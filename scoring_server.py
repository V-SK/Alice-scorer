#!/usr/bin/env python3
"""
Alice Protocol — Scoring Worker (Phase 1)
Standalone HTTP server for gradient scoring.
Runs on Mac (MPS) / GPU machine (CUDA) / CPU fallback.

Usage:
    # On Mac (MPS):
    python scoring_server.py \
        --model-path /path/to/alice_7b.pt \
        --validation-dir /path/to/validation_shards/ \
        --port 8090

    # On GPU:
    DEVICE=cuda python scoring_server.py ...

    # Default / recommended production mode:
    DEVICE=cpu python scoring_server.py ...

Architecture (Phase 1):
    PS main → POST /score {task_id, gradient_url, ...metadata}
    Worker  → GET gradient_url (pulls 16MB sparse gradient from nginx)
    Worker  → score (forward pass on validation set)
    Worker  → return {submission_id, score, loss_before, loss_after}

    Control plane (PS→Worker): metadata only, ~500 bytes
    Data plane (Worker→nginx): gradient fetch, ~16MB
    ✅ Control/data plane separation enforced
"""

import os
import sys
import json
import time
import struct
import zlib
import base64
import hashlib
import logging
import argparse
import asyncio
import threading
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import torch
import numpy as np

# --- aiohttp import (lightweight HTTP server) ---
try:
    from aiohttp import web
    import aiohttp
except ImportError:
    print("pip install aiohttp --break-system-packages")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ScoringWorker] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scoring_worker")

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_PORT = 8090
VALIDATION_SHARD_RANGE = (59951, 60000)  # 50 shards, indices 59951-60000
NUM_VALIDATION_SHARDS = 5  # Use 5 of 50 for speed (configurable)
FETCH_TIMEOUT = 30  # seconds to fetch gradient from storage
SCORE_TIMEOUT = 120  # max seconds per scoring operation


# =============================================================================
# Model loading — import from alice codebase
# =============================================================================

def load_model(model_path: str, device: str) -> torch.nn.Module:
    """
    Load Alice-7B model from checkpoint.
    
    Integration note: This imports alice's model definition.
    On the Worker machine, you need:
      - src/model.py (or wherever AliceModel is defined)
      - The checkpoint file (~13GB FP16)
    
    Adjust the import path to match your deployment.
    """
    # Try importing from alice codebase
    try:
        # Option A: alice-project is on PYTHONPATH
        from src.model import AliceForCausalLM, AliceConfig
    except ImportError:
        try:
            # Option B: alice_model module
            from src.model import AliceForCausalLM, AliceConfig
        except ImportError:
            log.error(
                "Cannot import AliceForCausalLM/AliceConfig from src.model.\n"
                "  Option 1: export PYTHONPATH=/path/to/alice-project\n"
                "  Option 2: symlink src/model.py to this directory"
            )
            sys.exit(1)

    log.info(f"Loading model from {model_path} to {device}...")
    t0 = time.time()

    config = AliceConfig()
    model = AliceForCausalLM(config)

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    # Handle both raw state_dict and wrapped checkpoint
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()

    elapsed = time.time() - t0
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    log.info(f"Model loaded: {param_count:.1f}B params, {elapsed:.1f}s, device={device}")
    return model


# =============================================================================
# Validation data loading
# =============================================================================

def load_validation_shards(
    validation_dir: str,
    num_shards: int = NUM_VALIDATION_SHARDS,
    device: str = "cpu",
) -> list:
    """Load held-out validation shards aligned with PS (_init_validation_set)."""
    shards = []
    val_dir = Path(validation_dir)
    index_path = val_dir.parent / "shard_index.json"
    files = []
    try:
        if index_path.exists():
            index_data = json.loads(index_path.read_text())
            meta = index_data.get("shards", [])
            total = len(meta)
            if total >= 60000:
                ids = list(range(59996, min(60001, total)))
            else:
                ids = list(range(max(0, total - 1), total))
            ids = ids[:num_shards]
            for sid in ids:
                fn = meta[sid].get("filename")
                if fn:
                    files.append((sid, val_dir / fn))
        else:
            # fallback: take latest shard_* files
            for path in sorted(val_dir.glob("shard_*.pt"))[-num_shards:]:
                match = re.search(r"(\d+)", path.stem)
                shard_id = int(match.group(1)) if match else None
                files.append((shard_id, path))
    except Exception as e:
        log.warning(f"Failed reading shard_index: {e}")
        files = []
        for path in sorted(val_dir.glob("shard_*.pt"))[-num_shards:]:
            match = re.search(r"(\d+)", path.stem)
            shard_id = int(match.group(1)) if match else None
            files.append((shard_id, path))

    for shard_id, sf in files:
        try:
            data = torch.load(sf, map_location="cpu", weights_only=True)
            if isinstance(data, dict) and "tokens" in data:
                data = data["tokens"]
            shards.append({
                "shard_id": shard_id,
                "tokens": data,
            })
            if shard_id is None:
                log.info(f"Loaded validation shard: {sf.name}")
            else:
                log.info(f"Loaded validation shard: {sf.name} (id={shard_id})")
        except Exception as e:
            log.warning(f"Failed to load {sf}: {e}")

    log.info(f"Loaded {len(shards)} validation shards")
    return shards



# =============================================================================
# Gradient deserialization (from binary_v2 format)
# =============================================================================

def decompress_gradients_sparse(payload: list) -> Dict[str, dict]:
    """
    Decompress binary_v2 gradient payload to sparse format.
    
    Input: JSON list of {name, shape, fmt, k, data} per parameter
    Output: Dict[param_name] → {indices: Tensor(int64), values: Tensor(fp16/fp32), shape: tuple}
    
    This is the same logic as PS's decompress_gradients_sparse().
    ~16MB sparse, NOT 13GB dense.
    """
    result = {}
    # Support both dict format {param_name: {shape,k,data,...}} and list format [{name,shape,k,data,...}]
    if isinstance(payload, dict):
        items = [(k, v) for k, v in payload.items() if isinstance(v, dict) and "k" in v]
    else:
        items = [(item["name"], item) for item in payload]
    for name, item in items:
        shape = tuple(item["shape"])
        k = item["k"]
        raw = zlib.decompress(base64.b64decode(item["data"]))

        # Detect precision from buffer size
        # raw = [values: k * value_bytes] + [indices: k * 4]
        indices_bytes = k * 4
        values_bytes = len(raw) - indices_bytes

        if values_bytes == k * 2:
            # FP16 values
            values = torch.frombuffer(bytearray(raw[:values_bytes]), dtype=torch.float16).clone()
        elif values_bytes == k * 4:
            # FP32 values
            values = torch.frombuffer(bytearray(raw[:values_bytes]), dtype=torch.float32).clone()
        else:
            raise ValueError(
                f"Param {name}: buffer mismatch. len={len(raw)}, k={k}, "
                f"expected values={k*2} or {k*4} + indices={indices_bytes}"
            )

        indices = torch.frombuffer(
            bytearray(raw[values_bytes:]), dtype=torch.int32
        ).to(torch.int64).clone()

        result[name] = {"indices": indices, "values": values, "shape": shape}

    return result


# =============================================================================
# Scoring logic (extracted from PS _score_gradient_sparse)
# =============================================================================

@torch.no_grad()
def score_gradient(
    model: torch.nn.Module,
    sparse_gradient: Dict[str, dict],
    validation_shards: list,
    device: str,
) -> Tuple[float, float, float]:
    """
    Score a sparse gradient against the validation set.
    
    1. Compute loss_before on validation data
    2. Apply sparse gradient to model (in-place)
    3. Compute loss_after
    4. Restore model to original state (exact reversal)
    5. Return (score, loss_before, loss_after)
    
    score = max(0, loss_before - loss_after)
    
    Uses saved_originals + try/finally for guaranteed model restoration.
    Uses nextafter nudge for FP16 precision (from sparse scoring patch).
    """
    if not validation_shards:
        raise RuntimeError("No validation shards loaded — cannot score")

    model.eval()

    # --- Step 1: Compute loss_before ---
    loss_before = _compute_validation_loss(model, validation_shards, device)

    # --- Step 2: Apply gradient, compute loss_after, restore ---
    saved_originals = {}
    try:
        # Apply sparse gradient in-place
        for name, grad_info in sparse_gradient.items():
            param = dict(model.named_parameters()).get(name)
            if param is None:
                continue

            indices = grad_info["indices"].to(device)
            values = grad_info["values"].to(param.dtype).to(device)

            flat = param.data.view(-1)

            # Save originals for exact restoration
            saved_originals[name] = flat[indices].clone()

            # Apply gradient (subtract = gradient descent direction)
            # Note: the convention depends on how miners compute gradients.
            # If miner sends raw gradients, PS subtracts: param -= lr * gradient
            # If miner sends deltas (already scaled), PS adds: param += delta
            # Match whatever PS currently does in _score_gradient_sparse()
            flat[indices] += values

            # FP16 nextafter nudge: if value didn't change due to precision,
            # nudge by one ULP to ensure the model actually changes
            unchanged = flat[indices] == saved_originals[name]
            if unchanged.any():
                nudge_vals = values[unchanged]
                direction = torch.where(nudge_vals > 0, torch.ones_like(nudge_vals), -torch.ones_like(nudge_vals))
                flat[indices[unchanged]] = torch.nextafter(flat[indices[unchanged]], flat[indices[unchanged]] + direction)

        # --- Step 3: Compute loss_after ---
        loss_after = _compute_validation_loss(model, validation_shards, device)

    finally:
        # --- Step 4: Restore model exactly ---
        for name, orig_values in saved_originals.items():
            param = dict(model.named_parameters()).get(name)
            if param is None:
                continue
            indices = sparse_gradient[name]["indices"].to(device)
            param.data.view(-1)[indices] = orig_values

    score = max(0.0, loss_before - loss_after)
    return score, loss_before, loss_after


def _compute_validation_loss(
    model: torch.nn.Module,
    validation_shards: list,
    device: str,
) -> float:
    """
    Average cross-entropy loss over validation shards.
    Uses FP32 for precision (even if model is FP16).
    """
    total_loss = 0.0
    total_tokens = 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    max_seq_len = int(
        getattr(getattr(model, "config", None), "max_position_embeddings", 2048)
        or 2048
    )
    stride = max(1, max_seq_len - 1)

    for shard_data in validation_shards:
        if isinstance(shard_data, torch.Tensor):
            tokens = shard_data
        elif isinstance(shard_data, dict):
            if "input_ids" in shard_data:
                tokens = shard_data["input_ids"]
            elif "tokens" in shard_data:
                tokens = shard_data["tokens"]
            else:
                continue
        else:
            continue

        if not isinstance(tokens, torch.Tensor):
            continue

        if tokens.dim() == 1:
            token_rows = [tokens]
        elif tokens.dim() == 2:
            token_rows = [row for row in tokens]
        else:
            continue

        for row in token_rows:
            row = row.reshape(-1)
            if row.numel() < 2:
                continue

            for start in range(0, row.numel() - 1, stride):
                chunk = row[start:start + max_seq_len]
                if chunk.numel() < 2:
                    continue

                input_ids = chunk.unsqueeze(0).to(device=device, dtype=torch.long)
                outputs = model(input_ids, labels=None)

                if isinstance(outputs, tuple):
                    logits = outputs[0]
                elif hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs

                shift_logits = logits[..., :-1, :].contiguous().float()
                shift_labels = input_ids[..., 1:].contiguous()

                loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                num_tokens = shift_labels.numel()
                total_loss += loss.item()
                total_tokens += num_tokens

    if total_tokens == 0:
        return float("inf")

    return total_loss / total_tokens


# =============================================================================
# HTTP Server
# =============================================================================

class ScoringServer:
    def __init__(self, model, validation_shards, device, model_version=0,
                 ps_url="", model_path=""):
        self.model = model
        self.validation_shards = validation_shards
        self.validation_shard_map = {
            int(shard.get("shard_id")): shard
            for shard in validation_shards
            if isinstance(shard, dict) and shard.get("shard_id") is not None
        }
        self.device = device
        self.model_version = model_version
        self.ps_url = ps_url.rstrip("/") if ps_url else ""
        self.model_path = model_path  # Path to current model file on disk
        self.busy = False
        self.scored_count = 0
        self.total_time = 0.0
        self._scored_ids = set()  # Idempotency tracking
        self._scored_results = {}  # Cache for idempotent responses
        self._model_lock = threading.Lock()

        # Start background model update loop (checks PS every 5 min)
        if self.ps_url:
            threading.Thread(target=self._model_update_loop, daemon=True,
                             name="model_update").start()
            log.info(f"[AUTO-UPDATE] Enabled, checking {self.ps_url} every 300s")

    def _select_validation_shards(self, shard_ids: list) -> Tuple[list, list]:
        if not shard_ids:
            return list(self.validation_shards), []

        requested = []
        for value in shard_ids:
            try:
                requested.append(int(value))
            except Exception:
                continue

        selected = []
        missing = []
        for shard_id in requested:
            shard = self.validation_shard_map.get(shard_id)
            if shard is None:
                missing.append(shard_id)
            else:
                selected.append(shard)
        return selected, missing

    def _score_submission_blocking(self, raw_data: bytes) -> Tuple[float, float, float]:
        payload = json.loads(raw_data)
        sparse_gradient = decompress_gradients_sparse(payload)
        with self._model_lock:
            return score_gradient(self.model, sparse_gradient, self.validation_shards, self.device)

    def _validate_blocking(self, selected_shards: list) -> float:
        with self._model_lock:
            return _compute_validation_loss(self.model, selected_shards, self.device)

    async def handle_score(self, request: web.Request) -> web.Response:
        """
        POST /score
        
        Request body (JSON):
        {
            "submission_id": "uuid-or-hash",
            "model_version": 42,
            "shard_id": 12345,
            "miner_id": "aXXX...alice-address",
            "epoch_id": 7,
            "gradient_url": "http://65.109.84.107:8888/gradients/uuid.bin"
        }
        
        Response (JSON):
        {
            "submission_id": "uuid-or-hash",
            "score": 0.000827,
            "loss_before": 11.6276,
            "loss_after": 11.6268,
            "model_version": 42,
            "elapsed_ms": 4823
        }
        """
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)

        # --- Validate required fields ---
        required = ["submission_id", "model_version", "shard_id", "miner_id", "epoch_id", "gradient_url"]
        missing = [f for f in required if f not in body]
        if missing:
            return web.json_response({"error": f"missing fields: {missing}"}, status=400)

        sid = body["submission_id"]

        # --- Idempotency check ---
        if sid in self._scored_results:
            log.info(f"[IDEMPOTENT] Returning cached result for {sid}")
            return web.json_response(self._scored_results[sid])

        # --- Model version check ---
        if body["model_version"] != self.model_version:
            log.warning(
                f"Model version mismatch: worker={self.model_version}, "
                f"request={body['model_version']}"
            )
            # Could return error or proceed with warning
            # For Phase 1: proceed but flag it
            # return web.json_response({"error": "model_version_mismatch"}, status=409)

        # --- Busy check (single worker, one score at a time) ---
        if self.busy:
            return web.json_response(
                {"error": "worker_busy", "submission_id": sid}, status=503
            )

        self.busy = True
        t0 = time.time()

        try:
            # --- Fetch gradient from storage (data plane) ---
            gradient_url = body["gradient_url"]
            log.info(f"[SCORE] {sid[:12]}... fetching gradient from {gradient_url}")

            async with aiohttp.ClientSession() as session:
                async with session.get(gradient_url, timeout=aiohttp.ClientTimeout(total=FETCH_TIMEOUT)) as resp:
                    if resp.status != 200:
                        return web.json_response(
                            {"error": f"gradient_fetch_failed: HTTP {resp.status}", "submission_id": sid},
                            status=502,
                        )
                    raw_data = await resp.read()

            score, loss_before, loss_after = await asyncio.to_thread(
                self._score_submission_blocking,
                raw_data,
            )

            elapsed_ms = int((time.time() - t0) * 1000)

            result = {
                "submission_id": sid,
                "score": round(score, 8),
                "loss_before": round(loss_before, 6),
                "loss_after": round(loss_after, 6),
                "model_version": self.model_version,
                "elapsed_ms": elapsed_ms,
            }

            # Cache for idempotency
            self._scored_results[sid] = result
            # Limit cache size (keep last 1000)
            if len(self._scored_results) > 1000:
                oldest = list(self._scored_results.keys())[:500]
                for k in oldest:
                    del self._scored_results[k]

            self.scored_count += 1
            self.total_time += elapsed_ms / 1000

            log.info(
                f"[SCORE] {sid[:12]}... done: score={score:.6f}, "
                f"loss={loss_before:.4f}→{loss_after:.4f}, {elapsed_ms}ms"
            )

            return web.json_response(result)

        except asyncio.TimeoutError:
            log.error(f"[SCORE] {sid[:12]}... timeout fetching gradient")
            return web.json_response(
                {"error": "gradient_fetch_timeout", "submission_id": sid}, status=504
            )
        except Exception as e:
            log.error(f"[SCORE] {sid[:12]}... error: {e}", exc_info=True)
            return web.json_response(
                {"error": str(e), "submission_id": sid}, status=500
            )
        finally:
            self.busy = False

    # ================================================================
    # Background model auto-update (delta / full download from PS)
    # ================================================================


    def _ensure_ps_token(self):
        """Register with PS to get auth token for delta requests."""
        if hasattr(self, '_ps_token') and self._ps_token:
            return self._ps_token
        import requests as _requests
        try:
            resp = _requests.post(
                f"{self.ps_url}/register",
                json={
                    "address": f"scorer-{id(self):x}",
                    "instance_id": f"scorer-{id(self):x}",
                    "protocol_version": "1.0",
                    "data_format": "tensor",
                    "device_type": "cpu",
                    "memory_gb": 64,
                },
                timeout=15,
            )
            if resp.status_code == 200:
                self._ps_token = resp.json().get("token", "")
                if self._ps_token:
                    log.info("[AUTO-UPDATE] Got PS auth token")
                    return self._ps_token
        except Exception as e:
            log.warning(f"[AUTO-UPDATE] Failed to get PS token: {e}")
        self._ps_token = None
        return None

    def _model_update_loop(self):
        """Background thread: check PS for model updates every 300s."""
        time.sleep(30)  # Initial delay — let server start up
        while True:
            try:
                self._check_and_apply_updates()
            except Exception as e:
                log.error(f"[AUTO-UPDATE] loop error: {e}", exc_info=True)
            time.sleep(300)  # 5 minutes

    def _check_and_apply_updates(self):
        """Check PS model version; pull deltas or full model if behind."""
        import requests as _requests  # stdlib-compat, no new dep

        try:
            resp = _requests.get(f"{self.ps_url}/model/info", timeout=15)
            if resp.status_code != 200:
                log.warning(f"[AUTO-UPDATE] /model/info returned {resp.status_code}")
                return
            info = resp.json()
        except Exception as e:
            log.warning(f"[AUTO-UPDATE] failed to reach PS: {e}")
            return

        ps_version = info.get("model_version", 0)
        if ps_version <= self.model_version:
            return  # Already up to date

        gap = ps_version - self.model_version
        log.info(f"[AUTO-UPDATE] PS v{ps_version} > local v{self.model_version} (gap={gap})")

        if gap > 10:
            # Too far behind — full download
            log.info(f"[AUTO-UPDATE] gap={gap} > 10, downloading full model...")
            self._download_full_model_sync(ps_version)
        else:
            # Incremental delta updates
            current = self.model_version
            for v in range(current, ps_version):
                ok = self._fetch_and_apply_delta(v)
                if not ok:
                    log.warning(f"[AUTO-UPDATE] delta from v{v} failed, falling back to full download")
                    self._download_full_model_sync(ps_version)
                    return
            log.info(f"[AUTO-UPDATE] ✅ Incremental update complete → v{self.model_version}")

    def _fetch_and_apply_delta(self, from_version: int) -> bool:
        """Fetch single delta from PS and apply to in-memory model."""
        import requests as _requests

        try:
            headers = {}
            token = self._ensure_ps_token()
            if token:
                headers["Authorization"] = f"Bearer {token}"
            resp = _requests.get(
                f"{self.ps_url}/model/delta",
                params={"from_version": from_version},
                headers=headers,
                timeout=120,
            )
            if resp.status_code != 200:
                log.warning(f"[AUTO-UPDATE] delta from v{from_version}: HTTP {resp.status_code}")
                return False

            delta_payload = resp.json()
        except Exception as e:
            log.warning(f"[AUTO-UPDATE] delta fetch failed: {e}")
            return False

        return self._apply_delta(delta_payload, from_version)

    def _apply_delta(self, delta_payload: dict, from_version: int) -> bool:
        """
        Apply compressed delta to the in-memory model.
        delta_payload is binary_v2 compressed format (same as miner gradients).
        Uses decompress_gradients_sparse (already in this file) then scatters to model params.
        """
        try:
            # decompress_gradients_sparse returns {name: {indices, values, shape}}
            sparse_delta = decompress_gradients_sparse(delta_payload)

            with self._model_lock:
                param_dict = dict(self.model.named_parameters())
                updated = 0
                for name, sdata in sparse_delta.items():
                    param = param_dict.get(name)
                    if param is None:
                        continue
                    indices = sdata["indices"]
                    values = sdata["values"].to(param.dtype).to(param.device)
                    # Scatter into dense parameter
                    flat = param.data.view(-1)
                    flat[indices.to(param.device)] += values
                    updated += 1

                self.model_version = from_version + 1
                self._scored_results.clear()  # Invalidate scoring cache

            log.info(f"[AUTO-UPDATE] Applied delta v{from_version}→v{self.model_version}, {updated} params updated")
            return True

        except Exception as e:
            log.error(f"[AUTO-UPDATE] _apply_delta failed: {e}", exc_info=True)
            return False

    def _download_full_model_sync(self, target_version: int):
        """Download full model from PS and hot-reload (blocking, runs in bg thread)."""
        import requests as _requests

        model_dir = Path(os.environ.get("MODEL_DIR", "/tmp/alice-models"))
        model_dir.mkdir(parents=True, exist_ok=True)
        dest = model_dir / f"model_v{target_version}.pt"
        tmp_path = str(dest) + ".downloading"

        # Get download URL from PS
        try:
            info_resp = _requests.get(f"{self.ps_url}/model/info", timeout=15)
            info = info_resp.json() if info_resp.status_code == 200 else {}
            download_url = info.get("download_url", "")
            if not download_url:
                download_url = f"{self.ps_url}/models/v{target_version}_full.pt"
            elif download_url.startswith("/"):
                download_url = f"{self.ps_url}{download_url}"
        except Exception:
            download_url = f"{self.ps_url}/models/v{target_version}_full.pt"

        log.info(f"[AUTO-UPDATE] Downloading full model v{target_version} from {download_url}")

        try:
            resp = _requests.get(download_url, stream=True, timeout=3600)
            if resp.status_code != 200:
                log.error(f"[AUTO-UPDATE] Full model download failed: HTTP {resp.status_code}")
                return

            downloaded = 0
            last_log = 0
            total = int(resp.headers.get("content-length", 0))
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded - last_log >= 50 * 1024 * 1024:
                        if total:
                            pct = downloaded * 100 / total
                            log.info(f"[AUTO-UPDATE] Download: {downloaded/1e6:.0f}MB / {total/1e6:.0f}MB ({pct:.1f}%)")
                        else:
                            log.info(f"[AUTO-UPDATE] Download: {downloaded/1e6:.0f}MB")
                        last_log = downloaded

            os.replace(tmp_path, str(dest))
            log.info(f"[AUTO-UPDATE] Downloaded {downloaded/1e6:.1f}MB → {dest}")

            # Hot reload
            with self._model_lock:
                new_model = load_model(str(dest), self.device)
                self.model = new_model
                self.model_version = target_version
                self._scored_results.clear()

            log.info(f"[AUTO-UPDATE] ✅ Full model v{target_version} loaded, hot-reloaded")

        except Exception as e:
            log.error(f"[AUTO-UPDATE] full download failed: {e}", exc_info=True)
            # Clean up partial download
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    async def handle_health(self, request: web.Request) -> web.Response:
        """GET /health — liveness + status"""
        avg_ms = (self.total_time / self.scored_count * 1000) if self.scored_count > 0 else 0
        return web.json_response({
            "status": "ok",
            "device": self.device,
            "model_version": self.model_version,
            "busy": self.busy,
            "scored_count": self.scored_count,
            "avg_score_ms": round(avg_ms),
            "validation_shards": len(self.validation_shards),
        })

    async def handle_validate(self, request: web.Request) -> web.Response:
        """POST /validate — compute validation loss on held-out shards."""
        if self.busy:
            return web.json_response({"error": "worker_busy"}, status=503)

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)

        requested_version = body.get("model_version")
        if requested_version is not None and int(requested_version) != int(self.model_version):
            return web.json_response(
                {
                    "error": "model_version_mismatch",
                    "worker_model_version": self.model_version,
                    "requested_model_version": requested_version,
                },
                status=409,
            )

        shard_ids = body.get("shard_ids") or []
        selected_shards, missing = self._select_validation_shards(shard_ids)
        if missing:
            return web.json_response(
                {
                    "error": "validation_shards_missing",
                    "missing_shard_ids": missing,
                    "available_shard_ids": sorted(self.validation_shard_map.keys()),
                },
                status=400,
            )

        if not selected_shards:
            return web.json_response({"error": "no_validation_shards_selected"}, status=400)

        self.busy = True
        try:
            avg_loss = await asyncio.to_thread(self._validate_blocking, selected_shards)
            return web.json_response(
                {
                    "status": "ok",
                    "avg_loss": round(float(avg_loss), 6),
                    "num_shards": len(selected_shards),
                    "model_version": self.model_version,
                }
            )
        except Exception as e:
            log.error(f"[VALIDATE] error: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)
        finally:
            self.busy = False

    async def _download_model(self, url: str, dest: str) -> str:
        """Download model from URL with streaming + progress logging."""
        log.info(f"[RELOAD] Downloading model from {url}")
        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = str(dest_path) + ".downloading"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=3600)) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Download failed: HTTP {resp.status}")
                total = resp.content_length or 0
                downloaded = 0
                last_log = 0
                with open(tmp_path, "wb") as f:
                    async for chunk in resp.content.iter_chunked(1024 * 1024):  # 1MB chunks
                        f.write(chunk)
                        downloaded += len(chunk)
                        # Log progress every 50MB
                        if downloaded - last_log >= 50 * 1024 * 1024:
                            if total:
                                pct = downloaded * 100 / total
                                log.info(f"[RELOAD] Download progress: {downloaded / 1e6:.0f}MB / {total / 1e6:.0f}MB ({pct:.1f}%)")
                            else:
                                log.info(f"[RELOAD] Download progress: {downloaded / 1e6:.0f}MB")
                            last_log = downloaded

        os.rename(tmp_path, dest)
        log.info(f"[RELOAD] Download complete: {downloaded / 1e6:.1f}MB → {dest}")
        return dest

    async def handle_reload_model(self, request: web.Request) -> web.Response:
        """
        POST /reload
        
        Request body (JSON):
        {
            "model_path": "/path/to/new/checkpoint.pt",
            "model_version": 43,
            "download_url": "https://dl.aliceprotocol.org/v43_full.pt"  (optional)
        }
        
        If download_url is provided, downloads the model first then loads it.
        If only model_path is provided, loads directly from local path.
        Triggers model reload without restarting the Worker.
        Called by PS after aggregation bumps model_version.
        """
        if self.busy:
            return web.json_response({"error": "worker_busy"}, status=503)

        try:
            body = await request.json()
            model_path = body.get("model_path")
            download_url = body.get("download_url")
            new_version = body.get("model_version", self.model_version + 1)

            if not model_path and not download_url:
                return web.json_response(
                    {"error": "model_path or download_url required"}, status=400
                )

            self.busy = True

            # If download_url provided, download first
            if download_url:
                if not model_path:
                    # Default download destination
                    model_dir = Path(os.environ.get("MODEL_DIR", "/tmp/alice-models"))
                    model_path = str(model_dir / f"model_v{new_version}.pt")
                log.info(f"[RELOAD] Will download v{new_version} from URL → {model_path}")
                await self._download_model(download_url, model_path)

            log.info(f"[RELOAD] Loading model v{new_version} from {model_path}")

            # Reload model
            self.model = load_model(model_path, self.device)
            self.model_version = new_version
            self._scored_results.clear()  # Invalidate cache

            log.info(f"[RELOAD] Model v{new_version} loaded successfully")
            return web.json_response({
                "status": "reloaded",
                "model_version": self.model_version,
                "model_path": model_path,
                "downloaded": bool(download_url),
            })
        except Exception as e:
            log.error(f"[RELOAD] Failed: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)
        finally:
            self.busy = False


# =============================================================================
# Main
# =============================================================================

def detect_device() -> str:
    env_device = os.environ.get("DEVICE", "cpu")
    if env_device != "auto":
        return env_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Alice Scoring Worker (Phase 1)")
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--validation-dir", required=True, help="Path to validation shard directory")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"HTTP port (default: {DEFAULT_PORT})")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--device", default="cpu", help="Device: cpu, auto, cuda, mps")
    parser.add_argument("--model-version", type=int, default=0, help="Initial model version")
    parser.add_argument("--num-val-shards", type=int, default=NUM_VALIDATION_SHARDS, help="Number of validation shards to use")
    parser.add_argument("--ps-url", default="", help="Parameter Server URL for auto-update (empty = disabled)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device != "auto":
        os.environ["DEVICE"] = args.device
    device = detect_device()
    log.info(f"Using device: {device}")

    # Load model
    model = load_model(args.model_path, device)

    # Load validation shards
    validation_shards = load_validation_shards(
        args.validation_dir, 
        num_shards=args.num_val_shards,
        device=device,
    )

    # Create server
    server = ScoringServer(
        model=model,
        validation_shards=validation_shards,
        device=device,
        model_version=args.model_version,
        ps_url=args.ps_url,
        model_path=args.model_path,
    )

    app = web.Application()
    app.router.add_post("/score", server.handle_score)
    app.router.add_get("/health", server.handle_health)
    app.router.add_post("/validate", server.handle_validate)
    app.router.add_post("/reload", server.handle_reload_model)

    log.info(f"Starting scoring worker on {args.host}:{args.port}")
    log.info(f"  Device: {device}")
    log.info(f"  Model version: {args.model_version}")
    log.info(f"  Validation shards: {len(validation_shards)}")
    log.info(f"  Endpoints: POST /score, POST /validate, GET /health, POST /reload")

    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
