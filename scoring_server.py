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
import platform
import sys
import json
import time
import math
import gc
import inspect
import struct
import zlib
import base64
import hashlib
import logging
import argparse
import asyncio
import threading
import re
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn.functional as F
import numpy as np
import requests

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
FETCH_TIMEOUT = 120  # seconds to fetch gradient from storage
SCORE_TIMEOUT = 120  # max seconds per scoring operation
SAFE_MAX_SEQ_LEN = int(os.environ.get("ALICE_SAFE_MAX_SEQ_LEN", "2048"))
VALIDATION_SEQ_LEN = int(os.environ.get("ALICE_VALIDATION_SEQ_LEN", "128"))
VALIDATION_BATCHES_PER_SHARD = int(os.environ.get("ALICE_VALIDATION_BATCHES_PER_SHARD", "1"))
VALIDATION_MICROBATCH_SIZE = max(1, int(os.environ.get("ALICE_VALIDATION_MICROBATCH_SIZE", "8")))
SCORER_REGISTER_INTERVAL_S = max(15, int(os.environ.get("ALICE_SCORER_REGISTER_INTERVAL_S", "60")))
SCORER_REGISTER_BOOT_POLL_S = max(2, int(os.environ.get("ALICE_SCORER_REGISTER_BOOT_POLL_S", "5")))


def _read_cpu_model() -> str:
    try:
        system = platform.system()
        if system == "Darwin":
            import subprocess
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
            ).strip()
        if system == "Linux":
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        if system == "Windows":
            return platform.processor().strip()
    except Exception:
        pass
    return platform.processor().strip() or "Unknown"


def detect_device_info(device_override: Optional[str] = None) -> Dict[str, Any]:
    try:
        import psutil  # type: ignore
    except Exception:
        psutil = None

    selected = str(device_override or detect_device()).strip().lower()
    ram_gb = 0.0
    if psutil is not None:
        try:
            ram_gb = round(psutil.virtual_memory().total / 1e9, 1)
        except Exception:
            ram_gb = 0.0

    info: Dict[str, Any] = {
        "os": platform.system(),
        "platform": platform.system().lower(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ram_gb": ram_gb,
        "system_memory_gb": ram_gb,
        "cpu_model": _read_cpu_model(),
        "cpu_count": os.cpu_count() or 0,
        "device_type": "cpu",
        "device": "cpu",
        "device_name": "CPU",
        "gpu_model": "CPU-only",
        "gpu_vram_gb": 0.0,
        "vram_gb": 0.0,
        "unified_memory_gb": 0.0,
        "gpu_count": 0,
        "vendor": "cpu",
        "memory_gb": ram_gb,
    }

    if selected == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = round(props.total_memory / 1e9, 1)
        gpu_model = torch.cuda.get_device_name(0)
        info.update({
            "device_type": "cuda",
            "device": "cuda",
            "device_name": gpu_model,
            "gpu_model": gpu_model,
            "gpu_vram_gb": vram_gb,
            "vram_gb": vram_gb,
            "gpu_count": torch.cuda.device_count(),
            "vendor": "nvidia",
            "memory_gb": vram_gb,
        })
        return info

    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if selected == "mps" and mps_available:
        gpu_model = info["cpu_model"] or f"Apple Silicon ({platform.machine()})"
        info.update({
            "device_type": "mps",
            "device": "mps",
            "device_name": gpu_model,
            "gpu_model": gpu_model,
            "gpu_vram_gb": ram_gb,
            "unified_memory_gb": ram_gb,
            "vendor": "apple",
            "memory_gb": ram_gb,
        })
        return info

    return info


def format_device_log_line(info: Dict[str, Any]) -> str:
    device_type = str(info.get("device_type") or "cpu").lower()
    if device_type == "cuda":
        return f"[Device] {info.get('gpu_model', 'Unknown CUDA GPU')}, {float(info.get('gpu_vram_gb', 0.0)):.1f}GB VRAM, CUDA"
    if device_type == "mps":
        return f"[Device] {info.get('gpu_model', 'Apple Silicon')}, {float(info.get('ram_gb', 0.0)):.1f}GB unified memory, MPS"
    return f"[Device] {info.get('cpu_model', 'Unknown CPU')}, {float(info.get('ram_gb', 0.0)):.1f}GB RAM, CPU-only"


# =============================================================================
# Model loading — import from alice codebase
# =============================================================================

def resolve_model_dtype(dtype_name: str, device: str) -> tuple[torch.dtype, str]:
    normalized = (dtype_name or "float16").strip().lower()
    if normalized == "auto":
        normalized = "float16" if platform.machine() in ("arm64", "aarch64") else "float32"

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    dtype = mapping.get(normalized)
    if dtype is None:
        raise ValueError(f"Unsupported model dtype: {dtype_name}")
    return dtype, ("float16" if dtype == torch.float16 else "bfloat16" if dtype == torch.bfloat16 else "float32")


def _supports_kwarg(func: Any, name: str) -> bool:
    try:
        return name in inspect.signature(func).parameters
    except Exception:
        return False


def get_process_memory_mb() -> float:
    try:
        import psutil  # type: ignore
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 1)
    except Exception:
        pass

    try:
        with open("/proc/self/status", "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return round(float(parts[1]) / 1024.0, 1)
    except Exception:
        pass
    return 0.0


def load_model(model_path: str, device: str, model_dtype_name: str) -> tuple[torch.nn.Module, str, Optional[int]]:
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

    target_dtype, resolved_dtype_name = resolve_model_dtype(model_dtype_name, device)
    log.info(f"Loading model from {model_path} to {device} with dtype={resolved_dtype_name}...")
    t0 = time.time()

    config = AliceConfig()
    previous_default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(target_dtype)
        model = AliceForCausalLM(config)
    finally:
        torch.set_default_dtype(previous_default_dtype)

    load_kwargs = {"map_location": "cpu"}
    if _supports_kwarg(torch.load, "weights_only"):
        load_kwargs["weights_only"] = True
    if _supports_kwarg(torch.load, "mmap"):
        load_kwargs["mmap"] = True

    try:
        checkpoint = torch.load(model_path, **load_kwargs)
    except TypeError as exc:
        if load_kwargs.pop("mmap", None) is None:
            raise
        log.warning(f"[MODEL-LOAD] torch.load mmap unsupported, retrying without mmap: {exc}")
        checkpoint = torch.load(model_path, **load_kwargs)
    checkpoint_version = checkpoint.get("model_version") if isinstance(checkpoint, dict) else None
    # Handle both raw state_dict and wrapped checkpoint
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict_kwargs = {"strict": False}
    if _supports_kwarg(model.load_state_dict, "assign"):
        state_dict_kwargs["assign"] = True
    try:
        model.load_state_dict(state_dict, **state_dict_kwargs)
    except TypeError as exc:
        if state_dict_kwargs.pop("assign", None) is None:
            raise
        log.warning(f"[MODEL-LOAD] load_state_dict assign unsupported, retrying without assign: {exc}")
        model.load_state_dict(state_dict, **state_dict_kwargs)
    for param in model.parameters():
        param.requires_grad_(False)
    del state_dict
    del checkpoint
    gc.collect()

    model = model.to(device=device, dtype=target_dtype)
    model.eval()

    elapsed = time.time() - t0
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    configured_max_seq_len = int(
        getattr(getattr(model, "config", None), "max_position_embeddings", 2048)
        or 2048
    )
    if configured_max_seq_len < 2 or configured_max_seq_len > SAFE_MAX_SEQ_LEN:
        log.warning(
            f"[MODEL-CONFIG] max_position_embeddings={configured_max_seq_len} outside safe range; "
            f"runtime validation window will clamp to {SAFE_MAX_SEQ_LEN}"
        )
    log.info(
        f"Model loaded: {param_count:.1f}B params, {elapsed:.1f}s, "
        f"device={device}, dtype={resolved_dtype_name}"
    )
    log.info(
        f"[MODEL-CONFIG] max_position_embeddings={configured_max_seq_len} "
        f"safe_max_seq_len={SAFE_MAX_SEQ_LEN}"
    )
    return model, resolved_dtype_name, int(checkpoint_version) if checkpoint_version is not None else None


def _read_version_file(path: Path) -> Optional[int]:
    try:
        raw = path.read_text(encoding="utf-8").strip()
        return int(raw) if raw else None
    except Exception:
        return None


def _parse_version_hint(path_str: str) -> Optional[int]:
    match = re.search(r"[_/]v(\d+)_full\.pt$", path_str)
    if not match:
        match = re.search(r"[_/]model_v(\d+)\.pt$", path_str)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def resolve_startup_baseline(model_path: str, requested_version: int) -> Tuple[str, int]:
    requested = Path(model_path)
    model_dir = requested.parent
    current_model = model_dir / "current_full.pt"
    current_version = model_dir / "current_version.txt"

    if current_model.exists():
        resolved_version = _read_version_file(current_version)
        if resolved_version is None:
            resolved_version = requested_version or _parse_version_hint(str(current_model)) or 0
        return str(current_model), int(resolved_version)

    fallback: Optional[Path] = requested if requested.exists() else None
    if fallback is None:
        candidates = sorted(
            (
                path for path in model_dir.glob("v*_full.pt")
                if _parse_version_hint(str(path)) is not None
            ),
            key=lambda p: int(_parse_version_hint(str(p)) or 0),
            reverse=True,
        )
        if candidates:
            fallback = candidates[0]

    if fallback is None:
        raise FileNotFoundError(
            f"No scorer baseline found in {model_dir} "
            f"(expected {current_model.name} or v*_full.pt bootstrap)"
        )

    resolved_version = requested_version or _parse_version_hint(str(fallback)) or 0
    return str(fallback), int(resolved_version)


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
            shape = tuple(data.shape) if isinstance(data, torch.Tensor) else None
            if shard_id is None:
                log.info(f"Loaded validation shard: {sf.name} shape={shape}")
            else:
                log.info(f"Loaded validation shard: {sf.name} (id={shard_id}) shape={shape}")
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

def _resolve_sample_seq_len(model: torch.nn.Module) -> int:
    configured_max_seq_len = int(
        getattr(getattr(model, "config", None), "max_position_embeddings", 2048)
        or 2048
    )
    max_seq_len = max(2, min(configured_max_seq_len, SAFE_MAX_SEQ_LEN))
    return max(2, min(VALIDATION_SEQ_LEN, max_seq_len))


def _extract_validation_rows(validation_shards: list) -> list[dict]:
    extracted = []
    for shard_index, shard_data in enumerate(validation_shards):
        if isinstance(shard_data, torch.Tensor):
            tokens = shard_data
            shard_id = shard_index
        elif isinstance(shard_data, dict):
            if "input_ids" in shard_data:
                tokens = shard_data["input_ids"]
            elif "tokens" in shard_data:
                tokens = shard_data["tokens"]
            else:
                continue
            shard_id = shard_data.get("shard_id", shard_index)
        else:
            continue

        if not isinstance(tokens, torch.Tensor):
            continue

        if tokens.dim() == 1:
            token_rows = [tokens.reshape(-1)]
        elif tokens.dim() == 2:
            token_rows = [row.reshape(-1) for row in tokens]
        else:
            continue

        extracted.append({
            "shard_id": int(shard_id) if shard_id is not None else int(shard_index),
            "rows": token_rows,
        })
    return extracted


def build_validation_plan(
    model: torch.nn.Module,
    validation_shards: list,
    model_version: int,
) -> tuple[list[dict], int]:
    sample_seq_len = _resolve_sample_seq_len(model)
    extracted = _extract_validation_rows(validation_shards)
    plan = []
    for shard_index, shard in enumerate(extracted):
        shard_id = int(shard["shard_id"])
        for row_index, row in enumerate(shard["rows"]):
            if row.numel() <= sample_seq_len + 1:
                continue
            max_start = max(1, int(row.numel()) - sample_seq_len - 1)
            for batch_index in range(VALIDATION_BATCHES_PER_SHARD):
                seed_material = (
                    f"{int(model_version)}:{shard_id}:{row_index}:{batch_index}:"
                    f"{int(row.numel())}:{sample_seq_len}"
                )
                seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
                start = seed % max_start
                plan.append({
                    "shard_index": shard_index,
                    "row_index": row_index,
                    "start": int(start),
                })
    return plan, sample_seq_len


def prepare_validation_batches(
    validation_shards: list,
    validation_plan: list[dict],
    sample_seq_len: int,
    *,
    microbatch_size: int = VALIDATION_MICROBATCH_SIZE,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    extracted = _extract_validation_rows(validation_shards)
    prepared: list[tuple[torch.Tensor, torch.Tensor]] = []
    input_batch: list[torch.Tensor] = []
    label_batch: list[torch.Tensor] = []

    def flush() -> None:
        if not input_batch:
            return
        prepared.append((
            torch.stack(input_batch, dim=0).to(dtype=torch.long).contiguous(),
            torch.stack(label_batch, dim=0).to(dtype=torch.long).contiguous(),
        ))
        input_batch.clear()
        label_batch.clear()

    for item in validation_plan:
        shard = extracted[int(item["shard_index"])]
        row = shard["rows"][int(item["row_index"])]
        start = int(item["start"])
        chunk = row[start : start + sample_seq_len + 1]
        if chunk.numel() < sample_seq_len + 1:
            continue
        input_batch.append(chunk[:-1])
        label_batch.append(chunk[1:])
        if len(input_batch) >= microbatch_size:
            flush()

    flush()
    return prepared

@torch.no_grad()
def score_gradient(
    model: torch.nn.Module,
    sparse_gradient: Dict[str, dict],
    validation_shards: list,
    device: str,
    *,
    learning_rate: float,
    param_dict: Optional[Dict[str, torch.nn.Parameter]] = None,
    validation_plan: Optional[list[dict]] = None,
    prepared_batches: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
    loss_before: Optional[float] = None,
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
    param_map = param_dict or dict(model.named_parameters())

    # --- Step 1: Compute loss_before ---
    if loss_before is None:
        loss_before = _compute_validation_loss(
            model,
            validation_shards,
            device,
            validation_plan=validation_plan,
            prepared_batches=prepared_batches,
        )

    # --- Step 2: Apply gradient, compute loss_after, restore ---
    saved_originals = {}
    try:
        # Apply sparse gradient in-place
        for name, grad_info in sparse_gradient.items():
            param = param_map.get(name)
            if param is None:
                continue

            indices = grad_info["indices"].to(device)
            values = grad_info["values"].to(param.dtype).to(device)

            flat = param.data.view(-1)

            # Save originals for exact restoration
            saved_originals[name] = flat[indices].clone()

            # Miner submissions are sparse gradients. Match PS scoring/update semantics:
            # param := param - learning_rate * gradient
            current_fp32 = flat[indices].float()
            updated_fp32 = current_fp32 - (float(learning_rate) * values.float())
            updated_cast = updated_fp32.to(flat.dtype)

            # FP16 nextafter nudge: if value didn't change due to precision,
            # nudge by one ULP to ensure the model actually changes
            unchanged = updated_cast == flat[indices]
            if unchanged.any():
                direction = -(float(learning_rate) * values.float())
                toward = torch.where(
                    direction < 0,
                    torch.full_like(current_fp32, float('-inf')),
                    torch.full_like(current_fp32, float('inf'))
                )
                nudged = torch.nextafter(
                    flat[indices][unchanged].float(),
                    toward[unchanged]
                ).to(flat.dtype)
                updated_cast = updated_cast.clone()
                updated_cast[unchanged] = nudged

            flat[indices] = updated_cast

        # --- Step 3: Compute loss_after ---
        loss_after = _compute_validation_loss(
            model,
            validation_shards,
            device,
            validation_plan=validation_plan,
            prepared_batches=prepared_batches,
        )

    finally:
        # --- Step 4: Restore model exactly ---
        for name, orig_values in saved_originals.items():
            param = param_map.get(name)
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
    *,
    validation_plan: Optional[list[dict]] = None,
    prepared_batches: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
) -> float:
    """
    Average cross-entropy loss over validation shards.
    Uses FP32 for precision (even if model is FP16).
    """
    total_loss = 0.0
    total_batches = 0
    sample_seq_len = _resolve_sample_seq_len(model)
    log.info(
        f"[VAL] sample_seq_len={sample_seq_len} safe_max_seq_len={SAFE_MAX_SEQ_LEN} "
        f"batches_per_shard={VALIDATION_BATCHES_PER_SHARD} microbatch={VALIDATION_MICROBATCH_SIZE} "
        f"plan={'prepared' if prepared_batches is not None else 'fixed' if validation_plan else 'random'}"
    )
    extracted = _extract_validation_rows(validation_shards) if prepared_batches is None else []

    with torch.inference_mode():
        if prepared_batches is not None:
            for input_ids, labels in prepared_batches:
                input_ids = input_ids.to(device=device, dtype=torch.long)
                labels = labels.to(device=device, dtype=torch.long)
                outputs = model(input_ids, labels=None)

                if isinstance(outputs, tuple):
                    logits = outputs[0]
                elif hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs

                shift_logits = logits[..., :-1, :].contiguous().float().transpose(1, 2)
                shift_labels = labels[..., 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits,
                    shift_labels,
                    ignore_index=-100,
                )
                batch_size = int(input_ids.size(0))
                total_loss += float(loss.item()) * batch_size
                total_batches += batch_size
        elif validation_plan is not None:
            for item in validation_plan:
                shard = extracted[int(item["shard_index"])]
                row = shard["rows"][int(item["row_index"])]
                start = int(item["start"])
                chunk = row[start : start + sample_seq_len + 1]
                if chunk.numel() < sample_seq_len + 1:
                    continue

                input_ids = chunk[:-1].unsqueeze(0).to(device=device, dtype=torch.long)
                labels = chunk[1:].unsqueeze(0).to(device=device, dtype=torch.long)
                outputs = model(input_ids, labels=None)

                if isinstance(outputs, tuple):
                    logits = outputs[0]
                elif hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs

                shift_logits = logits[..., :-1, :].contiguous().float()
                shift_labels = labels[..., 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                total_loss += float(loss.item())
                total_batches += 1
        else:
            for shard in extracted:
                for row in shard["rows"]:
                    if row.numel() <= sample_seq_len + 1:
                        continue

                    for _ in range(VALIDATION_BATCHES_PER_SHARD):
                        max_start = max(1, row.numel() - sample_seq_len - 1)
                        start = int(torch.randint(0, max_start, (1,)).item())
                        chunk = row[start : start + sample_seq_len + 1]
                        if chunk.numel() < sample_seq_len + 1:
                            continue

                        input_ids = chunk[:-1].unsqueeze(0).to(device=device, dtype=torch.long)
                        labels = chunk[1:].unsqueeze(0).to(device=device, dtype=torch.long)
                        outputs = model(input_ids, labels=None)

                        if isinstance(outputs, tuple):
                            logits = outputs[0]
                        elif hasattr(outputs, "logits"):
                            logits = outputs.logits
                        else:
                            logits = outputs

                        shift_logits = logits[..., :-1, :].contiguous().float()
                        shift_labels = labels[..., 1:].contiguous()

                        loss = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            ignore_index=-100,
                        )
                        total_loss += float(loss.item())
                        total_batches += 1

    if total_batches == 0:
        return float("inf")

    return total_loss / total_batches


# =============================================================================
# HTTP Server
# =============================================================================

class ScoringServer:
    def __init__(self, model, validation_shards, device, model_version=0,
                 ps_url="", model_path="", model_dtype="float16"):
        self.model = model
        self.validation_shards = validation_shards
        self.validation_shard_map = {
            int(shard.get("shard_id")): shard
            for shard in validation_shards
            if isinstance(shard, dict) and shard.get("shard_id") is not None
        }
        self.device = device
        self.model_dtype = model_dtype
        self.model_version = model_version
        self.ps_url = ps_url.rstrip("/") if ps_url else ""
        self.model_path = model_path  # Path to current model file on disk
        self.device_info = detect_device_info(device)
        self._baseline_dir = Path(model_path).resolve().parent
        self._current_model_path = self._baseline_dir / "current_full.pt"
        self._current_version_path = self._baseline_dir / "current_version.txt"
        self.busy = False
        self.scored_count = 0
        self.total_time = 0.0
        self.last_score_at: Optional[float] = None
        self.last_loss: Optional[float] = None
        self.error_count = 0
        self._scored_ids = set()  # Idempotency tracking
        self._scored_results = {}  # Cache for idempotent responses
        self._model_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._param_dict = dict(self.model.named_parameters())
        self._validation_plan: list[dict] = []
        self._validation_plan_sample_seq_len = _resolve_sample_seq_len(self.model)
        self._prepared_validation_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._baseline_loss_cache: Dict[tuple[int, int, int, int], float] = {}
        self.started_at = time.time()
        self.last_ready_at: Optional[float] = None
        self.last_state_change_at = self.started_at
        self.state = "starting"
        self.state_detail = "initializing"
        self.ready = False
        self.accepting_scores = False
        self.last_baseline_warm_at: Optional[float] = None
        self.last_baseline_warm_ms: Optional[float] = None
        self.last_baseline_loss: Optional[float] = None
        self._refresh_validation_profile()
        self._set_runtime_state("warming", "baseline_cache", ready=False)
        threading.Thread(target=self._finish_startup_warm, daemon=True, name="startup_warm").start()

        # Start background model update loop (checks PS every 5 min)
        if self.ps_url:
            threading.Thread(target=self._model_update_loop, daemon=True,
                             name="model_update").start()
            log.info(f"[AUTO-UPDATE] Enabled, checking {self.ps_url} every 300s")

    def request_stop(self) -> None:
        if not self._stop_event.is_set():
            log.info("[LIFECYCLE] stop requested")
            self._stop_event.set()

    def _finish_startup_warm(self) -> None:
        try:
            with self._model_lock:
                self._warm_baseline_cache(reason="startup")
        except Exception:
            self._set_runtime_state("degraded", "startup_warm_failed", ready=False)
            return
        self._set_runtime_state("ready", "serving", ready=True)

    def _set_runtime_state(self, state: str, detail: str = "", *, ready: Optional[bool] = None) -> None:
        now = time.time()
        self.state = str(state or "unknown")
        self.state_detail = str(detail or "").strip()
        self.last_state_change_at = now
        if ready is not None:
            self.ready = bool(ready)
            if self.ready:
                self.last_ready_at = now
        self.accepting_scores = bool(self.ready and not self.busy)

    def _warm_baseline_cache(self, reason: str) -> Optional[float]:
        started = time.time()
        try:
            loss = float(self._get_cached_loss_before())
            elapsed_ms = (time.time() - started) * 1000.0
            self.last_baseline_warm_at = time.time()
            self.last_baseline_warm_ms = elapsed_ms
            self.last_baseline_loss = loss
            log.info(
                "[VAL] baseline cache warmed: reason=%s loss=%.6f elapsed_ms=%.0f cache_entries=%d",
                reason,
                loss,
                elapsed_ms,
                len(self._baseline_loss_cache),
            )
            return loss
        except Exception as exc:
            self.error_count += 1
            self.last_baseline_warm_at = time.time()
            self.last_baseline_warm_ms = (time.time() - started) * 1000.0
            log.error(f"[VAL] baseline cache warm failed during {reason}: {exc}", exc_info=True)
            raise

    def _status_payload(self) -> Dict[str, Any]:
        avg_ms = (self.total_time / self.scored_count * 1000) if self.scored_count > 0 else 0
        device_info = dict(self.device_info)
        accepting_scores = bool(self.ready and not self.busy)
        self.accepting_scores = accepting_scores
        return {
            "status": "ok",
            "state": self.state,
            "state_detail": self.state_detail,
            "ready": bool(self.ready),
            "accepting_scores": accepting_scores,
            "busy": self.busy,
            "device": self.device,
            "model_dtype": self.model_dtype,
            "model_version": self.model_version,
            "scored_count": self.scored_count,
            "last_score_at": self.last_score_at,
            "last_loss": self.last_loss,
            "errors": self.error_count,
            "avg_score_ms": round(avg_ms),
            "validation_shards": len(self.validation_shards),
            "gpu_model": device_info.get("gpu_model"),
            "gpu_vram_gb": device_info.get("gpu_vram_gb"),
            "cpu_model": device_info.get("cpu_model"),
            "ram_gb": device_info.get("ram_gb"),
            "os": device_info.get("os"),
            "arch": device_info.get("arch"),
            "memory_mb": get_process_memory_mb(),
            "queue_size": 1 if self.busy else 0,
            "started_at": self.started_at,
            "last_ready_at": self.last_ready_at,
            "last_state_change_at": self.last_state_change_at,
            "baseline_cache_entries": len(self._baseline_loss_cache),
            "last_baseline_warm_at": self.last_baseline_warm_at,
            "last_baseline_warm_ms": round(self.last_baseline_warm_ms) if self.last_baseline_warm_ms is not None else None,
            "last_baseline_loss": self.last_baseline_loss,
        }

    def _baseline_cache_key(self) -> tuple[int, int, int, int]:
        return (
            int(self.model_version or 0),
            int(len(self._validation_plan)),
            int(self._validation_plan_sample_seq_len),
            int(len(self.validation_shards)),
        )

    def _refresh_validation_profile(self) -> None:
        self._param_dict = dict(self.model.named_parameters())
        self._validation_plan, self._validation_plan_sample_seq_len = build_validation_plan(
            self.model,
            self.validation_shards,
            int(self.model_version or 0),
        )
        self._prepared_validation_batches = prepare_validation_batches(
            self.validation_shards,
            self._validation_plan,
            self._validation_plan_sample_seq_len,
        )
        self._baseline_loss_cache.clear()
        log.info(
            "[VAL] validation profile refreshed: model_version=%s plan_items=%s sample_seq_len=%s shards=%s prepared_batches=%s",
            int(self.model_version or 0),
            int(len(self._validation_plan)),
            int(self._validation_plan_sample_seq_len),
            int(len(self.validation_shards)),
            int(len(self._prepared_validation_batches)),
        )

    def _get_cached_loss_before(self) -> float:
        cache_key = self._baseline_cache_key()
        cached = self._baseline_loss_cache.get(cache_key)
        if cached is not None:
            return float(cached)
        loss = _compute_validation_loss(
            self.model,
            self.validation_shards,
            self.device,
            validation_plan=self._validation_plan,
            prepared_batches=self._prepared_validation_batches,
        )
        self._baseline_loss_cache[cache_key] = float(loss)
        return float(loss)

    def _persist_version_marker(self, version: int) -> None:
        tmp_version = Path(f"{self._current_version_path}.tmp")
        tmp_version.write_text(f"{int(version)}\n", encoding="utf-8")
        os.replace(tmp_version, self._current_version_path)

    def _verify_persisted_baseline(self, version: int) -> bool:
        try:
            if not self._current_model_path.exists():
                log.warning(f"[AUTO-UPDATE] baseline model missing after persist: {self._current_model_path}")
                return False
            persisted_version = _read_version_file(self._current_version_path)
            if persisted_version != int(version):
                log.warning(
                    f"[AUTO-UPDATE] baseline version marker mismatch: "
                    f"expected={int(version)} actual={persisted_version}"
                )
                return False
            model_size_mb = self._current_model_path.stat().st_size / 1e6
            log.info(
                f"[AUTO-UPDATE] Baseline verified → {self._current_model_path} "
                f"(v{int(version)}, size={model_size_mb:.1f}MB)"
            )
            return True
        except Exception as exc:
            log.error(f"[AUTO-UPDATE] baseline verification failed: {exc}", exc_info=True)
            return False

    def _promote_checkpoint_baseline(self, checkpoint_path: str, version: int) -> bool:
        source = Path(checkpoint_path)
        if not source.exists():
            log.warning(f"[AUTO-UPDATE] baseline source missing: {source}")
            return False
        try:
            self._current_model_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_model = Path(f"{self._current_model_path}.tmp")
            shutil.copy2(source, tmp_model)
            os.replace(tmp_model, self._current_model_path)
            self._persist_version_marker(version)
            self.model_path = str(self._current_model_path)
            log.info(f"[AUTO-UPDATE] Promoted checkpoint baseline → {self._current_model_path} (v{version})")
            return self._verify_persisted_baseline(version)
        except Exception as exc:
            log.error(f"[AUTO-UPDATE] failed to promote checkpoint baseline: {exc}", exc_info=True)
            return False

    def _persist_current_baseline(self, version: int) -> bool:
        try:
            self._current_model_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_model = Path(f"{self._current_model_path}.tmp")
            with self._model_lock:
                checkpoint = {
                    "model_state_dict": self.model.state_dict(),
                    "model_version": int(version),
                }
                torch.save(checkpoint, tmp_model)
            os.replace(tmp_model, self._current_model_path)
            self._persist_version_marker(version)
            self.model_path = str(self._current_model_path)
            log.info(f"[AUTO-UPDATE] Persisted current baseline → {self._current_model_path} (v{version})")
            return self._verify_persisted_baseline(version)
        except Exception as exc:
            log.error(f"[AUTO-UPDATE] failed to persist current baseline: {exc}", exc_info=True)
            return False

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

    def _score_submission_blocking(self, raw_data: bytes, learning_rate: float) -> Tuple[float, float, float]:
        payload = json.loads(raw_data)
        sparse_gradient = decompress_gradients_sparse(payload)
        with self._model_lock:
            loss_before = self._get_cached_loss_before()
            return score_gradient(
                self.model,
                sparse_gradient,
                self.validation_shards,
                self.device,
                learning_rate=float(learning_rate),
                param_dict=self._param_dict,
                validation_plan=self._validation_plan,
                prepared_batches=self._prepared_validation_batches,
                loss_before=loss_before,
            )

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
            "learning_rate": 0.001,
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
        if not self.ready:
            return web.json_response(
                {"error": "worker_not_ready", "state": self.state, "state_detail": self.state_detail},
                status=503,
            )
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

        raw_learning_rate = body.get("learning_rate")
        if raw_learning_rate is None:
            raw_learning_rate = (
                os.getenv("SCORER_DEFAULT_LEARNING_RATE", "").strip()
                or os.getenv("PS_LR", "").strip()
                or "1.0"
            )
            log.warning(
                "[SCORE] submission=%s missing learning_rate, defaulting to %s",
                sid[:12],
                raw_learning_rate,
            )
        try:
            learning_rate = float(raw_learning_rate)
        except Exception:
            return web.json_response({"error": "invalid learning_rate"}, status=400)
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            return web.json_response({"error": "invalid learning_rate"}, status=400)

        # --- Busy check (single worker, one score at a time) ---
        if self.busy:
            return web.json_response(
                {"error": "worker_busy", "submission_id": sid}, status=503
            )

        self.busy = True
        self.accepting_scores = False
        t0 = time.time()

        try:
            # --- Fetch gradient from storage (data plane) ---
            gradient_url = body["gradient_url"]
            log.info(f"[SCORE] {sid[:12]}... fetching gradient from {gradient_url}")

            fetch_started = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.get(gradient_url, timeout=aiohttp.ClientTimeout(total=FETCH_TIMEOUT)) as resp:
                    if resp.status != 200:
                        return web.json_response(
                            {"error": f"gradient_fetch_failed: HTTP {resp.status}", "submission_id": sid},
                            status=502,
                        )
                    raw_data = await resp.read()
            fetch_elapsed = time.time() - fetch_started
            log.info(
                f"[SCORE] {sid[:12]}... fetched gradient bytes={len(raw_data)} fetch_seconds={fetch_elapsed:.2f}"
            )

            score, loss_before, loss_after = await asyncio.to_thread(
                self._score_submission_blocking,
                raw_data,
                learning_rate,
            )

            elapsed_ms = int((time.time() - t0) * 1000)

            result = {
                "submission_id": sid,
                "score": round(score, 8),
                "loss_before": round(loss_before, 6),
                "loss_after": round(loss_after, 6),
                "model_version": self.model_version,
                "learning_rate": learning_rate,
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
            self.last_score_at = time.time()
            self.last_loss = float(loss_after)

            log.info(
                f"[SCORE] {sid[:12]}... done: score={score:.6f}, "
                f"loss={loss_before:.4f}→{loss_after:.4f}, {elapsed_ms}ms"
            )

            return web.json_response(result)

        except asyncio.TimeoutError:
            self.error_count += 1
            log.error(f"[SCORE] {sid[:12]}... timeout fetching gradient")
            return web.json_response(
                {"error": "gradient_fetch_timeout", "submission_id": sid}, status=504
            )
        except Exception as e:
            self.error_count += 1
            log.error(f"[SCORE] {sid[:12]}... error: {e}", exc_info=True)
            return web.json_response(
                {"error": str(e), "submission_id": sid}, status=500
            )
        finally:
            self.busy = False
            self.accepting_scores = bool(self.ready)

    # ================================================================
    # Background model auto-update (delta / full download from PS)
    # ================================================================


    def _ensure_ps_token(self, force_refresh: bool = False):
        """Return optional PS bearer token for private deployments.

        Delta updates are served from a public read-only endpoint in production,
        so scorers should not try to self-register as pseudo-miners just to fetch
        model deltas. If a private deployment wants authenticated reads, provide
        an explicit token via environment.
        """
        if force_refresh:
            self._ps_token = None
        if hasattr(self, '_ps_token') and self._ps_token:
            return self._ps_token

        token = (
            os.getenv("ALICE_PS_TOKEN", "").strip()
            or os.getenv("PS_AUTH_TOKEN", "").strip()
        )
        if token:
            self._ps_token = token
            return self._ps_token

        self._ps_token = None
        return None

    def _model_update_loop(self):
        """Background thread: check PS for model updates every 300s."""
        if self._stop_event.wait(30):
            return
        while not self._stop_event.is_set():
            try:
                self._check_and_apply_updates()
            except Exception as e:
                log.error(f"[AUTO-UPDATE] loop error: {e}", exc_info=True)
            if self._stop_event.wait(300):
                return

    def _fetch_ps_model_info(self) -> Dict[str, Any]:
        import requests as _requests  # stdlib-compat, no new dep

        resp = _requests.get(f"{self.ps_url}/model/info", timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"/model/info returned HTTP {resp.status_code}")
        info = resp.json()
        published_version = info.get("model_version")
        if published_version is None:
            published_version = info.get("version", 0)
        info["model_version"] = int(published_version or 0)
        return info

    def _normalize_remote_url(self, url: str) -> str:
        normalized = str(url or "").strip()
        if normalized.startswith("/"):
            normalized = f"{self.ps_url}{normalized}"
        return normalized.rstrip("/")

    def _candidate_download_urls_from_info(self, info: Dict[str, Any], target_version: int) -> Tuple[int, list[str]]:
        published_version = int(info.get("model_version") or info.get("version") or 0)
        if published_version <= 0:
            raise RuntimeError("no published model version available")
        if int(target_version) != published_version:
            raise RuntimeError(
                f"requested v{int(target_version)} but published version is v{published_version}"
            )

        candidates = []
        explicit = self._normalize_remote_url(str(info.get("download_url", "") or ""))
        if explicit:
            candidates.append(explicit)

        for base_url in info.get("base_urls", []) or []:
            normalized_base = self._normalize_remote_url(str(base_url or ""))
            if normalized_base:
                candidates.append(f"{normalized_base}/v{published_version}_full.pt")

        single_base = self._normalize_remote_url(str(info.get("base_url", "") or ""))
        if single_base:
            candidates.append(f"{single_base}/v{published_version}_full.pt")

        unique_candidates = []
        seen = set()
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                unique_candidates.append(candidate)
        if not unique_candidates:
            raise RuntimeError(f"no download URL available for published model v{published_version}")
        def _priority(url: str) -> tuple[int, str]:
            if explicit and url == explicit:
                return (0, url)
            if "ps.aliceprotocol.org/models" in url:
                return (1, url)
            if "huggingface.co/" in url:
                return (2, url)
            return (3, url)
        unique_candidates.sort(key=_priority)
        return published_version, unique_candidates

    def _resolve_download_url_from_info(self, info: Dict[str, Any], target_version: int) -> str:
        _, candidates = self._candidate_download_urls_from_info(info, target_version)
        for candidate in candidates:
            if candidate:
                return candidate
        raise RuntimeError("no download URL candidates available")

    def _resolve_checksum_url_from_info(self, info: Dict[str, Any], download_url: str) -> str:
        for key in ("checksum_url", "sha256_url"):
            candidate = self._normalize_remote_url(str(info.get(key, "") or ""))
            if candidate:
                return candidate
        normalized_download = self._normalize_remote_url(download_url)
        return f"{normalized_download}.sha256" if normalized_download else ""

    def _fetch_remote_checksum(self, checksum_url: str) -> str:
        import requests as _requests  # stdlib-compat, no new dep

        normalized_url = self._normalize_remote_url(checksum_url)
        if not normalized_url:
            raise RuntimeError("checksum URL missing")
        resp = _requests.get(normalized_url, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"checksum fetch failed: HTTP {resp.status_code}")
        checksum_text = resp.text.strip()
        if not checksum_text:
            raise RuntimeError("checksum response empty")
        checksum = checksum_text.split()[0].strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise RuntimeError(f"invalid checksum payload from {normalized_url}")
        return checksum

    def _sha256_file(self, path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _check_and_apply_updates(self):
        """Check PS model version; pull deltas or full model if behind."""
        if self.busy:
            log.info("[AUTO-UPDATE] worker busy, defer update check")
            return

        self.busy = True
        try:
            info = self._fetch_ps_model_info()
        except Exception as e:
            log.warning(f"[AUTO-UPDATE] failed to reach PS: {e}")
            self.busy = False
            return

        try:
            ps_version = info.get("model_version")
            if ps_version is None:
                ps_version = info.get("version", 0)
            if ps_version <= self.model_version:
                return  # Already up to date

            gap = ps_version - self.model_version
            self._set_runtime_state("updating", f"ps_v{int(ps_version)}", ready=False)
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
                self._warm_baseline_cache(reason=f"delta_v{current}_to_v{ps_version}")
                log.info(f"[AUTO-UPDATE] ✅ Incremental update complete → v{self.model_version}")
        finally:
            if not self._stop_event.is_set() and self.model is not None:
                self._set_runtime_state("ready", "serving", ready=True)
            self.busy = False

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
            if resp.status_code in (401, 403):
                log.warning(f"[AUTO-UPDATE] delta from v{from_version}: HTTP {resp.status_code}, refreshing token")
                token = self._ensure_ps_token(force_refresh=True)
                headers = {"Authorization": f"Bearer {token}"} if token else {}
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
        delta_payload is a precomputed model diff from PS /model/delta, not a miner gradient.
        It must be applied as param += delta.
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
                self._refresh_validation_profile()

            log.info(f"[AUTO-UPDATE] Applied delta v{from_version}→v{self.model_version}, {updated} params updated")
            if not self._persist_current_baseline(self.model_version):
                log.warning(f"[AUTO-UPDATE] Delta v{from_version}→v{self.model_version} applied but baseline persist failed")
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

        try:
            info = self._fetch_ps_model_info()
            download_url = self._resolve_download_url_from_info(info, target_version)
            checksum_url = self._resolve_checksum_url_from_info(info, download_url)
            expected_sha256 = self._fetch_remote_checksum(checksum_url)
        except Exception as exc:
            log.error(f"[AUTO-UPDATE] Full model v{target_version} preflight failed: {exc}")
            return

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

            actual_sha256 = self._sha256_file(tmp_path)
            if actual_sha256 != expected_sha256:
                try:
                    os.remove(tmp_path)
                except FileNotFoundError:
                    pass
                log.error(
                    f"[AUTO-UPDATE] checksum mismatch for v{target_version}: "
                    f"expected={expected_sha256} actual={actual_sha256}"
                )
                return

            os.replace(tmp_path, str(dest))
            log.info(f"[AUTO-UPDATE] Downloaded {downloaded/1e6:.1f}MB → {dest}")

            if not self._promote_checkpoint_baseline(str(dest), target_version):
                log.warning(f"[AUTO-UPDATE] Full model v{target_version} loaded but baseline promotion failed")
                return

            if self.device == "cpu":
                log.warning(
                    f"[AUTO-UPDATE] Full model v{target_version} staged on CPU baseline; "
                    "exiting for low-memory restart instead of in-process hot-reload"
                )
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(17)

            # Hot reload for non-CPU devices where memory headroom is expected.
            with self._model_lock:
                new_model, resolved_dtype_name, loaded_version = load_model(str(dest), self.device, self.model_dtype)
                self.model = new_model
                self.model_dtype = resolved_dtype_name
                self.model_version = int(loaded_version or target_version)
                self._scored_results.clear()
                self._refresh_validation_profile()
            self._warm_baseline_cache(reason=f"full_download_v{self.model_version}")
            self._set_runtime_state("ready", "serving", ready=True)

            log.info(f"[AUTO-UPDATE] ✅ Full model v{self.model_version} loaded, hot-reloaded")

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
        return web.json_response(self._status_payload())

    async def handle_status(self, request: web.Request) -> web.Response:
        payload = self._status_payload()
        payload["dtype"] = str(getattr(self.model, "dtype", self.model_dtype))
        return web.json_response(payload)

    async def handle_ready(self, request: web.Request) -> web.Response:
        payload = self._status_payload()
        if self.ready:
            return web.json_response(payload)
        return web.json_response(payload, status=503)

    async def handle_validate(self, request: web.Request) -> web.Response:
        """POST /validate — compute validation loss on held-out shards."""
        if not self.ready:
            return web.json_response(
                {"error": "worker_not_ready", "state": self.state, "state_detail": self.state_detail},
                status=503,
            )
        if self.busy:
            return web.json_response({"error": "worker_busy"}, status=503)

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)

        requested_version = body.get("model_version")
        _max_gap = int(os.environ.get("ALICE_MAX_VERSION_GAP", "50"))
        if requested_version is not None and abs(int(requested_version) - int(self.model_version)) > _max_gap:
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
        self.accepting_scores = False
        try:
            avg_loss = await asyncio.to_thread(self._validate_blocking, selected_shards)
            self.last_score_at = time.time()
            self.last_loss = float(avg_loss)
            return web.json_response(
                {
                    "status": "ok",
                    "avg_loss": round(float(avg_loss), 6),
                    "num_shards": len(selected_shards),
                    "model_version": self.model_version,
                }
            )
        except Exception as e:
            self.error_count += 1
            log.error(f"[VALIDATE] error: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)
        finally:
            self.busy = False
            self.accepting_scores = bool(self.ready)

    async def _download_model(self, url: str, dest: str, expected_sha256: Optional[str] = None) -> str:
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

        if expected_sha256:
            actual_sha256 = self._sha256_file(tmp_path)
            if actual_sha256 != expected_sha256:
                try:
                    os.remove(tmp_path)
                except FileNotFoundError:
                    pass
                raise RuntimeError(
                    f"Checksum mismatch: expected={expected_sha256} actual={actual_sha256}"
                )

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
            checksum_url = body.get("checksum_url")
            new_version = int(body.get("model_version", self.model_version + 1))

            if not model_path and not download_url:
                return web.json_response(
                    {"error": "model_path or download_url required"}, status=400
                )

            # If download_url provided, download first
            if download_url:
                try:
                    published_info = await asyncio.to_thread(self._fetch_ps_model_info)
                    _, allowed_download_urls = self._candidate_download_urls_from_info(
                        published_info,
                        new_version,
                    )
                    normalized_download_url = self._normalize_remote_url(str(download_url or ""))
                    selected_download_url = normalized_download_url or allowed_download_urls[0]
                    if selected_download_url not in allowed_download_urls:
                        return web.json_response(
                            {
                                "error": "reload_url_not_published",
                                "requested_url": selected_download_url,
                                "published_url": allowed_download_urls[0],
                                "published_urls": allowed_download_urls,
                                "published_version": int(published_info.get("model_version", 0) or 0),
                            },
                            status=409,
                        )
                    checksum_url = checksum_url or self._resolve_checksum_url_from_info(
                        published_info,
                        selected_download_url,
                    )
                    expected_sha256 = await asyncio.to_thread(self._fetch_remote_checksum, checksum_url)
                    download_url = selected_download_url
                except Exception as exc:
                    return web.json_response(
                        {
                            "error": "reload_preflight_failed",
                            "message": str(exc),
                        },
                        status=409,
                    )
                if not model_path:
                    # Default download destination
                    model_dir = Path(os.environ.get("MODEL_DIR", "/tmp/alice-models"))
                    model_path = str(model_dir / f"model_v{new_version}.pt")
                log.info(f"[RELOAD] Will download v{new_version} from URL → {model_path}")

            self.busy = True
            self.accepting_scores = False
            self._set_runtime_state("reloading", "reload_request", ready=False)

            if download_url:
                await self._download_model(download_url, model_path, expected_sha256=expected_sha256)

            log.info(f"[RELOAD] Loading model v{new_version} from {model_path}")

            # Reload model
            self.model, resolved_dtype_name, loaded_version = load_model(model_path, self.device, self.model_dtype)
            self.model_dtype = resolved_dtype_name
            self.model_version = int(loaded_version or new_version)
            self._scored_results.clear()  # Invalidate cache
            self._refresh_validation_profile()
            self._warm_baseline_cache(reason=f"reload_v{int(self.model_version)}")
            if loaded_version is not None and int(loaded_version) != int(new_version):
                log.warning(
                    f"[RELOAD] Requested v{int(new_version)} but checkpoint metadata says v{int(loaded_version)}; "
                    f"using metadata version"
                )
            effective_version = int(self.model_version)
            if not self._promote_checkpoint_baseline(model_path, effective_version):
                if not self._persist_current_baseline(effective_version):
                    log.warning(f"[RELOAD] Model v{effective_version} loaded but baseline persist failed")

            log.info(f"[RELOAD] Model v{effective_version} loaded successfully")
            self._set_runtime_state("ready", "serving", ready=True)
            return web.json_response({
                "status": "reloaded",
                "model_dtype": self.model_dtype,
                "model_version": self.model_version,
                "model_path": model_path,
                "downloaded": bool(download_url),
            })
        except Exception as e:
            log.error(f"[RELOAD] Failed: {e}", exc_info=True)
            self._set_runtime_state("degraded", "reload_failed", ready=False)
            return web.json_response({"error": str(e)}, status=500)
        finally:
            self.busy = False
            self.accepting_scores = bool(self.ready and not self.busy)


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


def register_scorer_endpoint(ps_url: str, scorer_address: str, public_endpoint: str, model_version: int) -> bool:
    if not ps_url or not scorer_address or not public_endpoint:
        return False
    try:
        import requests as _requests
        headers = {}
        scorer_key = os.environ.get("SCORER_API_KEY", "").strip()
        if scorer_key:
            headers["Authorization"] = f"Bearer {scorer_key}"

        resp = _requests.post(
            f"{ps_url.rstrip('/')}/scorer/register-endpoint",
            json={
                "address": scorer_address,
                "endpoint": public_endpoint,
                "model_version": int(model_version or 0),
            },
            headers=headers,
            timeout=20,
        )
        if resp.status_code == 200:
            log.info(f"[SCORER REGISTER] registered endpoint {public_endpoint} for {scorer_address[:12]}")
            return True
        log.warning(f"[SCORER REGISTER] failed: HTTP {resp.status_code} body={resp.text[:200]}")
        return False
    except Exception as exc:
        log.warning(f"[SCORER REGISTER] failed: {exc}")
        return False


def start_endpoint_registration_loop(
    ps_url: str,
    scorer_address: str,
    public_endpoint: str,
    model_version_ref,
    stop_event: threading.Event,
    local_ready_url: str,
):
    if not ps_url or not scorer_address or not public_endpoint:
        return

    def _loop():
        warned_not_ready = False
        while not stop_event.is_set():
            try:
                ready_resp = requests.get(local_ready_url, timeout=5)
                if ready_resp.status_code != 200:
                    if not warned_not_ready:
                        log.info(
                            "[SCORER REGISTER] waiting for local readiness before registering endpoint: %s",
                            local_ready_url,
                        )
                        warned_not_ready = True
                    if stop_event.wait(SCORER_REGISTER_BOOT_POLL_S):
                        return
                    continue
                warned_not_ready = False
                register_scorer_endpoint(
                    ps_url=ps_url,
                    scorer_address=scorer_address,
                    public_endpoint=public_endpoint,
                    model_version=int(model_version_ref()),
                )
            except Exception as exc:
                log.warning(f"[SCORER REGISTER] loop error: {exc}")
            if stop_event.wait(SCORER_REGISTER_INTERVAL_S):
                return

    threading.Thread(target=_loop, daemon=True, name="scorer_endpoint_register").start()


def parse_args():
    parser = argparse.ArgumentParser(description="Alice Scoring Worker (Phase 1)")
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--validation-dir", required=True, help="Path to validation shard directory")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"HTTP port (default: {DEFAULT_PORT})")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--device", default="cpu", help="Device: cpu, auto, cuda, mps")
    parser.add_argument("--model-dtype", default="float16", help="Model dtype: float16, bfloat16, float32, auto")
    parser.add_argument("--model-version", type=int, default=0, help="Initial model version")
    parser.add_argument("--num-val-shards", type=int, default=NUM_VALIDATION_SHARDS, help="Number of validation shards to use")
    parser.add_argument("--ps-url", default="", help="Parameter Server URL for auto-update (empty = disabled)")
    parser.add_argument("--scorer-address", default="", help="On-chain scorer address used for endpoint registration")
    parser.add_argument("--public-endpoint", default="", help="Public scorer endpoint URL, e.g. http://my-ip:8090")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device != "auto":
        os.environ["DEVICE"] = args.device
    device = detect_device()
    device_info = detect_device_info(device)
    log.info(f"Using device: {device}")
    _, resolved_dtype_name = resolve_model_dtype(args.model_dtype, device)
    log.info(f"Using model dtype: {resolved_dtype_name} (platform={platform.machine()})")
    log.info(format_device_log_line(device_info))

    resolved_model_path, resolved_model_version = resolve_startup_baseline(
        args.model_path,
        args.model_version,
    )
    if resolved_model_path != args.model_path or resolved_model_version != args.model_version:
        log.info(
            f"Resolved scorer baseline: path={resolved_model_path} version={resolved_model_version} "
            f"(requested path={args.model_path} version={args.model_version})"
        )

    # Load model
    model, resolved_dtype_name, embedded_version = load_model(resolved_model_path, device, args.model_dtype)
    if embedded_version is not None and int(embedded_version) != int(resolved_model_version):
        log.warning(
            f"Baseline version marker mismatch: resolved={int(resolved_model_version)} "
            f"checkpoint={int(embedded_version)}; using checkpoint metadata"
        )
        resolved_model_version = int(embedded_version)

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
        model_dtype=resolved_dtype_name,
        model_version=resolved_model_version,
        ps_url=args.ps_url,
        model_path=resolved_model_path,
    )

    app = web.Application()
    app.router.add_post("/score", server.handle_score)
    app.router.add_get("/health", server.handle_health)
    app.router.add_get("/status", server.handle_status)
    app.router.add_get("/ready", server.handle_ready)
    app.router.add_post("/validate", server.handle_validate)
    app.router.add_post("/reload", server.handle_reload_model)

    log.info(f"Starting scoring worker on {args.host}:{args.port}")
    log.info(f"  Device: {device}")
    log.info(f"  Model dtype: {resolved_dtype_name}")
    log.info(f"  Model version: {resolved_model_version}")
    log.info(f"  Model path: {resolved_model_path}")
    log.info(f"  Validation shards: {len(validation_shards)}")
    log.info(f"  Endpoints: POST /score, POST /validate, GET /health, GET /ready, GET /status, POST /reload")

    if args.ps_url and args.scorer_address and args.public_endpoint:
        start_endpoint_registration_loop(
            ps_url=args.ps_url,
            scorer_address=args.scorer_address,
            public_endpoint=args.public_endpoint,
            model_version_ref=lambda: server.model_version,
            stop_event=server._stop_event,
            local_ready_url=f"http://127.0.0.1:{int(args.port)}/ready",
        )
    elif args.scorer_address or args.public_endpoint:
        log.warning("[SCORER REGISTER] both --scorer-address and --public-endpoint are required for PS endpoint registration")

    async def _on_shutdown(_app):
        server.request_stop()

    app.on_shutdown.append(_on_shutdown)
    app.on_cleanup.append(_on_shutdown)
    web.run_app(app, host=args.host, port=args.port, print=None, shutdown_timeout=5.0)


if __name__ == "__main__":
    main()
