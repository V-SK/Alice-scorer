import gc
import os
import time

import torch

from src.model import AliceConfig, AliceForCausalLM


MODEL_PATH = "/root/alice-scorer/models/current_full.pt"
SEQ_LEN = 128
VOCAB_SIZE = 32000
RUNS = 10


def read_proc_kb(key: str) -> int:
    with open("/proc/self/status", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(f"{key}:"):
                parts = line.split()
                return int(parts[1])
    return 0


def rss_gb() -> float:
    return read_proc_kb("VmRSS") / 1024 / 1024


def peak_rss_gb() -> float:
    return read_proc_kb("VmHWM") / 1024 / 1024


def fmt(seconds: float) -> str:
    return f"{seconds:.1f}s"


def make_model() -> AliceForCausalLM:
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        model = AliceForCausalLM(AliceConfig())
    finally:
        torch.set_default_dtype(previous)
    return model


def load_model() -> tuple[torch.nn.Module, float]:
    start = time.perf_counter()
    model = make_model()
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(device="cpu", dtype=torch.float16)
    model.eval()
    model.requires_grad_(False)
    del checkpoint
    gc.collect()
    return model, time.perf_counter() - start


def benchmark_forward(model: torch.nn.Module, input_ids: torch.Tensor) -> list[float]:
    times = []
    with torch.inference_mode():
        _ = model(input_ids=input_ids)
        for _ in range(RUNS):
            start = time.perf_counter()
            _ = model(input_ids=input_ids)
            times.append(time.perf_counter() - start)
    return times


def benchmark_full_scoring(model: torch.nn.Module, input_ids: torch.Tensor) -> tuple[float, list[float]]:
    times = []
    with torch.inference_mode():
        start = time.perf_counter()
        for _ in range(10):
            lap = time.perf_counter()
            _ = model(input_ids=input_ids)
            times.append(time.perf_counter() - lap)
        total = time.perf_counter() - start
    return total, times


def main() -> None:
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    model, load_seconds = load_model()
    rss_after_load = rss_gb()

    input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), dtype=torch.long)

    single_forward = benchmark_forward(model, input_ids)
    full_scoring_total, full_scoring_parts = benchmark_full_scoring(model, input_ids)

    print("=== FP16 Scorer Benchmark (VPS3: Ryzen 3600, DDR4) ===")
    print(f"Model load: {fmt(load_seconds)} | RSS after load: {rss_after_load:.1f} GB")
    print(f"Single forward avg ({RUNS} runs): {sum(single_forward) / len(single_forward):.1f}s")
    print(f"Full scoring (10 forward): {full_scoring_total:.1f}s")
    print(f"Peak RSS: {peak_rss_gb():.1f} GB")
    print()
    print("Reference:")
    print("Mac Studio M1 Max: ~25-30s full scoring")
    print("VPS1 CPU (pre-FP16 path): ~68s full scoring")
    print()
    print("Single forward runs:", ", ".join(f"{t:.2f}" for t in single_forward))
    print("Full scoring forward runs:", ", ".join(f"{t:.2f}" for t in full_scoring_parts))


if __name__ == "__main__":
    main()
