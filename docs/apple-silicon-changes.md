# Apple Silicon Metrics Changes

## Problem
On MPS (Apple Silicon), two metrics in `train.py` are broken:
- **`peak_vram_mb`** always reports `0.0` — the code only calls `torch.cuda.max_memory_allocated()`, which doesn't exist on MPS
- **`mfu_percent`** is calculated against `H100_BF16_PEAK_FLOPS = 989.5 TFLOPS`, giving ~0.3% on Mac — meaningless

## Solution
1. Auto-detect chip via `sysctl -n machdep.cpu.brand_string` (e.g. "Apple M4 Pro" → "m4-pro")
2. Parameterized `APPLE_SILICON_FLOPS` dict with per-chip FP32 peak TFLOPS
3. `torch.mps.driver_allocated_memory()` for real memory reporting on MPS

## Apple Silicon FP32 Peak TFLOPS Reference

Values derived from core counts, clock speeds, and ALU configs (Flopper.io, cpu-monkey).
Apple does not publish official FP32 TFLOPS. MPS runs FP32 (no native bfloat16).

### M4 Series (2024)

| Chip     | GPU Cores | FP32 TFLOPS | Memory BW  |
|----------|-----------|-------------|------------|
| M4       | 10        | 4.26        | 120 GB/s   |
| M4 Pro   | 20        | 9.2         | 273 GB/s   |
| M4 Max   | 40        | 18.4        | 546 GB/s   |

### M5 Series (Base: Oct 2025, Pro/Max: Mar 2026)

| Chip     | GPU Cores | FP32 TFLOPS | Memory BW  |
|----------|-----------|-------------|------------|
| M5       | 10        | 4.15        | 154 GB/s   |
| M5 Pro   | 20        | 8.3         | 307 GB/s   |
| M5 Max   | 40        | 16.6        | 614 GB/s   |

### Notes
- M5 base FP32 is slightly lower than M4 base (4.15 vs 4.26). This is misleading — M5 GPU cores include a Neural Accelerator and Apple claims 30% faster overall graphics. Raw FP32 TFLOPS doesn't capture architectural changes.
- 16-core Pro and 32-core Max variants exist; values scale linearly from the max-core configs listed above.
- Memory bandwidth is listed for reference — useful context for understanding throughput bottlenecks.

## Auto-Detection
The chip is detected automatically at startup via:
```python
import subprocess
brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
# "Apple M4 Pro" → "m4-pro"
chip_key = brand.replace("Apple ", "").lower().replace(" ", "-")
```
No manual configuration needed — just move the Mac and run.

If the chip isn't in the FLOPS dict (e.g. a new chip), a warning is printed and MFU shows 0%. Add the new chip's TFLOPS to `APPLE_SILICON_FLOPS` to fix.

## Changes Made
- `train.py` line ~526: Replaced `H100_BF16_PEAK_FLOPS` with auto-detected `APPLE_SILICON_FLOPS` dict + `DEVICE_PEAK_FLOPS`
- `train.py` lines ~658, ~689: Updated MFU calculations to use `DEVICE_PEAK_FLOPS`
- `train.py` lines ~690-693: Added `torch.mps.driver_allocated_memory()` for MPS memory reporting
