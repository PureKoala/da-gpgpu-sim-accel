# GPGPU-Sim Distribution – Project Overview

Date: 2025-11-09  
Branch: `dev`

## What is this project?
This repository contains a forked distribution of GPGPU-Sim (v4.2.0) with AccelWattch power modeling enabled and several research extensions for deformable attention (DeformAttn) and FMR (a warp-cooperative tile loader optimization for attention workloads). It supports functional and timing-accurate simulation of NVIDIA-like GPUs using PTX-level execution.

Key use-cases:
- Reproducing GPU micro-architecture behaviors (warp scheduling, SIMT stack, memory coalescing, caches/DRAM)
- Modeling Deformable Attention kernels with specialized tile loading (FMR) paths
- Evaluating performance and power trade-offs under configurable architectures

## Repository layout (high-level)
- `aerialvision/`, `bin/aerialvision.py` – GUI tools for PTX visualization and debugging.
- `build/`, `lib/` – Build artifacts and static libraries (organized by compiler/CUDA version).
- `configs/` – Simulator configs (SM/Caches/DRAM). See tested configs subfolders for reference.
- `cuobjdump_to_ptxplus/` – PTX+ tools (lexer/parser) used by the simulator.
- `doc/` – Documentation (this overview file, doxygen configs, etc.).
- `libcuda/`, `libopencl/` – CUDA/OpenCL runtime shims used by GPGPU-Sim.
- `src/` – Core simulator sources:
  - `cuda-sim/` – PTX execution engine (functional model), instruction semantics.
  - `gpgpu-sim/` – Timing model: pipelines, caches, DRAM, interconnects.
  - `intersim2/` – On-chip/off-chip interconnect modeling.
- `deAttn/`, `deAttn_fp16/`, `deAttn_fp16_fmr/`, `deAttn_test_fmr/` – Deformable Attention kernels, runner scripts, and FMR research variants (FP16, timing-model focused). Each subproject contains its own `src/`, `sim/`, and `run.sh`/`syn_gpgpu_sim.sh` helpers.
- `memory-bank/` – Design notes and deep dives on memory banks, FMR plans, reviews, and timing model considerations.
- `scripts/` – Utility scripts (e.g., `gen_ptxinfo`).

## Quick start
### Prerequisites
- Linux (x86_64)
- CUDA toolkit (e.g., 11.3 as used by current build trees)
- GCC toolchain (7.5.0 compatible builds present; newer toolchains supported via dedicated build dirs)

### Build the simulator
From repo root:
```bash
# 1) Setup env (paths, CUDA version detection, etc.)
source setup_environment

# 2) Build (will populate build/gcc-*/cuda-*/release)
make -j$(nproc)
```

### Run the Deformable Attention + FMR test
A curated runner is provided under `deAttn_fp16_fmr`:
```bash
cd deAttn_fp16_fmr
bash run.sh
```
The script compiles `test.cu` and DeformAttn CUDA sources (with FMR optimization), then runs under GPGPU-Sim using an RTX3070-like config and writes outputs to `out/`.

## Architecture at a glance
- Functional model (PTX execution):
  - Implements PTX instruction semantics per thread: `src/cuda-sim/instructions.cc`.
  - Thread execution entry: `ptx_thread_info::ptx_exec_inst()` in `src/cuda-sim/cuda-sim.cc`.
  - SIMT control flow (post-dominator stack) maintained in timing model and consulted for fetch/issue.
- Timing model (micro-architecture):
  - Warp schedulers, scoreboarding, pipeline control, caches and DRAM in `src/gpgpu-sim/`.
  - Memory coalescing and transaction generation in `warp_inst_t::generate_mem_accesses()`.

### FMR – warp-cooperative tile loader (DeformAttn)
- Entry in PTX: `call __fmr_sample(...)` in kernel.
- Intercepted in functional executor to trigger a warp-level tile load (TMA-like batching of GMEM addresses):
  - Functional path: `ld_sample_fmr_impl_functional()` copies data for correctness.
  - Timing path: `ld_sample_fmr_impl()` sets `inst.op = FMR_SAMPLE_OP`, batches addresses via `inst.set_addr()` and activates slots for `generate_mem_accesses()`.
- Goal: model interleaved SMEM writes and realistic global memory transactions for the tile fetch.

## Extending and developing
- Add new PTX instruction behaviors in `src/cuda-sim/instructions.cc`, wire opcodes in `opcodes.def`.
- Modify memory subsystem policies in `src/gpgpu-sim/` (e.g., `gpu-cache.cc`, `dram*.cc`).
- Adjust timing parameters via configuration files in `configs/`.
- DeformAttn runners (`deAttn*` folders) are self-contained harnesses for compiling and running target kernels on the simulator.

## Common troubleshooting
- “Run `source setup_environment` before `make`”: environment not initialized – source the script at repo root.
- OpenCL warnings during build: benign if you’re not using OpenCL; set NVOPENCL paths to enable.
- PTX assert on PC mismatch: indicates functional/timing desynchronization. Re-run with debug prints or consult notes in `memory-bank/`.
- CUDA include headers not found: ensure `CUDA_HOME`/`/usr/local/cuda-*` matches your installed toolkit.

## Useful files
- `README.md` – Upstream GPGPU-Sim readme and usage notes.
- `doc/doxygen/` – Doxygen configuration (generate API docs if needed).
- `memory-bank/` – In-depth design/analysis notes (FMR design, bank conflicts, timing reviews).

## License
See `COPYRIGHT` for GPGPU-Sim licensing terms.

## Acknowledgements
This distribution builds upon the GPGPU-Sim project and integrates research changes for Deformable Attention and FMR modeling.
