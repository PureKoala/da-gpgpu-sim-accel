# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a forked distribution of GPGPU-Sim (v4.2.0) with AccelWattch power modeling and research extensions for deformable attention (DeformAttn) and FMR (warp-cooperative tile loader optimization). It's a cycle-level GPU simulator that models NVIDIA-like GPUs using PTX-level execution.

## Current Development Status (2025-01-23)

**Active Feature**: Deformable Attention Hardware Accelerator Implementation
**Current Phase**: P1 - Functional Verification (Numerical Correctness Testing)
**Last Milestone**: ✅ P0 Complete - Parameter passing, debug output, compilation verified
**Next Milestone**: 🧪 P1 - Baseline vs Optimized numerical consistency

### Quick Status Check
- ✅ Function call interception working
- ✅ All 4 `deform_*_impl()` functions with complete parameter reading
- ✅ Compilation passes (gcc-13.3.0)
- ✅ Debug output via `-gpgpu_tensorcore_debug 1`
- ⏳ Awaiting numerical verification tests
- ⏳ Performance statistics not yet implemented

### For New Contributors
If working on DeformAttn implementation, refer to:
- `DevDocs/IMPLEMENTATION.md` - Current implementation details and plans (PRIMARY SOURCE)
- `DevDocs/TODO.md` - Detailed task list and priorities (TASK TRACKING)
- `DevDocs/ARCHITECTURE.md` - Historical design documentation (ARCHIVED - for reference only)

**Recent Changes (2025-01-23)**:
- ✅ P0 Complete: Full parameter passing implementation in all 4 `deform_*_impl()` functions
- ✅ Debug output added: Use `-gpgpu_tensorcore_debug 1` to see PCB/TBC/TMA/INTERP details
- ✅ Compilation verified: Successfully builds with gcc-13.3.0
- 🧪 P1 In Progress: Numerical verification testing (Baseline vs Optimized comparison)

**Focus Areas for Current Work**:
1. **P1 Testing**: Run and compare Baseline vs Optimized kernels (highest priority)
2. **Debug Analysis**: Verify parameter passing through debug logs
3. **Edge Cases**: Test zero-weight, out-of-bounds, and discrete scenarios
4. **DO NOT**: Attempt to integrate execution units or pipeline dispatch - not used in this architecture

## Build Commands

### Prerequisites
- Set `CUDA_INSTALL_PATH` environment variable (e.g., `/usr/local/cuda`)
- Ensure `nvcc` is in your PATH

### Build the Simulator
```bash
# 1. Setup environment (required before any build)
source setup_environment

# 2. Build (release mode by default)
make -j$(nproc)

# 3. Build with debug symbols
source setup_environment debug
make

# 4. Build documentation
make docs

# 5. Clean build
make clean
```

### CMake Build (Alternative)
```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j$(nproc)
```

### Running Deformable Attention Tests
```bash
cd deAttn_fp16_fmr
bash run.sh
```

## Architecture

### Core Components

#### 1. Functional Model (PTX Execution)
- **Location**: `src/cuda-sim/`
- **Key Files**:
  - `instructions.cc` - PTX instruction semantics per thread
  - `cuda-sim.cc` - Thread execution entry (`ptx_thread_info::ptx_exec_inst()`)
  - `ptx.y/ptx.l` - PTX parser (Bison/Flex)
  - `opcodes.def` - Instruction opcode definitions
- **Purpose**: Executes PTX instructions functionally (correctness)

#### 2. Timing Model (Micro-architecture)
- **Location**: `src/gpgpu-sim/`
- **Key Files**:
  - `gpu-sim.cc/h` - Main GPU simulation engine
  - `shader.*` - Shader core, warp schedulers, pipelines
  - `gpu-cache.cc/h` - Cache hierarchy (L1/L2)
  - `dram.cc/h` - DRAM controller
  - `l2cache.cc/h` - L2 cache
- **Purpose**: Models timing, pipelines, memory hierarchy, interconnects

#### 3. Interconnect Simulator
- **Location**: `src/intersim2/`
- **Purpose**: On-chip/off-chip network modeling

#### 4. Power Model (AccelWattch)
- **Location**: `src/accelwattch/`
- **Purpose**: Power consumption modeling for GPU components

#### 5. Runtime Shims
- **Location**: `libcuda/`, `libopencl/`
- **Purpose**: CUDA/OpenCL runtime interceptors that redirect to GPGPU-Sim

### Deformable Attention Extensions

#### FMR (Warp-Cooperative Tile Loader)
- **Location**: `deAttn_fp16_fmr/`
- **Integration Points**:
  - PTX instruction: `call __fmr_sample(...)`
  - Functional path: `ld_sample_fmr_impl_functional()` in `src/cuda-sim/instructions.cc`
  - Timing path: `ld_sample_fmr_impl()` sets `inst.op = FMR_SAMPLE_OP`
- **Purpose**: Optimized tile loading for attention workloads

#### Deformable Attention Accelerator
- **Location**: `src/cuda-sim/` (implementation), `DevDocs/` (design docs)
- **Status**: P0 Complete (compilation verified), P1 In Progress (numerical testing)
- **Components**: 5-stage pipeline (PCB → GTC → TBC → TMA & Storage → Interpolation)
- **Integration**: Function call interception ONLY (no PTX pseudo-instructions)
- **See Section**: "Deformable Attention Implementation Notes" for complete details

### Algorithm Requirements (from Orient.md)

#### Baseline vs Optimized Comparison

| Feature | Baseline (Naive) | Optimized (5-Stage Pipeline) |
|---------|------------------|------------------------------|
| **稀疏剪枝** | ❌ 无 | ✅ PCB (1 cycle) |
| **稀疏感知计算** | ❌ 无 | ✅ GTC (0 cycle, 并行) |
| **空间聚合决策** | ❌ 无 | ✅ TBC (4-7 cycles, 自适应) |
| **数据加载策略** | ❌ 直接离散访问 | ✅ TMA & Storage (22 cycles) |
| **计算单元** | ❌ GPU流水线串行 | ✅ Interpolation (3 cycles) |
| **Shared Memory** | ❌ 无 | ✅ 有 (Tile Buffer) |
| **Bank冲突** | ❌ 高 (68.3%) | ✅ 0% (自适应映射) |
| **总延迟** | ❌ 不可控 (依赖内存) | ✅ 可控 (30-33 cycles) |
| **性能** | ❌ 最差情况 | ✅ 优化后 (4-8×提升) |

#### Module Mapping to Kernel Execution Flow

| 优化阶段 | 对应模块 | 功能 | 延迟 | 关键技术 |
|---------|---------|------|------|----------|
| **阶段 I: 稀疏剪枝** | **PCB (Pre-Check Block)** | 权重预筛选，生成稀疏掩码 | 1 cycle (固定) | 阈值比较，mask生成 |
| **阶段 II: 稀疏感知计算** | **GTC (Gated Tensor Core)** | 操作数隔离，标记有效坐标 | 0 cycle (并行) | 无效点坐标置零 |
| **阶段 III: 空间聚合决策** | **TBC (Tile Boundary Check)** | 自适应聚合决策，决定加载策略 | 4-7 cycles (自适应) | 两阶段决策，模式检测 |
| **阶段 IV: 自适应数据加载** | **TMA & Storage** | Tile加载 + 无冲突存储管理 | 22 cycles (16+6) | 自适应Bank映射，Tracker管理 |
| **阶段 V: 无冲突计算** | **Interpolation** | 双线性插值计算 | 3 cycles | 无冲突读取，Swizzling |

#### Total Latency Budget
- **低聚集度场景**（Discrete模式）: ~30 cycles
  - PCB(1) + GTC(0) + TBC(4) + TMA(16) + Storage(6) + Interpolation(3) = 30 cycles
- **高聚集度场景**（Tile模式）: ~33 cycles
  - PCB(1) + GTC(0) + TBC(7) + TMA(16) + Storage(6) + Interpolation(3) = 33 cycles

#### DeformAttn Test Projects
- **Location**: `deAttn/`, `deAttn_fp16/`, `deAttn_fp16_fmr/`, `deAttn_test_fmr/`
- **Purpose**: CUDA kernels and test harnesses for Deformable Attention workloads
- **Usage**: Self-contained harnesses for compiling and running target kernels on simulator

## Development Workflow

### Adding New PTX Instructions
1. Add opcode in `src/cuda-sim/opcodes.def`
2. Implement semantics in `src/cuda-sim/instructions.cc`
3. Update parser if needed in `src/cuda-sim/ptx.y`

### Modifying Timing Models
- Cache: `src/gpgpu-sim/gpu-cache.cc`
- DRAM: `src/gpgpu-sim/dram.cc`
- Pipeline: `src/gpgpu-sim/shader.*`

### Running Simulations
```bash
# Setup environment
source setup_environment

# Copy config files to application directory (use RTX 3070 for DeformAttn)
cp configs/RTX3070/* /path/to/app/

# Run application (uses GPGPU-Sim libraries instead of real CUDA)
./your_cuda_application
```

### Configuration Files
- Location: `configs/`
- Key configs: `QuadroFX5800/`, `GTX480/`, **`RTX3070/` (Primary test target)**
- Common options:
  - `-gpgpu_ptx_force_max_capability <version>`
  - `-power_simulation_enabled 1`
  - `-gpgpu_deform_attn_avail 1`

**Target GPU for DeformAttn Testing**: RTX 3070
- Use config from `configs/RTX3070/` or `deAttn_fp16_fmr/gpgpusim.config`
- SM version: 8.6 (Ampere architecture)

## Key Files Reference

### Core Simulator
- `src/abstract_hardware_model.h` - Hardware model abstractions
- `src/cuda-sim/ptx_parser.cc` - PTX instruction parser
- `src/gpgpu-sim/gpu-sim.cc` - Main simulation loop
- `src/gpgpusim_entrypoint.cc` - Entry point for intercepted CUDA calls

### Configuration & Build
- `setup_environment` - Environment setup script (must source before build)
- `Makefile` - Top-level build system
- `CMakeLists.txt` - CMake build configuration
- `configs/RTX3070/` - RTX 3070 configuration (primary test target)

## Common Tasks

### Clean Build After Changes
```bash
make clean
source setup_environment [debug|release]
make -j$(nproc)
```

### Run Single Test Application
```bash
cd /path/to/test
source setup_environment
cp configs/RTX3070/* .  # Use RTX 3070 config (primary test target)
./test_application
```

### Generate Documentation
```bash
make docs
# View at: doc/doxygen/html/index.html
```

### Profile Deformable Attention
```bash
cd deAttn_fp16_fmr
bash run.sh
cat out/test_RTX3070.txt
```

## Environment Variables

- `CUDA_INSTALL_PATH` - Path to CUDA toolkit (required)
- `GPGPUSIM_ROOT` - Root directory of GPGPU-Sim (auto-set by setup_environment)
- `LD_LIBRARY_PATH` - Modified by setup_environment to use GPGPU-Sim libraries
- `OPENCL_REMOTE_GPU_HOST` - For OpenCL remote compilation (optional)

## Deformable Attention Implementation Notes

### ⚠️ CRITICAL ARCHITECTURE DECISION: Function Call Only

**Implementation Strategy**:
- ✅ **Function Call Interception ONLY**: All functionality via `cuda-sim.cc` string matching
- ❌ **NO PTX Pseudo-Instructions**: CUDA compiler cannot recognize custom PTX opcodes
- ❌ **NO Execution Units**: `deform_attn_exec_unit` class NOT used (exists for reference only)
- ❌ **NO Pipeline Dispatch**: No integration in `shader.cc` `issue_warp()` or `create_exec_pipeline()`
- ❌ **NO Opcode Definitions**: No entries in `opcodes.def`

**Why This Approach**:
- CUDA/nvcc compilation produces PTX that cannot reference custom instructions
- Function call interception is the ONLY viable path without modifying CUDA toolkit
- Successfully proven by FMR implementation (`ld_sample_fmr`)

### Integration Approach
- **Function Call Interception**: Use CUDA Function Call mechanism to intercept DeformAttn operations
- **Fixed Latency Model**: Each module has fixed cycle count (1-16 cycles) for simplicity
- **Functional Model**: Focus on functional correctness with fixed latency simulation
- **No Timing-Level**: No register-transfer-level (RTL) implementation - this is a functional model

### Implementation Status

#### ✅ Completed (as of 2025-01-23)
1. **Core Data Structures** (`src/cuda-sim/deform_attn_unit.h/.cc`)
   - 5-stage pipeline: PCB → GTC → TBC → TMA & Storage → Interpolation
   - Top-level unit: `deform_attn_unit` with pipeline management
   - Fixed latency: 1 + 0 + 4-7 + 16 + 6 + 3 = ~30-33 cycles

2. **Function Call Interception** (`src/cuda-sim/cuda-sim.cc`)
   - `__deform_pcb()` → `deform_pcb_impl()` (1 cycle)
   - `__deform_tbc()` → `deform_tbc_impl()` (4-7 cycles)
   - `__deform_tma()` → `deform_tma_impl()` (16 cycles)
   - `__deform_interp()` → `deform_interp_impl()` (3 cycles)
   - **P0 COMPLETE**: Full parameter passing implementation using `.param` space

3. **Configuration Options** (`src/gpgpu-sim/gpu-sim.cc`)
   - 8 configuration options for latency and availability
   - Example: `-gpgpu_deform_pcb_latency 1`

4. **Functional Model Implementation** (`src/cuda-sim/instructions.cc`)
   - All 4 `deform_*_impl()` functions with complete parameter reading
   - Debug output support via `-gpgpu_tensorcore_debug 1`
   - PCB: Displays pruning rate and valid point counts
   - TBC: Displays bounding box, tile parameters, access patterns

5. **Test Files** (`deAttn_optim/`, `deAttn_base/`)
   - Baseline kernel: Numerical reference implementation
   - Optimized kernel: Hardware-accelerated with fallback paths
   - Test harness and build system

6. **Compilation Verified** ✅
   - Successfully compiles with gcc-13.3.0
   - All DeformAttn support modules integrated

#### ⏳ In Progress / Pending
1. **P1 - Functional Verification** (Current Priority)
   - Numerical consistency testing: Baseline vs Optimized
   - Run with `-gpgpu_deform_attn_avail 0` (Baseline)
   - Run with `-gpgpu_deform_attn_avail 1` (Optimized)
   - Verify output difference < 1e-3 using `torch.allclose()`

2. **P2 - Performance Statistics** (Next Phase)
   - Add performance counters to `shader_core_stats`
   - PCB pruning rate statistics
   - TBC mode distribution (Discrete/Tile16/Tile32h/Tile32v)
   - TMA tile loading and reuse statistics

3. **P2 - Edge Case Testing**
   - Zero-weight scenarios (full PCB pruning)
   - Discrete coordinates (no aggregation in TBC)
   - Out-of-bounds coordinates (TMA boundary checks)

### Implementation Phases (Updated 2025-01-23)

1. **Phase 1: Function Call Interception** ✅ **COMPLETED**
   - CUDA Function Call interception mechanism
   - 4 function interception handlers in `cuda-sim.cc`

2. **Phase 2: Functional Model Implementation** ✅ **COMPLETED**
   - 5 module functional models with fixed latency
   - Complete parameter reading using `.param` space (FMR-style)
   - Debug output support integrated

3. **Phase 3: Compilation & Integration** ✅ **COMPLETED**
   - Successful compilation with gcc-13.3.0
   - Configuration system integrated into `gpu-sim.cc`
   - Baseline and Optimized kernels implemented

4. **Phase 4: Validation & Testing** 🧪 **IN PROGRESS**
   - P1: Numerical correctness verification (Current)
   - Edge case testing (Zero weights, OOB coordinates, discrete patterns)
   - Debug log analysis to verify execution flow

5. **Phase 5: Performance Statistics** ⏳ **PENDING**
   - Performance counters (pruning rate, mode distribution, tile reuse)
   - Latency histograms per stage
   - Memory access statistics

6. **Phase 6: Analysis & Optimization** ⏳ **PENDING**
   - Performance profiling and bottleneck analysis
   - Parameter sensitivity studies
   - Final experimental data for publication

### Key Design Decisions
- **Function Call Interception ONLY**: NO PTX pseudo-instructions, NO execution units, NO pipeline dispatch
  - Reason: CUDA compiler cannot recognize custom PTX instructions
  - All functionality achieved through function call interception in `cuda-sim.cc`
- **Fixed Latency Model**: Each module has fixed cycle count (1-16 cycles) for simplicity
  - PCB: 1 cycle, TBC: 4-7 cycles (adaptive), TMA: 16 cycles, Storage: 6 cycles, Interpolation: 3 cycles
- **Adaptive Behavior**: TBC adapts based on sampling point density (First-8 voting mechanism)
- **Conflict-Free Storage**: Self-adaptive Bank mapping to eliminate conflicts
- **Parameter Passing**: Uses `.param` space reading (FMR-style implementation)

### Development Tips
- **Function Call First**: All hardware acceleration via function interception, NOT execution units
- **Reference FMR**: Parameter passing mechanism follows `ld_sample_fmr_impl()` pattern
- **Test Incrementally**: Use `-gpgpu_tensorcore_debug 1` to verify parameter correctness
- **Focus on Functionality**: Functional correctness is the priority over timing accuracy
- **No Pipeline Dispatch**: The `deform_attn_exec_unit` class exists but is NOT used

### Configuration Example
```
-gpgpu_deform_attn_avail 1
-gpgpu_num_deform_units 1
-gpgpu_deform_pcb_latency 1
-gpgpu_deform_tbc_phase1_latency 4
-gpgpu_deform_tbc_phase2_latency 3
-gpgpu_deform_tma_latency 16
-gpgpu_deform_storage_latency 6
-gpgpu_deform_interp_latency 3
```

### CUDA Usage Example
```cpp
// Device function declarations (see deAttn_optim/src/deform_attn/cuda/deform_attn_accelerator.cuh)
extern "C" __device__ void __deform_pcb(float* weights, float threshold, bool enable, bool* mask);
extern "C" __device__ void __deform_tbc(float* abs_coords, bool* valid_coords, int* mode, ...);
extern "C" __device__ void __deform_tma(float* tile_data, unsigned long long gmem_base, ...);
extern "C" __device__ void __deform_interp(float* output, float* tile_data, ...);

// Usage: See deAttn_optim/src/deform_attn/cuda/ms_deform_attn_im2col_cuda.cuh for complete example
```


## Testing DeformAttn Implementation

### Current Testing Phase: P1 Functional Verification

**Objective**: Verify numerical correctness between Baseline and Optimized kernels

```bash
# Step 1: Build GPGPU-Sim with DeformAttn support
cd $GPGPUSIM_ROOT
source setup_environment
make clean && make -j$(nproc)

# Step 2: Run Baseline (reference implementation)
cd deAttn_base
make
./test_deform_attn > baseline_output.txt

# Step 3: Run Optimized (hardware-accelerated)
cd ../deAttn_optim
make
export GPGPUSIM_CONFIG_FILE=./gpgpusim.config
# Ensure gpgpusim.config contains:
#   -gpgpu_deform_attn_avail 1
#   -gpgpu_tensorcore_debug 1  # For debug output
./test_deform_attn > optimized_output.txt

# Step 4: Numerical comparison
# Expected: Output difference < 1e-3
python -c "
import numpy as np
baseline = np.loadtxt('baseline_output.txt')
optimized = np.loadtxt('optimized_output.txt')
diff = np.abs(baseline - optimized)
print(f'Max difference: {diff.max()}')
print(f'Mean difference: {diff.mean()}')
assert diff.max() < 1e-3, 'Numerical consistency check failed'
print('✅ Numerical verification PASSED')
"
```

### Debug Output Analysis

Enable detailed logging to verify parameter passing:

```bash
# Add to gpgpusim.config
-gpgpu_tensorcore_debug 1

# Expected debug output format:
# [PCB] Threshold: 0.05, Valid points: 187/256 (73.0%), Pruning rate: 27.0%
# [TBC] BBox: (10,15)-(26,31), Mode: Horizontal, Tile size: 16x16
# [TMA] Loading tile from GMEM@0x... to SMEM@0x..., Mode: 0 (Horizontal)
# [INTERP] Local coord: (3.2, 7.8), Result: 0.xyz
```

### Known Issues & Troubleshooting

1. **Function calls not intercepted**
   - Check: `grep "deform_pcb" simulator_output.txt` should show calls
   - Verify: Function names match exactly (`__deform_pcb`, etc.)
   - Debug: Add printf in `cuda-sim.cc` interception code

2. **Numerical differences > 1e-3**
   - Check: Baseline and Optimized use same input data
   - Verify: Fallback path in Optimized kernel works correctly
   - Debug: Compare intermediate values (after PCB, after TBC, etc.)

3. **Compilation errors**
   - Verify: gcc version (tested with gcc-13.3.0)
   - Check: CUDA_INSTALL_PATH is set correctly
   - Clean: `make clean` before rebuilding

## Quick Reference

### Target GPU Configuration
- **Primary GPU**: NVIDIA RTX 3070 (Ampere, SM 8.6)
- **Config Location**: `configs/RTX3070/` or `deAttn_fp16_fmr/gpgpusim.config`
- **Test Output**: `deAttn_fp16_fmr/out/test_RTX3070.txt`

### DeformAttn Key Commands
```bash
# Build simulator
source setup_environment && make -j$(nproc)

# Run DeformAttn test (RTX 3070 config)
cd deAttn_fp16_fmr && bash run.sh

# Check results
cat out/test_RTX3070.txt
```

### Debug Configuration
```bash
# Enable debug output in gpgpusim.config
-gpgpu_tensorcore_debug 1      # PCB/TBC/TMA/INTERP debug logs
-gpgpu_deform_attn_avail 1     # Enable DeformAttn accelerator
```

## General Notes

- This is a research simulator - performance differs from real hardware
- Functional model (PTX) is cycle-accurate for instruction semantics
- Timing model (micro-arch) is configurable for different GPU architectures
- DeformAttn extensions are functional models with fixed latencies (no timing-level implementation)
- **DeformAttn Status**: Compilation complete ✅, Numerical verification in progress 🧪
- **Target GPU**: RTX 3070 (Ampere SM 8.6)