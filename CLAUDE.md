# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a forked distribution of GPGPU-Sim (v4.2.0) with AccelWattch power modeling and research extensions for deformable attention (DeformAttn) and FMR (warp-cooperative tile loader optimization). It's a cycle-level GPU simulator that models NVIDIA-like GPUs using PTX-level execution.

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

#### Deformable Attention Function Model (In Development)
- **Location**: `DevDocs/` (design docs), `src/` (implementation)
- **Design Documents**:
  - `DevDocs/DeformAttn_Architecture.md` - Architecture design and module specifications
  - `DevDocs/DeformAttn_Development_Guide.md` - Implementation guide and development workflow
  - `DevDocs/Orient.md` - Algorithm requirements and module mapping (CRITICAL)
- **Components** (5-stage pipeline):
  - **PCB** (Pre-Check Block): Weight pruning, generates sparse mask (1 cycle)
  - **GTC** (Gated Tensor Core): Sparse-aware computation with operand isolation (0 cycle, parallel)
  - **TBC** (Tile Boundary Check): Boundary check + mode detection (4-7 cycles, adaptive)
  - **TMA & Storage** (Tile Memory Access): Adaptive tile loading + conflict-free storage (16+6 cycles)
  - **Interpolation**: Bilinear interpolation with conflict-free access (3 cycles)
- **Integration Strategy**:
  - Use CUDA Function Call interception mechanism
  - Map operations to pseudo-instructions (opcodes 0xD0-0xD3)
  - Functional model + fixed latency simulation (no timing-level implementation)
  - Target: Function correctness, end-to-end latency estimation, performance data collection

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

# Copy config files to application directory
cp configs/QuadroFX5800/* /path/to/app/

# Run application (uses GPGPU-Sim libraries instead of real CUDA)
./your_cuda_application
```

### Configuration Files
- Location: `configs/`
- Key configs: `QuadroFX5800/`, `GTX480/`
- Common options:
  - `-gpgpu_ptx_force_max_capability <version>`
  - `-power_simulation_enabled 1`
  - `-gpgpu_deform_attn_enable 1`

## Testing

### Unit Tests
- Individual modules in `src/cuda-sim/` and `src/gpgpu-sim/`
- DeformAttn tests in `deAttn_test_fmr/`

### Regression Tests
- Use Docker for consistent testing (see README.md)
- Command: `docker run --privileged ... aamodt/gpgpu-sim_regress:latest`

### Debugging
```bash
# Build debug version
source setup_environment debug
make

# Run with gdb
gdb --args ./your_application
```

### Performance Profiling
- Enable tracing: Set `TRACE=1` during build
- Stats output: Check `gpgpu_sim` stats in simulation output
- DeformAttn profiling: See `deAttn_fp16_fmr/PERFORMANCE_PROFILING.md`

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
- `configs/` - Architecture configuration files

### Deformable Attention
- `deAttn/README.md` - DeformAttn algorithm overview
- `deAttn_fp16_fmr/PROJECT_OVERVIEW.md` - FMR extension details
- `DevDocs/DeformAttn_Architecture.md` - Architecture design and module specifications
- `DevDocs/DeformAttn_Development_Guide.md` - Implementation guide and development workflow
- `DevDocs/FMR_Implementation_Summary.md` - FMR implementation reference

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
cp configs/QuadroFX5800/* .
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

### Integration Approach
- **Function Call Interception**: Use CUDA Function Call mechanism to intercept DeformAttn operations
- **Fixed Latency Model**: Each module has fixed cycle count (1-16 cycles) for simplicity
- **Functional Model**: Focus on functional correctness with fixed latency simulation
- **No Timing-Level**: No register-transfer-level (RTL) implementation - this is a functional model

### Implementation Status

#### ✅ Completed
1. **Core Data Structures** (`src/cuda-sim/deform_attn_unit.h/.cc`)
   - 5-stage pipeline: PCB → GTC → TBC → TMA & Storage → Interpolation
   - Top-level unit: `deform_attn_unit` with流水线管理
   - Fixed latency: 1 + 0 + 4-7 + 16 + 6 + 3 = ~30-33 cycles

2. **Function Call Interception** (`src/cuda-sim/cuda-sim.cc`)
   - `__deform_pcb()` → `deform_pcb_impl()` (1 cycle)
   - `__deform_tbc()` → `deform_tbc_impl()` (4-7 cycles)
   - `__deform_tma()` → `deform_tma_impl()` (16 cycles)
   - `__deform_interp()` → `deform_interp_impl()` (3 cycles)

3. **Configuration Options** (`src/gpgpu-sim/gpu-sim.cc`)
   - 10 configuration options for latency and availability
   - Example: `-gpgpu_deform_pcb_latency 1`

4. **Pipeline Integration** (`src/gpgpu-sim/shader.h`)
   - Added `ID_OC_DEFORM`, `OC_EX_DEFORM` pipeline stages
   - Added `deform_attn_exec_unit` class declaration
   - Added `deform_attn_unit *m_deform_attn_unit` member
   - Added DeformAttn configuration options

5. **Test Files** (`deAttn_fp16_fmr/`)
   - `test_deform_attn.cu`: CUDA test kernel
   - `Makefile.test`: Test build system

#### ⏳ Pending
1. **shader.cc Integration** (Highest Priority)
   - Implement `deform_attn_exec_unit` class
   - Create DeformAttn unit in `create_exec_pipeline()`
   - Add instruction dispatch in `issue_warp()`
   - Initialize configuration in `shader_core_config` constructor

2. **Performance Statistics** (`src/gpgpu-sim/stats.cc`)
   - Add DeformAttn performance counters
   - Collect cycle counts, operation counts, etc.

3. **Compilation & Testing**
   - Compile GPGPU-Sim with new code
   - Run test programs
   - Verify functionality

### Implementation Phases (Updated)

1. **Phase 1: Function Call Interception** ✅ **COMPLETED**
   - CUDA Function Call 拦截机制
   - 4 个函数的拦截逻辑

2. **Phase 2: Functional Model Implementation** ✅ **COMPLETED**
   - 5 个模块的功能模型（固定延迟）
   - Top-Level unit 管理

3. **Phase 3: Pipeline Integration** ⏳ **IN PROGRESS**
   - `deform_attn_exec_unit` 类实现
   - 集成到 `shader_core_ctx`
   - 指令分发逻辑

4. **Phase 4: Memory Integration** ⏳ **PENDING**
   - TMA 与 L2 Cache 集成
   - Storage 与 Shared Memory 集成
   - Bank 冲突检测

5. **Phase 5: Performance Statistics** ⏳ **PENDING**
   - 添加性能计数器
   - 添加延迟直方图
   - 添加访存统计

6. **Phase 6: Validation & Debugging** ⏳ **PENDING**
   - 单元测试
   - 集成测试
   - 性能分析

### Key Design Decisions
- **Fixed Latency**: Each module has fixed cycle count (1-16 cycles) for simplicity
- **Function Call Interception Only**: No pseudo-instructions (FMR's `ld.sample.fmr` 实际未使用)
- **Adaptive Behavior**: TBC adapts based on sampling point density (low/high聚集度)
- **Conflict-Free Storage**: Self-adaptive Bank mapping to eliminate conflicts
- **Simplified CAM**: Use hash table instead of actual CAM for tracker management

### Development Tips
- **Start with shader.cc**: Implement `deform_attn_exec_unit` first
- **Reference FMR**: 80% of code structure can be借鉴
- **Test Incrementally**: Test each module separately
- **Use Fixed Latency**: No need to simulate dynamic memory delays
- **Focus on Functionality**: Functional correctness is the priority

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
extern "C" __device__ void __deform_pcb(float* weights, float threshold, bool enable, bool* mask);
extern "C" __device__ void __deform_tbc(float* abs_coords, bool* valid_coords, int* mode, int* tile_size, int* base_x, int* base_y);
extern "C" __device__ void __deform_tma(float* tile_data, unsigned long long gmem_base, unsigned long long smem_base, int tile_x, int tile_y, int pitch, int mode, int tile_size);
extern "C" __device__ void __deform_interp(float* output, float* tile_data, float* coords, float* weights);

__global__ void my_kernel(float* output, float* weights, float* coords, float* tile_data, float threshold) {
    extern __shared__ float smem[];

    bool mask[16*16];
    __deform_pcb(weights, threshold, true, mask);

    int mode, tile_size, base_x, base_y;
    bool valid[16*16];
    __deform_tbc(coords, valid, &mode, &tile_size, &base_x, &base_y);

    __deform_tma(tile_data, gmem_base, smem_base, base_x, base_y, pitch, mode, tile_size);

    __deform_interp(output, tile_data, coords, weights);
}
```

### Related Documents
- `DevDocs/DeformAttn_Architecture.md` - Architecture design and module specifications
- `DevDocs/DeformAttn_Development_Guide.md` - Implementation guide and development workflow
- `DevDocs/Orient.md` - Algorithm requirements and module mapping (CRITICAL)
- `DevDocs/FMR_Implementation_Summary.md` - FMR implementation reference

### Key Algorithm Requirements (from Orient.md)

#### Baseline (Naive) Implementation
- **Core Feature**: No sparse pruning, no shared memory buffering, no coalesced memory access
- **Performance**: Worst case baseline (100% DRAM access, 68.3% bank conflicts)
- **Corresponding Modules**: None - all computation in GPU pipeline serial execution

#### Optimized (5-Stage Pipeline) Implementation
- **Core Feature**: Sparse pruning, tile-based cooperation, shared memory buffering
- **Performance**: 4-8× improvement over baseline
- **Corresponding Modules**: Complete 5-stage pipeline architecture

#### Module Mapping to Kernel Execution Flow

| Optimization Phase | Corresponding Module | Function | Latency | Key Technology |
|-------------------|---------------------|----------|---------|----------------|
| **Phase I: Sparsity Pruning** | **PCB (Pre-Check Block)** | Weight pre-filtering, generates sparse mask | 1 cycle (fixed) | Threshold comparison, mask generation |
| **Phase II: Sparse-Aware Compute** | **GTC (Gated Tensor Core)** | Operand isolation, marks valid coordinates | 0 cycle (parallel) | Invalid point coordinate zeroing |
| **Phase III: Spatial Aggregation Decision** | **TBC (Tile Boundary Check)** | Adaptive aggregation decision, determines loading strategy | 4-7 cycles (adaptive) | Two-stage decision, mode detection |
| **Phase IV: Adaptive Data Loading** | **TMA & Storage** | Tile loading + conflict-free storage management | 22 cycles (16+6) | Adaptive bank mapping, tracker management |
| **Phase V: Conflict-Free Compute** | **Interpolation** | Bilinear interpolation computation | 3 cycles | Conflict-free read, Swizzling |

#### Total Latency Budget
- **Low clustering scenario** (Discrete mode): ~30 cycles
  - PCB(1) + GTC(0) + TBC(4) + TMA(16) + Storage(6) + Interpolation(3) = 30 cycles
- **High clustering scenario** (Tile mode): ~33 cycles
  - PCB(1) + GTC(0) + TBC(7) + TMA(16) + Storage(6) + Interpolation(3) = 33 cycles

## General Notes

- This is a research simulator - performance differs from real hardware
- Functional model (PTX) is cycle-accurate for instruction semantics
- Timing model (micro-arch) is configurable for different GPU architectures
- DeformAttn extensions are functional models with fixed latencies (no timing-level implementation)
- FMR optimization models warp-cooperative tile loading for attention workloads