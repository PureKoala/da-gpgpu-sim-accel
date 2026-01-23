# Deformable Attention with FP16 Tensor Core

This directory contains the **FP16 Tensor Core** implementation of Deformable Attention, optimized for GPGPU-Sim analysis.

## Key Features

### 🚀 FP16 Tensor Core Implementation
- Uses native FP16 data type for Tensor Core operations
- **No quantization overhead** compared to INT8 version
- Direct FP32 accumulator output (eliminates dequantization step)
- Better numerical accuracy with simplified implementation

### 📊 Performance Benefits
- **Eliminated steps**:
  - ❌ Scale/zero-point computation
  - ❌ FP32 → INT8 quantization
  - ❌ INT32 → FP32 dequantization kernel
  
- **Simplified pipeline**:
  ```
  FP32 → FP16 conversion → Tensor Core (FP16×FP16→FP32) → Ready!
  ```

### 🎯 Numerical Accuracy
- **FP16 precision**: ~3 decimal digits
- **FP16 range**: ±65,504
- **No quantization error**: Preserves data distribution
- **Better for training**: Improved gradient flow

## Directory Structure

```
deAttn_fp16/
├── src/
│   ├── test.cu                              # Main test program (FP16)
│   └── deform_attn/
│       ├── deform_attn_cuda.h               # API definitions (FP16)
│       └── cuda/
│           ├── ms_deform_attn_cuda_kernel.cu        # Sampling/aggregation
│           ├── ms_deform_attn_predict_cuda.cuh      # FP16 prediction kernels
│           └── ms_deform_attn_predict_wrapper.cu    # FP16 API wrappers
├── Makefile                                 # Build configuration
├── README_FP16.md                          # This file
└── FP16_MIGRATION_SUMMARY.md               # Detailed migration guide
```

## Building

### Prerequisites
- CUDA Toolkit (11.0+)
- GPU with Tensor Core support (Compute Capability 7.0+)
  - Volta (SM70): FP16 Tensor Core
  - Turing (SM75): FP16 Tensor Core
  - Ampere (SM80/86): FP16 & INT8 Tensor Core
  - Ada (SM89): FP16 & INT8 Tensor Core

### Compile
```bash
# Default (SM86 - RTX 3090/4090)
make

# For different architectures
make CUDA_ARCH=sm_70    # Volta (V100)
make CUDA_ARCH=sm_75    # Turing (RTX 2080)
make CUDA_ARCH=sm_80    # Ampere (A100)
make CUDA_ARCH=sm_86    # Ampere (RTX 3090)
make CUDA_ARCH=sm_89    # Ada (RTX 4090)
```

### Run
```bash
# Run the compiled test
make run

# Or directly
./bin/test
```

### Clean
```bash
make clean
```

## Test Configuration

The test program (`src/test.cu`) demonstrates:

1. **Prediction Phase** (FP16 Tensor Core):
   - Query projection: Q [batch×num_query, 256] (FP16)
   - Sampling offset prediction: SO = Q × W_SO (FP16 → FP32)
   - Attention weight prediction: A = Q × W_A (FP16 → FP32)

2. **Sampling & Aggregation Phase** (FP32):
   - Deformable sampling based on predicted offsets
   - Weighted aggregation based on predicted attention weights

### Default Parameters
```cpp
batch_size = 1
num_query = 256
num_heads = 8
channels = 32
num_levels = 4
num_point = 4
base_size = 512×512
```

## Performance Analysis

### Expected Output
```
========================================
  性能统计摘要 (FP16 Tensor Core)
========================================
预测阶段:
  SO预测时间:        XXX.XXX us
  Attn预测时间:      XXX.XXX us
  FP32->FP16转换:    XXX.XXX ms

采样聚合阶段:
  Kernel执行时间:    XXX.XXX us

总体性能:
  内存传输时间:      XXX.XXX us
  总时间:            XXX.XXX us
  计算操作数:        XXXXXXXXX
  计算性能:          XX.XX GFLOPS
  内存占用:          XX.XX MB

Tensor Core利用率分析 (N维度限制):
  N=16场景利用率:    100.0% (FP16最优配置)
  实际场景利用率:    XX.X%

优势 vs INT8版本:
  ✓ 无需量化/反量化
  ✓ 更高数值精度
  ✓ 简化的数据流
========================================
```

### Tensor Core Utilization
- **Optimal**: N = 16 (m16n16k16 WMMA shape)
- **Current config**: SO output = 256, Attn output = 128 (both aligned to 16)
- **Utilization**: 100% (no padding waste!)

## Comparison: FP16 vs INT8

| Aspect | FP16 | INT8 |
|--------|------|------|
| **Precision** | ~3 decimal digits | Integer only |
| **Range** | ±65,504 | ±127 |
| **Quantization** | ❌ Not needed | ✅ Required |
| **Dequantization** | ❌ Not needed | ✅ Required |
| **Memory (inputs)** | 2 bytes | 1 byte |
| **Memory (accum)** | 4 bytes (FP32) | 4 bytes (INT32) |
| **Accuracy** | Higher | Lower |
| **Implementation** | Simpler | More complex |
| **Debug** | Easier | Harder |

## WMMA Configuration

### FP16 Tensor Core
```cpp
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

// FP16 input, FP32 accumulator
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
```

### Shape Requirements
- **M dimension**: Multiple of 16 (optimal)
- **N dimension**: Multiple of 16 (optimal for FP16)
- **K dimension**: Multiple of 16 (required)

## GPGPU-Sim Integration

### For GPGPU-Sim Analysis
1. **Compile with GPGPU-Sim**:
   ```bash
   source $GPGPU_SIM_ROOT/setup_environment
   make clean && make
   ```

2. **Run with simulation**:
   ```bash
   ./bin/test
   ```

3. **Analyze Tensor Core usage**:
   - WMMA instruction traces
   - Memory access patterns
   - Warp utilization
   - Pipeline efficiency

### Key Observations for GPGPU-Sim
- **FP16 WMMA**: `mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32`
- **No dequantization overhead**: Simpler execution trace
- **Direct FP32 output**: Fewer memory transactions
- **Better padding efficiency**: N=16 optimal (vs INT8's N=8 minimum)

## Troubleshooting

### Compilation Issues
```bash
# Check CUDA installation
nvcc --version

# Verify GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Clean and rebuild
make clean && make
```

### Runtime Errors
- **Out of memory**: Reduce batch_size or num_query in test.cu
- **Invalid WMMA configuration**: Ensure dimensions are multiples of 16
- **Wrong architecture**: Compile with correct CUDA_ARCH

## References

- **CUDA Programming Guide**: [Warp Matrix Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
- **FP16 Tensor Core**: [Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)
- **GPGPU-Sim**: [Documentation](http://gpgpu-sim.org/)

## License

Copyright (c) 2025. Licensed under Apache License 2.0.

## Contact

For questions or issues, please refer to the parent project documentation.
