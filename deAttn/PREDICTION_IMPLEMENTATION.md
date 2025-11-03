# Deformable Attention 预测阶段实现文档

## INT8 Ampere Tensor Core 实现

本文档详细描述了 Deformable Attention 预测阶段的 INT8 Tensor Core 实现，用于展示 K-Padding 导致的计算资源浪费，为"切分K"创新提供实验支持。

---

## 1. 实现概述

### 1.1 背景

现有的 Deformable Attention CUDA 实现**只包含采样与聚合阶段**，缺失了从 Query 张量生成采样偏移量 (Sampling Offset) 和注意力权重 (Attention Weight) 的**预测阶段**。

预测阶段的核心是两个 GEMM 运算：
- **SO = Q × W_SO**：生成采样偏移量
- **A = Q × W_A**：生成注意力权重

### 1.2 实现目标

1. **功能完整性**：实现完整的端到端 Deformable Attention（预测 → 采样 → 聚合）
2. **性能展示**：使用 INT8 Tensor Core (Ampere `mma.sync.aligned.m16n8k16`) 执行预测阶段
3. **浪费分析**：量化 K-Padding 导致的计算资源浪费，为论文创新点提供数据支持

---

## 2. 架构设计

### 2.1 数据流设计

```
┌─────────────────────────────────────────────────────────────────┐
│                      预测阶段 (Prediction)                        │
│                                                                  │
│  Q [batch×query, C_in]  ───┐                                    │
│         (FP32)              │                                    │
│                             ├──> 量化 ──> Q_s8 (INT8)           │
│                             │                                    │
│  W_SO [C_in, 8]  ───────────┼──> 量化+Padding ──> W_SO_s8 [C_in, 16] (INT8) │
│  W_A  [C_in, 4]  ───────────┘    量化+Padding ──> W_A_s8  [C_in, 16] (INT8) │
│                                                                  │
│  ┌─────────────────┐        ┌─────────────────┐                │
│  │ predict_so_     │ INT32  │ dequantize_     │ FP32           │
│  │ int8_ampere     │ ────>  │ so_kernel       │ ────> sampling_loc │
│  │ (K=8→16, 50%浪费)│        │                 │                │
│  └─────────────────┘        └─────────────────┘                │
│                                                                  │
│  ┌─────────────────┐        ┌─────────────────┐                │
│  │ predict_attn_   │ INT32  │ dequantize_     │ FP32           │
│  │ int8_ampere     │ ────>  │ attn_kernel     │ ────> attn_weight │
│  │ (K=4→16, 75%浪费)│        │                 │                │
│  └─────────────────┘        └─────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              采样与聚合阶段 (Sampling & Aggregation)              │
│                                                                  │
│  ms_deform_attn_cuda_forward(                                   │
│      value, sampling_loc, attn_weight, ...                      │
│  )                                                              │
│                                                                  │
│  输出: aggregated_features [batch×query, num_heads×channels]    │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 K-Padding 策略

| 矩阵 | 原始维度 | 填充后维度 | 浪费比例 | Tensor Core 利用率 |
|------|---------|-----------|---------|-------------------|
| W_SO | C_in × 8 | C_in × 16 | 50% | 50% |
| W_A  | C_in × 4 | C_in × 16 | 75% | 25% |

**关键发现**：
- Ampere 架构的 `mma.sync.aligned.m16n8k16` 指令要求 **K 维度最小为 16**
- DDetr 预测阶段的实际 K 维度为 4 和 8
- 硬件执行 K=16 的指令，但只有少部分元素有效数据
- **这为"切分K"创新提供了强有力的论据**

---

## 3. 核心实现

### 3.1 文件结构

```
deAttn/src/deform_attn/
├── deform_attn_cuda.h                      # 公共API头文件（新增预测阶段API）
└── cuda/
    ├── ms_deform_attn_predict_cuda.cuh     # 预测阶段Kernel实现（新增）
    ├── ms_deform_attn_predict_wrapper.cu   # API包装器（新增）
    ├── ms_deform_attn_cuda_kernel.cu       # 采样聚合阶段包装器（已有）
    └── ms_deform_attn_im2col_cuda.cuh      # 采样聚合Kernel（已有）
```

### 3.2 关键 Kernel 实现

#### 3.2.1 采样偏移量预测 (K=8 → K=16)

```cuda
template <int TILE_M = 16, int TILE_N = 8, int TILE_K = 16>
static __global__ void predict_so_int8_ampere_kernel(
    const int8_t* Q,          // [T × C_in], T = batch × num_query
    const int8_t* W_SO,       // [C_in × 16] (padded from 8)
    int32_t* SO_out,          // [T × 16]
    const int T,
    const int C_in,
    const int N_out           // = 16 (after padding)
) {
    // ... 省略加载和计算逻辑 ...
    
    // **CRITICAL INTERCEPTION POINT FOR GPGPU-SIM**
    // 硬件执行 K=16，但只有前 8 个元素有效
    #if __CUDA_ARCH__ >= 800
    asm volatile (
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+r"(C_frag[0]), "+r"(C_frag[1]), "+r"(C_frag[2]), "+r"(C_frag[3])
        : "r"(*reinterpret_cast<int32_t*>(&A_frag[0])),
          "r"(*reinterpret_cast<int32_t*>(&A_frag[2])),
          "r"(*reinterpret_cast<int32_t*>(&B_frag[0]))
    );
    #endif
    
    // 50% 的 K 维度计算被浪费（K=16 中只有 8 个有效）
}
```

#### 3.2.2 注意力权重预测 (K=4 → K=16)

```cuda
template <int TILE_M = 16, int TILE_N = 8, int TILE_K = 16>
static __global__ void predict_attn_int8_ampere_kernel(
    const int8_t* Q,          // [T × C_in]
    const int8_t* W_A,        // [C_in × 16] (padded from 4)
    int32_t* A_out,           // [T × 16]
    const int T,
    const int C_in,
    const int N_out           // = 16 (after padding from 4)
) {
    // 同样的 mma.sync.aligned.m16n8k16 指令
    // 但是 75% 的 K 维度被浪费（K=16 中只有 4 个有效）
    // 如果 N 维度也需要填充（4→8），则总利用率仅为 12.5%
}
```

#### 3.2.3 反量化 Kernel

```cuda
static __global__ void dequantize_so_kernel(
    const int32_t* SO_int32,
    float* sampling_loc,
    const float* reference_points,
    const float scale,
    const int zero_point,
    const int total_elements,
    const bool add_reference
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    // INT32 → FP32
    float dequant_val = static_cast<float>(SO_int32[idx] - zero_point) * scale;
    
    // 添加参考点偏移（Deformable Attention 特有）
    if (add_reference && reference_points != nullptr) {
        dequant_val += reference_points[idx];
    }
    
    sampling_loc[idx] = dequant_val;
}
```

### 3.3 主机端辅助函数

#### 量化函数
```cpp
inline void quantize_to_int8(const float* input, int8_t* output, size_t n, 
                              float scale, int zero_point) {
    for (size_t i = 0; i < n; i++) {
        float val = input[i] / scale + zero_point;
        val = fmax(-128.0f, fmin(127.0f, val));
        output[i] = static_cast<int8_t>(roundf(val));
    }
}
```

#### 填充函数
```cpp
inline void pad_weights_k_dimension(const int8_t* input, int8_t* output, 
                                     int C, int K_orig, int K_padded) {
    for (int c = 0; c < C; c++) {
        for (int k = 0; k < K_orig; k++) {
            output[c * K_padded + k] = input[c * K_orig + k];
        }
        // 零填充剩余元素
        for (int k = K_orig; k < K_padded; k++) {
            output[c * K_padded + k] = 0;
        }
    }
}
```

---

## 4. API 接口

### 4.1 新增的公共 API

```cpp
// 1. 采样偏移量预测
void ms_deform_attn_predict_so_cuda(
    const int8_t* d_Q,
    const int8_t* d_W_SO,
    int32_t* d_SO_out,
    int batch_size,
    int num_query,
    int C_in,
    int N_out,
    cudaStream_t stream = 0);

// 2. 注意力权重预测
void ms_deform_attn_predict_attn_cuda(
    const int8_t* d_Q,
    const int8_t* d_W_A,
    int32_t* d_A_out,
    int batch_size,
    int num_query,
    int C_in,
    int N_out,
    cudaStream_t stream = 0);

// 3. 采样偏移量反量化
void ms_deform_attn_dequantize_so_cuda(
    const int32_t* d_SO_int32,
    float* d_sampling_loc,
    const float* d_reference_points,
    float scale,
    int zero_point,
    int total_elements,
    bool add_reference,
    cudaStream_t stream = 0);

// 4. 注意力权重反量化
void ms_deform_attn_dequantize_attn_cuda(
    const int32_t* d_A_int32,
    float* d_attn_weight,
    float scale,
    int zero_point,
    int total_elements,
    cudaStream_t stream = 0);
```

### 4.2 量化参数结构

```cpp
struct QuantizationParams {
    float scale;
    int zero_point;
    
    QuantizationParams() : scale(1.0f), zero_point(0) {}
    QuantizationParams(float s, int zp) : scale(s), zero_point(zp) {}
};
```

---

## 5. 使用示例

### 5.1 完整的端到端流程

```cpp
// 1. 准备输入数据 (FP32)
float* h_Q = ...;         // [batch × num_query × C_in]
float* h_W_SO = ...;      // [C_in × 8]
float* h_W_A = ...;       // [C_in × 4]

// 2. 计算量化参数
QuantizationParams q_params = compute_quantization_params(h_Q, Q_size);
QuantizationParams w_so_params = compute_quantization_params(h_W_SO, W_SO_size);
QuantizationParams w_a_params = compute_quantization_params(h_W_A, W_A_size);

// 3. 量化 FP32 → INT8
int8_t* h_Q_s8 = ...;
int8_t* h_W_SO_s8 = ...;
int8_t* h_W_A_s8 = ...;
quantize_to_int8(h_Q, h_Q_s8, Q_size, q_params.scale, q_params.zero_point);
quantize_to_int8(h_W_SO, h_W_SO_s8, W_SO_size, w_so_params.scale, w_so_params.zero_point);
quantize_to_int8(h_W_A, h_W_A_s8, W_A_size, w_a_params.scale, w_a_params.zero_point);

// 4. K-Padding
int8_t* h_W_SO_padded = ...;
int8_t* h_W_A_padded = ...;
pad_weights_k_dimension(h_W_SO_s8, h_W_SO_padded, C_in, 8, 16);
pad_weights_k_dimension(h_W_A_s8, h_W_A_padded, C_in, 4, 16);

// 5. 传输到设备
cudaMemcpy(d_Q_s8, h_Q_s8, ...);
cudaMemcpy(d_W_SO_padded, h_W_SO_padded, ...);
cudaMemcpy(d_W_A_padded, h_W_A_padded, ...);

// 6. 执行预测阶段（INT8 Tensor Core）
int32_t *d_SO_s32, *d_A_s32;
ms_deform_attn_predict_so_cuda(d_Q_s8, d_W_SO_padded, d_SO_s32, ...);
ms_deform_attn_predict_attn_cuda(d_Q_s8, d_W_A_padded, d_A_s32, ...);

// 7. 反量化（INT32 → FP32）
float *d_sampling_loc, *d_attn_weight;
ms_deform_attn_dequantize_so_cuda(d_SO_s32, d_sampling_loc, ...);
ms_deform_attn_dequantize_attn_cuda(d_A_s32, d_attn_weight, ...);

// 8. 执行采样聚合阶段（现有实现）
ms_deform_attn_cuda_forward(d_value, d_spatial_shapes, d_level_start_index,
                             d_sampling_loc, d_attn_weight, d_output, ...);
```

---

## 6. 性能分析

### 6.1 计算资源利用率

| Kernel | K维度 | 硬件K | 浪费比例 | 有效利用率 | GPGPU-Sim指令计数 |
|--------|------|------|---------|----------|----------------|
| predict_so | 8 | 16 | 50% | **50%** | `mma.sync.aligned.m16n8k16` |
| predict_attn | 4 | 16 | 75% | **25%** | `mma.sync.aligned.m16n8k16` |

### 6.2 性能统计输出

```
========================================
         性能统计摘要
========================================
预测阶段:
  SO预测时间:        X.XXX ms
  Attn预测时间:      X.XXX ms
  反量化时间:        X.XXX ms
  主机量化时间:      X.XXX ms

采样聚合阶段:
  Kernel执行时间:    X.XXX ms

总体性能:
  内存传输时间:      X.XXX ms
  总时间:            X.XXX ms
  计算操作数:        XXXXXXXXX
  计算性能:          XX.XX GFLOPS
  内存占用:          XX.XX MB

Tensor Core利用率分析:
  K=8场景利用率:     50.0% (浪费50%)
  K=4场景利用率:     25.0% (浪费75%)
========================================
```

### 6.3 GPGPU-Sim 分析重点

运行 GPGPU-Sim 仿真后，重点分析：

1. **指令统计**：`gpgpu_inst_stats.txt` 中 `mma.sync.aligned.m16n8k16` 的执行次数
2. **Warp效率**：预测阶段 Warp 的实际利用率
3. **内存访问模式**：量化/反量化阶段的内存带宽利用率
4. **对比实验**：
   - 有预测阶段 vs. 无预测阶段的总体性能
   - INT8 路径 vs. FP32 路径的精度损失

---

## 7. 论文创新点支持

### 7.1 实验数据收集

本实现为"切分K"创新提供以下实验数据：

1. **浪费量化**：
   - K=8 场景：50% 计算周期浪费
   - K=4 场景：75% 计算周期浪费

2. **GPGPU-Sim 拦截证据**：
   - PTX 指令级别的 `mma.sync` 执行统计
   - Tensor Core 硬件资源占用时间分析

3. **性能对比**：
   - 如果支持 `m16n8k8` 和 `m16n8k4` 指令，理论加速比为 2× 和 4×

### 7.2 创新论述

> 传统的 Ampere 架构 INT8 Tensor Core 指令 `mma.sync.aligned.m16n8k16` 要求 K 维度最小为 16。然而，在 Deformable DETR 的预测阶段，实际的 K 维度仅为 4 和 8。这导致硬件执行了大量无效计算：
>
> - **采样偏移量预测**：K=8 时，50% 的 Tensor Core 计算能力被浪费在处理零填充元素上。
> - **注意力权重预测**：K=4 时，高达 75% 的计算资源未被有效利用。
>
> 我们提出的"切分K"机制，通过在硬件层面支持更细粒度的 K 维度（如 K=8, K=4），可以直接消除这种浪费，在相同硬件资源下实现 2-4× 的吞吐量提升。

---

## 8. 编译与运行

### 8.1 编译

```bash
cd /home/koala/gpgpu-sim_distribution/deAttn
make clean
make
```

### 8.2 运行测试

```bash
# 真实 GPU 执行
./bin/test

# GPGPU-Sim 仿真
bash run.sh
```

### 8.3 查看结果

```bash
# 查看仿真输出
cat out/test_RTX3070.txt

# 分析指令统计
cat sim/test_RTX3070/gpgpu_inst_stats.txt | grep "mma.sync"
```

---

## 9. 未来工作

### 9.1 短期改进

1. **完整集成**：当前 `test.cu` 仍使用随机初始化的 `sampling_loc` 和 `attn_weight`。需要完全移除随机初始化，改用预测阶段的输出。

2. **精度验证**：添加量化误差分析，对比 INT8 路径和 FP32 路径的输出差异（MSE, PSNR）。

3. **性能优化**：
   - 使用 CUDA Graphs 减少 Kernel 启动开销
   - 使用 CUDA Streams 实现预测和采样的流水线并行

### 9.2 长期扩展

1. **支持更多架构**：
   - Turing Volta WMMA 接口
   - Hopper FP8 Tensor Core

2. **端到端训练**：实现 INT8 量化感知训练 (QAT)

3. **硬件提案**：基于本实验数据，提出 Tensor Core 指令集扩展

---

## 10. 参考文献

1. **Deformable DETR**: Zhu et al., "Deformable DETR: Deformable Transformers for End-to-End Object Detection", ICLR 2021
2. **NVIDIA Ampere Architecture**: [NVIDIA Ampere GA102 GPU Architecture Whitepaper](https://www.nvidia.com/en-us/data-center/ampere-architecture/)
3. **PTX ISA**: [NVIDIA PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
4. **GPGPU-Sim**: [GPGPU-Sim Manual](http://gpgpu-sim.org/manual/)

---

## 附录

### A. 编译警告说明

编译时出现的警告：
- `variable "warp_id" was declared but never referenced`：保留用于调试，不影响功能
- `variable "SO_out_padded" was declared but never referenced`：预留字段，暂未使用

这些警告不影响程序正确性。

### B. 文件列表

新增文件：
- `src/deform_attn/cuda/ms_deform_attn_predict_cuda.cuh` (~600 行)
- `src/deform_attn/cuda/ms_deform_attn_predict_wrapper.cu` (~75 行)

修改文件：
- `src/deform_attn/deform_attn_cuda.h` (+60 行)
- `src/test.cu` (+80 行)
- `Makefile` (+1 行)

---

**文档版本**: v1.0  
**最后更新**: 2025-10-31  
**作者**: AI Assistant  
**许可证**: Apache License 2.0
