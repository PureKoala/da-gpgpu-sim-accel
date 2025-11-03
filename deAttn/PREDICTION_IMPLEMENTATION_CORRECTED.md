# Deformable Attention 预测阶段实现文档（修正版）

## INT8 Ampere Tensor Core 实现

本文档详细描述了 Deformable Attention 预测阶段的 INT8 Tensor Core 实现，用于展示 **N 维度填充**导致的计算资源浪费，为"支持更小 N 维度粒度"的硬件创新提供实验支持。

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
3. **浪费分析**：量化 **N 维度填充**导致的计算资源浪费，为论文创新点提供数据支持

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
│  W_SO [C_in, N_SO]  ────────┼──> 量化+N-Padding ──> W_SO_s8 [C_in, N_SO_pad] (INT8) │
│  W_A  [C_in, N_A]   ────────┘    量化+N-Padding ──> W_A_s8  [C_in, N_A_pad]  (INT8) │
│                                                                  │
│  其中: N_SO_pad = ((N_SO + 7) / 8) * 8  (向上取整到8的倍数)     │
│       N_A_pad  = ((N_A  + 7) / 8) * 8  (向上取整到8的倍数)     │
│                                                                  │
│  ┌─────────────────┐        ┌─────────────────┐                │
│  │ predict_so_     │ INT32  │ dequantize_     │ FP32           │
│  │ int8_ampere     │ ────>  │ so_kernel       │ ────> sampling_loc │
│  │ (N填充浪费)     │        │                 │                │
│  └─────────────────┘        └─────────────────┘                │
│                                                                  │
│  ┌─────────────────┐        ┌─────────────────┐                │
│  │ predict_attn_   │ INT32  │ dequantize_     │ FP32           │
│  │ int8_ampere     │ ────>  │ attn_kernel     │ ────> attn_weight │
│  │ (N填充浪费)     │        │                 │                │
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

### 2.2 N 维度填充问题

**关键发现：INT8 Tensor Core 的 N 维度限制**

| 矩阵 | 原始 N 维度 | 填充后 N 维度 | 浪费比例 | Tensor Core 利用率 |
|------|------------|--------------|---------|-------------------|
| W_SO | num_heads × num_levels × num_point × 2<br>(例如: 8×4×4×2 = 256) | 向上取整到 8 的倍数<br>(例如: 256 → 256, 无浪费) | 取决于实际值 | 取决于填充 |
| W_A  | num_heads × num_levels × num_point<br>(例如: 8×4×4 = 128) | 向上取整到 8 的倍数<br>(例如: 128 → 128, 无浪费) | 取决于实际值 | 取决于填充 |

**示例场景分析：**

1. **无浪费场景**（N 已经是 8 的倍数）：
   - SO: N = 256 → N_pad = 256 (0% 浪费)
   - A:  N = 128 → N_pad = 128 (0% 浪费)

2. **中等浪费场景**（N 需要少量填充）：
   - SO: N = 10 → N_pad = 16 (37.5% 浪费)
   - A:  N = 5  → N_pad = 8  (37.5% 浪费)

3. **严重浪费场景**（N << 8）：
   - SO: N = 4 → N_pad = 8 (50% 浪费)
   - A:  N = 2 → N_pad = 8 (75% 浪费)
   - A:  N = 1 → N_pad = 8 (87.5% 浪费)

**硬件原因：**
- Ampere 架构的 INT8 Tensor Core 指令 `mma.sync.aligned.m16n8k16` 要求：
  - M = 16 (输出行数)
  - **N = 8 (输出列数，这是最小值！)**
  - K = 16 (内积长度)
- 当实际输出特征维度 N < 8 时，必须填充到 N=8
- 硬件执行完整的 N=8 计算，但填充列的结果被丢弃
- **这为"支持更小 N 维度粒度"的硬件创新提供了强有力的论据**

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

#### 3.2.1 采样偏移量预测 (N 维度填充)

```cuda
/**
 * WMMA 指令: mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32
 *   - M=16: 每次处理 16 行
 *   - N=8:  每次处理 8 列 (最小值，无法更小！)
 *   - K=16: 内积长度 16
 */
template <int WMMA_M = 16, int WMMA_N = 8, int WMMA_K = 16>
static __global__ void predict_so_int8_ampere_kernel(
    const int8_t* __restrict__ Q,          // [T × C_in]
    const int8_t* __restrict__ W_SO,       // [C_in × N_out], N_out 必须是 8 的倍数
    int32_t* __restrict__ SO_out,          // [T × N_out]
    const int T,
    const int C_in,
    const int N_out                        // 填充后的 N 维度 (≥ 8)
) {
    // ... 省略 warp 和 fragment 初始化 ...
    
    // **关键点：硬件最小 N=8 的限制**
    // 如果实际需要的 N < 8，必须填充到 8
    // 填充列的计算结果会被丢弃，浪费计算资源
    
    for (int k = 0; k < C_in; k += WMMA_K) {
        wmma::load_matrix_sync(a_frag, Q + aRow * C_in + k, C_in);
        wmma::load_matrix_sync(b_frag, W_SO + k * N_out + bCol, N_out);
        
        // **此 WMMA 指令处理 N=8 列，即使只需要 N < 8**
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    wmma::store_matrix_sync(SO_out + aRow * N_out + bCol, c_frag, N_out, wmma::mem_row_major);
}
```

**浪费分析：**
- 如果 N_actual = 4，N_padded = 8：50% 的输出列计算被浪费
- 如果 N_actual = 2，N_padded = 8：75% 的输出列计算被浪费
- 如果 N_actual = 1，N_padded = 8：87.5% 的输出列计算被浪费

#### 3.2.2 注意力权重预测 (N 维度填充)

```cuda
// 与 predict_so 类似，唯一区别是输出维度不同
template <int WMMA_M = 16, int WMMA_N = 8, int WMMA_K = 16>
static __global__ void predict_attn_int8_ampere_kernel(
    const int8_t* __restrict__ Q,
    const int8_t* __restrict__ W_A,        // [C_in × N_out], N_out 必须是 8 的倍数
    int32_t* __restrict__ A_out,
    const int T,
    const int C_in,
    const int N_out
) {
    // 同样受 N=8 最小值的限制
    // 如果注意力权重维度 < 8，必须填充到 8
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

#### N 维度填充函数
```cpp
// 注意：函数名保留为 pad_weights_k_dimension 以兼容历史代码
// 但实际上是在填充 N 维度（输出特征维度）
inline void pad_weights_k_dimension(const int8_t* input, int8_t* output, 
                             int C, int N_orig, int N_padded) {
    for (int c = 0; c < C; c++) {
        // 复制有效数据
        for (int n = 0; n < N_orig; n++) {
            output[c * N_padded + n] = input[c * N_orig + n];
        }
        // 零填充剩余列
        for (int n = N_orig; n < N_padded; n++) {
            output[c * N_padded + n] = 0;
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
    int N_out,                // 必须是 8 的倍数
    cudaStream_t stream = 0);

// 2. 注意力权重预测
void ms_deform_attn_predict_attn_cuda(
    const int8_t* d_Q,
    const int8_t* d_W_A,
    int32_t* d_A_out,
    int batch_size,
    int num_query,
    int C_in,
    int N_out,                // 必须是 8 的倍数
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
float* h_W_SO = ...;      // [C_in × N_SO_orig]
float* h_W_A = ...;       // [C_in × N_A_orig]

// 2. 计算填充后的维度
int N_SO_pad = ((N_SO_orig + 7) / 8) * 8;  // 向上取整到 8 的倍数
int N_A_pad  = ((N_A_orig  + 7) / 8) * 8;

// 3. 计算量化参数
QuantizationParams q_params = compute_quantization_params(h_Q, Q_size);
QuantizationParams w_so_params = compute_quantization_params(h_W_SO, W_SO_size);
QuantizationParams w_a_params = compute_quantization_params(h_W_A, W_A_size);

// 4. 量化 FP32 → INT8（同时进行 N 维度填充）
int8_t* h_Q_s8 = ...;
int8_t* h_W_SO_s8_padded = ...;  // [C_in × N_SO_pad]
int8_t* h_W_A_s8_padded = ...;   // [C_in × N_A_pad]

quantize_to_int8(h_Q, h_Q_s8, Q_size, q_params.scale, q_params.zero_point);

// 对权重矩阵进行量化和 N 维度填充
// （填充部分用零填充，量化后仍为接近 zero_point 的值）
// ... 量化和填充逻辑 ...

// 5. 传输到设备
cudaMemcpy(d_Q_s8, h_Q_s8, ...);
cudaMemcpy(d_W_SO_padded, h_W_SO_s8_padded, ...);
cudaMemcpy(d_W_A_padded, h_W_A_s8_padded, ...);

// 6. 执行预测阶段（INT8 Tensor Core）
int32_t *d_SO_s32, *d_A_s32;
ms_deform_attn_predict_so_cuda(d_Q_s8, d_W_SO_padded, d_SO_s32, 
                                 batch, query, C_in, N_SO_pad, ...);
ms_deform_attn_predict_attn_cuda(d_Q_s8, d_W_A_padded, d_A_s32, 
                                   batch, query, C_in, N_A_pad, ...);

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

### 6.1 计算资源利用率（基于 N 维度）

| 实际 N 维度 | 填充后 N | 浪费比例 | 有效利用率 |
|-----------|---------|---------|-----------|
| N = 8     | 8       | 0%      | **100%**  |
| N = 7     | 8       | 12.5%   | **87.5%** |
| N = 6     | 8       | 25%     | **75%**   |
| N = 5     | 8       | 37.5%   | **62.5%** |
| N = 4     | 8       | 50%     | **50%**   |
| N = 3     | 8       | 62.5%   | **37.5%** |
| N = 2     | 8       | 75%     | **25%**   |
| N = 1     | 8       | 87.5%   | **12.5%** |

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

Tensor Core利用率分析 (N维度限制):
  SO预测 N=256 → 256:  100.0% (完全利用)
  Attn预测 N=128 → 128: 100.0% (完全利用)
  
  [或在需要填充的情况下:]
  SO预测 N=4 → 8:      50.0% (浪费50%)
  Attn预测 N=2 → 8:    25.0% (浪费75%)
========================================
```

### 6.3 GPGPU-Sim 分析重点

运行 GPGPU-Sim 仿真后，重点分析：

1. **指令统计**：`gpgpu_inst_stats.txt` 中 `wmma.mma.sync.aligned.m16n8k16` 的执行次数
2. **Warp效率**：预测阶段 Warp 的实际利用率
3. **内存访问模式**：量化/反量化阶段的内存带宽利用率
4. **对比实验**：
   - 有预测阶段 vs. 无预测阶段的总体性能
   - INT8 路径 vs. FP32 路径的精度损失
   - 不同 N 维度下的性能差异

---

## 7. 论文创新点支持

### 7.1 实验数据收集

本实现为"支持更小 N 维度粒度"的硬件创新提供以下实验数据：

1. **浪费量化**：
   - N=4 场景：50% 输出列计算浪费
   - N=2 场景：75% 输出列计算浪费
   - N=1 场景：87.5% 输出列计算浪费

2. **GPGPU-Sim 拦截证据**：
   - PTX 指令级别的 `wmma.mma.sync` 执行统计
   - Tensor Core 硬件资源占用时间分析
   - 实际有效计算 vs. 总计算周期的比例

3. **性能对比**：
   - 如果硬件支持 N=4 或 N=2 的 WMMA 指令，理论加速比为 2× 或 4×

### 7.2 创新论述

> 传统的 Ampere 架构 INT8 Tensor Core 指令 `wmma.mma.sync.aligned.m16n8k16` 要求 **N 维度最小为 8**。然而，在 Deformable DETR 的预测阶段，实际的输出特征维度可能远小于 8。这导致硬件执行了大量无效计算：
>
> - **采样偏移量预测**：如果 N < 8，必须填充到 N=8，浪费 (8-N)/8 的计算资源。
> - **注意力权重预测**：如果 N < 8，同样需要填充，浪费比例随 N 减小而增加。
>
> 我们提出的"支持更小 N 维度粒度"的硬件创新，通过引入 N=4、N=2 等更细粒度的 Tensor Core 指令（如 `m16n4k16`、`m16n2k16`），可以直接消除这种浪费，在相同硬件资源下实现 2-4× 的吞吐量提升。

### 7.3 硬件改进建议

**当前硬件限制：**
- INT8: `m16n8k16` (N_min = 8)
- FP16: `m16n8k8`  (N_min = 8)

**建议的新指令：**
- INT8: `m16n4k16`, `m16n2k16`, `m16n1k16`
- FP16: `m16n4k8`,  `m16n2k8`,  `m16n1k8`

**预期收益：**
- 减少内存带宽需求（无需传输填充数据）
- 提升计算效率（无需处理填充列）
- 降低功耗（减少无效计算）

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
cat sim/test_RTX3070/gpgpu_inst_stats.txt | grep "wmma.mma"
```

---

## 9. 常见问题

### Q1: 为什么是 N 维度而不是 K 维度？

**A:** 根据 NVIDIA PTX 文档，对于 GEMM 操作 C = A × B：
- A 矩阵维度：M × K
- B 矩阵维度：K × N
- C 矩阵维度：M × N

Tensor Core 指令 `m16n8k16` 的含义是：
- M=16: 输出矩阵的行数
- N=8:  输出矩阵的列数（对应输出特征维度）
- K=16: 内积长度（对应输入特征维度的分块大小）

在预测阶段，输出特征维度对应 N 维度，因此限制在 N 维度。

### Q2: 为什么不能使用 FP16 Tensor Core？

**A:** FP16 Tensor Core (`m16n8k8`) 同样要求 N ≥ 8。虽然 K 维度要求更小（K=8 vs K=16），但 N 维度的限制是相同的。

### Q3: 代码中的 `pad_weights_k_dimension` 函数名为什么不叫 `pad_weights_n_dimension`？

**A:** 为了保持向后兼容性和代码一致性，保留了原函数名。实际上该函数是在填充第二个维度，对于权重矩阵 [C_in × N_out]，第二个维度就是 N 维度（输出特征维度）。

---

## 10. 参考文献

1. **Deformable DETR**: Zhu et al., "Deformable DETR: Deformable Transformers for End-to-End Object Detection", ICLR 2021
2. **NVIDIA Ampere Architecture**: [NVIDIA Ampere GA102 GPU Architecture Whitepaper](https://www.nvidia.com/en-us/data-center/ampere-architecture/)
3. **PTX ISA**: [NVIDIA PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
4. **WMMA API**: [CUDA WMMA API Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
5. **GPGPU-Sim**: [GPGPU-Sim Manual](http://gpgpu-sim.org/manual/)

---

**文档版本**: v2.0 (修正版)  
**最后更新**: 2025-11-02  
**作者**: AI Assistant  
**修正说明**: 将错误的"K 维度填充"修正为"N 维度填充"，并更新所有相关描述和代码注释。
**许可证**: Apache License 2.0
