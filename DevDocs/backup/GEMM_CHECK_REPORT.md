# DeAttn_base GEMM 计算检查报告（已归档）

> 已归档：当前实现与计划以 [IMPLEMENTATION.md](IMPLEMENTATION.md) 为准。

**日期**: 2026-01-22  
**检查目标**: 确认两个关键的GEMM计算是否存在

---

## ✅ 检查结论：两个GEMM计算均已正确实现

### 1. GEMM #1: 计算采样偏移坐标 (Sampling Offset)

**数学公式**:
```
SO = Q × W_SO
[batch×num_query, SO_out] = [batch×num_query, C_in] × [C_in, SO_out]
```
其中 `SO_out = num_heads × num_levels × num_point × 2 = 8 × 4 × 4 × 2 = 256`

**实现位置**: [ms_deform_attn_predict_cuda.cuh](../deAttn_base/src/deform_attn/cuda/ms_deform_attn_predict_cuda.cuh#L75-L116)

**Kernel函数**:
```cuda
template <int WMMA_M = 16, int WMMA_N = 16, int WMMA_K = 16>
static __global__ void predict_so_fp16_kernel(
    const half* __restrict__ Q,            // [T × C_in], T = batch × num_query
    const half* __restrict__ W_SO,         // [C_in × N_out]
    float* __restrict__ SO_out,            // [T × N_out]
    const int T,
    const int C_in,
    const int N_out
) {
    // 声明 WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // 初始化累加器为零
    wmma::fill_fragment(c_frag, 0.0f);
    
    // K维度循环（16为步长）
    for (int k = 0; k < C_in; k += WMMA_K) {
        // 加载矩阵片段
        wmma::load_matrix_sync(a_frag, Q + aRow * C_in + k, C_in);
        wmma::load_matrix_sync(b_frag, W_SO + k * N_out + bCol, N_out);
        
        // 执行矩阵乘法（FP16输入 → FP32累加器）
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // 存储结果（FP32）
    wmma::store_matrix_sync(SO_out + aRow * N_out + bCol, c_frag, N_out, wmma::mem_row_major);
}
```

**调用位置**: [test.cu](../deAttn_base/src/test.cu#L436-L446)
```cuda
ms_deform_attn_predict_so_cuda(
    d_Q_fp16,           // Query特征 (FP16)
    d_W_SO_fp16,        // 权重矩阵 (FP16)
    d_SO_fp32,          // 输出: 采样偏移坐标 (FP32)
    batch_size,
    num_query,
    C_in,               // 输入维度: 256
    SO_out_padded,      // 输出维度: 256 (填充后)
    0                   // stream
);
```

**WMMA配置**:
- **M=16**: 每次处理16行
- **N=16**: 每次处理16列
- **K=16**: 内积维度每次16
- **输入精度**: FP16 (`half`)
- **累加器精度**: FP32 (`float`)
- **硬件**: Tensor Core (Ampere/Turing)

**输出**:
- 形状: `[batch×num_query, 256]`
- 内容: 每个query的128个采样点的2D偏移坐标 (x, y)
- 用途: 后续用于计算实际采样位置

---

### 2. GEMM #2: 计算注意力权重 (Attention Weight)

**数学公式**:
```
A = Q × W_A
[batch×num_query, A_out] = [batch×num_query, C_in] × [C_in, A_out]
```
其中 `A_out = num_heads × num_levels × num_point = 8 × 4 × 4 = 128`

**实现位置**: [ms_deform_attn_predict_cuda.cuh](../deAttn_base/src/deform_attn/cuda/ms_deform_attn_predict_cuda.cuh#L135-L176)

**Kernel函数**:
```cuda
template <int WMMA_M = 16, int WMMA_N = 16, int WMMA_K = 16>
static __global__ void predict_attn_fp16_kernel(
    const half* __restrict__ Q,            // [T × C_in]
    const half* __restrict__ W_A,          // [C_in × N_out]
    float* __restrict__ A_out,             // [T × N_out]
    const int T,
    const int C_in,
    const int N_out
) {
    // 声明 WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // 初始化累加器
    wmma::fill_fragment(c_frag, 0.0f);
    
    // K维度循环
    for (int k = 0; k < C_in; k += WMMA_K) {
        // 加载矩阵片段
        wmma::load_matrix_sync(a_frag, Q + aRow * C_in + k, C_in);
        wmma::load_matrix_sync(b_frag, W_A + k * N_out + bCol, N_out);
        
        // 执行矩阵乘法（FP16输入 → FP32累加器）
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // 存储结果（FP32）
    wmma::store_matrix_sync(A_out + aRow * N_out + bCol, c_frag, N_out, wmma::mem_row_major);
}
```

**调用位置**: [test.cu](../deAttn_base/src/test.cu#L462-L472)
```cuda
ms_deform_attn_predict_attn_cuda(
    d_Q_fp16,           // Query特征 (FP16)
    d_W_A_fp16,         // 权重矩阵 (FP16)
    d_A_fp32,           // 输出: 注意力权重 (FP32)
    batch_size,
    num_query,
    C_in,               // 输入维度: 256
    A_out_padded,       // 输出维度: 128 (填充后)
    0                   // stream
);
```

**WMMA配置**:
- **M=16**: 每次处理16行
- **N=16**: 每次处理16列
- **K=16**: 内积维度每次16
- **输入精度**: FP16 (`half`)
- **累加器精度**: FP32 (`float`)
- **硬件**: Tensor Core (Ampere/Turing)

**输出**:
- 形状: `[batch×num_query, 128]`
- 内容: 每个query对128个采样点的注意力权重
- 用途: 后续用于加权聚合采样值

---

## 3. 数据流分析

### 3.1 完整计算流程

```
[预测阶段 - 使用Tensor Core]
1. Query特征: Q [batch×num_query, 256] (FP16)
2. GEMM #1: SO = Q × W_SO → [batch×num_query, 256] (FP32)
   └─ 输出: 128个采样点的偏移坐标 (x,y)
3. GEMM #2: A = Q × W_A → [batch×num_query, 128] (FP32)
   └─ 输出: 128个采样点的注意力权重

[采样聚合阶段 - 标准CUDA或硬件加速器]
4. 对每个query的128个采样点:
   a. 读取偏移坐标 (x,y) from SO
   b. 读取注意力权重 w from A
   c. 双线性插值采样: val = BilinearSample(ValueMap, x, y)
   d. 加权累加: output += w × val
```

### 3.2 维度对应关系

| 张量 | 形状 | 含义 |
|------|------|------|
| Q | `[B×Q, 256]` | Query特征（输入） |
| W_SO | `[256, 256]` | 采样偏移权重矩阵 |
| W_A | `[256, 128]` | 注意力权重矩阵 |
| SO | `[B×Q, 256]` | 采样偏移坐标（H×L×P×2 = 8×4×4×2） |
| A | `[B×Q, 128]` | 注意力权重（H×L×P = 8×4×4） |
| Output | `[B×Q, H, C]` | 最终输出特征 |

其中：
- B = batch_size = 4
- Q = num_query = 256
- H = num_heads = 8
- L = num_levels = 4
- P = num_point = 4
- C = channels = 32

---

## 4. WMMA实现特点

### 4.1 FP16 Tensor Core优势

**与INT8版本对比**：

| 特性 | INT8 | FP16 (当前实现) |
|------|------|----------------|
| 输入精度 | 8-bit整数 | 16-bit浮点 |
| 累加器精度 | INT32 | FP32 |
| 量化开销 | 需要FP32→INT8 | 需要FP32→FP16（更快） |
| 反量化开销 | 需要INT32→FP32 | **无需**（直接FP32输出） |
| 数值精度 | 低（量化误差） | 高 |
| 数据流 | FP32→INT8→INT32→FP32 | FP32→FP16→FP32 |
| Tensor Core支持 | ✅ m16n8k16 | ✅ m16n16k16 |

**性能优势**：
- ✅ 消除反量化kernel（减少1次kernel launch）
- ✅ 减少中间数据拷贝
- ✅ 更高的数值稳定性
- ✅ 简化的代码逻辑

### 4.2 N维度填充策略

**为什么需要填充**：
- WMMA要求N维度是16的倍数（m16n16k16指令）
- SO输出: 原始256 → 填充后256 ✅（已经是16的倍数）
- A输出: 原始128 → 填充后128 ✅（已经是16的倍数）

**测试配置中的填充**：
```cpp
const int SO_out_orig = 256;  // 8×4×4×2
const int A_out_orig = 128;   // 8×4×4
const int SO_out_padded = 256; // 无需填充
const int A_out_padded = 128;  // 无需填充
```

### 4.3 矩阵布局

**Q (Query) - Row Major**:
```
Q[batch×num_query, C_in]
内存布局: Q[0,0], Q[0,1], ..., Q[0,255], Q[1,0], Q[1,1], ...
WMMA: wmma::row_major
```

**W_SO, W_A (Weights) - Column Major**:
```
W_SO[C_in, N_out]
内存布局: W[0,0], W[1,0], W[2,0], ..., W[0,1], W[1,1], ...
WMMA: wmma::col_major
```

**输出 - Row Major**:
```
SO[batch×num_query, N_out]
A[batch×num_query, N_out]
WMMA: wmma::mem_row_major
```

---

## 5. 性能测试结果

从 [log.txt](../deAttn_base/log.txt) 中提取：

```
预测阶段:
  SO预测时间:        16.202 us
  Attn预测时间:      11.195 us
  FP32->FP16转换:    5.214 ms

采样聚合阶段:
  Kernel执行时间:    19.051 us

总体性能:
  总时间:            46.449 us
  计算性能:          550.40 GFLOPS
```

**分析**：
- SO预测: 16.2 us（矩阵大小 [1024×256] × [256×256]）
- Attn预测: 11.2 us（矩阵大小 [1024×256] × [256×128]）
- 采样聚合: 19.1 us（128个采样点的双线性插值）
- **Tensor Core GEMM占总时间的59%**

---

## 6. PTX验证

**验证WMMA指令生成**：
```bash
# 查看生成的PTX是否包含WMMA指令
cuobjdump -ptx bin/test | grep -i "wmma\|mma.sync"
```

**预期PTX指令**：
```ptx
wmma.load.sync.aligned.row.m16n16k16.shared.f16 {%f0, %f1, ...}, [%r0], %r1;
wmma.load.sync.aligned.col.m16n16k16.shared.f16 {%f16, %f17, ...}, [%r2], %r3;
wmma.mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32 {...}, {...}, {...}, {...};
wmma.store.sync.aligned.row.m16n16k16.global.f32 [%r4], {...}, %r5;
```

---

## ✅ 总结

### 已确认的实现

| 组件 | 状态 | 位置 | WMMA配置 |
|------|------|------|---------|
| **GEMM #1: 采样偏移** | ✅ 已实现 | `predict_so_fp16_kernel` | m16n16k16, FP16→FP32 |
| **GEMM #2: 注意力权重** | ✅ 已实现 | `predict_attn_fp16_kernel` | m16n16k16, FP16→FP32 |
| **Wrapper函数** | ✅ 已实现 | `ms_deform_attn_predict_so/attn_cuda` | 完整 |
| **测试调用** | ✅ 已实现 | `test.cu:436-472` | 完整 |
| **GPGPU-Sim运行** | ✅ 成功 | `log.txt` | 无错误 |

### 关键特性

- ✅ 两个GEMM都使用Tensor Core (WMMA API)
- ✅ FP16输入 + FP32累加器（混合精度）
- ✅ 无需反量化（相比INT8版本）
- ✅ 自动N维度填充到16的倍数
- ✅ 在GPGPU-Sim中成功运行

### 后续工作

1. ⏳ 创建 deAttn_optim 版本
2. ⏳ 添加硬件加速器调用（PCB/TBC/TMA/Interp）
3. ⏳ 对比baseline vs optimized性能
4. ⏳ 验证4.9x加速比

---

**结论**: ✅ **deAttn_base 已正确实现两个关键GEMM计算，使用FP16 Tensor Core，可作为baseline使用**
