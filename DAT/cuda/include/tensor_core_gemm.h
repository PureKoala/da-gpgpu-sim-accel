#ifndef TENSOR_CORE_GEMM_H
#define TENSOR_CORE_GEMM_H

#include <cuda_runtime.h>
#include <mma.h>

/**
 * Tensor Core GEMM 使用 INT8 数据类型
 * 针对 Ampere 架构 (sm_86) 优化
 * 
 * 使用 CUDA wmma API 进行矩阵乘法
 * 支持的矩阵形状: 16x16x16 (M, N, K)
 */

namespace tensor_core {

// INT8 -> FP32 Tensor Core GEMM
// C (M x N) = A (M x K) * B (K x N)
// 使用 INT8 输入, FP32 累加器
template<int M, int N, int K>
__device__ void wmma_gemm_int8_fp32(
    const int8_t* A,        // (M, K)
    const int8_t* B,        // (K, N)
    float* C,               // (M, N)
    int lda,                // leading dimension of A
    int ldb,                // leading dimension of B
    int ldc,                // leading dimension of C
    float alpha = 1.0f,
    float beta = 0.0f
);

// 量化: FP32 -> INT8
__device__ __forceinline__ int8_t quantize_fp32_to_int8(
    float value,
    float scale,
    int zero_point = 0
) {
    int quantized = __float2int_rn(value * scale) + zero_point;
    quantized = max(-128, min(127, quantized));
    return static_cast<int8_t>(quantized);
}

// 反量化: INT8 -> FP32
__device__ __forceinline__ float dequantize_int8_to_fp32(
    int8_t value,
    float scale,
    int zero_point = 0
) {
    return (static_cast<float>(value) - zero_point) / scale;
}

// 计算量化scale (对称量化)
__device__ __forceinline__ float compute_quantization_scale(
    float abs_max,
    int num_bits = 8
) {
    float max_int = static_cast<float>((1 << (num_bits - 1)) - 1); // 127 for int8
    return max_int / (abs_max + 1e-8f);
}

// 主机端GEMM函数 (用于deformable_attention.cu)
cudaError_t gemm_int8_fp32(
    const int8_t* A,        // (M, K) INT8
    const int8_t* B,        // (K, N) INT8
    float* C,               // (M, N) FP32
    int M, int N, int K,
    float alpha,
    float beta,
    float scale_a,
    float scale_b,
    cudaStream_t stream = 0
);

} // namespace tensor_core

#endif // TENSOR_CORE_GEMM_H
