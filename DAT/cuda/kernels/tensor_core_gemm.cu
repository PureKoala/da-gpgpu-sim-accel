#include "../include/tensor_core_gemm.h"
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

/**
 * Tensor Core GEMM Kernel - INT8 输入, FP32 输出
 * 针对 Ampere 架构优化
 * 
 * 使用 m16n16k16 wmma 指令
 */

namespace tensor_core {

// WMMA 基础配置
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Block 配置
constexpr int BLOCK_M = 64;  // 每个block处理的M维度
constexpr int BLOCK_N = 64;  // 每个block处理的N维度
constexpr int BLOCK_K = 64;  // K维度的tile size

// Warp 配置
constexpr int WARP_M = 32;   // 每个warp处理的M维度
constexpr int WARP_N = 32;   // 每个warp处理的N维度

/**
 * INT8 Tensor Core GEMM Kernel
 * C = alpha * A * B + beta * C
 * A: (M, K) INT8
 * B: (K, N) INT8
 * C: (M, N) FP32
 */
__global__ void tensor_core_gemm_int8_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta,
    float scale_a, float scale_b
) {
    // Block 和 Warp 索引
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int warp_id = threadIdx.x / 32;
    
    // 计算该block负责的输出范围
    int c_row = block_row * BLOCK_M;
    int c_col = block_col * BLOCK_N;
    
    // 共享内存用于tile缓存
    __shared__ int8_t smem_A[BLOCK_M][BLOCK_K];
    __shared__ int8_t smem_B[BLOCK_K][BLOCK_N];
    
    // WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> c_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag_fp32;
    
    // 初始化累加器为0
    wmma::fill_fragment(c_frag, 0);
    
    // K维度循环
    for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
        // 加载A tile到共享内存
        for (int i = threadIdx.x; i < BLOCK_M * BLOCK_K; i += blockDim.x) {
            int row = i / BLOCK_K;
            int col = i % BLOCK_K;
            int global_row = c_row + row;
            int global_col = k_tile + col;
            
            if (global_row < M && global_col < K) {
                smem_A[row][col] = A[global_row * K + global_col];
            } else {
                smem_A[row][col] = 0;
            }
        }
        
        // 加载B tile到共享内存
        for (int i = threadIdx.x; i < BLOCK_K * BLOCK_N; i += blockDim.x) {
            int row = i / BLOCK_N;
            int col = i % BLOCK_N;
            int global_row = k_tile + row;
            int global_col = c_col + col;
            
            if (global_row < K && global_col < N) {
                smem_B[row][col] = B[global_row * N + global_col];
            } else {
                smem_B[row][col] = 0;
            }
        }
        
        __syncthreads();
        
        // 使用WMMA进行矩阵乘法
        // 每个warp处理多个WMMA tile
        int warp_row = (warp_id / 2) * WARP_M;
        int warp_col = (warp_id % 2) * WARP_N;
        
        for (int k = 0; k < BLOCK_K; k += WMMA_K) {
            for (int i = 0; i < WARP_M; i += WMMA_M) {
                for (int j = 0; j < WARP_N; j += WMMA_N) {
                    int tile_row = warp_row + i;
                    int tile_col = warp_col + j;
                    
                    if (tile_row < BLOCK_M && tile_col < BLOCK_N) {
                        // 加载A fragment
                        wmma::load_matrix_sync(
                            a_frag,
                            &smem_A[tile_row][k],
                            BLOCK_K
                        );
                        
                        // 加载B fragment
                        wmma::load_matrix_sync(
                            b_frag,
                            &smem_B[k][tile_col],
                            BLOCK_N
                        );
                        
                        // 矩阵乘累加
                        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // 转换INT32累加器到FP32并应用scale
    #pragma unroll
    for (int i = 0; i < c_frag.num_elements; i++) {
        c_frag_fp32.x[i] = static_cast<float>(c_frag.x[i]) / (scale_a * scale_b);
    }
    
    // 存储结果
    int warp_row = (warp_id / 2) * WARP_M;
    int warp_col = (warp_id % 2) * WARP_N;
    
    for (int i = 0; i < WARP_M; i += WMMA_M) {
        for (int j = 0; j < WARP_N; j += WMMA_N) {
            int tile_row = warp_row + i;
            int tile_col = warp_col + j;
            int out_row = c_row + tile_row;
            int out_col = c_col + tile_col;
            
            if (out_row < M && out_col < N) {
                // 如果beta != 0，需要先加载原有的C值
                if (beta != 0.0f) {
                    float temp[WMMA_M * WMMA_N];
                    for (int t = 0; t < WMMA_M * WMMA_N; t++) {
                        int local_row = t / WMMA_N;
                        int local_col = t % WMMA_N;
                        int global_row = out_row + local_row;
                        int global_col = out_col + local_col;
                        if (global_row < M && global_col < N) {
                            temp[t] = C[global_row * N + global_col];
                        }
                    }
                    
                    #pragma unroll
                    for (int t = 0; t < c_frag_fp32.num_elements; t++) {
                        c_frag_fp32.x[t] = alpha * c_frag_fp32.x[t] + beta * temp[t];
                    }
                }
                
                // 存储C fragment
                wmma::store_matrix_sync(
                    &C[out_row * N + out_col],
                    c_frag_fp32,
                    N,
                    wmma::mem_row_major
                );
            }
        }
    }
}

/**
 * 主机端调用的GEMM函数
 */
cudaError_t gemm_int8_fp32(
    const int8_t* A, const int8_t* B, float* C,
    int M, int N, int K,
    float alpha, float beta,
    float scale_a, float scale_b,
    cudaStream_t stream
) {
    // 配置block和grid
    dim3 block(256);  // 8 warps per block
    dim3 grid(
        (N + BLOCK_N - 1) / BLOCK_N,
        (M + BLOCK_M - 1) / BLOCK_M
    );
    
    tensor_core_gemm_int8_kernel<<<grid, block, 0, stream>>>(
        A, B, C, M, N, K, alpha, beta, scale_a, scale_b
    );
    
    return cudaGetLastError();
}

} // namespace tensor_core
