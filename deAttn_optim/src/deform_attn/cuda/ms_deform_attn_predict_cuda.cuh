/*!
**************************************************************************************************
* Deformable Attention Prediction Phase with FP16 Tensor Core
* Implements sampling offset and attention weight prediction using WMMA API with FP16
* Eliminates quantization overhead compared to INT8 version
* Copyright (c) 2025
**************************************************************************************************
*/

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdint>
#include <algorithm>

using namespace nvcuda;

/****************************************************************************************
 * Host-side Utility Functions: FP32 to FP16 Conversion and Padding
 ****************************************************************************************/

// Convert FP32 array to FP16
inline void convert_fp32_to_fp16(const float* input, half* output, size_t n) {
    for (size_t i = 0; i < n; i++) {
        output[i] = __float2half(input[i]);
    }
}

// Pad weight matrix in N dimension: [C × N_orig] → [C × N_padded]
// Used when N dimension needs padding (e.g., N=4 → N=8 for better Tensor Core alignment)
inline void pad_weights_fp16(const half* input, half* output,
                              int C, int N_orig, int N_padded) {
    for (int c = 0; c < C; c++) {
        for (int n = 0; n < N_orig; n++) {
            output[c * N_padded + n] = input[c * N_orig + n];
        }
        // Zero-pad remaining elements
        for (int n = N_orig; n < N_padded; n++) {
            output[c * N_padded + n] = __float2half(0.0f);
        }
    }
}

/****************************************************************************************
 * Device Kernels: FP16 GEMM using Tensor Core (WMMA API)
 ****************************************************************************************/

/**
 * Kernel: Sampling Offset Prediction using FP16 Tensor Core
 * 
 * Computation: SO = Q × W_SO
 * Matrix dimensions:
 *   - Q: [T × C_in] where T = batch × num_query (FP16, row-major)
 *   - W_SO: [C_in × N_out] (FP16, column-major for WMMA)
 *   - SO: [T × N_out] (FP32 accumulator for higher precision)
 * 
 * WMMA instruction: wmma::mma_sync for FP16
 *   - M=16: Output rows per instruction
 *   - N=16: Output columns per instruction (optimal for FP16)
 *   - K=16: Inner dimension per instruction
 * 
 * Advantages over INT8 version:
 *   - No quantization/dequantization overhead
 *   - Better numerical accuracy
 *   - Direct FP32 accumulator output (no need for separate dequantization kernel)
 *   - Still benefits from Tensor Core acceleration
 */
template <int WMMA_M = 16, int WMMA_N = 16, int WMMA_K = 16>
static __global__ void predict_so_fp16_kernel(
    const half* __restrict__ Q,            // [T × C_in], T = batch × num_query
    const half* __restrict__ W_SO,         // [C_in × N_out]
    float* __restrict__ SO_out,            // [T × N_out]
    const int T,                           // Total queries (batch × num_query)
    const int C_in,                        // Input feature dimension (must be multiple of 16)
    const int N_out                        // Output dimension
) {
    // Warp and lane IDs
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // Declare the fragments for WMMA operations (m16n16k16 for FP16)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // Initialize accumulator to zero
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Compute global position in output matrix
    int aRow = warpM * WMMA_M;
    int bCol = warpN * WMMA_N;
    
    // Boundary check
    if (aRow >= T || bCol >= N_out) return;
    
    // Loop over K dimension in chunks of WMMA_K (16)
    for (int k = 0; k < C_in; k += WMMA_K) {
        // Load the inputs from global memory to fragments
        // Matrix A (Q): row-major layout [T × C_in]
        wmma::load_matrix_sync(a_frag, Q + aRow * C_in + k, C_in);
        
        // Matrix B (W_SO): column-major layout [C_in × N_out]
        wmma::load_matrix_sync(b_frag, W_SO + k * N_out + bCol, N_out);
        
        // Perform the matrix multiplication
        // FP16 input, FP32 accumulator for better precision
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Store the output (FP32 accumulator) - no dequantization needed!
    wmma::store_matrix_sync(SO_out + aRow * N_out + bCol, c_frag, N_out, wmma::mem_row_major);
}

/**
 * Kernel: Attention Weight Prediction using FP16 Tensor Core
 * 
 * Computation: A = Q × W_A
 * Matrix dimensions:
 *   - Q: [T × C_in] where T = batch × num_query (FP16, row-major)
 *   - W_A: [C_in × N_out] (FP16, column-major for WMMA)
 *   - A: [T × N_out] (FP32 accumulator)
 * 
 * WMMA instruction: wmma::mma_sync for FP16
 *   - M=16: Output rows per instruction
 *   - N=16: Output columns per instruction
 *   - K=16: Inner dimension per instruction
 * 
 * Advantages over INT8 version:
 *   - No quantization/dequantization overhead
 *   - Direct FP32 output (ready for softmax)
 *   - Better numerical stability for attention weights
 */
template <int WMMA_M = 16, int WMMA_N = 16, int WMMA_K = 16>
static __global__ void predict_attn_fp16_kernel(
    const half* __restrict__ Q,            // [T × C_in]
    const half* __restrict__ W_A,          // [C_in × N_out]
    float* __restrict__ A_out,             // [T × N_out]
    const int T,
    const int C_in,
    const int N_out
) {
    // Warp and lane IDs
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // Declare WMMA fragments (m16n16k16 for FP16)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // Initialize accumulator
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Compute global position
    int aRow = warpM * WMMA_M;
    int bCol = warpN * WMMA_N;
    
    if (aRow >= T || bCol >= N_out) return;
    
    // Loop over K dimension in chunks of WMMA_K (16)
    for (int k = 0; k < C_in; k += WMMA_K) {
        // Load matrix A (Q)
        wmma::load_matrix_sync(a_frag, Q + aRow * C_in + k, C_in);
        
        // Load matrix B (W_A)
        wmma::load_matrix_sync(b_frag, W_A + k * N_out + bCol, N_out);
        
        // FP16 matrix multiplication with FP32 accumulator
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Store output (FP32)
    wmma::store_matrix_sync(A_out + aRow * N_out + bCol, c_frag, N_out, wmma::mem_row_major);
}

/****************************************************************************************
 * Optional: Add reference points kernel (if needed for deformable attention)
 ****************************************************************************************/

static __global__ void add_reference_kernel(
    float* __restrict__ sampling_loc,           // In-place modification
    const float* __restrict__ reference_points, // Reference points to add
    const int total_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    sampling_loc[idx] += reference_points[idx];
}

/****************************************************************************************
 * Host-side launcher functions
 ****************************************************************************************/

static inline void launch_predict_so_fp16(
    const half* d_Q,
    const half* d_W_SO,
    float* d_SO_out,
    int T,
    int C_in,
    int N_out,
    cudaStream_t stream = 0
) {
    // WMMA configuration: m16n16k16 for FP16 Tensor Core
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    // Calculate grid dimensions
    dim3 gridDim((T + WMMA_M - 1) / WMMA_M, (N_out + WMMA_N - 1) / WMMA_N);
    dim3 blockDim(32, 1);  // 1 warp per block
    
    predict_so_fp16_kernel<WMMA_M, WMMA_N, WMMA_K><<<gridDim, blockDim, 0, stream>>>(
        d_Q, d_W_SO, d_SO_out, T, C_in, N_out
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error launching predict_so_fp16 kernel: %s\n", cudaGetErrorString(err));
    }
}

static inline void launch_predict_attn_fp16(
    const half* d_Q,
    const half* d_W_A,
    float* d_A_out,
    int T,
    int C_in,
    int N_out,
    cudaStream_t stream = 0
) {
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    dim3 gridDim((T + WMMA_M - 1) / WMMA_M, (N_out + WMMA_N - 1) / WMMA_N);
    dim3 blockDim(32, 1);  // 1 warp per block
    
    predict_attn_fp16_kernel<WMMA_M, WMMA_N, WMMA_K><<<gridDim, blockDim, 0, stream>>>(
        d_Q, d_W_A, d_A_out, T, C_in, N_out
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error launching predict_attn_fp16 kernel: %s\n", cudaGetErrorString(err));
    }
}

static inline void launch_add_reference(
    float* d_sampling_loc,
    const float* d_reference_points,
    int total_elements,
    cudaStream_t stream = 0
) {
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    add_reference_kernel<<<blocks, threads, 0, stream>>>(
        d_sampling_loc, d_reference_points, total_elements
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error launching add_reference kernel: %s\n", cudaGetErrorString(err));
    }
}


