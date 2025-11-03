/*!
**************************************************************************************************
* Deformable Attention Prediction Phase with INT8 Tensor Core
* Implements sampling offset and attention weight prediction using WMMA API
* Demonstrates K-dimension padding overhead for GPGPU-Sim analysis
* Copyright (c) 2025
**************************************************************************************************
*/

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdint>
#include <algorithm>

using namespace nvcuda;

/****************************************************************************************
 * Host-side Utility Functions: Quantization and Padding
 ****************************************************************************************/

// Quantize FP32 array to INT8
inline void quantize_to_int8(const float* input, int8_t* output, size_t n, 
                      float scale, int zero_point) {
    for (size_t i = 0; i < n; i++) {
        float val = input[i] / scale + zero_point;
        val = fmax(-128.0f, fmin(127.0f, val)); // Clamp to INT8 range
        output[i] = static_cast<int8_t>(roundf(val));
    }
}

// Compute scale and zero_point for quantization
inline QuantizationParams compute_quantization_params(const float* data, size_t n) {
    float min_val = data[0];
    float max_val = data[0];
    
    for (size_t i = 1; i < n; i++) {
        min_val = fmin(min_val, data[i]);
        max_val = fmax(max_val, data[i]);
    }
    
    float scale = (max_val - min_val) / 255.0f;
    if (scale < 1e-8f) scale = 1e-8f; // Avoid division by zero
    
    int zero_point = static_cast<int>(roundf(-min_val / scale));
    zero_point = std::max(0, std::min(255, zero_point));
    
    return QuantizationParams(scale, zero_point);
}

// Pad weight matrix in K dimension: [C × K_orig] → [C × K_padded]
// Note: This function name is historical. It actually pads the second dimension
// which corresponds to N (output features) in our GEMM: C = A × B where B is [K×N]
inline void pad_weights_k_dimension(const int8_t* input, int8_t* output, 
                             int C, int K_orig, int K_padded) {
    for (int c = 0; c < C; c++) {
        for (int k = 0; k < K_orig; k++) {
            output[c * K_padded + k] = input[c * K_orig + k];
        }
        // Zero-pad remaining elements
        for (int k = K_orig; k < K_padded; k++) {
            output[c * K_padded + k] = 0;
        }
    }
}

// Pad weight matrix in both K and N dimensions: [C × N_orig] → [C × N_padded]
// Used when N dimension needs padding (e.g., N=4 → N=8 for INT8 Tensor Core minimum)
inline void pad_weights_kn_dimension(const int8_t* input, int8_t* output,
                              int C, int N_orig, int N_padded) {
    // For N dimension padding, same logic as K dimension
    pad_weights_k_dimension(input, output, C, N_orig, N_padded);
}

/****************************************************************************************
 * Device Kernels: INT8 GEMM using Ampere Tensor Core (PTX inline assembly)
 ****************************************************************************************/

/**
 * Kernel: Sampling Offset Prediction (N dimension padding)
 * 
 * Computation: SO = Q × W_SO
 * Matrix dimensions:
 *   - Q: [T × C_in] where T = batch × num_query (INT8, row-major)
 *   - W_SO: [C_in × N_out] (INT8, stored for col-major access in WMMA)
 *   - SO: [T × N_out] (INT32 accumulator)
 * 
 * WMMA instruction: mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32
 *   - M=16: Output rows per instruction
 *   - N=8: Output columns per instruction (MINIMUM for INT8 Tensor Core)
 *   - K=16: Inner dimension per instruction
 * 
 * Key observation for GPGPU-Sim (N-Padding Problem):
 *   - Actual N needed: num_heads × num_levels × num_point (e.g., 8 or less)
 *   - Hardware minimum N: 8 (for m16n8k16 instruction)
 *   - If actual N < 8: Must pad to N=8, wasting computation on padded columns
 *   - Example: N=4 → N=8 padding wastes 50% of output computation
 */
template <int WMMA_M = 16, int WMMA_N = 16, int WMMA_K = 16>
static __global__ void predict_so_int8_ampere_kernel(
    const int8_t* __restrict__ Q,          // [T × C_in], T = batch × num_query
    const int8_t* __restrict__ W_SO,       // [C_in × N_out], N_out must be multiple of 8
    int32_t* __restrict__ SO_out,          // [T × N_out]
    const int T,                           // Total queries (batch × num_query)
    const int C_in,                        // Input feature dimension (must be multiple of 16)
    const int N_out                        // Output dimension (must be multiple of 8 for INT8 TC)
) {
    // Warp and lane IDs
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // Declare the fragments for WMMA operations (m16n8k16 for INT8)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frag;
    
    // Initialize accumulator to zero
    wmma::fill_fragment(c_frag, 0);
    
    // Compute global position in output matrix
    int aRow = warpM * WMMA_M;
    int bCol = warpN * WMMA_N;
    
    // Boundary check
    if (aRow >= T || bCol >= N_out) return;
    
    // **CRITICAL INTERCEPTION POINT FOR GPGPU-SIM**
    // N-Padding Issue: If N_out < 8, some columns are padded zeros
    // Hardware processes full N=8 tile, but padded columns waste computation
    //
    // Loop over K dimension in chunks of WMMA_K (16)
    for (int k = 0; k < C_in; k += WMMA_K) {
        // Load the inputs from global memory to fragments
        // Matrix A (Q): row-major layout [T × C_in]
        wmma::load_matrix_sync(a_frag, Q + aRow * C_in + k, C_in);
        
        // Matrix B (W_SO): column-major layout [C_in × N_out]
        // Note: WMMA expects col-major for B, but our W is stored row-major [C_in × N_out]
        // We need to transpose the indexing
        wmma::load_matrix_sync(b_frag, W_SO + k * N_out + bCol, N_out);
        
        // Perform the matrix multiplication
        // **INT8 TC LIMITATION: Minimum N=8, wastes computation if actual N < 8**
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    // Store the output (INT32 accumulator)
    wmma::store_matrix_sync(SO_out + aRow * N_out + bCol, c_frag, N_out, wmma::mem_row_major);
}

/**
 * Kernel: Attention Weight Prediction (N dimension padding)
 * 
 * Computation: A = Q × W_A
 * Matrix dimensions:
 *   - Q: [T × C_in] where T = batch × num_query (INT8, row-major)
 *   - W_A: [C_in × N_out] (INT8, stored for col-major access in WMMA)
 *   - A: [T × N_out] (INT32 accumulator)
 * 
 * WMMA instruction: mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32
 *   - M=16: Output rows per instruction
 *   - N=8: Output columns per instruction (MINIMUM for INT8 Tensor Core)
 *   - K=16: Inner dimension per instruction
 * 
 * Key observation for GPGPU-Sim (N-Padding Problem):
 *   - Actual N needed: num_heads × num_levels × num_point (e.g., 4 or less)
 *   - Hardware minimum N: 8 (for m16n8k16 instruction)
 *   - If actual N < 8: Must pad to N=8, wasting computation on padded columns
 *   - Example: N=4 → N=8 padding wastes 50% of output computation
 *   - Worse case: If both input/output need alignment, utilization drops further
 */
template <int WMMA_M = 16, int WMMA_N = 16, int WMMA_K = 16>
static __global__ void predict_attn_int8_ampere_kernel(
    const int8_t* __restrict__ Q,          // [T × C_in]
    const int8_t* __restrict__ W_A,        // [C_in × N_out], N_out must be multiple of 8
    int32_t* __restrict__ A_out,           // [T × N_out]
    const int T,
    const int C_in,
    const int N_out                        // Must be multiple of 8 (padded if original < 8)
) {
    // Warp and lane IDs
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // Declare WMMA fragments (m16n8k16 for INT8)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frag;
    
    // Initialize accumulator
    wmma::fill_fragment(c_frag, 0);
    
    // Compute global position
    int aRow = warpM * WMMA_M;
    int bCol = warpN * WMMA_N;
    
    if (aRow >= T || bCol >= N_out) return;
    
    // **CRITICAL INTERCEPTION POINT FOR GPGPU-SIM**
    // N-Padding Issue: Hardware requires N=8 minimum for INT8 Tensor Core
    // If actual attention weight output dimension < 8, must pad to 8
    // This wastes computation on padded zero columns
    //
    // Loop over K dimension in chunks of WMMA_K (16)
    for (int k = 0; k < C_in; k += WMMA_K) {
        // Load matrix A (Q)
        wmma::load_matrix_sync(a_frag, Q + aRow * C_in + k, C_in);
        
        // Load matrix B (W_A)
        wmma::load_matrix_sync(b_frag, W_A + k * N_out + bCol, N_out);
        
        // **INT8 TC LIMITATION: Minimum N=8, wastes computation if actual N < 8**
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Store output
    wmma::store_matrix_sync(A_out + aRow * N_out + bCol, c_frag, N_out, wmma::mem_row_major);
}

/****************************************************************************************
 * Dequantization Kernel: INT32 → FP32 with reference point addition
 ****************************************************************************************/

/**
 * Kernel: Dequantize and convert INT32 results to FP32
 * 
 * For sampling offsets:
 *   sampling_loc[i] = (SO_int32[i] - zero_point) * scale + reference_point[i]
 * 
 * For attention weights:
 *   attn_weight[i] = (A_int32[i] - zero_point) * scale
 *   Then apply softmax normalization (optional, done in separate kernel)
 */
static __global__ void dequantize_so_kernel(
    const int32_t* __restrict__ SO_int32,    // INT32 accumulator output
    float* __restrict__ sampling_loc,         // FP32 output
    const float* __restrict__ reference_points, // Optional reference points to add
    const float scale,
    const int zero_point,
    const int total_elements,
    const bool add_reference                  // Whether to add reference points
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    // Dequantize: convert INT32 → FP32
    float dequant_val = static_cast<float>(SO_int32[idx] - zero_point) * scale;
    
    // Add reference point if needed (for deformable attention)
    if (add_reference && reference_points != nullptr) {
        dequant_val += reference_points[idx];
    }
    
    sampling_loc[idx] = dequant_val;
}

static __global__ void dequantize_attn_kernel(
    const int32_t* __restrict__ A_int32,     // INT32 accumulator output
    float* __restrict__ attn_weight,          // FP32 output
    const float scale,
    const int zero_point,
    const int total_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    // Dequantize: convert INT32 → FP32
    float dequant_val = static_cast<float>(A_int32[idx] - zero_point) * scale;
    
    // Clamp to [0, 1] for attention weights (before softmax)
    attn_weight[idx] = fmaxf(0.0f, fminf(1.0f, dequant_val));
}

/****************************************************************************************
 * Host-side launcher functions
 ****************************************************************************************/

static inline void launch_predict_so_int8_ampere(
    const int8_t* d_Q,
    const int8_t* d_W_SO,
    int32_t* d_SO_out,
    int T,
    int C_in,
    int N_out,
    cudaStream_t stream = 0
) {
    // WMMA configuration: m16n16k16 for INT8 Ampere Tensor Core
    const int WMMA_M = 16;
    const int WMMA_N = 16;  // Using N=16 for WMMA API compatibility
    
    // Calculate grid dimensions
    // Each block contains warps that process different output tiles
    dim3 gridDim((T + WMMA_M - 1) / WMMA_M, (N_out + WMMA_N - 1) / WMMA_N);
    dim3 blockDim(32, 1);  // 1 warp per block
    
    predict_so_int8_ampere_kernel<WMMA_M, WMMA_N, 16><<<gridDim, blockDim, 0, stream>>>(
        d_Q, d_W_SO, d_SO_out, T, C_in, N_out
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error launching predict_so kernel: %s\n", cudaGetErrorString(err));
    }
}

static inline void launch_predict_attn_int8_ampere(
    const int8_t* d_Q,
    const int8_t* d_W_A,
    int32_t* d_A_out,
    int T,
    int C_in,
    int N_out,
    cudaStream_t stream = 0
) {
    const int WMMA_M = 16;
    const int WMMA_N = 16;  // Using N=16 for WMMA API compatibility
    
    dim3 gridDim((T + WMMA_M - 1) / WMMA_M, (N_out + WMMA_N - 1) / WMMA_N);
    dim3 blockDim(32, 1);  // 1 warp per block
    
    predict_attn_int8_ampere_kernel<WMMA_M, WMMA_N, 16><<<gridDim, blockDim, 0, stream>>>(
        d_Q, d_W_A, d_A_out, T, C_in, N_out
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error launching predict_attn kernel: %s\n", cudaGetErrorString(err));
    }
}

static inline void launch_dequantize_so(
    const int32_t* d_SO_int32,
    float* d_sampling_loc,
    const float* d_reference_points,
    float scale,
    int zero_point,
    int total_elements,
    bool add_reference,
    cudaStream_t stream = 0
) {
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    dequantize_so_kernel<<<blocks, threads, 0, stream>>>(
        d_SO_int32, d_sampling_loc, d_reference_points,
        scale, zero_point, total_elements, add_reference
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error launching dequantize_so kernel: %s\n", cudaGetErrorString(err));
    }
}

static inline void launch_dequantize_attn(
    const int32_t* d_A_int32,
    float* d_attn_weight,
    float scale,
    int zero_point,
    int total_elements,
    cudaStream_t stream = 0
) {
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    dequantize_attn_kernel<<<blocks, threads, 0, stream>>>(
        d_A_int32, d_attn_weight, scale, zero_point, total_elements
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error launching dequantize_attn kernel: %s\n", cudaGetErrorString(err));
    }
}
