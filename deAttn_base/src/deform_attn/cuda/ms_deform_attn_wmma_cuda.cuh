/*!
**************************************************************************************************
* Deformable Attention with Tensor Core (WMMA) Acceleration
* This is an experimental optimization that uses Tensor Cores for the attention computation
* Note: Standard deformable attention doesn't naturally map to matrix multiplication,
* but we can use WMMA for certain aggregation operations when batch sizes are large
* Modified for GPGPU-Sim compatibility
**************************************************************************************************
*/

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// Check if Tensor Core WMMA is available
#if __CUDA_ARCH__ >= 700

// Simplified Tensor Core accelerated kernel for attention weight aggregation
// This kernel assumes attention weights and values can be organized as a matrix multiplication
// Input: attention_matrix [M x K] (half precision)
//        value_matrix [K x N] (half precision)
// Output: output_matrix [M x N] (float or half precision)
__global__ void deform_attn_wmma_aggregate_kernel(
    const __half* __restrict__ attention_weights,  // [num_query, num_sample_points]
    const __half* __restrict__ sampled_values,     // [num_sample_points, channels]
    float* __restrict__ output,                     // [num_query, channels]
    int num_query,
    int num_sample_points,
    int channels)
{
    // Warp and tile indices
    const int warp_m = blockIdx.y * 16;  // Each warp processes 16x16 tile
    const int warp_n = blockIdx.x * 16;
    
    if (warp_m >= num_query || warp_n >= channels) return;

    // Declare fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    // Initialize accumulator to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Iterate over K dimension in steps of 16
    for (int k = 0; k < num_sample_points; k += 16) {
        if (k < num_sample_points) {
            // Load attention weights tile: [warp_m:warp_m+16, k:k+16]
            const __half* a_tile = attention_weights + warp_m * num_sample_points + k;
            wmma::load_matrix_sync(a_frag, a_tile, num_sample_points);
            
            // Load sampled values tile: [k:k+16, warp_n:warp_n+16]
            const __half* b_tile = sampled_values + k * channels + warp_n;
            wmma::load_matrix_sync(b_frag, b_tile, channels);
            
            // Perform matrix multiplication
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    // Store result
    float* c_tile = output + warp_m * channels + warp_n;
    wmma::store_matrix_sync(c_tile, c_frag, channels, wmma::mem_row_major);
}


// Helper function to launch WMMA kernel
void launch_deform_attn_wmma_aggregate(
    const __half* d_attention_weights,
    const __half* d_sampled_values,
    float* d_output,
    int num_query,
    int num_sample_points,
    int channels,
    cudaStream_t stream)
{
    // Ensure dimensions are multiples of 16 (required for WMMA)
    if (num_query % 16 != 0 || channels % 16 != 0) {
        fprintf(stderr, "Warning: WMMA requires dimensions to be multiples of 16. "
                "Falling back to standard kernel.\n");
        return;
    }

    dim3 grid((channels + 15) / 16, (num_query + 15) / 16);
    dim3 block(32);  // One warp per block

    deform_attn_wmma_aggregate_kernel<<<grid, block, 0, stream>>>(
        d_attention_weights,
        d_sampled_values,
        d_output,
        num_query,
        num_sample_points,
        channels
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "WMMA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

#else  // __CUDA_ARCH__ < 700

// Fallback for architectures without Tensor Core support
void launch_deform_attn_wmma_aggregate(
    const __half* d_attention_weights,
    const __half* d_sampled_values,
    float* d_output,
    int num_query,
    int num_sample_points,
    int channels,
    cudaStream_t stream)
{
    fprintf(stderr, "Tensor Core operations require compute capability >= 7.0\n");
}

#endif  // __CUDA_ARCH__ >= 700


/****************************************************************************************
 * Hybrid approach: Standard deformable attention with optional WMMA acceleration
 * 
 * The complete deformable attention can be broken into:
 * 1. Bilinear sampling (irregular memory access - must use standard CUDA cores)
 * 2. Attention weight application (can potentially use WMMA if reorganized)
 * 
 * For best results on modern GPUs:
 * - Use the standard im2col kernel for sampling
 * - Optionally use WMMA for final aggregation if batch size is large
 ****************************************************************************************/

// Note: Full integration of WMMA into deformable attention is complex because:
// 1. Bilinear sampling has irregular memory access patterns
// 2. The attention mechanism doesn't naturally form dense matrix multiplications
// 3. WMMA benefits are most pronounced with large, regular matrix operations
//
// For GPGPU-Sim testing, the standard CUDA core implementation in 
// ms_deform_attn_im2col_cuda.cuh is recommended as it's more compatible
// with simulation and provides better insights into memory access patterns.
