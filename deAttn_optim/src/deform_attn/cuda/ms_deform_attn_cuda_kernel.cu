/*!
**************************************************************************************************
* Deformable Attention CUDA Kernel Wrapper (No PyTorch/ATen dependencies)
* Modified for GPGPU-Sim compatibility
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0
**************************************************************************************************
*/

#include "../deform_attn_cuda.h"
#include "ms_deform_attn_im2col_cuda.cuh"
#include <cstring>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)


void ms_deform_attn_cuda_forward(
    const float* d_value,
    const int64_t* d_spatial_shapes,
    const int64_t* d_level_start_index,
    const float* d_sampling_loc,
    const float* d_attn_weight,
    float* d_output,
    int batch_size,
    int spatial_size,
    int num_heads,
    int channels,
    int num_levels,
    int num_query,
    int num_point,
    cudaStream_t stream)
{
    // Validate inputs
    if (!d_value || !d_spatial_shapes || !d_level_start_index || 
        !d_sampling_loc || !d_attn_weight || !d_output) {
        fprintf(stderr, "Error: NULL pointer passed to ms_deform_attn_cuda_forward\n");
        return;
    }

    // Initialize output to zero
    const size_t output_size = (size_t)batch_size * num_query * num_heads * channels;
    CUDA_CHECK(cudaMemsetAsync(d_output, 0, output_size * sizeof(float), stream));

    // Launch im2col kernel
    ms_deformable_im2col_cuda<float>(
        stream,
        d_value,
        d_spatial_shapes,
        d_level_start_index,
        d_sampling_loc,
        d_attn_weight,
        batch_size,
        spatial_size,
        num_heads,
        channels,
        num_levels,
        num_query,
        num_point,
        d_output);

    CUDA_CHECK(cudaGetLastError());
}


void ms_deform_attn_cuda_backward(
    const float* d_grad_output,
    const float* d_value,
    const int64_t* d_spatial_shapes,
    const int64_t* d_level_start_index,
    const float* d_sampling_loc,
    const float* d_attn_weight,
    float* d_grad_value,
    float* d_grad_sampling_loc,
    float* d_grad_attn_weight,
    int batch_size,
    int spatial_size,
    int num_heads,
    int channels,
    int num_levels,
    int num_query,
    int num_point,
    cudaStream_t stream)
{
    // Validate inputs
    if (!d_grad_output || !d_value || !d_spatial_shapes || !d_level_start_index || 
        !d_sampling_loc || !d_attn_weight || !d_grad_value || 
        !d_grad_sampling_loc || !d_grad_attn_weight) {
        fprintf(stderr, "Error: NULL pointer passed to ms_deform_attn_cuda_backward\n");
        return;
    }

    // Initialize gradients to zero
    const size_t grad_value_size = (size_t)batch_size * spatial_size * num_heads * channels;
    const size_t grad_sampling_loc_size = (size_t)batch_size * num_query * num_heads * num_levels * num_point * 2;
    const size_t grad_attn_weight_size = (size_t)batch_size * num_query * num_heads * num_levels * num_point;
    
    CUDA_CHECK(cudaMemsetAsync(d_grad_value, 0, grad_value_size * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(d_grad_sampling_loc, 0, grad_sampling_loc_size * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(d_grad_attn_weight, 0, grad_attn_weight_size * sizeof(float), stream));

    // Launch col2im kernel
    ms_deformable_col2im_cuda<float>(
        stream,
        d_grad_output,
        d_value,
        d_spatial_shapes,
        d_level_start_index,
        d_sampling_loc,
        d_attn_weight,
        batch_size,
        spatial_size,
        num_heads,
        channels,
        num_levels,
        num_query,
        num_point,
        d_grad_value,
        d_grad_sampling_loc,
        d_grad_attn_weight);

    CUDA_CHECK(cudaGetLastError());
}
