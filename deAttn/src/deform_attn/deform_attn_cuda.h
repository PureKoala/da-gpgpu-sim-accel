/*!
**************************************************************************************************
* Deformable Attention - Pure CUDA Implementation (No PyTorch)
* Modified for GPGPU-Sim compatibility
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0
**************************************************************************************************
*/

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Pure CUDA forward interface (no PyTorch dependencies)
// All pointers are to device memory
void ms_deform_attn_cuda_forward(
    const float* d_value,                    // [batch, spatial_size, num_heads, channels]
    const int64_t* d_spatial_shapes,         // [num_levels, 2] (height, width)
    const int64_t* d_level_start_index,      // [num_levels]
    const float* d_sampling_loc,             // [batch, num_query, num_heads, num_levels, num_point, 2]
    const float* d_attn_weight,              // [batch, num_query, num_heads, num_levels, num_point]
    float* d_output,                         // [batch, num_query, num_heads, channels]
    int batch_size,
    int spatial_size,
    int num_heads,
    int channels,
    int num_levels,
    int num_query,
    int num_point,
    cudaStream_t stream = 0);

// Pure CUDA backward interface (no PyTorch dependencies)
void ms_deform_attn_cuda_backward(
    const float* d_grad_output,              // [batch, num_query, num_heads, channels]
    const float* d_value,                    // [batch, spatial_size, num_heads, channels]
    const int64_t* d_spatial_shapes,         // [num_levels, 2]
    const int64_t* d_level_start_index,      // [num_levels]
    const float* d_sampling_loc,             // [batch, num_query, num_heads, num_levels, num_point, 2]
    const float* d_attn_weight,              // [batch, num_query, num_heads, num_levels, num_point]
    float* d_grad_value,                     // [batch, spatial_size, num_heads, channels]
    float* d_grad_sampling_loc,              // [batch, num_query, num_heads, num_levels, num_point, 2]
    float* d_grad_attn_weight,               // [batch, num_query, num_heads, num_levels, num_point]
    int batch_size,
    int spatial_size,
    int num_heads,
    int channels,
    int num_levels,
    int num_query,
    int num_point,
    cudaStream_t stream = 0);
