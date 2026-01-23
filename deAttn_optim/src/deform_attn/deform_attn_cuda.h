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

/****************************************************************************************
 * Prediction Phase APIs: FP16 Tensor Core GEMM for Sampling Offset & Attention Weight
 * Uses native FP16 without quantization for better accuracy and simpler implementation
 ****************************************************************************************/

// Predict sampling offsets using FP16 Tensor Core (Ampere/Turing wmma instructions)
// Input: Q [batch×num_query, C_in] (FP16), W_SO [C_in, N_out] (FP16)
// Output: SO [batch×num_query, N_out] (FP32 accumulator)
// Note: N_out must be multiple of 8 for optimal Tensor Core utilization (m16n8k16 instruction)
void ms_deform_attn_predict_so_cuda(
    const half* d_Q,
    const half* d_W_SO,
    float* d_SO_out,
    int batch_size,
    int num_query,
    int C_in,
    int N_out,
    cudaStream_t stream = 0);

// Predict attention weights using FP16 Tensor Core
// Input: Q [batch×num_query, C_in] (FP16), W_A [C_in, N_out] (FP16)
// Output: A [batch×num_query, N_out] (FP32 accumulator)
// Note: N_out must be multiple of 8 for optimal Tensor Core utilization (m16n8k16 instruction)
void ms_deform_attn_predict_attn_cuda(
    const half* d_Q,
    const half* d_W_A,
    float* d_A_out,
    int batch_size,
    int num_query,
    int C_in,
    int N_out,
    cudaStream_t stream = 0);

// Optional: Add reference points to sampling offsets (if needed)
void ms_deform_attn_add_reference_cuda(
    float* d_sampling_loc,
    const float* d_reference_points,
    int total_elements,
    cudaStream_t stream = 0);
