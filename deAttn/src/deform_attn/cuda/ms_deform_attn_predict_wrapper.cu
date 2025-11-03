/*!
**************************************************************************************************
* Deformable Attention Prediction Phase - API Wrappers
* Connects the INT8 Tensor Core kernels to the public API
* Copyright (c) 2025
**************************************************************************************************/

#include "../deform_attn_cuda.h"
#include "ms_deform_attn_predict_cuda.cuh"

void ms_deform_attn_predict_so_cuda(
    const int8_t* d_Q,
    const int8_t* d_W_SO,
    int32_t* d_SO_out,
    int batch_size,
    int num_query,
    int C_in,
    int N_out,
    cudaStream_t stream)
{
    int T = batch_size * num_query;
    launch_predict_so_int8_ampere(d_Q, d_W_SO, d_SO_out, T, C_in, N_out, stream);
}

void ms_deform_attn_predict_attn_cuda(
    const int8_t* d_Q,
    const int8_t* d_W_A,
    int32_t* d_A_out,
    int batch_size,
    int num_query,
    int C_in,
    int N_out,
    cudaStream_t stream)
{
    int T = batch_size * num_query;
    launch_predict_attn_int8_ampere(d_Q, d_W_A, d_A_out, T, C_in, N_out, stream);
}

void ms_deform_attn_dequantize_so_cuda(
    const int32_t* d_SO_int32,
    float* d_sampling_loc,
    const float* d_reference_points,
    float scale,
    int zero_point,
    int total_elements,
    bool add_reference,
    cudaStream_t stream)
{
    launch_dequantize_so(
        d_SO_int32, d_sampling_loc, d_reference_points,
        scale, zero_point, total_elements, add_reference, stream
    );
}

void ms_deform_attn_dequantize_attn_cuda(
    const int32_t* d_A_int32,
    float* d_attn_weight,
    float scale,
    int zero_point,
    int total_elements,
    cudaStream_t stream)
{
    launch_dequantize_attn(
        d_A_int32, d_attn_weight,
        scale, zero_point, total_elements, stream
    );
}
