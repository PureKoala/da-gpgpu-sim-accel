/*!
**************************************************************************************************
* Deformable Attention Prediction Phase - API Wrappers for FP16
* Connects the FP16 Tensor Core kernels to the public API
* Copyright (c) 2025
**************************************************************************************************/

#include "../deform_attn_cuda.h"
#include "ms_deform_attn_predict_cuda.cuh"

void ms_deform_attn_predict_so_cuda(
    const half* d_Q,
    const half* d_W_SO,
    float* d_SO_out,
    int batch_size,
    int num_query,
    int C_in,
    int N_out,
    cudaStream_t stream)
{
    int T = batch_size * num_query;
    launch_predict_so_fp16(d_Q, d_W_SO, d_SO_out, T, C_in, N_out, stream);
}

void ms_deform_attn_predict_attn_cuda(
    const half* d_Q,
    const half* d_W_A,
    float* d_A_out,
    int batch_size,
    int num_query,
    int C_in,
    int N_out,
    cudaStream_t stream)
{
    int T = batch_size * num_query;
    launch_predict_attn_fp16(d_Q, d_W_A, d_A_out, T, C_in, N_out, stream);
}

void ms_deform_attn_add_reference_cuda(
    float* d_sampling_loc,
    const float* d_reference_points,
    int total_elements,
    cudaStream_t stream)
{
    launch_add_reference(
        d_sampling_loc, d_reference_points, total_elements, stream
    );
}
