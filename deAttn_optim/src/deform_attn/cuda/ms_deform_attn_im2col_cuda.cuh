/*!
**************************************************************************************************
* Deformable Attention Im2Col CUDA Implementation (No PyTorch/ATen dependencies)
* Modified for GPGPU-Sim compatibility
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0
**************************************************************************************************
*/

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <algorithm>
#include "deform_attn_accelerator.cuh"  // 硬件加速器接口

// CUDA kernel loop macro
#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                         \
      i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;

inline int GET_BLOCKS(const int N, const int num_threads = CUDA_NUM_THREADS) {
  return (N + num_threads - 1) / num_threads;
}

// Atomic add for double precision (some older architectures need this)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

/****************************************************************************************
 * Forward: Deformable Attention Im2Col with Bilinear Interpolation
 ****************************************************************************************/

template <typename scalar_t>
__device__ scalar_t ms_deform_attn_im2col_bilinear(
    const scalar_t* __restrict__ bottom_data, 
    const int height, 
    const int width, 
    const int nheads, 
    const int channels,
    const scalar_t h, 
    const scalar_t w, 
    const int m, 
    const int c)
{
  const int h_low = floor(h);
  const int w_low = floor(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh;
  const scalar_t hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
  }
  
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }
  
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
  }
  
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
  }

  const scalar_t w1 = hh * hw;
  const scalar_t w2 = hh * lw;
  const scalar_t w3 = lh * hw;
  const scalar_t w4 = lh * lw;

  return (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
}


template <typename scalar_t>
__global__ void ms_deformable_im2col_gpu_kernel(
    const int n,
    const scalar_t* __restrict__ data_value, 
    const int64_t* __restrict__ data_spatial_shapes,
    const int64_t* __restrict__ data_level_start_index, 
    const scalar_t* __restrict__ data_sampling_loc,
    const scalar_t* __restrict__ data_attn_weight,
    const int batch_size, 
    const int spatial_size, 
    const int num_heads,
    const int channels, 
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t* __restrict__ data_col)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    // const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    scalar_t* data_col_ptr = data_col + index;
    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
    scalar_t col = 0;
    
    // 硬件加速器优化：处理每个level（保持与baseline一致的计算语义）
    for (int l_col = 0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const scalar_t* data_value_ptr = data_value + (data_value_ptr_init_offset + level_start_id * qid_stride);
      
      // 若num_point过大，直接使用baseline路径保证正确性
      constexpr int kMaxNumPoint = 64;
      if (num_point > kMaxNumPoint) {
        for (int p_col = 0; p_col < num_point; ++p_col) {
          const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr + p_col * 2];
          const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + p_col * 2 + 1];
          const scalar_t weight = data_attn_weight[data_weight_ptr + p_col];
          const scalar_t h_im = loc_h * spatial_h - 0.5;
          const scalar_t w_im = loc_w * spatial_w - 0.5;
          if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
            col += ms_deform_attn_im2col_bilinear(
                data_value_ptr, spatial_h, spatial_w, num_heads, channels,
                h_im, w_im, m_col, c_col) * weight;
          }
        }
        data_weight_ptr += num_point;
        data_loc_w_ptr += num_point * 2;
        continue;
      }

      // 线程私有缓冲区，避免shared内存竞争
      float s_coords[2 * kMaxNumPoint];
      float s_weights[kMaxNumPoint];
      bool s_mask[kMaxNumPoint];
      float s_tile_buffer[16 * 17]; // 17列避免bank冲突

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr + p_col * 2];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + p_col * 2 + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr + p_col];
        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        s_coords[p_col * 2] = (float)w_im;
        s_coords[p_col * 2 + 1] = (float)h_im;
        s_weights[p_col] = (float)weight;
        s_mask[p_col] = false;
      }

      int mode = 3; // 默认DISCRETE
      int base_x = 0, base_y = 0, tile_w = 16, tile_h = 16;

      #ifdef __GPGPU_SIM__
      // 硬件加速器调用 #1: PCB权重预筛选（阈值=0，保持数值一致性）
      const float threshold = 0.0f;
      __deform_pcb(s_weights, threshold, s_mask, num_point);

      // 硬件加速器调用 #2: TBC边界检查
      mode = __deform_tbc(s_coords, s_mask, num_point, &base_x, &base_y, &tile_w, &tile_h);

      // 安全边界：若tile越界或过大，退回DISCRETE模式
      if (mode != 3) {
        if (tile_w <= 0 || tile_h <= 0 || tile_w > 16 || tile_h > 16 ||
            base_x < 0 || base_y < 0 || base_x + tile_w > spatial_w || base_y + tile_h > spatial_h) {
          mode = 3;
        }
      }

      // 硬件加速器调用 #3: TMA Tile加载（如果聚集）
      if (mode != 3) {
        const scalar_t* tile_src = data_value_ptr +
            (base_y * spatial_w + base_x) * qid_stride +
            m_col * channels + c_col;
        __deform_tma(tile_src, s_tile_buffer, tile_w, tile_h, spatial_w * qid_stride, mode);
      }
      #endif

      // 插值和累加（保证与baseline相同的边界与权重语义）
      for (int p_col = 0; p_col < num_point; ++p_col)
      {
        if (!s_mask[p_col]) {
          const scalar_t weight = (scalar_t)s_weights[p_col];
          const scalar_t w_im = (scalar_t)s_coords[p_col * 2];
          const scalar_t h_im = (scalar_t)s_coords[p_col * 2 + 1];

          if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
          {
            scalar_t sampled_val;
            #ifdef __GPGPU_SIM__
            if (mode != 3) {
              const float local_x = (float)(w_im - base_x);
              const float local_y = (float)(h_im - base_y);
              if (local_x >= 0.0f && local_y >= 0.0f && local_x < tile_w && local_y < tile_h) {
                sampled_val = (scalar_t)__deform_interp(s_tile_buffer, local_x, local_y, mode);
              } else {
                sampled_val = ms_deform_attn_im2col_bilinear(
                    data_value_ptr, spatial_h, spatial_w, num_heads, channels,
                    h_im, w_im, m_col, c_col);
              }
            } else {
              sampled_val = ms_deform_attn_im2col_bilinear(
                  data_value_ptr, spatial_h, spatial_w, num_heads, channels,
                  h_im, w_im, m_col, c_col);
            }
            #else
            sampled_val = ms_deform_attn_im2col_bilinear(
                data_value_ptr, spatial_h, spatial_w, num_heads, channels,
                h_im, w_im, m_col, c_col);
            #endif
            col += sampled_val * weight;
          }
        }
      }

      data_weight_ptr += num_point;
      data_loc_w_ptr += num_point * 2;
    }
    *data_col_ptr = col;
  }
}


/****************************************************************************************
 * Backward: Deformable Attention Col2Im with Gradient Computation
 ****************************************************************************************/

template <typename scalar_t>
__device__ void ms_deform_attn_col2im_bilinear(
    const scalar_t* __restrict__ bottom_data, 
    const int height, 
    const int width, 
    const int nheads, 
    const int channels,
    const scalar_t h, 
    const scalar_t w, 
    const int m, 
    const int c,
    const scalar_t top_grad,
    const scalar_t attn_weight,
    scalar_t* __restrict__ grad_value, 
    scalar_t* grad_sampling_loc,
    scalar_t* grad_attn_weight)
{
  const int h_low = floor(h);
  const int w_low = floor(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh;
  const scalar_t hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  const scalar_t w1 = hh * hw;
  const scalar_t w2 = hh * lw;
  const scalar_t w3 = lh * hw;
  const scalar_t w4 = lh * lw;
  const scalar_t top_grad_value = top_grad * attn_weight;
  scalar_t grad_h_weight = 0;
  scalar_t grad_w_weight = 0;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_h_weight -= hw * v1;
    grad_w_weight -= hh * v1;
    atomicAdd(grad_value + ptr1, w1 * top_grad_value);
  }
  
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;
    atomicAdd(grad_value + ptr2, w2 * top_grad_value);
  }
  
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    atomicAdd(grad_value + ptr3, w3 * top_grad_value); 
  }
  
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    atomicAdd(grad_value + ptr4, w4 * top_grad_value);
  }

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  *grad_attn_weight = top_grad * val;
  *grad_sampling_loc = width * grad_w_weight * top_grad_value;
  *(grad_sampling_loc + 1) = height * grad_h_weight * top_grad_value;
}


template <typename scalar_t, unsigned int blockSize>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce(
    const int n,
    const scalar_t* __restrict__ grad_col,
    const scalar_t* __restrict__ data_value,
    const int64_t* __restrict__ data_spatial_shapes,
    const int64_t* __restrict__ data_level_start_index, 
    const scalar_t* __restrict__ data_sampling_loc,
    const scalar_t* __restrict__ data_attn_weight,
    const int batch_size, 
    const int spatial_size, 
    const int num_heads,
    const int channels, 
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t* __restrict__ grad_value,
    scalar_t* __restrict__ grad_sampling_loc,
    scalar_t* __restrict__ grad_attn_weight)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    __shared__ scalar_t cache_grad_sampling_loc[blockSize * 2];
    __shared__ scalar_t cache_grad_attn_weight[blockSize];
    unsigned int tid = threadIdx.x;
    
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    // const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t* data_value_ptr = data_value + value_ptr_offset;
      scalar_t* grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        
        *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
        {
          ms_deform_attn_col2im_bilinear(
            data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im, w_im, m_col, c_col,
            top_grad, weight, grad_value_ptr, 
            cache_grad_sampling_loc + (threadIdx.x << 1), 
            cache_grad_attn_weight + threadIdx.x);
        }
        
        __syncthreads();
        
        if (tid == 0)
        {
          scalar_t _grad_w = cache_grad_sampling_loc[0];
          scalar_t _grad_h = cache_grad_sampling_loc[1];
          scalar_t _grad_a = cache_grad_attn_weight[0];
          int sid = 2;
          for (unsigned int t = 1; t < blockSize; ++t)
          {
            _grad_w += cache_grad_sampling_loc[sid];
            _grad_h += cache_grad_sampling_loc[sid + 1];
            _grad_a += cache_grad_attn_weight[t];
            sid += 2;
          }
          
          *grad_sampling_loc = _grad_w;
          *(grad_sampling_loc + 1) = _grad_h;
          *grad_attn_weight = _grad_a;
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}


/****************************************************************************************
 * Host-side launcher functions
 ****************************************************************************************/

template <typename scalar_t>
void ms_deformable_im2col_cuda(
    cudaStream_t stream,
    const scalar_t* data_value,
    const int64_t* data_spatial_shapes, 
    const int64_t* data_level_start_index, 
    const scalar_t* data_sampling_loc,
    const scalar_t* data_attn_weight,
    const int batch_size,
    const int spatial_size, 
    const int num_heads, 
    const int channels, 
    const int num_levels, 
    const int num_query,
    const int num_point,
    scalar_t* data_col)
{
  const int num_kernels = batch_size * num_query * num_heads * channels;
  const int num_threads = CUDA_NUM_THREADS;
  
  ms_deformable_im2col_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_kernels, num_threads), num_threads, 0, stream>>>(
      num_kernels, data_value, data_spatial_shapes, data_level_start_index, 
      data_sampling_loc, data_attn_weight, 
      batch_size, spatial_size, num_heads, channels, num_levels, num_query, num_point, 
      data_col);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in ms_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}


template <typename scalar_t>
void ms_deformable_col2im_cuda(
    cudaStream_t stream,
    const scalar_t* grad_col,
    const scalar_t* data_value,
    const int64_t* data_spatial_shapes,
    const int64_t* data_level_start_index, 
    const scalar_t* data_sampling_loc,
    const scalar_t* data_attn_weight,
    const int batch_size, 
    const int spatial_size, 
    const int num_heads,
    const int channels, 
    const int num_levels,
    const int num_query,
    const int num_point, 
    scalar_t* grad_value,
    scalar_t* grad_sampling_loc,
    scalar_t* grad_attn_weight)
{
  const int num_kernels = batch_size * num_query * num_heads * channels;
  const int num_threads = (channels > CUDA_NUM_THREADS) ? CUDA_NUM_THREADS : channels;

  // Use blocksize 64 for better occupancy on various architectures
  ms_deformable_col2im_gpu_kernel_shm_reduce<scalar_t, 64>
      <<<GET_BLOCKS(num_kernels, num_threads), num_threads, 0, stream>>>(
      num_kernels, grad_col, data_value, data_spatial_shapes, data_level_start_index, 
      data_sampling_loc, data_attn_weight,
      batch_size, spatial_size, num_heads, channels, num_levels, num_query, num_point,
      grad_value, grad_sampling_loc, grad_attn_weight);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in ms_deformable_col2im_cuda: %s\n", cudaGetErrorString(err));
  }
}
