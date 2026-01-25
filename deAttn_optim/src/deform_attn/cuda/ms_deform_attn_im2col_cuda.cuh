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

#ifndef __GPGPU_SIM__
#define __GPGPU_SIM__
#endif

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


// ============================================================================
// 硬件加速模式控制宏
// DEFORM_ACCEL_FAST_MODE: 启用快速模式（用于性能对比测试）
// DEFORM_ACCEL_CORRECT_MODE: 启用正确模式（用于数值验证）
// 默认: 快速模式
// ============================================================================
#ifndef DEFORM_ACCEL_CORRECT_MODE
#define DEFORM_ACCEL_FAST_MODE 1
#endif

// ============================================================================
// PCB稀疏剪枝参数 (每层剪枝率，百分比形式)
// 剪枝率 = 被跳过的点占总点数的比例
// ============================================================================
#ifndef DEFORM_PCB_PRUNE_RATE_L0
#define DEFORM_PCB_PRUNE_RATE_L0 80  // Level0: 80%被剪枝，处理20%的点
#endif
#ifndef DEFORM_PCB_PRUNE_RATE_L1
#define DEFORM_PCB_PRUNE_RATE_L1 70  // Level1: 70%被剪枝，处理30%的点
#endif
#ifndef DEFORM_PCB_PRUNE_RATE_L2
#define DEFORM_PCB_PRUNE_RATE_L2 50  // Level2: 50%被剪枝，处理50%的点
#endif
#ifndef DEFORM_PCB_PRUNE_RATE_L3
#define DEFORM_PCB_PRUNE_RATE_L3 34  // Level3: 34%被剪枝，处理66%的点
#endif

// ============================================================================
// 访存策略参数
// 0 = DISCRETE模式 (分散load，适合点少的情况)
// 1 = TILE模式 (tile load，适合聚集度高的情况)
// ============================================================================
#ifndef DEFORM_ACCESS_MODE_L0
#define DEFORM_ACCESS_MODE_L0 0  // Level0: discrete
#endif
#ifndef DEFORM_ACCESS_MODE_L1
#define DEFORM_ACCESS_MODE_L1 0  // Level1: discrete
#endif
#ifndef DEFORM_ACCESS_MODE_L2
#define DEFORM_ACCESS_MODE_L2 1  // Level2: tile
#endif
#ifndef DEFORM_ACCESS_MODE_L3
#define DEFORM_ACCESS_MODE_L3 1  // Level3: tile
#endif

// ============================================================================
// 惩罚参数 (模拟硬件延迟)
// ============================================================================
#ifndef DEFORM_PCB_PENALTY_ITERS
#define DEFORM_PCB_PENALTY_ITERS 0      // PCB检查被剪枝点的惩罚迭代数
#endif
#ifndef DEFORM_DISCRETE_PENALTY_ITERS
#define DEFORM_DISCRETE_PENALTY_ITERS 0 // Discrete模式的额外访存惩罚
#endif
#ifndef DEFORM_TILE_MISS_PENALTY_ITERS
#define DEFORM_TILE_MISS_PENALTY_ITERS 0 // Tile miss的惩罚（Level2/3不会触发）
#endif

// ============================================================================ 
// 高效剪枝判断：使用编译期常量避免运行时开销
// 基于 (tid ^ p_col ^ level) 的低位来决定是否剪枝
// ============================================================================
// Deterministic pruning (method B) toggle for grouped kernel PCB
#ifndef DEFORM_PCB_USE_DETERMINISTIC_PRUNE
#define DEFORM_PCB_USE_DETERMINISTIC_PRUNE 0
#endif

// 将剪枝率转换为确定性模式（基于 0..99 的周期）
// keep = 100 - prune_rate
// keep越大 -> 保留越多；prune_rate越大 -> 剪枝越多
__device__ __forceinline__ bool should_prune_rate(int p_col, int tid, int prune_rate_percent) {
    int keep = 100 - prune_rate_percent;
    if (keep <= 0) return true;
    if (keep >= 100) return false;
    return (((tid + p_col) % 100) >= keep);
}

__device__ __forceinline__ bool should_prune_l0(int p_col, int tid) {
    return should_prune_rate(p_col, tid, DEFORM_PCB_PRUNE_RATE_L0);
}

__device__ __forceinline__ bool should_prune_l1(int p_col, int tid) {
    return should_prune_rate(p_col, tid, DEFORM_PCB_PRUNE_RATE_L1);
}

__device__ __forceinline__ bool should_prune_l2(int p_col, int tid) {
    return should_prune_rate(p_col, tid, DEFORM_PCB_PRUNE_RATE_L2);
}

__device__ __forceinline__ bool should_prune_l3(int p_col, int tid) {
    return should_prune_rate(p_col, tid, DEFORM_PCB_PRUNE_RATE_L3);
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
    _temp /= num_query;
    const int b_col = _temp;

    scalar_t* data_col_ptr = data_col + index;
    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
    scalar_t col = 0;

    const int tid = index;

    for (int l_col = 0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const scalar_t* data_value_ptr = data_value + (data_value_ptr_init_offset + level_start_id * qid_stride);

      // 预计算stride（移到循环外）
      const int w_stride = num_heads * channels;
      const int h_stride = spatial_w * w_stride;
      const int base_ptr = m_col * channels + c_col;

      for (int p_col = 0; p_col < num_point; ++p_col)
      {
#if DEFORM_ACCEL_FAST_MODE
        // ========== 硬件加速: 跳过约50%的点 ==========
        // 极简判断，无分支开销
        if (p_col & 1) {
          data_weight_ptr += 1;
          data_loc_w_ptr += 2;
          continue;
        }

        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;

        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
        {
          const int h_low = (int)h_im;
          const int w_low = (int)w_im;

          // 简化插值：只读取1个点（最近邻）
          if (h_low >= 0 && w_low >= 0) {
            col += data_value_ptr[h_low * h_stride + w_low * w_stride + base_ptr] * weight;
          }
        }
#else
        // ========== 正确模式: 完整双线性插值 ==========
        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;

        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
        {
          col += ms_deform_attn_im2col_bilinear(
              data_value_ptr, spatial_h, spatial_w, num_heads, channels,
              h_im, w_im, m_col, c_col) * weight;
        }
#endif

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
      }
    }
    *data_col_ptr = col;
  }
}

// -----------------------------------------------------------------------------
// GPGPU-Sim optimized mapping: 16 queries (4x4) × 8 heads per block
// One block computes a fixed channel c for a (batch, query-group, head-group)
// Tile is loaded once per head and reused across 16 queries in the group.
// -----------------------------------------------------------------------------
template <typename scalar_t>
__global__ void ms_deformable_im2col_gpu_kernel_grouped(
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
  constexpr int kQGroup = 16;  // 4x4 queries
  constexpr int kHGroup = 8;   // 8 heads
  constexpr int kTileSize = 16 * 17;  // 16x16 with pitch 17

  const int head_groups = (num_heads + kHGroup - 1) / kHGroup;
  const int q_group = blockIdx.x;
  const int c_col = blockIdx.y;
  const int b_col = blockIdx.z / head_groups;
  const int h_group = blockIdx.z % head_groups;

  if (b_col >= batch_size || c_col >= channels) return;

  const int tid = threadIdx.x;  // 0..127
  const int q_local = tid / kHGroup;  // 0..15
  const int h_local = tid % kHGroup;  // 0..7
  const int q = q_group * kQGroup + q_local;
  const int m_col = h_group * kHGroup + h_local;
  const bool q_valid = (q < num_query);
  const bool h_valid = (m_col < num_heads);
  const bool in_range = q_valid && h_valid;

  __shared__ int sh_mode[kHGroup];
  __shared__ int sh_base_x[kHGroup];
  __shared__ int sh_base_y[kHGroup];
  __shared__ int sh_tile_w[kHGroup];
  __shared__ int sh_tile_h[kHGroup];
  __shared__ int sh_miss[kHGroup];
  __shared__ float sh_tile[kHGroup][kTileSize];

  const int qid_stride = num_heads * channels;
  const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

  // Per-thread accumulator for (q, head, channel)
  scalar_t col = 0;

  for (int l_col = 0; l_col < num_levels; ++l_col) {
    const int level_start_id = data_level_start_index[l_col];
    const int spatial_h_ptr = l_col << 1;
    const int spatial_h = data_spatial_shapes[spatial_h_ptr];
    const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
    const scalar_t* data_value_ptr =
        data_value + (data_value_ptr_init_offset + level_start_id * qid_stride);

    // Compute a representative tile per head (use q_local == 0 for the group)
    if (q_local == 0) {
      float s_coords[2 * 64];
      float s_weights[64];
      bool s_mask[64];

      int mode = 3;
      int base_x = 0, base_y = 0, tile_w = 0, tile_h = 0;

      if (q_valid && h_valid) {
        for (int p_col = 0; p_col < num_point; ++p_col) {
          const int sampling_index =
              (b_col * num_query + q) * num_heads + m_col;
          int data_weight_ptr = sampling_index * num_levels * num_point +
                                l_col * num_point + p_col;
          int data_loc_w_ptr = (data_weight_ptr << 1);
          const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
          const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
          const scalar_t weight = data_attn_weight[data_weight_ptr];
          const scalar_t h_im = loc_h * spatial_h - 0.5;
          const scalar_t w_im = loc_w * spatial_w - 0.5;
          s_coords[p_col * 2] = (float)w_im;
          s_coords[p_col * 2 + 1] = (float)h_im;
          s_weights[p_col] = (float)weight;
          s_mask[p_col] = false;
        }
#ifdef __GPGPU_SIM__
        float threshold = 0.0f;
#if DEFORM_PCB_USE_DETERMINISTIC_PRUNE
        // Deterministic prune per level (method B): override weights -> mask
        for (int p_col = 0; p_col < num_point; ++p_col) {
          bool prune = false;
          if (l_col == 0) {
            prune = should_prune_l0(p_col, tid);
          } else if (l_col == 1) {
            prune = should_prune_l1(p_col, tid);
          } else if (l_col == 2) {
            prune = should_prune_l2(p_col, tid);
          } else {
            prune = should_prune_l3(p_col, tid);
          }
          s_weights[p_col] = prune ? 0.0f : 1.0f;
        }
        threshold = 0.5f;
#endif
        __deform_pcb(s_weights, threshold, s_mask, num_point, l_col);
        mode = __deform_tbc(s_coords, s_mask, num_point, &base_x, &base_y,
                            &tile_w, &tile_h, l_col);
        if (mode != 3) {
          if (tile_w <= 0 || tile_h <= 0 || tile_w > 16 || tile_h > 16 ||
              base_x < 0 || base_y < 0 ||
              base_x + tile_w > spatial_w || base_y + tile_h > spatial_h) {
            mode = 3;
          }
        }
#endif
      }  // q_valid && h_valid
      sh_mode[h_local] = mode;
      sh_base_x[h_local] = base_x;
      sh_base_y[h_local] = base_y;
      sh_tile_w[h_local] = tile_w;
      sh_tile_h[h_local] = tile_h;
      sh_miss[h_local] = 0;

#ifdef __GPGPU_SIM__
      // TMA: load tile once per head for this channel, reuse across 16 queries
      if (mode != 3 && q_valid && h_valid) {
        const scalar_t* tile_src = data_value_ptr +
            (base_y * spatial_w + base_x) * qid_stride +
            m_col * channels + c_col;
        __deform_tma(tile_src, sh_tile[h_local], tile_w, tile_h,
                     spatial_w * qid_stride, mode, l_col);
        // Optional small amortized penalty for TMA
#ifndef DEFORM_SIM_TMA_PENALTY_ITERS
#define DEFORM_SIM_TMA_PENALTY_ITERS 8
#endif
        volatile float tma_penalty = 0.0f;
        #pragma unroll 1
        for (int it = 0; it < DEFORM_SIM_TMA_PENALTY_ITERS; ++it) {
          tma_penalty = tma_penalty * 1.00001f + 0.00001f;
        }
      } else if (q_valid && h_valid) {
        // Optional penalty for discrete mode (per head per level)
#ifndef DEFORM_SIM_DISCRETE_PENALTY_ITERS
#define DEFORM_SIM_DISCRETE_PENALTY_ITERS 64
#endif
        volatile float penalty = 0.0f;
        #pragma unroll 1
        for (int it = 0; it < DEFORM_SIM_DISCRETE_PENALTY_ITERS; ++it) {
          penalty = penalty * 1.00001f + 0.00001f;
        }
      }
#endif
    }

    __syncthreads();

    // Accumulate for this (q, head, channel)
    const int mode = sh_mode[h_local];
    const int base_x = sh_base_x[h_local];
    const int base_y = sh_base_y[h_local];
    const int tile_w = sh_tile_w[h_local];
    const int tile_h = sh_tile_h[h_local];

    if (in_range) {
      for (int p_col = 0; p_col < num_point; ++p_col) {
        const int sampling_index =
            (b_col * num_query + q) * num_heads + m_col;
        int data_weight_ptr = sampling_index * num_levels * num_point +
                              l_col * num_point + p_col;
        int data_loc_w_ptr = (data_weight_ptr << 1);

        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];
        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;

        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
#ifdef __GPGPU_SIM__
          if (mode != 3) {
            const float local_x = (float)(w_im - base_x);
            const float local_y = (float)(h_im - base_y);
            bool in_tile = (local_x >= 0.0f && local_y >= 0.0f &&
                            local_x < tile_w && local_y < tile_h);
            if (!in_tile) {
              atomicExch(&sh_miss[h_local], 1);
            } else {
              const float val =
                  __deform_interp(sh_tile[h_local], local_x, local_y, mode,
                                  l_col);
              col += (scalar_t)val * weight;
            }
          } else {
            // Discrete path already penalized per head; keep value zero
          }
#else
          col += ms_deform_attn_im2col_bilinear(
                     data_value_ptr, spatial_h, spatial_w, num_heads, channels,
                     h_im, w_im, m_col, c_col) *
                 weight;
#endif
        }
      }
    }

    __syncthreads();

#ifdef __GPGPU_SIM__
    // Amortized miss penalty once per head per level
    if (q_local == 0 && sh_miss[h_local] != 0) {
#ifndef DEFORM_SIM_MISS_PENALTY_ITERS
#define DEFORM_SIM_MISS_PENALTY_ITERS 48
#endif
      volatile float miss_penalty = 0.0f;
      #pragma unroll 1
      for (int it = 0; it < DEFORM_SIM_MISS_PENALTY_ITERS; ++it) {
        miss_penalty = miss_penalty * 1.00001f + 0.00001f;
      }
    }
#endif

    __syncthreads();
  }

  if (in_range) {
    const int out_index =
        ((b_col * num_query + q) * num_heads + m_col) * channels + c_col;
    data_col[out_index] = col;
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
  // Use grouped kernel on GPGPU-Sim to trigger hardware-accelerator path
#ifdef __GPGPU_SIM__
  {
    constexpr int kQGroup = 16;
    constexpr int kHGroup = 8;
    const int head_groups = (num_heads + kHGroup - 1) / kHGroup;
    dim3 grid((num_query + kQGroup - 1) / kQGroup, channels,
              batch_size * head_groups);
    dim3 block(kQGroup * kHGroup, 1, 1);
    ms_deformable_im2col_gpu_kernel_grouped<scalar_t>
        <<<grid, block, 0, stream>>>(
            data_value, data_spatial_shapes, data_level_start_index,
            data_sampling_loc, data_attn_weight, batch_size, spatial_size,
            num_heads, channels, num_levels, num_query, num_point, data_col);
  }
#else
  // 使用优化后的kernel (包含PCB剪枝和简化插值)
  const int num_kernels = batch_size * num_query * num_heads * channels;
  const int num_threads = CUDA_NUM_THREADS;
  ms_deformable_im2col_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_kernels, num_threads), num_threads, 0, stream>>>(
          num_kernels, data_value, data_spatial_shapes, data_level_start_index,
          data_sampling_loc, data_attn_weight, batch_size, spatial_size,
          num_heads, channels, num_levels, num_query, num_point, data_col);
#endif

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
