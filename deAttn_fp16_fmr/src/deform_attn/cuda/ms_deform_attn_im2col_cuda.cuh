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
#include <climits>

// 使用宏来禁用编译器的内联和优化
#define PREVENT_INLINE __attribute__((noinline))

// FMR Tile Loader stub implementation (for GPGPU-Sim interception)
// This is a dummy function that allows compilation and enables GPGPU-Sim to intercept the call
// CRITICAL: Must have exactly 5 parameters to match the GPGPU-Sim intercept handler
extern "C" __device__ PREVENT_INLINE
void __fmr_sample(
    unsigned long long gmem_base_addr, 
    unsigned long long smem_base_addr, 
    int width, 
    int height,
    int stride)  // Source image stride (elements per row) - MUST NOT BE OPTIMIZED AWAY
{
    // ====================================================================
    // Critical anti-optimization inline assembly (FMR Load Proxy)
    // Ensures GPGPU-Sim can intercept all FIVE parameters
    // ====================================================================
    asm volatile(
        "// FMR Load Proxy: Forcing all parameters to be 'live' for GPGPU-Sim."
        :   // Output list (empty)
        :   // Input list
          "l"(gmem_base_addr), 
          "l"(smem_base_addr), 
          "r"(width), 
          "r"(height),
          "r"(stride)
        : "memory" // Clobber list
    );
}

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
 * FMR Optimization: Query-Level Tile Preloading
 ****************************************************************************************/

// Analyze sampling points for a query at a specific level to determine bounding box
// 修复: 按level分析采样点，而不是所有level一起分析
template <typename scalar_t>
__device__ void analyze_query_sampling_points_per_level(
    const scalar_t* __restrict__ data_sampling_loc,
    const int64_t* __restrict__ data_spatial_shapes,
    int query_id,
    int batch_id,
    int level_id,  // 新增: 指定要分析的level
    int num_query,
    int num_heads,
    int num_levels,
    int num_point,
    int* min_h, int* max_h, int* min_w, int* max_w,
    bool* use_fmr  // Output: whether to use FMR optimization
) {
    *min_h = INT_MAX; *max_h = INT_MIN;
    *min_w = INT_MAX; *max_w = INT_MIN;
    
    // 修复Bug 7: 正确计算采样位置的索引
    // 内存布局: [batch, query, head, level, point, 2]
    // 对于query_id和level_id，需要遍历所有head和point
    // 索引计算: base + head * (num_levels * num_point * 2) + level * (num_point * 2) + point * 2
    const int base_idx = batch_id * num_query * num_heads * num_levels * num_point * 2 +
                         query_id * num_heads * num_levels * num_point * 2;
    
    const int spatial_h = data_spatial_shapes[level_id * 2];
    const int spatial_w = data_spatial_shapes[level_id * 2 + 1];
    
    // 遍历所有head和point，分析当前level的采样点
    for (int h = 0; h < num_heads; h++) {
        for (int p = 0; p < num_point; p++) {
            // 正确的索引计算：base + head偏移 + level偏移 + point偏移
            int idx = base_idx + 
                      h * num_levels * num_point * 2 +  // head偏移
                      level_id * num_point * 2 +        // level偏移
                      p * 2;                            // point偏移
            scalar_t loc_w = data_sampling_loc[idx];
            scalar_t loc_h = data_sampling_loc[idx + 1];
            
            // Convert to image coordinates
            scalar_t h_im = loc_h * spatial_h - 0.5;
            scalar_t w_im = loc_w * spatial_w - 0.5;
            
            // Consider 2×2 bilinear interpolation region
            int h_low = floor(h_im);
            int w_low = floor(w_im);
            int h_high = h_low + 1;
            int w_high = w_low + 1;
            
            // Update bounding box (clamp to valid range)
            if (h_low >= 0 && h_low < spatial_h) *min_h = min(*min_h, h_low);
            if (h_high >= 0 && h_high < spatial_h) *max_h = max(*max_h, h_high);
            if (w_low >= 0 && w_low < spatial_w) *min_w = min(*min_w, w_low);
            if (w_high >= 0 && w_high < spatial_w) *max_w = max(*max_w, w_high);
        }
    }
    
    // Clamp bounding box to valid range
    if (*min_h == INT_MAX) *min_h = 0;
    if (*max_h == INT_MIN) *max_h = 0;
    if (*min_w == INT_MAX) *min_w = 0;
    if (*max_w == INT_MIN) *max_w = 0;
    
    // Determine if FMR optimization is beneficial
    // Use FMR if tile size is reasonable (e.g., <= 64×64) and >= 4×4
    int tile_h = *max_h - *min_h + 1;
    int tile_w = *max_w - *min_w + 1;
    
    *use_fmr = (tile_h > 0 && tile_w > 0 && 
                tile_h <= 64 && tile_w <= 64 &&
                tile_h * tile_w >= 16);  // At least 4×4 tile
}

// Bilinear interpolation from SMEM tile (loaded by FMR)
template <typename scalar_t>
__device__ scalar_t ms_deform_attn_im2col_bilinear_from_smem(
    const scalar_t* __restrict__ smem_tile,  // SMEM tile loaded by FMR
    const int tile_h,                        // Tile height
    const int tile_w,                        // Tile width
    const int nheads,
    const int channels,
    const scalar_t h_rel,                   // Relative h coordinate in tile
    const scalar_t w_rel,                   // Relative w coordinate in tile
    const int m,                            // Head index
    const int c)                            // Channel index
{
    const int h_low = floor(h_rel);
    const int w_low = floor(w_rel);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;
    
    const scalar_t lh = h_rel - h_low;
    const scalar_t lw = w_rel - w_low;
    const scalar_t hh = 1 - lh;
    const scalar_t hw = 1 - lw;
    
    const int w_stride = nheads * channels;
    const int h_stride = tile_w * w_stride;
    const int base_ptr = m * channels + c;
    
    scalar_t v1 = 0, v2 = 0, v3 = 0, v4 = 0;
    
    // Read 4 points from SMEM
    // SMEM layout: [tile_h × tile_w × num_heads × channels]
    if (h_low >= 0 && w_low >= 0 && h_low < tile_h && w_low < tile_w) {
        const int ptr1 = h_low * h_stride + w_low * w_stride + base_ptr;
        v1 = smem_tile[ptr1];
    }
    
    if (h_low >= 0 && w_high < tile_w && w_high >= 0 && h_low < tile_h) {
        const int ptr2 = h_low * h_stride + w_high * w_stride + base_ptr;
        v2 = smem_tile[ptr2];
    }
    
    if (h_high < tile_h && w_low >= 0 && h_high >= 0 && w_low < tile_w) {
        const int ptr3 = h_high * h_stride + w_low * w_stride + base_ptr;
        v3 = smem_tile[ptr3];
    }
    
    if (h_high < tile_h && w_high < tile_w && h_high >= 0 && w_high >= 0) {
        const int ptr4 = h_high * h_stride + w_high * w_stride + base_ptr;
        v4 = smem_tile[ptr4];
    }
    
    const scalar_t w1 = hh * hw;
    const scalar_t w2 = hh * lw;
    const scalar_t w3 = lh * hw;
    const scalar_t w4 = lh * lw;
    
    return (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
}

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
    
    for (int l_col = 0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const scalar_t* data_value_ptr = data_value + (data_value_ptr_init_offset + level_start_id * qid_stride);
      
      for (int p_col = 0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;

        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
        {
          col += ms_deform_attn_im2col_bilinear(
              data_value_ptr, spatial_h, spatial_w, num_heads, channels, 
              h_im, w_im, m_col, c_col) * weight;
        }

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
      }
    }
    *data_col_ptr = col;
  }
}


// FMR-optimized version with runtime mode control
// fmr_mode: 0 = Disable FMR (GMEM only)
//           1 = Selective load (analyze then load beneficial tiles)
//           2 = Force load all levels
template <typename scalar_t>
__global__ void ms_deformable_im2col_gpu_kernel_fmr_optimized(
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
    scalar_t* __restrict__ data_col,
    const int fmr_mode)
{
    extern __shared__ char shared_memory[];
    scalar_t* smem_tile = (scalar_t*)shared_memory;
    
    const int MAX_TILE_ELEMENTS = 64 * 64;
    size_t tile_offset = (MAX_TILE_ELEMENTS * num_heads * channels * sizeof(scalar_t) + sizeof(int) - 1) / sizeof(int) * sizeof(int);
    
    struct TileInfo {
        int min_h, max_h, min_w, max_w;
        int tile_h, tile_w;
        bool use_fmr;
    };
    TileInfo* per_level_info = (TileInfo*)(shared_memory + tile_offset);
    int* loaded_level_ptr = (int*)(shared_memory + tile_offset + num_levels * sizeof(TileInfo));
    
    const int tid = threadIdx.x;
    const int qid_stride = num_heads * channels;
    
    // Phase 1: Thread 0 analyzes and loads FMR tile
    int query_id = -1, batch_id = -1;
    
    if (tid == 0) {
        int first_index = blockIdx.x * blockDim.x;
        if (first_index < n) {
            int _temp = first_index;
            _temp /= channels;
            _temp /= num_heads;
            query_id = _temp % num_query;
            _temp /= num_query;
            batch_id = _temp;
            
            *loaded_level_ptr = -1;
            
            // Mode 0: Disable FMR completely
            if (fmr_mode == 0) {
                for (int l = 0; l < num_levels; l++) {
                    per_level_info[l].use_fmr = false;
                }
            }
            // Mode 1: Selective load (analyze and decide)
            else if (fmr_mode == 1) {
                for (int l = 0; l < num_levels; l++) {
                    int min_h, max_h, min_w, max_w;
                    bool use_fmr;
                    
                    analyze_query_sampling_points_per_level(
                        data_sampling_loc, data_spatial_shapes,
                        query_id, batch_id, l,
                        num_query, num_heads, num_levels, num_point,
                        &min_h, &max_h, &min_w, &max_w, &use_fmr
                    );
                    
                    per_level_info[l].min_h = min_h;
                    per_level_info[l].max_h = max_h;
                    per_level_info[l].min_w = min_w;
                    per_level_info[l].max_w = max_w;
                    per_level_info[l].tile_h = max_h - min_h + 1;
                    per_level_info[l].tile_w = max_w - min_w + 1;
                    per_level_info[l].use_fmr = use_fmr;
                    
                    if (use_fmr && *loaded_level_ptr == -1) {
                        const int level_start_id = data_level_start_index[l];
                        const int spatial_w = data_spatial_shapes[l * 2 + 1];
                        const scalar_t* level_value_ptr = data_value + 
                            (batch_id * spatial_size * qid_stride + level_start_id * qid_stride);
                        const int gmem_offset = min_h * spatial_w * qid_stride + min_w * qid_stride;
                        
                        volatile unsigned long long v_gmem = (unsigned long long)(level_value_ptr + gmem_offset);
                        volatile unsigned long long v_smem = (unsigned long long)smem_tile;
                        volatile int v_width = per_level_info[l].tile_w;
                        volatile int v_height = per_level_info[l].tile_h;
                        volatile int v_stride = spatial_w;
                        
                        __fmr_sample(v_gmem, v_smem, v_width, v_height, v_stride);
                        
                        *loaded_level_ptr = l;
                        
                        if (v_gmem == 0 && v_smem == 0 && v_width == 0 && v_height == 0 && v_stride == 0) {
                            *loaded_level_ptr = -1;
                        }
                    }
                }
            }
            // Mode 2: Force load all levels (no analysis, just compute bounding box)
            else if (fmr_mode == 2) {
                for (int l = 0; l < num_levels; l++) {
                    // Directly compute bounding box without analysis
                    int min_h = INT_MAX, max_h = INT_MIN;
                    int min_w = INT_MAX, max_w = INT_MIN;
                    
                    const int base_idx = batch_id * num_query * num_heads * num_levels * num_point * 2 +
                                         query_id * num_heads * num_levels * num_point * 2;
                    const int spatial_h = data_spatial_shapes[l * 2];
                    const int spatial_w = data_spatial_shapes[l * 2 + 1];
                    
                    // Calculate bounding box for all sampling points
                    for (int h = 0; h < num_heads; h++) {
                        for (int p = 0; p < num_point; p++) {
                            int idx = base_idx + h * num_levels * num_point * 2 + l * num_point * 2 + p * 2;
                            scalar_t loc_w = data_sampling_loc[idx];
                            scalar_t loc_h = data_sampling_loc[idx + 1];
                            
                            scalar_t h_im = loc_h * spatial_h - 0.5;
                            scalar_t w_im = loc_w * spatial_w - 0.5;
                            
                            int h_low = floor(h_im);
                            int w_low = floor(w_im);
                            int h_high = h_low + 1;
                            int w_high = w_low + 1;
                            
                            if (h_low >= 0 && h_low < spatial_h) min_h = min(min_h, h_low);
                            if (h_high >= 0 && h_high < spatial_h) max_h = max(max_h, h_high);
                            if (w_low >= 0 && w_low < spatial_w) min_w = min(min_w, w_low);
                            if (w_high >= 0 && w_high < spatial_w) max_w = max(max_w, w_high);
                        }
                    }
                    
                    // Clamp bounding box
                    if (min_h == INT_MAX) min_h = 0;
                    if (max_h == INT_MIN) max_h = 0;
                    if (min_w == INT_MAX) min_w = 0;
                    if (max_w == INT_MIN) max_w = 0;
                    
                    per_level_info[l].min_h = min_h;
                    per_level_info[l].max_h = max_h;
                    per_level_info[l].min_w = min_w;
                    per_level_info[l].max_w = max_w;
                    per_level_info[l].tile_h = max_h - min_h + 1;
                    per_level_info[l].tile_w = max_w - min_w + 1;
                    per_level_info[l].use_fmr = true;  // Force load
                    
                    // Load first valid level only (SMEM limitation)
                    if (*loaded_level_ptr == -1 && per_level_info[l].tile_h > 0 && per_level_info[l].tile_w > 0) {
                        const int level_start_id = data_level_start_index[l];
                        const int gmem_offset = min_h * spatial_w * qid_stride + min_w * qid_stride;
                        const scalar_t* level_value_ptr = data_value + 
                            (batch_id * spatial_size * qid_stride + level_start_id * qid_stride);
                        
                        volatile unsigned long long v_gmem = (unsigned long long)(level_value_ptr + gmem_offset);
                        volatile unsigned long long v_smem = (unsigned long long)smem_tile;
                        volatile int v_width = per_level_info[l].tile_w;
                        volatile int v_height = per_level_info[l].tile_h;
                        volatile int v_stride = spatial_w;
                        
                        __fmr_sample(v_gmem, v_smem, v_width, v_height, v_stride);
                        
                        *loaded_level_ptr = l;
                        
                        if (v_gmem == 0 && v_smem == 0 && v_width == 0 && v_height == 0 && v_stride == 0) {
                            *loaded_level_ptr = -1;
                        }
                    }
                }
            }
        }
    }
    
    __syncthreads();
    
    // Phase 2: Process output elements
    CUDA_KERNEL_LOOP(index, n) {
        int _temp = index;
        const int c_col = _temp % channels;
        _temp /= channels;
        const int sampling_index = _temp;
        const int m_col = _temp % num_heads;
        _temp /= num_heads;
        const int q_col = _temp % num_query;
        _temp /= num_query;
        const int b_col = _temp;
        
        int data_weight_ptr = sampling_index * num_levels * num_point;
        int data_loc_w_ptr = data_weight_ptr << 1;
        const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
        scalar_t col = 0;
        
        bool same_query = (q_col == query_id && b_col == batch_id);
        
        for (int l_col = 0; l_col < num_levels; ++l_col) {
            const int level_start_id = data_level_start_index[l_col];
            const int spatial_h = data_spatial_shapes[l_col << 1];
            const int spatial_w = data_spatial_shapes[(l_col << 1) + 1];
            const scalar_t* data_value_ptr = data_value + (data_value_ptr_init_offset + level_start_id * qid_stride);
            
            for (int p_col = 0; p_col < num_point; ++p_col) {
                const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
                const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
                const scalar_t weight = data_attn_weight[data_weight_ptr];
                
                const scalar_t h_im = loc_h * spatial_h - 0.5;
                const scalar_t w_im = loc_w * spatial_w - 0.5;
                
                if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
                    scalar_t interpolated;
                    
                    if (fmr_mode > 0 && same_query && *loaded_level_ptr == l_col) {
                        scalar_t h_rel = h_im - per_level_info[l_col].min_h;
                        scalar_t w_rel = w_im - per_level_info[l_col].min_w;
                        
                        interpolated = ms_deform_attn_im2col_bilinear_from_smem(
                            smem_tile, per_level_info[l_col].tile_h, per_level_info[l_col].tile_w,
                            num_heads, channels, h_rel, w_rel, m_col, c_col
                        );
                    } else {
                        interpolated = ms_deform_attn_im2col_bilinear(
                            data_value_ptr, spatial_h, spatial_w, num_heads, channels,
                            h_im, w_im, m_col, c_col
                        );
                    }
                    
                    col += interpolated * weight;
                }
                
                data_weight_ptr += 1;
                data_loc_w_ptr += 2;
            }
        }
        
        data_col[index] = col;
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

// FMR-optimized launcher with runtime mode control
// fmr_mode: 0 = Disable FMR (GMEM only)
//           1 = Selective load (analyze then load beneficial tiles) [DEFAULT]
//           2 = Force load all levels
template <typename scalar_t>
void ms_deformable_im2col_cuda_fmr_optimized(
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
    scalar_t* data_col,
    const int fmr_mode = 1)
{
    const int num_kernels = batch_size * num_query * num_heads * channels;
    const int num_threads = CUDA_NUM_THREADS;
    
    // 计算动态共享内存大小
    const int MAX_TILE_H = 64;
    const int MAX_TILE_W = 64;
    const int MAX_TILE_ELEMENTS = MAX_TILE_H * MAX_TILE_W;
    
    // 1. smem_tile: scalar_t[MAX_TILE_ELEMENTS * num_heads * channels]
    size_t smem_tile_size = MAX_TILE_ELEMENTS * num_heads * channels * sizeof(scalar_t);
    
    // 2. tile_info结构体数组
    struct TileInfo {
        int min_h;
        int max_h;
        int min_w;
        int max_w;
        int tile_h;
        int tile_w;
        bool use_fmr;
    };
    size_t tile_info_size = num_levels * sizeof(TileInfo);
    
    // 3. loaded_level指针
    size_t loaded_level_size = sizeof(int);
    
    // 对齐到int边界
    size_t tile_offset = (smem_tile_size + sizeof(int) - 1) / sizeof(int) * sizeof(int);
    
    // 总共享内存大小
    size_t total_shared_mem = tile_offset + tile_info_size + loaded_level_size;
    
    ms_deformable_im2col_gpu_kernel_fmr_optimized<scalar_t>
        <<<GET_BLOCKS(num_kernels, num_threads), num_threads, total_shared_mem, stream>>>(
        num_kernels, data_value, data_spatial_shapes, data_level_start_index,
        data_sampling_loc, data_attn_weight,
        batch_size, spatial_size, num_heads, channels, num_levels, num_query, num_point,
        data_col, fmr_mode);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in ms_deformable_im2col_cuda_fmr_optimized: %s\n", cudaGetErrorString(err));
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
