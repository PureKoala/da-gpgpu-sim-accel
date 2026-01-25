/*!
**************************************************************************************************
* Deformable Attention Hardware Accelerator Stubs
* These functions will be intercepted by GPGPU-Sim and routed to hardware accelerator models
* Copyright (c) 2026
**************************************************************************************************
*/

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

/****************************************************************************************
 * Simulator-only knobs
 ****************************************************************************************/
#ifdef __GPGPU_SIM__
#ifndef DEFORM_SIM_DUMMY_ITERS
#define DEFORM_SIM_DUMMY_ITERS 0
#endif
#endif

/****************************************************************************************
 * Hardware Accelerator Function Declarations
 * These are stub functions that will be intercepted by GPGPU-Sim
 ****************************************************************************************/

/**
 * PCB (Pre-Check Block): 权重预筛选
 * 
 * 功能: 基于阈值过滤低权重采样点
 * GPGPU-Sim延迟: 1 cycle (固定)
 * 
 * @param weights      输入权重数组 [num_points]
 * @param threshold    权重阈值
 * @param mask         输出mask数组 [num_points], true表示低权重需要跳过
 * @param num_points   采样点数量
 * @param level        特征层级索引（用于延迟建模）
 */
__device__ __noinline__ void __deform_pcb(
    const float* weights,
    float threshold,
    bool* mask,
    int num_points,
    int level
);

/**
 * TBC (Tile Boundary Check): Tile边界检查
 * 
 * 功能: 分析采样点的空间聚集性，决定访存模式
 * GPGPU-Sim延迟: 4-7 cycles
 * 
 * @param coords       采样坐标数组 [num_points][2] (x, y)
 * @param mask         有效点mask [num_points]
 * @param num_points   采样点数量
 * @param base_x       输出: Tile基准x坐标
 * @param base_y       输出: Tile基准y坐标
 * @param tile_w       输出: Tile宽度
 * @param tile_h       输出: Tile高度
 * @param level        特征层级索引（用于延迟建模）
 * @return int         访存模式: 0=Horizontal, 1=Vertical, 2=XOR, 3=Discrete
 */
__device__ __noinline__ int __deform_tbc(
    const float* coords,
    const bool* mask,
    int num_points,
    int* base_x,
    int* base_y,
    int* tile_w,
    int* tile_h,
    int level
);

/**
 * TMA (Tile Memory Access): Tile内存访问
 * 
 * 功能: 从全局内存批量加载16×16 tile到共享内存
 * GPGPU-Sim延迟: 22 cycles (16c TMA + 6c Storage)
 * 
 * @param src          源地址 (value_map全局内存)
 * @param dst          目标地址 (tile_buffer共享内存)
 * @param tile_w       Tile宽度
 * @param tile_h       Tile高度
 * @param src_pitch    源数据行跨度
 * @param mode         访存模式 (从TBC返回)
 * @param level        特征层级索引（用于延迟建模）
 */
__device__ __noinline__ void __deform_tma(
    const float* src,
    float* dst,
    int tile_w,
    int tile_h,
    int src_pitch,
    int mode,
    int level
);

/**
 * Interpolation: 硬件加速双线性插值
 * 
 * 功能: 从tile_buffer中进行双线性插值
 * GPGPU-Sim延迟: 3 cycles (固定)
 * 
 * @param tile_buffer  Tile数据缓冲区 (共享内存)
 * @param local_x      局部x坐标 (相对于tile)
 * @param local_y      局部y坐标 (相对于tile)
 * @param mode         访存模式 (决定swizzle方式)
 * @param level        特征层级索引（用于延迟建模）
 * @return float       插值结果
 */
__device__ __noinline__ float __deform_interp(
    const float* tile_buffer,
    float local_x,
    float local_y,
    int mode,
    int level
);

/****************************************************************************************
 * 单点版本接口 (用于逐点处理的kernel)
 * 这些函数会被GPGPU-Sim拦截，模拟硬件单元的延迟
 ****************************************************************************************/

/**
 * PCB单点检查: 判断单个采样点是否应该被剪枝
 * GPGPU-Sim延迟: 由模拟器控制
 *
 * @param weight    采样点权重
 * @param point_idx 采样点索引（用于确定性剪枝模式）
 * @return bool     true=被剪枝（跳过），false=保留
 */
__device__ __forceinline__ bool __deform_pcb_check(float weight, int point_idx) {
    // 这个函数会被GPGPU-Sim拦截
    // 实现：50%的点被剪枝（基于索引的确定性模式）
    return (point_idx & 1) != 0;
}

/**
 * 单点插值: 硬件加速的单点采样
 * GPGPU-Sim延迟: 由模拟器控制
 */
__device__ __forceinline__ float __deform_sample_point(
    const float* data_ptr,
    int h_low, int w_low,
    int h_stride, int w_stride,
    int base_ptr
) {
    // 这个函数会被GPGPU-Sim拦截
    return data_ptr[h_low * h_stride + w_low * w_stride + base_ptr];
}

/****************************************************************************************
 * Implementation Stubs (Empty for real hardware, intercepted by GPGPU-Sim)
 ****************************************************************************************/

__device__ __noinline__ void __deform_pcb(
    const float* weights,
    float threshold,
    bool* mask,
    int num_points,
    int level
) {
    // 这个函数会被GPGPU-Sim拦截，函数体内容不会被执行
    // 为防止编译器优化掉参数，使用简单的内联汇编
    #ifdef __GPGPU_SIM__
    // 模拟器路径：GPGPU-Sim会在CALL指令时拦截并执行deform_pcb_impl()
    // 函数体留空，拦截器会处理所有逻辑
    asm volatile ("" : : "l"(weights), "f"(threshold), "l"(mask), "r"(num_points), "r"(level) : "memory");
    #else
    // 真实硬件fallback（不应该在模拟器中执行到这里）
    if (threadIdx.x == 0) {
        for (int i = 0; i < num_points; i++) {
            mask[i] = (fabsf(weights[i]) < threshold);
        }
    }
    #endif
}

__device__ __noinline__ int __deform_tbc(
    const float* coords,
    const bool* mask,
    int num_points,
    int* base_x,
    int* base_y,
    int* tile_w,
    int* tile_h,
    int level
) {
    // 这个函数会被GPGPU-Sim拦截，函数体内容不会被执行
    #ifdef __GPGPU_SIM__
    // 模拟器路径：GPGPU-Sim会在CALL指令时拦截并执行deform_tbc_impl()
    asm volatile ("" : : "l"(coords), "l"(mask), "r"(num_points), "l"(base_x), "l"(base_y), "l"(tile_w), "l"(tile_h), "r"(level) : "memory");
    return 0; // 返回值会被拦截器覆盖
    #else
    // 真实硬件fallback
    float min_x = 1e9f, max_x = -1e9f;
    float min_y = 1e9f, max_y = -1e9f;
    for (int i = 0; i < num_points; i++) {
        if (!mask[i]) {
            float x = coords[i * 2];
            float y = coords[i * 2 + 1];
            min_x = fminf(min_x, x);
            max_x = fmaxf(max_x, x);
            min_y = fminf(min_y, y);
            max_y = fmaxf(max_y, y);
        }
    }
    *base_x = (int)min_x;
    *base_y = (int)min_y;
    *tile_w = (int)(max_x - min_x) + 1;
    *tile_h = (int)(max_y - min_y) + 1;
    if (*tile_w > 16 || *tile_h > 16) {
        return 3;
    }
    return 0;
    #endif
}

__device__ __noinline__ void __deform_tma(
    const float* src,
    float* dst,
    int tile_w,
    int tile_h,
    int src_pitch,
    int mode,
    int level
) {
    // 这个函数会被GPGPU-Sim拦截，函数体内容不会被执行
    #ifdef __GPGPU_SIM__
    // 模拟器路径：GPGPU-Sim会在CALL指令时拦截并执行deform_tma_impl()
    asm volatile ("" : : "l"(src), "l"(dst), "r"(tile_w), "r"(tile_h), "r"(src_pitch), "r"(mode), "r"(level) : "memory");
    #else
    // 真实硬件fallback
    for (int ty = 0; ty < tile_h; ty++) {
        for (int tx = 0; tx < tile_w; tx++) {
            dst[ty * 17 + tx] = src[ty * src_pitch + tx];
        }
    }
    #endif
}

__device__ __noinline__ float __deform_interp(
    const float* tile_buffer,
    float local_x,
    float local_y,
    int mode,
    int level
) {
    // 这个函数会被GPGPU-Sim拦截，函数体内容不会被执行
    #ifdef __GPGPU_SIM__
    // 模拟器路径：GPGPU-Sim会在CALL指令时拦截并执行deform_interp_impl()
    asm volatile ("" : : "l"(tile_buffer), "f"(local_x), "f"(local_y), "r"(mode), "r"(level) : "memory");
    return 0.0f; // 返回值会被拦截器覆盖
    #else
    // 真实硬件fallback
    int x0 = (int)floorf(local_x);
    int y0 = (int)floorf(local_y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float fx = local_x - x0;
    float fy = local_y - y0;
    float v00 = (x0 >= 0 && y0 >= 0) ? tile_buffer[y0 * 17 + x0] : 0.0f;
    float v01 = (x1 < 16 && y0 >= 0) ? tile_buffer[y0 * 17 + x1] : 0.0f;
    float v10 = (x0 >= 0 && y1 < 16) ? tile_buffer[y1 * 17 + x0] : 0.0f;
    float v11 = (x1 < 16 && y1 < 16) ? tile_buffer[y1 * 17 + x1] : 0.0f;
    return (1-fx)*(1-fy)*v00 + fx*(1-fy)*v01 + (1-fx)*fy*v10 + fx*fy*v11;
    #endif
}
