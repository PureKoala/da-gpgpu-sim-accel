#ifndef DEFORMABLE_ATTENTION_H
#define DEFORMABLE_ATTENTION_H

#include <cuda_runtime.h>
#include <stdint.h>

/**
 * DeformableAttention 算子 - CUDA实现
 * 
 * 针对第一个Block优化:
 * - 输入维度: 56x56 (3136 tokens)
 * - embed_dim: 128
 * - num_heads: 4
 * - grid_size: 7x7 (49 sampling points)
 * - offset_groups: 1
 * 
 * 架构: RTX3070 (Ampere sm_86)
 * 优化: 使用Tensor Core (INT8)
 */

// 参数结构体
struct DeformableAttnParams {
    // 输入输出
    const float* input;          // (B, C, H, W) = (B, 128, 56, 56)
    float* output;               // (B, N, C) = (B, 3136, 128)
    
    // 权重矩阵
    const float* q_weight;       // (C, C, 1, 1) = (128, 128, 1, 1)
    const float* k_weight;       // (C, C, 1, 1)
    const float* v_weight;       // (C, C, 1, 1)
    const float* out_weight;     // (C, C, 1, 1)
    
    const float* q_bias;         // (C,) = (128,)
    const float* k_bias;         // (C,)
    const float* v_bias;         // (C,)
    const float* out_bias;       // (C,)
    
    // Offset生成网络权重
    const float* offset_dw_weight;      // (C, 1, 3, 3) depthwise conv
    const float* offset_conv_weight;    // (2*G, C, 1, 1) = (2, 128, 1, 1)
    const float* offset_norm_weight;    // (C,) = (128,)
    const float* offset_norm_bias;      // (C,)
    
    // RPB (相对位置偏置)
    const float* rpb_table;      // (M, 2*grid_size-1, 2*grid_size-1) = (4, 13, 13)
    
    // Reference points
    const float* reference_points;  // (1, 49, G, 2)
    
    // 维度参数
    int batch_size;              // B
    int num_tokens;              // N = H * W = 3136
    int embed_dim;               // C = 128
    int num_heads;               // M = 4
    int head_dim;                // D_h = C / M = 32
    int offset_groups;           // G = 1
    int grid_size;               // 7
    int num_sampling_points;     // 49
    int H, W;                    // 56, 56
    
    // 临时缓冲区 (workspace)
    float* workspace;            // 用于中间结果
    size_t workspace_size;
};

// 主函数接口
cudaError_t launchDeformableAttention(
    const DeformableAttnParams& params,
    cudaStream_t stream = 0
);

// 计算所需workspace大小
size_t getDeformableAttentionWorkspaceSize(
    int batch_size,
    int num_tokens,
    int embed_dim,
    int num_heads,
    int num_sampling_points
);

#endif // DEFORMABLE_ATTENTION_H
