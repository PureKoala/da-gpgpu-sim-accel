#include "../include/deformable_attention.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * 测试程序 - DeformableAttention CUDA实现
 */

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 初始化随机数据
void init_random_data(float* data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

int main() {
    printf("=== DeformableAttention CUDA Test ===\n\n");

    // 通过修改这里的宏定义来切换测试配置 (BLOCK1, BLOCK2, BLOCK3, BLOCK4)
#define BLOCK1

#ifdef BLOCK1
    // --- Block 1 参数 ---
    int batch_size = 1;
    int H = 56, W = 56;
    int embed_dim = 128;
    int num_heads = 4;
    int offset_groups = 1;
    int grid_size = 7;
#elif defined(BLOCK2)
    // --- Block 2 参数 ---
    int batch_size = 1;
    int H = 28, W = 28;
    int embed_dim = 256;
    int num_heads = 8;
    int offset_groups = 1;
    int grid_size = 7;
#elif defined(BLOCK3)
    // --- Block 3 参数 ---
    int batch_size = 1;
    int H = 14, W = 14;
    int embed_dim = 512;
    int num_heads = 16;
    int offset_groups = 1;
    int grid_size = 7;
#elif defined(BLOCK4)
    // --- Block 4 参数 ---
    int batch_size = 1;
    int H = 7, W = 7;
    int embed_dim = 1024;
    int num_heads = 32;
    int offset_groups = 1;
    int grid_size = 7;
#endif

    // 基于配置计算派生参数
    int num_tokens = H * W;
    int head_dim = embed_dim / num_heads;
    int num_sampling_points = grid_size * grid_size;
    
    printf("Configuration:\n");
    printf("  Batch Size: %d\n", batch_size);
    printf("  Input Size: %dx%d (%d tokens)\n", H, W, num_tokens);
    printf("  Embed Dim: %d\n", embed_dim);
    printf("  Num Heads: %d\n", num_heads);
    printf("  Head Dim: %d\n", head_dim);
    printf("  Grid Size: %dx%d (%d points)\n", grid_size, grid_size, num_sampling_points);
    printf("\n");
    
    // 计算数据大小
    size_t input_size = batch_size * num_tokens * embed_dim;
    size_t weight_size = embed_dim * embed_dim;
    size_t bias_size = embed_dim;
    size_t rpb_size = num_heads * (2 * grid_size - 1) * (2 * grid_size - 1);
    
    // 分配主机内存
    float *h_input = (float*)malloc(input_size * sizeof(float));
    float *h_output = (float*)malloc(input_size * sizeof(float));
    float *h_q_weight = (float*)malloc(weight_size * sizeof(float));
    float *h_k_weight = (float*)malloc(weight_size * sizeof(float));
    float *h_v_weight = (float*)malloc(weight_size * sizeof(float));
    float *h_out_weight = (float*)malloc(weight_size * sizeof(float));
    float *h_bias = (float*)malloc(bias_size * sizeof(float));
    float *h_rpb_table = (float*)malloc(rpb_size * sizeof(float));
    
    // 初始化随机数据
    srand(time(NULL));
    printf("Initializing random data...\n");
    init_random_data(h_input, input_size);
    init_random_data(h_q_weight, weight_size);
    init_random_data(h_k_weight, weight_size);
    init_random_data(h_v_weight, weight_size);
    init_random_data(h_out_weight, weight_size);
    init_random_data(h_bias, bias_size);
    init_random_data(h_rpb_table, rpb_size);
    
    // 分配设备内存
    float *d_input, *d_output;
    float *d_q_weight, *d_k_weight, *d_v_weight, *d_out_weight;
    float *d_bias, *d_rpb_table;
    float *d_workspace;
    
    CHECK_CUDA(cudaMalloc(&d_input, input_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, input_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_q_weight, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_k_weight, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_v_weight, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out_weight, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_bias, bias_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_rpb_table, rpb_size * sizeof(float)));
    
    // 分配workspace
    size_t workspace_size = getDeformableAttentionWorkspaceSize(
        batch_size, num_tokens, embed_dim, num_heads, num_sampling_points
    );
    CHECK_CUDA(cudaMalloc(&d_workspace, workspace_size));
    
    printf("Memory allocation:\n");
    printf("  Input/Output: %.2f MB\n", input_size * sizeof(float) / 1024.0 / 1024.0);
    printf("  Weights: %.2f MB\n", weight_size * sizeof(float) * 4 / 1024.0 / 1024.0);
    printf("  Workspace: %.2f MB\n", workspace_size / 1024.0 / 1024.0);
    printf("  Total: %.2f MB\n", 
           (input_size * 2 + weight_size * 4 + workspace_size / sizeof(float)) * sizeof(float) / 1024.0 / 1024.0);
    printf("\n");
    
    // 拷贝数据到设备
    printf("Copying data to device...\n");
    CHECK_CUDA(cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_q_weight, h_q_weight, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_k_weight, h_k_weight, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_v_weight, h_v_weight, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_out_weight, h_out_weight, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias, h_bias, bias_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rpb_table, h_rpb_table, rpb_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // 创建参数结构体
    DeformableAttnParams params;
    params.input = d_input;
    params.output = d_output;
    params.q_weight = d_q_weight;
    params.k_weight = d_k_weight;
    params.v_weight = d_v_weight;
    params.out_weight = d_out_weight;
    params.q_bias = d_bias;
    params.k_bias = d_bias;
    params.v_bias = d_bias;
    params.out_bias = d_bias;
    params.rpb_table = d_rpb_table;
    params.batch_size = batch_size;
    params.num_tokens = num_tokens;
    params.embed_dim = embed_dim;
    params.num_heads = num_heads;
    params.head_dim = head_dim;
    params.offset_groups = offset_groups;
    params.grid_size = grid_size;
    params.num_sampling_points = num_sampling_points;
    params.H = H;
    params.W = W;
    params.workspace = d_workspace;
    params.workspace_size = workspace_size;
    
    // 预热
    printf("Warming up...\n");
    for (int i = 0; i < 3; i++) {
        CHECK_CUDA(launchDeformableAttention(params));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 性能测试
    printf("\nPerformance test (100 iterations)...\n");
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        CHECK_CUDA(launchDeformableAttention(params));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    
    float avg_time = milliseconds / 100.0f;
    printf("Average time: %.3f ms\n", avg_time);
    printf("Throughput: %.2f GFLOPS\n", 
           (2.0 * batch_size * num_tokens * embed_dim * num_sampling_points / 1e9) / (avg_time / 1000.0));
    
    // 拷贝结果回主机
    CHECK_CUDA(cudaMemcpy(h_output, d_output, input_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 简单验证
    printf("\nOutput sample (first 10 values):\n");
    for (int i = 0; i < 10; i++) {
        printf("  %.6f\n", h_output[i]);
    }
    
    // 清理
    printf("\nCleaning up...\n");
    free(h_input);
    free(h_output);
    free(h_q_weight);
    free(h_k_weight);
    free(h_v_weight);
    free(h_out_weight);
    free(h_bias);
    free(h_rpb_table);
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_q_weight);
    cudaFree(d_k_weight);
    cudaFree(d_v_weight);
    cudaFree(d_out_weight);
    cudaFree(d_bias);
    cudaFree(d_rpb_table);
    cudaFree(d_workspace);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("\n=== Test completed successfully! ===\n");
    
    return 0;
}
