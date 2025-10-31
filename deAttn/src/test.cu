/*!
**************************************************************************************************
* Deformable Attention Test Program (Pure CUDA, No PyTorch)
* Modified for GPGPU-Sim compatibility with performance statistics
* Copyright (c) 2025
**************************************************************************************************
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Include our pure CUDA deformable attention implementation
#include "deform_attn/deform_attn_cuda.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA错误 %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err__)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/****************************************************************************************
 * 性能统计结构
 ****************************************************************************************/
struct PerformanceStats {
    float kernel_time_ms;
    float memory_transfer_time_ms;
    float total_time_ms;
    long long num_operations;
    double gflops;
    size_t memory_footprint_bytes;
    
    void print() const {
        printf("\n");
        printf("========================================\n");
        printf("         性能统计摘要\n");
        printf("========================================\n");
        printf("Kernel执行时间:    %.3f ms\n", kernel_time_ms);
        printf("内存传输时间:      %.3f ms\n", memory_transfer_time_ms);
        printf("总时间:            %.3f ms\n", total_time_ms);
        printf("计算操作数:        %lld\n", num_operations);
        printf("计算性能:          %.2f GFLOPS\n", gflops);
        printf("内存占用:          %.2f MB\n", memory_footprint_bytes / (1024.0 * 1024.0));
        printf("========================================\n");
    }
};

/****************************************************************************************
 * 辅助函数：初始化数据
 ****************************************************************************************/

// 初始化随机浮点数组
void init_random_float(float* arr, size_t n, float min_val, float max_val) {
    for (size_t i = 0; i < n; i++) {
        arr[i] = min_val + (max_val - min_val) * ((float)rand() / RAND_MAX);
    }
}

// 初始化int64数组
void init_int64(int64_t* arr, size_t n, int64_t val) {
    for (size_t i = 0; i < n; i++) {
        arr[i] = val;
    }
}

// 初始化空间形状
void init_spatial_shapes(int64_t* shapes, int num_levels, int base_h, int base_w) {
    for (int i = 0; i < num_levels; i++) {
        shapes[2 * i] = base_h >> i;      // height
        shapes[2 * i + 1] = base_w >> i;  // width
    }
}

// 计算level起始索引
void compute_level_start_index(int64_t* start_index, const int64_t* spatial_shapes, 
                                int num_levels, int num_heads, int channels) {
    start_index[0] = 0;
    for (int i = 1; i < num_levels; i++) {
        int64_t prev_h = spatial_shapes[2 * (i - 1)];
        int64_t prev_w = spatial_shapes[2 * (i - 1) + 1];
        start_index[i] = start_index[i - 1] + prev_h * prev_w;
    }
}

/****************************************************************************************
 * 结果验证函数
 ****************************************************************************************/

void verify_output(const float* output, int batch, int num_query, int num_heads, 
                   int channels, const char* name) {
    const int total = batch * num_query * num_heads * channels;
    float sum = 0.0f;
    float max_val = -1e9f;
    float min_val = 1e9f;
    int nan_count = 0;
    int inf_count = 0;
    
    for (int i = 0; i < total; i++) {
        float val = output[i];
        if (isnan(val)) {
            nan_count++;
        } else if (isinf(val)) {
            inf_count++;
        } else {
            sum += val;
            max_val = fmax(max_val, val);
            min_val = fmin(min_val, val);
        }
    }
    
    printf("\n%s 验证结果:\n", name);
    printf("  总元素数: %d\n", total);
    printf("  求和: %.6f\n", sum);
    printf("  平均值: %.6f\n", sum / total);
    printf("  最大值: %.6f\n", max_val);
    printf("  最小值: %.6f\n", min_val);
    printf("  NaN数量: %d\n", nan_count);
    printf("  Inf数量: %d\n", inf_count);
    
    if (nan_count > 0 || inf_count > 0) {
        printf("  ⚠️ 警告: 输出包含异常值!\n");
    } else {
        printf("  ✓ 输出正常\n");
    }
}

/****************************************************************************************
 * 主测试函数
 ****************************************************************************************/

int main(int argc, char** argv) {
    printf("========================================\n");
    printf("  Deformable Attention CUDA测试\n");
    printf("========================================\n\n");
    
    // 设置随机种子
    srand(time(NULL));
    
    // ==================== 配置参数 ====================
    // 可以根据需要调整这些参数
    const int batch_size = 1;          // 批量大小
    const int num_query = 256;         // 查询点数量
    const int num_heads = 8;           // 注意力头数
    const int channels = 32;           // 每个头的通道数
    const int num_levels = 4;          // 特征金字塔层级数
    const int num_point = 4;           // 每个查询的采样点数
    const int base_height = 512;        // 基础特征图高度
    const int base_width = 512;         // 基础特征图宽度
    
    printf("配置参数:\n");
    printf("  batch_size:   %d\n", batch_size);
    printf("  num_query:    %d\n", num_query);
    printf("  num_heads:    %d\n", num_heads);
    printf("  channels:     %d\n", channels);
    printf("  num_levels:   %d\n", num_levels);
    printf("  num_point:    %d\n", num_point);
    printf("  base_size:    %dx%d\n\n", base_height, base_width);
    
    // ==================== 计算维度 ====================
    // spatial_size: 所有层级的空间位置总数
    int spatial_size = 0;
    for (int l = 0; l < num_levels; l++) {
        int h = base_height >> l;
        int w = base_width >> l;
        spatial_size += h * w;
    }
    
    printf("计算的spatial_size: %d\n\n", spatial_size);
    
    // ==================== 分配主机内存 ====================
    printf("分配主机内存...\n");
    
    // 输入张量
    size_t value_size = (size_t)batch_size * spatial_size * num_heads * channels;
    size_t spatial_shapes_size = (size_t)num_levels * 2;
    size_t level_start_index_size = (size_t)num_levels;
    size_t sampling_loc_size = (size_t)batch_size * num_query * num_heads * num_levels * num_point * 2;
    size_t attn_weight_size = (size_t)batch_size * num_query * num_heads * num_levels * num_point;
    size_t output_size = (size_t)batch_size * num_query * num_heads * channels;
    
    float* h_value = (float*)malloc(value_size * sizeof(float));
    int64_t* h_spatial_shapes = (int64_t*)malloc(spatial_shapes_size * sizeof(int64_t));
    int64_t* h_level_start_index = (int64_t*)malloc(level_start_index_size * sizeof(int64_t));
    float* h_sampling_loc = (float*)malloc(sampling_loc_size * sizeof(float));
    float* h_attn_weight = (float*)malloc(attn_weight_size * sizeof(float));
    float* h_output = (float*)malloc(output_size * sizeof(float));
    
    if (!h_value || !h_spatial_shapes || !h_level_start_index || 
        !h_sampling_loc || !h_attn_weight || !h_output) {
        fprintf(stderr, "主机内存分配失败!\n");
        return EXIT_FAILURE;
    }
    
    // ==================== 初始化数据 ====================
    printf("初始化测试数据...\n");
    
    // 初始化value (特征值)
    init_random_float(h_value, value_size, -1.0f, 1.0f);
    
    // 初始化spatial_shapes (空间形状)
    init_spatial_shapes(h_spatial_shapes, num_levels, base_height, base_width);
    
    // 初始化level_start_index (层级起始索引)
    compute_level_start_index(h_level_start_index, h_spatial_shapes, num_levels, num_heads, channels);
    
    // 初始化sampling_loc (采样位置，归一化到[0,1])
    init_random_float(h_sampling_loc, sampling_loc_size, 0.1f, 0.9f);
    
    // 初始化attn_weight (注意力权重，使用softmax友好的值)
    init_random_float(h_attn_weight, attn_weight_size, 0.0f, 1.0f);
    // 对每组权重进行归一化
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t q = 0; q < num_query; q++) {
            for (size_t h = 0; h < num_heads; h++) {
                float sum = 0.0f;
                size_t base_idx = ((b * num_query + q) * num_heads + h) * num_levels * num_point;
                for (size_t i = 0; i < num_levels * num_point; i++) {
                    sum += h_attn_weight[base_idx + i];
                }
                if (sum > 0) {
                    for (size_t i = 0; i < num_levels * num_point; i++) {
                        h_attn_weight[base_idx + i] /= sum;
                    }
                }
            }
        }
    }
    
    printf("数据初始化完成\n\n");
    
    // ==================== 分配设备内存 ====================
    printf("分配设备内存...\n");
    
    float *d_value, *d_sampling_loc, *d_attn_weight, *d_output;
    int64_t *d_spatial_shapes, *d_level_start_index;
    
    CUDA_CHECK(cudaMalloc(&d_value, value_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_spatial_shapes, spatial_shapes_size * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_level_start_index, level_start_index_size * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_sampling_loc, sampling_loc_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_weight, attn_weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(float)));
    
    size_t total_device_memory = value_size * sizeof(float) + 
                                  spatial_shapes_size * sizeof(int64_t) +
                                  level_start_index_size * sizeof(int64_t) +
                                  sampling_loc_size * sizeof(float) +
                                  attn_weight_size * sizeof(float) +
                                  output_size * sizeof(float);
    
    printf("设备内存分配完成 (%.2f MB)\n\n", total_device_memory / (1024.0 * 1024.0));
    
    // ==================== 创建CUDA事件用于计时 ====================
    cudaEvent_t start_h2d, stop_h2d, start_kernel, stop_kernel, start_d2h, stop_d2h;
    CUDA_CHECK(cudaEventCreate(&start_h2d));
    CUDA_CHECK(cudaEventCreate(&stop_h2d));
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));
    CUDA_CHECK(cudaEventCreate(&start_d2h));
    CUDA_CHECK(cudaEventCreate(&stop_d2h));
    
    // ==================== 主机到设备内存传输 ====================
    printf("传输数据到设备...\n");
    CUDA_CHECK(cudaEventRecord(start_h2d));
    
    CUDA_CHECK(cudaMemcpy(d_value, h_value, value_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_spatial_shapes, h_spatial_shapes, spatial_shapes_size * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_level_start_index, h_level_start_index, level_start_index_size * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sampling_loc, h_sampling_loc, sampling_loc_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_attn_weight, h_attn_weight, attn_weight_size * sizeof(float), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaEventRecord(stop_h2d));
    CUDA_CHECK(cudaEventSynchronize(stop_h2d));
    
    float h2d_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_time, start_h2d, stop_h2d));
    printf("H2D传输完成 (%.3f ms)\n\n", h2d_time);
    
    // ==================== 执行Deformable Attention Forward ====================
    printf("执行 Deformable Attention Forward...\n");
    
    CUDA_CHECK(cudaEventRecord(start_kernel));
    
    ms_deform_attn_cuda_forward(
        d_value,
        d_spatial_shapes,
        d_level_start_index,
        d_sampling_loc,
        d_attn_weight,
        d_output,
        batch_size,
        spatial_size,
        num_heads,
        channels,
        num_levels,
        num_query,
        num_point,
        0  // default stream
    );
    
    CUDA_CHECK(cudaEventRecord(stop_kernel));
    CUDA_CHECK(cudaEventSynchronize(stop_kernel));
    
    float kernel_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel));
    printf("Kernel执行完成 (%.3f ms)\n\n", kernel_time);
    
    // ==================== 设备到主机内存传输 ====================
    printf("传输结果回主机...\n");
    CUDA_CHECK(cudaEventRecord(start_d2h));
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaEventRecord(stop_d2h));
    CUDA_CHECK(cudaEventSynchronize(stop_d2h));
    
    float d2h_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_time, start_d2h, stop_d2h));
    printf("D2H传输完成 (%.3f ms)\n", d2h_time);
    
    // ==================== 验证结果 ====================
    verify_output(h_output, batch_size, num_query, num_heads, channels, "输出");
    
    // ==================== 计算性能指标 ====================
    PerformanceStats stats;
    stats.kernel_time_ms = kernel_time;
    stats.memory_transfer_time_ms = h2d_time + d2h_time;
    stats.total_time_ms = stats.kernel_time_ms + stats.memory_transfer_time_ms;
    
    // 估算操作数：对于每个输出元素，需要进行 num_levels * num_point 次采样和乘加
    stats.num_operations = (long long)batch_size * num_query * num_heads * channels * 
                          num_levels * num_point * 10;  // 约10个操作/采样点
    stats.gflops = (stats.num_operations / 1e9) / (stats.kernel_time_ms / 1000.0);
    stats.memory_footprint_bytes = total_device_memory;
    
    stats.print();
    
    // ==================== 清理资源 ====================
    printf("\n清理资源...\n");
    
    // 释放设备内存
    cudaFree(d_value);
    cudaFree(d_spatial_shapes);
    cudaFree(d_level_start_index);
    cudaFree(d_sampling_loc);
    cudaFree(d_attn_weight);
    cudaFree(d_output);
    
    // 释放主机内存
    free(h_value);
    free(h_spatial_shapes);
    free(h_level_start_index);
    free(h_sampling_loc);
    free(h_attn_weight);
    free(h_output);
    
    // 销毁事件
    cudaEventDestroy(start_h2d);
    cudaEventDestroy(stop_h2d);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);
    cudaEventDestroy(start_d2h);
    cudaEventDestroy(stop_d2h);
    
    printf("\n");
    printf("========================================\n");
    printf("  测试完成!\n");
    printf("========================================\n");
    
    return EXIT_SUCCESS;
}