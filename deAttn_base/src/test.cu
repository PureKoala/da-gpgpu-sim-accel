/*!
**************************************************************************************************
* Deformable Attention Test Program (Pure CUDA, No PyTorch)
* Modified for GPGPU-Sim compatibility with FP16 Tensor Core support
* 
* Key Features:
* - Uses FP16 Tensor Core for prediction phase (no quantization overhead)
* - Direct FP32 accumulator output (eliminates dequantization step)
* - Better numerical accuracy compared to INT8 version
* - Simplified data flow: FP32 -> FP16 -> FP32 (vs FP32 -> INT8 -> INT32 -> FP32)
* 
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
#include "deform_attn/cuda/ms_deform_attn_predict_cuda.cuh"

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
    
    // 新增：预测阶段性能统计
    float predict_so_time_ms;
    float predict_attn_time_ms;
    float conversion_time_ms;  // FP32->FP16转换时间
    double tensor_core_utilization_n16;  // N=16时的利用率 (FP16最优)
    double tensor_core_utilization_actual;  // 实际N维度的利用率
    
    void print() const {
        printf("\n");
        printf("========================================\n");
        printf("         性能统计摘要 (FP16 Tensor Core)\n");
        printf("========================================\n");
        printf("预测阶段:\n");
        printf("  SO预测时间:        %.3f us\n", predict_so_time_ms);
        printf("  Attn预测时间:      %.3f us\n", predict_attn_time_ms);
        printf("  FP32->FP16转换:    %.3f ms\n", conversion_time_ms);
        printf("\n采样聚合阶段:\n");
        printf("  Kernel执行时间:    %.3f us\n", kernel_time_ms);
        printf("\n总体性能:\n");
        printf("  内存传输时间:      %.3f us\n", memory_transfer_time_ms);
        printf("  总时间:            %.3f us\n", total_time_ms);
        printf("  计算操作数:        %lld\n", num_operations);
        printf("  计算性能:          %.2f GFLOPS\n", gflops);
        printf("  内存占用:          %.2f MB\n", memory_footprint_bytes / (1024.0 * 1024.0));
        printf("\nTensor Core利用率分析 (N维度限制):\n");
        printf("  N=16场景利用率:    %.1f%% (FP16最优配置)\n", tensor_core_utilization_n16 * 100);
        printf("  实际场景利用率:    %.1f%%\n", tensor_core_utilization_actual * 100);
        printf("\n优势 vs INT8版本:\n");
        printf("  ✓ 无需量化/反量化\n");
        printf("  ✓ 更高数值精度\n");
        printf("  ✓ 简化的数据流\n");
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
    printf("[DEBUG] 程序开始执行\n");
    printf("========================================\n");
    printf("  Deformable Attention CUDA测试\n");
    printf("========================================\n\n");
    
    // 设置随机种子
    printf("[DEBUG] 设置随机种子\n");
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
    printf("[DEBUG] 开始分配主机内存\n");
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
    printf("[DEBUG] 主机内存分配成功\n");
    
    // ==================== 初始化数据 ====================
    printf("[DEBUG] 开始初始化测试数据\n");
    printf("初始化测试数据...\n");
    
    // 初始化value (特征值)
    init_random_float(h_value, value_size, -1.0f, 1.0f);
    
    // 初始化spatial_shapes (空间形状)
    init_spatial_shapes(h_spatial_shapes, num_levels, base_height, base_width);
    
    // 初始化level_start_index (层级起始索引)
    compute_level_start_index(h_level_start_index, h_spatial_shapes, num_levels, num_heads, channels);
    
    // ==================== 预测阶段：初始化 Q, W_SO, W_A (FP16) ====================
    printf("[DEBUG] 开始初始化预测阶段数据 (FP16)\n");
    printf("初始化预测阶段数据 (FP16 Tensor Core - 无需量化)...\n");
    
    // 定义预测阶段的维度
    const int C_in = 256;              // 输入特征维度 (Query维度)
    const int SO_out_orig = num_heads * num_levels * num_point * 2;  // 采样偏移输出 (原始)
    const int A_out_orig = num_heads * num_levels * num_point;        // 注意力权重输出 (原始)
    
    // N-Padding: FP16 Tensor Core WMMA API optimal N is 16 (m16n16k16)
    // Padding to multiples of 16 for optimal performance
    const int SO_out_padded = ((SO_out_orig + 15) / 16) * 16;  // Round up to multiple of 16
    const int A_out_padded = ((A_out_orig + 15) / 16) * 16;    // Round up to multiple of 16
    
    printf("  SO输出维度: 原始=%d, 填充后=%d (浪费%.1f%%)\n", 
           SO_out_orig, SO_out_padded, 
           100.0f * (SO_out_padded - SO_out_orig) / SO_out_padded);
    printf("  A输出维度:  原始=%d, 填充后=%d (浪费%.1f%%)\n", 
           A_out_orig, A_out_padded,
           100.0f * (A_out_padded - A_out_orig) / A_out_padded);
    
    // 分配主机端预测阶段内存 (FP32)
    size_t Q_size = (size_t)batch_size * num_query * C_in;
    size_t W_SO_size = (size_t)C_in * SO_out_padded;  // 使用填充后的维度
    size_t W_A_size = (size_t)C_in * A_out_padded;    // 使用填充后的维度
    
    float* h_Q = (float*)malloc(Q_size * sizeof(float));
    float* h_W_SO = (float*)malloc(W_SO_size * sizeof(float));
    float* h_W_A = (float*)malloc(W_A_size * sizeof(float));
    
    if (!h_Q || !h_W_SO || !h_W_A) {
        fprintf(stderr, "预测阶段主机内存分配失败!\n");
        return EXIT_FAILURE;
    }
    printf("[DEBUG] 预测阶段主机内存分配成功\n");
    
    // 初始化预测阶段数据 (FP32)
    printf("[DEBUG] 初始化Q数据\n");
    init_random_float(h_Q, Q_size, -1.0f, 1.0f);
    
    // 初始化权重（只初始化有效部分，填充部分置零）
    printf("[DEBUG] 初始化W_SO数据（有效部分：%d，填充后：%d）\n", 
           C_in * SO_out_orig, (int)W_SO_size);
    memset(h_W_SO, 0, W_SO_size * sizeof(float));  // 先全部置零
    for (int i = 0; i < C_in; i++) {
        for (int j = 0; j < SO_out_orig; j++) {
            h_W_SO[i * SO_out_padded + j] = -0.5f + ((float)rand() / RAND_MAX);
        }
    }
    
    printf("[DEBUG] 初始化W_A数据（有效部分：%d，填充后：%d）\n",
           C_in * A_out_orig, (int)W_A_size);
    memset(h_W_A, 0, W_A_size * sizeof(float));  // 先全部置零
    for (int i = 0; i < C_in; i++) {
        for (int j = 0; j < A_out_orig; j++) {
            h_W_A[i * A_out_padded + j] = -0.5f + ((float)rand() / RAND_MAX);
        }
    }
    
    // 转换：FP32 → FP16 (在主机端进行)
    printf("[DEBUG] 开始FP32->FP16转换\n");
    clock_t convert_start = clock();
    
    printf("[DEBUG] 分配FP16内存\n");
    half* h_Q_fp16 = (half*)malloc(Q_size * sizeof(half));
    half* h_W_SO_fp16 = (half*)malloc(W_SO_size * sizeof(half));
    half* h_W_A_fp16 = (half*)malloc(W_A_size * sizeof(half));
    
    if (!h_Q_fp16 || !h_W_SO_fp16 || !h_W_A_fp16) {
        fprintf(stderr, "FP16内存分配失败!\n");
        return EXIT_FAILURE;
    }
    
    printf("[DEBUG] 转换Q到FP16\n");
    convert_fp32_to_fp16(h_Q, h_Q_fp16, Q_size);
    printf("[DEBUG] 转换W_SO到FP16\n");
    convert_fp32_to_fp16(h_W_SO, h_W_SO_fp16, W_SO_size);
    printf("[DEBUG] 转换W_A到FP16\n");
    convert_fp32_to_fp16(h_W_A, h_W_A_fp16, W_A_size);
    
    clock_t convert_end = clock();
    float convert_time_ms = (float)(convert_end - convert_start) * 1000.0f / CLOCKS_PER_SEC;
    
    printf("预测阶段数据准备完成 (FP32->FP16转换时间: %.3f ms)\n", convert_time_ms);
    printf("  Q维度: [%d × %d] → FP16\n", batch_size * num_query, C_in);
    printf("  W_SO维度: [%d × %d] → FP16 (填充后)\n", C_in, SO_out_padded);
    printf("  W_A维度: [%d × %d] → FP16 (填充后)\n\n", C_in, A_out_padded);
    
    printf("数据初始化完成\n\n");
    
    // ==================== 分配设备内存（预测阶段 + 采样聚合阶段） ====================
    printf("[DEBUG] 开始分配设备内存\n");
    printf("分配设备内存...\n");
    
    // 采样聚合阶段的设备内存
    float *d_value, *d_output;
    // 注意：d_sampling_loc 和 d_attn_weight 不再需要，直接使用预测结果
    int64_t *d_spatial_shapes, *d_level_start_index;
    
    printf("[DEBUG] cudaMalloc d_value\n");
    CUDA_CHECK(cudaMalloc(&d_value, value_size * sizeof(float)));
    printf("[DEBUG] cudaMalloc d_spatial_shapes\n");
    CUDA_CHECK(cudaMalloc(&d_spatial_shapes, spatial_shapes_size * sizeof(int64_t)));
    printf("[DEBUG] cudaMalloc d_level_start_index\n");
    CUDA_CHECK(cudaMalloc(&d_level_start_index, level_start_index_size * sizeof(int64_t)));
    // 不再分配 d_sampling_loc 和 d_attn_weight，直接使用预测结果
    // printf("[DEBUG] cudaMalloc d_sampling_loc\n");
    // CUDA_CHECK(cudaMalloc(&d_sampling_loc, sampling_loc_size * sizeof(float)));
    // printf("[DEBUG] cudaMalloc d_attn_weight\n");
    // CUDA_CHECK(cudaMalloc(&d_attn_weight, attn_weight_size * sizeof(float)));
    printf("[DEBUG] cudaMalloc d_output\n");
    CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(float)));
    
    // 预测阶段的设备内存 (FP16 Tensor Core)
    half *d_Q_fp16, *d_W_SO_fp16, *d_W_A_fp16;
    float *d_SO_fp32, *d_A_fp32;
    
    printf("[DEBUG] cudaMalloc d_Q_fp16\n");
    CUDA_CHECK(cudaMalloc(&d_Q_fp16, Q_size * sizeof(half)));
    printf("[DEBUG] cudaMalloc d_W_SO_fp16\n");
    CUDA_CHECK(cudaMalloc(&d_W_SO_fp16, W_SO_size * sizeof(half)));
    printf("[DEBUG] cudaMalloc d_W_A_fp16\n");
    CUDA_CHECK(cudaMalloc(&d_W_A_fp16, W_A_size * sizeof(half)));
    printf("[DEBUG] cudaMalloc d_SO_fp32\n");
    CUDA_CHECK(cudaMalloc(&d_SO_fp32, (size_t)batch_size * num_query * SO_out_padded * sizeof(float)));
    printf("[DEBUG] cudaMalloc d_A_fp32\n");
    CUDA_CHECK(cudaMalloc(&d_A_fp32, (size_t)batch_size * num_query * A_out_padded * sizeof(float)));
    
    size_t total_device_memory = value_size * sizeof(float) + 
                                  spatial_shapes_size * sizeof(int64_t) +
                                  level_start_index_size * sizeof(int64_t) +
                                  // sampling_loc_size * sizeof(float) +  // 不再需要
                                  // attn_weight_size * sizeof(float) +   // 不再需要
                                  output_size * sizeof(float) +
                                  Q_size * sizeof(half) +
                                  W_SO_size * sizeof(half) +
                                  W_A_size * sizeof(half) +
                                  (size_t)batch_size * num_query * SO_out_padded * sizeof(float) +
                                  (size_t)batch_size * num_query * A_out_padded * sizeof(float);
    
    printf("设备内存分配完成 (%.2f MB)\n\n", total_device_memory / (1024.0 * 1024.0));
    
    // ==================== 创建CUDA事件用于计时 ====================
    printf("[DEBUG] 创建CUDA事件\n");
    cudaEvent_t start_h2d, stop_h2d, start_pred_so, stop_pred_so, start_pred_attn, stop_pred_attn;
    cudaEvent_t start_kernel, stop_kernel, start_d2h, stop_d2h;
    CUDA_CHECK(cudaEventCreate(&start_h2d));
    CUDA_CHECK(cudaEventCreate(&stop_h2d));
    CUDA_CHECK(cudaEventCreate(&start_pred_so));
    CUDA_CHECK(cudaEventCreate(&stop_pred_so));
    CUDA_CHECK(cudaEventCreate(&start_pred_attn));
    CUDA_CHECK(cudaEventCreate(&stop_pred_attn));
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));
    CUDA_CHECK(cudaEventCreate(&start_d2h));
    CUDA_CHECK(cudaEventCreate(&stop_d2h));
    
    // ==================== 主机到设备内存传输 ====================
    printf("[DEBUG] 开始H2D传输\n");
    printf("传输数据到设备...\n");
    CUDA_CHECK(cudaEventRecord(start_h2d));
    
    // 传输采样聚合阶段所需数据
    CUDA_CHECK(cudaMemcpy(d_value, h_value, value_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_spatial_shapes, h_spatial_shapes, spatial_shapes_size * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_level_start_index, h_level_start_index, level_start_index_size * sizeof(int64_t), cudaMemcpyHostToDevice));
    
    // 传输预测阶段所需数据 (FP16)
    CUDA_CHECK(cudaMemcpy(d_Q_fp16, h_Q_fp16, Q_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W_SO_fp16, h_W_SO_fp16, W_SO_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W_A_fp16, h_W_A_fp16, W_A_size * sizeof(half), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaEventRecord(stop_h2d));
    CUDA_CHECK(cudaEventSynchronize(stop_h2d));
    
    float h2d_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_time, start_h2d, stop_h2d));
    printf("H2D传输完成 (%.3f us)\n\n", h2d_time);
    
    // ==================== 执行预测阶段 (FP16 Tensor Core) ====================
    printf("[DEBUG] 开始执行预测阶段 (FP16)\n");
    printf("执行预测阶段 (FP16 Tensor Core GEMM - 无需量化/反量化)...\n");
    
    // 1. 采样偏移量预测: SO = Q × W_SO (FP16 输入, FP32 累加器输出)
    printf("  [1/2] 预测采样偏移量 (实际N=%d, 填充到N=%d, 浪费%.1f%%)...\n",
           SO_out_orig, SO_out_padded, 
           100.0f * (SO_out_padded - SO_out_orig) / SO_out_padded);
    printf("[DEBUG] 调用 ms_deform_attn_predict_so_cuda (FP16)\n");
    CUDA_CHECK(cudaEventRecord(start_pred_so));
    
    ms_deform_attn_predict_so_cuda(
        d_Q_fp16,
        d_W_SO_fp16,
        d_SO_fp32,          // 直接输出FP32
        batch_size,
        num_query,
        C_in,
        SO_out_padded,
        0  // default stream
    );
    
    printf("[DEBUG] ms_deform_attn_predict_so_cuda 调用完成\n");
    CUDA_CHECK(cudaEventRecord(stop_pred_so));
    CUDA_CHECK(cudaEventSynchronize(stop_pred_so));
    
    float pred_so_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&pred_so_time, start_pred_so, stop_pred_so));
    printf("  SO预测完成 (%.3f us)\n", pred_so_time);
    
    // 2. 注意力权重预测: A = Q × W_A (FP16 输入, FP32 累加器输出)
    printf("  [2/2] 预测注意力权重 (实际N=%d, 填充到N=%d, 浪费%.1f%%)...\n",
           A_out_orig, A_out_padded,
           100.0f * (A_out_padded - A_out_orig) / A_out_padded);
    printf("[DEBUG] 调用 ms_deform_attn_predict_attn_cuda (FP16)\n");
    CUDA_CHECK(cudaEventRecord(start_pred_attn));
    
    ms_deform_attn_predict_attn_cuda(
        d_Q_fp16,
        d_W_A_fp16,
        d_A_fp32,           // 直接输出FP32
        batch_size,
        num_query,
        C_in,
        A_out_padded,
        0  // default stream
    );
    
    printf("[DEBUG] ms_deform_attn_predict_attn_cuda 调用完成\n");
    CUDA_CHECK(cudaEventRecord(stop_pred_attn));
    CUDA_CHECK(cudaEventSynchronize(stop_pred_attn));
    
    float pred_attn_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&pred_attn_time, start_pred_attn, stop_pred_attn));
    printf("  Attn预测完成 (%.3f us)\n", pred_attn_time);
    
    // 注意：直接使用d_SO_fp32和d_A_fp32作为输入，无需memcpy！
    // 原始维度 SO_out_orig=%d 和 A_out_orig=%d 已经足够
    // 填充的额外元素不影响forward kernel（它只读取有效部分）
    printf("  [3/3] 跳过memcpy，直接使用预测结果 (零拷贝优化)...\n");
    
    printf("预测阶段总时间: %.3f us (无需量化/反量化开销，无memcpy开销!)\n\n", pred_so_time + pred_attn_time);
    
    // ==================== 执行Deformable Attention Forward (采样聚合阶段) ====================
    printf("[DEBUG] 开始采样聚合阶段\n");
    printf("执行 Deformable Attention Forward (采样聚合阶段)...\n");
    
    CUDA_CHECK(cudaEventRecord(start_kernel));
    
    printf("[DEBUG] 调用 ms_deform_attn_cuda_forward\n");
    ms_deform_attn_cuda_forward(
        d_value,
        d_spatial_shapes,
        d_level_start_index,
        d_SO_fp32,        // 直接使用预测结果，无需memcpy
        d_A_fp32,         // 直接使用预测结果，无需memcpy
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
    
    printf("[DEBUG] ms_deform_attn_cuda_forward 调用完成\n");
    CUDA_CHECK(cudaEventRecord(stop_kernel));
    CUDA_CHECK(cudaEventSynchronize(stop_kernel));
    
    float kernel_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel));
    printf("Kernel执行完成 (%.3f us)\n\n", kernel_time);
    
    // ==================== 设备到主机内存传输 ====================
    printf("[DEBUG] 开始D2H传输\n");
    printf("传输结果回主机...\n");
    CUDA_CHECK(cudaEventRecord(start_d2h));
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaEventRecord(stop_d2h));
    CUDA_CHECK(cudaEventSynchronize(stop_d2h));
    
    float d2h_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_time, start_d2h, stop_d2h));
    printf("D2H传输完成 (%.3f us)\n", d2h_time);
    
    // ==================== 验证结果 ====================
    printf("[DEBUG] 开始验证结果\n");
    verify_output(h_output, batch_size, num_query, num_heads, channels, "输出");
    
    // ==================== 计算性能指标 ====================
    PerformanceStats stats;
    stats.kernel_time_ms = kernel_time;
    stats.memory_transfer_time_ms = h2d_time + d2h_time;
    
    // 预测阶段性能统计 (FP16)
    stats.predict_so_time_ms = pred_so_time;
    stats.predict_attn_time_ms = pred_attn_time;
    stats.conversion_time_ms = convert_time_ms;
    
    // 计算实际的Tensor Core利用率（基于N维度）
    // FP16最优配置是N=16 (m16n16k16指令)
    stats.tensor_core_utilization_n16 = 1.0;  // N=16时完全利用
    stats.tensor_core_utilization_actual = (double)SO_out_orig / SO_out_padded;  // 实际利用率
    
    // 更新总时间 (不包括主机端的FP32->FP16转换，因为可以预先离线完成)
    stats.total_time_ms = stats.predict_so_time_ms + stats.predict_attn_time_ms + 
                          stats.kernel_time_ms + stats.memory_transfer_time_ms;
    
    // 估算操作数：对于每个输出元素，需要进行 num_levels * num_point 次采样和乘加
    stats.num_operations = (long long)batch_size * num_query * num_heads * channels * 
                          num_levels * num_point * 10;  // 约10个操作/采样点
    stats.gflops = (stats.num_operations / 1e9) / (stats.kernel_time_ms / 1000000.0);  // 转换us到秒
    stats.memory_footprint_bytes = total_device_memory;
    
    stats.print();
    
    // ==================== 清理资源 ====================
    printf("[DEBUG] 开始清理资源\n");
    printf("\n清理资源...\n");
    
    // 释放设备内存
    cudaFree(d_value);
    cudaFree(d_spatial_shapes);
    cudaFree(d_level_start_index);
    // cudaFree(d_sampling_loc);  // 不再需要
    // cudaFree(d_attn_weight);   // 不再需要
    cudaFree(d_output);
    
    // 释放预测阶段设备内存
    cudaFree(d_Q_fp16);
    cudaFree(d_W_SO_fp16);
    cudaFree(d_W_A_fp16);
    cudaFree(d_SO_fp32);
    cudaFree(d_A_fp32);
    
    // 释放主机内存
    free(h_value);
    free(h_spatial_shapes);
    free(h_level_start_index);
    free(h_sampling_loc);
    free(h_attn_weight);
    free(h_output);
    
    // 释放预测阶段主机内存
    free(h_Q);
    free(h_W_SO);
    free(h_W_A);
    free(h_Q_fp16);
    free(h_W_SO_fp16);
    free(h_W_A_fp16);
    
    // 销毁事件
    cudaEventDestroy(start_h2d);
    cudaEventDestroy(stop_h2d);
    cudaEventDestroy(start_pred_so);
    cudaEventDestroy(stop_pred_so);
    cudaEventDestroy(start_pred_attn);
    cudaEventDestroy(stop_pred_attn);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);
    cudaEventDestroy(start_d2h);
    cudaEventDestroy(stop_d2h);
    
    printf("[DEBUG] 资源清理完成\n");
    printf("\n");
    printf("========================================\n");
    printf("  测试完成!\n");
    printf("========================================\n");
    
    printf("[DEBUG] 程序正常退出\n");
    return EXIT_SUCCESS;
}