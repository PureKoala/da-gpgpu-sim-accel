#include "../include/deformable_attention.h"
#include "../include/tensor_core_gemm.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cfloat>

/**
 * DeformableAttention CUDA 实现
 * 针对第一个Block: 56x56输入, 128维, 4头, 7x7采样网格
 * 
 * 所有矩阵乘法使用 Tensor Core INT8 GEMM
 */

// ==================== 量化/反量化 Kernel ====================

/**
 * 计算量化scale: scale = 127 / max(abs(tensor))
 */
__global__ void compute_scale_kernel(
    const float* __restrict__ input,
    float* __restrict__ scale,
    int size
) {
    __shared__ float smem[32];
    
    // 找最大绝对值
    float max_val = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        max_val = fmaxf(max_val, fabsf(input[i]));
    }
    
    // Warp reduce
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    
    if (lane == 0) smem[warp_id] = max_val;
    __syncthreads();
    
    if (warp_id == 0) {
        max_val = (lane < (blockDim.x + 31) / 32) ? smem[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
        }
        if (lane == 0) {
            *scale = (max_val > 1e-8f) ? (127.0f / max_val) : 1.0f;
        }
    }
}

/**
 * FP32 -> INT8 量化
 */
__global__ void quantize_fp32_to_int8_kernel(
    const float* __restrict__ input,
    int8_t* __restrict__ output,
    float scale,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx] * scale;
        val = fmaxf(-127.0f, fminf(127.0f, val));
        output[idx] = static_cast<int8_t>(rintf(val));
    }
}

/**
 * INT8 -> FP32 反量化
 */
__global__ void dequantize_int8_to_fp32_kernel(
    const int8_t* __restrict__ input,
    float* __restrict__ output,
    float scale,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = static_cast<float>(input[idx]) / scale;
    }
}

// ==================== 辅助Kernel ====================

/**
 * Depthwise Convolution 3x3
 * 用于offset特征提取
 */
__global__ void depthwise_conv3x3_kernel(
    const float* __restrict__ input,   // (B, C, H, W)
    const float* __restrict__ weight,  // (C, 1, 3, 3)
    float* __restrict__ output,        // (B, C, H, W)
    int B, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * W;
    
    if (idx < total) {
        int w = idx % W;
        int h = (idx / W) % H;
        int c = (idx / (W * H)) % C;
        int b = idx / (W * H * C);
        
        float sum = 0.0f;
        
        // 3x3卷积
        #pragma unroll
        for (int kh = 0; kh < 3; kh++) {
            #pragma unroll
            for (int kw = 0; kw < 3; kw++) {
                int h_in = h + kh - 1;  // padding=1
                int w_in = w + kw - 1;
                
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    int in_idx = ((b * C + c) * H + h_in) * W + w_in;
                    int weight_idx = (c * 3 + kh) * 3 + kw;
                    sum += input[in_idx] * weight[weight_idx];
                }
            }
        }
        
        output[idx] = sum;
    }
}

/**
 * 固化的 AdaptiveAvgPool2d: 56x56 -> 7x7
 * stride = 8, kernel = 8x8
 */
__global__ void adaptive_avgpool_56to7_kernel(
    const float* __restrict__ input,   // (B, C, 56, 56)
    float* __restrict__ output,        // (B, C, 7, 7)
    int B, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * 7 * 7;
    
    if (idx < total) {
        int out_w = idx % 7;
        int out_h = (idx / 7) % 7;
        int c = (idx / 49) % C;
        int b = idx / (49 * C);
        
        // 计算输入区域 (8x8 kernel, stride 8)
        int in_h_start = out_h * 8;
        int in_w_start = out_w * 8;
        
        float sum = 0.0f;
        #pragma unroll
        for (int kh = 0; kh < 8; kh++) {
            #pragma unroll
            for (int kw = 0; kw < 8; kw++) {
                int in_h = in_h_start + kh;
                int in_w = in_w_start + kw;
                if (in_h < 56 && in_w < 56) {
                    int in_idx = ((b * C + c) * 56 + in_h) * 56 + in_w;
                    sum += input[in_idx];
                }
            }
        }
        
        output[idx] = sum / 64.0f;  // 8x8 = 64
    }
}

/**
 * LayerNorm kernel
 */
__global__ void layer_norm_kernel(
    float* __restrict__ data,           // (B, N, C)
    const float* __restrict__ weight,   // (C,)
    const float* __restrict__ bias,     // (C,)
    int B, int N, int C
) {
    int b = blockIdx.y;
    int n = blockIdx.x;
    
    if (b < B && n < N) {
        float* row = data + (b * N + n) * C;
        
        // 计算均值
        float mean = 0.0f;
        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            mean += row[c];
        }
        
        // Warp reduce
        __shared__ float smem[32];
        int lane = threadIdx.x % 32;
        int warp_id = threadIdx.x / 32;
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            mean += __shfl_down_sync(0xffffffff, mean, offset);
        }
        
        if (lane == 0) smem[warp_id] = mean;
        __syncthreads();
        
        if (warp_id == 0) {
            mean = (lane < (blockDim.x + 31) / 32) ? smem[lane] : 0.0f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                mean += __shfl_down_sync(0xffffffff, mean, offset);
            }
        }
        __syncthreads();
        
        mean = smem[0] / C;
        
        // 计算方差
        float var = 0.0f;
        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            float diff = row[c] - mean;
            var += diff * diff;
        }
        
        // Warp reduce
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            var += __shfl_down_sync(0xffffffff, var, offset);
        }
        
        if (lane == 0) smem[warp_id] = var;
        __syncthreads();
        
        if (warp_id == 0) {
            var = (lane < (blockDim.x + 31) / 32) ? smem[lane] : 0.0f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                var += __shfl_down_sync(0xffffffff, var, offset);
            }
        }
        __syncthreads();
        
        var = smem[0] / C;
        float inv_std = rsqrtf(var + 1e-5f);
        
        // 归一化并应用weight和bias
        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            row[c] = (row[c] - mean) * inv_std * weight[c] + bias[c];
        }
    }
}

/**
 * GELU 激活函数
 */
__device__ __forceinline__ float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void gelu_kernel(
    float* __restrict__ data,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = gelu(data[idx]);
    }
}

/**
 * 将(B, C, H, W)重排为(C, B*H*W)用于GEMM
 * 列优先布局以适配GEMM的转置
 */
__global__ void reshape_BCHW_to_CHW_kernel(
    const float* __restrict__ input,   // (B, C, H, W)
    float* __restrict__ output,        // (C, B*H*W) 或者说 (B*H*W, C) 转置
    int B, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * H * W;
    int total = C * N;
    
    if (idx < total) {
        int n = idx % N;  // spatial position in B*H*W
        int c = idx / N;  // channel
        
        int b = n / (H * W);
        int hw = n % (H * W);
        int h = hw / W;
        int w = hw % W;
        
        int in_idx = ((b * C + c) * H + h) * W + w;
        output[idx] = input[in_idx];
    }
}

/**
 * 将(C, B*H*W) GEMM输出重排回(B, C, H, W)
 */
__global__ void reshape_CHW_to_BCHW_kernel(
    const float* __restrict__ input,   // (C, B*H*W)
    float* __restrict__ output,        // (B, C, H, W)
    int B, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * W;
    
    if (idx < total) {
        int w = idx % W;
        int h = (idx / W) % H;
        int c = (idx / (W * H)) % C;
        int b = idx / (W * H * C);
        
        int n = b * H * W + h * W + w;
        int in_idx = c * (B * H * W) + n;
        
        output[idx] = input[in_idx];
    }
}

/**
 * 1x1 Convolution 使用 Tensor Core GEMM实现
 * Conv1x1(input, weight) = Reshape(GEMM(weight, Reshape(input)))
 * 
 * input: (B, C_in, H, W) -> reshape to (C_in, B*H*W)
 * weight: (C_out, C_in, 1, 1) -> (C_out, C_in)
 * output: (C_out, B*H*W) -> reshape to (B, C_out, H, W)
 */
cudaError_t conv1x1_tensor_core(
    const float* input,       // (B, C_in, H, W)
    const float* weight,      // (C_out, C_in, 1, 1)
    float* output,            // (B, C_out, H, W)
    int B, int C_in, int C_out, int H, int W,
    float* input_reshaped,    // 工作缓冲区 (C_in, B*H*W)
    int8_t* input_int8,       // 工作缓冲区
    int8_t* weight_int8,      // 工作缓冲区
    float* output_temp,       // 工作缓冲区 (C_out, B*H*W)
    float* scale_buf,         // 2个float
    cudaStream_t stream
) {
    int M = C_out;
    int N = B * H * W;
    int K = C_in;
    
    int threads = 256;
    
    // 1. Reshape input: (B, C_in, H, W) -> (C_in, B*H*W)
    {
        int total = K * N;
        int blocks = (total + threads - 1) / threads;
        reshape_BCHW_to_CHW_kernel<<<blocks, threads, 0, stream>>>(
            input, input_reshaped, B, C_in, H, W
        );
    }
    
    // 2. 计算量化scale
    {
        compute_scale_kernel<<<1, threads, 0, stream>>>(
            input_reshaped, &scale_buf[0], K * N
        );
        compute_scale_kernel<<<1, threads, 0, stream>>>(
            weight, &scale_buf[1], M * K
        );
    }
    
    // 同步获取scale
    float scales[2];
    cudaMemcpyAsync(scales, scale_buf, 2 * sizeof(float), 
                   cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // 3. 量化
    {
        int blocks_input = (K * N + threads - 1) / threads;
        int blocks_weight = (M * K + threads - 1) / threads;
        
        quantize_fp32_to_int8_kernel<<<blocks_input, threads, 0, stream>>>(
            input_reshaped, input_int8, scales[0], K * N
        );
        quantize_fp32_to_int8_kernel<<<blocks_weight, threads, 0, stream>>>(
            weight, weight_int8, scales[1], M * K
        );
    }
    
    // 4. Tensor Core GEMM: output_temp = weight @ input_reshaped
    // (M, K) @ (K, N) = (M, N)
    {
        tensor_core::gemm_int8_fp32(
            weight_int8,    // (M, K)
            input_int8,     // (K, N)
            output_temp,    // (M, N)
            M, N, K,
            1.0f, 0.0f,
            scales[1], scales[0],
            stream
        );
    }
    
    // 5. Reshape output: (C_out, B*H*W) -> (B, C_out, H, W)
    {
        int total = B * C_out * H * W;
        int blocks = (total + threads - 1) / threads;
        reshape_CHW_to_BCHW_kernel<<<blocks, threads, 0, stream>>>(
            output_temp, output, B, C_out, H, W
        );
    }
    
    return cudaGetLastError();
}

/**
 * 生成deformed sampling points
 * reference_points + offsets -> deformed_points
 */
__global__ void generate_deformed_points_kernel(
    const float* __restrict__ offsets,         // (B, 49, 1, 2) for G=1
    const float* __restrict__ reference_points, // (1, 49, 1, 2)
    float* __restrict__ deformed_points,        // (B, 49, 1, 2)
    int B
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * 49 * 2;  // G=1
    
    if (idx < total) {
        int coord = idx % 2;  // x or y
        int point = (idx / 2) % 49;
        int b = idx / (49 * 2);
        
        float ref = reference_points[point * 2 + coord];
        float offset = offsets[(b * 49 + point) * 2 + coord];
        float deformed = ref + offset;
        
        // tanh clamp to [-1, 1]
        deformed = tanhf(deformed);
        
        deformed_points[(b * 49 + point) * 2 + coord] = deformed;
    }
}

/**
 * Grid Sample (Bilinear Interpolation)
 * 从feature map采样特征
 */
__global__ void grid_sample_kernel(
    const float* __restrict__ input,       // (B, C, H, W)
    const float* __restrict__ grid,        // (B, Ns*G, 2) in [-1, 1]
    float* __restrict__ output,            // (B, C, Ns*G)
    int B, int C, int H, int W, int Ns, int G
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * Ns * G;
    
    if (idx < total) {
        int point = idx % (Ns * G);
        int c = (idx / (Ns * G)) % C;
        int b = idx / (C * Ns * G);
        
        // 获取采样坐标 (x, y) in [-1, 1]
        float x = grid[(b * Ns * G + point) * 2 + 0];
        float y = grid[(b * Ns * G + point) * 2 + 1];
        
        // 转换到像素坐标
        float x_pixel = ((x + 1.0f) / 2.0f) * (W - 1);
        float y_pixel = ((y + 1.0f) / 2.0f) * (H - 1);
        
        // 双线性插值
        int x0 = floorf(x_pixel);
        int y0 = floorf(y_pixel);
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        
        float wx1 = x_pixel - x0;
        float wy1 = y_pixel - y0;
        float wx0 = 1.0f - wx1;
        float wy0 = 1.0f - wy1;
        
        float value = 0.0f;
        
        if (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H) {
            int in_idx = ((b * C + c) * H + y0) * W + x0;
            value += input[in_idx] * wx0 * wy0;
        }
        if (x1 >= 0 && x1 < W && y0 >= 0 && y0 < H) {
            int in_idx = ((b * C + c) * H + y0) * W + x1;
            value += input[in_idx] * wx1 * wy0;
        }
        if (x0 >= 0 && x0 < W && y1 >= 0 && y1 < H) {
            int in_idx = ((b * C + c) * H + y1) * W + x0;
            value += input[in_idx] * wx0 * wy1;
        }
        if (x1 >= 0 && x1 < W && y1 >= 0 && y1 < H) {
            int in_idx = ((b * C + c) * H + y1) * W + x1;
            value += input[in_idx] * wx1 * wy1;
        }
        
        output[idx] = value;
    }
}

/**
 * Sample RPB (Relative Position Bias)
 */
__global__ void sample_rpb_kernel(
    const float* __restrict__ rpb_table,      // (M, 13, 13)
    const float* __restrict__ deformed_points, // (B, Ns, G, 2)
    float* __restrict__ rpb,                   // (B, M, 1, Ns)
    int B, int M, int Ns, int G
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * M * Ns;
    
    if (idx < total) {
        int point = idx % Ns;
        int head = (idx / Ns) % M;
        int b = idx / (Ns * M);
        
        // 对于G=1，直接使用deformed_points
        float x = deformed_points[(b * Ns + point) * 2 + 0];  // [-1, 1]
        float y = deformed_points[(b * Ns + point) * 2 + 1];
        
        // 转换到rpb_table索引 [0, 12]
        float x_idx = ((x + 1.0f) / 2.0f) * 12.0f;
        float y_idx = ((y + 1.0f) / 2.0f) * 12.0f;
        
        // 双线性插值采样rpb
        int x0 = floorf(x_idx);
        int y0 = floorf(y_idx);
        int x1 = min(x0 + 1, 12);
        int y1 = min(y0 + 1, 12);
        x0 = max(x0, 0);
        y0 = max(y0, 0);
        
        float wx1 = x_idx - x0;
        float wy1 = y_idx - y0;
        float wx0 = 1.0f - wx1;
        float wy0 = 1.0f - wy1;
        
        float bias = 0.0f;
        bias += rpb_table[(head * 13 + y0) * 13 + x0] * wx0 * wy0;
        bias += rpb_table[(head * 13 + y0) * 13 + x1] * wx1 * wy0;
        bias += rpb_table[(head * 13 + y1) * 13 + x0] * wx0 * wy1;
        bias += rpb_table[(head * 13 + y1) * 13 + x1] * wx1 * wy1;
        
        rpb[(b * M + head) * Ns + point] = bias;
    }
}

/**
 * Softmax (along last dimension)
 */
__global__ void softmax_kernel(
    float* __restrict__ data,  // (B, M, N, Ns)
    int B, int M, int N, int Ns
) {
    int idx = blockIdx.x;
    int b = idx / (M * N);
    int m = (idx / N) % M;
    int n = idx % N;
    
    if (b < B && m < M && n < N) {
        float* row = data + ((b * M + m) * N + n) * Ns;
        
        // 找最大值
        float max_val = row[0];
        for (int i = threadIdx.x; i < Ns; i += blockDim.x) {
            max_val = fmaxf(max_val, row[i]);
        }
        
        // Reduce max
        __shared__ float smem[32];
        int lane = threadIdx.x % 32;
        int warp_id = threadIdx.x / 32;
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
        }
        
        if (lane == 0) smem[warp_id] = max_val;
        __syncthreads();
        
        if (warp_id == 0) {
            max_val = (lane < (blockDim.x + 31) / 32) ? smem[lane] : -INFINITY;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
            }
        }
        __syncthreads();
        max_val = smem[0];
        
        // 计算exp和sum
        float sum = 0.0f;
        for (int i = threadIdx.x; i < Ns; i += blockDim.x) {
            row[i] = expf(row[i] - max_val);
            sum += row[i];
        }
        
        // Reduce sum
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        if (lane == 0) smem[warp_id] = sum;
        __syncthreads();
        
        if (warp_id == 0) {
            sum = (lane < (blockDim.x + 31) / 32) ? smem[lane] : 0.0f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
        }
        __syncthreads();
        sum = smem[0];
        
        // 归一化
        float inv_sum = 1.0f / (sum + 1e-8f);
        for (int i = threadIdx.x; i < Ns; i += blockDim.x) {
            row[i] *= inv_sum;
        }
    }
}

/**
 * Element-wise Add with Broadcasting
 * A: (B, M, N, P)
 * B: (B, M, 1, P) broadcast along N dimension
 */
__global__ void add_bias_kernel(
    float* __restrict__ A,
    const float* __restrict__ B,
    int B_size, int M, int N, int P
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B_size * M * N * P;
    
    if (idx < total) {
        int p = idx % P;
        int m = (idx / (P * N)) % M;
        int b = idx / (P * N * M);
        
        int bias_idx = ((b * M + m) * 1 + 0) * P + p;
        A[idx] += B[bias_idx];
    }
}

/**
 * Reshape and Transpose helper
 * 将 (B, N, C) reshape到 (B, N, M, head_dim) 然后转置为 (B, M, N, head_dim)
 */
__global__ void reshape_transpose_kernel(
    const float* __restrict__ input,   // (B, N, C)
    float* __restrict__ output,        // (B, M, N, head_dim)
    int B, int N, int M, int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * M * N * head_dim;
    
    if (idx < total) {
        int d = idx % head_dim;
        int n = (idx / head_dim) % N;
        int m = (idx / (head_dim * N)) % M;
        int b = idx / (head_dim * N * M);
        
        // input: (B, N, M, head_dim)
        int in_idx = ((b * N + n) * M + m) * head_dim + d;
        output[idx] = input[in_idx];
    }
}

/**
 * Transpose back: (B, M, N, head_dim) -> (B, N, M, head_dim) -> (B, N, C)
 */
__global__ void transpose_reshape_kernel(
    const float* __restrict__ input,   // (B, M, N, head_dim)
    float* __restrict__ output,        // (B, N, C)
    int B, int N, int M, int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int C = M * head_dim;
    int total = B * N * C;
    
    if (idx < total) {
        int c = idx % C;
        int n = (idx / C) % N;
        int b = idx / (C * N);
        
        int m = c / head_dim;
        int d = c % head_dim;
        
        // input: (B, M, N, head_dim)
        int in_idx = ((b * M + m) * N + n) * head_dim + d;
        output[idx] = input[in_idx];
    }
}

// ==================== 主函数 ====================

size_t getDeformableAttentionWorkspaceSize(
    int batch_size,
    int num_tokens,
    int embed_dim,
    int num_heads,
    int num_sampling_points
) {
    const int G = 1;  // 固定为1
    const int head_dim = embed_dim / num_heads;
    const int H = 56, W = 56;
    size_t size = 0;
    
    // Workspace布局:
    // 1. offset_feat after dwconv: B * C * H * W * sizeof(float)
    size += batch_size * embed_dim * H * W * sizeof(float);
    
    // 2. offset_feat after pool: B * C * 7 * 7 * sizeof(float)
    size += batch_size * embed_dim * 7 * 7 * sizeof(float);
    
    // 3. offsets: B * 49 * G * 2 * sizeof(float)
    size += batch_size * 49 * G * 2 * sizeof(float);
    
    // 4. deformed_points: B * 49 * G * 2 * sizeof(float)
    size += batch_size * 49 * G * 2 * sizeof(float);
    
    // 5. sampled_features: B * C * 49 * G * sizeof(float)
    size += batch_size * embed_dim * 49 * G * sizeof(float);
    
    // 6. Q, K, V: B * N * C * sizeof(float) each
    size += 3 * batch_size * num_tokens * embed_dim * sizeof(float);
    
    // 7. attention: B * M * N * (Ns*G) * sizeof(float)
    size += batch_size * num_heads * num_tokens * (num_sampling_points * G) * sizeof(float);
    
    // 8. rpb: B * M * G * Ns * sizeof(float)
    size += batch_size * num_heads * G * num_sampling_points * sizeof(float);
    
    // 9. output_before_proj: B * N * C * sizeof(float)
    size += batch_size * num_tokens * embed_dim * sizeof(float);
    
    // ==== Tensor Core专用缓冲区 ====
    
    // 10-14: 每个Conv1x1/Linear层需要的缓冲区
    // 最大的操作是 QKV projection: (B*H*W, C) @ (C, 3*C)
    int max_gemm_size = batch_size * H * W * embed_dim * 3;  // 最大的GEMM
    
    // INT8量化缓冲区
    size += 2 * max_gemm_size * sizeof(int8_t);  // input_int8, weight_int8
    
    // FP32 reshape缓冲区
    size += 2 * max_gemm_size * sizeof(float);  // input_reshaped, output_temp
    
    // Scale缓冲区
    size += 16 * sizeof(float);  // 多个scale值
    
    // 15-18: Reshape后的Q, K, V缓冲区
    size += batch_size * num_heads * num_tokens * head_dim * sizeof(float);  // Q_reshaped
    size += batch_size * num_heads * (num_sampling_points * G) * head_dim * sizeof(float);  // K_reshaped
    size += batch_size * num_heads * (num_sampling_points * G) * head_dim * sizeof(float);  // V_reshaped
    size += batch_size * num_heads * num_tokens * head_dim * sizeof(float);  // output_temp
    
    // 19-20: Attention矩阵乘法的INT8缓冲区
    // Q @ K^T: (B*M, N, head_dim) @ (B*M, head_dim, Ns)
    size += batch_size * num_heads * num_tokens * head_dim * sizeof(int8_t);  // Q_int8 for attention
    size += batch_size * num_heads * (num_sampling_points * G) * head_dim * sizeof(int8_t);  // K_int8 for attention
    
    // 21: attn @ V的INT8缓冲区
    size += batch_size * num_heads * num_tokens * (num_sampling_points * G) * sizeof(int8_t);  // attn_int8
    size += batch_size * num_heads * (num_sampling_points * G) * head_dim * sizeof(int8_t);  // V_int8 for output
    
    return size;
}

cudaError_t launchDeformableAttention(
    const DeformableAttnParams& params,
    cudaStream_t stream
) {
    // 参数解包
    const int B = params.batch_size;
    const int N = params.num_tokens;
    const int C = params.embed_dim;
    const int M = params.num_heads;
    const int Ns = params.num_sampling_points;
    const int G = params.offset_groups;
    const int H = params.H, W = params.W;  // 从params获取
    const int head_dim = params.head_dim;
    
    // Workspace指针分配
    char* workspace_ptr = reinterpret_cast<char*>(params.workspace);
    size_t offset = 0;
    
    auto allocate = [&](size_t size) -> void* {
        void* ptr = workspace_ptr + offset;
        offset += size;
        return ptr;
    };
    
    // 1. offset_feat after dwconv
    float* offset_feat_dwconv = static_cast<float*>(allocate(B * C * H * W * sizeof(float)));
    
    // 2. offset_feat after pool
    float* offset_feat_pool = static_cast<float*>(allocate(B * C * 7 * 7 * sizeof(float)));
    
    // 3. offsets
    float* offsets = static_cast<float*>(allocate(B * 49 * G * 2 * sizeof(float)));
    
    // 4. deformed_points
    float* deformed_points = static_cast<float*>(allocate(B * 49 * G * 2 * sizeof(float)));
    
    // 5. sampled_features
    float* sampled_features = static_cast<float*>(allocate(B * C * Ns * G * sizeof(float)));
    
    // 6. Q, K, V
    float* Q = static_cast<float*>(allocate(B * N * C * sizeof(float)));
    float* K = static_cast<float*>(allocate(B * N * C * sizeof(float)));
    float* V = static_cast<float*>(allocate(B * N * C * sizeof(float)));
    
    // 7. attention weights
    float* attn = static_cast<float*>(allocate(B * M * N * Ns * sizeof(float)));
    
    // 8. rpb
    float* rpb = static_cast<float*>(allocate(B * M * G * Ns * sizeof(float)));
    
    // 9. output_before_proj
    float* output_before_proj = static_cast<float*>(allocate(B * N * C * sizeof(float)));
    
    // 10-14: Tensor Core专用缓冲区
    int max_gemm_size = B * H * W * C * 3;
    int8_t* gemm_input_int8 = static_cast<int8_t*>(allocate(max_gemm_size * sizeof(int8_t)));
    int8_t* gemm_weight_int8 = static_cast<int8_t*>(allocate(max_gemm_size * sizeof(int8_t)));
    float* gemm_input_reshaped = static_cast<float*>(allocate(max_gemm_size * sizeof(float)));
    float* gemm_output_temp = static_cast<float*>(allocate(max_gemm_size * sizeof(float)));
    float* scale_buf = static_cast<float*>(allocate(16 * sizeof(float)));
    
    // 临时缓冲区用于reshape
    float* Q_reshaped = static_cast<float*>(allocate(B * M * N * head_dim * sizeof(float)));
    float* K_reshaped = static_cast<float*>(allocate(B * M * Ns * G * head_dim * sizeof(float)));
    float* V_reshaped = static_cast<float*>(allocate(B * M * Ns * G * head_dim * sizeof(float)));
    float* output_attn_temp = static_cast<float*>(allocate(B * M * N * head_dim * sizeof(float)));
    
    // Attention矩阵乘法的INT8缓冲区
    int8_t* Q_int8_attn = static_cast<int8_t*>(allocate(B * M * N * head_dim * sizeof(int8_t)));
    int8_t* K_int8_attn = static_cast<int8_t*>(allocate(B * M * Ns * G * head_dim * sizeof(int8_t)));
    int8_t* attn_int8 = static_cast<int8_t*>(allocate(B * M * N * (Ns * G) * sizeof(int8_t)));
    int8_t* V_int8_attn = static_cast<int8_t*>(allocate(B * M * Ns * G * head_dim * sizeof(int8_t)));
    
    // ==================== Stage 1: Offset生成 ====================
    
    // 1.1 Depthwise Conv 3x3
    {
        int total = B * C * H * W;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        depthwise_conv3x3_kernel<<<blocks, threads, 0, stream>>>(
            params.input,              // (B, C, H, W)
            params.offset_dw_weight,   // (C, 1, 3, 3)
            offset_feat_dwconv,        // (B, C, H, W)
            B, C, H, W
        );
    }
    
    // 1.2 LayerNorm
    {
        dim3 blocks(N, B);
        int threads = 256;
        layer_norm_kernel<<<blocks, threads, 0, stream>>>(
            offset_feat_dwconv,  // reshape to (B, N, C)
            params.offset_norm_weight,
            params.offset_norm_bias,
            B, N, C
        );
    }
    
    // 1.3 GELU
    {
        int total = B * N * C;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        gelu_kernel<<<blocks, threads, 0, stream>>>(
            offset_feat_dwconv,
            total
        );
    }
    
    // 1.4 AdaptiveAvgPool 56x56 -> 7x7
    {
        int total = B * C * 7 * 7;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        adaptive_avgpool_56to7_kernel<<<blocks, threads, 0, stream>>>(
            offset_feat_dwconv,  // (B, C, 56, 56)
            offset_feat_pool,    // (B, C, 7, 7)
            B, C
        );
    }
    
    // 1.5 1x1 Conv生成offset (C -> 2*G) - 使用Tensor Core
    {
        int C_out = 2 * G;  // x, y offsets for each group
        
        conv1x1_tensor_core(
            offset_feat_pool,           // (B, C, 7, 7)
            params.offset_conv_weight,  // (2*G, C, 1, 1)
            offsets,                    // (B, 2*G, 7, 7) -> (B, 49, G, 2)
            B, C, C_out, 7, 7,
            gemm_input_reshaped,
            gemm_input_int8,
            gemm_weight_int8,
            gemm_output_temp,
            scale_buf,
            stream
        );
    }
    
    // 1.6 生成deformed points
    {
        int total = B * 49 * G * 2;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        generate_deformed_points_kernel<<<blocks, threads, 0, stream>>>(
            offsets,               // (B, 49, G, 2)
            params.reference_points,  // (1, 49, G, 2)
            deformed_points,       // (B, 49, G, 2)
            B
        );
    }
    
    // ==================== Stage 2: 特征采样 ====================
    
    // 2.1 Grid Sample
    {
        int total = B * C * Ns * G;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        grid_sample_kernel<<<blocks, threads, 0, stream>>>(
            params.input,      // (B, C, H, W)
            deformed_points,   // (B, Ns*G, 2)
            sampled_features,  // (B, C, Ns*G)
            B, C, H, W, Ns, G
        );
    }
    
    // 2.2 Sample RPB
    {
        int total = B * M * Ns * G;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        sample_rpb_kernel<<<blocks, threads, 0, stream>>>(
            params.rpb_table,     // (M, 13, 13)
            deformed_points,      // (B, Ns, G, 2)
            rpb,                  // (B, M, G, Ns)
            B, M, Ns, G
        );
    }
    
    // ==================== Stage 3: Q, K, V计算 - 全部使用Tensor Core ====================
    
    // 3.1 计算 Q = input @ W_q - 使用Tensor Core GEMM
    {
        conv1x1_tensor_core(
            params.input,      // (B, C, H, W)
            params.q_weight,   // (C, C, 1, 1) - Q权重
            Q,                 // (B, C, H, W) -> (B, N, C)
            B, C, C, H, W,
            gemm_input_reshaped,
            gemm_input_int8,
            gemm_weight_int8,
            gemm_output_temp,
            scale_buf,
            stream
        );
        
        // Reshape Q: (B, N, C) -> (B, M, N, head_dim)
        int total = B * M * N * head_dim;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        reshape_transpose_kernel<<<blocks, threads, 0, stream>>>(
            Q, Q_reshaped, B, N, M, head_dim
        );
    }
    
    // 3.2 计算 K (从sampled_features) - 使用Tensor Core GEMM
    {
        conv1x1_tensor_core(
            sampled_features,                  // (B, C, Ns*G, 1) 视为 (B, C, Ns*G, 1)
            params.k_weight,                   // K权重 (C, C, 1, 1)
            K,                                 // (B, C, Ns*G, 1)
            B, C, C, Ns * G, 1,
            gemm_input_reshaped,
            gemm_input_int8,
            gemm_weight_int8,
            gemm_output_temp,
            scale_buf,
            stream
        );
        
        // Reshape K: (B, Ns*G, C) -> (B, M, Ns*G, head_dim)
        int total = B * M * (Ns * G) * head_dim;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        reshape_transpose_kernel<<<blocks, threads, 0, stream>>>(
            K, K_reshaped, B, Ns * G, M, head_dim
        );
    }
    
    // 3.3 计算 V (从sampled_features) - 使用Tensor Core GEMM
    {
        conv1x1_tensor_core(
            sampled_features,                  // (B, C, Ns*G, 1)
            params.v_weight,                   // V权重 (C, C, 1, 1)
            V,                                 // (B, C, Ns*G, 1)
            B, C, C, Ns * G, 1,
            gemm_input_reshaped,
            gemm_input_int8,
            gemm_weight_int8,
            gemm_output_temp,
            scale_buf,
            stream
        );
        
        // Reshape V: (B, Ns*G, C) -> (B, M, Ns*G, head_dim)
        int total = B * M * (Ns * G) * head_dim;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        reshape_transpose_kernel<<<blocks, threads, 0, stream>>>(
            V, V_reshaped, B, Ns * G, M, head_dim
        );
    }
    
    // ==================== Stage 4: Attention计算 - 使用Tensor Core ====================
    
    // 4.1 计算 Q @ K^T / sqrt(head_dim) - 使用Tensor Core GEMM
    {
        // attn = Q @ K^T / sqrt(head_dim)
        // Q_reshaped: (B, M, N, head_dim) -> 视为 (B*M, N, head_dim)
        // K_reshaped: (B, M, Ns*G, head_dim) -> 视为 (B*M, Ns*G, head_dim)
        // 需要计算 (B*M, N, head_dim) @ (B*M, head_dim, Ns*G) = (B*M, N, Ns*G)
        
        int batch_gemm = B * M;
        int M_gemm = N;
        int N_gemm = Ns * G;
        int K_gemm = head_dim;
        float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
        
        int threads = 256;
        
        // 量化Q和K
        {
            compute_scale_kernel<<<1, threads, 0, stream>>>(
                Q_reshaped, &scale_buf[0], batch_gemm * M_gemm * K_gemm
            );
            compute_scale_kernel<<<1, threads, 0, stream>>>(
                K_reshaped, &scale_buf[1], batch_gemm * N_gemm * K_gemm
            );
            
            float scales[2];
            cudaMemcpyAsync(scales, scale_buf, 2 * sizeof(float), 
                           cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            
            int blocks_q = (batch_gemm * M_gemm * K_gemm + threads - 1) / threads;
            int blocks_k = (batch_gemm * N_gemm * K_gemm + threads - 1) / threads;
            
            quantize_fp32_to_int8_kernel<<<blocks_q, threads, 0, stream>>>(
                Q_reshaped, Q_int8_attn, scales[0], batch_gemm * M_gemm * K_gemm
            );
            quantize_fp32_to_int8_kernel<<<blocks_k, threads, 0, stream>>>(
                K_reshaped, K_int8_attn, scales[1], batch_gemm * N_gemm * K_gemm
            );
            
            // Batch GEMM循环
            for (int bm = 0; bm < batch_gemm; bm++) {
                tensor_core::gemm_int8_fp32(
                    Q_int8_attn + bm * M_gemm * K_gemm,    // (M_gemm, K_gemm)
                    K_int8_attn + bm * N_gemm * K_gemm,    // (K_gemm, N_gemm) - 转置
                    attn + bm * M_gemm * N_gemm,           // (M_gemm, N_gemm)
                    M_gemm, N_gemm, K_gemm,
                    scale, 0.0f,
                    scales[0], scales[1],
                    stream
                );
            }
        }
    }
    
    // 4.2 加上RPB
    {
        int total = B * M * N * (Ns * G);
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        
        add_bias_kernel<<<blocks, threads, 0, stream>>>(
            attn,  // (B, M, N, Ns*G)
            rpb,   // (B, M, 1, Ns*G)
            B, M, N, Ns * G
        );
    }
    
    // 4.3 Softmax
    {
        int blocks = B * M * N;
        int threads = 256;
        softmax_kernel<<<blocks, threads, 0, stream>>>(
            attn,  // (B, M, N, Ns*G)
            B, M, N, Ns * G
        );
    }
    
    // ==================== Stage 5: 输出计算 - 使用Tensor Core ====================
    
    // 5.1 attn @ V - 使用Tensor Core GEMM
    {
        // output = attn @ V
        // attn: (B*M, N, Ns*G)
        // V_reshaped: (B*M, Ns*G, head_dim)
        // output: (B*M, N, head_dim)
        
        int batch_gemm = B * M;
        int M_gemm = N;
        int K_gemm = Ns * G;
        int N_gemm = head_dim;
        
        int threads = 256;
        
        // 量化attn和V
        {
            compute_scale_kernel<<<1, threads, 0, stream>>>(
                attn, &scale_buf[0], batch_gemm * M_gemm * K_gemm
            );
            compute_scale_kernel<<<1, threads, 0, stream>>>(
                V_reshaped, &scale_buf[1], batch_gemm * K_gemm * N_gemm
            );
            
            float scales[2];
            cudaMemcpyAsync(scales, scale_buf, 2 * sizeof(float), 
                           cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            
            int blocks_attn = (batch_gemm * M_gemm * K_gemm + threads - 1) / threads;
            int blocks_v = (batch_gemm * K_gemm * N_gemm + threads - 1) / threads;
            
            quantize_fp32_to_int8_kernel<<<blocks_attn, threads, 0, stream>>>(
                attn, attn_int8, scales[0], batch_gemm * M_gemm * K_gemm
            );
            quantize_fp32_to_int8_kernel<<<blocks_v, threads, 0, stream>>>(
                V_reshaped, V_int8_attn, scales[1], batch_gemm * K_gemm * N_gemm
            );
            
            // Batch GEMM循环
            for (int bm = 0; bm < batch_gemm; bm++) {
                tensor_core::gemm_int8_fp32(
                    attn_int8 + bm * M_gemm * K_gemm,      // (M_gemm, K_gemm)
                    V_int8_attn + bm * K_gemm * N_gemm,    // (K_gemm, N_gemm)
                    output_attn_temp + bm * M_gemm * N_gemm, // (M_gemm, N_gemm)
                    M_gemm, N_gemm, K_gemm,
                    1.0f, 0.0f,
                    scales[0], scales[1],
                    stream
                );
            }
        }
        
        // Transpose back: (B, M, N, head_dim) -> (B, N, C)
        int total = B * N * C;
        int blocks = (total + threads - 1) / threads;
        transpose_reshape_kernel<<<blocks, threads, 0, stream>>>(
            output_attn_temp, output_before_proj, B, N, M, head_dim
        );
    }
    
    // 5.2 Output Projection - 使用Tensor Core GEMM
    {
        conv1x1_tensor_core(
            output_before_proj,     // (B, N, C) 视为 (B, C, N, 1)
            params.out_weight,      // (C, C, 1, 1)
            params.output,          // (B, C, N, 1)
            B, C, C, N, 1,
            gemm_input_reshaped,
            gemm_input_int8,
            gemm_weight_int8,
            gemm_output_temp,
            scale_buf,
            stream
        );
    }
    
    // 5.3 Residual + LayerNorm (如果需要)
    {
        dim3 blocks(N, B);
        int threads = 256;
        layer_norm_kernel<<<blocks, threads, 0, stream>>>(
            params.output,       // (B, N, C)
            params.offset_norm_weight,
            params.offset_norm_bias,
            B, N, C
        );
    }
    
    return cudaGetLastError();
}
