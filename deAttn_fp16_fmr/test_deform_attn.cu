/*
 * Deformable Attention Test Kernel
 *
 * This file demonstrates how to use the Deformable Attention functions
 * in CUDA code. These functions will be intercepted by GPGPU-Sim and
 * simulated with fixed latency.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// ============================================================================
// Deformable Attention Function Prototypes
// These functions are intercepted by GPGPU-Sim
// ============================================================================

extern "C" __device__ void __deform_pcb(
    float* weights,
    float threshold,
    bool enable,
    bool* mask
);

extern "C" __device__ void __deform_tbc(
    float* abs_coords,
    bool* valid_coords,
    int* mode,
    int* tile_size,
    int* base_tile_x,
    int* base_tile_y
);

extern "C" __device__ void __deform_tma(
    float* tile_data,
    unsigned long long gmem_base,
    unsigned long long smem_base,
    int tile_x,
    int tile_y,
    int pitch,
    int mode,
    int tile_size
);

extern "C" __device__ void __deform_interp(
    float* output,
    float* tile_data,
    float* coords,
    float* weights
);

// ============================================================================
// Test Kernel: Simple Deformable Attention
// ============================================================================

__global__ void test_deform_attn_simple(
    float* output,
    float* weights,
    float* coords,
    float* tile_data,
    float threshold,
    int width,
    int height
) {
    extern __shared__ float smem[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Only lane 0 executes the DeformAttn operations
    if (lane_id == 0) {
        // Step 1: PCB - Pre-Check Block
        // 权重过滤，生成稀疏掩码
        bool mask[16 * 16];
        __deform_pcb(weights, threshold, true, mask);

        // Step 2: TBC - Tile Boundary Check
        // 边界检查 + 模式检测
        int mode, tile_size, base_x, base_y;
        bool valid[16 * 16];
        __deform_tbc(coords, valid, &mode, &tile_size, &base_x, &base_y);

        // Step 3: TMA - Tile Memory Access
        // Tile 加载（GMEM → SMEM）
        unsigned long long gmem_base = (unsigned long long)tile_data;
        unsigned long long smem_base = (unsigned long long)smem;
        __deform_tma(tile_data, gmem_base, smem_base,
                     base_x, base_y, width, mode, tile_size);

        // Step 4: Interpolation
        // 双线性插值计算
        __deform_interp(output, tile_data, coords, weights);
    }
}

// ============================================================================
// Test Kernel: Deformable Attention with Multiple Queries
// ============================================================================

__global__ void test_deform_attn_multi_query(
    float* output,
    float* weights,
    float* coords,
    float* tile_data,
    float threshold,
    int num_queries,
    int width,
    int height
) {
    extern __shared__ float smem[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // 每个 warp 处理一个 query tile
    int query_tile_id = blockIdx.x * blockDim.x / 32 + warp_id;

    if (query_tile_id < num_queries && lane_id == 0) {
        // 计算当前 query tile 的偏移
        int tile_offset = query_tile_id * 16 * 16;

        // Step 1: PCB
        bool mask[16 * 16];
        __deform_pcb(weights + tile_offset, threshold, true, mask);

        // Step 2: TBC
        int mode, tile_size, base_x, base_y;
        bool valid[16 * 16];
        __deform_tbc(coords + tile_offset * 2, valid, &mode, &tile_size, &base_x, &base_y);

        // Step 3: TMA
        unsigned long long gmem_base = (unsigned long long)(tile_data + tile_offset);
        unsigned long long smem_base = (unsigned long long)smem;
        __deform_tma(tile_data + tile_offset, gmem_base, smem_base,
                     base_x, base_y, width, mode, tile_size);

        // Step 4: Interpolation
        __deform_interp(output + tile_offset, tile_data + tile_offset,
                        coords + tile_offset * 2, weights + tile_offset);
    }
}

// ============================================================================
// Host Helper Functions
// ============================================================================

// Initialize test data
void init_test_data(float* weights, float* coords, float* tile_data,
                    int width, int height, float threshold) {
    // Initialize weights (random values)
    for (int i = 0; i < 16 * 16; i++) {
        weights[i] = (float)rand() / RAND_MAX;
    }

    // Initialize coordinates (within valid range)
    for (int i = 0; i < 16 * 16; i++) {
        coords[i * 2] = (float)(rand() % width);     // x
        coords[i * 2 + 1] = (float)(rand() % height); // y
    }

    // Initialize tile data
    for (int i = 0; i < 16 * 16; i++) {
        tile_data[i] = (float)rand() / RAND_MAX;
    }
}

// Run test
void run_test() {
    printf("=== Deformable Attention Test ===\n");

    // Configuration
    const int width = 32;
    const int height = 32;
    const float threshold = 0.5f;
    const int num_queries = 1;

    // Allocate host memory
    float* h_weights = new float[16 * 16];
    float* h_coords = new float[16 * 16 * 2];
    float* h_tile_data = new float[16 * 16];
    float* h_output = new float[16 * 16];

    // Initialize test data
    init_test_data(h_weights, h_coords, h_tile_data, width, height, threshold);

    // Allocate device memory
    float* d_weights;
    float* d_coords;
    float* d_tile_data;
    float* d_output;

    cudaMalloc(&d_weights, sizeof(float) * 16 * 16);
    cudaMalloc(&d_coords, sizeof(float) * 16 * 16 * 2);
    cudaMalloc(&d_tile_data, sizeof(float) * 16 * 16);
    cudaMalloc(&d_output, sizeof(float) * 16 * 16);

    // Copy data to device
    cudaMemcpy(d_weights, h_weights, sizeof(float) * 16 * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_coords, h_coords, sizeof(float) * 16 * 16 * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tile_data, h_tile_data, sizeof(float) * 16 * 16, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block(32);  // 32 threads (1 warp)
    dim3 grid(1);

    size_t shared_mem_size = 16 * 16 * sizeof(float);  // SMEM for tile data

    printf("Launching test kernel...\n");
    test_deform_attn_simple<<<grid, block, shared_mem_size>>>(
        d_output, d_weights, d_coords, d_tile_data, threshold, width, height
    );

    cudaDeviceSynchronize();
    printf("Kernel completed\n");

    // Copy result back
    cudaMemcpy(h_output, d_output, sizeof(float) * 16 * 16, cudaMemcpyDeviceToHost);

    // Print some results
    printf("First 10 output values:\n");
    for (int i = 0; i < 10; i++) {
        printf("  output[%d] = %.4f\n", i, h_output[i]);
    }

    // Cleanup
    delete[] h_weights;
    delete[] h_coords;
    delete[] h_tile_data;
    delete[] h_output;

    cudaFree(d_weights);
    cudaFree(d_coords);
    cudaFree(d_tile_data);
    cudaFree(d_output);

    printf("Test completed successfully!\n");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("Deformable Attention Test Program\n");
    printf("==================================\n\n");

    // Check CUDA device
    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("CUDA devices found: %d\n", device_count);

    if (device_count == 0) {
        printf("No CUDA devices found. Running in simulation mode.\n");
    }

    // Run test
    run_test();

    return 0;
}