#include <cuda_runtime.h> // 包含 CUDA 运行时 API
#include <cuda_fp16.h>    // 包含 half (fp16) 类型支持
#include <stdio.h>        // 用于 C 语言的 printf
#include <stdint.h>       // 用于 uint64_t 和 uintptr_t
#include <stdlib.h>       // 用于 getenv, atoi, strtol, rand, malloc, free

// 使用宏来禁用编译器的内联和优化
#define PREVENT_INLINE __attribute__((noinline))

/*
========================================================================
  FMR Tile Loader 代理函数
========================================================================
*/
extern "C" __device__ PREVENT_INLINE
void __fmr_sample(
    unsigned long long gmem_base_addr, 
    unsigned long long smem_base_addr, 
    int width, 
    int height,
    int stride)  // 源图像的stride（每行元素数）
{
    // ====================================================================
    // 关键的防优化汇编 (FMR Load Proxy)
    // 确保 GPGPU-Sim 可以拦截所有五个参数
    // ====================================================================
    asm volatile(
        "// FMR Load Proxy: Forcing all parameters to be 'live' for GPGPU-Sim."
        :   // 输出列表 (空)
        :   // 输入列表
          "l"(gmem_base_addr), 
          "l"(smem_base_addr), 
          "r"(width), 
          "r"(height),
          "r"(stride)
        : "memory" // 破坏列表
    );
}


/*
========================================================================
  [精简后的 Kernel]
  仅用于测试 FMR Tile Loader 代理
========================================================================
*/
__global__ void test_fmr_kernel(
    const half* feature_map,   // [GMEM] 输入的完整特征图
    int feature_width,         // 完整特征图的宽度 (stride)
    int tile_width,            // 瓦块宽度
    int tile_height)           // 瓦块高度
{
    // 声明 __shared__ 内存用于 FMR 重组
    extern __shared__ half smem_feature[];
    
    // 1. 获取 SMEM 基地址
    unsigned long long smem_addr = (unsigned long long)smem_feature;

    // 2. 获取 GMEM 基地址
    // GPGPU-Sim 拦截器应使用 blockIdx.x, feature_width 等
    // 来计算此块的实际 GMEM 瓦块地址
    unsigned long long gmem_addr = (unsigned long long)feature_map;

    // 3. 让 0 号线程调用加载代理函数
    if (threadIdx.x == 0 && threadIdx.y == 0) 
    {
        if (blockIdx.x == 0) { // 仅让第一个块打印
             printf("ThreadBlock 0 calling FMR Load Proxy:\n");
             printf("  GMEM Addr (base): %llu\n", gmem_addr);
             printf("  SMEM Addr (base): %llu\n", smem_addr);
             printf("  Tile Width:     %d\n", tile_width);
             printf("  Tile Height:    %d\n", tile_height);
             printf("  Source Stride:  %d\n", feature_width);
        }

        // 调用 FMR 加载代理函数（支持strided访问）
        __fmr_sample(gmem_addr, smem_addr, tile_width, tile_height, feature_width);
    }
}


// ========================================================================
// Host (CPU) 代码 [精简版]
// ========================================================================

int main()
{
    // --- 1. 定义参数 ---
    int block_dim_x    = 32;
    int block_dim_y    = 32;
    int feature_width  = 128;
    int feature_height = 128;
    int tile_width     = 32;
    int tile_height    = 32;
    int grid_size      = 4; // 默认启动 4 个线程块

    const char* env = NULL;
    if ((env = getenv("BLOCK_DIM_X")))    block_dim_x    = atoi(env);
    if ((env = getenv("BLOCK_DIM_Y")))    block_dim_y    = atoi(env);
    if ((env = getenv("FEATURE_WIDTH")))  feature_width  = atoi(env);
    if ((env = getenv("FEATURE_HEIGHT"))) feature_height = atoi(env);
    if ((env = getenv("TILE_WIDTH")))     tile_width     = atoi(env);
    if ((env = getenv("TILE_HEIGHT")))    tile_height    = atoi(env);
    if ((env = getenv("GRID_SIZE")))      grid_size      = atoi(env);


    // 简单有效性检查与修正
    if (block_dim_x <= 0) block_dim_x = 32;
    if (block_dim_y <= 0) block_dim_y = 32;
    if (feature_width  <= 0) feature_width  = 128;
    if (feature_height <= 0) feature_height = 128;
    if (tile_width     <= 0) tile_width     = 32;
    if (tile_height    <= 0) tile_height    = 32;
    if (grid_size      <= 0) grid_size      = 4;

    int block_size = block_dim_x * block_dim_y;
    
    // 动态分配的 __shared__ 内存大小 (in bytes)
    size_t smem_size = tile_width * tile_height * sizeof(half);

    printf("启动 FMR [Loader] 测试...\n");
    printf("Grid Size: %d, Block Size: %d, Shared Mem: %zu bytes\n", grid_size, block_size, smem_size);
    printf("Test Params: FeatW=%d, FeatH=%d, TileW=%d, TileH=%d\n",
        feature_width, feature_height, tile_width, tile_height);


    // --- 2. 分配 Host 内存 (仅需 feature map) ---
    half* h_feature = (half*)malloc(feature_width * feature_height * sizeof(half));
    if (!h_feature) {
        fprintf(stderr, "Host malloc failed!\n");
        return 1;
    }
    printf("[TEST] 分配 Host 内存完成 \n");

    // --- 3. 初始化 Host 数据 (伪数据) ---
    for (int i = 0; i < feature_width * feature_height; ++i) {
        h_feature[i] = __float2half((float)i); // 用索引值填充
    }
    printf("[TEST] 初始化 Host 数据完成 \n");

    // --- 4. 分配 Device 内存 (仅需 feature map) ---
    half* d_feature;
    cudaError_t err = cudaMalloc(&d_feature, feature_width * feature_height * sizeof(half));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("[TEST] 分配 Device 内存完成 \n");

    // --- 5. 拷贝 Host -> Device ---
    cudaMemcpy(d_feature, h_feature, feature_width * feature_height * sizeof(half), cudaMemcpyHostToDevice);
    printf("[TEST] 拷贝 Host -> Device 完成 \n");

    printf("[TEST] 启动 FMR [Loader] 测试...\n");
    // --- 6. 启动 Kernel ---
    dim3 gridDim(grid_size, 1, 1);
    dim3 blockDim(block_dim_x, block_dim_y, 1);
    
    test_fmr_kernel<<<gridDim, blockDim, smem_size>>>(
        d_feature,
        feature_width,
        tile_width,
        tile_height
    );
    
    cudaDeviceSynchronize(); // 等待 Kernel 完成
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    printf("[TEST] FMR [Loader] 测试完成\n");

    // --- 7. [已删除] 拷贝 Device -> Host ---
    // --- 8. [已删除] 验证结果 ---
    printf("Kernel 执行完毕。\n");

    // --- 9. 释放内存 ---
    free(h_feature);
    cudaFree(d_feature);

    return 0;
}