cmake_minimum_required(VERSION 3.18)
project(DeformableAttentionCUDA LANGUAGES CXX CUDA)

# CUDA设置
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 3070 (Ampere sm_86)

# CUDA编译选项
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -gencode arch=compute_86,code=sm_86
    --use_fast_math
    --expt-relaxed-constexpr
    -O3
    -Xptxas -v
)

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)

# CUTLASS路径 (需要根据实际安装位置调整)
# set(CUTLASS_DIR "/path/to/cutlass")
# include_directories(${CUTLASS_DIR}/include)

# 源文件
set(KERNEL_SOURCES
    kernels/tensor_core_gemm.cu
    kernels/deformable_attention.cu
)

# 编译为静态库
add_library(deformable_attn_cuda STATIC ${KERNEL_SOURCES})

# 设置CUDA属性
set_target_properties(deformable_attn_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# 测试程序
add_executable(test_deformable_attn tests/test_main.cu)
target_link_libraries(test_deformable_attn deformable_attn_cuda)

# 设置输出目录
set_target_properties(deformable_attn_cuda PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

set_target_properties(test_deformable_attn PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 打印信息
message(STATUS "CUDA Version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
