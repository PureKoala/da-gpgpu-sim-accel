# Makefile for Deformable Attention CUDA kernels
# Target: RTX 3070 (Ampere sm_86)

NVCC := nvcc
CXX := g++

# CUDA架构
ARCH := -gencode arch=compute_86,code=sm_86

# 编译标志
NVCCFLAGS := $(ARCH) -O3 --use_fast_math --expt-relaxed-constexpr -Xptxas -v
CXXFLAGS := -O3 -std=c++17

# 包含目录
INCLUDES := -I./include -I$(CUDA_PATH)/include

# 源文件
KERNEL_SRCS := kernels/tensor_core_gemm.cu kernels/deformable_attention.cu
TEST_SRCS := tests/test_main.cu

# 目标文件
KERNEL_OBJS := $(KERNEL_SRCS:.cu=.o)
TEST_OBJS := $(TEST_SRCS:.cu=.o)

# 输出
LIB := lib/libdeformable_attn_cuda.a
TEST_BIN := bin/test_deformable_attn

.PHONY: all clean lib test

all: lib test

lib: $(LIB)

test: $(TEST_BIN)

$(LIB): $(KERNEL_OBJS)
	@mkdir -p lib
	ar rcs $@ $^

$(TEST_BIN): $(TEST_OBJS) $(LIB)
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $^ -o $@ -L./lib -ldeformable_attn_cuda

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -dc $< -o $@

clean:
	rm -f $(KERNEL_OBJS) $(TEST_OBJS) $(LIB) $(TEST_BIN)
	rm -rf lib bin

# 运行测试
run: $(TEST_BIN)
	./$(TEST_BIN)

# 查看PTX和SASS
ptx: kernels/tensor_core_gemm.cu
	$(NVCC) $(ARCH) --ptx $< -o kernels/tensor_core_gemm.ptx

sass: kernels/tensor_core_gemm.cu
	$(NVCC) $(ARCH) -cubin $< -o kernels/tensor_core_gemm.cubin
	cuobjdump -sass kernels/tensor_core_gemm.cubin > kernels/tensor_core_gemm.sass

.PHONY: ptx sass
