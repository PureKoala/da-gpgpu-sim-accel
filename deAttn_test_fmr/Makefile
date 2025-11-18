CUDA_HOME := /usr/local/cuda-11.3
NVCC := $(CUDA_HOME)/bin/nvcc

# 使用 -ptx 标志来生成 PTX，而不是可执行文件
# 使用 -O0 (关闭优化) 来确保 nvcc 不会优化掉您的代理函数
all: test.ptx

test.ptx: src/test.cu
	$(NVCC) -arch=sm_86 -ptx -O0 -std=c++11 -I$(CUDA_HOME)/include src/test.cu -o test.ptx

# 一个用于运行 GPGPU-Sim 的目标
run: test.ptx
	../configs/tested-cfgs/SM86_RTX3070 -config $(CONFIG_FILE) -entry test_fmr_kernel test.ptx

clean:
	rm -f *.ptx *.o