# Makefile for Deformable Attention CUDA Implementation
# Pure CUDA version without PyTorch dependencies

# Compiler and flags
NVCC = nvcc
CUDA_ARCH ?= sm_86
CUDA_FLAGS = -arch=$(CUDA_ARCH) --cudart shared -O3 -Xcompiler -fPIC
INCLUDES = -I./src

# Directories
SRC_DIR = src
BIN_DIR = bin
OBJ_DIR = obj

# Source files
SOURCES = $(SRC_DIR)/test.cu \
          $(SRC_DIR)/deform_attn/cuda/ms_deform_attn_cuda_kernel.cu \
          $(SRC_DIR)/deform_attn/cuda/ms_deform_attn_predict_wrapper.cu

# Target executable
TARGET = $(BIN_DIR)/test

# Create directories
$(shell mkdir -p $(BIN_DIR) $(OBJ_DIR))

# Default target
all: $(TARGET)

# Compile the main executable
$(TARGET): $(SOURCES)
	@echo "编译 Deformable Attention 测试程序..."
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) $(SOURCES) -o $(TARGET)
	@echo "编译完成: $(TARGET)"

# Clean build artifacts
clean:
	@echo "清理编译产物..."
	rm -rf $(BIN_DIR) $(OBJ_DIR) 
	@echo "清理完成"

# Run the test
run: $(TARGET)
	@echo "运行测试..."
	./$(TARGET)

# Help
help:
	@echo "Deformable Attention Makefile"
	@echo ""
	@echo "使用方法:"
	@echo "  make           - 编译项目"
	@echo "  make clean     - 清理编译产物"
	@echo "  make run       - 编译并运行测试"
	@echo "  make help      - 显示此帮助信息"
	@echo ""
	@echo "变量:"
	@echo "  CUDA_ARCH      - CUDA架构 (默认: sm_86)"
	@echo ""
	@echo "示例:"
	@echo "  make CUDA_ARCH=sm_75    - 为SM75架构编译"
	@echo "  make clean && make      - 完全重新编译"

.PHONY: all clean run help
