#!/bin/bash

# DeformableAttention CUDA 构建脚本
# 目标: RTX 3070 (sm_86)

set -e

echo "========================================="
echo "DeformableAttention CUDA Build Script"
echo "Target: RTX 3070 (Ampere sm_86)"
echo "========================================="
echo ""

# 检查CUDA
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA (nvcc) not found!"
    echo "Please install CUDA Toolkit first."
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
echo "CUDA Version: $CUDA_VERSION"
echo ""

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    echo "Available GPUs:"
    nvidia-smi --list-gpus
    echo ""
fi

# 选择构建方式
echo "Select build method:"
echo "1) CMake (recommended)"
echo "2) Makefile"
read -p "Choice [1/2]: " choice

case $choice in
    1)
        echo ""
        echo "Building with CMake..."
        mkdir -p build
        cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release
        make -j$(nproc)
        echo ""
        echo "Build completed!"
        echo "Executable: build/bin/test_deformable_attn"
        echo ""
        echo "Run test:"
        echo "  cd build && ./bin/test_deformable_attn"
        ;;
    2)
        echo ""
        echo "Building with Makefile..."
        make clean
        make -j$(nproc)
        echo ""
        echo "Build completed!"
        echo "Executable: bin/test_deformable_attn"
        echo ""
        echo "Run test:"
        echo "  ./bin/test_deformable_attn"
        ;;
    *)
        echo "Invalid choice!"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "Additional commands:"
echo "  make ptx     - Generate PTX assembly"
echo "  make sass    - Generate SASS assembly"
echo "  make run     - Run test directly"
echo "========================================="
