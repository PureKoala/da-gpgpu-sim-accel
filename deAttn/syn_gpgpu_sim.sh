#!/bin/bash
# GPGPU-Sim Source Code Recompilation Script
# Use this script to quickly recompile GPGPU-Sim after modifying source code

# Configuration parameters
BUILD_TYPE=release  # release or debug
CLEAN_BUILD=0       # 1=clean build; 0=incremental build
JOBS=16             # number of parallel jobs for compilation (-j parameter)

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# GPGPU-Sim root directory
GPGPUSIM_ROOT=$(cd "$(dirname "$0")/.." && pwd)

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}GPGPU-Sim Recompilation Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "GPGPU-Sim Path: ${GPGPUSIM_ROOT}"
echo -e "Build Type: ${BUILD_TYPE}"
echo -e "Parallel Jobs: ${JOBS}"
echo ""

# Check GPGPU-Sim directory
if [ ! -d "${GPGPUSIM_ROOT}" ]; then
    echo -e "${RED}Error: GPGPU-Sim directory not found${NC}"
    exit 1
fi

# Check Makefile
if [ ! -f "${GPGPUSIM_ROOT}/Makefile" ]; then
    echo -e "${RED}Error: Makefile not found${NC}"
    exit 1
fi

cd "${GPGPUSIM_ROOT}" || exit 1

# Setup CUDA path (auto-detect)
if [ -z "$CUDA_INSTALL_PATH" ]; then
    if command -v nvcc &> /dev/null; then
        export CUDA_INSTALL_PATH=$(dirname $(dirname $(which nvcc)))
        echo -e "${YELLOW}Auto-detected CUDA: ${CUDA_INSTALL_PATH}${NC}"
    else
        echo -e "${RED}Error: nvcc not found, set CUDA_INSTALL_PATH${NC}"
        exit 1
    fi
else
    echo -e "Using CUDA: ${CUDA_INSTALL_PATH}"
fi

# Load GPGPU-Sim environment
echo -e "${YELLOW}Loading GPGPU-Sim environment...${NC}"
if [ -f "${GPGPUSIM_ROOT}/setup_environment" ]; then
    source "${GPGPUSIM_ROOT}/setup_environment" ${BUILD_TYPE}
else
    echo -e "${RED}Error: setup_environment not found${NC}"
    exit 1
fi

# Clean build if needed
if [ ${CLEAN_BUILD} -eq 1 ]; then
    echo -e "${YELLOW}Cleaning build...${NC}"
    make clean
    if [ $? -ne 0 ]; then
        echo -e "${RED}Clean failed${NC}"
        exit 1
    fi
    echo -e "${GREEN}Clean completed${NC}"
    echo ""
fi

# Start compilation
echo -e "${YELLOW}Compiling GPGPU-Sim...${NC}"
echo -e "${YELLOW}Command: make -j${JOBS}${NC}"
echo ""

START_TIME=$(date +%s)

make -j${JOBS}

MAKE_STATUS=$?
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))

echo ""
echo -e "${GREEN}========================================${NC}"

if [ ${MAKE_STATUS} -eq 0 ]; then
    echo -e "${GREEN}✓ Compilation successful!${NC}"
    echo -e "${GREEN}Time elapsed: ${ELAPSED_TIME} seconds${NC}"
    echo ""
    echo -e "Library files location:"
    echo -e "  ${GPGPUSIM_ROOT}/lib/"
    echo ""
    echo -e "Usage:"
    echo -e "  1. Navigate to your application directory"
    echo -e "  2. Run: source ${GPGPUSIM_ROOT}/setup_environment ${BUILD_TYPE}"
    echo -e "  3. Execute your CUDA program"
else
    echo -e "${RED}✗ Compilation failed!${NC}"
    echo -e "${RED}Please check the error messages above${NC}"
    exit 1
fi

echo -e "${GREEN}========================================${NC}"

# Show compiled libraries
if [ -d "${GPGPUSIM_ROOT}/lib" ]; then
    echo ""
    echo -e "${YELLOW}Compiled libraries:${NC}"
    find "${GPGPUSIM_ROOT}/lib" -name "*.so*" -type f 2>/dev/null | head -n 10
fi

exit 0
