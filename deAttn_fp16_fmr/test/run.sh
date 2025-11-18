#!/bin/bash

# FMR Test Run Script
# This script sets up the environment and runs the FMR test with GPGPU-Sim

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_msg() {
    local color=$1
    shift
    echo -e "${color}$@${NC}"
}

print_msg $BLUE "=========================================="
print_msg $BLUE "  FMR Bilinear Sampling Test"
print_msg $BLUE "=========================================="
echo

# Step 1: Check GPGPU-Sim setup
print_msg $YELLOW "[1/5] Checking GPGPU-Sim setup..."

if [ -z "$GPGPUSIM_ROOT" ]; then
    export GPGPUSIM_ROOT=$(cd ../.. && pwd)
    print_msg $YELLOW "  Setting GPGPUSIM_ROOT=$GPGPUSIM_ROOT"
fi

if [ ! -f "$GPGPUSIM_ROOT/setup_environment" ]; then
    print_msg $RED "  Error: GPGPU-Sim setup_environment not found!"
    print_msg $RED "  Please set GPGPUSIM_ROOT or run from correct directory"
    exit 1
fi

# Source GPGPU-Sim environment
source $GPGPUSIM_ROOT/setup_environment
print_msg $GREEN "  ✓ GPGPU-Sim environment loaded"

# Step 2: Check configuration files
print_msg $YELLOW "[2/5] Checking configuration files..."

if [ ! -f "gpgpusim.config" ]; then
    print_msg $RED "  Error: gpgpusim.config not found!"
    exit 1
fi

if [ ! -f "config_fermi_islip.icnt" ]; then
    print_msg $RED "  Error: config_fermi_islip.icnt not found!"
    exit 1
fi

print_msg $GREEN "  ✓ Configuration files found"

# Step 3: Build test
print_msg $YELLOW "[3/5] Building test executable..."

if [ ! -f "test_fmr_manual" ] || [ "test_fmr_manual.cu" -nt "test_fmr_manual" ]; then
    make clean > /dev/null 2>&1 || true
    if make test_fmr_manual 2>&1 | tee build.log; then
        print_msg $GREEN "  ✓ Build successful"
    else
        print_msg $RED "  ✗ Build failed! Check build.log for details"
        exit 1
    fi
else
    print_msg $GREEN "  ✓ Using existing executable"
fi

# Step 4: Clean previous run outputs
print_msg $YELLOW "[4/5] Preparing for simulation..."

rm -f gpgpusim_power_report__*.log
rm -f gpgpu_inst_stats.txt
rm -f _app_cuda_version_*.ptx
rm -f _cuobjdump_complete_output_*
print_msg $GREEN "  ✓ Cleaned previous outputs"

# Step 5: Run simulation
print_msg $YELLOW "[5/5] Running FMR test with GPGPU-Sim..."
echo
print_msg $BLUE "----------------------------------------"
print_msg $BLUE "  Simulation Output"
print_msg $BLUE "----------------------------------------"
echo

# Run the test
./test_fmr_manual 2>&1 | tee simulation.log

# Check simulation result
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo
    print_msg $GREEN "=========================================="
    print_msg $GREEN "  Test PASSED!"
    print_msg $GREEN "=========================================="
    echo
    
    # Show statistics if available
    if [ -f "gpgpu_inst_stats.txt" ]; then
        print_msg $BLUE "Checking for FMR instructions in statistics..."
        if grep -i "fmr\|sample" gpgpu_inst_stats.txt > /dev/null 2>&1; then
            echo
            print_msg $YELLOW "FMR Instruction Statistics:"
            grep -i "fmr\|sample" gpgpu_inst_stats.txt | head -10
        else
            print_msg $YELLOW "  (No FMR-specific statistics found yet)"
        fi
    fi
    
    exit 0
else
    echo
    print_msg $RED "=========================================="
    print_msg $RED "  Test FAILED!"
    print_msg $RED "=========================================="
    echo
    print_msg $YELLOW "Check simulation.log for details"
    exit 1
fi
