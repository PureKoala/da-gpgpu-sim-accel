#!/bin/bash
# Test script for FMR mode control in deformable attention kernel

echo "========================================"
echo "FMR Mode Testing Script"
echo "========================================"
echo ""
echo "Testing 3 FMR modes:"
echo "  Mode 0: Disable FMR (GMEM only)"
echo "  Mode 1: Selective load (analyze and decide)"
echo "  Mode 2: Force load all levels"
echo ""

# Ensure the binary is built
if [ ! -f "./bin/test" ]; then
    echo "Error: Binary not found. Please run 'make' first."
    exit 1
fi

# Test Mode 0: Disable FMR
echo "========================================"
echo "Testing Mode 0: Disable FMR (GMEM only)"
echo "========================================"
FMR_MODE=0 ./bin/test
echo ""
echo ""

# Test Mode 1: Selective load (default)
echo "========================================"
echo "Testing Mode 1: Selective load"
echo "========================================"
FMR_MODE=1 ./bin/test
echo ""
echo ""

# Test Mode 2: Force load all
echo "========================================"
echo "Testing Mode 2: Force load all levels"
echo "========================================"
FMR_MODE=2 ./bin/test
echo ""
echo ""

echo "========================================"
echo "All FMR mode tests completed!"
echo "========================================"
