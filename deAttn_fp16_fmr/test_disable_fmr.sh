#!/bin/bash

echo "==================================="
echo "Testing FMR Analysis Control"
echo "==================================="

# Clean previous build
make clean

echo ""
echo "--- Test 1: Default (FMR Analysis Enabled) ---"
make -j8
if [ $? -eq 0 ]; then
    echo "✓ Build successful with FMR analysis enabled"
else
    echo "✗ Build failed"
    exit 1
fi

# Check for FMR-related symbols
echo "Checking for analyze_query_sampling_points_per_level in binary..."
if objdump -t bin/deform_attn_forward_test 2>/dev/null | grep -q "analyze_query_sampling_points_per_level"; then
    echo "✓ FMR analysis function found in binary"
else
    echo "✗ FMR analysis function NOT found (expected with ENABLE_FMR_ANALYSIS=1)"
fi

echo ""
echo "--- Test 2: FMR Analysis Disabled ---"
make clean
make CXXFLAGS="-DENABLE_FMR_ANALYSIS=0" -j8
if [ $? -eq 0 ]; then
    echo "✓ Build successful with FMR analysis disabled"
else
    echo "✗ Build failed"
    exit 1
fi

echo ""
echo "==================================="
echo "All tests completed!"
echo "==================================="
echo ""
echo "Usage:"
echo "  Default (FMR enabled):  make"
echo "  Disable FMR analysis:   make CXXFLAGS=\"-DENABLE_FMR_ANALYSIS=0\""
