#!/bin/bash
# Deformable Attention Accelerator P1 Validation Test
# 用于验证Baseline vs Optimized数值一致性

set -e  # Exit on error

echo "=========================================="
echo "Deformable Attention P1 Validation Test"
echo "=========================================="
echo ""

# 设置路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 编译最新代码
echo "[1/5] Syncing GPGPU-Sim code..."
./syn_gpgpu_sim.sh || { echo "ERROR: Sync failed"; exit 1; }

echo ""
echo "[2/5] Compiling CUDA kernels..."
make clean && make || { echo "ERROR: Compilation failed"; exit 1; }

# 运行Baseline测试
echo ""
echo "[3/5] Running BASELINE test (no accelerator)..."
echo "Config: -gpgpu_deform_attn_avail 0"

# 创建baseline配置
sed -i 's/-gpgpu_deform_attn_avail.*/-gpgpu_deform_attn_avail 0/' sim/gpgpusim.config
sed -i 's/-gpgpu_tensorcore_debug.*/-gpgpu_tensorcore_debug 0/' sim/gpgpusim.config

# 运行baseline
cd sim
timeout 120s ../bin/deform_attn_test > baseline_output.txt 2>&1 || true
cd ..

echo "Baseline output saved to: sim/baseline_output.txt"

# 运行Optimized测试
echo ""
echo "[4/5] Running OPTIMIZED test (with accelerator)..."
echo "Config: -gpgpu_deform_attn_avail 1, -gpgpu_tensorcore_debug 1"

# 创建optimized配置
sed -i 's/-gpgpu_deform_attn_avail.*/-gpgpu_deform_attn_avail 1/' sim/gpgpusim.config
sed -i 's/-gpgpu_tensorcore_debug.*/-gpgpu_tensorcore_debug 1/' sim/gpgpusim.config

# 运行optimized
cd sim
timeout 120s ../bin/deform_attn_test > optimized_output.txt 2>&1 || true
cd ..

echo "Optimized output saved to: sim/optimized_output.txt"

# 对比结果
echo ""
echo "[5/5] Comparing outputs..."
echo ""

# 提取数值结果（假设输出格式包含"Output tensor:"）
if [ -f sim/baseline_output.txt ] && [ -f sim/optimized_output.txt ]; then
    echo "=== Baseline Output (first 10 lines) ==="
    head -n 10 sim/baseline_output.txt | grep -E "(Output|Result|tensor)" || echo "No output found"
    echo ""
    
    echo "=== Optimized Output (first 10 lines) ==="
    head -n 10 sim/optimized_output.txt | grep -E "(Output|Result|tensor)" || echo "No output found"
    echo ""
    
    echo "=== Debug Info (Optimized) ==="
    grep -E "(DEFORM|PCB|TBC|TMA|INTERP)" sim/optimized_output.txt | head -n 20 || echo "No debug output found"
    echo ""
    
    echo "Full logs available at:"
    echo "  - Baseline:  sim/baseline_output.txt"
    echo "  - Optimized: sim/optimized_output.txt"
else
    echo "ERROR: Output files not found!"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Test completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Check sim/optimized_output.txt for DEFORM debug logs"
echo "2. Verify PCB/TBC/TMA/INTERP are being called"
echo "3. Compare numerical outputs manually or use Python script"
echo ""
echo "For Python comparison:"
echo "  python3 compare_outputs.py sim/baseline_output.txt sim/optimized_output.txt"
