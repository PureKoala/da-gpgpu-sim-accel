#!/bin/bash

# 性能分析脚本
# 使用 Nsight Compute 和 nvprof

set -e

EXECUTABLE="./bin/test_deformable_attn"

if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable not found: $EXECUTABLE"
    echo "Please build the project first."
    exit 1
fi

echo "========================================="
echo "DeformableAttention Performance Analysis"
echo "========================================="
echo ""

# 1. 基础性能测试
echo "1. Running basic performance test..."
$EXECUTABLE
echo ""

# 2. Nsight Compute 分析
if command -v ncu &> /dev/null; then
    echo "2. Running Nsight Compute analysis..."
    echo "   This may take a few minutes..."
    
    # 完整分析
    ncu --set full \
        --export nsight_compute_report \
        --force-overwrite \
        $EXECUTABLE
    
    echo "   Report saved to: nsight_compute_report.ncu-rep"
    echo "   Open with: ncu-ui nsight_compute_report.ncu-rep"
    echo ""
    
    # 快速分析 (关键指标)
    echo "   Quick analysis (key metrics):"
    ncu --metrics \
        sm__throughput.avg.pct_of_peak_sustained_elapsed,\
        dram__throughput.avg.pct_of_peak_sustained_elapsed,\
        l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
        l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
        smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,\
        smsp__sass_thread_inst_executed_op_dmul_pred_on.sum \
        $EXECUTABLE 2>&1 | grep -A 20 "Metric Name"
else
    echo "2. Nsight Compute not found, skipping..."
fi
echo ""

# 3. nvprof 分析 (如果可用)
if command -v nvprof &> /dev/null; then
    echo "3. Running nvprof analysis..."
    
    # GPU trace
    nvprof --print-gpu-trace \
           --log-file nvprof_trace.log \
           $EXECUTABLE
    
    echo "   GPU trace saved to: nvprof_trace.log"
    
    # API trace
    nvprof --print-api-trace \
           --log-file nvprof_api.log \
           $EXECUTABLE
    
    echo "   API trace saved to: nvprof_api.log"
    
    # Metrics
    nvprof --metrics achieved_occupancy,\
                    gld_efficiency,\
                    gst_efficiency,\
                    sm_efficiency,\
                    warp_execution_efficiency \
           --log-file nvprof_metrics.log \
           $EXECUTABLE
    
    echo "   Metrics saved to: nvprof_metrics.log"
else
    echo "3. nvprof not found, skipping..."
fi
echo ""

# 4. CUDA memcheck
if command -v cuda-memcheck &> /dev/null; then
    echo "4. Running CUDA memcheck..."
    cuda-memcheck --leak-check full \
                  --log-file cuda_memcheck.log \
                  $EXECUTABLE
    echo "   Memcheck report saved to: cuda_memcheck.log"
else
    echo "4. cuda-memcheck not found, skipping..."
fi
echo ""

echo "========================================="
echo "Analysis completed!"
echo "========================================="
echo ""
echo "Generated files:"
ls -lh *.log *.ncu-rep 2>/dev/null || echo "  No report files found"
echo ""
echo "Tips:"
echo "  - Open .ncu-rep files with ncu-ui for detailed analysis"
echo "  - Check .log files for performance metrics"
echo "  - Look for memory coalescing issues"
echo "  - Check Tensor Core utilization"
echo "========================================="
