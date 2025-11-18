# FMR性能分析指南

## 概述

本文档说明如何在GPGPU-Sim中统计和分析FMR优化版本的性能。

## 方法1：GPGPU-Sim自动统计（最准确）

### 1.1 运行并捕获输出

```bash
cd /home/koala/gpgpu-sim_distribution/deAttn_fp16_fmr
bash run.sh 2>&1 | tee performance.log
```

### 1.2 查看关键性能指标

#### Kernel执行周期数
```bash
grep "gpu_tot_sim_cycle\|gpu_sim_cycle" performance.log
```

**输出示例**：
```
gpu_sim_cycle = 1234567
gpu_tot_sim_cycle = 1234567
```

#### Kernel执行时间（IPC统计）
```bash
grep "kernel_name\|gpu_ipc" performance.log
```

#### 内存访问统计
```bash
grep "L1D_total_cache_accesses\|L2_total_cache_accesses\|gpgpu_n_mem_read_global\|gpgpu_n_mem_write_global" performance.log
```

#### Cache命中率
```bash
grep "L1D_total_cache_miss_rate\|L2_total_cache_miss_rate" performance.log
```

### 1.3 对比两个版本

#### 运行原始版本
```bash
cd deAttn_fp16_fmr
# 修改 src/deform_attn/deform_attn_cuda.cu 使用原始kernel
# ms_deform_attn_cuda_forward 调用 ms_deformable_im2col_cuda 而非 ms_deformable_im2col_cuda_fmr_optimized
make clean && make
bash run.sh 2>&1 | tee perf_original.log
```

#### 运行FMR优化版本
```bash
# 使用 ms_deformable_im2col_cuda_fmr_optimized
make clean && make
bash run.sh 2>&1 | tee perf_fmr.log
```

#### 对比周期数
```bash
echo "=== Original Version ==="
grep "gpu_tot_sim_cycle" perf_original.log | tail -1

echo "=== FMR Optimized Version ==="
grep "gpu_tot_sim_cycle" perf_fmr.log | tail -1
```

## 方法2：主机端CUDA Events计时（已实现）

代码中已经使用CUDA Events进行计时：

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
// ... kernel launch ...
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float time_ms;
cudaEventElapsedTime(&time_ms, start, stop);
```

**优点**：
- ✓ 准确的kernel执行时间
- ✓ 自动包含在输出中

**注意**：
- GPGPU-Sim会模拟cudaEventElapsedTime，返回模拟的时间

## 方法3：解析GPGPU-Sim统计文件

GPGPU-Sim会生成详细的统计文件（如果配置中启用）：

```bash
# 在配置文件中启用统计输出
# gpgpusim.config 添加:
# -gpgpu_runtime_stat <filename>
```

## 方法4：使用gpgpu_卡统计器

### 4.1 查看所有可用统计
```bash
grep "^gpgpu_" performance.log | head -50
```

### 4.2 关键性能指标

| 指标 | 含义 | grep命令 |
|------|------|----------|
| `gpu_tot_sim_cycle` | 总模拟周期数 | `grep gpu_tot_sim_cycle` |
| `gpu_tot_ipc` | 平均IPC | `grep gpu_tot_ipc` |
| `gpgpu_n_load_insn` | Load指令数 | `grep gpgpu_n_load_insn` |
| `gpgpu_n_store_insn` | Store指令数 | `grep gpgpu_n_store_insn` |
| `L1D_total_cache_miss_rate` | L1D缺失率 | `grep L1D_total_cache_miss_rate` |
| `L2_total_cache_miss_rate` | L2缺失率 | `grep L2_total_cache_miss_rate` |
| `gpgpu_n_mem_read_global` | 全局内存读次数 | `grep gpgpu_n_mem_read_global` |

## 方法5：性能对比脚本

创建自动化对比脚本：

```bash
#!/bin/bash
# compare_performance.sh

echo "========== Performance Comparison =========="

echo -e "\n[Original Version]"
ORIG_CYCLES=$(grep "gpu_tot_sim_cycle" perf_original.log | tail -1 | awk '{print $3}')
ORIG_IPC=$(grep "gpu_tot_ipc" perf_original.log | tail -1 | awk '{print $3}')
echo "Cycles: $ORIG_CYCLES"
echo "IPC: $ORIG_IPC"

echo -e "\n[FMR Optimized Version]"
FMR_CYCLES=$(grep "gpu_tot_sim_cycle" perf_fmr.log | tail -1 | awk '{print $3}')
FMR_IPC=$(grep "gpu_tot_ipc" perf_fmr.log | tail -1 | awk '{print $3}')
echo "Cycles: $FMR_CYCLES"
echo "IPC: $FMR_IPC"

echo -e "\n[Speedup]"
SPEEDUP=$(echo "scale=2; $ORIG_CYCLES / $FMR_CYCLES" | bc)
echo "Speedup: ${SPEEDUP}x"

echo -e "\n[Memory Statistics]"
echo "Original - L1D Miss Rate:"
grep "L1D_total_cache_miss_rate" perf_original.log | tail -1

echo "FMR - L1D Miss Rate:"
grep "L1D_total_cache_miss_rate" perf_fmr.log | tail -1
```

## 诊断性能下降

如果FMR优化版本性能下降，检查以下指标：

### 1. 共享内存使用
```bash
grep "shared_mem\|smem" performance.log
```

### 2. Bank冲突
```bash
grep "bank_conflict\|shared_bank" performance.log
```

### 3. Warp占用率
```bash
grep "Warp Occupancy\|achieved_occupancy" performance.log
```

### 4. 指令吞吐量
```bash
grep "inst_executed\|ipc" performance.log
```

### 5. FMR调用统计
```bash
grep "FMR\|fmr_sample" performance.log
```

## 性能优化建议

基于统计结果：

1. **如果L1D缺失率高** → FMR tile size可能太大
2. **如果IPC低** → 可能存在同步或依赖问题
3. **如果共享内存Bank冲突** → 调整访问模式
4. **如果Warp占用率低** → 减少共享内存使用或寄存器压力

## 快速诊断命令

```bash
# 一键对比关键指标
echo "=== Cycles ===" && \
grep "gpu_tot_sim_cycle" perf_*.log | tail -2 && \
echo -e "\n=== IPC ===" && \
grep "gpu_tot_ipc" perf_*.log | tail -2 && \
echo -e "\n=== L1D Miss Rate ===" && \
grep "L1D_total_cache_miss_rate" perf_*.log | tail -2 && \
echo -e "\n=== Memory Access ===" && \
grep "gpgpu_n_mem_read_global\|gpgpu_n_mem_write_global" perf_*.log | tail -4
```

## 总结

**推荐流程**：
1. 使用方法1（GPGPU-Sim自动统计）获取详细数据
2. 使用方法2（CUDA Events）验证
3. 使用方法5（对比脚本）快速比较
4. 根据诊断建议优化

**关键指标**：
- `gpu_tot_sim_cycle`：越小越好
- `gpu_tot_ipc`：越大越好
- Cache miss rate：越小越好
- FMR调用开销：应该小于原始内存访问开销
