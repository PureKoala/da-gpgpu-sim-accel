# Deformable Attention in GPGPU-Sim — Current Implementation & Plan

**Version**: v2.0 (2026-01-23)  
**Scope**: 单一可信来源（当前实现 + 后续计划）。历史设计与调研内容已归档至 DevDocs/backup。

---

## 1. 当前实现（as-is）

### 1.1 集成策略
- **Function Call Only**：通过 CUDA 函数调用拦截实现加速器功能
- **无自定义 PTX 指令**：CUDA 编译器不识别自定义 PTX
- **无执行单元/流水线分发**：`deform_attn_exec_unit` 与 `shader.cc` 流水线分发不使用

### 1.2 Function Call 拦截与功能模型
- **拦截位置**：`src/cuda-sim/cuda-sim.cc`
    - 依据函数名匹配 `__deform_pcb/__deform_tbc/__deform_tma/__deform_interp`
- **功能模型实现**：
    - `src/cuda-sim/deform_attn_unit.h/.cc`
    - `src/cuda-sim/deform_attn_impl.cc`
- **模块覆盖**：PCB / GTC / TBC / TMA+Storage / Interpolation
- **建模方式**：功能模拟 + 固定延迟（配置可调）

### 1.3 CUDA Kernel 实现
- **Baseline（数值基准）**：
    - `deAttn_base/src/deform_attn/cuda/ms_deform_attn_im2col_cuda.cuh`
- **Optimized（硬件调用 + fallback）**：
    - `deAttn_optim/src/deform_attn/cuda/ms_deform_attn_im2col_cuda.cuh`
- **可选 WMMA 聚合路径**（仅满足尺寸对齐时触发）：
    - `deAttn_base/src/deform_attn/cuda/ms_deform_attn_wmma_cuda.cuh`
    - `deAttn_optim/src/deform_attn/cuda/ms_deform_attn_wmma_cuda.cuh`

### 1.4 配置选项（关键项）
```
-gpgpu_deform_attn_avail 1
-gpgpu_num_deform_units 1
-gpgpu_deform_pcb_latency 1
-gpgpu_deform_tbc_latency 7
-gpgpu_deform_tma_latency 16
-gpgpu_deform_storage_latency 6
-gpgpu_deform_interp_latency 3
```

### 1.5 明确不使用的路径
- PTX 伪指令 / 新增操作码
- `deform_attn_exec_unit` 执行单元
- `shader.cc` 内的流水线创建与指令派发逻辑

---

## 2. 当前状态清单

### ✅ 已完成
- Function Call 拦截机制
- 5 模块功能模拟（PCB/GTC/TBC/TMA&Storage/Interpolation）
- Baseline/Optimized Kernel 实现
- GPGPU-Sim 配置参数注册
- 编译通过（含 DeformAttn 支持）

### ⏳ 未完成（按优先级）
- **P0 参数传递修复**：修正 `deform_*_impl()` 的参数读取逻辑
- **P1 功能验证**：Baseline vs Optimized 数值一致性验证
- **P2 性能统计**：PCB 剪枝率、TBC 模式分布、TMA 复用率

---

## 3. 后续计划

### P0：参数传递修复（最高优先级）
- 对齐 FMR 参数传递机制
- 增加调试输出以验证参数一致性

### P1：功能验证
- 编译并运行 baseline/optimized kernels
- 采集输出并做数值对比

### P2：性能统计
- 增加统计计数器（PCB/TBC/TMA）
- 输出格式与采样策略统一

### 可选计划（非当前关注）
- 动态延迟模型
- 多精度支持扩展

---

## 4. 历史与归档
- 设计与调研类文档已归档至 `DevDocs/backup/`
- 如需查看详细架构推导或历史讨论，请查阅归档文档

---

# Deformable Attention 实现指南（归档）

> 以下内容为历史材料，仅供参考。

**版本**: v1.0 (2026-01-22)  
**范围**: 实现状态、GPGPU-Sim集成、Kernel开发、测试验证

---

## 1. 实现现状总览

### 1.1 已完成模块 ✅

#### 功能模型实现
- **文件**: `src/cuda-sim/deform_attn_unit.h/.cc`
- **内容**: 5个模块类的完整实现
  - `deform_pcb_model` - 权重预筛选
  - `deform_gtc_model` - 操作数隔离  
  - `deform_tbc_model` - 边界检查与聚合决策
  - `deform_tma_model` + `deform_storage_model` - Tile加载与存储
  - `deform_interp_model` - 双线性插值

#### Kernel实现（采样/插值）
- **Baseline**: 纯CUDA采样与双线性插值（数值基准）
- **Optimized**: GPGPU-Sim硬件加速调用 + 回退路径（与Baseline一致）
- **文件**:
    - `deAttn_base/src/deform_attn/cuda/ms_deform_attn_im2col_cuda.cuh`
    - `deAttn_optim/src/deform_attn/cuda/ms_deform_attn_im2col_cuda.cuh`

#### Function Call拦截机制  
- **文件**: `src/cuda-sim/cuda-sim.cc` (参考1863-1891行)
- **机制**: 基于FMR成功经验的CUDA函数调用拦截
```cpp
if (fname.find("__deform_pcb") != std::string::npos) {
    deform_pcb_impl(pI, core, inst);
    skip = true;
}
```

#### 实现函数
- **文件**: `src/cuda-sim/deform_attn_impl.cc`  
- **函数**: 
  - `deform_pcb_impl()` - PCB功能与延迟模拟
  - `deform_tbc_impl()` - TBC两阶段决策算法
  - `deform_tma_impl()` - TMA Tile加载模拟
  - `deform_interp_impl()` - 插值计算实现

#### 配置系统
- **文件**: `src/gpgpu-sim/gpu-sim.cc`
- **选项**: 完整的GPGPU-Sim配置参数注册

### 1.2 进行中任务 ⏳

#### GPGPU-Sim流水线集成
**目标**: 在shader.cc中实现执行单元创建与调度

**关键任务**:
```cpp
// 1. 执行单元类定义
class deform_attn_exec_unit : public exec_unit_t {
public:
    deform_attn_exec_unit(shader_core_ctx* core);
    void cycle();
    bool can_issue(warp_inst_t &inst);
    void issue(warp_inst_t &inst);
};

// 2. 流水线创建
void shader_core_ctx::create_exec_pipeline() {
    if (m_config->deform_attn_avail()) {
        m_deform_attn_unit = new deform_attn_exec_unit(this);
    }
}

#### 性能统计
**目标**: 采集加速器命中情况与模式分布

**关键项**:
- PCB剪枝命中率
- TBC模式分布
- TMA Tile复用率

// 3. 指令分发
void shader_core_ctx::issue_warp(warp_inst_t &inst) {
    if (is_deform_attn_call(inst)) {
        if (m_deform_attn_unit->can_issue(inst)) {
            m_deform_attn_unit->issue(inst);
        }
    }
}
```

---

## 2. CUDA Kernel 实现指导

### 2.1 数据布局设计

#### 全局参数
```cpp
const int NUM_HEADS = 8;
const int NUM_LEVELS = 4;  
const int NUM_POINTS_PER_HEAD = 4;
const int TOTAL_POINTS_PER_QUERY = 128;  // 8×4×4
```

#### 张量布局（推荐flattened格式）
```cpp
// 输入张量
float* attn_weights;      // [Q*H*L*P] = [Q*128]
float2* sampling_locs;    // [Q*H*L*P] = [Q*128] 
float* value_maps[4];     // L个多尺度特征图

// 输出张量
float* output;            // [Q]
```

### 2.2 Baseline Kernel - 功能验证版

**设计目标**: 提供数值基准，验证算法正确性

```cpp
__global__ void deform_attn_baseline(
    float* value_maps[], 
    float2* sampling_locs,
    float* attn_weights,
    float* output,
    int num_queries
) {
    // 线程映射：1 thread = 1 query
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= num_queries) return;
    
    float acc = 0.0f;
    
    // 串行处理128个采样点
    for (int t = 0; t < 128; t++) {
        // 解包索引 t -> (h,l,p)
        int p = t % 4;
        int l = (t / 4) % 4;  
        int h = t / 16;
        
        // 读取权重和坐标
        float w = attn_weights[q * 128 + t];
        float2 coord = sampling_locs[q * 128 + t];
        
        // 双线性采样（4次全局内存访问）
        float val = bilinear_sample_global(value_maps[l], coord.x, coord.y);
        acc += val * w;
    }
    
    output[q] = acc;
}

__device__ float bilinear_sample_global(
    float* feature_map, 
    float x, float y
) {
    // 边界检查与clamp
    int width = LEVEL_WIDTHS[level];  // 需要传入或使用常量
    int height = LEVEL_HEIGHTS[level];
    
    x = fmaxf(0.0f, fminf(x, width - 1.0f));
    y = fmaxf(0.0f, fminf(y, height - 1.0f));
    
    // 整数与分数部分
    int x0 = (int)x, y0 = (int)y;
    int x1 = x0 + 1, y1 = y0 + 1;
    float fx = x - x0, fy = y - y0;
    
    // 4点读取（确保边界安全）
    float p00 = feature_map[y0 * width + x0];
    float p01 = (x1 < width) ? feature_map[y0 * width + x1] : p00;
    float p10 = (y1 < height) ? feature_map[y1 * width + x0] : p00;  
    float p11 = (x1 < width && y1 < height) ? feature_map[y1 * width + x1] : p00;
    
    // 双线性插值
    return (1-fx) * (1-fy) * p00 + fx * (1-fy) * p01 + 
           (1-fx) * fy * p10 + fx * fy * p11;
}
```

### 2.3 Optimized Kernel - 硬件协同版

**设计目标**: 与5级流水线硬件协同，实现性能优化

#### 线程映射策略
```cpp
// Block配置：16 queries/block × 256 threads/block
// 分块策略：128 points/query ÷ 16 points/chunk = 8 chunks  
// 线程映射：tid -> (q_local, t_in_chunk)

__global__ void deform_attn_optimized(
    float* value_maps[],
    float2* sampling_locs, 
    float* attn_weights,
    float* output,
    float threshold,
    int num_queries
) {
    // 线程与块索引映射
    int tid = threadIdx.x;                    // 0..255
    int q_local = tid / 16;                   // 0..15 (local query index)
    int t_in_chunk = tid % 16;                // 0..15 (point within chunk)
    int q_global = blockIdx.x * 16 + q_local; // 全局query索引
    
    if (q_global >= num_queries) return;
    
    // 共享内存声明（添加padding避免bank冲突）
    __shared__ float tile_buffer[16][17];     // +1 padding
    __shared__ int shared_mode, shared_base_x, shared_base_y;
    __shared__ float2 coords_buffer[256];
    __shared__ bool valid_buffer[256];
    
    float acc_local = 0.0f;  // 当前线程的局部累加
```

#### 5阶段处理流程
```cpp
    // === 处理8个chunks ===
    for (int chunk_id = 0; chunk_id < 8; chunk_id++) {
        int t_global = chunk_id * 16 + t_in_chunk;  // 全局point索引
        
        // === Stage I: PCB权重筛选 ===
        float w = attn_weights[q_global * 128 + t_global];
        bool skip = __deform_pcb(&w, threshold);  // 硬件调用
        
        // === Stage II: GTC操作数隔离 ===  
        float2 coord = make_float2(0.0f, 0.0f);
        if (!skip) {
            coord = sampling_locs[q_global * 128 + t_global];
        }
        
        // === Stage III: TBC聚合决策 ===
        // 收集当前chunk内所有有效坐标
        coords_buffer[tid] = coord;
        valid_buffer[tid] = !skip;
        __syncthreads();
        
        // 单线程执行TBC决策
        if (tid == 0) {
            float2 active_coords[256];
            int num_active = 0;
            
            for (int i = 0; i < 256; i++) {
                if (valid_buffer[i]) {
                    active_coords[num_active++] = coords_buffer[i];
                }
            }
            
            int mode, base_x, base_y;
            __deform_tbc(active_coords, num_active, &mode, &base_x, &base_y);
            
            shared_mode = mode;
            shared_base_x = base_x; 
            shared_base_y = base_y;
        }
        __syncthreads();
        
        // === Stage IV: TMA & Storage ===
        if (shared_mode != 3) {  // 非Discrete模式
            // 256线程协作加载16×16 tile
            int tx = tid % 16;
            int ty = tid / 16;
            
            if (ty < 16) {  // 仅前256线程参与加载
                int gx = shared_base_x + tx;
                int gy = shared_base_y + ty;
                int level = (t_global / 4) % 4;
                
                // 硬件TMA调用
                float tile_val = __deform_tma(gx, gy, level, shared_mode);
                tile_buffer[ty][tx] = tile_val;  // 无bank冲突写入
            }
            __syncthreads();
        }
        
        // === Stage V: Interpolation ===
        if (!skip) {  // 仅处理有效点
            float val;
            int level = (t_global / 4) % 4;
            
            if (shared_mode != 3) {
                // Tile模式：从共享内存插值
                int lx = (int)coord.x - shared_base_x;
                int ly = (int)coord.y - shared_base_y;
                float fx = coord.x - (int)coord.x;
                float fy = coord.y - (int)coord.y;
                
                // 硬件插值调用
                val = __deform_interp(tile_buffer, lx, ly, fx, fy, shared_mode);
            } else {
                // Discrete模式：全局内存fallback
                val = bilinear_sample_global(value_maps[level], coord.x, coord.y);
            }
            
            acc_local += val * w;
        }
    }
    
    // === 每query的16线程归约 ===
    __shared__ float acc_buffer[256];
    acc_buffer[tid] = acc_local;
    __syncthreads();
    
    // 每个query选择一个线程做最终写回
    if (t_in_chunk == 0) {
        float sum = 0.0f;
        for (int i = 0; i < 16; i++) {
            sum += acc_buffer[q_local * 16 + i];
        }
        output[q_global] = sum;
    }
}
```

### 2.4 CUDA最佳实践要点

#### 内存访问优化
```cpp
// ✅ 合并访问模式
float w = attn_weights[q * 128 + t];  // 连续访问

// ❌ 避免跨步访问
float w = attn_weights[q + t * NUM_QUERIES];  // 大跨步
```

#### Bank冲突避免
```cpp
// ✅ 添加padding
__shared__ float tile_buffer[16][17];  // +1避免bank冲突

// ❌ 32的倍数维度 
__shared__ float tile_buffer[16][32];  // 32-way冲突
```

#### Warp Divergence控制
```cpp
// 使用warp-level原语减少分支发散
if (__all_sync(0xffffffff, mode != 3)) {
    // 整个warp都走Tile路径
} else if (__all_sync(0xffffffff, mode == 3)) {
    // 整个warp都走Discrete路径  
} else {
    // 混合处理
}
```

---

## 3. 实现数据流（无代码）

### 3.1 Baseline 数据流
1. 读取输入张量：`data_value`、`data_spatial_shapes`、`data_level_start_index`、`data_sampling_loc`、`data_attn_weight`。
2. 对每个输出元素解码索引（batch、head、channel、sampling index）。
3. 对每个 level 与 point 计算采样坐标 $(h_{im}, w_{im})$ 并做边界判断。
4. 在有效范围内执行标准双线性插值并按权重累加。
5. 写回 `data_col`。

### 3.2 Optimized 数据流
1. 输入与索引映射与 Baseline 相同，确保输出维度与数值语义一致。
2. 生成当前 level 的采样坐标与权重缓冲。
3. 在 GPGPU-Sim 下依次调用：`__deform_pcb`（权重筛选）、`__deform_tbc`（聚合判定）、`__deform_tma`（Tile加载）、`__deform_interp`（插值）。
4. 若聚合失败或越界，回退到标准双线性插值路径。
5. 写回 `data_col`。

### 3.3 注意力聚合阶段
注意力权重与采样值的矩阵聚合存在可选 WMMA 实验路径，仅在尺寸满足 16 对齐时触发；采样/插值阶段为优化差异的核心。

## 4. 测试与验证策略

### 3.1 单模块测试

#### PCB模块测试
```cpp
// 测试权重筛选逻辑
void test_pcb_functionality() {
    float weights[16*16] = {...};  // 测试数据
    bool expected_mask[16*16] = {...};
    bool actual_mask[16*16];
    
    deform_pcb_impl(weights, 0.1f, actual_mask);
    assert_masks_equal(expected_mask, actual_mask);
}
```

#### TBC模块测试
```cpp
// 测试聚合决策算法
void test_tbc_mode_detection() {
    // 测试用例1：高局部性 -> Horizontal模式
    float2 coords_clustered[] = {{10,10}, {11,10}, {12,10}, ...};
    int mode = test_tbc_decision(coords_clustered, 16);
    assert(mode == HORIZONTAL_MODE);
    
    // 测试用例2：随机分布 -> Discrete模式  
    float2 coords_random[] = {{1,5}, {50,100}, {200,300}, ...};
    mode = test_tbc_decision(coords_random, 16);
    assert(mode == DISCRETE_MODE);
}
```

### 3.2 端到端验证

#### 数值正确性验证
```cpp
void test_numerical_correctness() {
    // 1. 生成相同输入数据
    float* weights = generate_test_weights();
    float2* coords = generate_test_coords(); 
    float* value_maps = generate_test_maps();
    
    // 2. 分别运行两个kernel
    float* baseline_output = run_baseline_kernel(weights, coords, value_maps);
    float* optimized_output = run_optimized_kernel(weights, coords, value_maps);
    
    // 3. 对比结果（允许小幅浮点误差）
    for (int q = 0; q < num_queries; q++) {
        float diff = fabs(baseline_output[q] - optimized_output[q]);
        assert(diff < 1e-5);  // 数值精度要求
    }
}
```

#### 性能基准测试
```cpp
void test_performance_improvement() {
    // 测试不同聚合度下的性能表现
    float aggregation_ratios[] = {0.2, 0.5, 0.8, 0.95};
    
    for (auto ratio : aggregation_ratios) {
        auto coords = generate_coords_with_locality(ratio);
        
        auto baseline_time = benchmark_baseline_kernel(coords);
        auto optimized_time = benchmark_optimized_kernel(coords); 
        
        float speedup = baseline_time / optimized_time;
        printf("Aggregation %.1f: Speedup %.2fx\n", ratio, speedup);
    }
}
```

### 3.3 GPGPU-Sim集成测试

#### Function Call拦截验证
```cpp
void test_function_call_interception() {
    // 验证硬件函数调用被正确拦截
    launch_test_kernel();  // 包含__deform_pcb调用
    
    // 检查统计计数器
    assert(get_deform_pcb_call_count() > 0);
    assert(get_deform_tbc_call_count() > 0);
    assert(get_deform_tma_call_count() > 0);
}
```

#### 延迟建模验证
```cpp
void test_latency_modeling() {
    // 验证各阶段延迟统计正确
    run_deform_attn_workload();
    
    auto stats = get_deform_attn_stats();
    assert(stats.pcb_total_cycles == stats.pcb_calls * 1);      // PCB: 1 cycle
    assert(stats.tbc_total_cycles <= stats.tbc_calls * 7);     // TBC: ≤7 cycles
    assert(stats.tma_total_cycles == stats.tma_calls * 16);    // TMA: 16 cycles
}
```

---

## 4. 调试与问题排查

### 4.1 常见问题及解决方案

#### 问题1：Function Call拦截失败
**症状**: kernel运行但无硬件加速效果
**排查**:
```cpp
// 1. 确认函数名匹配
if (fname.find("__deform_pcb") != std::string::npos) // 精确匹配

// 2. 检查skip标志设置
skip = true;  // 必须设置，避免进入标准CALL处理

// 3. 验证所有warp线程PC同步
// 参考FMR实现中的线程同步机制
```

#### 问题2：共享内存Bank冲突
**症状**: 性能不达预期，smem访问延迟高
**解决**:
```cpp
// 添加padding维度
__shared__ float tile_buffer[16][17];  // 而非[16][16]

// 或使用交错访问模式
int offset = threadIdx.x % 16;
smem[threadIdx.y][(threadIdx.x + offset) % 17] = data;
```

#### 问题3：数值精度问题  
**症状**: baseline与optimized结果差异大
**排查**:
```cpp
// 1. 检查边界处理一致性
// 2. 验证插值算法精度
// 3. 确认浮点运算顺序（FP16 vs FP32）
```

### 4.2 性能分析与调试工具

#### 4.2.1 Nsight Compute性能分析

**编译选项配置**

```bash
# 编译时添加调试信息以支持性能分析
nvcc -O3 -arch=sm_80 \
     -lineinfo \
     -src-in-ptx \
     -g -G \
     deform_attn_kernel.cu -o kernel.out
```

**关键性能指标收集**

```bash
# 使用Nsight Compute分析kernel性能
ncu --set full \
    --export profile_report \
    --force-overwrite \
    ./kernel.out

# 重点关注的指标
ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
    smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
    smsp__average_warps_issue_stalled_short_scoreboard.pct \
    ./kernel.out
```

**性能瓶颈分析脚本**

```python
# analyze_performance.py
import pandas as pd
import json

def analyze_ncu_report(report_file):
    with open(report_file, 'r') as f:
        data = json.load(f)
    
    metrics = data['metrics']
    
    # 分析计算吞吐量
    sm_throughput = metrics['sm__throughput.avg.pct_of_peak_sustained_elapsed']
    if sm_throughput < 50:
        print(f"⚠️  Low SM utilization: {sm_throughput}%")
        print("    Possible causes: insufficient parallelism, branch divergence")
    
    # 分析内存带宽
    dram_throughput = metrics['dram__throughput.avg.pct_of_peak_sustained_elapsed']
    if dram_throughput > 70:
        print(f"⚠️  Memory bound: {dram_throughput}% DRAM utilization")
        print("    Suggestion: optimize memory access patterns, use shared memory")
    
    # 分析Bank冲突
    bank_conflicts = metrics['l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum']
    if bank_conflicts > 1000:
        print(f"⚠️  High shared memory bank conflicts: {bank_conflicts}")
        print("    Suggestion: add padding or use XOR addressing")
    
    # 分析内存合并
    coalescing = metrics['smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct']
    if coalescing < 80:
        print(f"⚠️  Poor memory coalescing: {coalescing}%")
        print("    Suggestion: ensure contiguous memory access patterns")

if __name__ == "__main__":
    analyze_ncu_report("profile_report.ncu-rep")
```

#### 4.2.2 GPGPU-Sim性能分析器

**详细延迟分析器实现**

```cpp
// 文件: src/gpgpu-sim/deform_attn_profiler.h
class deform_attn_profiler {
public:
    deform_attn_profiler() { reset(); }
    
    // 记录各阶段延迟
    void record_stage_latency(int stage, unsigned cycles) {
        m_stage_cycles[stage] += cycles;
        m_stage_calls[stage]++;
        
        // 延迟直方图
        if (cycles < 10) m_latency_hist[stage][0]++;
        else if (cycles < 50) m_latency_hist[stage][1]++;
        else if (cycles < 100) m_latency_hist[stage][2]++;
        else m_latency_hist[stage][3]++;
    }
    
    // 记录模式分布
    void record_tbc_mode(int mode) {
        switch(mode) {
            case HORIZONTAL_MODE: m_horizontal_count++; break;
            case VERTICAL_MODE: m_vertical_count++; break;
            case XOR_MODE: m_xor_count++; break;
            case DISCRETE_MODE: m_discrete_count++; break;
        }
    }
    
    // 记录PCB剪枝效果
    void record_pcb_result(int total_points, int masked_points) {
        m_pcb_total_points += total_points;
        m_pcb_masked_points += masked_points;
    }
    
    // 打印分析报告
    void print_stage_breakdown(FILE* fp) {
        fprintf(fp, "\n=== Deformable Attention Performance Report ===\n");
        
        // 各阶段延迟分解
        fprintf(fp, "\nStage Latency Breakdown:\n");
        const char* stage_names[] = {"PCB", "GTC", "TBC", "TMA", "Storage", "Interp"};
        
        unsigned long long total_cycles = 0;
        for (int i = 0; i < 6; i++) {
            unsigned long long avg = m_stage_calls[i] > 0 ? 
                                    m_stage_cycles[i] / m_stage_calls[i] : 0;
            fprintf(fp, "  %s: %llu calls, %llu total cycles, %.2f avg cycles\n",
                   stage_names[i], m_stage_calls[i], m_stage_cycles[i], 
                   (double)avg);
            total_cycles += m_stage_cycles[i];
        }
        
        fprintf(fp, "  Total: %llu cycles\n", total_cycles);
        
        // 延迟直方图
        fprintf(fp, "\nLatency Distribution:\n");
        for (int i = 0; i < 6; i++) {
            fprintf(fp, "  %s: <10c:%llu, 10-50c:%llu, 50-100c:%llu, >100c:%llu\n",
                   stage_names[i],
                   m_latency_hist[i][0], m_latency_hist[i][1],
                   m_latency_hist[i][2], m_latency_hist[i][3]);
        }
        
        // TBC模式分布
        fprintf(fp, "\nTBC Mode Distribution:\n");
        unsigned long long total_tbc = m_horizontal_count + m_vertical_count + 
                                       m_xor_count + m_discrete_count;
        if (total_tbc > 0) {
            fprintf(fp, "  Horizontal: %llu (%.1f%%)\n", 
                   m_horizontal_count, 100.0 * m_horizontal_count / total_tbc);
            fprintf(fp, "  Vertical:   %llu (%.1f%%)\n",
                   m_vertical_count, 100.0 * m_vertical_count / total_tbc);
            fprintf(fp, "  XOR:        %llu (%.1f%%)\n",
                   m_xor_count, 100.0 * m_xor_count / total_tbc);
            fprintf(fp, "  Discrete:   %llu (%.1f%%)\n",
                   m_discrete_count, 100.0 * m_discrete_count / total_tbc);
        }
        
        // PCB剪枝效果
        fprintf(fp, "\nPCB Pruning Effectiveness:\n");
        if (m_pcb_total_points > 0) {
            double prune_rate = 100.0 * m_pcb_masked_points / m_pcb_total_points;
            fprintf(fp, "  Total points: %llu\n", m_pcb_total_points);
            fprintf(fp, "  Masked points: %llu (%.1f%%)\n", 
                   m_pcb_masked_points, prune_rate);
            fprintf(fp, "  Compute savings: %.1f%%\n", prune_rate);
        }
    }
    
    // 瓶颈分析
    void analyze_bottlenecks(FILE* fp) {
        fprintf(fp, "\n=== Bottleneck Analysis ===\n");
        
        // 找出最耗时的阶段
        int max_stage = 0;
        unsigned long long max_cycles = 0;
        for (int i = 0; i < 6; i++) {
            if (m_stage_cycles[i] > max_cycles) {
                max_cycles = m_stage_cycles[i];
                max_stage = i;
            }
        }
        
        const char* stage_names[] = {"PCB", "GTC", "TBC", "TMA", "Storage", "Interp"};
        fprintf(fp, "Primary bottleneck: %s (%.1f%% of total time)\n",
               stage_names[max_stage],
               100.0 * max_cycles / (m_stage_cycles[0] + m_stage_cycles[1] + 
                                    m_stage_cycles[2] + m_stage_cycles[3] + 
                                    m_stage_cycles[4] + m_stage_cycles[5]));
        
        // 分析Discrete模式比例
        unsigned long long total_tbc = m_horizontal_count + m_vertical_count + 
                                       m_xor_count + m_discrete_count;
        if (total_tbc > 0) {
            double discrete_ratio = 100.0 * m_discrete_count / total_tbc;
            if (discrete_ratio > 30) {
                fprintf(fp, "\n⚠️  High Discrete mode ratio: %.1f%%\n", discrete_ratio);
                fprintf(fp, "   Suggestion: Low spatial locality in sampling patterns\n");
                fprintf(fp, "   Consider: Adjusting TBC aggregation threshold\n");
            }
        }
        
        // 分析PCB效率
        if (m_pcb_total_points > 0) {
            double prune_rate = 100.0 * m_pcb_masked_points / m_pcb_total_points;
            if (prune_rate < 10) {
                fprintf(fp, "\n⚠️  Low PCB pruning rate: %.1f%%\n", prune_rate);
                fprintf(fp, "   Suggestion: Weight distribution is uniform\n");
                fprintf(fp, "   Consider: Increasing threshold or disable PCB\n");
            }
        }
    }
    
    void reset() {
        memset(m_stage_cycles, 0, sizeof(m_stage_cycles));
        memset(m_stage_calls, 0, sizeof(m_stage_calls));
        memset(m_latency_hist, 0, sizeof(m_latency_hist));
        m_horizontal_count = m_vertical_count = 0;
        m_xor_count = m_discrete_count = 0;
        m_pcb_total_points = m_pcb_masked_points = 0;
    }
    
private:
    unsigned long long m_stage_cycles[6];  // PCB, GTC, TBC, TMA, Storage, Interp
    unsigned long long m_stage_calls[6];
    unsigned long long m_latency_hist[6][4];  // <10, 10-50, 50-100, >100 cycles
    
    // TBC模式统计
    unsigned long long m_horizontal_count;
    unsigned long long m_vertical_count;
    unsigned long long m_xor_count;
    unsigned long long m_discrete_count;
    
    // PCB统计
    unsigned long long m_pcb_total_points;
    unsigned long long m_pcb_masked_points;
};
```

#### 4.2.3 自定义性能计数器

```cpp
// Kernel内性能计数
__global__ void instrumented_kernel(...) {
    // 使用clock64()进行细粒度计时
    unsigned long long start_time = clock64();
    
    // 执行PCB阶段
    perform_pcb(...);
    unsigned long long pcb_time = clock64() - start_time;
    
    // 记录到global memory
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        timing_data[STAGE_PCB] = pcb_time;
    }
    
    // 继续其他阶段...
}
```

---

## 5. 下一步开发计划

### 5.1 短期目标（1-2周）
1. **完成GPGPU-Sim流水线集成** - shader.cc执行单元实现
2. **实现Baseline kernel** - 建立数值基准
3. **基础功能验证** - 确保Function Call拦截正常工作

### 5.2 中期目标（1个月）
1. **完成Optimized kernel** - 5阶段硬件协同实现
2. **端到端测试** - 数值正确性与性能验证  
3. **性能统计完善** - 详细的分析报告

### 5.3 长期目标（2-3个月）
1. **动态延迟建模** - 基于内存系统状态的精确建模
2. **多级优化** - L1/L2 Cache感知的优化策略
3. **论文支撑** - 完整的实验数据与分析

---

**开发重点**: 当前最高优先级是完成GPGPU-Sim流水线集成，这是后续所有工作的基础。建议先实现最小可工作版本，再逐步完善功能。