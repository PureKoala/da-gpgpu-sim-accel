# Deformable Attention GPGPU-Sim 集成指南

**版本**: v1.1 (2026-01-22)  
**目标**: 整合硬件架构、kernel实现与GPGPU-Sim集成的完整指南  
**状态**: 硬件模型已实现，流水线集成进行中

---

## 概述

本指南整合了Deformable Attention在GPGPU-Sim中的完整实现方案，包括：
- ✅ **硬件功能模型**：5级流水线的功能实现与延迟建模  
- ✅ **Function Call拦截**：基于FMR经验的成功实现方案  
- ⏳ **GPGPU-Sim集成**：流水线单元创建与调度（进行中）  
- 📋 **Kernel实现指导**：Baseline与Optimized两版本的代码指导

---

## 第一部分：硬件架构总览

### 1.1 设计目标与解决方案

**核心挑战**：
- 稀疏采样的随机访存 → **Predict-then-Execute** 架构
- 低权重点的无效计算 → **权重预筛选 + 操作数隔离**

**解决方案**：
```
GPU前端计算 → PCB权重筛选 → GTC操作数隔离 → TBC聚合决策 → TMA&Storage → Interpolation
```

### 1.2 5级流水线详细设计

#### Stage 1: PCB (Pre-Check Block) - 权重预筛选 ✅
```cpp
// 硬件功能：阈值比较 + mask生成
__deform_pcb(weights[16][16], threshold, enable) → mask[16][16]
```
- **延迟**: 1 cycle（固定）
- **实现**: `deform_pcb_impl()` in deform_attn_impl.cc
- **Kernel调用点**: 每个16×16 processing tile开始时

#### Stage 2: GTC (Gated Tensor Core) - 操作数隔离 ✅
```cpp
// kernel内实现：基于PCB mask的坐标谓词化
if (mask[qi][pi]) {
    coords[qi][pi] = 0.0;  // 隔离无效坐标
}
```
- **延迟**: 0 cycle（与PCB并行）
- **实现**: Kernel内逻辑，无需硬件调用

#### Stage 3: TBC (Tile Boundary Check) - 聚合决策 ✅
```cpp
// 硬件功能：边界检查 + 模式检测
__deform_tbc(coords_active) → (mode, base_x, base_y)
```
- **延迟**: 4 cycles (Phase1) + 3 cycles (Phase2)
- **模式**: Horizontal/Vertical/XOR/Discrete
- **实现**: `deform_tbc_impl()` with First-8 voting + bbox分析

#### Stage 4: TMA & Storage - Tile加载与存储 ✅
```cpp
// 硬件功能：协作式Tile加载
__deform_tma(base_tile, mode, level) → tile_loaded
```
- **TMA延迟**: 16 cycles
- **Storage延迟**: 6 cycles  
- **实现**: `deform_tma_impl()` + `deform_storage_impl()`
- **特性**: Swizzle布局 + Bank冲突避免

#### Stage 5: Interpolation - 双线性插值 ✅
```cpp
// 硬件功能：2×2邻域插值
__deform_interp(neighbors, frac_coords) → interpolated_value
```
- **延迟**: 3 cycles（固定）
- **实现**: `deform_interp_impl()`
- **路径**: Tile模式（smem）或Discrete模式（global）

---

## 第二部分：GPGPU-Sim集成现状

### 2.1 已完成工作 ✅

#### Function Call拦截机制（基于FMR成功经验）
```cpp
// 文件: src/cuda-sim/cuda-sim.cc (参考1863-1891行)
if (inst_opcode == CALL_OP && lane_id == 0) {
    std::string fname = target_func->get_name();
    if (fname.find("__deform_pcb") != std::string::npos) {
        deform_pcb_impl(pI, core, inst);
        skip = true;  // 跳过普通CALL处理
    }
    // 类似处理 __deform_tbc, __deform_tma, __deform_interp
}
```

#### 功能模型实现
- ✅ `src/cuda-sim/deform_attn_unit.h/.cc` - 5个模块类定义
- ✅ `src/cuda-sim/deform_attn_impl.cc` - 功能实现函数
- ✅ 延迟建模 + 参数读取 + 统计收集

#### 配置项注册
```bash
-gpgpu_deform_attn_avail 1
-gpgpu_num_deform_units 1  
-gpgpu_deform_pcb_latency 1
-gpgpu_deform_tbc_phase1_latency 4
-gpgpu_deform_tbc_phase2_latency 3
-gpgpu_deform_tma_latency 16
-gpgpu_deform_storage_latency 6
-gpgpu_deform_interp_latency 3
```

### 2.2 待完成工作 ⏳

#### GPGPU-Sim流水线集成（最高优先级）
```cpp
// 需要在 src/gpgpu-sim/shader.cc 中实现：

// 1. 执行单元创建
class deform_attn_exec_unit : public exec_unit_t {
    void cycle();
    bool can_issue(warp_inst_t &inst);
    // ...
};

// 2. 流水线初始化
void shader_core_ctx::create_exec_pipeline() {
    if (config->deform_attn_avail()) {
        m_deform_attn_unit = new deform_attn_exec_unit(this);
    }
}

// 3. 指令分发
void shader_core_ctx::issue_warp() {
    if (is_deform_attn_inst(inst) && m_deform_attn_unit->can_issue(inst)) {
        m_deform_attn_unit->issue(inst);
    }
}
```

#### 性能统计收集
- PCB剪枝命中率统计
- TBC模式分布统计（Horizontal/Vertical/XOR/Discrete比例）
- TMA Tile复用率统计
- 端到端延迟分析

---

## 第三部分：Kernel实现指导

### 3.1 数据布局与线程映射

#### 全局参数
```cpp
const int NUM_HEADS = 8;
const int NUM_LEVELS = 4;  
const int NUM_POINTS_PER_HEAD = 4;
const int TOTAL_POINTS_PER_QUERY = 128;  // 8×4×4
```

#### 张量布局
```cpp
float attn_weights[Q][H][L][P];     // 权重
float2 sampling_locs[Q][H][L][P];   // 采样坐标  
float* value_maps[L];               // 多尺度特征图
float output[Q];                    // 输出结果
```

### 3.2 Baseline Kernel（功能验证版）

#### 线程映射
```cpp
// 1 thread = 1 query，串行处理128 points
__global__ void deform_attn_baseline(
    float* value_maps[], 
    float2* sampling_locs,
    float* attn_weights,
    float* output
) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    
    for (int t = 0; t < 128; t++) {
        // 解包 (h,l,p) 索引
        int p = t % 4;
        int l = (t / 4) % 4;  
        int h = t / 16;
        
        float w = attn_weights[q*128 + t];
        float2 coord = sampling_locs[q*128 + t];
        
        // 直接全局内存采样（4次读取 + 插值）
        float val = bilinear_sample_global(value_maps[l], coord.x, coord.y);
        acc += val * w;
    }
    
    output[q] = acc;
}
```

### 3.4 CUDA优化要点（基于NVIDIA最佳实践）

#### 内存访问优化
```cpp
// ✅ 好的模式：合并访问
float2 coord = sampling_locs[q * 128 + t];  // 连续内存访问

// ❌ 避免的模式：跨步访问  
float2 coord = sampling_locs[q + t * NUM_QUERIES];  // 大跨步访问
```

#### 共享内存Bank冲突避免
```cpp
// ✅ 好的模式：添加padding
__shared__ float tile_buffer[16][17];  // +1 padding避免bank冲突

// ❌ 避免的模式：32的倍数维度
__shared__ float tile_buffer[16][32];  // 32-way bank conflict
```

#### Warp Divergence最小化
```cpp
// ✅ 好的模式：统一控制流
if (__all_sync(0xffffffff, !mask)) {
    // 整个warp都走Tile路径
    tile_based_processing();
} else if (__all_sync(0xffffffff, mask)) {
    // 整个warp都走Discrete路径
    discrete_processing();
} else {
    // 混合处理
    mixed_processing();
}
```

#### 寄存器压力管理
```cpp
// 限制循环展开，避免寄存器溢出
#pragma unroll 4  // 适度展开，不是越多越好
for (int i = 0; i < num_levels; i++) {
    process_level(i);
}
```

#### 线程映射与分块策略
```cpp
// Block粒度：16 queries × 256 threads
// Chunking：128 points/query ÷ 16 points/chunk = 8 chunks  
// 线程映射：tid → (q_local, t_in_chunk)
// 优化要点：确保内存访问合并与避免bank冲突

__global__ void deform_attn_optimized(
    float* value_maps[],
    float2* sampling_locs, 
    float* attn_weights,
    float* output,
    float threshold
) {
    // 线程与块映射
    int tid = threadIdx.x;                    // 0..255
    int q_local = tid / 16;                   // 0..15 (query within block)
    int t_in_chunk = tid % 16;                // 0..15 (point within chunk)
    int q = blockIdx.x * 16 + q_local;       // 全局query索引
    
    // 共享内存声明 - 添加padding避免bank冲突
    __shared__ float tile_buffer[16][17];     // +1 padding避免bank冲突
    __shared__ int shared_mode;               // TBC决策结果
    __shared__ int shared_base_x, shared_base_y;
    
    float acc_q = 0.0f;  // 当前线程的累加值
    
    // 处理8个chunks
    for (int chunk_id = 0; chunk_id < 8; chunk_id++) {
        int t = chunk_id * 16 + t_in_chunk;  // 全局point索引
        
        // === Stage I: PCB权重筛选 ===
        float w = attn_weights[q * 128 + t];
        bool mask = __deform_pcb(&w, threshold, true);  // 硬件调用
        
        // === Stage II: GTC操作数隔离 ===
        float2 coord;
        if (!mask) {
            coord = sampling_locs[q * 128 + t];
        } else {
            coord = make_float2(0.0f, 0.0f);  // 隔离无效坐标
        }
        
        // === Stage III: TBC聚合决策 ===
        // 收集有效坐标（优化：减少warp divergence）
        __shared__ float2 coords_buffer[256];
        __shared__ bool valid_buffer[256];
        
        // 使用连续内存访问模式，提高cache效率
        coords_buffer[tid] = coord;
        valid_buffer[tid] = !mask;
        __syncthreads();
        
        if (tid == 0) {
            // 构建有效坐标集合并调用TBC
            float2 active_coords[256];
            int num_active = 0;
            
            // 串行扫描避免atomic操作
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
            // 协作加载16×16 tile - 确保内存访问合并
            int tx = tid % 16;
            int ty = tid / 16;
            int gx = shared_base_x + tx;
            int gy = shared_base_y + ty;
            
            // 获取当前点的level
            int level = ((chunk_id * 16 + t_in_chunk) / 4) % 4;
            
            if (ty < 16) {  // 只用前256个线程加载，确保warp利用率
                float tile_val = __deform_tma(gx, gy, level, shared_mode);
                tile_buffer[ty][tx] = tile_val;  // 无bank冲突的写入模式
            }
            __syncthreads();
        }
        
        // === Stage V: Interpolation ===
        if (!mask) {  // 仅处理有效点
            float val;
            if (shared_mode != 3) {
                // Tile模式：从共享内存插值
                int lx = (int)coord.x - shared_base_x;
                int ly = (int)coord.y - shared_base_y;
                float fx = coord.x - (int)coord.x;
                float fy = coord.y - (int)coord.y;
                
                val = __deform_interp(tile_buffer, lx, ly, fx, fy, shared_mode);
            } else {
                // Discrete模式：直接全局内存采样
                int level = (t / 4) % 4;
                val = bilinear_sample_global(value_maps[level], coord.x, coord.y);
            }
            
            acc_q += val * w;
        }
    }
    
    // === 每query归约 + 写回 ===
    // 同一query的16个线程需要归约
    __shared__ float acc_shared[256];
    acc_shared[tid] = acc_q;
    __syncthreads();
    
    if (t_in_chunk == 0) {  // 每query选一个线程做归约
        float sum = 0.0f;
        for (int i = 0; i < 16; i++) {
            sum += acc_shared[q_local * 16 + i];
        }
        output[q] = sum;
    }
}
```

---

## 第四部分：实施计划与验证策略

### 4.1 实施优先级

#### Phase 1: GPGPU-Sim集成完成 ⏳
1. 实现 `deform_attn_exec_unit` 类
2. 集成到 `create_exec_pipeline()` 
3. 添加指令分发逻辑
4. 验证Function Call拦截工作正常

#### Phase 2: Kernel功能验证 📋
1. 实现Baseline kernel，验证数值正确性
2. 实现Optimized kernel基础版本（仅PCB剪枝）
3. 添加TBC决策与Tile/Discrete路径
4. 完整端到端测试

#### Phase 3: 性能优化与分析 📋
1. 添加详细性能统计
2. 延迟建模精度验证
3. 与真实硬件性能对比
4. 撰写性能分析报告

### 4.2 验证策略

#### 功能正确性验证
- Baseline vs Optimized数值对比
- 单模块功能测试（PCB/TBC/TMA/Interp）
- 边界条件测试

#### 性能模型验证  
- 延迟统计与预期对比
- 访存模式分析
- 能效比估算

---

## 第五部分：技术总结

### 5.1 关键技术决策

1. **Function Call拦截 vs PTX伪指令**：选择Function Call拦截，降低实现复杂度
2. **固定延迟 vs 动态建模**：采用固定延迟模型，专注功能验证
3. **16×16 processing tile**：与硬件模块粒度对齐，便于验证

### 5.2 成功经验借鉴

**FMR实现经验**：
- Lane 0执行 + 全warp PC同步
- 参数通过 `.param` 空间读取
- 功能模型与时序模型结合

### 5.3 未来扩展方向

- **动态延迟建模**：基于实际内存系统状态
- **多级优化**：L1/L2 Cache感知的Tile加载
- **能效分析**：功耗建模与能效比优化

---

**总结与下一步**：

本文档成功整合了DevDocs中分散的技术文档，实现了以下目标：

✅ **硬件架构与kernel实现对齐**：明确了5级流水线的硬件调用点与kernel代码结构的对应关系

✅ **GPGPU-Sim集成路径清晰**：基于FMR成功经验，制定了Function Call拦截到流水线集成的完整方案

✅ **CUDA最佳实践融入**：结合NVIDIA官方指南，优化了内存访问模式、bank冲突避免和warp divergence处理

✅ **实施优先级明确**：从GPGPU-Sim集成→功能验证→性能优化的清晰路线图

**关键技术贡献**：
- 16×16 processing tile与硬件模块粒度的精确对齐
- Baseline/Optimized两版本kernel的完整实现指导
- Function Call拦截机制的成熟应用
- 端到端延迟建模的可验证方案

**下一步工作**：专注于GPGPU-Sim流水线集成（shader.cc中的执行单元实现），这是当前的最高优先级任务。
