# Deformable Attention GPGPU-Sim 架构设计

**Document Type:** 架构设计文档
**Version:** v1.0 (2026-01-18)
**Scope:** 系统架构、模块设计、数据流、设计决策

---

## 1. 设计概述

### 1.1 设计目标

本设计旨在在 GPGPU-Sim 中实现 Deformable Attention 的功能模型，解决传统 GPU 模拟中 Deformable Attention 算子的两大挑战：

1. **稀疏采样的随机访存**：离散的采样点导致访存碎片化，带宽利用率低
2. **低权重点的无效计算**：大量低重要性采样点消耗计算资源但对结果贡献小

### 1.2 核心设计思想

采用 **"Predict-then-Execute"（先预测，后执行）** 的架构理念：

- **前端优化**：通过**权重预测**生成稀疏掩码，使用**操作数隔离**技术降低无效计算
- **后端优化**：通过**访存聚合**和**自适应存储布局**技术，优化访存模式

### 1.3 GPGPU-Sim 集成理念

**功能模型定位**：在 GPGPU-Sim 中实现 Deformable Attention 的功能模型
- 使用 CUDA Function Call 拦截机制
- 固定延迟模拟（无时序级实现）
- 模拟目标：功能正确性验证、端到端延迟估算、性能数据收集

---

## 2. 系统架构

### 2.1 处理粒度

以 **Query Tile** 为基本处理单元：
- 每个 Tile 包含多个查询点（典型配置：16 queries）
- 每个查询对应多级特征图的多个采样点（典型：4 levels × 4 points）
- 总采样点数：数百个离散空间位置

### 2.2 5 级流水线架构

```
GPU Pipeline (外部) → [1] PCB → [2] GTC → [3] TBC → [4] TMA & Storage → [5] Interpolation
```

**GPU Pipeline（前置）**：
- 标准 MMA 计算：Query × Weight → Attention Weights
- 标准 MMA 计算：Query × Offset Weights → Offsets
- 基础坐标运算：Reference + Offset → Absolute Coordinates

**功能模型（加速器）**：
- **[1] Pre-Check Block (PCB)**：权重过滤与掩码生成
- **[2] Gated Tensor Core (GTC)**：稀疏计算与功耗优化
- **[3] Tile Boundary Check (TBC)**：坐标验证、聚合决策与模式分析
- **[4] TMA & Storage**：自适应Tile加载与无冲突存储
- **[5] Interpolation**：无冲突插值计算

### 2.3 数据流

```
GPU Pipeline
  ├─> PCB (权重 → 稀疏掩码)
  ├─> GTC (掩码 + 坐标 → 有效坐标)
  ├─> TBC (坐标 → Tile 参数)
  ├─> TMA & Storage (Tile 参数 → Tile 数据)
  └─> Interpolation (Tile 数据 + 坐标 → 输出)
```

---

## 3. 模块详细设计

### 3.1 PCB (Pre-Check Block) - 权重预筛选

**功能**: 权重过滤，生成稀疏掩码

**输入**:
- `weights[16][16]`: FP16 权重矩阵
- `threshold`: 剪枝阈值
- `enable`: 使能标志

**输出**:
- `mask[16][16]`: 稀疏掩码（1=跳过，0=计算）

**算法**:
```cpp
for (qi, pi) in 16×16:
    mask[qi][pi] = (|weight[qi][pi]| < threshold) ? 1 : 0
```

**延迟**: 1 cycle（固定）

**实现要点**:
- 从 `.param` 空间读取权重和阈值
- 生成掩码并存储到输出缓冲区
- 作为计算操作（非内存操作）

---

### 3.2 GTC (Gated Tensor Core) - 稀疏感知计算

**功能**: 操作数隔离，标记有效坐标

**输入**:
- `mask[16][16]`: PCB 生成的稀疏掩码
- `coordinates[16][16][2]`: 2D 坐标

**输出**:
- `valid_coords[16][16]`: 有效坐标标记

**算法**:
```cpp
for (qi, pi) in 16×16:
    if (mask[qi][pi] == 0):  // 有效点
        valid_coords[qi][pi] = 1
    else:                    // 无效点
        valid_coords[qi][pi] = 0
        coordinates[qi][pi] = 0  // 操作数隔离
```

**延迟**: 0 cycle（与 PCB 并行）

**实现要点**:
- 与 PCB 并行处理
- 不需要独立的流水线阶段
- 可以在 PCB 的输出阶段直接处理

---

### 3.3 TBC (Tile Boundary Check) - 自适应聚合决策

**功能**: 两阶段决策，判断是否适合 Tile 加载

**Phase 1: 快速聚集判断**
- 输入: `abs_coords[16][16]`, `valid_coords[16][16]`
- 输出: `mode` (0=Horizontal, 1=Vertical, 2=XOR, 3=Discrete)
- 延迟: 4 cycles

**Phase 2: 精细模式分析** (仅 Phase 1 通过后执行)
- 输入: Phase 1 的结果
- 输出: `tile_size`, `base_tile_x`, `base_tile_y`
- 延迟: +3 cycles

**算法**:
```cpp
// Phase 1: First-8 Voting
1. 边界检查（整数比较）
2. 提取前 8 个有效点
3. 计算锚点 Tile 的命中率
4. 判断：命中率 < 50% → mode=3 (Discrete)
        命中率 ≥ 50% → 进入 Phase 2

// Phase 2: 模式检测
1. 计算前 8 点的包围盒（min_x, max_x, min_y, max_y）
2. 计算 delta_x, delta_y
3. 检测模式：
   - delta_x > 2×delta_y → mode=0 (Horizontal)
   - delta_y > 2×delta_x → mode=1 (Vertical)
   - 其他 → mode=2 (XOR)
```

**延迟**: 4 cycles（低聚集度）或 7 cycles（高聚集度）

**实现要点**:
- 整数比较和计算
- 简单的包围盒计算
- 模式检测逻辑

---

### 3.4 TMA & Storage - 自适应存储子系统

**功能**: Tile 加载 + 无冲突存储管理

**TMA (Tile Memory Access)**:
- 输入: `base_tile_x`, `base_tile_y`, `tile_size`, `mode`
- 输出: Tile 数据（加载到 Shared Memory）
- 延迟: 16 cycles（16 行 × 1 cycle）

**Storage (Tracker + Bank Mapping)**:
- 输入: Warp 查询（32 线程 × 2×2 像素）
- 输出: 2×2 邻域数据
- 延迟: 6 cycles

**算法**:
```cpp
// TMA: Tile 加载
for row_id in 0..15:
    addr = base_addr + row_id * stride
    tile_data[row_id] = memory_read(addr, 32 bytes)
    apply_swizzle(tile_data[row_id], mode)

// Storage: Bank 映射
bank_id = (word_offset XOR xor_key) % 32
```

**延迟**: 16 + 6 = 22 cycles

**实现要点**:
- 作为内存操作（MEM__OP）
- 通过内存系统访问
- 自适应 Bank 映射

---

### 3.5 Interpolation - 双线性插值

**功能**: 无冲突双线性插值计算

**输入**:
- `p00, p01, p10, p11[16][16]`: 2×2 邻域像素
- `wx, wy[16][16]`: 插值权重

**输出**:
- `result[16][16]`: 插值结果

**算法**:
```cpp
for (qi, pi) in 16×16:
    result[qi][pi] = p00×(1-wx)×(1-wy) + p01×wx×(1-wy) +
                     p10×(1-wx)×wy + p11×wx×wy
```

**延迟**: 3 cycles

**实现要点**:
- 浮点运算
- 固定延迟
- 无冲突读取（依赖 Storage）

---

## 4. GPGPU-Sim 集成点

### 4.1 核心集成点

| 组件 | 集成方式 | 说明 |
|------|---------|------|
| Function Call 拦截 | `src/cuda-sim/cuda-sim.cc` | 拦截 `__deform_*()` 函数调用 |
| 功能模型实现 | `src/cuda-sim/deform_attn_impl.cc` | 4 个实现函数 |
| 时序模型 | `src/cuda-sim/deform_attn_unit.cc` | 5 个模块 + Top-Level |
| 流水线集成 | `src/gpgpu-sim/shader.cc` | 集成到 `shader_core_ctx` |
| 配置选项 | `src/gpgpu-sim/gpu-sim.cc` | 10 个配置项 |
| 性能统计 | `src/gpgpu-sim/stats.cc` | 收集性能数据 |

### 4.2 数据结构

#### Deformable Attention 请求
```cpp
struct deform_attn_request {
    unsigned query_tile_id;        // Query Tile ID
    unsigned num_queries;          // 典型 16
    unsigned num_points_per_query; // 典型 16

    // PCB 数据
    float* weights;                // 16×16 FP16
    float threshold;
    bool enable_prune;

    // TBC 数据
    float* abs_coords;             // 16×16 × 2D
    bool* valid_coords;            // 16×16

    // TMA 数据
    addr_t row_base_addr;
    unsigned tile_x, tile_y;
    unsigned pitch;
    unsigned mode;                 // 0:Horizontal, 1:Vertical, 2:XOR, 3:Discrete
    unsigned tile_size;            // 0:16×16, 1:16×32, 2:32×16

    // 输出
    float* output;                 // 16×16 FP16
};
```

#### 功能单元状态
```cpp
class deform_attn_unit {
private:
    // 子单元
    deform_pcb_model pcb;
    deform_tbc_model tbc;
    deform_tma_model tma;
    deform_storage_model storage;
    deform_interp_model interp;

    // 请求队列
    std::queue<deform_attn_request*> request_queue;

    // 当前处理状态
    enum stage_t {IDLE, PCB, GTC, TBC_PHASE1, TBC_PHASE2,
                  TMA, STORAGE, INTERP, COMPLETE};
    stage_t m_current_stage;
    unsigned m_remaining_cycles;

    // 中间结果
    bool m_mask[16][16];
    bool m_valid_coords[16][16];
    deform_tbc_model::mode_t m_mode;
    deform_tbc_model::tile_size_t m_tile_size;
    unsigned m_base_tile_x;
    unsigned m_base_tile_y;
    float m_tile_data[16][16];
    float m_output[16][16];

    // 性能统计
    struct stats_t {
        unsigned num_ops;
        unsigned num_cycles;
        unsigned num_sparse_skip;
        unsigned num_discrete_fallback;
        unsigned num_tile_loads;
        unsigned num_high聚集度;
        unsigned num_low聚集度;
    } m_stats;

public:
    void issue(deform_attn_request* req);
    void cycle();
    bool is_idle();
    void collect_stats();
};
```

---

## 5. 关键技术点

### 5.1 CUDA Function Call 拦截

**拦截点**: `src/cuda-sim/cuda-sim.cc` (1896-1938行)

**拦截逻辑**:
```cpp
if (inst_opcode == CALL_OP && lane_id == 0) {
    std::string fname = target_func->get_name();

    if (fname.find("deform_pcb") != std::string::npos) {
        deform_pcb_impl(pI, core, inst);
        is_deform_call = true;
        skip = true;
    }
    // ... 其他函数类似
}
```

**关键点**:
- Lane 0 执行
- 跳过 CALL 语义（不压栈/弹栈）
- 所有线程 PC 同步
- **不需要伪指令接口**（仅 Function Call 拦截）

---

### 5.2 参数读取

**参考 FMR 实现**:
```cpp
// 从 .param 空间读取
const operand_info &weights_op = pI->operand_lookup(1);
thread->m_local_mem->read(weights_op.get_symbol()->get_address(),
                          16*16*sizeof(float), weights);
```

**应用到 DeformAttn**:
```cpp
// PCB 参数读取
const operand_info &weights_op = pI->operand_lookup(1);
const operand_info &threshold_op = pI->operand_lookup(2);
const operand_info &enable_op = pI->operand_lookup(3);

thread->m_local_mem->read(weights_op.get_symbol()->get_address(),
                          sizeof(float) * 16 * 16, weights);
thread->m_local_mem->read(threshold_op.get_symbol()->get_address(),
                          sizeof(float), &threshold);
```

---

### 5.3 固定延迟实现

**在构造函数中指定固定延迟**:
```cpp
// PCB 单元（固定 1 cycle）
deform_pcb_unit::deform_pcb_unit(...)
    : pipelined_simd_unit(result_port, config, 1, core, issue_reg_id) {
    m_name = "DEFORM_PCB";
}

// TBC 单元（固定 4 或 7 cycles）
deform_tbc_unit::deform_tbc_unit(...)
    : pipelined_simd_unit(result_port, config, 4, core, issue_reg_id) {
    m_name = "DEFORM_TBC";
}
```

**关键点**:
- 在构造函数中指定固定延迟
- 不需要动态延迟计算
- 简化实现

---

### 5.4 内存操作标记（TMA）

**TMA 作为内存操作**:
```cpp
inst.op = DEFORM_TMA_OP;
inst.memory_op = memory_load;
inst.data_size = 32;  // 每行 32 bytes
```

**关键点**:
- 作为 `MEM__OP` 通过内存系统
- 设置 `inst.op` 和 `inst.memory_op`
- **固定延迟**：TMA 延迟 16 cycles（配置项）
- **不依赖**实际内存延迟变化

---

## 6. 流水线集成架构

### 6.1 GPU 协作模型

**核心思想**：与标准 GPU 计算流程协作

- **职责分离**：
  - GPU负责：标准MMA计算、基础坐标运算
  - 加速器负责：稀疏优化、访存加速

- **收益**：
  - 简化加速器设计复杂度
  - 提升系统集成灵活性
  - 便于适配不同GPU架构

### 6.2 流水线阶段

**添加的流水线阶段**:
- `ID_OC_DEFORM`: Issue → Operand Collector (DeformAttn)
- `OC_EX_DEFORM`: Operand Collector → Execute (DeformAttn)

**执行单元**:
- `deform_attn_exec_unit`: 继承自 `pipelined_simd_unit`
- 固定延迟：33 cycles（最大）
- 操作类型：SPECIALIZED__OP

### 6.3 指令分发

**在 `issue_warp()` 中**:
```cpp
// Deformable Attention instruction dispatch
if ((pI->op == DEFORM_PCB_OP || pI->op == DEFORM_TBC_OP ||
     pI->op == DEFORM_TMA_OP || pI->op == DEFORM_INTERP_OP) &&
    m_deform_out->has_free(m_shader->m_config->sub_core_model, m_issue_reg_id)) {

    bool deform_pipe_avail = (m_shader->m_config->gpgpu_num_deform_units > 0) &&
                             m_deform_out->has_free(m_shader->m_config->sub_core_model,
                                                    m_issue_reg_id);

    if (deform_pipe_avail) {
        m_shader->issue_warp(*m_deform_out, pI, active_mask, warp_id, ...);
        previous_issued_inst_exec_type = exec_unit_type_t::DEFORM;
    }
}
```

---

## 7. 自适应优化机制

### 7.1 TBC 自适应决策

**两阶段决策**:
1. **Phase 1**: 快速聚集判断（4 cycles）
   - First-8 Voting
   - 命中率 < 50% → 回退离散加载

2. **Phase 2**: 精细模式分析（+3 cycles）
   - 包围盒计算
   - 模式检测（Horizontal/Vertical/XOR）
   - Tile 尺寸选择

**自适应行为**:
- 低聚集度：~30 cycles（离散加载）
- 高聚集度：~33 cycles（Tile 加载 + 自适应映射）

### 7.2 Storage 自适应 Bank 映射

**三层自适应**:
1. **Tile 尺寸自适应**: 16×16 / 16×32 / 32×16
2. **布局模式自适应**: Horizontal / Vertical / XOR
3. **Bank 映射自适应**: `bank_id = (word_offset XOR xor_key) % 32`

**无冲突保证**:
- 相邻行的像素映射到不同 Bank
- 单线程邻域访问 100% 无冲突

---

## 8. 配置选项

### 8.1 GPGPU-Sim 配置

**添加的配置选项**:
```
-gpgpu_deform_attn_avail 1
-gpgpu_num_deform_units 1
-gpgpu_deform_pcb_latency 1
-gpgpu_deform_tbc_phase1_latency 4
-gpgpu_deform_tbc_phase2_latency 3
-gpgpu_deform_tma_latency 16
-gpgpu_deform_storage_latency 6
-gpgpu_deform_interp_latency 3
-gpgpu_operand_collector_num_in_ports_deform 1
-gpgpu_operand_collector_num_out_ports_deform 1
```

### 8.2 配置说明

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `gpgpu_deform_attn_avail` | 0 | 是否启用 DeformAttn 单元 |
| `gpgpu_num_deform_units` | 1 | DeformAttn 单元数量 |
| `deform_pcb_latency` | 1 | PCB 延迟（固定） |
| `deform_tbc_phase1_latency` | 4 | TBC Phase 1 延迟（固定） |
| `deform_tbc_phase2_latency` | 3 | TBC Phase 2 延迟（固定） |
| `deform_tma_latency` | 16 | TMA 延迟（固定） |
| `deform_storage_latency` | 6 | Storage 延迟（固定） |
| `deform_interp_latency` | 3 | Interpolation 延迟（固定） |

---

## 9. 关键设计决策

### 9.1 固定延迟 vs 动态延迟

**选择固定延迟的原因**:
- 简化实现复杂度
- 专注于功能正确性
- 足够用于性能估算
- 避免复杂的内存系统建模

**权衡**:
- ✅ 简化实现
- ✅ 可预测的延迟
- ❌ 不如动态延迟准确

### 9.2 Function Call 拦截 vs 伪指令

**选择 Function Call 拦截的原因**:
- FMR 的伪指令 `ld.sample.fmr` 实际未使用
- 实际使用的是 `__fmr_sample()` 函数调用拦截
- 简化实现，避免伪指令复杂性
- 更自然的 CUDA 编程模型

**优势**:
- ✅ 简化接口设计
- ✅ 避免伪指令定义
- ✅ 更直观的使用方式

### 9.3 模块化设计

**5 个独立模块**:
1. PCB: 权重过滤
2. GTC: 操作数隔离
3. TBC: 自适应决策
4. TMA & Storage: 内存管理
5. Interpolation: 计算

**优势**:
- ✅ 每个模块独立测试
- ✅ 易于理解和维护
- ✅ 便于扩展和优化

### 9.4 自适应优化

**自适应行为**:
- TBC 根据聚集度自适应选择模式
- Storage 根据模式自适应选择 Bank 映射
- TMA 根据 tile_size 自适应选择尺寸

**优势**:
- ✅ 适应不同工作负载
- ✅ 优化内存访问模式
- ✅ 消除 Bank 冲突

---

## 10. 性能指标

### 10.1 延迟预算

| 模块 | 延迟 | 说明 |
|------|------|------|
| PCB | 1 cycle | 权重过滤 |
| GTC | 0 cycle | 与 PCB 并行 |
| TBC Phase 1 | 4 cycles | 快速聚集判断 |
| TBC Phase 2 | +3 cycles | 精细模式分析（仅高聚集度）|
| TMA | 16 cycles | Tile 加载（16 行）|
| Storage | 6 cycles | Tracker + Bank 映射 |
| Interpolation | 3 cycles | 双线性插值 |
| **总延迟** | **~30-33 cycles** | 根据场景自适应 |

### 10.2 预期性能提升

| 指标 | 基线 | DeformAttn | 提升 |
|------|------|-----------|------|
| 稀疏跳过率 | 0% | 0-70% | 动态 |
| DRAM 访问 | 100% | 12-25% | 4-8× 减少 |
| Bank 冲突 | 68.3% | 0% | 完全消除 |
| 插值吞吐 | 1/4 cycles | 1/1 cycles | 4× 提升 |

---

## 11. 文件结构

### 11.1 核心实现文件

```
src/cuda-sim/
├── deform_attn_unit.h          # 单元定义（5 个模块 + Top-Level）
├── deform_attn_unit.cc         # 单元实现（固定延迟）
├── deform_attn_impl.cc         # 功能模型实现函数
├── cuda-sim.cc                 # Function Call 拦截逻辑
└── Makefile                    # 编译配置

src/gpgpu-sim/
├── shader.h                    # 流水线阶段、配置、执行单元声明
├── shader.cc                   # 集成到 shader_core_ctx（待完成）
├── gpu-sim.cc                  # 配置选项注册
└── stats.cc                    # 性能统计（待完成）
```

### 11.2 测试文件

```
deAttn_fp16_fmr/
├── test_deform_attn.cu         # 测试 CUDA Kernel
├── Makefile.test               # 测试 Makefile
└── run.sh                      # 运行脚本
```

---

## 12. 开发建议

### 12.1 实现顺序

1. **Phase 1**: Function Call 拦截（已完成）
2. **Phase 2**: 功能模型实现（已完成）
3. **Phase 3**: shader.cc 集成（进行中）
4. **Phase 4**: 性能统计（待完成）
5. **Phase 5**: 编译测试（待完成）
6. **Phase 6**: 验证调试（待完成）

### 12.2 调试技巧

- 启用 `debug_tensorcore` 标志
- 使用 `printf` 输出中间结果
- 对比 Golden 参考值
- 从小规模测试开始

### 12.3 测试策略

- 每个模块单独测试
- 小规模集成测试
- 逐步增加复杂度
- 对比预期延迟

---

## 13. 总结

### 13.1 核心特性

1. **固定延迟模型**: 每个模块有固定的 cycle 延迟
2. **Function Call 拦截**: 简化接口设计
3. **自适应优化**: TBC 和 Storage 自适应行为
4. **模块化设计**: 5 个独立模块
5. **无冲突存储**: 自适应 Bank 映射

### 13.2 关键优势

- ✅ 功能正确性保证
- ✅ 可预测的延迟
- ✅ 自适应优化
- ✅ 模块化设计
- ✅ 易于集成

### 13.3 设计权衡

- **固定延迟**: 简化实现 vs 动态延迟准确性
- **Function Call 拦截**: 简化接口 vs 伪指令灵活性
- **功能模型**: 专注于正确性 vs 时序级准确性

---

## 14. 相关文档

- `DeformAttn_Development_Guide.md` - 开发指南
- `DeformAttn_Progress_Tracker.md` - 进度跟踪
- `FMR_Implementation_Summary.md` - FMR 实现参考
- `deAttn/README.md` - Deformable Attention 算法说明

---

**文档历史**：
- v1.0 (2026-01-18): 初始版本，合并自 Golden.md 和 DeformAttn_Implementation_Plan.md