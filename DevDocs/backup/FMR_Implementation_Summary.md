# FMR (Feature Map Reorganizer) 实现总结

**Document Type:** FMR 指令实现技术总结
**Version:** v1.0 (2026-01-18)
**Scope:** FMR 指令的实现方式、集成点、技术细节

---

## 1. FMR 概述

### 1.1 设计目标
FMR (Feature Map Reorganizer) 是为 Deformable Attention 工作负载设计的**内存优化单元**，主要解决：
- **Tile 加载优化**：将离散的全局内存访问转换为连续的共享内存访问
- **Warp 协作**：32 线程协作加载 Tile 数据，提高内存带宽利用率
- **Bank 冲突避免**：通过交错布局减少共享内存 Bank 冲突

### 1.2 核心特性
- **TMA-like 设计**：类似 NVIDIA Tensor Memory Accelerator 的地址生成机制
- **内存操作标记**：作为内存操作（MEM__OP）而非特殊单元操作
- **动态延迟**：依赖实际内存系统（L1/L2/DRAM）而非固定延迟
- **功能模型 + 时序模型**：在 `ld_sample_fmr_impl()` 中同时实现功能正确性和时序模拟

---

## 2. 指令定义

### 2.1 操作码定义
**文件**: `src/cuda-sim/opcodes.def`
```cpp
OP_W_DEF(LD_SAMPLE_FMR_OP, ld_sample_fmr_impl, "ld.sample.fmr", 1, 5)
```

**关键点**:
- `OP_W_DEF`: Warp-level 指令（需要 `core_t*` 和 `warp_inst_t&` 参数）
- `LD_SAMPLE_FMR_OP`: 操作码枚举值
- `ld_sample_fmr_impl`: 实现函数
- `1`: 目标操作数数量
- `5`: 操作数分类（内存操作）

### 2.2 伪指令语法
**文件**: `src/cuda-sim/ptx.l`
```cpp
ld.sample.fmr TC; yylval->int_value = LD_SAMPLE_FMR_OP; return OPCODE;
```

---

## 3. CUDA Function Call 拦截机制

### 3.1 拦截点
**文件**: `src/cuda-sim/cuda-sim.cc` (1863-1891行)

**拦截逻辑**:
```cpp
// 在 ptx_thread_info::ptx_exec_inst() 中
if (inst_opcode == CALL_OP && lane_id == 0) {
    const operand_info &target = pI->func_addr();
    if (target.is_function_address()) {
        std::string fname = target_func->get_name();

        if (fname.find("fmr_sample") != std::string::npos) {
            // FMR 拦截处理
            core_t *core = get_core();
            ld_sample_fmr_impl(pI, core, inst);

            is_fmr_call = true;  // 标记为 FMR 调用
            skip = true;         // 跳过普通 CALL 处理
        }
    }
}
```

### 3.2 拦截策略
**关键设计决策**:
1. **Lane 0 执行**: 仅在 lane 0 执行 FMR 逻辑（warp 同步）
2. **跳过 CALL 语义**: 不压栈/弹栈，避免 callstack 混淆
3. **所有线程 PC 同步**: 所有线程都执行 `ptx_exec_inst()`，确保 PC 一致
4. **功能模型集成**: 在 `ld_sample_fmr_impl()` 中同时处理功能和时序

### 3.3 为什么需要拦截？
**问题**: CUDA Function Call 的参数传递机制
- 参数通过 `.param` 空间传递
- 普通 `CALL_OP` 会压栈/弹栈
- FMR 需要访问参数并生成内存访问，但不能作为真实函数调用

**解决方案**: 拦截后直接调用 `ld_sample_fmr_impl()`，跳过标准 CALL 处理

---

## 4. 功能模型实现

### 4.1 核心函数
**文件**: `src/cuda-sim/instructions.cc` (3585-3809行)

**函数签名**:
```cpp
void ld_sample_fmr_impl(const ptx_instruction *pI, core_t *core, warp_inst_t &inst);
```

### 4.2 参数读取
**参数位置** (从 `.param` 空间):
```cpp
// operand[1]: gmem_base_addr (u64) - 全局内存基地址
// operand[2]: smem_base_addr (u64) - 共享内存基地址
// operand[3]: width (s32)          - Tile 宽度
// operand[4]: height (s32)         - Tile 高度
// operand[5]: stride (s32)         - 源图像跨度
```

**读取方式**:
```cpp
thread->m_local_mem->read(gmem_base_op.get_symbol()->get_address(),
                          sizeof(unsigned long long), &gmem_base);
```

### 4.3 地址生成 (TMA-like)
**核心逻辑**:
```cpp
// Step 1: 生成所有 GMEM 地址
for (int row = 0; row < height; row++) {
    int loads_per_row = (width + 7) / 8;  // 每8个元素一次加载
    for (int load = 0; load < loads_per_row; load++) {
        int start_col = load * 8;
        addr_t gmem_addr = gmem_base + (row * stride + start_col) * elem_size;
        all_gmem_addrs.push_back(gmem_addr);
    }
}
```

**关键参数**:
- `elem_size = 2` (FP16 = 2 bytes)
- `MAX_ACCESSES_PER_INSN_PER_THREAD = 8` (每个线程槽最多8个地址)

### 4.4 批处理机制
**问题**: 地址数量可能超过 warp 线程数
**解决方案**: 批处理 + 线程槽复用

```cpp
// Step 2: 批处理地址到线程槽
int num_batches = (total_txns + MAX_ACCESSES_PER_INSN_PER_THREAD - 1) /
                  MAX_ACCESSES_PER_INSN_PER_THREAD;

// 限制到可用线程槽
if (num_batches > warp_size) {
    num_batches = warp_size;
}

// 批处理循环
for (int batch = 0; batch < num_batches; batch++) {
    // 准备批次地址数组
    new_addr_type batch_addrs[MAX_ACCESSES_PER_INSN_PER_THREAD];
    // ... 填充地址

    // 存储到线程槽
    inst.set_addr(batch, batch_addrs, (unsigned)batch_size);

    // 激活线程槽
    active_mask.set(batch);
}
```

**关键点**:
- 每个线程槽存储最多8个地址
- `inst.set_addr()` 用于存储地址
- `active_mask` 标记哪些线程槽有地址

### 4.5 功能模拟 (数据搬运)
**Step 3: 功能模拟 - Thread 0 处理所有数据**
```cpp
// Thread 0 执行所有数据搬运
thread = core->get_thread_info()[tid];  // Thread 0

for (size_t addr_idx = 0; addr_idx < all_gmem_addrs.size(); addr_idx++) {
    int row = addr_to_row_col[addr_idx].first;
    int start_col = addr_to_row_col[addr_idx].second;
    int end_col = std::min(start_col + 8, width);

    for (int col = start_col; col < end_col; col++) {
        addr_t gmem_read_addr = gmem_base + (row * stride + col) * elem_size;
        int tile_idx = row * width + col;
        addr_t smem_write_addr = smem_base + tile_idx * elem_size;

        // 读取 GMEM
        uint16_t data;
        gmem->read(gmem_read_addr, elem_size, &data);

        // 写入 SMEM
        smem->write(smem_write_addr, elem_size, &data, thread, pI);
    }
}
```

**关键点**:
- Thread 0 执行所有数据搬运（功能正确性）
- GMEM 读取：`gmem->read()`
- SMEM 写入：`smem->write()`
- 逻辑地址连续，硬件处理 Bank 映射

### 4.6 时序模型设置
**Step 4: 设置时序模型信息**
```cpp
inst.op = FMR_SAMPLE_OP;           // 操作码
inst.space = gmem_space;            // 内存空间
inst.data_size = 16;                // 128-bit = 16 bytes per transaction
inst.memory_op = memory_load;       // 内存操作类型
```

**关键点**:
- `inst.op = FMR_SAMPLE_OP`: 标记为 FMR 操作
- `inst.memory_op = memory_load`: 标记为加载操作
- `inst.data_size = 16`: 每次事务 128-bit
- 地址已通过 `inst.set_addr()` 存储，后续由 `generate_mem_accesses()` 处理

---

## 5. 时序模型集成

### 5.1 FMR 单元定义
**文件**: `src/gpgpu-sim/shader.h` (1269-1280行)

```cpp
class fmr_unit : public pipelined_simd_unit {
public:
    fmr_unit(register_set *result_port, const shader_core_config *config,
             shader_core_ctx *core, unsigned issue_reg_id);

    virtual void issue(register_set &source_reg);
    virtual void active_lanes_in_pipeline();
};
```

### 5.2 FMR 单元实现
**文件**: `src/gpgpu-sim/shader.cc` (2441-2497行)

**关键实现**:
```cpp
void fmr_unit::issue(register_set &source_reg) {
    warp_inst_t **ready_reg = source_reg.get_ready(m_config->sub_core_model, m_issue_reg_id);

    // FMR 是内存操作，不是特殊单元操作
    (*ready_reg)->op_pipe = MEM__OP;

    // 关键：不计算固定延迟，依赖实际内存系统
    // 内存事务已在 instructions.cc 中生成
    // 实际延迟由 L1/L2/DRAM 决定

    pipelined_simd_unit::issue(source_reg);
}
```

**设计决策**:
- **MEM__OP**: FMR 作为内存操作，通过 `ldst_unit` 处理
- **无固定延迟**: 不在 `issue()` 中计算延迟，依赖实际内存系统
- **TMA-like**: 类似 Tensor Memory Accelerator 的设计

### 5.3 流水线配置
**文件**: `src/gpgpu-sim/gpu-sim.cc` (626-643行)

```cpp
// FMR 配置选项
option_parser_register(opp, "-gpgpu_fmr_avail", OPT_UINT32,
                       &gpgpu_fmr_avail,
                       "FMR Available (default=0)", "0");

option_parser_register(opp, "-gpgpu_num_fmr_units", OPT_UINT32,
                       &gpgpu_num_fmr_units,
                       "Number of FMR units (default=1)", "0");

option_parser_register(opp, "-gpgpu_fmr_latency", OPT_UINT32, &fmr_latency,
                       "FMR operation latency in cycles (default=4)", "4");
```

**配置示例**:
```
-gpgpu_fmr_avail 1
-gpgpu_num_fmr_units 1
-gpgpu_fmr_latency 4
```

### 5.4 流水线阶段
**文件**: `src/gpgpu-sim/shader.h` (1523-1524行)

```cpp
enum pipeline_stage_t {
    // ... 其他阶段
    ID_OC_FMR,        // Issue → Operand Collector (FMR)
    OC_EX_FMR,        // Operand Collector → Execute (FMR)
    // ... 其他阶段
};
```

**流水线路径**:
```
ID_OC_FMR → OC_EX_FMR → EX_WB
```

---

## 6. 内存系统集成

### 6.1 内存事务生成
**文件**: `src/gpgpu-sim/shader.cc` (1509-1524行)

```cpp
if ((pI->op == FMR_SAMPLE_OP) &&
    m_fmr_out->has_free(m_shader->m_config->sub_core_model, m_issue_reg_id)) {

    bool fmr_pipe_avail = (m_shader->m_config->gpgpu_num_fmr_units > 0) && ...;

    if (fmr_pipe_avail) {
        m_shader->issue_warp(*m_fmr_out, pI, active_mask, warp_id, ...);
        previous_issued_inst_exec_type = exec_unit_type_t::FMR;
    }
}
```

### 6.2 内存访问流程
**完整路径**:
```
1. instructions.cc: ld_sample_fmr_impl()
   - 生成 GMEM 地址
   - 存储到 inst.set_addr()
   - 设置 inst.op = FMR_SAMPLE_OP

2. shader.cc: warp_scheduler::issue_warp()
   - 选择 FMR 执行单元
   - 发送到 ID_OC_FMR 寄存器

3. operand_collector: issue()
   - 收集操作数
   - 发送到 OC_EX_FMR

4. fmr_unit: issue()
   - 标记为 MEM__OP
   - 转发到内存系统

5. ldst_unit: 处理内存事务
   - L1 Cache 访问
   - L2 Cache 访问
   - DRAM 访问（如需要）

6. 写回 SMEM
   - 通过 ldst_unit 完成
   - 更新共享内存
```

### 6.3 内存操作类型
**关键点**:
- **GMEM → SMEM**: FMR 是 DMA-like 操作
- **加载操作**: `inst.memory_op = memory_load`
- **128-bit 事务**: `inst.data_size = 16` bytes
- **批量处理**: 多个地址通过线程槽存储

---

## 7. 关键技术细节

### 7.1 地址生成策略
**TMA-like 设计**:
- **集中式地址生成**: Thread 0 生成所有地址
- **批量存储**: 地址分批存储到线程槽
- **无冗余计算**: 避免每个线程重复计算地址

**批处理逻辑**:
```cpp
// 每个线程槽最多存储 8 个地址
int loads_per_row = (width + 7) / 8;  // Ceiling division
total_txns = height * loads_per_row
num_batches = ceil(total_txns / 8)
```

### 7.2 线程槽管理
**active_mask 使用**:
```cpp
active_mask_t active_mask;
active_mask.reset();

for (int batch = 0; batch < num_batches; batch++) {
    // 存储地址到线程槽
    inst.set_addr(batch, batch_addrs, batch_size);

    // 激活线程槽
    active_mask.set(batch);
}

// FIX: 不修改 inst.set_active()
// inst.set_active(active_mask);  // ❌ 错误：会导致 PC 不同步
```

**为什么不能修改 set_active()**:
- 所有线程都需要执行 `ptx_exec_inst()` 来推进 PC
- 只有带地址的线程槽才会生成内存访问
- `generate_mem_accesses()` 会自然跳过无地址的线程

### 7.3 功能正确性保证
**Thread 0 执行所有数据搬运**:
```cpp
// Thread 0 读取所有 GMEM 数据
// Thread 0 写入所有 SMEM 数据
// 确保功能正确性，不受批处理限制
```

**优势**:
- 功能正确性不依赖批处理
- 即使批处理被截断，功能仍正确
- 时序模型可能受影响，但功能不受影响

### 7.4 Bank 冲突避免
**交错布局** (在 SMEM 中):
```cpp
// 逻辑地址连续，硬件处理 Bank 映射
// Row[even] → Banks[0,1]
// Row[odd]  → Banks[2,3]
```

**实现方式**:
- 逻辑地址: `tile_idx = row * width + col`
- 物理 Bank: 由 SMEM 硬件控制器处理
- 无需软件干预

---

## 8. 配置和使用

### 8.1 编译配置
**启用 FMR**:
```bash
# 在 gpgpusim.config 中添加
-gpgpu_fmr_avail 1
-gpgpu_num_fmr_units 1
-gpgpu_fmr_latency 4
```

### 8.2 CUDA 调用示例
**CUDA 代码**:
```cpp
// 定义 FMR 函数原型
extern "C" __device__ void __fmr_sample(
    void* gmem_base,
    void* smem_base,
    int width,
    int height,
    int stride
);

// 使用示例
__global__ void deform_attn_kernel(...) {
    extern __shared__ float smem[];

    // 调用 FMR 加载 Tile
    __fmr_sample(gmem_tile, smem_tile, width, height, stride);

    // 继续处理...
}
```

### 8.3 调试选项
**启用调试输出**:
```cpp
// 在代码中设置
core->get_gpu()->gpgpu_ctx->debug_tensorcore = true;
```

**调试输出包括**:
- Tile 参数（gmem/smem 地址、宽高、跨度）
- 批处理信息（总事务数、批次数、激活线程槽）
- Bank 分布统计

---

## 9. 性能统计

### 9.1 统计信息
**文件**: `src/abstract_hardware_model.h` (1078-1083行)

```cpp
// FMR tile metadata
int m_fmr_tile_width;   // tile width (number of f16 elements per row)
int m_fmr_tile_height;  // tile height (number of rows)
int m_fmr_stride;       // stride for GMEM access (source image width)
addr_t m_fmr_gmem_base; // global memory base address
addr_t m_fmr_smem_base; // shared memory base address
```

### 9.2 性能指标
**可收集的统计**:
- FMR 操作次数
- 总内存事务数
- 平均 Tile 尺寸
- 批处理分布
- 内存带宽利用率

---

## 10. 与 Deformable Attention 的关系

### 10.1 FMR 在 Deformable Attention 中的角色
**Deformable Attention 流水线**:
```
GPU Pipeline (外部) → [1] PCB → [2] GTC → [3] TBC → [4] FMR (TMA) → [5] Interpolation
```

**FMR 职责**:
- **TMA-like Tile 加载**: 将离散 GMEM 访问转换为连续 SMEM 访问
- **内存优化**: 提高带宽利用率，减少 DRAM 访问
- **功能模型**: 提供功能正确性 + 时序模拟

### 10.2 与未来 DeformAttn 功能模型的关系
**FMR 是 DeformAttn 的子集**:
- FMR: 专注于 Tile 加载（TMA-like）
- DeformAttn: 完整的 5 级流水线（PCB → GTC → TBC → TMA+Storage → Interpolation）

**扩展方向**:
- FMR 可以扩展为完整的 TMA+Storage 模块
- 保持相同的拦截机制和内存集成方式
- 添加自适应布局和 Bank 映射

---

## 11. 总结

### 11.1 核心设计
1. **CUDA Function Call 拦截**: 通过函数名匹配拦截 `__fmr_sample()`
2. **TMA-like 地址生成**: Thread 0 生成所有地址，批量存储到线程槽
3. **功能 + 时序集成**: 在 `ld_sample_fmr_impl()` 中同时处理
4. **内存操作标记**: 作为 MEM__OP 通过 ldst_unit 处理
5. **动态延迟**: 依赖实际内存系统而非固定延迟

### 11.2 关键技术点
- **批处理机制**: 解决地址数量超过线程数的问题
- **Thread 0 执行**: 保证功能正确性
- **不修改 set_active()**: 保证 PC 同步
- **MEM__OP 标记**: 通过标准内存路径处理

### 11.3 优势
- ✅ 功能正确性保证
- ✅ 时序模型准确（依赖实际内存系统）
- ✅ 与现有 GPGPU-Sim 内存系统无缝集成
- ✅ 可扩展性强（可扩展为完整 DeformAttn 模块）

### 11.4 局限性
- ⚠️ 批处理可能截断（地址数 > warp_size * 8）
- ⚠️ 无自适应 Bank 映射（依赖硬件控制器）
- ⚠️ 无 CAM 管理（简单批处理）

---

## 12. 相关文件

### 12.1 核心实现
- `src/cuda-sim/opcodes.def` - 操作码定义
- `src/cuda-sim/ptx.l` - 伪指令语法
- `src/cuda-sim/cuda-sim.cc` - 拦截逻辑 (1863-1891行)
- `src/cuda-sim/instructions.cc` - 功能模型 (3585-3809行)

### 12.2 时序模型
- `src/gpgpu-sim/shader.h` - FMR 单元定义 (1269-1280行)
- `src/gpgpu-sim/shader.cc` - FMR 单元实现 (2441-2497行)
- `src/gpgpu-sim/gpu-sim.cc` - 配置选项 (626-643行)

### 12.3 数据结构
- `src/abstract_hardware_model.h` - FMR 元数据 (1078-1083行)
- `src/abstract_hardware_model.cc` - FMR 操作码检查 (289行)

---

**文档历史**：
- v1.0 (2026-01-18): 初始版本，基于代码分析创建