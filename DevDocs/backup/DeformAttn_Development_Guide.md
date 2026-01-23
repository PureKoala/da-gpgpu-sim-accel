# Deformable Attention 开发指南

**Document Type:** 开发指南与测试策略
**Version:** v1.0 (2026-01-18)
**Scope:** 实现步骤、测试策略、配置选项、使用示例

---

## 1. 快速开始

### 1.1 已完成工作 ✅

**核心实现文件**:
- ✅ `src/cuda-sim/deform_attn_unit.h/.cc` - 5 个模块 + Top-Level
- ✅ `src/cuda-sim/deform_attn_impl.cc` - 功能模型实现函数
- ✅ `src/cuda-sim/cuda-sim.cc` - CUDA Function Call 拦截逻辑
- ✅ `src/gpgpu-sim/shader.h` - 流水线阶段和配置
- ✅ `src/gpgpu-sim/gpu-sim.cc` - 配置选项注册
- ✅ `src/cuda-sim/Makefile` - 编译配置
- ✅ `deAttn_fp16_fmr/test_deform_attn.cu` - 测试 CUDA Kernel
- ✅ `deAttn_fp16_fmr/Makefile.test` - 测试 Makefile

**实现的模块**:
1. **PCB (Pre-Check Block)** - 权重预筛选（1 cycle）
2. **GTC (Gated Tensor Core)** - 稀疏感知计算（0 cycle，并行）
3. **TBC (Tile Boundary Check)** - 自适应聚合决策（4-7 cycles）
4. **TMA & Storage** - Tile 加载 + 无冲突存储（22 cycles）
5. **Interpolation** - 双线性插值（3 cycles）

**Function Call 拦截**:
- `__deform_pcb()` → `deform_pcb_impl()` (1 cycle)
- `__deform_tbc()` → `deform_tbc_impl()` (4-7 cycles)
- `__deform_tma()` → `deform_tma_impl()` (16 cycles)
- `__deform_interp()` → `deform_interp_impl()` (3 cycles)

### 1.2 下一步工作（更新版 TODO） ⏳

**最高优先级（GPGPU-Sim 内部集成）**:
1. ⏳ **实现 `deform_attn_exec_unit` 类** (shader.cc)
2. ⏳ **在 `create_exec_pipeline()` 中创建单元** (shader.cc)
3. ⏳ **在 `issue_warp()` 中添加指令分发** (shader.cc)
4. ⏳ **配置初始化** (shader.cc)
5. ⏳ **添加性能统计** (stats.cc)
6. ⏳ **编译与回归测试**（功能正确性 + 周期统计）

**并行任务（CUDA kernel 驱动侧）**:
7. ⏳ **Baseline kernel 驱动**：完整实现 `bilinear_sample_global()` 与正确性对照
8. ⏳ **Optimized kernel 驱动**：chunking + per-query reduce + Discrete fallback
9. ⏳ **伪指令调用映射**：`__deform_pcb/__deform_tbc/__deform_tma/__deform_interp` 在 kernel 中的调用位置对齐

**说明（边界划分）**:
- CUDA kernel 负责：线程映射、控制流、reduce、Discrete 路径。
- GPGPU-Sim 内部负责：PCB/TBC/TMA/Interpolation 的固定延迟建模（通过 Function Call 拦截）。

---

## 2. 实现步骤

### Phase 1: Function Call 拦截接口 ✅

**目标**: 建立 CUDA Function Call 拦截机制

**任务**:
- ✅ **TODO 1.1**: 在 `cuda-sim.cc` 中添加 CUDA Function Call 拦截逻辑
  - 拦截 `__deform_pcb()`, `__deform_tbc()`, `__deform_tma()`, `__deform_interp()`
  - 参考 FMR 的拦截方式（1863-1891行）
  - 在 `CALL_OP` 处理中添加函数名匹配

**注意**:
- ❌ **不需要**在 `opcodes.def` 中定义伪指令
- ❌ **不需要**在 `ptx.l` 中添加伪指令语法
- ❌ **不需要**在 `abstract_hardware_model.h` 中添加指令枚举
- ✅ **只需要** CUDA Function Call 拦截机制

**参考**:
- `src/cuda-sim/cuda-sim.cc` (1896-1938行) - 拦截逻辑

---

### Phase 2: 功能模型实现 ✅

**目标**: 实现 5 个模块的功能模型

**任务**:
- ✅ **TODO 2.1**: 创建 `deform_attn_unit.h/.cc` 文件
- ✅ **TODO 2.2**: 实现 `deform_pcb_model` 类
  - 功能：权重阈值比较，生成 mask
  - 延迟：1 cycle（固定）
- ✅ **TODO 2.3**: 实现 `deform_tbc_model` 类
  - 功能：边界检查 + First-8 Voting + 模式检测
  - 延迟：4 cycles（Phase 1）或 7 cycles（Phase 1+2）（固定）
- ✅ **TODO 2.4**: 实现 `deform_tma_model` 类
  - 功能：Tile 加载（模拟内存访问）
  - 延迟：16 cycles（固定）
  - 集成点：调用 `mem_fetch_interface` 发起访存
- ✅ **TODO 2.5**: 实现 `deform_storage_model` 类
  - 功能：Tracker 管理 + 自适应 Bank 映射
  - 延迟：6 cycles（固定）
  - 简化：使用哈希表模拟 CAM
- ✅ **TODO 2.6**: 实现 `deform_interp_model` 类
  - 功能：双线性插值计算
  - 延迟：3 cycles（固定）

**参考**:
- `src/cuda-sim/instructions.cc` (3585-3809行) - FMR 功能模型
- **关键差异**: FMR 使用动态延迟，DeformAttn 使用固定延迟

---

### Phase 3: Pipeline 集成 ⏳

**目标**: 将模块集成到 GPU 流水线

**任务**:
- ⏳ **TODO 3.1**: 实现 `deform_attn_exec_unit` 类
  - 文件: `src/gpgpu-sim/shader.cc`
  - 参考: `fmr_unit` (2441-2497行)
  - 延迟: 33 cycles（最大）
  - 操作类型: SPECIALIZED__OP

- ⏳ **TODO 3.2**: 在 `create_exec_pipeline()` 中创建单元
  - 文件: `src/gpgpu-sim/shader.cc`
  - 参考: FMR 单元创建 (458-462行)
  - 添加到 `m_fu` 向量
  - 添加 dispatch_port 和 issue_port

- ⏳ **TODO 3.3**: 在 `shader_core_ctx` 构造函数中实例化
  - 文件: `src/gpgpu-sim/shader.cc`
  - 初始化 `m_deform_attn_unit`

- ⏳ **TODO 3.4**: 在 `issue_warp()` 中添加指令分发
  - 文件: `src/gpgpu-sim/shader.cc`
  - 参考: FMR 指令分发 (1509-1524行)
  - 检查 `inst.op` 是否为 DeformAttn 操作
  - 检查执行单元是否可用
  - 发送到对应的寄存器集

- ⏳ **TODO 3.5**: 配置初始化
  - 文件: `src/gpgpu-sim/shader.cc`
  - 在 `shader_core_config` 构造函数中初始化配置

- ⏳ **TODO 3.6**: 在 `cycle()` 中调用
  - 文件: `src/gpgpu-sim/shader.cc`
  - 在 `shader_core_ctx::cycle()` 中调用 `m_deform_attn_unit->cycle()`

**参考**:
- `src/gpgpu-sim/shader.h` (1269-1280行) - FMR 单元定义
- `src/gpgpu-sim/shader.cc` (2441-2497行) - FMR 单元实现

---

### Phase 4: 内存集成 ⏳

**目标**: TMA 与内存系统集成

**任务**:
- ⏳ **TODO 4.1**: TMA 与 L2 Cache 集成
  - 发起 16 次内存读取请求（每行 32 bytes）
  - 接收内存响应并缓存
  - 参考 FMR 的内存访问

- ⏳ **TODO 4.2**: Storage 与 Shared Memory 集成
  - 模拟 32-bank SRAM 访问
  - 实现自适应 Bank 映射逻辑
  - 参考 FMR 的 SMEM 写入

- ⏳ **TODO 4.3**: Bank 冲突检测（可选）
  - 统计 Bank 冲突率
  - 验证自适应 Swizzling 效果

**参考**:
- `src/cuda-sim/instructions.cc` (3767-3773行) - SMEM 写入
- `src/gpgpu-sim/shader.cc` (1509-1524行) - FMR 内存集成

---

### Phase 5: 性能统计 ⏳

**目标**: 收集性能数据

**任务**:
- ⏳ **TODO 5.1**: 添加性能计数器
  - `num_deform_attn_ops`: 总操作数
  - `num_deform_attn_cycles`: 总 cycle 数
  - `num_deform_sparse_skip`: 稀疏跳过的点数
  - `num_deform_discrete_fallback`: 离散加载次数
  - `num_deform_tile_loads`: Tile 加载次数

- ⏳ **TODO 5.2**: 添加延迟直方图
  - 低聚集度场景延迟分布（~30 cycles）
  - 高聚集度场景延迟分布（~33 cycles）

- ⏳ **TODO 5.3**: 添加访存统计
  - L2 Cache 访问次数
  - DRAM 访问次数
  - Bank 冲突次数

**参考**:
- `src/abstract_hardware_model.h` (1078-1083行) - FMR 元数据
- `src/gpgpu-sim/stats.cc` - 性能统计

---

### Phase 6: 验证与调试 ⏳

**目标**: 验证功能正确性，收集性能数据

**任务**:
- ⏳ **TODO 6.1**: 单元测试
  - 测试每个模块的功能正确性
  - 对比 Golden 参考值
  - PCB 测试：权重过滤正确性
  - TBC 测试：模式检测正确性
  - TMA 测试：Tile 加载正确性
  - Interpolation 测试：插值计算正确性

- ⏳ **TODO 6.2**: 集成测试
  - 运行简单的 CUDA Kernel（调用伪指令）
  - 验证端到端延迟
  - 对比预期的 cycle 数（30-33 cycles）

- ⏳ **TODO 6.3**: 性能分析
  - 收集性能数据
  - 分析瓶颈（计算 vs 内存）
  - 验证自适应优化效果

---

## 3. 关键技术点

### 3.1 CUDA Function Call 拦截（核心机制）

**参考 FMR 实现**:
```cpp
// src/cuda-sim/cuda-sim.cc (1863-1891行)
if (inst_opcode == CALL_OP && lane_id == 0) {
    std::string fname = target_func->get_name();

    if (fname.find("fmr_sample") != std::string::npos) {
        ld_sample_fmr_impl(pI, core, inst);
        is_fmr_call = true;
        skip = true;
    }
}
```

**应用到 DeformAttn**:
```cpp
if (fname.find("deform_pcb") != std::string::npos) {
    deform_pcb_impl(pI, core, inst);
    is_deform_call = true;
    skip = true;
}
```

**关键点**:
- Lane 0 执行
- 跳过 CALL 语义（不压栈/弹栈）
- 所有线程 PC 同步
- **不需要伪指令接口**（FMR 的 `ld.sample.fmr` 实际未使用）

**注意事项**:
- ❌ **不需要**在 `opcodes.def` 中定义伪指令
- ❌ **不需要**在 `ptx.l` 中添加伪指令语法
- ❌ **不需要**在 `abstract_hardware_model.h` 中添加指令枚举
- ✅ **只需要**在 `cuda-sim.cc` 中添加 Function Call 拦截逻辑

---

### 3.2 参数读取

**参考 FMR 实现**:
```cpp
// src/cuda-sim/instructions.cc (3596-3634行)
const operand_info &gmem_base_op = pI->operand_lookup(1);
thread->m_local_mem->read(gmem_base_op.get_symbol()->get_address(),
                          sizeof(unsigned long long), &gmem_base);
```

**应用到 DeformAttn (PCB)**:
```cpp
const operand_info &weights_op = pI->operand_lookup(1);
const operand_info &threshold_op = pI->operand_lookup(2);

thread->m_local_mem->read(weights_op.get_symbol()->get_address(),
                          16*16*sizeof(float), weights);
thread->m_local_mem->read(threshold_op.get_symbol()->get_address(),
                          sizeof(float), &threshold);
```

---

### 3.3 固定延迟实现

**参考 FMR 实现**:
```cpp
// src/gpgpu-sim/shader.cc (2441-2497行)
fmr_unit::fmr_unit(register_set *result_port, const shader_core_config *config,
                   shader_core_ctx *core, unsigned issue_reg_id)
    : pipelined_simd_unit(result_port, config, config->fmr_latency, core, issue_reg_id) {
    m_name = "FMR";
}
```

**应用到 DeformAttn**:
```cpp
// PCB 单元（固定 1 cycle）
deform_pcb_unit::deform_pcb_unit(register_set *result_port,
                                 const shader_core_config *config,
                                 shader_core_ctx *core, unsigned issue_reg_id)
    : pipelined_simd_unit(result_port, config, 1, core, issue_reg_id) {
    m_name = "DEFORM_PCB";
}

// TBC 单元（固定 4 或 7 cycles）
deform_tbc_unit::deform_tbc_unit(register_set *result_port,
                                 const shader_core_config *config,
                                 shader_core_ctx *core, unsigned issue_reg_id)
    : pipelined_simd_unit(result_port, config, 4, core, issue_reg_id) {
    m_name = "DEFORM_TBC";
}
```

**关键点**:
- 在构造函数中指定固定延迟
- 不需要动态延迟计算
- 简化实现

---

### 3.4 内存操作标记（TMA）

**参考 FMR 实现**:
```cpp
// src/cuda-sim/instructions.cc (3779行)
inst.op = FMR_SAMPLE_OP;
inst.memory_op = memory_load;
inst.data_size = 16;
```

**应用到 DeformAttn (TMA)**:
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

**注意事项**:
- FMR 使用动态延迟（依赖 L1/L2/DRAM）
- DeformAttn 使用固定延迟（简化实现）
- TMA 的 16 cycles 是配置的固定值

---

## 4. 文件修改清单

### 4.1 新增文件

| 文件 | 说明 | 参考 |
|------|------|------|
| `src/cuda-sim/deform_attn_unit.h` | DeformAttn 单元定义 | `fmr_unit` in `shader.h` |
| `src/cuda-sim/deform_attn_unit.cc` | DeformAttn 单元实现 | `fmr_unit` in `shader.cc` |
| `src/cuda-sim/deform_attn_impl.cc` | 功能模型实现函数 | `ld_sample_fmr_impl()` |
| `deAttn_fp16_fmr/test_deform_attn.cu` | 测试 CUDA Kernel | - |
| `deAttn_fp16_fmr/Makefile.test` | 测试 Makefile | - |

### 4.2 修改文件

| 文件 | 修改内容 | 参考位置 |
|------|---------|---------|
| `src/cuda-sim/cuda-sim.cc` | 添加拦截逻辑 | FMR 拦截 (1863-1891行) |
| `src/gpgpu-sim/shader.h` | 添加流水线阶段和配置 | `ID_OC_FMR` (line 1523) |
| `src/gpgpu-sim/shader.cc` | 添加时序模型和分发 | `fmr_unit` (2441-2497行) |
| `src/gpgpu-sim/gpu-sim.cc` | 添加配置选项 | FMR 配置 (626-643行) |
| `src/gpgpu-sim/stats.cc` | 添加性能统计 | - |
| `src/cuda-sim/Makefile` | 添加编译目标 | - |

**注意**:
- ❌ **不需要**修改 `opcodes.def`（FMR 的伪指令实际未使用）
- ❌ **不需要**修改 `ptx.l`（FMR 的伪指令语法实际未使用）
- ❌ **不需要**修改 `abstract_hardware_model.h`（除非需要添加统计变量）

---

## 5. 测试策略

### 5.1 单元测试

#### PCB 测试
```cpp
// 输入
float weights[16*16] = {0.1, 0.2, 0.3, ...};
float threshold = 0.5;
bool enable = true;

// 预期输出
bool mask[16*16];  // 根据阈值生成

// 测试代码
deform_pcb_model pcb;
pcb.execute(weights, threshold, enable, mask);
```

#### TBC 测试
```cpp
// 输入
float abs_coords[16*16*2] = {...};
bool valid_coords[16*16] = {...};

// 预期输出
deform_tbc_model::mode_t mode;
deform_tbc_model::tile_size_t tile_size;
unsigned base_tile_x, base_tile_y;

// 测试代码
deform_tbc_model tbc;
bool high_gather = tbc.execute(abs_coords, valid_coords,
                               mode, tile_size,
                               base_tile_x, base_tile_y);
```

#### TMA 测试
```cpp
// 输入
addr_t gmem_base = 0x1000;
addr_t smem_base = 0x2000;
unsigned tile_x = 0, tile_y = 0;
unsigned pitch = 32;
unsigned mode = 2;  // XOR
unsigned tile_size = 0;  // 16×16

// 测试代码
deform_tma_model tma;
tma.execute(gmem_base, smem_base, tile_x, tile_y, pitch,
            mode, tile_size, gmem, smem, smid, thread, pI);
```

#### Interpolation 测试
```cpp
// 输入
float p00[16*16], p01[16*16], p10[16*16], p11[16*16];
float wx[16*16], wy[16*16];

// 预期输出
float result[16*16];

// 测试代码
deform_interp_model interp;
interp.execute(p00, p01, p10, p11, wx, wy, result);
```

### 5.2 集成测试

#### CUDA Kernel 示例
```cpp
// CUDA Kernel - 使用 Deformable Attention
__global__ void deform_attn_kernel(
    float* output,
    float* weights,
    float* coords,
    float* tile_data,
    float threshold
) {
    extern __shared__ float smem[];

    // Step 1: PCB - 权重过滤
    bool mask[16*16];
    __deform_pcb(weights, threshold, true, mask);

    // Step 2: TBC - 边界检查 + 模式检测
    int mode, tile_size, base_x, base_y;
    bool valid[16*16];
    __deform_tbc(coords, valid, mode, tile_size, base_x, base_y);

    // Step 3: TMA - Tile 加载
    __deform_tma(tile_data, gmem_base, smem_base,
                 base_x, base_y, pitch, mode, tile_size);

    // Step 4: Interpolation - 插值计算
    __deform_interp(output, tile_data, coords, weights);
}
```

#### 测试配置
```
-gpgpu_deform_attn_avail 1
-gpgpu_deform_pcb_latency 1
-gpgpu_deform_tbc_phase1_latency 4
-gpgpu_deform_tbc_phase2_latency 3
-gpgpu_deform_tma_latency 16
-gpgpu_deform_storage_latency 6
-gpgpu_deform_interp_latency 3
```

---

## 6. 编译与测试

### 6.1 编译 GPGPU-Sim

```bash
# 1. 设置环境
source setup_environment

# 2. 清理旧编译
make clean

# 3. 编译
make -j$(nproc)
```

**预期输出**:
```
Compiling deform_attn_unit.cc...
Compiling deform_attn_impl.cc...
Linking...
Build complete.
```

### 6.2 编译测试程序

```bash
# 进入测试目录
cd deAttn_fp16_fmr

# 编译测试程序
make -f Makefile.test
```

### 6.3 运行测试

```bash
# 复制配置文件
cp configs/QuadroFX5800/* .

# 运行测试
./test_deform_attn
```

---

## 7. 验证指标

### 7.1 功能正确性

| 模块 | 验证方法 | 预期结果 |
|------|---------|---------|
| PCB | 对比阈值判断 | mask 符合预期 |
| TBC | 对比模式检测 | mode/tile_size 正确 |
| TMA | 检查 SMEM 数据 | Tile 数据正确加载 |
| Interpolation | 对比插值公式 | result 符合数学公式 |

### 7.2 性能指标

| 指标 | 预期值 | 说明 |
|------|--------|------|
| 总延迟（低聚集度） | ~30 cycles | PCB(1) + TBC(4) + TMA(16) + Storage(6) + Interp(3) |
| 总延迟（高聚集度） | ~33 cycles | PCB(1) + TBC(7) + TMA(16) + Storage(6) + Interp(3) |
| 稀疏跳过率 | 0-70% | 取决于权重分布 |
| 离散回退率 | 0-50% | 取决于采样点分布 |

### 7.3 内存统计

| 指标 | 预期值 | 说明 |
|------|--------|------|
| L2 Cache 访问次数 | 16-32 | Tile 加载次数 |
| DRAM 访问次数 | 1-4 | 取决于 Tile 位置 |
| Bank 冲突率 | 0% | 自适应映射保证无冲突 |

---

## 8. 调试技巧

### 8.1 启用调试输出

```cpp
// 在 CUDA Kernel 中设置
core->get_gpu()->gpgpu_ctx->debug_tensorcore = true;
```

### 8.2 预期调试输出

```
PCB: threshold=0.50, skip=48/256 (18.8%)
TBC: mode=2, tile_size=16x16, base=(0,0), high_gather=1
TMA: gmem=0x1000, smem=0x2000, tile=(0,0), size=16x16, mode=2
Interpolation: completed (3 cycles)
```

### 8.3 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 参数读取失败 | 参数类型错误 | 检查 `.param` 空间 |
| 延迟不正确 | 配置未生效 | 检查 `gpgpusim.config` |
| 内存访问错误 | 地址转换失败 | 检查 `generic_to_shared()` |

---

## 9. 配置选项

### 9.1 GPGPU-Sim 配置

**添加配置选项** (参考 FMR):
```cpp
// src/gpgpu-sim/gpu-sim.cc (645-677行)
option_parser_register(opp, "-gpgpu_deform_attn_avail", OPT_UINT32,
                       &gpgpu_deform_attn_avail,
                       "Deformable Attention Available (default=0)", "0");
option_parser_register(opp, "-gpgpu_num_deform_units", OPT_UINT32,
                       &gpgpu_num_deform_units,
                       "Number of DeformAttn units (default=1)", "0");
option_parser_register(opp, "-gpgpu_deform_pcb_latency", OPT_UINT32,
                       &deform_pcb_latency,
                       "PCB latency in cycles (default=1)", "1");
option_parser_register(opp, "-gpgpu_deform_tbc_phase1_latency", OPT_UINT32,
                       &deform_tbc_phase1_latency,
                       "TBC Phase 1 latency (default=4)", "4");
option_parser_register(opp, "-gpgpu_deform_tbc_phase2_latency", OPT_UINT32,
                       &deform_tbc_phase2_latency,
                       "TBC Phase 2 latency (default=3)", "3");
option_parser_register(opp, "-gpgpu_deform_tma_latency", OPT_UINT32,
                       &deform_tma_latency,
                       "TMA latency (default=16)", "16");
option_parser_register(opp, "-gpgpu_deform_storage_latency", OPT_UINT32,
                       &deform_storage_latency,
                       "Storage latency (default=6)", "6");
option_parser_register(opp, "-gpgpu_deform_interp_latency", OPT_UINT32,
                       &deform_interp_latency,
                       "Interpolation latency (default=3)", "3");
```

**配置示例**:
```
-gpgpu_deform_attn_avail 1
-gpgpu_num_deform_units 1
-gpgpu_deform_pcb_latency 1
-gpgpu_deform_tbc_phase1_latency 4
-gpgpu_deform_tbc_phase2_latency 3
-gpgpu_deform_tma_latency 16
-gpgpu_deform_storage_latency 6
-gpgpu_deform_interp_latency 3
```

---

## 10. CUDA 使用示例

### 10.1 函数原型

```cpp
extern "C" __device__ void __deform_pcb(
    float* weights,
    float threshold,
    bool enable,
    bool* mask
);

extern "C" __device__ void __deform_tbc(
    float* abs_coords,
    bool* valid_coords,
    int* mode,
    int* tile_size,
    int* base_tile_x,
    int* base_tile_y
);

extern "C" __device__ void __deform_tma(
    float* tile_data,
    unsigned long long gmem_base,
    unsigned long long smem_base,
    int tile_x,
    int tile_y,
    int pitch,
    int mode,
    int tile_size
);

extern "C" __device__ void __deform_interp(
    float* output,
    float* tile_data,
    float* coords,
    float* weights
);
```

### 10.2 使用示例

```cpp
__global__ void my_deform_attn_kernel(
    float* output,
    float* weights,
    float* coords,
    float* tile_data,
    float threshold
) {
    extern __shared__ float smem[];

    // Step 1: PCB
    bool mask[16*16];
    __deform_pcb(weights, threshold, true, mask);

    // Step 2: TBC
    int mode, tile_size, base_x, base_y;
    bool valid[16*16];
    __deform_tbc(coords, valid, &mode, &tile_size, &base_x, &base_y);

    // Step 3: TMA
    __deform_tma(tile_data, gmem_base, smem_base,
                 base_x, base_y, pitch, mode, tile_size);

    // Step 4: Interpolation
    __deform_interp(output, tile_data, coords, weights);
}
```

---

## 11. 时间估算

| 阶段 | 工作量 | 依赖 | 说明 |
|------|-------|------|------|
| Phase 1: Function Call 拦截 | 0.5-1 天 | - | 仅需添加拦截逻辑，非常简单 |
| Phase 2: 功能模型 | 5-7 天 | Phase 1 | 5 个模块，中等复杂度 |
| Phase 3: 流水线集成 | 3-4 天 | Phase 2 | 需要理解寄存器机制 |
| Phase 4: 内存集成 | 4-5 天 | Phase 3 | TMA 内存访问，中等复杂度 |
| Phase 5: 性能统计 | 2-3 天 | Phase 4 | 相对简单 |
| Phase 6: 验证调试 | 3-5 天 | Phase 5 | 需要编写测试用例 |
| **总计** | **18-25 天** | - | 5 个模块，固定延迟简化 |

**对比 FMR**:
- FMR: 3-5 天（单一模块，动态延迟，有伪指令接口）
- DeformAttn: 18-25 天（5 个模块，固定延迟，无伪指令接口）
- 复杂度比 FMR 高 5-8 倍
- **Phase 1 大幅简化**：仅需 0.5-1 天（仅添加拦截逻辑）

---

## 12. 验证策略

### 12.1 功能验证

**测试场景 1：低聚集度**
- 输入：256 个随机分布的采样点
- 预期：TBC 快速回退，mode=3
- 延迟：~30 cycles

**测试场景 2：高聚集度**
- 输入：256 个聚集在 16×16 Tile 内的采样点
- 预期：TBC 通过，进入 Tile 加载
- 延迟：~33 cycles

**测试场景 3：边界情况**
- 输入：坐标在 padding 边缘
- 预期：边界检查正确过滤无效点

### 12.2 性能验证

**对比预期值**：
- 运行相同的测试案例
- 对比 GPGPU-Sim 和预期的 cycle 数
- 验证功能正确性

**对比实际硬件**（如果可用）：
- 运行相同的 Workload
- 对比吞吐量和延迟
- 分析差异来源

---

## 13. 简化假设

### 13.1 不实现的功能

- ❌ 时序级仿真（寄存器传输级）
- ❌ 流水线冒险处理（假设无冒险）
- ❌ 动态电压/频率调节
- ❌ 详细的功耗模型

### 13.2 简化模型

- ✅ 固定延迟（无访存延迟变化）
- ✅ 理想 L2 Cache（假设命中率 100%）
- ✅ 简化 Bank 冲突（统计但不模拟延迟）
- ✅ 哈希表代替 CAM（O(1) 查找）

---

## 14. 总结

### 14.1 核心要点

1. **Function Call 拦截**: 仅需在 `cuda-sim.cc` 中添加拦截逻辑
2. **固定延迟模型**: 每个模块有固定的 cycle 延迟
3. **模块化设计**: 5 个独立模块，易于测试
4. **自适应优化**: TBC 和 Storage 自适应行为
5. **简化接口**: 仅使用 Function Call 拦截，避免伪指令复杂性

### 14.2 成功关键

1. **充分参考 FMR**: 80% 的代码结构可以借鉴（特别是 Function Call 拦截）
2. **固定延迟简化**: 不模拟动态功耗和访存变化
3. **模块化设计**: 每个模块独立测试
4. **逐步集成**: 从单个模块到完整流水线
5. **充分测试**: 单元测试 + 集成测试
6. **简化接口**: 仅使用 Function Call 拦截，避免伪指令复杂性

### 14.3 关键差异

| 方面 | FMR (参考) | DeformAttn (目标) |
|------|-----------|------------------|
| 模块数 | 1 个 | 5 个 |
| 延迟模型 | 动态（内存系统） | 固定（简化） |
| 伪指令接口 | 有（但实际未使用） | 无（仅 Function Call 拦截） |
| 复杂度 | 中等 | 高 |
| 实现时间 | 3-5 天 | 18-25 天 |
| 内存操作 | 是 (MEM__OP) | 部分 (TMA) |
| 计算操作 | 否 | 是 (PCB, GTC, Interp) |

---

## 15. 相关文档

- `DeformAttn_Architecture.md` - 顶层架构设计
- `DeformAttn_Progress_Tracker.md` - 进度跟踪
- `FMR_Implementation_Summary.md` - FMR 实现参考
- `deAttn/README.md` - Deformable Attention 算法说明
- `deAttn_fp16_fmr/PROJECT_OVERVIEW.md` - FMR 扩展说明

---

**文档历史**：
- v1.0 (2026-01-18): 初始版本，合并自 DeformAttn_Implementation_Plan.md、DeformAttn_Quick_Start.md 和 DeformAttn_Test_Summary.md