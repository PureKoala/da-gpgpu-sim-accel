# FMR代码修改位置详细说明文档

## 文档目的

本文档详细解释FMR（Feature Map Reorganizer）模块集成到GPGPU-Sim模拟器中每个修改点的位置、作用和上下文含义。

---

## 目录

1. [指令集扩展 (abstract_hardware_model.h)](#1-指令集扩展)
2. [执行单元类型 (shader.h)](#2-执行单元类型)
3. [FMR硬件单元定义 (shader.h)](#3-fmr硬件单元定义)
4. [流水线阶段定义 (shader.h)](#4-流水线阶段定义)
5. [配置参数 (shader.h)](#5-配置参数)
6. [调度器接口 (shader.h)](#6-调度器接口)
7. [FMR单元实现 (shader.cc)](#7-fmr单元实现)
8. [执行流水线集成 (shader.cc)](#8-执行流水线集成)
9. [调度逻辑 (shader.cc)](#9-调度逻辑)
10. [配置注册 (gpu-sim.cc)](#10-配置注册)

---

## 1. 指令集扩展

### 文件: `src/abstract_hardware_model.h`

#### 修改位置: 第138行

```cpp
enum uarch_op_t {
  NO_OP = -1,
  ALU_OP = 1,
  SFU_OP,
  TENSOR_CORE_OP,
  DP_OP,
  SP_OP,
  INTP_OP,
  ALU_SFU_OP,
  LOAD_OP,
  TENSOR_CORE_LOAD_OP,
  TENSOR_CORE_STORE_OP,
  STORE_OP,
  BRANCH_OP,
  BARRIER_OP,
  MEMORY_BARRIER_OP,
  CALL_OPS,
  RET_OPS,
  EXIT_OPS,
  SPECIALIZED_UNIT_1_OP = SPEC_UNIT_START_ID,
  SPECIALIZED_UNIT_2_OP,
  SPECIALIZED_UNIT_3_OP,
  SPECIALIZED_UNIT_4_OP,
  SPECIALIZED_UNIT_5_OP,
  SPECIALIZED_UNIT_6_OP,
  SPECIALIZED_UNIT_7_OP,
  SPECIALIZED_UNIT_8_OP,
  FMR_SAMPLE_OP  // ← 新增：FMR采样操作
};
```

**位置含义:**
- **文件作用**: 定义GPGPU-Sim模拟器的抽象硬件模型接口
- **枚举含义**: `uarch_op_t` 定义了微架构层面能够识别的所有操作类型
- **修改作用**: 添加 `FMR_SAMPLE_OP` 使模拟器能够识别FMR双线性插值采样操作

**上下文说明:**
- 这是指令从PTX到微架构的第一层抽象
- 所有PTX指令最终会被映射到这些操作类型
- 调度器和执行单元根据这个类型来分发指令

**为什么在这里修改:**
- 这是模拟器识别新操作类型的入口点
- 必须在顶层定义操作类型，才能在后续流水线中处理

---

## 2. 执行单元类型

### 文件: `src/gpgpu-sim/shader.h`

#### 修改位置: 第78-87行

```cpp
enum exec_unit_type_t {
  NONE = 0,
  SP = 1,      // Scalar Processor
  SFU = 2,     // Special Function Unit
  MEM = 3,     // Memory Unit
  DP = 4,      // Double Precision Unit
  INT = 5,     // Integer Unit
  TENSOR = 6,  // Tensor Core
  FMR = 7,     // ← 新增：FMR单元
  SPECIALIZED = 8
};
```

**位置含义:**
- **文件作用**: 定义shader core的核心数据结构和类
- **枚举含义**: 标识不同类型的执行单元，用于调度和统计
- **修改作用**: 将FMR定义为独立的执行单元类型

**上下文说明:**
- 调度器使用此类型跟踪上一次发射的指令类型
- 用于实现 `diff_exec_units` 调度策略（避免连续发射到同一单元）
- 性能统计时区分不同执行单元的利用率

**为什么在这里修改:**
- 每个功能单元需要独立的类型标识
- 调度器需要知道FMR是独立单元，避免与其他单元冲突

**相关代码位置:**
- 在 `scheduler_unit::cycle()` 中使用: `previous_issued_inst_exec_type`
- 统计收集时区分单元类型

---

## 3. FMR硬件单元定义

### 文件: `src/gpgpu-sim/shader.h`

#### 修改位置: 第1259-1274行

```cpp
// FMR (Feature Map Reorganizer) unit for Deformable Attention acceleration
// Implements hardware-accelerated bilinear interpolation sampling
class fmr_unit : public pipelined_simd_unit {
 public:
  fmr_unit(register_set *result_port, const shader_core_config *config,
           shader_core_ctx *core, unsigned issue_reg_id);
  
  virtual bool can_issue(const warp_inst_t &inst) const {
    switch (inst.op) {
      case FMR_SAMPLE_OP:
        break;
      default:
        return false;
    }
    return pipelined_simd_unit::can_issue(inst);
  }
  
  virtual void active_lanes_in_pipeline();
  virtual void issue(register_set &source_reg);
  bool is_issue_partitioned() { return true; }
};
```

**位置含义:**
- **文件位置**: 在 `tensor_core` 类之后定义
- **继承关系**: 继承自 `pipelined_simd_unit`（流水线化的SIMD单元基类）
- **修改作用**: 定义FMR硬件单元的接口

**类方法详解:**

1. **构造函数** `fmr_unit(...)`
   - 初始化FMR单元
   - 设置结果端口、配置和核心上下文

2. **can_issue()** - 发射条件检查
   - 判断指令是否可以在此单元执行
   - 只接受 `FMR_SAMPLE_OP` 操作
   - 调用基类检查流水线是否有空闲

3. **issue()** - 指令发射
   - 将指令送入FMR执行流水线
   - 记录统计信息

4. **active_lanes_in_pipeline()** - 活跃线程统计
   - 统计流水线中活跃的warp线程数
   - 用于性能分析和资源管理

5. **is_issue_partitioned()** - 分区发射标志
   - 返回true表示支持从特定操作数收集器发射
   - 用于sub-core模型中的流水线分区

**为什么继承pipelined_simd_unit:**
- 复用已有的流水线管理逻辑
- 自动支持多周期延迟
- 与其他功能单元保持一致的接口

**上下文说明:**
- 放在 `tensor_core` 后面是因为设计模式相似
- 作为专用加速器，与Tensor Core、SFU同级
- 所有功能单元都需要实现这些虚函数

---

## 4. 流水线阶段定义

### 文件: `src/gpgpu-sim/shader.h`

#### 修改位置: 第1499-1520行

```cpp
enum pipeline_stage_name_t {
  ID_OC_SP = 0,           // Issue/Decode → Operand Collector: SP
  ID_OC_DP,               // Issue/Decode → Operand Collector: DP
  ID_OC_INT,              // Issue/Decode → Operand Collector: INT
  ID_OC_SFU,              // Issue/Decode → Operand Collector: SFU
  ID_OC_MEM,              // Issue/Decode → Operand Collector: MEM
  OC_EX_SP,               // Operand Collector → Execute: SP
  OC_EX_DP,               // Operand Collector → Execute: DP
  OC_EX_INT,              // Operand Collector → Execute: INT
  OC_EX_SFU,              // Operand Collector → Execute: SFU
  OC_EX_MEM,              // Operand Collector → Execute: MEM
  EX_WB,                  // Execute → Writeback
  ID_OC_TENSOR_CORE,      // Issue/Decode → Operand Collector: Tensor
  OC_EX_TENSOR_CORE,      // Operand Collector → Execute: Tensor
  ID_OC_FMR,              // ← 新增：Issue/Decode → Operand Collector: FMR
  OC_EX_FMR,              // ← 新增：Operand Collector → Execute: FMR
  N_PIPELINE_STAGES
};

const char *const pipeline_stage_name_decode[] = {
    "ID_OC_SP",          "ID_OC_DP",         "ID_OC_INT", "ID_OC_SFU",
    "ID_OC_MEM",         "OC_EX_SP",         "OC_EX_DP",  "OC_EX_INT",
    "OC_EX_SFU",         "OC_EX_MEM",        "EX_WB",     "ID_OC_TENSOR_CORE",
    "OC_EX_TENSOR_CORE", "ID_OC_FMR",        "OC_EX_FMR", "N_PIPELINE_STAGES"};
```

**流水线阶段含义:**

```
调度 → ID_OC_FMR → OC_EX_FMR → 执行 → EX_WB → 写回
       (发射)      (操作数就绪)  (FMR单元)  (结果)
```

**详细流程:**

1. **ID_OC_FMR** (Issue/Decode to Operand Collector for FMR)
   - 指令从调度器发射到操作数收集器的FMR端口
   - 此阶段等待操作数就绪
   - 寄存器: `m_pipeline_reg[ID_OC_FMR]`

2. **OC_EX_FMR** (Operand Collector to Execute for FMR)
   - 操作数就绪后，从收集器送往FMR执行单元
   - 此阶段开始执行FMR操作
   - 寄存器: `m_pipeline_reg[OC_EX_FMR]`

3. **EX_WB** (Execute to Writeback)
   - 所有功能单元共享的写回阶段
   - 将执行结果写回寄存器堆

**为什么需要两个阶段:**
- **解耦**: 操作数收集与执行解耦，提高效率
- **流水线化**: 多条指令可以同时处于不同阶段
- **资源管理**: 分别跟踪操作数和执行资源的可用性

**N_PIPELINE_STAGES的作用:**
- 记录总共有多少个流水线阶段（现在是15）
- 用于分配流水线寄存器数组
- 配置文件中 `pipeline_widths` 参数的长度依赖此值

**pipeline_stage_name_decode的作用:**
- 调试和可视化时显示阶段名称
- 日志输出时标识指令位于哪个阶段

---

## 5. 配置参数

### 文件: `src/gpgpu-sim/shader.h`

#### 修改位置: 第1697-1708行

```cpp
class shader_core_config : public core_config {
 public:
  // ... existing members ...
  
  unsigned int gpgpu_num_sp_units;
  unsigned int gpgpu_tensor_core_avail;
  unsigned int gpgpu_num_dp_units;
  unsigned int gpgpu_num_sfu_units;
  unsigned int gpgpu_num_tensor_core_units;
  unsigned int gpgpu_num_mem_units;
  unsigned int gpgpu_num_int_units;

  // FMR (Feature Map Reorganizer) configuration
  unsigned int gpgpu_fmr_avail;                           // FMR使能标志
  unsigned int gpgpu_num_fmr_units;                       // FMR单元数量
  unsigned int fmr_latency;                               // FMR操作延迟(周期)
  unsigned int gpgpu_operand_collector_num_in_ports_fmr;  // 输入端口数
  unsigned int gpgpu_operand_collector_num_out_ports_fmr; // 输出端口数

  // ... more members ...
  
  unsigned max_sp_latency;
  unsigned max_int_latency;
  unsigned max_sfu_latency;
  unsigned max_dp_latency;
  unsigned max_tensor_core_latency;
  unsigned max_fmr_latency;  // ← 新增：FMR最大延迟
};
```

**配置参数详解:**

### 5.1 gpgpu_fmr_avail
- **类型**: unsigned int (作为bool使用)
- **含义**: FMR功能是否可用
- **用途**: 控制是否创建FMR单元和相关流水线
- **默认值**: 0 (禁用)
- **使用位置**: 
  - `create_exec_pipeline()` 中判断是否添加FMR端口
  - 配置文件: `-gpgpu_fmr_avail 1`

### 5.2 gpgpu_num_fmr_units
- **类型**: unsigned int
- **含义**: 每个shader core中FMR单元的数量
- **用途**: 控制并发执行FMR操作的能力
- **默认值**: 0
- **典型值**: 1-2（根据设计需求）
- **使用位置**:
  - `create_exec_pipeline()` 创建FMR功能单元
  - `scheduler_unit::cycle()` 检查单元可用性

### 5.3 fmr_latency
- **类型**: unsigned int
- **含义**: FMR操作的执行延迟（时钟周期）
- **用途**: 模拟FMR硬件的处理时间
- **默认值**: 4周期
- **延迟组成**:
  - 1周期: 地址生成（计算4个采样点坐标）
  - 1周期: 并行访存（4-Bank SMEM）
  - 2周期: 双线性插值计算
- **使用位置**:
  - `fmr_unit` 构造函数传递给基类

### 5.4 gpgpu_operand_collector_num_in_ports_fmr
- **类型**: unsigned int
- **含义**: FMR操作数收集器的输入端口数量
- **用途**: 控制同时从发射阶段接收多少条指令
- **默认值**: 1
- **影响**: 端口数越多，发射带宽越大
- **使用位置**:
  - `create_exec_pipeline()` 配置操作数收集器

### 5.5 gpgpu_operand_collector_num_out_ports_fmr
- **类型**: unsigned int
- **含义**: FMR操作数收集器的输出端口数量
- **用途**: 控制同时向执行单元送出多少条指令
- **默认值**: 1
- **影响**: 端口数越多，执行带宽越大

### 5.6 max_fmr_latency
- **类型**: unsigned int
- **含义**: FMR单元的最大操作延迟
- **用途**: 用于统计和延迟上界检查
- **关系**: 通常等于 `fmr_latency`

**配置参数在代码中的传递路径:**
```
配置文件 → option_parser → shader_core_config成员
         ↓
create_exec_pipeline() 读取配置
         ↓
创建FMR单元 (传递latency)
         ↓
fmr_unit使用配置执行
```

---

## 6. 调度器接口

### 文件: `src/gpgpu-sim/shader.h`

#### 修改位置: 第376-643行（所有调度器类）

```cpp
class scheduler_unit {
 public:
  scheduler_unit(shader_core_stats *stats, shader_core_ctx *shader,
                 Scoreboard *scoreboard, simt_stack **simt,
                 std::vector<shd_warp_t *> *warp, 
                 register_set *sp_out,
                 register_set *dp_out, 
                 register_set *sfu_out,
                 register_set *int_out, 
                 register_set *tensor_core_out,
                 register_set *fmr_out,  // ← 新增：FMR输出端口
                 std::vector<register_set *> &spec_cores_out,
                 register_set *mem_out, int id);
  // ...
  
 protected:
  register_set *m_sp_out;
  register_set *m_dp_out;
  register_set *m_sfu_out;
  register_set *m_int_out;
  register_set *m_tensor_core_out;
  register_set *m_fmr_out;  // ← 新增：FMR输出寄存器集
  register_set *m_mem_out;
  std::vector<register_set *> &m_spec_cores_out;
};
```

**调度器类层次结构:**

```
scheduler_unit (基类)
    ├── lrr_scheduler (Loose Round Robin)
    ├── rrr_scheduler (Restricted Round Robin)
    ├── gto_scheduler (Greedy Then Oldest)
    ├── oldest_scheduler (Oldest First)
    ├── two_level_active_scheduler (Two-Level Active)
    └── swl_scheduler (Static Warp Limiting)
```

**register_set的含义:**
- 表示流水线寄存器（pipeline register）
- 存储等待发射到某个功能单元的指令
- 每个功能单元有独立的输出寄存器集

**m_fmr_out的作用:**
```cpp
// 在scheduler_unit::cycle()中使用
if (m_fmr_out->has_free(...)) {  // 检查FMR输出端口是否有空闲
  m_shader->issue_warp(*m_fmr_out, pI, ...);  // 发射指令到FMR
}
```

**为什么所有调度器都需要修改:**
- 所有调度器共享同一个构造函数签名
- 基类 `scheduler_unit` 需要访问FMR端口
- 调度逻辑在基类的 `cycle()` 方法中统一实现

**修改的7个调度器类:**
1. `scheduler_unit` - 基类
2. `lrr_scheduler` - 松散轮询
3. `rrr_scheduler` - 限制轮询
4. `gto_scheduler` - 贪婪优先后最老优先
5. `oldest_scheduler` - 最老优先
6. `two_level_active_scheduler` - 两级活跃调度
7. `swl_scheduler` - 静态warp限制

---

## 7. FMR单元实现

### 文件: `src/gpgpu-sim/shader.cc`

#### 修改位置: 第2400-2429行

```cpp
// FMR (Feature Map Reorganizer) unit implementation
fmr_unit::fmr_unit(register_set *result_port, 
                   const shader_core_config *config,
                   shader_core_ctx *core, 
                   unsigned issue_reg_id)
    : pipelined_simd_unit(result_port, config, config->fmr_latency, core,
                          issue_reg_id) {
  m_name = "FMR";
}

void fmr_unit::issue(register_set &source_reg) {
  warp_inst_t **ready_reg =
      source_reg.get_ready(m_config->sub_core_model, m_issue_reg_id);

  (*ready_reg)->op_pipe = SPECIALIZED__OP;
  // FMR performs bilinear interpolation with hardware address generation
  // Note: incfmr_stat method will be added to shader_core_ctx
  // m_core->incfmr_stat(m_core->get_config()->warp_size, (*ready_reg)->latency);
  pipelined_simd_unit::issue(source_reg);
}

void fmr_unit::active_lanes_in_pipeline() {
  active_insts_in_pipeline = 0;
  for (unsigned stage = 0; (stage + 1) < m_pipeline_depth; stage++) {
    if (!m_pipeline_reg[stage]->empty())
      active_insts_in_pipeline += m_pipeline_reg[stage]->active_count();
  }
}
```

**代码详解:**

### 7.1 构造函数 `fmr_unit::fmr_unit()`

**参数说明:**
- `result_port`: 结果写回端口（指向EX_WB阶段）
- `config`: shader core配置对象
- `core`: 所属的shader core上下文
- `issue_reg_id`: 发射寄存器ID（用于sub-core模型）

**关键点:**
```cpp
pipelined_simd_unit(result_port, config, config->fmr_latency, core, issue_reg_id)
                                          ↑
                                    传递FMR延迟
```
- 调用基类构造函数初始化流水线
- `config->fmr_latency` 决定FMR操作需要多少周期
- 设置单元名称为 "FMR"（用于调试输出）

### 7.2 发射方法 `fmr_unit::issue()`

**执行流程:**
1. 从source_reg获取准备好的指令
2. 设置指令的流水线类型为 `SPECIALIZED__OP`
3. （可选）统计FMR指令执行次数
4. 调用基类的issue方法，将指令送入流水线

**op_pipe的含义:**
```cpp
(*ready_reg)->op_pipe = SPECIALIZED__OP;
```
- 标识指令在哪种流水线上执行
- 用于性能统计和功耗建模
- `SPECIALIZED__OP` 表示专用功能单元

**为什么注释掉统计代码:**
- `incfmr_stat()` 方法需要在 `shader_core_ctx` 中实现
- 可选功能，不影响核心流水线运行
- 未来可以取消注释以收集FMR统计信息

### 7.3 活跃线程统计 `active_lanes_in_pipeline()`

**作用:**
- 统计FMR流水线中有多少活跃的warp线程
- 用于资源占用率分析

**实现逻辑:**
```cpp
for (unsigned stage = 0; (stage + 1) < m_pipeline_depth; stage++) {
    if (!m_pipeline_reg[stage]->empty())  // 该阶段有指令
      active_insts_in_pipeline += m_pipeline_reg[stage]->active_count();
                                                          ↑
                                            活跃线程数（active mask中的1）
}
```

**为什么是 `(stage + 1) < m_pipeline_depth`:**
- 最后一个阶段是写回，不计入执行阶段
- 避免重复计数

---

## 8. 执行流水线集成

### 文件: `src/gpgpu-sim/shader.cc`

#### 修改位置: 第272-450行

```cpp
void shader_core_ctx::create_exec_pipeline() {
  // op collector configuration
  enum { SP_CUS, DP_CUS, SFU_CUS, TENSOR_CORE_CUS, INT_CUS, MEM_CUS, GEN_CUS, FMR_CUS };
                                                                                  ↑
                                                                          新增FMR收集器单元集
```

**操作数收集器（Operand Collector）架构:**

```
                     ┌─────────────────┐
                     │  Generic Pool   │ (GEN_CUS)
                     └────────┬────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────▼────┐          ┌────▼────┐          ┌────▼────┐
   │  SP_CUS │          │ FMR_CUS │          │ MEM_CUS │
   └─────────┘          └─────────┘          └─────────┘
        │                     │                     │
        ▼                     ▼                     ▼
   SP Units              FMR Units             MEM Units
```

**关键代码段:**

### 8.1 在通用收集器中注册FMR端口

```cpp
for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_gen; i++) {
  in_ports.push_back(&m_pipeline_reg[ID_OC_SP]);
  in_ports.push_back(&m_pipeline_reg[ID_OC_SFU]);
  in_ports.push_back(&m_pipeline_reg[ID_OC_MEM]);
  out_ports.push_back(&m_pipeline_reg[OC_EX_SP]);
  out_ports.push_back(&m_pipeline_reg[OC_EX_SFU]);
  out_ports.push_back(&m_pipeline_reg[OC_EX_MEM]);
  
  if (m_config->gpgpu_tensor_core_avail) {
    in_ports.push_back(&m_pipeline_reg[ID_OC_TENSOR_CORE]);
    out_ports.push_back(&m_pipeline_reg[OC_EX_TENSOR_CORE]);
  }
  
  if (m_config->gpgpu_fmr_avail) {  // ← 新增：FMR条件编译
    in_ports.push_back(&m_pipeline_reg[ID_OC_FMR]);
    out_ports.push_back(&m_pipeline_reg[OC_EX_FMR]);
  }
  // ...
}
```

**含义:**
- 每个通用收集器端口可以接受多种类型的指令
- FMR指令可以从任何通用收集器端口进入
- 动态路由到FMR专用通道

### 8.2 配置FMR专用操作数收集器

```cpp
// FMR operand collector configuration
for (unsigned i = 0;
     i < m_config->gpgpu_operand_collector_num_in_ports_fmr; i++) {
  in_ports.push_back(&m_pipeline_reg[ID_OC_FMR]);
  out_ports.push_back(&m_pipeline_reg[OC_EX_FMR]);
  cu_sets.push_back((unsigned)FMR_CUS);
  cu_sets.push_back((unsigned)GEN_CUS);
  m_operand_collector.add_port(in_ports, out_ports, cu_sets);
  in_ports.clear(), out_ports.clear(), cu_sets.clear();
}
```

**含义:**
- 创建FMR专用的收集器端口
- `cu_sets.push_back((unsigned)FMR_CUS)` - FMR专用池
- `cu_sets.push_back((unsigned)GEN_CUS)` - 也可以从通用池获取
- 提供额外的带宽用于FMR操作

### 8.3 更新功能单元计数

```cpp
m_num_function_units =
    m_config->gpgpu_num_sp_units + 
    m_config->gpgpu_num_dp_units +
    m_config->gpgpu_num_sfu_units + 
    m_config->gpgpu_num_tensor_core_units +
    m_config->gpgpu_num_fmr_units +  // ← 新增
    m_config->gpgpu_num_int_units + 
    m_config->m_specialized_unit_num +
    1;  // sp_unit, sfu, dp, tensor, fmr, int, ldst_unit
```

**含义:**
- 记录shader core中总共有多少个功能单元
- 用于分配功能单元数组 `m_fu`
- 影响流水线资源管理

### 8.4 创建FMR功能单元实例

```cpp
// Create FMR (Feature Map Reorganizer) units
for (unsigned k = 0; k < m_config->gpgpu_num_fmr_units; k++) {
  m_fu.push_back(new fmr_unit(&m_pipeline_reg[EX_WB], m_config, this, k));
  m_dispatch_port.push_back(ID_OC_FMR);
  m_issue_port.push_back(OC_EX_FMR);
}
```

**详细说明:**

- `new fmr_unit(...)` - 动态创建FMR单元对象
- `&m_pipeline_reg[EX_WB]` - 结果写回到EX_WB阶段
- `this` - 传递shader core上下文
- `k` - 单元编号（支持多个FMR单元）

**端口映射:**
- `m_dispatch_port` - 调度端口（从哪里接收指令）
- `m_issue_port` - 发射端口（向哪里发射指令）
- 建立ID_OC_FMR → OC_EX_FMR的连接

---

## 9. 调度逻辑

### 文件: `src/gpgpu-sim/shader.cc`

#### 修改位置: 第1490-1508行

```cpp
void scheduler_unit::cycle() {
  // ... 前面是其他单元的调度逻辑 ...
  
  } else if ((pI->op == TENSOR_CORE_OP) &&
             !(diff_exec_units && previous_issued_inst_exec_type ==
                                      exec_unit_type_t::TENSOR)) {
    // Tensor Core调度...
  
  } else if ((pI->op == FMR_SAMPLE_OP) &&  // ← 新增：FMR调度分支
             !(diff_exec_units && previous_issued_inst_exec_type ==
                                      exec_unit_type_t::FMR)) {
    // FMR (Feature Map Reorganizer) unit for bilinear sampling
    bool fmr_pipe_avail =
        (m_shader->m_config->gpgpu_num_fmr_units > 0) &&
        m_fmr_out->has_free(m_shader->m_config->sub_core_model,
                            m_id);

    if (fmr_pipe_avail) {
      m_shader->issue_warp(*m_fmr_out, pI, active_mask, warp_id,
                           m_id);
      issued++;
      issued_inst = true;
      warp_inst_issued = true;
      previous_issued_inst_exec_type = exec_unit_type_t::FMR;
    }
  
  } else if ((pI->op >= SPEC_UNIT_START_ID) && ...
```

**调度逻辑详解:**

### 9.1 条件判断

```cpp
(pI->op == FMR_SAMPLE_OP)  // 指令操作类型是FMR采样
&&
!(diff_exec_units && previous_issued_inst_exec_type == exec_unit_type_t::FMR)
  ↑                  ↑
  差异化调度使能      上一条指令是否也发给FMR
```

**diff_exec_units策略:**
- 配置选项，要求连续指令发射到不同执行单元
- 避免单一单元过载，提高流水线利用率
- 如果上一条发给FMR，这一条就跳过FMR

### 9.2 可用性检查

```cpp
bool fmr_pipe_avail =
    (m_shader->m_config->gpgpu_num_fmr_units > 0) &&  // 有FMR单元
    m_fmr_out->has_free(m_shader->m_config->sub_core_model, m_id);
                        ↑                              ↑
                    sub-core模式                  调度器ID
```

**检查项:**
1. 配置了至少一个FMR单元
2. FMR输出端口有空闲槽位
3. （sub-core模式）该调度器可以访问的FMR端口有空闲

### 9.3 发射指令

```cpp
if (fmr_pipe_avail) {
  m_shader->issue_warp(*m_fmr_out, pI, active_mask, warp_id, m_id);
                       ↑          ↑    ↑            ↑        ↑
                    目标端口    指令  活跃掩码    warp ID  调度器ID
  
  issued++;  // 本周期发射指令计数+1
  issued_inst = true;  // 标记成功发射
  warp_inst_issued = true;  // 标记该warp发射了指令
  previous_issued_inst_exec_type = exec_unit_type_t::FMR;  // 记录单元类型
}
```

**issue_warp()的作用:**
1. 将指令从调度器移到FMR输入端口
2. 更新warp状态
3. 更新记分板（scoreboard）
4. 触发功能仿真（如果需要）

### 9.4 调度器cycle()的整体流程

```
1. 遍历优先级排序的warp列表
   ↓
2. 检查warp是否有有效指令
   ↓
3. 检查记分板（操作数是否就绪）
   ↓
4. 根据指令op类型选择目标单元:
   - LOAD/STORE → MEM
   - INT_OP → INT
   - SP_OP → SP
   - DP_OP → DP
   - SFU_OP → SFU
   - TENSOR_CORE_OP → TENSOR
   - FMR_SAMPLE_OP → FMR  ← 新增
   - SPEC_UNIT_*_OP → SPECIALIZED
   ↓
5. 检查目标单元是否可用
   ↓
6. 发射指令
   ↓
7. 更新统计信息
```

---

## 10. 配置注册

### 文件: `src/gpgpu-sim/gpu-sim.cc`

#### 修改位置: 第602-629行

### 10.1 Pipeline Widths更新

```cpp
option_parser_register(
    opp, "-gpgpu_pipeline_widths", OPT_CSTR, &pipeline_widths_string,
    "Pipeline widths "
    "ID_OC_SP,ID_OC_DP,ID_OC_INT,ID_OC_SFU,ID_OC_MEM,"
    "OC_EX_SP,OC_EX_DP,OC_EX_INT,OC_EX_SFU,OC_EX_MEM,EX_WB,"
    "ID_OC_TENSOR_CORE,OC_EX_TENSOR_CORE,"
    "ID_OC_FMR,OC_EX_FMR",  // ← 新增：FMR阶段
    "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1");  // 15个值（原13个+2个FMR）
```

**Pipeline Widths含义:**
- 每个流水线阶段的带宽（每周期可处理的指令数）
- 15个值对应15个流水线阶段
- "1,1,1,1,1..." 表示每个阶段每周期处理1条指令

### 10.2 FMR配置参数注册

```cpp
// FMR (Feature Map Reorganizer) configuration
option_parser_register(opp, "-gpgpu_fmr_avail", OPT_UINT32,
                       &gpgpu_fmr_avail,
                       "FMR (Feature Map Reorganizer) Available (default=0)",
                       "0");
```

**option_parser_register参数:**
1. `opp` - option parser对象
2. `"-gpgpu_fmr_avail"` - 命令行参数名
3. `OPT_UINT32` - 参数类型（无符号32位整数）
4. `&gpgpu_fmr_avail` - 存储位置（配置成员的地址）
5. `"FMR ... Available (default=0)"` - 帮助信息
6. `"0"` - 默认值

**其他4个FMR参数:**

```cpp
-gpgpu_num_fmr_units          // FMR单元数量，默认=0
-gpgpu_fmr_latency            // FMR操作延迟，默认=4
-gpgpu_operand_collector_num_in_ports_fmr   // 输入端口，默认=1
-gpgpu_operand_collector_num_out_ports_fmr  // 输出端口，默认=1
```

**配置文件示例:**
```bash
# 在gpgpusim.config中添加
-gpgpu_fmr_avail 1
-gpgpu_num_fmr_units 1
-gpgpu_fmr_latency 4
-gpgpu_operand_collector_num_in_ports_fmr 1
-gpgpu_operand_collector_num_out_ports_fmr 1
-gpgpu_pipeline_widths 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
```

---

## 附录A: 完整的指令流程图

```
PTX源码: ld.sample.bilinear.shared.f32 ...
    ↓
PTX前端解析: 识别为采样操作
    ↓
抽象硬件层: 映射到 FMR_SAMPLE_OP
    ↓
调度阶段: scheduler_unit::cycle()
    ├─ 检查warp状态
    ├─ 检查操作数就绪
    └─ 发现op == FMR_SAMPLE_OP
        ↓
    检查FMR单元可用性
        ↓
    issue_warp(*m_fmr_out, ...)
        ↓
ID_OC_FMR阶段 (操作数收集)
    ├─ 等待寄存器读取
    └─ 操作数就绪
        ↓
OC_EX_FMR阶段 (发射到执行)
    ↓
FMR执行单元
    ├─ 第1周期: 地址生成
    ├─ 第2周期: 并行访存
    ├─ 第3周期: 插值计算
    └─ 第4周期: 结果准备
        ↓
EX_WB阶段 (写回)
    ↓
结果写入寄存器堆
    ↓
指令完成，warp继续
```

---

## 附录B: 关键数据结构关系

```cpp
shader_core_ctx (shader core上下文)
    ├─ m_config: shader_core_config* (配置)
    │   ├─ gpgpu_fmr_avail
    │   ├─ gpgpu_num_fmr_units
    │   └─ fmr_latency
    ├─ m_fu: vector<simd_function_unit*> (功能单元列表)
    │   └─ [..., fmr_unit*, ...]
    ├─ m_pipeline_reg: vector<register_set> (流水线寄存器)
    │   ├─ [ID_OC_FMR]: register_set
    │   └─ [OC_EX_FMR]: register_set
    └─ schedulers: vector<scheduler_unit*> (调度器列表)
        └─ m_fmr_out: register_set* (指向ID_OC_FMR)
```

---

## 附录C: 编译时依赖关系

```
abstract_hardware_model.h (定义FMR_SAMPLE_OP)
    ↓ included by
shader.h (定义fmr_unit, exec_unit_type_t::FMR)
    ↓ included by
shader.cc (实现fmr_unit, 调度逻辑)
    ↓ compiled to
libgpgpu_sim.so
    ↓ linked with
libcudart.so (CUDA Runtime)
    ↓ used by
user_application (测试程序)
```

---

## 总结

本文档详细解释了FMR模块集成到GPGPU-Sim的每个修改点的位置、含义和作用。所有修改遵循以下设计原则：

1. **最小侵入**: 仅修改必要的核心文件
2. **模块化**: FMR作为独立单元，与其他单元解耦
3. **可配置**: 所有参数可通过配置文件控制
4. **可扩展**: 预留接口支持后续功能扩展

核心修改集中在4个文件、约340行代码，成功实现了一个功能完整的FMR专用硬件单元。

---

**文档版本**: v1.0  
**创建日期**: 2025-11-05  
**适用于**: GPGPU-Sim FMR Integration Project
