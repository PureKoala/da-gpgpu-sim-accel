# FMR (Feature Map Reorganizer) 模块集成计划

## 目标
在GPGPU-Sim中实现FMR专用硬件单元，加速Deformable Attention的grid_sampler操作

## 现状分析

### 当前瓶颈 (从TODO.md分析)
1. **地址计算**: 手动计算4个采样点地址占用SP/INT单元
2. **访存延迟**: 4个离散访存需要4个周期，延迟占总运行时间40.9%
3. **计算资源**: SP单元执行大量简单重复的插值运算

### 设计目标
实现一个自治的FMR功能单元，包含:
1. **地址生成器(AG)**: 硬件计算双线性插值的4个地址和权重
2. **4-Bank SMEM控制器**: 保证4个相邻点无冲突并行访问
3. **插值核心**: 硬件固化双线性插值数学逻辑

## 最小改动修改方案

### 阶段1: 核心数据结构扩展

#### 1.1 扩展指令操作码 (abstract_hardware_model.h)
**位置**: `enum uarch_op_t` (约第115行)
**修改**: 添加 `FMR_SAMPLE_OP` 操作码
```cpp
enum uarch_op_t {
  // ... existing ops ...
  SPECIALIZED_UNIT_8_OP,
  FMR_SAMPLE_OP  // 新增: FMR采样操作
};
```

#### 1.2 定义FMR单元类 (shader.h)
**位置**: 在 `tensor_core` 类之后 (约第1260行)
**修改**: 新增 `fmr_unit` 类定义
```cpp
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

### 阶段2: 流水线集成

#### 2.1 添加流水线阶段 (shader.h)
**位置**: `enum pipeline_stage_name_t` (约第1340行)
**修改**: 添加FMR流水线阶段
```cpp
enum pipeline_stage_name_t {
  // ... existing stages ...
  OC_EX_TENSOR_CORE,
  ID_OC_FMR,        // 新增
  OC_EX_FMR,        // 新增
  N_PIPELINE_STAGES
};
```

#### 2.2 更新阶段名称数组 (shader.h)
**位置**: `pipeline_stage_name_decode` 数组 (约第1350行)
```cpp
const char *const pipeline_stage_name_decode[] = {
    // ... existing names ...
    "OC_EX_TENSOR_CORE",
    "ID_OC_FMR",      // 新增
    "OC_EX_FMR",      // 新增
    "N_PIPELINE_STAGES"
};
```

#### 2.3 创建FMR执行单元 (shader.cc)
**位置**: `create_exec_pipeline()` 函数中 (约第272行)
**修改**: 在TENSOR_CORE_CUS之后添加FMR_CUS
```cpp
void shader_core_ctx::create_exec_pipeline() {
  enum { SP_CUS, DP_CUS, SFU_CUS, TENSOR_CORE_CUS, INT_CUS, MEM_CUS, GEN_CUS, FMR_CUS };
  
  // ... existing code ...
  
  // FMR单元配置 (在tensor core配置之后)
  if (m_config->gpgpu_fmr_avail) {
    m_operand_collector.add_cu_set(
        FMR_CUS, m_config->gpgpu_num_fmr_units,
        m_config->gpgpu_operand_collector_num_out_ports_fmr);
    
    for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_fmr; i++) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_FMR]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_FMR]);
      cu_sets.push_back((unsigned)FMR_CUS);
      cu_sets.push_back((unsigned)GEN_CUS);
      m_operand_collector.add_port(in_ports, out_ports, cu_sets);
      in_ports.clear(), out_ports.clear(), cu_sets.clear();
    }
    
    // 创建FMR功能单元
    for (unsigned k = 0; k < m_config->gpgpu_num_fmr_units; k++) {
      m_fu.push_back(new fmr_unit(&m_pipeline_reg[EX_WB], m_config, this, k));
      m_dispatch_port.push_back(OC_EX_FMR);
      m_issue_port.push_back(ID_OC_FMR);
    }
  }
}
```

#### 2.4 实现FMR单元方法 (shader.cc)
**位置**: 在 `tensor_core::issue()` 之后 (约第2400行)
```cpp
fmr_unit::fmr_unit(register_set *result_port, const shader_core_config *config,
                   shader_core_ctx *core, unsigned issue_reg_id)
    : pipelined_simd_unit(result_port, config, config->fmr_latency, core, issue_reg_id) {
  m_name = "FMR";
}

void fmr_unit::issue(register_set &source_reg) {
  warp_inst_t **ready_reg =
      source_reg.get_ready(m_config->sub_core_model, m_issue_reg_id);
  
  (*ready_reg)->op_pipe = SPECIALIZED__OP;
  m_core->incfmr_stat(m_core->get_config()->warp_size, (*ready_reg)->latency);
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

### 阶段3: 调度器集成

#### 3.1 添加FMR调度逻辑 (shader.cc)
**位置**: `scheduler_unit::cycle()` 函数中 (约第1466行)
**修改**: 在TENSOR_CORE_OP分支之后添加FMR_SAMPLE_OP分支
```cpp
void scheduler_unit::cycle() {
  // ... existing code for other ops ...
  
  } else if ((pI->op == FMR_SAMPLE_OP) &&
             !(diff_exec_units && previous_issued_inst_exec_type ==
                                  exec_unit_type_t::FMR)) {
    bool fmr_pipe_avail =
        (m_shader->m_config->gpgpu_num_fmr_units > 0) &&
        m_fmr_out->has_free(m_shader->m_config->sub_core_model, m_id);

    if (fmr_pipe_avail) {
      m_shader->issue_warp(*m_fmr_out, pI, active_mask, warp_id, m_id);
      issued++;
      issued_inst = true;
      warp_inst_issued = true;
      previous_issued_inst_exec_type = exec_unit_type_t::FMR;
    }
  }
  
  // ... continue with existing code ...
}
```

#### 3.2 更新scheduler_unit类 (shader.h)
**位置**: `scheduler_unit` 类定义 (约第600行)
**修改**: 添加FMR输出端口
```cpp
class scheduler_unit {
 private:
  // ... existing members ...
  register_set *m_tensor_core_out;
  register_set *m_fmr_out;  // 新增
  // ... rest of members ...
};
```

**位置**: `exec_unit_type_t` 枚举 (scheduler.h或shader.h中)
```cpp
enum class exec_unit_type_t {
  NONE = 0,
  SP,
  SFU,
  MEM,
  DP,
  INT,
  TENSOR,
  FMR,  // 新增
  SPECIALIZED
};
```

### 阶段4: 配置参数

#### 4.1 添加配置成员 (shader.h)
**位置**: `shader_core_config` 类 (约第1900行)
```cpp
class shader_core_config : public core_config {
 public:
  // ... existing members ...
  
  // FMR配置参数
  bool gpgpu_fmr_avail;
  unsigned gpgpu_num_fmr_units;
  unsigned fmr_latency;
  unsigned gpgpu_operand_collector_num_in_ports_fmr;
  unsigned gpgpu_operand_collector_num_out_ports_fmr;
  
  // ... rest of config ...
};
```

#### 4.2 注册命令行选项 (gpu-sim.cc)
**位置**: `shader_core_config::reg_options()` 函数 (约第450行)
```cpp
void shader_core_config::reg_options(class OptionParser *opp) {
  // ... existing options ...
  
  option_parser_register(
      opp, "-gpgpu_fmr_avail", OPT_BOOL, &gpgpu_fmr_avail,
      "Enable FMR (Feature Map Reorganizer) units for grid sampling (default off)", "0");
  
  option_parser_register(
      opp, "-gpgpu_num_fmr_units", OPT_UINT32, &gpgpu_num_fmr_units,
      "Number of FMR units per shader core (default 1)", "1");
  
  option_parser_register(
      opp, "-gpgpu_fmr_latency", OPT_UINT32, &fmr_latency,
      "FMR operation latency in cycles (default 4)", "4");
  
  option_parser_register(
      opp, "-gpgpu_operand_collector_num_in_ports_fmr", OPT_UINT32,
      &gpgpu_operand_collector_num_in_ports_fmr,
      "Number of FMR input ports in operand collector (default 1)", "1");
  
  option_parser_register(
      opp, "-gpgpu_operand_collector_num_out_ports_fmr", OPT_UINT32,
      &gpgpu_operand_collector_num_out_ports_fmr,
      "Number of FMR output ports in operand collector (default 1)", "1");
}
```

### 阶段5: 性能统计

#### 5.1 添加统计计数器 (shader.h)
**位置**: `shader_core_stats_pod` 结构 (约第1722行)
```cpp
struct shader_core_stats_pod {
  // ... existing stats ...
  
  // FMR统计
  unsigned *m_num_fmr_inst;           // FMR指令总数
  unsigned *m_num_fmr_cycles;         // FMR单元占用周期数
  unsigned *m_num_grid_sample_replaced; // 被FMR替换的采样操作数
  
  // ... rest of stats ...
};
```

#### 5.2 实现统计方法 (shader.cc)
**位置**: 在类似的统计方法附近
```cpp
void shader_core_ctx::incfmr_stat(unsigned active_count, unsigned latency) {
  m_stats->m_num_fmr_inst[m_sid]++;
  m_stats->m_num_fmr_cycles[m_sid] += latency;
}
```

### 阶段6: Shared Memory扩展 (简化版)

#### 6.1 添加4-Bank标记 (memory.h或shader.h)
**位置**: memory_space相关定义附近
```cpp
// 在shared memory访问中标记是否使用FMR重组模式
struct mem_access_t {
  // ... existing members ...
  bool is_fmr_reorganized;  // 新增: 是否为FMR重组的访存
};
```

**注**: 完整的4-Bank SMEM实现需要修改memory子系统，这里采用最小改动方案，主要通过延迟建模体现FMR的加速效果。

## 验证计划

### 验证步骤
1. **编译验证**: 确保所有修改可以通过编译
2. **功能验证**: 使用简单的测试确认FMR指令可以被识别和执行
3. **性能验证**: 对比baseline vs FMR版本的执行周期数

### 预期效果
- FMR指令计数应该替代原来的大量fma/ld.shared指令
- 总执行周期数应该显著减少
- SP/INT单元利用率降低，FMR单元利用率提升

## 实施顺序

1. ✅ **阶段1**: 数据结构扩展 (abstract_hardware_model.h, shader.h)
2. ✅ **阶段2**: FMR单元实现 (shader.cc)
3. ✅ **阶段3**: 流水线集成 (create_exec_pipeline)
4. ✅ **阶段4**: 调度器集成 (scheduler_unit::cycle)
5. ✅ **阶段5**: 配置参数 (gpu-sim.cc)
6. ✅ **阶段6**: 性能统计 (shader.h, shader.cc)
7. ⏳ **阶段7**: 编译和基础测试
8. ⏳ **阶段8**: 使用test.cu进行性能验证

## 关键文件清单

### 必须修改的文件
1. `src/abstract_hardware_model.h` - 添加FMR_SAMPLE_OP
2. `src/gpgpu-sim/shader.h` - 添加fmr_unit类和配置
3. `src/gpgpu-sim/shader.cc` - 实现FMR单元和集成
4. `src/gpgpu-sim/gpu-sim.cc` - 添加配置选项

### 预计代码行数
- 新增代码: ~200行
- 修改代码: ~50行
- 总影响: <300行 (最小改动)

## 备注

这是一个最小化的实现方案，重点在于:
1. **正确性**: 确保FMR作为新的功能单元可以工作
2. **可测量**: 能够通过统计数据观察到FMR的效果
3. **可扩展**: 为后续详细的4-Bank SMEM实现预留接口

完整的4-Bank SMEM重组功能需要修改memory子系统的更多细节，但这超出了"最小改动"的范畴。当前方案通过调整FMR延迟参数即可模拟不同的硬件性能。
