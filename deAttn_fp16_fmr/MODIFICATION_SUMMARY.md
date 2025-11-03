# FMR模块集成 - 修改总结

## 已完成的修改

### 1. 指令集扩展 (`src/abstract_hardware_model.h`)
- ✅ 在 `enum uarch_op_t` 中添加了 `FMR_SAMPLE_OP` 操作码
- 位置: 第 138 行左右

### 2. FMR硬件单元定义 (`src/gpgpu-sim/shader.h`)
- ✅ 定义了 `fmr_unit` 类 (第 1259-1274 行)
  - 继承自 `pipelined_simd_unit`
  - 实现了 `can_issue()`, `issue()`, `active_lanes_in_pipeline()` 方法
  
- ✅ 添加了FMR流水线阶段到 `pipeline_stage_name_t` 枚举
  - `ID_OC_FMR`: FMR指令解码到操作数收集阶段
  - `OC_EX_FMR`: FMR操作数收集到执行阶段
  
- ✅ 更新了 `pipeline_stage_name_decode` 数组

- ✅ 在 `shader_core_config` 类中添加了配置参数:
  ```cpp
  unsigned int gpgpu_fmr_avail;                           // FMR使能开关
  unsigned int gpgpu_num_fmr_units;                       // FMR单元数量
  unsigned int gpgpu_operand_collector_num_in_ports_fmr;  // 输入端口数
  unsigned int gpgpu_operand_collector_num_out_ports_fmr; // 输出端口数
  unsigned int fmr_latency;                               // FMR延迟
  unsigned max_fmr_latency;                               // 最大延迟
  ```

### 3. FMR实现 (`src/gpgpu-sim/shader.cc`)
- ✅ 实现了 `fmr_unit` 的三个关键方法 (第 2400-2429 行):
  - `fmr_unit::fmr_unit()`: 构造函数，设置单元名称和延迟
  - `fmr_unit::issue()`: 发射指令到FMR流水线
  - `fmr_unit::active_lanes_in_pipeline()`: 统计流水线中活跃的线程

- ✅ 在 `create_exec_pipeline()` 中集成FMR:
  - 添加 `FMR_CUS` 到收集器单元枚举
  - 在通用操作数收集器中添加FMR端口 (第 299-301 行)
  - 配置FMR专用操作数收集器 (第 383-391 行)
  - 更新功能单元计数 (第 407行)
  - 创建FMR功能单元实例 (第 445-450 行)

## 待完成的修改

### 4. 调度器集成 (TODO)
**文件**: `src/gpgpu-sim/shader.cc` 和 `shader.h`

需要修改:
1. 在 `scheduler_unit` 类中添加 `register_set *m_fmr_out` 成员
2. 在 `scheduler_unit::cycle()` 中添加FMR_SAMPLE_OP的调度逻辑
3. 定义 `exec_unit_type_t::FMR` 枚举值

### 5. 配置参数注册 (TODO)
**文件**: `src/gpgpu-sim/gpu-sim.cc`

需要在 `shader_core_config::reg_options()` 中注册命令行参数:
```cpp
option_parser_register(opp, "-gpgpu_fmr_avail", OPT_BOOL, &gpgpu_fmr_avail, ...);
option_parser_register(opp, "-gpgpu_num_fmr_units", OPT_UINT32, &gpgpu_num_fmr_units, ...);
option_parser_register(opp, "-gpgpu_fmr_latency", OPT_UINT32, &fmr_latency, ...);
// ... 其他参数
```

### 6. 性能统计 (TODO)
**文件**: `src/gpgpu-sim/shader.h` 和 `shader.cc`

需要:
1. 在 `shader_core_stats_pod` 中添加FMR统计计数器
2. 在 `shader_core_ctx` 中实现 `incfmr_stat()` 方法
3. 取消注释 `fmr_unit::issue()` 中的统计调用

### 7. 配置文件初始化 (TODO)
**文件**: `src/gpgpu-sim/gpu-sim.cc` 或配置初始化相关文件

需要为FMR参数设置默认值:
```cpp
gpgpu_fmr_avail = 0;  // 默认关闭
gpgpu_num_fmr_units = 1;
fmr_latency = 4;
gpgpu_operand_collector_num_in_ports_fmr = 1;
gpgpu_operand_collector_num_out_ports_fmr = 1;
```

## 编译验证

### 当前状态
核心数据结构和FMR单元框架已经完成，代码应该可以编译通过（前提是完成调度器部分）。

### 下一步
1. 完成调度器集成（最关键）
2. 注册配置参数
3. 添加性能统计
4. 编译并测试

## 使用方法（实现完成后）

### 运行配置
在GPGPU-Sim配置文件中添加:
```
-gpgpu_fmr_avail 1
-gpgpu_num_fmr_units 1
-gpgpu_fmr_latency 4
-gpgpu_operand_collector_num_in_ports_fmr 1
-gpgpu_operand_collector_num_out_ports_fmr 1
```

### 预期效果
- FMR指令将替代手动的地址计算、访存和插值计算
- 总执行周期应该减少
- 可以通过调整 `fmr_latency` 参数模拟不同的硬件性能

## 技术要点

### FMR的工作原理
1. **输入**: 接收浮点坐标 (x, y) 和特征图基地址
2. **地址生成**: 硬件计算4个相邻点的地址和插值权重
3. **并行访存**: 利用4-Bank SMEM实现无冲突并行读取
4. **插值计算**: 硬件固化的双线性插值逻辑
5. **输出**: 直接输出插值结果

### 与其他单元的对比
- **Tensor Core**: 专用于矩阵乘法
- **SFU**: 专用于超越函数
- **FMR**: 专用于双线性插值采样

## 代码量统计
- 新增代码: ~180 行
- 修改代码: ~40 行
- 总影响: ~220 行代码

## 文件修改清单
1. ✅ `src/abstract_hardware_model.h` - 添加操作码
2. ✅ `src/gpgpu-sim/shader.h` - 定义类和配置
3. ✅ `src/gpgpu-sim/shader.cc` - 实现和集成
4. ⏳ `src/gpgpu-sim/shader.h` - 调度器成员
5. ⏳ `src/gpgpu-sim/shader.cc` - 调度逻辑
6. ⏳ `src/gpgpu-sim/gpu-sim.cc` - 配置参数
7. ⏳ `src/gpgpu-sim/shader.h` - 性能统计

## 设计哲学
采用**最小改动**原则:
- 复用现有的 `pipelined_simd_unit` 框架
- 参考 `tensor_core` 和 `sfu` 的实现模式
- 通过配置参数控制FMR的行为
- 暂不修改memory子系统（通过延迟建模体现性能）

这种设计便于:
1. 快速原型验证
2. 灵活的性能探索
3. 后续的详细实现扩展
