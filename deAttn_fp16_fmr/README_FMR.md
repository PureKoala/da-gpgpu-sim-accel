# FMR模块集成 - 快速指南

## 项目目标
在GPGPU-Sim中以最小改动集成FMR（Feature Map Reorganizer）专用硬件单元，用于加速Deformable Attention的双线性插值采样操作。

## 修改完成状态: 75%

### ✅ 已完成
1. ✅ 指令集扩展 (FMR_SAMPLE_OP)
2. ✅ FMR硬件单元类定义
3. ✅ 流水线阶段添加 (ID_OC_FMR, OC_EX_FMR)
4. ✅ 配置参数定义
5. ✅ 调度器签名更新（所有7个调度器类）
6. ✅ FMR单元实现（构造、issue、active_lanes）
7. ✅ 执行流水线集成
8. ✅ 调度器实例化更新

### ⏳ 待完成（预计1.5小时）
1. ⏳ FMR调度逻辑（scheduler_unit::cycle中添加FMR_SAMPLE_OP分支）
2. ⏳ 配置参数注册（gpu-sim.cc中的option_parser_register）
3. ⏳ swl_scheduler构造函数实现
4. ⏳ 配置默认值初始化

## 文档清单

| 文档 | 说明 |
|-----|------|
| `TODO.md` | FMR需求分析和设计目标 |
| `FMR_MODIFICATION_PLAN.md` | 详细的修改计划（包含代码示例）|
| `MODIFICATION_SUMMARY.md` | 修改总结（按阶段列出）|
| `FMR_IMPLEMENTATION_REPORT.md` | 完整实施报告（本文档）|

## 快速定位修改点

### 文件1: `src/abstract_hardware_model.h`
```
行号: ~138
修改: 添加 FMR_SAMPLE_OP 到 enum uarch_op_t
```

### 文件2: `src/gpgpu-sim/shader.h`  
```
行号: 1259-1274    - fmr_unit类定义
行号: 1499-1520    - 流水线阶段
行号: 1697-1706    - 配置参数
行号: 376-643      - 调度器构造函数（7个类）
```

### 文件3: `src/gpgpu-sim/shader.cc`
```
行号: 272-450      - create_exec_pipeline() 集成
行号: 207-251      - create_schedulers() 实例化
行号: 2400-2429    - fmr_unit实现
```

## 代码统计
- 新增代码: ~220行
- 修改代码: ~80行
- 影响文件: 3个核心文件

## 下一步
查看 `FMR_IMPLEMENTATION_REPORT.md` 的"下一步行动计划"章节。

## 核心理念
**最小改动 + 参数化设计 + 可扩展架构**
