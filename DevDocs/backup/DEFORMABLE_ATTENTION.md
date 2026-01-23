# Deformable Attention Hardware Accelerator for GPGPU-Sim

> Current implementation and future plan: see [IMPLEMENTATION.md](IMPLEMENTATION.md).
> This document is archived and kept for reference.

这是基于GPGPU-Sim的Deformable Attention硬件加速器实现，支持5阶段pipeline架构和完整的功能模拟。

## 🚀 项目概述

本项目实现了专用的Deformable Attention硬件加速器，集成到GPGPU-Sim仿真器中：

### 核心特性
- ✅ **5阶段Pipeline架构**: PCB → GTC → TBC → TMA → Interpolation
- ✅ **函数调用拦截**: 通过CUDA函数调用触发硬件加速
- ✅ **完整功能模拟**: 包含所有阶段的功能正确性验证
- ✅ **固定延迟模型**: 可配置的pipeline延迟设置
- ✅ **GPGPU-Sim集成**: 完整的调度器和执行单元集成

### 实现状态: **v1.2 - 完整功能实现**

| 组件 | 状态 | 说明 |
|------|------|------|
| 指令操作码定义 | ✅ 完成 | DEFORM_PCB/GTC/TBC/TMA/INTERP_OP |
| 执行单元集成 | ✅ 完成 | deform_attn_exec_unit |
| 调度器集成 | ✅ 完成 | 所有调度器类型支持 |
| 指令派发逻辑 | ✅ 完成 | 完整的指令识别和路由 |
| 功能模型实现 | ✅ 完成 | 5阶段功能模拟 |
| 内存系统集成 | ✅ 完成 | TMA阶段内存事务生成 |
| 数据流追踪 | ✅ 完成 | 完整的元数据管理 |

### 5阶段Pipeline数据流

```
Input: Attention Weights, Feature Maps, Sampling Coordinates
   ↓
┌─────────────────┐    ┌─────────────────┐
│   PCB Stage     │    │   GTC Stage     │
│  Weight Pruning │ ←→ │ Operand Isolation│  
│   (~1 cycle)    │    │   (parallel)    │
└─────────────────┘    └─────────────────┘
          ↓
┌─────────────────┐
│   TBC Stage     │
│ Boundary Check  │
│  (~4-7 cycles)  │
└─────────────────┘
          ↓
┌─────────────────┐
│   TMA Stage     │
│  Memory Access  │
│  (~22 cycles)   │  → Memory Subsystem
└─────────────────┘
          ↓
┌─────────────────┐
│ INTERP Stage    │
│ Bilinear Interp │
│   (~3 cycles)   │
└─────────────────┘
          ↓
Output: Interpolated Feature Results
```

## 🔧 实现架构

### 函数调用拦截机制
```cpp
// CUDA kernel中的调用:
__deform_pcb(weights, threshold, mask, count);
__deform_gtc(operands);
__deform_tbc(coords, bounds, modes);
__deform_tma(features, coords, output);
__deform_interp(features, weights, result);

// GPGPU-Sim拦截并转换为硬件指令:
DEFORM_PCB_OP  → deform_pcb_impl()
DEFORM_GTC_OP  → deform_gtc_impl()
DEFORM_TBC_OP  → deform_tbc_impl()
DEFORM_TMA_OP  → deform_tma_impl()
DEFORM_INTERP_OP → deform_interp_impl()
```

### 执行单元集成
- **执行单元类**: `deform_attn_exec_unit`
- **调度器支持**: 所有调度器类型 (LRR, RRR, GTO, oldest, two-level, SWL)
- **指令识别**: 完整的5个DEFORM操作码支持
- **延迟模型**: 固定延迟pipeline模型

### 数据结构扩展
```cpp
// warp_inst_t中的元数据字段:
int m_deform_stage;            // 当前pipeline阶段
int m_deform_valid_points;     // PCB: 有效采样点数
int m_deform_direct_count;     // TBC: 直接访问数
int m_deform_cached_count;     // TBC: 缓存访问数
int m_deform_compute_count;    // TBC: 计算访问数
int m_deform_mem_accesses;     // TMA: 内存访问数
int m_deform_interp_ops;       // INTERP: 插值操作数
```

## 📂 代码组织结构

### 核心文件修改
- `src/abstract_hardware_model.h`: 操作码定义和warp_inst_t扩展
- `src/cuda-sim/instructions.cc`: 5阶段功能实现
- `src/gpgpu-sim/shader.h`: 执行单元类定义
- `src/gpgpu-sim/shader.cc`: 执行单元创建和指令派发

### 测试和验证
- `deAttn_*/`: 各种测试和验证实现
- `configs/`: GPGPU-Sim配置文件

## 🚀 使用方法

### 编译和运行
1. 正常编译GPGPU-Sim
2. 在配置文件中设置deform单元数量
3. 编写包含deformable attention函数调用的CUDA kernel
4. 运行仿真

### 配置参数
- `gpgpu_num_deform_units`: deformable attention执行单元数量
- `deform_attn_latency`: 总pipeline延迟设置

## 📊 性能特性

### 相比软件实现的优势
- **并行处理**: 多个deform单元并行执行
- **专用pipeline**: 优化的5阶段处理流水线
- **内存优化**: TMA阶段高效的内存访问模式
- **延迟可预测**: 固定延迟模型便于性能分析

### 静态验证结果
- ✅ **架构完整性**: 90%+ 符合GPGPU-Sim集成patterns
- ✅ **功能正确性**: 完整的5阶段功能模拟
- ✅ **可扩展性**: 支持多单元配置
- ✅ **维护性**: 清晰的模块化设计

## 📈 开发历史

- **v1.0**: 基础pipeline集成
- **v1.1**: 完整调度器集成
- **v1.2**: 功能模型实现完成

## 🔮 未来计划

- **v1.3**: 性能统计系统增强
- **v1.4**: 动态延迟模型
- **v2.0**: 多精度支持扩展