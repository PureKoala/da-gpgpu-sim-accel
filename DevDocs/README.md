# Deformable Attention GPGPU-Sim 加速器（概览）

> 当前实现与后续计划请以 [IMPLEMENTATION.md](IMPLEMENTATION.md) 为准。
> 本文件其余内容为历史材料，仅供参考。

**项目概述**: 在GPGPU-Sim中实现Deformable Attention算子的硬件加速器模型，面向采样/插值阶段的硬件协同优化

**版本**: v1.2 (2026-01-22)  
**状态**: 硬件功能模型已完成 ✅，Kernel实现已完成 ✅，流水线集成已完成 ✅，功能模拟已完成 ✅

---

## 📚 项目文档

### 核心文档
- **[DEFORMABLE_ATTENTION.md](DEFORMABLE_ATTENTION.md)** - 完整的实现文档和使用指南
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - 架构设计文档
- **[IMPLEMENTATION.md](IMPLEMENTATION.md)** - 实现细节文档

### 快速开始

### 项目目标
解决传统GPU在Deformable Attention计算中的两大瓶颈：
- **稀疏采样的随机访存** → 通过Tile聚合优化
- **低权重点的无效计算** → 通过权重预筛选减少

### 核心技术方案
采用**5级流水线架构**，通过Function Call拦截机制在GPGPU-Sim中实现：
```
PCB权重筛选 → GTC操作数隔离 → TBC聚合决策 → TMA&Storage → Interpolation
```

### 当前实现状态（v1.2, 2026-01-23）

#### ✅ 已完成
- **架构决策**: Function Call Only（CUDA编译器无法识别自定义PTX指令）
- **Function Call拦截**: `cuda-sim.cc` 拦截4个函数调用
- **功能模型**: 5个模块实现（PCB/GTC/TBC/TMA&Storage/Interpolation）
- **配置系统**: 延迟和可用性配置选项
- **Baseline/Optimized Kernel**: 完整实现
- **编译通过**: GPGPU-Sim with DeformAttn support ✅

#### ⏳ 待完成
- **P0 参数传递修复**: 修改 `deform_*_impl()` 参数读取逻辑
- **P1 功能验证**: Baseline vs Optimized 数值对比
- **P2 性能统计**: PCB剪枝率、TBC模式分布、TMA复用率

---

## 文档导航

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - 算法原理与硬件架构设计
- **[IMPLEMENTATION.md](IMPLEMENTATION.md)** - 实现细节、数据流与开发指南
- **[TODO.md](TODO.md)** - 详细任务清单与开发进度

---

## 实现数据流（无代码）

### Baseline 数据流
1. 读取输入张量：`data_value`、`data_spatial_shapes`、`data_level_start_index`、`data_sampling_loc`、`data_attn_weight`。
2. 以输出元素为粒度解码索引（batch、head、channel、sampling index）。
3. 对每个 level、每个 point：计算采样坐标 $(h_{im}, w_{im})$，进行边界判断。
4. 在合法范围内执行标准双线性插值并按权重累加。
5. 写回 `data_col` 输出。对应实现见 [deAttn_base/src/deform_attn/cuda/ms_deform_attn_im2col_cuda.cuh](deAttn_base/src/deform_attn/cuda/ms_deform_attn_im2col_cuda.cuh)。

### Optimized 数据流 (GPGPU-Sim 硬件加速)

**设计说明**: 使用 **Function Call Only** 方案（无PTX伪指令）

1. **CUDA Kernel 调用阶段**：读取与 Baseline 相同的输入与索引映射，生成本 level 的采样坐标与权重缓冲。

2. **Function Call 拦截阶段**：GPGPU-Sim 在 `cuda-sim.cc` 中拦截硬件加速函数调用：
   - `__deform_pcb()` → `deform_pcb_impl()`（权重预筛选）
   - `__deform_tbc()` → `deform_tbc_impl()`（聚合决策）  
   - `__deform_tma()` → `deform_tma_impl()`（Tile加载）
   - `__deform_interp()` → `deform_interp_impl()`（双线性插值）

3. **硬件模型执行阶段**：每个 impl 函数直接执行功能模拟和延迟建模：
   - **PCB模块**：并行比较256个权重与阈值，生成剪枝mask，延迟1 cycle
   - **TBC模块**：分析有效点空间分布，决策访存模式（Horizontal/Vertical/XOR/Discrete），延迟4-7 cycles
   - **TMA模块**：协作加载16×16 Tile到共享内存，延迟16 cycles
   - **Storage模块**：无冲突Bank映射，应用Swizzle函数，延迟6 cycles
   - **Interpolation模块**：硬件加速双线性插值，延迟3 cycles

4. **结果处理阶段**：
   - 若聚合成功，使用硬件加速结果；若聚合失败或越界，回退为标准双线性插值路径
   - 保证数值与 Baseline 完全一致，写回 `data_col` 输出
   - 对应实现见 [deAttn_optim/src/deform_attn/cuda/ms_deform_attn_im2col_cuda.cuh](deAttn_optim/src/deform_attn/cuda/ms_deform_attn_im2col_cuda.cuh)

**注意**: 
- ❌ 不使用流水线分发（无操作码定义）
- ❌ 不创建执行单元（deform_attn_exec_unit已禁用）
- ✅ 仅通过Function Call拦截实现所有功能
- ✅ 原因：CUDA编译器无法识别自定义PTX指令

### GPGPU-Sim 集成架构（Function Call Only）
```
CUDA Kernel (deAttn_optim) 
    ↓ [Function Calls: __deform_pcb, __deform_tbc, __deform_tma, __deform_interp]
cuda-sim.cc (Function Call 拦截)
    ↓ [String Match: fname.find("deform_pcb"), etc.]
instructions.cc (功能模拟)
    ↓ [deform_pcb_impl, deform_tbc_impl, deform_tma_impl, deform_interp_impl]
    ↓ [功能模拟 + 延迟建模 (1-16 cycles)]
继续执行（无流水线分发）
```

**实现说明**:
- ✅ 使用Function Call拦截机制（复用FMR经验）
- ❌ 不使用PTX伪指令（CUDA编译器无法识别）
- ❌ 不使用流水线分发（已禁用 `deform_attn_exec_unit`）
- ✅ 直接在 `deform_*_impl()` 中完成功能模拟和延迟建模

### 注意力聚合阶段说明
注意力权重与采样值的矩阵聚合有可选的 WMMA 实验实现（仅在满足尺寸约束时使用），见 [deAttn_base/src/deform_attn/cuda/ms_deform_attn_wmma_cuda.cuh](deAttn_base/src/deform_attn/cuda/ms_deform_attn_wmma_cuda.cuh) 与 [deAttn_optim/src/deform_attn/cuda/ms_deform_attn_wmma_cuda.cuh](deAttn_optim/src/deform_attn/cuda/ms_deform_attn_wmma_cuda.cuh)。当前核心差异集中在采样/插值阶段是否调用专用加速器。

---

## 关键配置选项

```bash
# 基础配置
-gpgpu_deform_attn_avail 1
-gpgpu_num_deform_units 1

# 延迟配置（cycles）
-gpgpu_deform_pcb_latency 1      # PCB权重筛选
-gpgpu_deform_tbc_latency 7      # TBC聚合决策(4+3)  
-gpgpu_deform_tma_latency 16     # TMA Tile加载
-gpgpu_deform_storage_latency 6  # Storage无冲突访问
-gpgpu_deform_interp_latency 3   # Interpolation双线性插值
```

---

## 快速验证

```bash
# 1. 编译GPGPU-Sim（包含Deformable Attention支持）
cd $GPGPUSIM_ROOT
make

# 2. 运行功能测试
cd deAttn_fp16_fmr  
make test

# 3. 检查硬件调用统计
grep "deform_attn" simulator_output.txt
```

---

## 技术亮点

- **Function Call拦截机制**: 无需新增PTX指令，简化集成复杂度
- **固定延迟建模**: 专注功能验证，后续可扩展为动态建模
- **16×16处理粒度**: 与硬件模块设计精确对齐
- **多模式支持**: Tile聚合模式 + Discrete fallback路径

---

**联系**: 详细技术问题请参考ARCHITECTURE.md和IMPLEMENTATION.md文档