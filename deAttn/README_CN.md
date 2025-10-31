# Deformable Attention CUDA 实现

## 项目概述

本项目实现了一个不依赖PyTorch的纯CUDA版本的Deformable Attention算子，专门针对GPGPU-Sim仿真器优化。该实现包含Tensor Core加速支持，可以在GPGPU-Sim上进行性能分析和架构研究。

## 目录结构

```
deAttn/
├── README_CN.md                    # 中文说明文档（本文件）
├── run.sh                          # 主运行脚本，用于编译和在GPGPU-Sim上运行
├── syn_gpgpu_sim.sh               # GPGPU-Sim环境同步脚本
│
├── bin/                            # 编译后的可执行文件目录
│   └── test                       # test.cu编译后的二进制文件
│
├── out/                            # 输出结果目录
│   └── test_RTX3070.txt           # 运行结果输出文件
│
├── sim/                            # GPGPU-Sim仿真配置目录
│   └── test_RTX3070/              # 针对RTX3070配置的仿真环境
│       ├── gpgpusim.config        # GPU架构配置文件
│       ├── config_*.icnt          # 互连网络配置
│       └── ...                    # 其他GPGPU-Sim配置文件
│
├── src/                            # 源代码目录
│   ├── test.cu                    # 主测试程序（调用deformable attention kernel）
│   │
│   └── deform_attn/               # Deformable Attention实现
│       ├── deform_attn_cuda.h     # 纯CUDA接口头文件（无PyTorch依赖）
│       │
│       └── cuda/                  # CUDA kernel实现
│           ├── ms_deform_attn_im2col_cuda.cuh      # im2col实现（前向传播kernel）
│           ├── ms_deform_attn_im2col_wmma_cuda.cuh # Tensor Core优化版本（WMMA）
│           └── ms_deform_attn_cuda_kernel.cu       # kernel包装函数
│
└── ref/                            # 参考实现
    └── Deformable-DETR/           # 原始Deformable-DETR的PyTorch实现
        └── models/ops/src/        # 原始算子源码（保留作为参考）
```

## 文件说明

### 核心源文件

#### `src/test.cu`
- **作用**：主测试程序，负责调用deformable attention kernel
- **功能**：
  - 初始化测试数据（value、sampling_loc、attn_weight等）
  - 调用deformable attention forward kernel
  - 验证计算结果正确性
  - 统计性能指标（执行时间、吞吐量等）
  - 输出性能摘要

#### `src/deform_attn/deform_attn_cuda.h`
- **作用**：纯CUDA接口头文件
- **特点**：
  - 完全移除ATen/Torch依赖
  - 使用原生CUDA数据类型（float*, int64_t*等）
  - 定义了forward和backward函数接口
  - 兼容GPGPU-Sim环境

#### `src/deform_attn/cuda/ms_deform_attn_im2col_cuda.cuh`
- **作用**：实现deformable attention的im2col变换和双线性插值
- **核心功能**：
  - `ms_deform_attn_im2col_bilinear`：双线性插值采样函数
  - `ms_deformable_im2col_gpu_kernel`：前向传播主kernel
  - `ms_deformable_col2im_gpu_kernel`：反向传播主kernel
- **算法**：
  - 根据采样位置（sampling_loc）从输入特征图中进行双线性插值
  - 对插值结果应用注意力权重（attn_weight）
  - 累加多个采样点的结果

#### `src/deform_attn/cuda/ms_deform_attn_im2col_wmma_cuda.cuh`
- **作用**：Tensor Core优化版本（使用WMMA API）
- **优化点**：
  - 使用半精度（FP16）进行矩阵乘法
  - 利用WMMA fragments进行批量计算
  - 适用于Ampere及更新架构（SM80+）
- **应用场景**：大规模deformable attention计算

#### `src/deform_attn/cuda/ms_deform_attn_cuda_kernel.cu`
- **作用**：kernel包装和调度函数
- **功能**：
  - 根据硬件特性选择合适的kernel（标准版或Tensor Core版）
  - 计算grid和block维度
  - 处理输入参数和内存布局

### 脚本文件

#### `run.sh`
- **作用**：自动化编译和运行脚本
- **主要参数**：
  - `NAME`：源文件名（不含.cu后缀）
  - `CONFIG`：GPGPU-Sim配置（如RTX3070、GTX480等）
  - `ARCH`：CUDA架构（如sm_86、sm_75等）
  - `IFBUILD`：是否编译（1编译，0跳过）
  - `CONFIG_SELECT`：运行模式（0仅构建环境，1运行，2重建环境并运行）
- **工作流程**：
  1. 检查源文件和配置是否存在
  2. 创建/更新仿真环境（复制配置文件）
  3. 使用nvcc编译CUDA程序
  4. 设置GPGPU-Sim环境变量
  5. 在仿真环境中运行程序
  6. 输出结果到out/目录

#### `syn_gpgpu_sim.sh`
- **作用**：同步和更新GPGPU-Sim配置
- **用途**：从configs/目录同步最新配置到sim/目录

### 输出目录

#### `bin/`
- 存放编译后的CUDA可执行文件
- 文件名与源文件对应（如test.cu → bin/test）

#### `out/`
- 存放程序运行结果和GPGPU-Sim统计信息
- 格式：`${NAME}_${CONFIG}.txt`
- 包含：执行时间、kernel统计、内存访问等

#### `sim/`
- 存放各配置的GPGPU-Sim运行环境
- 每个子目录对应一个GPU配置
- 包含完整的GPGPU-Sim配置文件集

## Deformable Attention 算法说明

### 算法原理

Deformable Attention是一种灵活的注意力机制，核心思想是：
1. 不对所有空间位置计算注意力（传统self-attention）
2. 而是学习每个查询点的采样位置偏移（offset）
3. 只对采样位置处的特征计算注意力

### 数学公式

```
output(q) = Σ_k w_k · V(p_q + Δp_k)
```

其中：
- `q`：查询位置
- `k`：采样点索引
- `w_k`：注意力权重（learnable）
- `p_q`：参考位置
- `Δp_k`：位置偏移（learnable）
- `V(·)`：对value特征图的采样（使用双线性插值）

### 输入张量

1. **value**：`[batch, spatial_size, num_heads, channels]`
   - 输入特征图，被采样的对象

2. **spatial_shapes**：`[num_levels, 2]`
   - 每个特征层级的空间尺寸（高度和宽度）

3. **level_start_index**：`[num_levels]`
   - 每个层级在value中的起始索引

4. **sampling_loc**：`[batch, num_query, num_heads, num_levels, num_point, 2]`
   - 采样位置坐标（归一化到[0,1]）

5. **attn_weight**：`[batch, num_query, num_heads, num_levels, num_point]`
   - 注意力权重

### 输出张量

- **output**：`[batch, num_query, num_heads * channels]`
  - 聚合后的特征向量

## 编译和运行

### 前置要求

1. NVIDIA CUDA Toolkit（支持目标架构）
2. GPGPU-Sim（已正确安装和配置）
3. GCC编译器

### 编译

```bash
cd /home/koala/gpgpu-sim_distribution/deAttn

# 编译test程序
./run.sh
```

脚本会自动：
- 编译`src/test.cu`
- 生成`bin/test`可执行文件

### 运行

```bash
# 在GPGPU-Sim上运行（使用RTX3070配置）
./run.sh

# 查看输出结果
cat out/test_RTX3070.txt
```

### 自定义配置

修改`run.sh`中的参数：

```bash
NAME=test              # 程序名
CONFIG=RTX3070        # GPU配置（对应configs/tested-cfgs/中的目录）
ARCH=sm_86            # CUDA架构版本
IFBUILD=1             # 1=编译，0=直接运行已编译的程序
CONFIG_SELECT=2       # 0=仅构建环境，1=运行，2=重建并运行
```

## 性能优化

### Tensor Core版本

当满足以下条件时，自动启用Tensor Core优化：
- GPU架构 ≥ Ampere（SM80+）
- 批量大小足够大
- 数据对齐要求满足

### GPGPU-Sim适配

针对仿真器特性进行了优化：
- 简化复杂的控制流
- 减少寄存器使用
- 优化内存访问模式
- 避免使用仿真器不支持的特性

### 性能指标

程序输出包含：
- **执行时间**：kernel总运行时间（ms）
- **吞吐量**：每秒处理的查询数量
- **计算正确性**：输出误差统计
- **GPGPU-Sim统计**：指令数、缓存命中率等（在out/文件中）

## 调试技巧

### 启用调试模式

修改`run.sh`：
```bash
IFDEBUG=1  # 启用GDB调试
```

### 常见问题

1. **编译错误**：检查CUDA版本和架构是否匹配
2. **运行时错误**：查看`out/test_*.txt`中的错误信息
3. **结果不正确**：对比参考实现，检查输入数据

## 参考

- 原始论文：Deformable DETR (ICLR 2021)
- 原始实现：https://github.com/fundamentalvision/Deformable-DETR
- GPGPU-Sim文档：http://www.gpgpu-sim.org/

## 开发注意事项

1. **避免PyTorch依赖**：所有代码必须是纯CUDA
2. **GPGPU-Sim兼容性**：测试新特性时先验证仿真器支持
3. **内存管理**：使用`cudaMalloc`/`cudaFree`，避免ATen的内存分配器
4. **数据类型**：优先使用标准C++/CUDA类型，避免Torch专有类型

## 许可证

继承自Deformable-DETR的Apache License 2.0

---

**最后更新**：2025年10月
**维护者**：GPGPU-Sim团队
