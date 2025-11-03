## GPGPU-Sim Deformable Attention (DA) 加速计划：FMR 模块集成

**文档目的：** 本文档旨在概述 GPGPU-Sim 模拟器在处理 Deformable Attention (DA) Kernel 时的现状，并提出一个详细的硬件修改计划。该计划的核心是引入一个名为 **FMR (Feature Map Reorganizer)** 的新型专用硬件单元，以解决当前模拟方案中的严重性能瓶颈。

-----

### 1\. 模拟器现状 (Baseline)

您当前的 GPGPU-Sim 版本准确地模拟了在**没有**专用硬件支持的情况下，使用标准 CUDA Core 运行 DA Kernel `grid_sampler` 操作的“软件实现”方案。

  * **实现方式：** 双线性插值采样 (Bilinear Interpolation Sampling) 是通过一长串通用的 CUDA Core 指令“手动”实现的。
  * **模拟器中的具体表现：**
    1.  **地址计算开销：** 模拟器必须使用 **SP 或 INT 计算单元** 来执行一系列整数运算，以从浮点坐标 `(x, y)` 计算出 4 个相邻像素的整数地址。
    2.  **多次串行访存：** 模拟器接着执行 **4 次独立**的 `ld.shared` 指令，以串行方式从内存中获取这 4 个像素值。
    3.  **计算资源占用：** 模拟器随后使用 **SP 计算单元** 执行一系列 `fma` / `mul` / `add` 指令，来计算这 4 个像素值的加权平均值，完成插值。
  * **核心问题：** 这种“手动”实现方式在**地址计算**、**访存**和**插值计算**三个方面都极其低效，完全占用了通用计算和访存流水线。

-----

### 2\. 待解决的核心瓶颈

FMR 计划旨在解决由“手动”采样带来的三大瓶颈：

1.  **地址生成瓶颈：** 手动计算 4 个采样地址和插值权重会消耗宝贵的 SP/INT 单元的执行周期。
2.  **访存延迟瓶颈：**
      * 标准SRAM/Cache无法保证4个相邻像素的并行读取。
      * [cite\_start]如 RETA-AD 论文所分析，读取这 4 个离散的网格数据通常需要 **4 个周期** [cite: 487]。
      * RETA-AD 的 Table IV 数据显示，采样操作虽然只占总计算量的2.6%，但其**平均延迟占到了总运行时间的 40.9%**。
3.  **计算资源瓶颈：** SP 单元被大量用于执行简单的、重复的插值数学运算。

-----

### 3\. FMR 修改计划

我们的计划是在 GPGPU-Sim 中模拟一个全新的、与 SM (Streaming Multiprocessor) 紧密集成的专用功能单元——**FMR**。

#### 3.1 硬件设计 (Hardware Design)

FMR 模块是一个自治的(autonomous)硬件单元，包含三个核心组件：

1.  **FMR 地址生成器 (Address Generator, AG)：**

      * **实现：** 这是 FMR 单元内部的专用计算逻辑。
      * **功能：** **\<font color="\#00B050"\>（这是您新增的关键部分）\</font\>** AG 接收来自寄存器的浮点坐标 `(x, y)`，**自动在硬件中**计算出：
          * (a) 双线性插值所需的 4 个整数地址。
          * (b) [cite\_start]4 个插值权重 (即 RETA-AD 中的 $\alpha, \beta, \gamma, \lambda$) [cite: 515-521]。
      * **收益：** 此过程**完全不占用** SM 中的 SP/INT 计算单元。

2.  **FMR 内存控制器 (4-Bank SMEM)：**

      * **实现：** 修改 GPGPU-Sim 的 `memory.cc`，在 Shared Memory (SMEM) 模型中划分出一块“可重组”区域。
      * [cite\_start]**功能：** 当使用新的`st.rfm.shared`指令写入时，该控制器将根据像素的2D坐标奇偶性，自动将数据“编码”并**交错存入 4 个独立的 SMEM Bank** [cite: 599-602]。
      * **保证：** 这种设计从硬件上保证了**任意一个双线性插值所需的 4 个相邻点，都恰好落在 4 个不同的 Bank 中**。

3.  **FMR 插值核心 (Bilinear Interpolation Core)：**

      * **实现：** 在 `shader.cc` 中定义一个新的专用功能单元 (SFU)，即 `FMR_Unit`。
      * [cite\_start]**功能：** 该单元在硬件上固化了双线性插值所需的所有数学逻辑（即 4 个乘法器和一个 4-2 压缩器/加法树） [cite: 705]。它不占用通用的 SP 单元。

#### 3.2 指令集扩展 (Instruction Set Extension)

我们将引入一条新的 PTX 指令来调用 FMR 硬件，彻底取代“手动”采样代码：

  * **新指令：** `ld.sample.bilinear.shared.f32 (dst_reg), [base_addr], {coord_reg.x, coord_reg.y};`

  * **GPGPU-Sim 中的执行流程：**

    1.  `shader_core_ctx::decode()` 译码该指令。
    2.  `shader_core_ctx::issue()` 将其分发到空闲的 `FMR_Unit`。
    3.  `FMR_Unit` 执行：
        a.  **(1 周期):** 内部的**地址生成器(AG)** 从寄存器读取浮点坐标，**自动计算**出 4 个整数地址和 4 个插值权重。
        b.  [cite\_start]**(1 周期):** 内存控制器根据 4 个整数地址，**并行**向 4 个 SMEM Bank 发出读请求（由于 FMR 的设计，这 4 个请求保证无冲突） [cite: 568-570]。
        c.  **(N 周期):** SMEM 返回 4 个值。
        d.  **(M 周期):** 内部的**插值核心**使用 AG 算出的权重和 SMEM 返回的值，计算最终的 `f32` 结果。
    4.  `shader_core_ctx::writeback()` 将结果写入 `dst_reg`。

### 4\. 预期收益与验证

  * **现状 (Baseline) 统计：** `fma.f32` / `add.s32` / `mul.s32` 指令计数极高（用于地址计算+插值）。`ld.shared` 指令计数极高。SP/INT 和 LDST 流水线占用率高。
  * **FMR (Proposed) 统计：**
      * 用于采样的所有 `fma/add/mul/ld.shared` 指令计数**降为 0**。
      * 被替换为数量少得多的 `ld.sample.bilinear.shared` 指令。
      * SP/INT 和 LDST 流水线**被释放**，可用于处理 GEMM 等其他计算。
      * **总执行周期数 (Total Cycles) 显著降低**，这部分减少的量即是 FMR 硬件带来的收益，量化了对 DA Kernel 中 NCI 操作的端到端加速。