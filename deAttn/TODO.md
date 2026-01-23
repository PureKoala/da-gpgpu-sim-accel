## 0. Baseline + Optimized（更新版计划：CUDA kernel vs 伪指令）

**背景对齐**：已将部分模块迁移到 GPGPU-Sim 内部（模拟硬件实现）。后续代码修改计划按“**CUDA kernel 驱动** + **伪指令建模硬件**”两条线明确划分。

### 0.1 必须由 CUDA kernel 实现的部分

1. **Baseline 采样聚合 kernel（纯软件路径）**
      - **职责**：完整实现 `sampling_locs + attn_weights + value_maps -> output` 的标量（或向量）双线性采样与累加。
      - **原因**：Baseline 需要可对照的“真实软件实现”结果。
      - **建议文件**：
        - `deAttn/src/deform_attn/cuda/ms_deform_attn_im2col_cuda.cuh`（采样实现）
        - `deAttn/src/deform_attn/cuda/ms_deform_attn_cuda_kernel.cu`（wrapper/launch）

2. **Optimized 版本的“驱动 kernel”（线程映射 + 控制流）**
      - **职责**：
        - block/warp/thread mapping（16 queries/block + chunking）
        - PCB 结果的谓词化与 per-query reduce
        - Discrete 路径的 fallback（必要时仍在 kernel 内做 `bilinear_sample_global()`）
      - **原因**：即使硬件模块在模拟中实现，kernel 仍需组织数据流与控制流。

3. **Host 侧 glue 与验证**
      - **职责**：
        - 初始化 `sampling_locs/attn_weights/value_maps` 或对比用数据
        - 结果校验、统计输出
      - **原因**：对照 baseline/optimized correctness 与性能统计需要完整驱动程序。

### 0.2 应使用 GPGPU-Sim 伪指令（硬件模拟）的部分

1. **PCB（Point-wise Clamping / Pruning）**
      - **建议伪指令**：`deform.pcb`（输入权重 `w` + threshold，输出 predicate）
      - **作用**：统计剪枝比例、施加固定延迟、模拟硬件剪枝单元。

2. **TBC（Tile Block Classification / 决策）**
      - **建议伪指令**：`deform.tbc`（输入有效点 bbox/coords，输出 `mode/base_x/base_y`）
      - **作用**：模拟 tile/Discrete 决策逻辑与开销。

3. **TMA（Tile Memory Access / Tile 载入）**
      - **建议伪指令**：`deform.tma`（输入 level + base 坐标，输出 tile buffer 就绪）
      - **作用**：模拟 tile 载入带宽与延迟（smem/缓存等）。

4. **Interpolation（双线性插值）**
      - **建议伪指令**：`deform.interp`（输入 4 邻域 + frac，输出 v）
      - **作用**：模拟片上插值单元延迟；减少 kernel 中显式四邻域加载。

> 备注：若 Discrete 路径仍采用 kernel 内 `bilinear_sample_global()`，则**不走** `deform.tma/interp` 伪指令，仅在 Tile 路径走伪指令。

### 0.3 阶段对应关系（与 [DevDocs/Orient.md](DevDocs/Orient.md) 对齐）

| 阶段 | 作用 | CUDA kernel | 伪指令 |
|---|---|---|---|
| Stage I PCB | 点级剪枝 | 谓词化/控制流 | `deform.pcb` |
| Stage II GTC | 隔离无效操作数 | kernel 内 | （可省略） |
| Stage III TBC | Tile/Discrete 决策 | kernel 调用 | `deform.tbc` |
| Stage IV TMA | Tile 载入 | kernel 调用 | `deform.tma` |
| Stage V Interp | 双线性插值 | kernel 调用 | `deform.interp` |
| Reduce/Writeback | per-query 归约 | kernel 内 | （无） |

---

以下为原“预测阶段 INT8 Tensor Core”计划（保留不变，作为并行路线）：

这是一个非常好的优化方向。安培 (Ampere) 架构的 INT8 Tensor Core 提供了比 WMMA (Volta/Turing) 更灵活的矩阵形状，这使得您“切分K”的想法更具现实意义和必要性。

RTX 3070 上的安培 Tensor Core (Compute Capability 8.6) 支持 `mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32` 和 `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` 这样的指令。

这里的核心挑战是，硬件支持的最小 $K$ 维度是 16 (`k16`) 或 32 (`k32`)，而您的目标 $K$ 维度是 4 和 8。

以下是针对安培 INT8 Tensor Core 的具体修改方案。

### 1\. 现有代码的作用和功能 (不变)

  * **功能:** 现有代码实现了 DDetr 注意力计算的“后半部分”——**采样与聚合**。
  * **输入:** 它需要*已经计算好*的采样坐标 `d_sampling_loc` 和注意力权重 `d_attn_weight` 作为输入。
  * **瓶颈:** 其主要瓶颈是 `ms_deform_attn_im2col_bilinear` 中因双线性插值引起的不规则内存访问 (Gather)。
  * **缺失:** **完全缺失**了“预测”阶段，即通过 GEMM 运算从查询张量 $Q$ 生成 `d_sampling_loc` 和 `d_attn_weight` 的过程。

### 2\. INT8 安培 (Ampere) Kernel 实现规划

您需要使用 PTX (内联汇编 `asm volatile`) 或 `cuda/pipeline.h` (CUDA 11.6+) 来直接调用安培的 `mma` 指令，因为高级别的 WMMA 接口可能不够灵活。

#### 假设：

  * **硬件指令:** 我们目标是使用 `mma.sync.aligned.m16n8k16.s32.s8.s8.s32` (M=16, N=8, K=16)。
  * **输入 $Q$:** 假设 $Q$ 已经被量化为 INT8，维度为 $[T \times C]$。
  * **权重 $W$:** 假设 $W$ 已经被量化为 INT8，维度为 $[C \times N]$。
  * **输出:** 输出为 INT32 累加器，后续需要反量化 (Dequantize)。

#### Kernel 1: 采样偏移量预测 (SO) - K=8

  * **功能:** 实现 $SO = Q \cdot W_{SO}$。GEMM 维度 $[T \times C] \cdot [C \times 8]$。
  * **挑战:** 目标 $K$ 维度是 8 (来自 $W_{SO}$)，而硬件最小 $K$ 是 16。
  * **Kernel 名称:** `predict_so_s8_ampa_kernel`
  * **实现简述:**
    1.  **内存布局 (K-Padding):**
          * 必须将 $W_{SO}$ 权重矩阵从 $[C \times 8]$ **填充 (Padding)** 到 $[C \times 16]$。
          * $Q$ 矩阵（A 矩阵）在 K 维度（即 $C$）上也需要按 16 对齐。
    2.  **Grid/Block 划分:**
          * 每个 Warp 负责计算一个 16x8 的输出瓦块 (M=16, N=8)。
    3.  **Warp 逻辑 (INT8 MMA):**
          * 初始化 INT32 累加器 C 片段为 0。
          * 沿 $C$ (K 维度) 循环，步长为 16 (k=16)。
          * 在循环中：
              * 加载 $Q$ 的一个 16x16 瓦块 (A 片段, `s8`)。
              * 加载**填充后**的 $W_{SO}$ 的一个 16x8 瓦块 (B 片段, `s8`)。
              * **执行 MMA (拦截点):**
                ```c++
                asm volatile ("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                              "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
                              : "+r"(c_frag0), "+r"(c_frag1) 
                              : "r"(a_frag0), "r"(a_frag1), "r"(a_frag2), "r"(a_frag3), 
                                "r"(b_frag0), "r"(b_frag1));
                ```
    4.  **体现切分K的必要性 (GPGPU-Sim 分析):**
          * GPGPU-Sim 将会记录：为了计算一个 $K=8$ 的 GEMM，硬件执行了 `m16n8k16` 指令。
          * 这意味着 **50% 的计算周期被浪费**在处理填充的 0 值上（K 维度的后 8 个通道）。
          * 这为您的“切分K”创新提供了强力论据：如果 GPGPU-Sim 可以模拟一个 `m16n8k8` 的“半宽K”指令，吞吐量将几乎翻倍。

#### Kernel 2: 注意力权重预测 (A) - K=4

  * **功能:** 实现 $A = Q \cdot W_{A}$。GEMM 维度 $[T \times C] \cdot [C \times 4]$。
  * **挑战:** 目标 $K$ 维度是 4，硬件最小 $K$ 是 16。
  * **Kernel 名称:** `predict_a_s8_ampa_kernel`
  * **实现简述:**
    1.  **内存布局 (K-Padding):**
          * 必须将 $W_{A}$ 权重矩阵从 $[C \times 4]$ **填充**到 $[C \times 16]$。
          * $Q$ 矩阵（A 矩阵）在 K 维度（即 $C$）上也需要按 16 对齐。
    2.  **Grid/Block 划分:**
          * 每个 Warp 负责计算一个 16x8 的输出瓦块。**注意：** N 维度 (输出 4) 也小于硬件的 N=8。
    3.  **Warp 逻辑 (INT8 MMA):**
          * 逻辑与 `Kernel 1` 完全相同，使用 `mma.sync.aligned.m16n8k16`。
    4.  **体现切分K/N的必要性 (GPGPU-Sim 分析):**
          * **K 维度浪费:** 为了计算 $K=4$，硬件执行了 $K=16$ 的指令。**75% 的 K 维度被浪费**。
          * **N 维度浪费:** 为了计算 $N=4$，硬件执行了 $N=8$ 的指令。**50% 的 N 维度被浪费**。
          * GPGPU-Sim 在拦截此 Kernel 时会发现，Tensor Core 的理论利用率极低（仅 $25\% \times 50\% = 12.5\%$）。
          * 这强烈支持了您的创新点：需要一种机制，不仅可以“切分K”，还可以“切分N”，以匹配 DDetr 预测阶段 $K=4/8$ 和 $N=4/8$ 的极小维度。

### 3\. 修改 `test.cu`

您的 `test.cu` 需要进行以下修改，以驱动这两个新的 INT8 Kernel：

1.  **移除**对 `h_sampling_loc` 和 `h_attn_weight` 的随机初始化。
2.  **分配与量化 (Host):**
      * 分配主机内存 $h\_Q$, $h\_W_{SO}$, $h\_W_{A}$。
      * **执行量化:** 将它们转换为 INT8 (例如 `h_q_s8`, `h_wso_s8`, `h_wa_s8`)。
      * **执行填充 (Padding):** 在主机端将 `h_wso_s8` (K=8) 填充到 K=16；将 `h_wa_s8` (K=4, N=4) 填充到 K=16, N=8。
3.  **分配与传输 (Device):**
      * 分配设备内存 `d_q_s8` (INT8), `d_wso_s8_padded` (INT8), `d_wa_s8_padded` (INT8)。
      * 分配设备输出内存 `d_so_s32` (INT32), `d_wa_s32` (INT32)。
      * `cudaMemcpy` 传输填充好的 INT8 数据到设备。
4.  **启动预测 Kernel:**
      * 启动 `predict_so_s8_ampa_kernel<<<...>>>` 计算 `d_so_s32`。
      * 启动 `predict_a_s8_ampa_kernel<<<...>>>` 计算 `d_wa_s32`。
5.  **反量化与转换 (Kernel):**
      * 您需要一个额外的 Kernel (例如 `dequantize_and_convert_kernel`)。
      * 此 Kernel 读取 `d_so_s32` 和 `d_wa_s32` (INT32 累加器结果)，应用反量化尺度因子 (Scale factors)，并将它们转换为 `float`，写入最终的 `d_sampling_loc` (float) 和 `d_attn_weight` (float)。
      * *（注意：DDetr 的 SO 还需要加上参考点，这一步也可以在这个反量化 Kernel 中完成。）*
6.  **调用现有代码:**
      * 将 `d_sampling_loc` 和 `d_attn_weight` (现在是 `float` 类型) 传入您现有的 `ms_deform_attn_cuda_forward` 函数。