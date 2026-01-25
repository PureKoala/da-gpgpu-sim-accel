# deAttn_optim Kernel 实现与 DevDocs 方案对照 (2026-01-25)

本说明聚焦 deAttn_optim 目录下的 CUDA kernel 实现，并对照 DevDocs 中的“功能方案”描述当前真实代码路径。结论：**DevDocs 里描述的硬件加速/Function Call 方案对应的是“已实现但未被调用”的 grouped kernel**；当前实际执行的是简化版 kernel。

---

## 1. 关键文件与入口

- `deAttn_optim/src/deform_attn/cuda/ms_deform_attn_cuda_kernel.cu`  
  Forward/Backward wrapper，负责调用 im2col/col2im。
- `deAttn_optim/src/deform_attn/cuda/ms_deform_attn_im2col_cuda.cuh`  
  具体 CUDA kernels（包含两个 forward kernel：simple 与 grouped）。
- `deAttn_optim/src/deform_attn/cuda/deform_attn_accelerator.cuh`  
  `__deform_*` 硬件接口 stub，供 GPGPU‑Sim 拦截。
- DevDocs：  
  `DevDocs/IMPLEMENTATION.md`, `DevDocs/SIMULATION_REPORT.md`, `DevDocs/PERF_TUNING.md`

---

## 2. 当前实际运行的 kernel 路径（as-is）

**调用链：**
```
ms_deform_attn_cuda_forward
  -> ms_deformable_im2col_cuda
      -> ms_deformable_im2col_gpu_kernel  (simple kernel)
```
对应代码位置：`deAttn_optim/src/deform_attn/cuda/ms_deform_attn_im2col_cuda.cuh`

**线程映射与计算流程（simple kernel）：**
- 每个线程负责一个输出元素 `(b, q, head, c)`。
- 外层遍历 `levels` 与 `points`。
- `DEFORM_ACCEL_FAST_MODE` 默认开启（`DEFORM_ACCEL_CORRECT_MODE` 未定义时）。
  - Fast 模式逻辑：
    - 通过 `if (p_col & 1)` **直接跳过一半采样点**（固定 50%）。
    - 采样只做 **最近邻**（读取单点）而非双线性。
  - Correct 模式逻辑：
    - 使用 `ms_deform_attn_im2col_bilinear` 做完整双线性插值。
- **注意：这个路径不会调用任何 `__deform_*` 硬件接口**。

结论：目前的 forward 计算**不触发 Function Call 拦截**，也不会进入 DevDocs 里描述的 PCB/TBC/TMA/Interp 模块。

---

## 3. 已实现但未接入的 grouped kernel（对应 DevDocs 方案）

**kernel 名称**：`ms_deformable_im2col_gpu_kernel_grouped`  
**定义位置**：`deAttn_optim/src/deform_attn/cuda/ms_deform_attn_im2col_cuda.cuh`

**设计点（与 DevDocs/SIMULATION_REPORT 对齐）：**
- 一个 block 覆盖 `16 queries (4x4) × 8 heads`，固定一个 `channel`。
- 使用共享内存 `sh_tile[8][16*17]`，每个 head 只加载一次 tile。
- 关键步骤：
  1. `__deform_pcb`：权重筛选（PCB）
  2. `__deform_tbc`：聚合/Tile 边界与模式选择（TBC）
  3. `__deform_tma`：tile 加载（TMA + Storage）
  4. `__deform_interp`：tile 内插值（Interpolation）
- `#ifdef __GPGPU_SIM__` 分支会走硬件模拟接口，符合 DevDocs“Function Call Only”的方案描述。

**但是**：目前 `ms_deformable_im2col_cuda` **没有调用该 kernel**，因此它“存在但不生效”。

---

## 4. DevDocs 功能方案 vs 实际代码（差异点）

**DevDocs 的方案（IMPLEMENTATION / SIMULATION_REPORT）：**
- Function Call 拦截 `__deform_pcb/__deform_tbc/__deform_tma/__deform_interp`
- 5 级硬件模块（PCB/TBC/TMA+Storage/Interp/Accumulate）
- Query/Head 分组、tile 复用（grouped kernel 结构）

**当前代码的实际行为：**
- 仅调用 simple kernel。
- 不触发 `__deform_*`，因此 **硬件模块没有被使用**。
- grouped kernel 结构虽然存在，但未被 launcher 选中。

---

## 5. 未使用/失效的宏与函数（与你提到的一致）

下面这些宏/函数目前**只定义，未被实际代码引用**（等价“死代码”）：

**(1) 访存策略宏**
- `DEFORM_ACCESS_MODE_L0/L1/L2/L3`  
  只在宏定义处出现，**没有任何引用**。

**(2) 剪枝率宏**
- `DEFORM_PCB_PRUNE_RATE_L0/L1/L2/L3`  
  未在 kernel 中参与逻辑判断。

**(3) 剪枝判断函数**
- `should_prune_l0/l1/l2/l3`  
  定义于 `ms_deform_attn_im2col_cuda.cuh`，**没有被调用**。

**(4) 惩罚迭代宏**
- `DEFORM_PCB_PENALTY_ITERS`  
- `DEFORM_DISCRETE_PENALTY_ITERS`  
- `DEFORM_TILE_MISS_PENALTY_ITERS`  
  均未被引用。当前被使用的是另一组 `DEFORM_SIM_*_PENALTY_ITERS`（仅在 grouped kernel 的 `__GPGPU_SIM__` 分支里）。

**(5) 单点接口 stub**
- `__deform_pcb_check` / `__deform_sample_point`  
  在 `deform_attn_accelerator.cuh` 中定义，但在任何 kernel 中均未使用。

---

## 6. 如果要让实现与 DevDocs 方案一致（最小改动路径）

1. 在 `ms_deformable_im2col_cuda` 中加入条件分支，**切换到 grouped kernel**。  
   典型选择条件：`#ifdef __GPGPU_SIM__` 或显式宏开关。
2. 将 `DEFORM_ACCESS_MODE_L*`、`should_prune_l*` 融入 grouped kernel 的 per-level 决策逻辑。
3. 对齐 `DevDocs/PERF_TUNING.md` 中的宏说明，使其对应真实使用路径。

---

## 7. 小结

- **你观察到的 `DEFORM_ACCESS_MODE_L2` / `should_prune_l2` 确实未使用。**
- DevDocs 中描述的硬件加速方案**对应 grouped kernel 的设计**，但当前实际执行的是 simple kernel。
- 如果目标是验证 Function Call 拦截与硬件模块功能，应优先接入 grouped kernel。
