# Deformable Attention 软件数据流与执行方案（当前实现）

**版本**：2026-01-25

本文件描述仓库当前代码的**实际软件数据流与执行方案**。历史设计、旧计划与过期说明已归档到 `DevDocs/backup/`。

---

## 1. 关键路径概览

### 1.1 Host 入口
- `deAttn_optim/src/deform_attn/cuda/ms_deform_attn_cuda_kernel.cu`
  - `ms_deform_attn_cuda_forward()` → `ms_deformable_im2col_cuda()`
  - `ms_deform_attn_cuda_backward()` → `ms_deformable_col2im_cuda()`

### 1.2 Forward kernel（两条路径）
- **Simple kernel（默认）**：`ms_deformable_im2col_gpu_kernel`
  - 每线程负责一个 `(b, q, head, c)`
  - 遍历 `level × point` 执行采样与插值
  - **不触发 `__deform_*` 硬件接口**
- **Grouped kernel（硬件加速路径）**：`ms_deformable_im2col_gpu_kernel_grouped`
  - block 映射：`16 queries × 8 heads`，固定 `channel`
  - 使用共享 tile（16×17）复用
  - **触发 `__deform_pcb / __deform_tbc / __deform_tma / __deform_interp`**

> 注意：当前 launcher 仍调用 simple kernel。
> 若要走硬件路径，需要切换 `ms_deformable_im2col_cuda()` 调用 grouped kernel。

---

## 2. Forward 数据流（Grouped Kernel / 硬件路径）

### 2.1 每个 level 的处理流程（q_local==0 负责组织）
1. 采样坐标与权重打包到局部数组 `s_coords/s_weights`
2. 调用 `__deform_pcb` 产生 mask（PCB）
3. 调用 `__deform_tbc` 输出 `mode/base_x/base_y/tile_w/tile_h`（TBC）
4. 若非 DISCRETE：调用 `__deform_tma` 加载 tile（TMA+Storage）

### 2.2 每个点的插值与累加
- 若 mode ≠ DISCRETE：调用 `__deform_interp` 在 tile 内插值
- 若 DISCRETE：按分支保持为 0（或 fallback 路径）

---

## 3. 硬件加速拦截与功能模型

### 3.1 拦截点
`src/cuda-sim/cuda-sim.cc`：在 `CALL_OP` 时，lane0 拦截以下函数
- `__deform_pcb`
- `__deform_tbc`
- `__deform_tma`
- `__deform_interp`

### 3.2 功能模型
`src/cuda-sim/instructions.cc` 中实现：
- `deform_pcb_impl`：读取权重并写 mask
- `deform_tbc_impl`：bbox/tile 参数计算，决定 mode
- `deform_tma_impl`：生成 global→shared 的访存访问序列
- `deform_interp_impl`：从 tile 中做插值

---

## 4. Block 级动态延迟模型（方案 A）

### 4.1 核心原则
- **延迟按 block+level 计一次**，避免每线程叠加
- 记录表 key：`{kernel_uid, ctaid(x,y,z), level}`
- 每个 stage 仅首次应用 latency，其后同 block/level 设为 1 cycle

### 4.2 影响因子
- **level scale**：`-gpgpu_deform_level_scale_l0..l3`
- **sparsity factor**：基于 `valid_points/num_points`
- **mode penalty**：`-gpgpu_deform_mode_penalty_0..3`
- **TMA per elem / Interp per point**：线性附加项

---

## 5. Level 稀疏度控制（方法 B）

Grouped kernel 中提供确定性剪枝：
- 开关：`#define DEFORM_PCB_USE_DETERMINISTIC_PRUNE 1`
- 稀疏度由宏控制：
  - `DEFORM_PCB_PRUNE_RATE_L0`
  - `DEFORM_PCB_PRUNE_RATE_L1`
  - `DEFORM_PCB_PRUNE_RATE_L2`
  - `DEFORM_PCB_PRUNE_RATE_L3`

实现逻辑：
- `should_prune_l0..l3` 使用上述宏按比例剪枝
- 通过覆盖 `s_weights` + `threshold=0.5f` 强制 mask
- 使稀疏度可控且与权重分布解耦

---

## 6. 关键可调参数（默认不改变行为）

- `-gpgpu_deform_global_scale` (default=1.0)
- `-gpgpu_deform_level_scale_l0..l3` (default=1.0)
- `-gpgpu_deform_sparsity_alpha` (default=0.0)
- `-gpgpu_deform_sparsity_min_factor` (default=0.5)
- `-gpgpu_deform_sparsity_max_factor` (default=2.0)
- `-gpgpu_deform_mode_penalty_0..3` (default=0)
- `-gpgpu_deform_tma_per_elem` (default=0.0)
- `-gpgpu_deform_interp_per_point` (default=0.0)

---

## 7. 运行检查清单
- 编译时开启 `__GPGPU_SIM__`
- `gpgpusim.config` 中设置：
  - `-gpgpu_deform_functional_sim_enabled 1`
  - 若需 20% 延迟下降：`-gpgpu_deform_global_scale 0.8`
- 确认 grouped kernel 被调用（否则硬件路径不生效）

