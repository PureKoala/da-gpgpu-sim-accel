# Deformable Attention Kernel 书写指导文档（Baseline + Optimized）

**用途**：指导在 **GPGPU-Sim** 中实现 Deformable Attention 的两个版本 kernel（Baseline / Optimized）的**代码结构、线程映射、数据组织与伪代码**，并明确哪些步骤由**硬件模型（Function Call 拦截）**实现。
**定位**：这是一份“怎么写代码”的指导性文档；不强调设计优势，只强调**可实现**与**可验证**。
**版本**：v1.1（按你给的 v1.0 结构重写并补齐 chunking + 4×4 compute tile）

---

## 0. 实现假设与范围

1. **本指导以“标量采样”伪代码为主**：即 `bilinear_sample()` 返回一个 `float`。

* 如果你要模拟真实 DA 的向量通道（例如每头 `C_PER_HEAD = C_IN/NUM_HEADS`），只需在插值后加一层 `for c in range(C_PER_HEAD)` 的向量累加即可（结构不变）。

2. **GEMM / Softmax 可以在 kernel 外生成**（或在模拟中视作前置完成）：kernel 主要消费

* `sampling_locs[q,h,l,p] -> (x,y)`
* `attn_weights[q,h,l,p] -> w`
* `value_maps[l]`（多尺度特征图）

3. **Optimized 版本只要求体现：剪枝（PCB）、分块（TBC+TMA）、片上插值（smem）以及 4×4 compute tile 的组织方式**。其中 **PCB/TBC/TMA/Interpolation 由硬件模型实现**（Function Call 拦截 + 固定延迟），kernel 只负责调用与数据流组织。

---

## 0.1 硬件设计要点抽取（来自架构文档）

> 目的：把硬件设计“算法点”提取为 kernel 可执行的步骤与调用点，确保后续 kernel 书写计划与硬件模型对齐。

### 硬件架构概述（已实现）✅
1. **5 级流水线阶段**（固定延迟模型）：
   ```
   PCB (1c) → GTC (0c) → TBC (4-7c) → TMA&Storage (22c) → Interpolation (3c)
   ```

2. **Function Call 拦截机制**（基于FMR成功经验）✅
   - `__deform_pcb(weights_tile, threshold, enable)` → 1 cycle
   - `__deform_tbc(coords_active)` → 4-7 cycles 
   - `__deform_tma(base_tile, mode, level)` → 16 cycles
   - `__deform_interp(tile_neighbors, frac_coords)` → 3 cycles

### 各模块算法要点与kernel调用点
3. **PCB：权重阈值筛选**（✅已实现）
   - 硬件：输入 `weights[16][16]` + `threshold`，输出 `mask[16][16]`
   - Kernel调用点：每16×16 processing tile开始时调用

4. **GTC：操作数隔离**（✅已实现）
   - 硬件：与PCB并行执行，零延迟
   - Kernel实现：基于PCB mask进行坐标谓词化

5. **TBC：两阶段聚合决策**（✅已实现）
   - Phase1：First-8 voting + 命中率判断
   - Phase2：包围盒与模式检测（Horizontal/Vertical/XOR/Discrete）
   - Kernel调用点：收集16×16 tile内有效坐标后调用

6. **TMA & Storage：Tile加载与无冲突存储**（✅已实现）
   - TMA：按`base_tile_x/y`和`mode`进行tile载入（16 cycles）
   - Storage：bank映射与tracker逻辑（6 cycles）
   - Kernel调用点：TBC决策完成后，针对非Discrete模式

7. **Interpolation：双线性插值**（✅已实现）
   - 硬件：2×2邻域 + 权重 → 插值结果（3 cycles）
   - Kernel调用点：Tile模式在共享内存，Discrete模式在全局内存

### GPGPU-Sim集成状态
8. **已完成集成**：
   - ✅ CUDA Function Call 拦截逻辑（cuda-sim.cc）
   - ✅ 5个模块功能模型实现（deform_attn_unit.h/.cc）
   - ✅ 参数读取与延迟建模（deform_attn_impl.cc）

9. **待完成集成**：
   - ⏳ 流水线单元创建与调度（shader.cc）
   - ⏳ 指令分发与cycle调用（shader.cc）
   - ⏳ 性能统计收集（stats.cc）

---

## 0.2 后续 kernel 书写计划（与硬件模型对齐）

### A. Baseline kernel（软件对照）
1. 线程映射：1 thread = 1 query（串行 128 points）。
2. 仅使用 `bilinear_sample_global()`，不调用硬件阶段函数。
3. 作为“功能正确性”与“数值基线”。

### B. Optimized kernel（硬件协同）
1. **Chunking 与线程映射**：16 queries/block × 16 points/chunk，形成 16×16 处理块。
2. **Stage I PCB（硬件调用点）**：
    - 通过 `__deform_pcb(weights_tile, threshold, enable)` 生成 `mask`。
3. **Stage II GTC（kernel 内）**：
    - 对 `mask` 谓词化，清零无效坐标（不进入后续采样）。
4. **Stage III TBC（硬件调用点）**：
    - 通过 `__deform_tbc(coords_tile)` 输出 `mode/base_tile`。
5. **Stage IV TMA & Storage（硬件调用点）**：
    - 通过 `__deform_tma(base_tile, mode, level)` 完成 tile 载入与 swizzle。
6. **Stage V Interpolation（硬件调用点）**：
    - 通过 `__deform_interp(tile_or_neighbors, frac)` 输出插值结果。
7. **Discrete fallback（kernel 内）**：
    - `mode=Discrete` 时走 `bilinear_sample_global()`。
8. **Reduce + Writeback（kernel 内）**：
    - 每 query 的 16 线程归约，避免 atomic。

> 说明：所有 `__deform_*()` 调用由 GPGPU-Sim 内部实现固定延迟与统计；kernel 负责调度与数据流组织。

---

## 1. 全局参数（沿用）

```python
LEVEL_DIMS = [(480,270), (240,135), (120,67), (60,33)]
C_IN = 256
NUM_HEADS = 8
NUM_LEVELS = 4
NUM_POINTS_PER_HEAD = 4
TOTAL_POINTS_PER_QUERY = NUM_HEADS * NUM_LEVELS * NUM_POINTS_PER_HEAD  # 128
```

---

## 2. 数据布局与索引（强烈建议统一）

### 2.1 张量逻辑形状

* `attn_weights[q][h][l][p]`  (float)
* `sampling_locs[q][h][l][p]` (float2，归一化或像素坐标均可，但要统一)
* `value_maps[l]`：每层特征图（建议按 **HWC 或 CHW** 固定一种；下面示例用 `HWC` 的标量通道）

### 2.2 Flatten（便于在 kernel 内快速索引）

将 `(h,l,p)` 展开成 `t in [0,128)`：

```python
def t_to_hlp(t):
    # t = (((h * NUM_LEVELS) + l) * NUM_POINTS_PER_HEAD) + p
    p = t % NUM_POINTS_PER_HEAD
    t //= NUM_POINTS_PER_HEAD
    l = t % NUM_LEVELS
    h = t // NUM_LEVELS
    return h, l, p
```

---

## 3. Baseline Kernel（Naive）书写指导

### 3.1 线程映射（Baseline）

* **每个线程处理一个 query**：`thread_global = q`
* 线程内部串行遍历 128 points

### 3.2 Baseline 伪代码（Python 风格）

```python
def baseline_kernel(value_maps, sampling_locs, attn_weights, output):
    q = get_global_id()  # 1 thread = 1 query
    acc = 0.0

    for t in range(TOTAL_POINTS_PER_QUERY):  # 128
        h, l, p = t_to_hlp(t)
        w = attn_weights[q][h][l][p]
        x, y = sampling_locs[q][h][l][p]     # 像素坐标或归一化坐标

        # 直接全局内存的四邻域读取
        v = bilinear_sample_global(value_maps[l], x, y)  # 4 reads + interp
        acc += v * w

    output[q] = acc
```

### 3.3 Baseline 实现要点

* **不使用 shared memory**
* **不剪枝**
* `bilinear_sample_global()` 内部做边界处理（clamp 或 padding），否则容易越界

---

## 4. Optimized Kernel（指导性 5-stage）书写指导

> 关键补齐点：
>
> 1. **chunking**：`16 queries × 16 points` 组成一个 16×16 的处理块（对应你文档里的 mask/valid_coords 维度）
> 2. **4×4 compute tile**：每个 query 的每个 chunk 有 16 个点，天然就是一个 `4×4` micro-tile（计算组织单位）

---

### 4.1 线程映射（Optimized 推荐落地版）

**Block 粒度**：1 block 处理 `QG = 16` 个 queries（一个 query group）

* `q_base = block_id * 16`
* `q_local in [0..15]`

**Chunk 切分**：每个 query 有 128 points，按 16 points/chunk 切分：

* `NUM_CHUNKS = 128 / 16 = 8`
* `chunk_id in [0..7]`
* `t_in_chunk in [0..15]`

**线程数**：推荐 `256 threads/block`（简单直观）

* `tid in [0..255]`
* `q_local = tid // 16`
* `t_in_chunk = tid % 16`

这样每个线程对应一个 **(query, point_in_chunk)**，刚好对应 `mask[16][16]`。

---

### 4.2 shared memory 组织（最小可实现）

对于 **Tile 模式**（聚集访问）：

* `tile_buffer[16][16]`：缓存一个 level 的 16×16 tile（标量示例）
* 如果要扩展到通道维，可改为 `tile_buffer[C_TILE][16][16]` 或按 `C_PER_HEAD` 组织

对于 **Discrete 模式**（不聚集）：

* 最简单落地：不强制用 tile_buffer；直接对有效点做 global 采样（但保留 PCB 剪枝）
* 如果你坚持“都走 smem”：可以用 `gather_buffer[num_active][4]` 存每个点的 4 邻域（实现复杂度略高）

下面伪代码以“两条路径都可实现”为目标：Tile 用 smem；Discrete 直接 global（更好写、更稳）。

---

### 4.3 关键函数（指导性 stub）

> 说明：`__deform_*()` 代表**硬件实现的阶段函数**（由 GPGPU-Sim 拦截）。
> kernel 侧只做**调用、同步与数据流组织**。

```python
def pcb_mask(w, threshold):
    # 硬件实现（Function Call 拦截）
    return __deform_pcb(w, threshold)  # True=skip

def tbc_decide(coords_active):
    """
    输入：本chunk内所有有效点的坐标集合（跨16 queries）
    输出：mode, base_x, base_y
    mode: 0=Horizontal, 1=Vertical, 2=XOR, 3=Discrete
    """
    # 硬件实现（Function Call 拦截）
    return __deform_tbc(coords_active)

def swizzle(mode, x, y):
    # 给出最小可用的三种排布；你也可以后续替换为真实策略
    if mode == 0:      # Horizontal-friendly
        return x, y
    elif mode == 1:    # Vertical-friendly（示意：转置映射）
        return y, x
    elif mode == 2:    # XOR-friendly（示意：x与y低位异或）
        return x ^ (y & 0x7), y
    else:              # Discrete 不用 swizzle
        return x, y
```

---

## 5. Optimized Kernel 伪代码（按“阶段+chunk”可直接写）

> 说明：为了“指导代码书写”，下面写成一个结构清晰、容易翻译成 CUDA/GPGPU-Sim kernel 的版本。
> 你在 GPGPU-Sim 里如果需要固定延迟，可以把每个阶段封装成 `__deform_*()` intrinsic 或者用计数/空操作模拟。

```python
def optimized_kernel(value_maps, sampling_locs, attn_weights, output, threshold):
    # --- block / thread mapping ---
    tid = thread_id()
    block = block_id()

    q_base = block * 16
    q_local = tid // 16          # 0..15
    t_in_chunk = tid % 16        # 0..15
    q = q_base + q_local

    # --- per-block shared memory (tile mode) ---
    # 这里只示意一个level的标量tile；真实实现可按level循环重用
    __shared__ float tile_buffer[16][16]

    # 每个query的最终累加（示意标量）
    # 如果你做向量通道，这里应是一个向量acc[c]
    acc_q = 0.0

    NUM_CHUNKS = 8  # 128/16

    for chunk_id in range(NUM_CHUNKS):
        # 本chunk内，该线程对应的全局point索引 t
        t = chunk_id * 16 + t_in_chunk  # 0..127
        h, l, p = t_to_hlp(t)

        # ---------------------------
        # Stage I: PCB（点级剪枝）
        # ---------------------------
        w = attn_weights[q][h][l][p]
        skip = pcb_mask(w, threshold)

        # ---------------------------
        # Stage II: GTC（操作数隔离/谓词化）
        # ---------------------------
        # 注意：不要 return；只对“这个点”做continue/谓词化
        if not skip:
            x, y = sampling_locs[q][h][l][p]
        else:
            x, y = 0.0, 0.0  # 被隔离的坐标不会参与后续聚合/访存

        # ---------------------------
        # Stage III: TBC（block级决策：本chunk是否Tile）
        # ---------------------------
        # 先收集本block里所有有效坐标（伪代码写法；CUDA中可用shared+compaction）
        coords_active = block_collect_active_coords(skip, x, y)

        if tid == 0:
            mode, base_x, base_y = tbc_decide(coords_active)
            shared_mode = mode
            shared_base_x = base_x
            shared_base_y = base_y
        block_sync()

        mode = shared_mode
        base_x = shared_base_x
        base_y = shared_base_y

        # ---------------------------
        # Stage IV: TMA（Tile模式：加载16×16到smem）
        # ---------------------------
        if mode != 3:
            # 256线程协作加载：每线程加载一个tile元素（或按你需要改成更合并的方式）
            # 映射：tid -> (tx, ty)
            tx = tid % 16
            ty = tid // 16
            gx = base_x + tx
            gy = base_y + ty

            tile_buffer[ty][tx] = global_load(value_maps[l], gx, gy)
            block_sync()

        # ---------------------------
        # Stage V: Interpolation（4×4 compute micro-tile）
        # ---------------------------
        if skip:
            # 本点被剪枝，直接跳过贡献
            continue

        # 4×4 micro-tile 显式出现：每个 query 的 chunk 16点 = 4×4
        mt_x = t_in_chunk % 4    # 0..3
        mt_y = t_in_chunk // 4   # 0..3
        # 注：mt_x/mt_y用于“计算组织”，真正采样坐标仍来自(x,y)

        if mode != 3:
            # Tile路径：从 smem 读四邻域
            # 将全局坐标映射到tile内坐标（相对base）
            lx = int(x) - base_x
            ly = int(y) - base_y
            fx = x - int(x)
            fy = y - int(y)

            # swizzle映射（示意）
            sx0, sy0 = swizzle(mode, lx,   ly)
            sx1, sy1 = swizzle(mode, lx+1, ly)
            sx2, sy2 = swizzle(mode, lx,   ly+1)
            sx3, sy3 = swizzle(mode, lx+1, ly+1)

            p00 = tile_buffer[sy0][sx0]
            p01 = tile_buffer[sy1][sx1]
            p10 = tile_buffer[sy2][sx2]
            p11 = tile_buffer[sy3][sx3]

            v = bilinear_interp(p00, p01, p10, p11, fx, fy)
        else:
            # Discrete路径：直接全局四邻域采样（最稳、最容易先跑通）
            v = bilinear_sample_global(value_maps[l], x, y)

        # 点贡献累加到“该query的局部acc”
        acc_q += v * w

        # （可选）如果你要严格模拟“阶段固定延迟”，可在这里插入 stage_latency() hook

    # --- 写回 output：避免atomic，做每query一条写回 ---
    # 这里示意：同一个query在block内有16个线程（t_in_chunk=0..15）都在累加acc_q，
    # 你需要做一次 reduce，把这16个线程的acc_q归约到一个线程再写回
    sum_q = reduce_sum_across_16_threads_same_query(acc_q)

    if t_in_chunk == 0:
        output[q] = sum_q
```

---

## 6. 你在 CUDA/GPGPU-Sim 里“照着写”时的落地提示

### 6.1 block_collect_active_coords 的最小实现方式（建议先写能跑的）

第一版不做复杂 compaction，直接用 shared memory 存每线程的 `(valid, x, y)`：

* `smem_valid[256]`
* `smem_x[256]`
* `smem_y[256]`
  然后由 `tid==0` 扫描 256 项构建 bbox / 统计即可。
  这会慢，但**最容易对**，而且这是“指导文档”的目标：先正确，再逐步优化。

### 6.2 reduce_sum_across_16_threads_same_query（每query 16线程归约）

最简单版：

* `smem_acc[256] = acc_q`
* `__syncthreads()`
* `t_in_chunk==0` 的线程把同 query 的 16 项相加写回

### 6.3 关于“每 level 单独 tile”

上面伪代码在 chunk 内自然携带 `l`（level）。如果一个 chunk 内的点跨多个 level：

* 最简单写法：仍然按点各自的 `l` 做离散处理（会降低 tile 复用）
* 更“规整”的写法：在 chunk 内再按 level 分组处理（实现复杂度更高）
  **建议**：第一版先不按 level 分组，确保功能正确即可。

---

## 7. 实现优先级（写代码时按这个顺序最稳）

1. **Baseline**：先把 `bilinear_sample_global()` 跑通，验证输出正确
2. **Optimized v0（只加 PCB）**：在 baseline 循环里加 `if w<thr: continue`，验证数值变化符合预期
3. **Optimized v1（chunking + per-query reduce）**：引入 16 queries/block、8 chunks、16线程归约
4. **Optimized v2（TBC + Tile）**：先用 bbox 判定 Tile/Discrete；Tile 路径只实现 `mode=XOR` 一种 swizzle
5. **Optimized v3（扩展 mode）**：补齐 Horizontal/Vertical/XOR 三种映射（Discrete 继续 fallback）

---

## 8.（可直接照抄的）GPGPU-Sim 配置字段占位

> 你可以保留这些参数作为“阶段延迟占位”，但在代码层面先不强制体现它们的优势，只作为模拟 hook。

```cpp
-gpgpu_deform_attn_avail 1
-gpgpu_num_deform_units 1
-gpgpu_deform_pcb_latency 1
-gpgpu_deform_tbc_phase1_latency 4
-gpgpu_deform_tbc_phase2_latency 3
-gpgpu_deform_tma_latency 16
-gpgpu_deform_storage_latency 6
-gpgpu_deform_interp_latency 3
```

---

如果你下一步要我“直接改写你原文档的全文版本（保持你原来的章节编号与格式）”，我也可以按这份 v1.1 的落地结构，把你 v1.0 中不适合指导写代码的部分（比如 `return`、atomicAdd、16×16 语义不清）全部替换成**可直接翻译到 kernel 的版本**。
