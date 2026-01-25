# Deformable Attention 加速器仿真实现汇报

## 1. 报告目的
本报告用于组内汇报与毕业设计材料归档，完整说明 Deformable Attention 加速器在 GPGPU‑Sim 中的实现结构、内核组织方式与四个模块（PCB / TBC / TMA+Storage / Interpolation）的真实代码落点。

---

## 2. 仿真环境与参数

| 配置项 | 值 |
|---|---|
| 模拟器 | GPGPU‑Sim v4.2.0 |
| 目标 GPU | RTX 3070 (SM 8.6 Ampere) |
| 规模 | batch=1, query=256, heads=8, channels=32, levels=4, points=4 |

---

## 3. 总体结构与内核组织

### 3.1 五级流水线结构
```
PCB → TBC → TMA&Storage → Interpolation → Accumulate
```

### 3.2 Query/Head 复用组织（核心加速点）
- Query Group：16 个 query（4×4）
- Head Group：8 个 head
- 同一 level 内，**每个 head 只触发一次 TMA 加载**，在 16 个 query 内复用  
- tile 存储使用 pitch=17（16×17）以满足无冲突访问

对应内核组织代码（`deAttn_optim/src/deform_attn/cuda/ms_deform_attn_im2col_cuda.cuh`）：

```cpp
constexpr int kQGroup = 16;     // 4x4 queries
constexpr int kHGroup = 8;      // 8 heads
constexpr int kTileSize = 16 * 17;  // 16x16 tile, pitch=17

const int head_groups = (num_heads + 7) / 8;
dim3 grid((num_query + 15) / 16, channels, batch_size * head_groups);
dim3 block(128, 1, 1);  // 16 queries × 8 heads

__shared__ float sh_tile[kHGroup][kTileSize];
```

---

## 4. 四模块实现（真实代码落点）

以下 4 个模块均在内核路径中出现，并与硬件加速器接口一致。

### 4.1 PCB（Pre‑Check Block）
文件：`deAttn_optim/src/deform_attn/cuda/ms_deform_attn_im2col_cuda.cuh`

```cpp
// PCB: 权重筛选
const float threshold = 0.0f;
__deform_pcb(s_weights, threshold, s_mask, num_point);
```

### 4.2 TBC（Tile Boundary Check）
文件：`deAttn_optim/src/deform_attn/cuda/ms_deform_attn_im2col_cuda.cuh`

```cpp
// TBC: 计算 tile 边界与访问模式
mode = __deform_tbc(s_coords, s_mask, num_point,
                    &base_x, &base_y, &tile_w, &tile_h);
```

### 4.3 TMA + Storage（含 swizzle / pitch=17）
文件：`deAttn_optim/src/deform_attn/cuda/ms_deform_attn_im2col_cuda.cuh`

```cpp
// TMA: 加载 tile 到共享存储（pitch=17）
if (mode != 3) {
  const scalar_t* tile_src = data_value_ptr +
      (base_y * spatial_w + base_x) * qid_stride +
      m_col * channels + c_col;
  __deform_tma(tile_src, sh_tile[h_local],
               tile_w, tile_h, spatial_w * qid_stride, mode);
}
```

### 4.4 Interpolation（片上双线性插值）
文件：`deAttn_optim/src/deform_attn/cuda/ms_deform_attn_im2col_cuda.cuh`

```cpp
// Interp: 在 tile 内插值
const float val = __deform_interp(sh_tile[h_local],
                                  local_x, local_y, mode);
col += (scalar_t)val * weight;
```

---

## 5. 模块接口定义（与硬件模型一致）
文件：`deAttn_optim/src/deform_attn/cuda/deform_attn_accelerator.cuh`

```cpp
__device__ void  __deform_pcb(const float* weights, float threshold,
                              bool* mask, int num_points);
__device__ int   __deform_tbc(const float* coords, const bool* mask,
                              int num_points, int* base_x, int* base_y,
                              int* tile_w, int* tile_h);
__device__ void  __deform_tma(const float* src, float* dst,
                              int tile_w, int tile_h, int src_pitch, int mode);
__device__ float __deform_interp(const float* tile_buffer,
                                 float local_x, float local_y, int mode);
```

---

## 6. 模拟器内部实现（四模块功能模型）

以下展示 GPGPU‑Sim 内部对四模块的实现位置与关键逻辑片段。  
文件：`src/cuda-sim/instructions.cc`

### 6.1 PCB（权重筛选）
```cpp
// deform_pcb_impl(...)
for (int i = 0; i < num_points; i++) {
  float weight;
  weights_mem->read(weights_ptr + i * sizeof(float), sizeof(float), &weight);
  bool valid = (fabsf(weight) >= threshold);
  uint8_t mask_bit = valid ? 0 : 1;  // 1=invalid
  mask_mem->write(mask_ptr + i, sizeof(uint8_t), &mask_bit, thread, pI);
  if (valid) valid_count++;
}
```

### 6.2 TBC（边界/模式决策）
```cpp
// deform_tbc_impl(...)
float min_x = 1e9f, max_x = -1e9f;
float min_y = 1e9f, max_y = -1e9f;
for (int i = 0; i < num_points; i++) {
  uint8_t mask_bit;
  mask_mem->read(mask_ptr + i, sizeof(uint8_t), &mask_bit);
  if (mask_bit == 0) {
    float x, y;
    coords_mem->read(coords_ptr + i * 2 * sizeof(float), sizeof(float), &x);
    coords_mem->read(coords_ptr + (i * 2 + 1) * sizeof(float), &y);
    min_x = fminf(min_x, x); max_x = fmaxf(max_x, x);
    min_y = fminf(min_y, y); max_y = fmaxf(max_y, y);
  }
}
int tile_w = (int)ceilf(max_x) - base_x + 1;
int tile_h = (int)ceilf(max_y) - base_y + 1;
int mode = (tile_w > 16 || tile_h > 16) ? 3 : 0;
```

### 6.3 TMA + Storage（tile 加载 + pitch=17）
```cpp
// deform_tma_impl(...)
const int dst_pitch = 17;
for (int row = 0; row < tile_h; row++) {
  for (int col = 0; col < tile_w; col++) {
    addr_t src_addr = src_ptr + (row * src_pitch + col) * sizeof(float);
    addr_t dst_addr = dst_ptr + (row * dst_pitch + col) * sizeof(float);
    float value;
    src_mem->read(src_addr, sizeof(float), &value);
    dst_mem->write(dst_addr, sizeof(float), &value, thread, pI);
  }
}
```

### 6.4 Interpolation（片上双线性插值）
```cpp
// deform_interp_impl(...)
const int tile_pitch = 17;
addr_t addr00 = tile_buffer_ptr + (y0 * tile_pitch + x0) * sizeof(float);
tile_mem->read(addr00, sizeof(float), &v00);
// ...
float result = (1.0f - fx) * (1.0f - fy) * v00 + fx * (1.0f - fy) * v01 +
               (1.0f - fx) * fy * v10 + fx * fy * v11;
inst.m_deform_interp_result = result;
```

---

## 7. 模块级延迟与复用模型（结构一致的估算）

表 1 给出模块默认延迟（与配置项一致），用于说明流水线各阶段的占比与均摊关系：

| 模块 | 延迟 (cycles) | 说明 |
|---|---:|---|
| PCB | 1 | 权重筛选 |
| TBC Phase1 | 4 | 初步聚合判断 |
| TBC Phase2 | 3 | 模式/包围盒决策 |
| TMA | 16 | Tile 批量加载 |
| Storage | 6 | 无冲突存储（pitch=17） |
| Interp | 3 | 片上双线性插值 |

**复用模型**  
同一 level 内，**每个 head 的 TMA/Storage 只执行一次**，在 Query Group 内复用：  

```
T_level = PCB + TBC + (TMA+Storage)/Reuse + Interp
Reuse ≈ Q_group(16) × H_group(8)
```

---

## 8. 调用链（文本流程图）

```
test.cu
  └─ ms_deform_attn_cuda_forward(...)
      └─ ms_deformable_im2col_cuda(...)
          └─ ms_deformable_im2col_gpu_kernel_grouped<<<grid, block>>>()
              ├─ __deform_pcb(...)
              ├─ __deform_tbc(...)
              ├─ __deform_tma(...)
              └─ __deform_interp(...)
```

该路径与硬件加速模块一一对应，体现 Query/Head 组内复用。

---

## 9. 实验结果（日志摘录）

来自 `deAttn_base/out/test_RTX3070.txt` 与 `deAttn_optim/out/test_RTX3070.txt`：

| 指标 | Baseline | Optimized |
|---|---:|---:|
| SO 预测时间 | 16.201 μs | 16.201 μs |
| Attn 预测时间 | 11.081 μs | 11.081 μs |
| 采样聚合 Kernel 时间 | 19.091 μs | **14.313 μs** |
| 总时间 | 46.373 μs | **41.595 μs** |
| 计算性能 | 549.25 GFLOPS | **732.62 GFLOPS** |

**阶段性加速比**（采样聚合）：  
19.091 μs → 14.313 μs，约 **25.0%** 时间下降（≈1.33×）。

---

## 10. 性能收益来源（结构性解释）

1) **TMA/Storage 代价均摊**  
同一 level 内，每个 head 只加载一次 tile，并在 16 个 query 内复用。  
→ 访存开销在 Query/Head 组内均摊。

2) **片上插值减少全局访存**  
Interpolation 在 tile 内完成，避免每个点 4 次全局读取。

---

## 11. 测试流程

```bash
# Baseline
cd deAttn_base && make && bash run.sh

# Optimized
cd deAttn_optim && make && bash run.sh
```

---

## 12. 结论

本实现完整落地了 PCB / TBC / TMA+Storage / Interpolation 四模块，  
并采用 Query/Head 组内复用结构，使 tile 加载与插值成本在组内均摊。  
与 baseline 相比，采样聚合阶段显著降低访存开销并获得稳定性能提升。  
该实现结构与硬件加速器设计一致，具备可复用与可扩展性。
