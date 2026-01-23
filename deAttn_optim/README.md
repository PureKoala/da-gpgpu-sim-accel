# Deformable Attention Optimized Implementation

使用硬件加速器优化的Deformable Attention实现。

## 关键特性

### 1. WMMA矩阵计算（与baseline相同）
- **GEMM #1**: 采样偏移预测 `SO = Q × W_SO`
- **GEMM #2**: 注意力权重预测 `A = Q × W_A`
- **Tensor Core**: FP16输入 + FP32累加器

### 2. 硬件加速器优化（新增）
- **PCB**: 权重预筛选，剔除低权重采样点（~30%剪枝）
- **TBC**: Tile边界检查，分析空间聚集性
- **TMA**: Tile内存访问，批量加载16×16 tile
- **Interpolation**: 硬件加速双线性插值

## 与Baseline对比

| 特性 | Baseline | Optimized (本版本) |
|------|----------|-------------------|
| WMMA GEMM | ✅ 使用 | ✅ 使用（相同） |
| 采样方式 | 随机全局访问 | Tile批量访问 |
| 插值计算 | 标准CUDA | 硬件加速 |
| 权重筛选 | 无 | PCB剪枝 |
| 预期加速比 | 1.0x | 4.9x |

## 编译和运行

```bash
# 编译
make clean && make

# 直接运行
./bin/test

# GPGPU-Sim仿真
./run.sh
```

## 硬件加速器调用

在采样聚合kernel中添加了以下函数调用：

```cuda
// 1. 权重预筛选
__deform_pcb(attention_weights, threshold, mask);

// 2. Tile边界检查
int mode = __deform_tbc(sampling_locations, num_points, 
                        &base_x, &base_y);

// 3. Tile内存访问
if (mode != DISCRETE_MODE) {
    __deform_tma(value_map, tile_buffer, 
                 tile_w, tile_h, pitch, mode);
}

// 4. 硬件加速插值
for (int p = 0; p < num_point; ++p) {
    if (!mask[p]) {
        float val = __deform_interp(tile_buffer, 
                                     local_x, local_y, mode);
        output += weight[p] * val;
    }
}
```

## 性能优化原理

### 访存优化
- **Baseline**: 128点 × 4次访存/点 = 512次随机访存
- **Optimized**: 1次Tile加载(16×16) + 128次shared memory访问
- **加速比**: ~4.9x

### 计算优化
- **PCB剪枝**: 减少30%无效计算
- **提升**: ~1.4x

### 总体加速
- **理论加速比**: 4.9x × 1.4x ≈ **6.9x**

## GPGPU-Sim集成

硬件加速器函数会被GPGPU-Sim拦截并模拟：

- `__deform_pcb` → `deform_pcb_impl()` (1 cycle)
- `__deform_tbc` → `deform_tbc_impl()` (7 cycles)
- `__deform_tma` → `deform_tma_impl()` (22 cycles)
- `__deform_interp` → `deform_interp_impl()` (3 cycles)

延迟模型在GPGPU-Sim的`deform_attn_impl.cc`和`deform_attn_unit.cc`中实现。
