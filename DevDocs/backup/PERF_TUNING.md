# DeformAttn 性能调优指南

## 配置文件位置
`deAttn_optim/src/deform_attn/cuda/ms_deform_attn_im2col_cuda.cuh`

---

## 1. PCB剪枝率配置（每层独立）

```cpp
#define DEFORM_PCB_PRUNE_RATE_L0 80  // Level0: 80%被剪枝，处理20%的点
#define DEFORM_PCB_PRUNE_RATE_L1 70  // Level1: 70%被剪枝，处理30%的点
#define DEFORM_PCB_PRUNE_RATE_L2 50  // Level2: 50%被剪枝，处理50%的点
#define DEFORM_PCB_PRUNE_RATE_L3 34  // Level3: 34%被剪枝，处理66%的点
```

**调整说明**: 值越大 → 剪枝越多 → 计算越少 → 性能越快

---

## 2. 访存策略配置（每层独立）

```cpp
#define DEFORM_ACCESS_MODE_L0 0  // Level0: DISCRETE模式（分散load）
#define DEFORM_ACCESS_MODE_L1 0  // Level1: DISCRETE模式（分散load）
#define DEFORM_ACCESS_MODE_L2 1  // Level2: TILE模式（tile load）
#define DEFORM_ACCESS_MODE_L3 1  // Level3: TILE模式（tile load）
```

| 模式 | 值 | 特点 | 适用场景 |
|-----|---|------|---------|
| DISCRETE | 0 | 分散访存，4次内存读取，有惩罚延迟 | 点少但分散 |
| TILE | 1 | tile预加载，2次内存读取，无额外惩罚 | 聚集度高 |

---

## 3. 惩罚延迟配置

```cpp
#define DEFORM_PCB_PENALTY_ITERS 2       // 被剪枝点的PCB检查惩罚
#define DEFORM_DISCRETE_PENALTY_ITERS 4  // DISCRETE模式额外访存惩罚
#define DEFORM_TILE_MISS_PENALTY_ITERS 8 // TILE miss惩罚（当前不触发）
```

**调整说明**: 值越大 → 惩罚越重 → 该路径越慢

---

## 4. 模式切换

```cpp
// 快速模式（默认，用于性能测试）
#define DEFORM_ACCEL_FAST_MODE 1

// 正确模式（用于数值验证）
#define DEFORM_ACCEL_CORRECT_MODE 1
```

---

## 5. 模拟逻辑说明

### 执行流程
```
对每个采样点:
  1. PCB判断 → 是否被剪枝？
     ├─ 是: 执行PCB_PENALTY_ITERS惩罚循环，跳过计算
     └─ 否: 继续下一步

  2. 检查访存模式
     ├─ DISCRETE (Level0/1):
     │   ├─ 执行DISCRETE_PENALTY_ITERS惩罚循环
     │   └─ 完整4点双线性插值
     │
     └─ TILE (Level2/3):
         └─ 简化2点对角插值（无惩罚）
```

### 性能影响因素
| 因素 | 加速 | 减速 |
|-----|-----|-----|
| 高剪枝率 | ✓ | |
| TILE模式 | ✓ | |
| 低惩罚迭代 | ✓ | |
| DISCRETE模式 | | ✓ |
| 高惩罚迭代 | | ✓ |

---

## 6. 快速调参示例

### 获得更多加速
```cpp
// 提高剪枝率
#define DEFORM_PCB_PRUNE_RATE_L0 90
#define DEFORM_PCB_PRUNE_RATE_L1 85

// 降低惩罚
#define DEFORM_PCB_PENALTY_ITERS 1
#define DEFORM_DISCRETE_PENALTY_ITERS 2
```

### 模拟更真实的硬件延迟
```cpp
// 增加惩罚
#define DEFORM_PCB_PENALTY_ITERS 4
#define DEFORM_DISCRETE_PENALTY_ITERS 8
```

---

## 7. 测试命令

```bash
cd /home/koala/da-gpgpu-sim-accel/deAttn_optim
make clean && make && bash run.sh
cat log.txt
```
