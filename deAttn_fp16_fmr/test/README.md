# FMR Bilinear Sampling Test

## 目录结构

```
test/
├── test_fmr_basic.cu         # FMR测试程序源码
├── gpgpusim.config           # GPGPU-Sim配置文件
├── config_fermi_islip.icnt   # 互连网络配置
├── Makefile                  # 编译配置
├── run.sh                    # 运行脚本
└── README.md                 # 本文件
```

## 测试说明

### 测试内容

这个测试程序验证FMR（Feature Map Reorganizer）指令的基本功能：

1. **初始化特征图**: 在共享内存中创建 8x8 的特征图
2. **设置采样点**: 定义4个浮点坐标采样点
3. **调用FMR指令**: 使用内联PTX调用 `ld.sample.fmr.f16`
4. **验证结果**: 与CPU上的双线性插值参考实现对比

### FMR指令格式

```ptx
ld.sample.fmr.f16 %result, [%base_addr], %x, %y, %stride;
```

**操作数说明**:
- `%result`: f16类型的采样结果
- `[%base_addr]`: 特征图基地址（共享内存）
- `%x`: 采样x坐标（f32）
- `%y`: 采样y坐标（f32）
- `%stride`: 特征图单行元素数量（s32）

### 测试用例

| Sample | Coordinate | Expected Behavior |
|--------|-----------|-------------------|
| 0 | (1.5, 1.5) | 正中心插值 |
| 1 | (3.2, 2.8) | 一般情况插值 |
| 2 | (2.0, 3.0) | 整数坐标 |
| 3 | (4.7, 1.3) | 接近边界插值 |

## 快速开始

### 方法1：使用run.sh（推荐）

```bash
cd /home/koala/gpgpu-sim_distribution/deAttn_fp16_fmr/test
./run.sh
```

run.sh会自动：
1. ✅ 检查GPGPU-Sim环境
2. ✅ 验证配置文件
3. ✅ 编译测试程序
4. ✅ 运行模拟
5. ✅ 显示结果和统计

### 方法2：手动运行

```bash
# 1. 设置环境
cd /home/koala/gpgpu-sim_distribution
source setup_environment

# 2. 进入测试目录
cd deAttn_fp16_fmr/test

# 3. 编译
make clean
make all

# 4. 运行
./test_fmr_basic
```

## 预期输出

### 成功运行的输出示例

```
=== FMR Bilinear Sampling Test ===

Launching kernel: 1 blocks x 32 threads

=== Results ===
Sample   Coord    FMR Result   Expected     Diff         Status
---------------------------------------------------------------
0        (1.5,1.5) 15.00        15.00        0.0000       PASS
1        (3.2,2.8) 28.20        28.20        0.0000       PASS
2        (2.0,3.0) 32.00        32.00        0.0000       PASS
3        (4.7,1.3) 18.21        18.21        0.0000       PASS
---------------------------------------------------------------
Passed: 4/4, Max Error: 0.0000

✓ All tests PASSED!
```

### GPGPU-Sim统计输出

在 `gpgpu_inst_stats.txt` 中应该能看到：
```
ld.sample.fmr: 4    # FMR指令执行次数
```

## 配置说明

### GPGPU-Sim配置 (gpgpusim.config)

关键的FMR相关配置：

```bash
# FMR使能和单元配置
-gpgpu_fmr_avail 1              # 启用FMR
-gpgpu_num_fmr_units 1          # 1个FMR单元
-gpgpu_fmr_latency 4            # 4周期延迟

# FMR操作数收集器端口
-gpgpu_operand_collector_num_in_ports_fmr 1
-gpgpu_operand_collector_num_out_ports_fmr 1

# 流水线宽度（15个阶段，包括FMR）
-gpgpu_pipeline_widths 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
```

### 调整配置

如果要测试不同的FMR配置：

```bash
# 增加FMR单元数量
-gpgpu_num_fmr_units 2

# 调整延迟
-gpgpu_fmr_latency 2

# 增加带宽
-gpgpu_operand_collector_num_in_ports_fmr 2
```

## 调试

### 启用详细日志

编辑 `gpgpusim.config`，添加：

```bash
-gpgpu_ptx_verbose 1                    # PTX详细日志
-gpgpu_ptx_instruction_classification 1 # 指令分类信息
```

### 查看PTX代码

```bash
ls _app_cuda_version_*.ptx
```

### 检查生成的指令

```bash
grep -i "ld.sample.fmr" _app_cuda_version_*.ptx
```

## 故障排查

### 问题1: 找不到ld.sample.fmr指令

**症状**: PTX文件中没有FMR指令

**原因**: NVCC可能优化掉了内联汇编

**解决**: 
```bash
# 使用-G禁用优化
make clean
make debug
./test_fmr_basic
```

### 问题2: 编译错误

**症状**: `nvcc: command not found`

**解决**:
```bash
export PATH=/usr/local/cuda-11.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH
```

### 问题3: GPGPU-Sim环境未设置

**症状**: `libcudart.so` 版本不匹配

**解决**:
```bash
cd /home/koala/gpgpu-sim_distribution
source setup_environment
```

### 问题4: 运行时错误

**检查**:
1. 查看 `simulation.log`
2. 查看 `gpgpusim_power_report__*.log`
3. 检查PTX解析是否正确

## 文件说明

### 输出文件

运行后会生成以下文件：

- `simulation.log` - 完整的模拟输出
- `build.log` - 编译日志
- `gpgpu_inst_stats.txt` - 指令统计
- `gpgpusim_power_report__*.log` - 功耗报告
- `_app_cuda_version_*.ptx` - 生成的PTX代码
- `_cuobjdump_complete_output_*` - CUDA对象转储

### 清理

```bash
make clean          # 清理编译生成的文件
rm -f *.log *.txt   # 清理日志文件
```

## 进阶测试

### 测试更多采样点

修改 `test_fmr_basic.cu` 中的 `NUM_SAMPLES`:

```cpp
#define NUM_SAMPLES 100  // 增加到100个采样点
```

### 测试更大的特征图

```cpp
#define WIDTH 64
#define HEIGHT 64
```

**注意**: 共享内存大小限制为 49152 字节 (24K half)

### 性能测试

创建 `test_fmr_performance.cu` 进行性能对比：
- Baseline: 使用多条ld指令手动实现
- FMR: 使用单条ld.sample.fmr指令

比较两者的：
- 指令数量
- 执行周期
- 能耗

## 参考文档

- `../FMR_IMPLEMENTATION_PLAN_V2.md` - FMR实施计划
- `../FMR_PHASE1_COMPLETION_REPORT.md` - 阶段一完成报告
- `../FMR_CODE_LOCATION_GUIDE.md` - 代码位置详细说明
- `../TODO.md` - FMR模块原始需求

## 联系与支持

如有问题，请检查：
1. GPGPU-Sim是否正确编译
2. FMR相关代码是否正确集成
3. 配置文件是否正确设置

---

**测试创建时间**: 2025-11-05  
**GPGPU-Sim版本**: 4.2.0  
**CUDA版本**: 11.3  
**目标架构**: SM_70 (Volta)
