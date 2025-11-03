# FMR集成 - 快速检查清单

## ✅ 修改完成状态

### 核心修改 (100%完成)

- [x] **abstract_hardware_model.h** (1处)
  - [x] 添加 `FMR_SAMPLE_OP` 到 `enum uarch_op_t`

- [x] **shader.h** (15+处)
  - [x] 添加 `FMR = 7` 到 `exec_unit_type_t`
  - [x] 定义 `fmr_unit` 类
  - [x] 添加 `ID_OC_FMR`, `OC_EX_FMR` 流水线阶段
  - [x] 更新 `pipeline_stage_name_decode` 数组
  - [x] 添加5个FMR配置参数
  - [x] 更新所有7个调度器类构造函数签名

- [x] **shader.cc** (10+处)
  - [x] 实现 `fmr_unit` 构造函数
  - [x] 实现 `fmr_unit::issue()`
  - [x] 实现 `fmr_unit::active_lanes_in_pipeline()`
  - [x] 在 `create_exec_pipeline()` 中添加FMR_CUS枚举
  - [x] 注册FMR操作数收集器
  - [x] 创建FMR功能单元实例
  - [x] 更新功能单元计数
  - [x] 在 `scheduler_unit::cycle()` 中添加FMR调度逻辑
  - [x] 更新所有6个调度器实例化（添加fmr_out参数）
  - [x] 实现 `swl_scheduler` 构造函数

- [x] **gpu-sim.cc** (6处)
  - [x] 注册 `-gpgpu_fmr_avail`
  - [x] 注册 `-gpgpu_num_fmr_units`
  - [x] 注册 `-gpgpu_fmr_latency`
  - [x] 注册 `-gpgpu_operand_collector_num_in_ports_fmr`
  - [x] 注册 `-gpgpu_operand_collector_num_out_ports_fmr`
  - [x] 更新 `pipeline_widths` 说明

---

## 🔍 关键代码行号快速定位

| 文件 | 行号 | 修改内容 |
|-----|------|---------|
| `abstract_hardware_model.h` | ~138 | `FMR_SAMPLE_OP` |
| `shader.h` | ~85 | `exec_unit_type_t::FMR = 7` |
| `shader.h` | 1259-1274 | `fmr_unit` 类定义 |
| `shader.h` | 1499-1520 | FMR流水线阶段 |
| `shader.h` | 1697-1708 | FMR配置参数 |
| `shader.h` | 376-643 | 调度器构造函数更新 |
| `shader.cc` | 2400-2429 | `fmr_unit` 实现 |
| `shader.cc` | 272-450 | `create_exec_pipeline()` 集成 |
| `shader.cc` | 1490-1508 | FMR调度逻辑 |
| `shader.cc` | 207-251 | 调度器实例化 |
| `shader.cc` | 1720-1733 | `swl_scheduler` 构造 |
| `gpu-sim.cc` | 602-607 | `pipeline_widths` 更新 |
| `gpu-sim.cc` | 610-629 | FMR配置注册 |

---

## 🧪 预编译检查

### 1. 头文件包含检查
```bash
grep -n "FMR_SAMPLE_OP" src/abstract_hardware_model.h
# 应该看到: 138:  FMR_SAMPLE_OP
```

### 2. FMR单元定义检查
```bash
grep -n "class fmr_unit" src/gpgpu-sim/shader.h
# 应该看到: 1260:class fmr_unit : public pipelined_simd_unit
```

### 3. 流水线阶段检查
```bash
grep -n "ID_OC_FMR" src/gpgpu-sim/shader.h
# 应该看到多处匹配
```

### 4. 配置参数检查
```bash
grep -n "gpgpu_fmr_avail" src/gpgpu-sim/gpu-sim.cc
# 应该看到配置注册代码
```

---

## 📋 编译前准备

### 确保环境变量设置
```bash
cd /home/koala/gpgpu-sim_distribution
source setup_environment
echo $GPGPUSIM_ROOT  # 应该输出当前目录
```

### 清理旧构建
```bash
make clean
rm -rf build/gcc-*/cuda-*  # 清理所有构建产物
```

---

## 🔨 编译命令

### 完整编译
```bash
make -j8 2>&1 | tee fmr_build.log
```

### 检查编译结果
```bash
# 检查是否有错误
grep -i "error:" fmr_build.log

# 检查警告（可忽略某些警告）
grep -i "warning:" fmr_build.log | grep -v "unused"
```

---

## ⚠️ 常见编译问题

### 问题1: N_PIPELINE_STAGES计数错误
**症状**: `array index out of bounds`  
**原因**: 流水线阶段数组长度未更新  
**解决**: N_PIPELINE_STAGES = 15 (原13 + 2个FMR阶段)

### 问题2: m_fmr_out未声明
**症状**: `'m_fmr_out' was not declared in this scope`  
**原因**: 某个调度器类未添加fmr_out成员  
**解决**: 检查scheduler_unit基类中是否有 `register_set *m_fmr_out;`

### 问题3: FMR相关配置变量未定义
**症状**: `'gpgpu_fmr_avail' was not declared`  
**原因**: shader_core_config类中缺少配置成员  
**解决**: 确认shader.h:1697-1708行的配置定义

---

## ✅ 编译成功标志

编译成功后应该看到：
```
Linking C++ executable ../../lib/gcc-X.X.X/cuda-XXXXX/release/libcudart.so
Built target cudart
...
[100%] Built target gpgpu-sim
```

---

## 🚀 快速测试

### 最小测试配置
```bash
cd configs/tested-cfgs/SM7_QV100
cp gpgpusim.config gpgpusim.config.bak

# 添加到配置文件末尾
echo "" >> gpgpusim.config
echo "# FMR Configuration" >> gpgpusim.config
echo "-gpgpu_fmr_avail 1" >> gpgpusim.config
echo "-gpgpu_num_fmr_units 1" >> gpgpusim.config
echo "-gpgpu_fmr_latency 4" >> gpgpusim.config
```

### 运行测试
```bash
cd /home/koala/gpgpu-sim_distribution/deAttn_fp16_fmr
./test
```

---

## 📊 验证指标

### 成功指标
- ✅ 编译无错误
- ✅ 链接成功
- ✅ 测试程序可运行
- ✅ 可以识别FMR配置参数
- ✅ 统计信息中出现FMR相关数据

### 性能指标（预期）
- 📉 总执行周期减少
- 📉 SP/INT单元利用率下降
- 📈 FMR单元利用率提升
- 📉 采样操作延迟减少 ~40%

---

## 📚 文档索引

- **FMR_MODIFICATION_PLAN.md** - 详细修改计划
- **FMR_IMPLEMENTATION_REPORT.md** - 实施过程记录
- **FMR_FINAL_REPORT.md** - 完整完成报告
- **README_FMR.md** - 快速入门
- **本文档** - 检查清单

---

**最后更新**: 2025-11-03  
**状态**: ✅ 所有核心修改完成  
**下一步**: 编译验证 → 功能测试 → 性能评估
