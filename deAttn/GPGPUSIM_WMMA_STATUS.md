# GPGPU-Sim WMMA Tensor Core 支持状态

## 当前实现状态

### ✅ 成功部分
1. **WMMA Kernel 实现完成**
   - 使用 `nvcuda::wmma` API 实现 INT8 Tensor Core GEMM
   - 生成正确的 WMMA PTX 指令
   - 编译成功，无错误

2. **生成的 PTX 指令（test.1.sm_86.ptx）**
   ```ptx
   wmma.load.a.sync.aligned.row.m16n16k16.global.s8 {%r63, %r64}, [%rd15], %r32;
   wmma.load.b.sync.aligned.col.m16n16k16.global.s8 {%r66, %r67}, [%rd11], %r33;
   wmma.mma.sync.aligned.row.col.m16n16k16.s32.s8.s8.s32 
       {%r86, %r85, %r84, %r83, %r82, %r81, %r80, %r79},
       {%r63, %r64}, 
       {%r66, %r67}, 
       {%r86, %r85, %r84, %r83, %r82, %r81, %r80, %r79};
   wmma.store.d.sync.aligned.row.m16n16k16.global.s32 [%rd14], {...}, %r33;
   ```

3. **PTX 解析成功**
   - GPGPU-Sim 成功解析 PTX 文件
   - 识别出 kernel 函数：
     * `_Z29predict_so_int8_ampere_kernelILi16ELi16ELi16EEvPKaS1_Piiii`
     * `_Z31predict_attn_int8_ampere_kernelILi16ELi16ELi16EEvPKaS1_Piiii`

### ❌ GPGPU-Sim 运行时错误

**错误信息：**
```
test: ./ptx_ir.h:1055: int ptx_instruction::get_type2() const: 
Assertion `m_scalar_type.size() == 2' failed.
Aborted (core dumped)
```

**根本原因：**
- GPGPU-Sim 的 PTX 指令表示 (`ptx_ir.h`) 假设指令最多有 2 个类型参数
- INT8 WMMA 指令有 **4 个类型参数**：`m16n16k16.s32.s8.s8.s32`
  * `s32` - 累加器类型 (output/accumulator)
  * `s8` - 矩阵 A 输入类型
  * `s8` - 矩阵 B 输入类型  
  * `s32` - 矩阵 C 输入类型 (accumulator input)

**代码位置：** `/home/koala/gpgpu-sim_distribution/src/cuda-sim/ptx_ir.h:1055`
```cpp
int get_type2() const {
    assert(m_scalar_type.size() == 2);  // ← 这里断言失败
    return m_scalar_type.back();
}
```

## 技术分析

### WMMA 指令格式对比

| WMMA Type | Type Parameters | GPGPU-Sim Support |
|-----------|----------------|-------------------|
| FP16 (Volta/Turing) | `m16n16k16.f16.f16` (2个类型) | ✅ 支持 |
| INT8 (Ampere) | `m16n16k16.s32.s8.s8.s32` (4个类型) | ❌ 不支持 |

### K-Padding 浪费分析（已在 PTX 中体现）

尽管运行时出错，但生成的 PTX 代码已经清晰展示了 K-Padding 问题：

1. **采样偏移预测 (SO)**
   - 实际需要：K=8
   - WMMA 要求：K=16
   - PTX 中循环迭代：`k += 16`
   - **浪费：50% 的 K 维度计算**

2. **注意力权重预测 (Attn)**
   - 实际需要：K=4
   - WMMA 要求：K=16
   - PTX 中循环迭代：`k += 16`
   - **浪费：75% 的 K 维度计算**

## 解决方案选项

### 选项 1：修改 GPGPU-Sim 源码（推荐用于研究）
**步骤：**
1. 修改 `src/cuda-sim/ptx_ir.h` 和 `src/cuda-sim/ptx_ir.cc`
   - 将 `m_scalar_type` 从最多 2 个扩展到 4 个类型参数
   - 添加 `get_type3()` 和 `get_type4()` 方法
   
2. 修改 `src/cuda-sim/ptx_parser.y` 
   - 更新 WMMA 指令语法规则以支持 4 个类型参数

3. 修改 `src/cuda-sim/instructions.cc`
   - 更新 WMMA 指令执行逻辑

**优点：** 
- 真正支持 INT8 Tensor Core 仿真
- 可以获得准确的性能数据
- 适合发表学术论文

**缺点：** 
- 需要深入理解 GPGPU-Sim 内部机制
- 修改量较大，可能引入新 bug

### 选项 2：使用 FP16 WMMA 代替（快速验证）
将 INT8 WMMA 改为 FP16 WMMA（2 个类型参数），GPGPU-Sim 已支持。

**修改：**
```cpp
// 使用 FP16 而不是 INT8
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
```

**优点：** 
- 可以立即运行
- 仍然能展示 K-Padding 问题
- GPGPU-Sim 完全支持

**缺点：** 
- 与论文声称的 INT8 Tensor Core 不符
- 性能特征与 INT8 不同

### 选项 3：PTX 层面拦截分析（当前可行）
不运行仿真，只分析生成的 PTX 代码：

**可以提取的数据：**
1. WMMA 指令数量和类型
2. K 维度迭代次数（padding 证据）
3. 内存访问模式
4. 寄存器使用情况

**分析工具：**
```bash
# 统计 WMMA 指令
grep "wmma\." test.1.sm_86.ptx | wc -l

# 查看 K 维度循环
grep -A 5 "k_tile.*WMMA_K" test.1.sm_86.ptx
```

## 当前推荐方案

### 用于论文的策略

**方案 A：理论分析 + PTX 证据**
1. 在论文中使用当前生成的 PTX 代码作为证据
2. 标注出 K-Padding 位置（代码注释已经很清晰）
3. 理论计算浪费比例：50% (SO) 和 75% (Attn)
4. 声明："由于 GPGPU-Sim 对 Ampere INT8 WMMA 支持限制，我们通过 PTX 代码分析验证"

**方案 B：修改为 FP16 并运行仿真**
1. 快速改为 FP16 WMMA
2. 运行完整仿真，获取性能数据
3. 在论文中说明："使用 FP16 验证概念，INT8 具有相同的 K-Padding 问题"

**方案 C：混合方法（最完整）**
1. 保留当前 INT8 代码和生成的 PTX
2. 另外创建 FP16 版本用于实际仿真
3. 论文中展示两者：PTX 证明 INT8 可行性，FP16 提供性能数据

## 代码资产总结

### 已完成的实现（无需修改）
- ✅ `ms_deform_attn_predict_cuda.cuh` - WMMA kernel 实现
- ✅ `ms_deform_attn_predict_wrapper.cu` - API 包装
- ✅ `test.cu` - 完整的端到端测试
- ✅ 生成的 PTX 文件 - 包含真实的 WMMA 指令

### 论文中可以使用的内容
1. **PTX 代码片段** - 展示真实的 WMMA 指令
2. **Kernel 实现** - 展示如何使用 WMMA API
3. **K-Padding 注释** - 代码中的 `**CRITICAL INTERCEPTION POINT**` 标记
4. **理论分析** - 50% 和 75% 浪费计算

## 下一步行动

根据研究目标和时间限制，建议：

**如果时间充裕（2-3天）：**
→ 实施选项 1，修改 GPGPU-Sim 以支持 INT8 WMMA

**如果需要快速结果（1天内）：**
→ 实施选项 2，改为 FP16 WMMA 并运行仿真

**如果只需要概念验证：**
→ 使用当前 PTX 代码和理论分析（已完成）

---
**生成时间：** 2025-10-31  
**GPGPU-Sim 版本：** 4.2.0  
**CUDA 版本：** 11.0  
**GPU 配置：** RTX 3070 (SM86)
