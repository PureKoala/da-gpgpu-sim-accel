# FMR实施计划 V2.0 - 详细实施路线图

## 文档版本信息
- **版本**: V2.0
- **日期**: 2025-11-05
- **状态**: 已完成微架构框架，待实现PTX前端和功能逻辑

---

## 当前状态评估

### ✅ 已完成：微架构框架（后端）
1. **指令操作码定义** (`abstract_hardware_model.h`)
   - 已添加 `FMR_SAMPLE_OP` 到 `uarch_op_t` 枚举
   
2. **执行单元类型** (`shader.h`)
   - 已添加 `exec_unit_type_t::FMR = 7`
   
3. **FMR硬件单元** (`shader.h` + `shader.cc`)
   - 已定义 `fmr_unit` 类（继承自 `pipelined_simd_unit`）
   - 已实现基础方法：构造函数、`issue()`、`can_issue()`、`active_lanes_in_pipeline()`
   
4. **流水线阶段** (`shader.h`)
   - 已添加 `ID_OC_FMR` 和 `OC_EX_FMR` 流水线阶段
   
5. **调度器集成** (`shader.h` + `shader.cc`)
   - 已更新所有7个调度器类的构造函数签名
   - 已在 `scheduler_unit::cycle()` 中添加FMR调度逻辑
   
6. **配置系统** (`gpu-sim.cc`)
   - 已注册5个FMR配置参数

### ⏳ 待实现：前端和功能逻辑

---

## 核心设计决策

### 指令格式设计（基于用户需求）

根据您的要求，新的PTX指令应包含以下操作数：

```ptx
ld.sample.fmr.f16 %dst, [%base_addr], %coord_x, %coord_y, %stride;
```

**操作数说明：**
1. `%dst` - 目标寄存器（f16），存储采样结果
2. `[%base_addr]` - 特征图基地址（共享内存地址）
3. `%coord_x` - 采样点x坐标（f32）
4. `%coord_y` - 采样点y坐标（f32）
5. `%stride` - 特征图单行元素数量（s32），用于计算下一行地址

**设计理由：**
- **stride参数**：使FMR单元能够正确计算4个采样点的地址
  - `top_left = base_addr + floor(y) * stride + floor(x)`
  - `top_right = base_addr + floor(y) * stride + ceil(x)`
  - `bottom_left = base_addr + ceil(y) * stride + floor(x)`
  - `bottom_right = base_addr + ceil(y) * stride + ceil(x)`

### 功能分工（基于用户意见）

您建议插值计算使用CUDA Core完成，因此FMR单元职责调整为：

**FMR单元负责（硬件加速）：**
1. ✅ 地址生成（4个采样点地址）
2. ✅ 4-Bank并行访存（从共享内存读取4个值）
3. ✅ 插值权重计算（α, β, γ, λ）

**CUDA Core负责（软件计算）：**
1. ⏳ 双线性插值计算（4个乘法 + 3个加法）
2. ⏳ 使用FMR返回的4个值和4个权重

**指令执行流程（修订版）：**
```
ld.sample.fmr.f16指令
    ↓
FMR_Unit接收
    ↓
[周期1-2] 地址生成器(AG)
    - 计算floor(x), floor(y), ceil(x), ceil(y)
    - 计算4个地址偏移量
    - 计算4个插值权重
    ↓
[周期3] 4-Bank SMEM并行访问
    - 发出4个并行读请求（无Bank冲突）
    - 读取4个f16值
    ↓
[周期4] 结果打包
    - 将4个值 + 4个权重打包返回
    ↓
写回到目标寄存器（可能是多个寄存器）
    ↓
后续CUDA指令完成插值计算
    - 4 × fma.f16 或 mul.f16 + add.f16
```

---

## 三阶段实施计划

---

## 阶段一：PTX前端集成（指令识别与解析）

### 目标
使GPGPU-Sim能够识别和解析新的 `ld.sample.fmr` PTX指令，并将其映射到 `FMR_SAMPLE_OP`。

---

### 任务 1.1：定义PTX指令Token（词法分析）

**文件**: `src/cuda-sim/ptx.l`

**修改内容**:
```lex
/* 在现有的内存指令token定义区域添加 */
"ld.sample.fmr"     { return TOKEN_LD_SAMPLE_FMR; }
```

**说明**:
- 这让词法分析器能识别 `ld.sample.fmr` 字符串
- 返回一个新的token类型 `TOKEN_LD_SAMPLE_FMR`

---

### 任务 1.2：添加Token定义（语法分析器）

**文件**: `src/cuda-sim/ptx.y`

**修改位置**: 在token声明区域
```yacc
/* Token declarations */
%token TOKEN_LD_SAMPLE_FMR
```

---

### 任务 1.3：定义语法规则（语法分析）

**文件**: `src/cuda-sim/ptx.y`

**修改内容**: 在指令语法规则区域添加
```yacc
instruction:
    /* ...existing rules... */
    | TOKEN_LD_SAMPLE_FMR operand_list {
        // 创建FMR采样指令
        const operand_info &dst = $2[0];        // 目标寄存器
        const operand_info &base = $2[1];       // 基地址
        const operand_info &coord_x = $2[2];    // x坐标
        const operand_info &coord_y = $2[3];    // y坐标
        const operand_info &stride = $2[4];     // stride
        
        // 创建指令对象
        ptx_instruction *inst = new ptx_instruction();
        inst->set_opcode(FMR_SAMPLE_OP);        // 关键：设置操作码
        inst->set_dst(dst);
        inst->add_src(base);
        inst->add_src(coord_x);
        inst->add_src(coord_y);
        inst->add_src(stride);
        
        $$ = inst;
    }
    ;
```

**说明**:
- 解析5个操作数（1个目标 + 4个源）
- **关键步骤**：`set_opcode(FMR_SAMPLE_OP)` 将指令映射到我们定义的微架构操作码
- 这是连接前端（PTX文本）和后端（微架构执行）的桥梁

---

### 任务 1.4：验证PTX解析

**测试代码**: 创建简单的PTX测试文件
```ptx
.version 7.0
.target sm_70
.address_size 64

.visible .entry test_fmr_parse()
{
    .reg .f16 %result;
    .reg .u64 %base;
    .reg .f32 %x, %y;
    .reg .s32 %stride;
    
    mov.u64 %base, 0;
    mov.f32 %x, 1.5;
    mov.f32 %y, 2.5;
    mov.s32 %stride, 64;
    
    // 新指令
    ld.sample.fmr.f16 %result, [%base], %x, %y, %stride;
    
    ret;
}
```

**验证方法**:
1. 编译修改后的GPGPU-Sim
2. 使用 `-gpgpu_ptx_verbose 1` 运行
3. 检查日志输出，确认指令被正确解析为 `FMR_SAMPLE_OP`

---

## 阶段二：FMR功能逻辑实现（功能仿真）

### 目标
实现FMR单元的功能行为，模拟地址生成、4-Bank访存和权重计算。

---

### 任务 2.1：扩展PTX指令类

**文件**: `src/cuda-sim/ptx_ir.h`

**修改内容**: 在 `ptx_instruction` 类中添加FMR特定字段
```cpp
class ptx_instruction {
public:
    // ...existing members...
    
    // FMR specific data
    struct fmr_sampling_data {
        addr_t base_address;      // 特征图基地址
        float coord_x;            // x坐标
        float coord_y;            // y坐标
        int stride;               // 单行元素数量
        
        // 计算得到的4个采样点地址
        addr_t addr_top_left;
        addr_t addr_top_right;
        addr_t addr_bottom_left;
        addr_t addr_bottom_right;
        
        // 计算得到的4个插值权重
        float weight_alpha;   // (1-dx) * (1-dy)
        float weight_beta;    // dx * (1-dy)
        float weight_gamma;   // (1-dx) * dy
        float weight_lambda;  // dx * dy
    } fmr_data;
    
    bool has_fmr_data() const { return m_opcode == FMR_SAMPLE_OP; }
};
```

---

### 任务 2.2：实现FMR功能仿真核心

**文件**: `src/cuda-sim/ptx_inst.cc`

**修改内容**: 在 `ptx_thread_info::execute()` 中添加FMR case
```cpp
void ptx_thread_info::execute(const ptx_instruction &inst, 
                               ptx_thread_info *thread,
                               unsigned warp_id) {
    switch (inst.get_opcode()) {
    // ...existing cases...
    
    case FMR_SAMPLE_OP: {
        // ============ FMR功能仿真 ============
        
        // 步骤1: 读取操作数
        addr_t base_addr = thread->get_operand_value(inst.get_src(0)).u64;
        float coord_x = thread->get_operand_value(inst.get_src(1)).f32;
        float coord_y = thread->get_operand_value(inst.get_src(2)).f32;
        int stride = thread->get_operand_value(inst.get_src(3)).s32;
        
        // 步骤2: 地址生成（模拟AG硬件）
        int x0 = (int)floor(coord_x);
        int y0 = (int)floor(coord_y);
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        
        // 计算4个采样点地址（假设f16，每个元素2字节）
        addr_t addr_tl = base_addr + (y0 * stride + x0) * 2;  // top-left
        addr_t addr_tr = base_addr + (y0 * stride + x1) * 2;  // top-right
        addr_t addr_bl = base_addr + (y1 * stride + x0) * 2;  // bottom-left
        addr_t addr_br = base_addr + (y1 * stride + x1) * 2;  // bottom-right
        
        // 步骤3: 计算插值权重
        float dx = coord_x - x0;  // 小数部分
        float dy = coord_y - y0;
        float weight_alpha = (1.0f - dx) * (1.0f - dy);
        float weight_beta = dx * (1.0f - dy);
        float weight_gamma = (1.0f - dx) * dy;
        float weight_lambda = dx * dy;
        
        // 步骤4: 并行访存（模拟4-Bank SMEM）
        // 注意：这里我们调用共享内存读取函数
        half val_tl, val_tr, val_bl, val_br;
        
        thread->read_shared_memory(addr_tl, 2, &val_tl);
        thread->read_shared_memory(addr_tr, 2, &val_tr);
        thread->read_shared_memory(addr_bl, 2, &val_bl);
        thread->read_shared_memory(addr_br, 2, &val_br);
        
        // 步骤5: 打包结果（由于不在硬件中做插值，我们返回所有数据）
        // 方案A: 使用向量寄存器返回（需要4+4=8个值）
        // 方案B: 立即完成插值（当前实现）
        
        // 这里先实现方案B，稍后可以改为方案A
        float result = weight_alpha * __half2float(val_tl) +
                      weight_beta * __half2float(val_tr) +
                      weight_gamma * __half2float(val_bl) +
                      weight_lambda * __half2float(val_br);
        
        // 步骤6: 写回结果
        ptx_reg_t dst_data;
        dst_data.f16 = __float2half(result);
        thread->set_operand_value(inst.get_dst(), dst_data);
        
        break;
    }
    
    // ...other cases...
    }
}
```

**说明**:
- 这是FMR的"黄金模型"，确保功能正确性
- 模拟了AG（地址生成器）的行为
- 模拟了4-Bank并行访存
- **临时实现**：在这里完成插值计算（后续可以改为返回8个值）

---

### 任务 2.3：实现共享内存读取接口

**文件**: `src/cuda-sim/ptx_thread_info.h` 和 `.cc`

**修改内容**: 确保存在共享内存读取方法
```cpp
class ptx_thread_info {
public:
    // 读取共享内存
    void read_shared_memory(addr_t addr, size_t size, void *data);
    
    // 写入共享内存（用于未来的st.rfm.shared指令）
    void write_shared_memory(addr_t addr, size_t size, const void *data);
};
```

---

### 任务 2.4：添加FMR统计收集

**文件**: `src/gpgpu-sim/shader.h`

**修改内容**: 在 `shader_core_stats` 类中添加
```cpp
class shader_core_stats {
public:
    // ...existing members...
    
    // FMR statistics
    unsigned long long m_num_fmr_insns;          // FMR指令数
    unsigned long long m_num_fmr_active_lanes;   // FMR活跃线程数
    
    void inc_fmr_insns(unsigned lanes) {
        m_num_fmr_insns++;
        m_num_fmr_active_lanes += lanes;
    }
};
```

**文件**: `src/gpgpu-sim/shader.cc`

**修改内容**: 取消注释 `fmr_unit::issue()` 中的统计代码
```cpp
void fmr_unit::issue(register_set &source_reg) {
    warp_inst_t **ready_reg =
        source_reg.get_ready(m_config->sub_core_model, m_issue_reg_id);

    (*ready_reg)->op_pipe = SPECIALIZED__OP;
    
    // 收集FMR统计
    m_core->get_stats()->inc_fmr_insns((*ready_reg)->active_count());
    
    pipelined_simd_unit::issue(source_reg);
}
```

---

## 阶段三：测试与验证

### 目标
创建端到端的测试用例，验证从PTX解析到功能仿真再到微架构执行的完整流程。

---

### 任务 3.1：创建CUDA测试核函数

**文件**: `deAttn_fp16_fmr/test/test_fmr_single.cu`

**内容**:
```cuda
#include <cuda_fp16.h>
#include <stdio.h>

__global__ void test_fmr_sampling(
    half *feature_map,      // 输入特征图（共享内存）
    half *output,           // 输出采样结果
    float *coords_x,        // 采样x坐标
    float *coords_y,        // 采样y坐标
    int stride,             // 特征图宽度
    int num_samples)        // 采样点数量
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_samples) return;
    
    // 加载特征图到共享内存（简化示例）
    __shared__ half smem_feature[64 * 64];  // 假设64x64特征图
    
    // 简单的加载逻辑（实际应该更复杂）
    int load_idx = threadIdx.x;
    while (load_idx < 64 * 64) {
        smem_feature[load_idx] = feature_map[load_idx];
        load_idx += blockDim.x;
    }
    __syncthreads();
    
    // 获取采样坐标
    float x = coords_x[tid];
    float y = coords_y[tid];
    
    // 调用FMR指令（使用内联PTX）
    half result;
    asm volatile(
        "ld.sample.fmr.f16 %0, [%1], %2, %3, %4;"
        : "=h"(result)                              // 输出：f16寄存器
        : "l"((unsigned long long)smem_feature),    // 输入：基地址
          "f"(x),                                   // 输入：x坐标
          "f"(y),                                   // 输入：y坐标
          "r"(stride)                               // 输入：stride
    );
    
    // 写回结果
    output[tid] = result;
}

int main() {
    // 测试参数
    const int width = 64;
    const int height = 64;
    const int num_samples = 10;
    
    // 分配主机内存
    half *h_feature = new half[width * height];
    half *h_output = new half[num_samples];
    float *h_coords_x = new float[num_samples];
    float *h_coords_y = new float[num_samples];
    
    // 初始化测试数据
    for (int i = 0; i < width * height; i++) {
        h_feature[i] = __float2half((float)(i % 256) / 255.0f);
    }
    
    for (int i = 0; i < num_samples; i++) {
        h_coords_x[i] = 10.5f + i * 2.3f;
        h_coords_y[i] = 20.3f + i * 1.7f;
    }
    
    // 分配设备内存
    half *d_feature, *d_output;
    float *d_coords_x, *d_coords_y;
    cudaMalloc(&d_feature, width * height * sizeof(half));
    cudaMalloc(&d_output, num_samples * sizeof(half));
    cudaMalloc(&d_coords_x, num_samples * sizeof(float));
    cudaMalloc(&d_coords_y, num_samples * sizeof(float));
    
    // 复制数据到设备
    cudaMemcpy(d_feature, h_feature, width * height * sizeof(half), 
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_coords_x, h_coords_x, num_samples * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_coords_y, h_coords_y, num_samples * sizeof(float),
               cudaMemcpyHostToDevice);
    
    // 启动kernel
    int threads = 256;
    int blocks = (num_samples + threads - 1) / threads;
    test_fmr_sampling<<<blocks, threads>>>(
        d_feature, d_output, d_coords_x, d_coords_y, width, num_samples);
    
    // 复制结果回主机
    cudaMemcpy(h_output, d_output, num_samples * sizeof(half),
               cudaMemcpyDeviceToHost);
    
    // 验证结果
    printf("FMR Sampling Test Results:\n");
    for (int i = 0; i < num_samples; i++) {
        printf("Sample %d: coord(%.2f, %.2f) -> %.4f\n",
               i, h_coords_x[i], h_coords_y[i], __half2float(h_output[i]));
    }
    
    // 清理
    delete[] h_feature;
    delete[] h_output;
    delete[] h_coords_x;
    delete[] h_coords_y;
    cudaFree(d_feature);
    cudaFree(d_output);
    cudaFree(d_coords_x);
    cudaFree(d_coords_y);
    
    return 0;
}
```

---

### 任务 3.2：编译和运行测试

**步骤**:

1. **编译GPGPU-Sim**:
```bash
cd /home/koala/gpgpu-sim_distribution
source setup_environment
make -j8
```

2. **编译测试程序**:
```bash
cd deAttn_fp16_fmr/test
nvcc -arch=sm_70 -o test_fmr_single test_fmr_single.cu
```

3. **运行测试**:
```bash
./test_fmr_single
```

4. **检查GPGPU-Sim输出**:
```bash
# 查看FMR指令统计
grep "FMR" gpgpusim_power_report__*.log
grep "fmr" gpgpu_inst_stats.txt
```

---

### 任务 3.3：验证检查点

**验证项目**:

✅ **PTX解析验证**:
- [ ] 日志中出现 `Decoded instruction: FMR_SAMPLE_OP`
- [ ] 没有PTX解析错误

✅ **功能仿真验证**:
- [ ] 输出结果在合理范围内（0.0 - 1.0）
- [ ] 采样结果符合双线性插值预期

✅ **微架构执行验证**:
- [ ] `gpgpu_inst_stats.txt` 中有FMR指令计数
- [ ] FMR流水线阶段有活跃记录
- [ ] 执行周期符合配置的延迟（默认4周期）

✅ **性能统计验证**:
- [ ] `m_num_fmr_insns` 计数正确
- [ ] FMR单元利用率统计正常

---

## 后续优化方向（可选）

### 4.1 返回4值+4权重方案

如果未来想让CUDA Core完成插值计算，可以修改为：

```cpp
// 在ptx_inst.cc的FMR_SAMPLE_OP case中
// 不立即计算插值，而是返回8个值

// 使用向量寄存器（需要2个.v4）
// .reg .v4 .f16 values;    // 4个采样值
// .reg .v4 .f32 weights;   // 4个权重

// 将8个值打包到向量寄存器
// 后续由CUDA指令完成：
// mul.f16 tmp0, values.x, weights.x;
// mul.f16 tmp1, values.y, weights.y;
// mul.f16 tmp2, values.z, weights.z;
// mul.f16 tmp3, values.w, weights.w;
// add.f16 result, tmp0, tmp1;
// add.f16 result, result, tmp2;
// add.f16 result, result, tmp3;
```

### 4.2 4-Bank SMEM重组

**文件**: `src/gpgpu-sim/memory.cc`

实现 `st.rfm.shared` 指令，在写入时自动重组数据到4个Bank。

### 4.3 功耗建模

**文件**: `src/accelwattch/accelwattch_stat_logger.cc`

为FMR单元添加功耗统计。

---

## 实施时间线估算

| 阶段 | 任务 | 预计时间 | 优先级 |
|------|------|----------|--------|
| 阶段一 | 1.1-1.4 PTX前端集成 | 2-3天 | 🔴 高 |
| 阶段二 | 2.1-2.4 功能仿真 | 3-4天 | 🔴 高 |
| 阶段三 | 3.1-3.3 测试验证 | 2天 | 🔴 高 |
| 优化 | 4.1-4.3 可选优化 | 5-7天 | 🟡 中 |

**总计**：7-9天（核心功能）+ 5-7天（可选优化）

---

## 依赖文件清单

### 需要修改的文件（按优先级）

#### 高优先级（必须修改）
1. `src/cuda-sim/ptx.l` - 词法分析器
2. `src/cuda-sim/ptx.y` - 语法分析器
3. `src/cuda-sim/ptx_ir.h` - PTX指令类定义
4. `src/cuda-sim/ptx_inst.cc` - 指令执行逻辑
5. `src/gpgpu-sim/shader.h` - 统计类更新
6. `src/gpgpu-sim/shader.cc` - 统计收集

#### 中优先级（可能需要修改）
7. `src/cuda-sim/ptx_thread_info.h/.cc` - 线程状态管理
8. `src/gpgpu-sim/memory.h/.cc` - 共享内存访问

#### 低优先级（可选）
9. `src/accelwattch/*.cc` - 功耗模型

---

## 关键技术点

### 1. PTX指令编码
- Token定义：`TOKEN_LD_SAMPLE_FMR`
- Opcode映射：`FMR_SAMPLE_OP`
- 操作数顺序：dst, base, x, y, stride

### 2. 地址计算公式
```cpp
addr = base + (y * stride + x) * sizeof(element)
```

### 3. 插值权重公式
```cpp
dx = frac(x), dy = frac(y)
α = (1-dx) * (1-dy)  // top-left
β = dx * (1-dy)      // top-right
γ = (1-dx) * dy      // bottom-left
λ = dx * dy          // bottom-right
```

### 4. 流水线时序
```
Cycle 0: 指令发射 (scheduler → ID_OC_FMR)
Cycle 1: 操作数收集 (ID_OC_FMR → OC_EX_FMR)
Cycle 2: 地址生成 + 权重计算
Cycle 3: 4-Bank并行访存
Cycle 4: 结果打包 (OC_EX_FMR → EX_WB)
Cycle 5: 写回寄存器
```

---

## 调试建议

### 启用详细日志
```bash
# 在gpgpusim.config中添加
-gpgpu_ptx_verbose 1
-gpgpu_ptx_instruction_classification 1
-gpgpu_pipeline_debug 1
```

### 关键检查点
1. PTX解析阶段：`ptxinfo_*` 日志文件
2. 功能仿真阶段：添加printf在 `ptx_inst.cc`
3. 微架构执行：查看 `pipeline_debug` 输出
4. 统计验证：`gpgpu_inst_stats.txt`

---

## 成功标准

### 功能正确性
- [ ] PTX指令正确解析
- [ ] 地址计算正确（4个地址）
- [ ] 权重计算正确（总和=1.0）
- [ ] 采样结果符合双线性插值

### 性能目标
- [ ] FMR指令替代原有的多条指令
- [ ] SP/INT/LDST单元占用率降低
- [ ] 总执行周期减少（相比baseline）

### 代码质量
- [ ] 通过编译（无警告）
- [ ] 通过基础测试用例
- [ ] 代码有充分注释

---

## 总结

本计划将分三个阶段实现FMR单元的完整功能：

1. **阶段一（PTX前端）**：让模拟器识别新指令
2. **阶段二（功能仿真）**：实现正确的功能行为
3. **阶段三（测试验证）**：端到端验证

核心创新点：
- ✅ 5操作数指令设计（包含stride）
- ✅ 硬件地址生成（释放CUDA Core）
- ✅ 4-Bank并行访存（减少访存延迟）
- ⏳ 灵活的插值计算位置（FMR或CUDA Core）

**下一步行动**：开始阶段一任务1.1，修改 `ptx.l` 添加token定义。
