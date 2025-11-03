# INT8 Tensor Core WMMA Support

## Overview
This document describes the modifications made to GPGPU-Sim to support INT8 WMMA (Warp Matrix Multiply-Accumulate) instructions for Tensor Cores. These changes enable simulation of neural networks using INT8 quantization for improved performance.

## Supported Instructions

### WMMA Load Instructions
- `wmma.load.matrix.sync.aligned.{row|col}.m16n16k16.s8.shared`
  - Loads 16x16 INT8 matrix fragments from shared memory
  - Supports both row-major and column-major layouts

### WMMA Compute Instructions
- `wmma.mma.sync.aligned.row.col.m16n16k16.s32.s8.s8.s32`
  - Performs matrix multiply-accumulate: D = A × B + C
  - Input matrices A, B: INT8 (signed 8-bit integers)
  - Accumulator matrices C, D: INT32 (signed 32-bit integers)
  - Matrix dimensions: 16×16×16 (M×N×K)

## Implementation Details

### File Modified
- **File**: `src/cuda-sim/instructions.cc`
- **Functions modified**:
  1. `thread_group_offset()` - Thread-to-element mapping
  2. `mma_ld_impl()` - WMMA load instruction implementation
  3. `mma_impl()` - WMMA compute instruction implementation

### Key Changes

#### 1. `mma_ld_impl()` - INT8 Data Loading

**Problem**: Original code assumed 16-bit FP16 elements, reading 4 elements (8 bytes) per thread. INT8 elements are 8-bit, requiring 8 elements (8 bytes) per thread.

**Solution**:
- Detect INT8 type: `type == S8_TYPE || type == U8_TYPE`
- Adjust data loading: Read **8 elements** instead of 4 for INT8
- Data packing: Pack **4 INT8 values** into each 32-bit register
  ```cpp
  // Pack 4 INT8 values into one uint32
  nw_data[k] = (data[k*4+0] & 0xff) | 
               ((data[k*4+1] & 0xff) << 8) |
               ((data[k*4+2] & 0xff) << 16) | 
               ((data[k*4+3] & 0xff) << 24);
  ```
- Register usage: **2 registers** per thread (instead of 4 for FP16)
  - Each register holds 4 packed INT8 values
  - Total: 2 regs × 4 values = 8 INT8 values per thread

**Memory Layout**:
```
INT8 Memory:  [b0 b1 b2 b3 b4 b5 b6 b7]  (8 bytes)
                ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
Packed Regs:  [b0|b1|b2|b3] [b4|b5|b6|b7]  (2 x uint32)
```

#### 2. `mma_impl()` - INT8 Matrix Multiplication

**Problem**: Original code performed floating-point operations on FP16 data. INT8 WMMA requires integer arithmetic with INT32 accumulation.

**Solution**:

##### a) INT8 Detection
```cpp
bool is_int8_wmma = (pI->get_scalar_type().size() == 4);
// 4-parameter type specification: m16n16k16.s32.s8.s8.s32
```

##### b) Data Unpacking
Unpack 4 INT8 values from each 32-bit register with sign extension:
```cpp
// Extract 4 signed INT8 values from packed uint32
for (k = 0; k < nelem; k++) {
  nw_v[k*4 + 0].s32 = (v[k].s64 & 0xff);
  nw_v[k*4 + 1].s32 = ((v[k].s64 >> 8) & 0xff);
  nw_v[k*4 + 2].s32 = ((v[k].s64 >> 16) & 0xff);
  nw_v[k*4 + 3].s32 = ((v[k].s64 >> 24) & 0xff);
  
  // Sign extend if type is signed INT8 (S8_TYPE)
  if (type == S8_TYPE) {
    if (nw_v[k*4 + 0].s32 & 0x80) nw_v[k*4 + 0].s32 |= 0xffffff00;
    if (nw_v[k*4 + 1].s32 & 0x80) nw_v[k*4 + 1].s32 |= 0xffffff00;
    // ... same for other elements
  }
}
```

##### c) Matrix Multiplication
Separate INT8 and FP16 computation paths:
```cpp
if (is_int8_wmma) {
  // INT8: Integer matrix multiplication
  // D[i][j] = Σ(A[i][k] * B[k][j]) + C[i][j]
  for (i = 0; i < 16; i++) {
    for (j = 0; j < 16; j++) {
      matrix_d[i][j].s32 = matrix_c[i][j].s32;  // Initialize with C
      for (k = 0; k < 16; k++) {
        // Integer multiply-accumulate
        matrix_d[i][j].s32 += matrix_a[i][k].s32 * matrix_b[k][j].s32;
      }
    }
  }
} else {
  // FP16: Floating-point multiplication (original code)
  // ...
}
```

##### d) Result Write-back
- Use **type2** (output type = S32) instead of **type** (input type = S8) for accumulator mapping
- Write **INT32 results** (8 values per thread) to destination registers
```cpp
mapping(thrd, LOAD_C, ROW, type2, k, 16, row, col, offset);
// type2 = S32_TYPE for INT8 WMMA
```

#### 3. `thread_group_offset()` - Element Mapping

**Result**: No changes needed. The existing offset tables work correctly for INT8 because they represent element indices, not byte offsets. The same thread-to-element mapping applies for both INT8 and FP16 data types.

## Data Type Specifications

### PTX Type Constants
- `S8_TYPE` (303): Signed 8-bit integer
- `U8_TYPE` (304): Unsigned 8-bit integer
- `S32_TYPE` (305): Signed 32-bit integer (accumulator)
- `F16_TYPE`: 16-bit floating point
- `F32_TYPE` (312): 32-bit floating point

### Register Usage Comparison

| Data Type | Elements per Thread | Bytes per Element | Registers per Thread | Values per Register |
|-----------|-------------------|------------------|---------------------|-------------------|
| FP16      | 8                 | 2                | 4                   | 2 (32-bit/16-bit) |
| INT8      | 8                 | 1                | 2                   | 4 (32-bit/8-bit)  |

### Matrix Operation Types

#### FP16 WMMA (Original)
- **A, B matrices**: FP16 (16-bit float)
- **C, D matrices**: FP16 or FP32 (16-bit or 32-bit float)
- **Operation**: Floating-point multiply-accumulate

#### INT8 WMMA (New)
- **A, B matrices**: INT8 (8-bit signed integer)
- **C, D matrices**: INT32 (32-bit signed integer)
- **Operation**: Integer multiply-accumulate with 32-bit accumulation

## Design Rationale

### Minimal Modification Approach
The implementation follows a minimal modification strategy:
1. **Reuse existing code paths**: Use conditional branches (`if (is_int8_wmma)`) instead of duplicating code
2. **Preserve FP16 functionality**: All changes are additive; no FP16 code removed
3. **Consistent data flow**: Follow the same load → unpack → compute → pack → store pattern

### Why Separate INT8 Computation?
- **Type safety**: INT8×INT8 → INT32 arithmetic is fundamentally different from FP16×FP16 → FP16/FP32
- **Accuracy**: Integer operations must avoid floating-point rounding errors
- **Accumulator width**: INT8 products require INT32 accumulation to prevent overflow (max product: 127×127 = 16129)

### Packing Strategy Justification
- **Register efficiency**: Pack 4 INT8 values per 32-bit register (100% utilization)
- **Memory bandwidth**: Read 8 bytes per thread (same as FP16 WMMA)
- **Warp uniformity**: All 32 threads in warp use identical packing scheme

## Validation

### Test Case
- **Network**: Deformable Attention (deAttn)
- **Architecture**: sm_86 (NVIDIA RTX 3070 configuration)
- **Result**: Successfully executed to completion with correct output

### Verification Steps
1. Program launches and initializes GPGPU-Sim correctly
2. INT8 WMMA load instructions execute without crashes
3. INT8 WMMA compute instructions produce correct matrix results
4. Memory operations (shared memory loads) work correctly
5. Output values match expected neural network behavior
6. Program exits cleanly (exit code 0)

## Future Enhancements

### Potential Improvements
1. **Additional INT8 variants**:
   - Unsigned INT8 (U8×U8 → U32)
   - Mixed precision (U8×S8 → S32)
   - Different matrix sizes (m8n8k32, etc.)

2. **Performance optimizations**:
   - SIMD optimizations for packing/unpacking
   - Vectorized integer multiply-accumulate
   - Cache-friendly matrix traversal

3. **Extended accumulator support**:
   - INT8 → INT16 accumulation (for smaller networks)
   - INT8 → INT64 accumulation (for higher precision)

## References

### NVIDIA Documentation
- PTX ISA Documentation: WMMA Instructions
- CUDA C++ Programming Guide: Tensor Cores
- Turing Architecture Whitepaper: INT8 Tensor Core Operations

### GPGPU-Sim Documentation
- `src/cuda-sim/instructions.h`: PTX instruction definitions
- `src/abstract_hardware_model.h`: Type definitions and constants
- `doc/doxygen/`: API documentation

## Authors and Contributors
- Implementation: GPGPU-Sim modifications for INT8 WMMA support
- Testing: Deformable Attention network validation

## Version Information
- **GPGPU-Sim Version**: 4.2.0
- **Target CUDA Version**: 11.0.3
- **Supported Compute Capability**: sm_70 and above (Tensor Core architectures)
