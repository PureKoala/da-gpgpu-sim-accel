# Deformable Attention CUDA Implementation

A high-performance CUDA implementation of Deformable Attention for computer vision applications, specifically designed for GPGPU-Sim simulation and performance analysis.

## Overview

Deformable Attention is a flexible attention mechanism that learns to attend to specific spatial locations dynamically, rather than attending to all spatial positions uniformly. This implementation provides a complete CUDA kernel for both forward and backward passes, with comprehensive performance profiling capabilities.

## Algorithm Background

### What is Deformable Attention?

Traditional self-attention mechanisms compute attention weights between all query-key pairs in a sequence, which can be computationally expensive for large spatial resolutions. Deformable Attention addresses this by:

1. **Learning sampling locations**: Instead of attending to all spatial positions, the model learns where to sample from
2. **Bilinear interpolation**: Uses smooth sampling via bilinear interpolation
3. **Multi-scale features**: Operates on feature pyramids with multiple resolution levels

### Mathematical Formulation

For each query position $q$, the output is computed as:

$
output(q) = Σ_{k=1}^K Σ_{m=1}^M A_{q,k,m} · V(p_q + Δp_{q,k,m})
$

Where:
- $A_{q,k,m}$ is the attention weight for query $q$, level $k$, and sample point $m$
- $V(·)$ represents bilinear sampling from the value feature map
- $p_q$ is the reference position for query $q$
- $Δp_{q,k,m}$ is the learnable offset for sampling location

## Parameter Configuration Explained

The test program uses the following default configuration, each parameter corresponding to specific aspects of the Deformable Attention algorithm:

### Core Algorithm Parameters

#### `batch_size: 2`
- **Algorithm role**: Number of independent samples processed in parallel
- **Memory impact**: All tensors have batch dimension as the first axis
- **Parallelization**: Each batch item can be processed independently across different GPU blocks
- **Typical range**: 1-32 depending on memory constraints

#### `num_query: 256` 
- **Algorithm role**: Number of query positions where attention is computed
- **Physical meaning**: 
  - In object detection: Number of object queries
  - In segmentation: Number of pixel/region queries  
  - In general: Spatial locations where we want to aggregate information
- **Computation**: Each query requires sampling from multiple levels and points
- **Memory scaling**: Linear impact on output tensor size

#### `num_heads: 8`
- **Algorithm role**: Number of parallel attention heads (multi-head attention)
- **Purpose**: 
  - Allows model to attend to different representation subspaces
  - Each head learns different types of spatial relationships
  - Improves model expressiveness and robustness
- **Implementation**: Each head processes independently with separate attention weights
- **Memory scaling**: Multiplies intermediate tensor sizes

#### `channels: 32`
- **Algorithm role**: Feature dimension per attention head
- **Physical meaning**: 
  - Number of channels in the value features for each head
  - Determines the richness of information aggregated at each query
- **Total output channels**: `num_heads × channels = 8 × 32 = 256`
- **Computation**: Each channel is processed independently during aggregation

#### `num_levels: 4`
- **Algorithm role**: Number of feature pyramid levels
- **Multi-scale processing**: 
  - Level 0: Full resolution (32×32)
  - Level 1: Half resolution (16×16)  
  - Level 2: Quarter resolution (8×8)
  - Level 3: Eighth resolution (4×4)
- **Purpose**: Enables attention across multiple scales for better context understanding
- **Memory layout**: Features are concatenated along spatial dimension

#### `num_point: 4`
- **Algorithm role**: Number of sampling points per query per level
- **Sampling strategy**: Each query samples 4 locations from each pyramid level
- **Total samples per query**: `num_levels × num_point = 4 × 4 = 16`
- **Learnable offsets**: Each sampling point has learnable 2D offset (x, y)
- **Attention weights**: Each sampling point has its own learned attention weight

#### `base_size: 32×32`
- **Algorithm role**: Spatial resolution of the highest (level 0) feature map
- **Pyramid construction**:
  - Level 0: 32×32 = 1,024 spatial positions
  - Level 1: 16×16 = 256 spatial positions
  - Level 2: 8×8 = 64 spatial positions  
  - Level 3: 4×4 = 16 spatial positions
  - **Total spatial_size**: 1,024 + 256 + 64 + 16 = 1,360

## Memory Layout and Tensor Shapes

### Input Tensors

1. **Value Features**: `[batch_size, spatial_size, num_heads, channels]`
   - Shape: `[2, 1360, 8, 32]`
   - Content: Feature representations to be sampled from
   - Layout: Concatenated multi-level features

2. **Sampling Locations**: `[batch_size, num_query, num_heads, num_levels, num_point, 2]`
   - Shape: `[2, 256, 8, 4, 4, 2]`
   - Content: Normalized coordinates (x, y) ∈ [0, 1] for each sampling point
   - Last dimension: (x_offset, y_offset)

3. **Attention Weights**: `[batch_size, num_query, num_heads, num_levels, num_point]`
   - Shape: `[2, 256, 8, 4, 4]`
   - Content: Learned importance scores for each sampling point
   - Constraint: Usually normalized (sum to 1) across levels and points

4. **Spatial Shapes**: `[num_levels, 2]`
   - Shape: `[4, 2]`
   - Content: `[[32,32], [16,16], [8,8], [4,4]]`
   - Format: (height, width) for each pyramid level

5. **Level Start Index**: `[num_levels]`
   - Shape: `[4]`
   - Content: `[0, 1024, 1280, 1344]`
   - Purpose: Starting indices for each level in the concatenated spatial dimension

### Output Tensor

- **Output Features**: `[batch_size, num_query, num_heads, channels]`
  - Shape: `[2, 256, 8, 32]`
  - Content: Aggregated features for each query position

## Computational Complexity

### Per-Query Computation
For each of the 256 queries:
1. **Sampling**: 16 bilinear interpolations (4 levels × 4 points)
2. **Weighting**: 16 scalar multiplications  
3. **Aggregation**: 16 weighted additions per channel
4. **Total per query**: ~16 × 10 = 160 operations per channel

### Total Computation
- **Total operations**: `batch_size × num_query × num_heads × channels × num_levels × num_point × 10`
- **For default config**: `2 × 256 × 8 × 32 × 4 × 4 × 10 ≈ 21M operations`

## Performance Characteristics

### Memory Access Patterns
- **Value sampling**: Irregular, cache-unfriendly due to learned offsets
- **Attention weights**: Sequential, cache-friendly
- **Output writing**: Sequential, optimal for coalescing

### Parallelization Strategy
- **Block-level**: Different queries processed by different thread blocks
- **Warp-level**: Multiple heads processed within each warp
- **Thread-level**: Individual channels processed by threads

## Usage

### Compilation
```bash
# Using Makefile
make clean && make

# Using run.sh for GPGPU-Sim
bash run.sh
```

### Running Tests
```bash
# Direct execution (requires compatible NVIDIA driver)
./bin/test

# GPGPU-Sim simulation
bash run.sh
cat out/test_RTX3070.txt
```

### Customizing Parameters

Edit the configuration section in `src/test.cu`:

```cpp
// Adjust these parameters based on your needs
const int batch_size = 4;          // Increase for better GPU utilization
const int num_query = 512;         // More queries for complex scenes  
const int num_heads = 16;          // More heads for richer representations
const int channels = 64;           // Higher feature dimensions
const int num_levels = 3;          // Fewer levels for speed
const int num_point = 8;           // More sampling points for accuracy
const int base_height = 64;        // Higher resolution inputs
const int base_width = 64;
```

## Performance Analysis

The implementation provides detailed performance metrics:

- **Kernel execution time**: Measures GPU computation time
- **Memory transfer time**: H2D and D2H transfer costs
- **GFLOPS**: Effective computational throughput
- **Memory footprint**: Total GPU memory usage
- **Output validation**: Checks for numerical stability

### Sample Output
```
========================================
         性能统计摘要
========================================
Kernel执行时间:    156000.000 ms
内存传输时间:      0.000 ms  
总时间:            156000.000 ms
计算操作数:        20971520
计算性能:          0.00 GFLOPS
内存占用:          3.91 MB
========================================
```

## Architecture Optimization

### For Real GPUs
- Increase batch size for better occupancy
- Ensure tensor dimensions are multiples of 32 (warp size)
- Use CUDA streams for overlapping computation and memory transfer

### For GPGPU-Sim Analysis
- Use smaller problem sizes for faster simulation
- Focus on cache hit rates and memory access patterns
- Analyze warp utilization and branch divergence
- Study the impact of irregular sampling on memory hierarchy

## Applications

This implementation is suitable for:

1. **Object Detection**: Deformable DETR, Sparse R-CNN
2. **Semantic Segmentation**: Learning adaptive sampling patterns
3. **Video Understanding**: Temporal deformable attention
4. **Architecture Research**: GPU memory hierarchy studies
5. **Performance Analysis**: GPGPU-Sim simulation studies

## File Structure

```
deAttn/
├── README.md                           # This file
├── README_CN.md                        # Chinese documentation  
├── IMPLEMENTATION_SUMMARY.md           # Detailed implementation notes
├── Makefile                            # Build system
├── run.sh                              # GPGPU-Sim execution script
│
├── src/
│   ├── test.cu                         # Main test program
│   └── deform_attn/
│       ├── deform_attn_cuda.h          # CUDA API header
│       └── cuda/
│           ├── ms_deform_attn_im2col_cuda.cuh      # Core kernels
│           ├── ms_deform_attn_cuda_kernel.cu       # Kernel wrappers
│           └── ms_deform_attn_wmma_cuda.cuh        # Tensor Core support
│
├── bin/test                            # Compiled executable
├── out/test_RTX3070.txt               # Simulation results
└── sim/test_RTX3070/                  # GPGPU-Sim environment
```

## References

1. **Deformable DETR**: "Deformable DETR: Deformable Transformers for End-to-End Object Detection", ICLR 2021
2. **Original Implementation**: [fundamentalvision/Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR)
3. **GPGPU-Sim**: [GPGPU-Sim Documentation](http://www.gpgpu-sim.org/)

## License

This implementation maintains compatibility with the original Apache License 2.0 from the Deformable-DETR project.