# Deformable Attention Baseline Implementation

Pure CUDA implementation of Deformable Attention without hardware acceleration or WMMA.

## Key Features

- **No WMMA**: Uses standard FP32 operations instead of Tensor Cores
- **No Hardware Acceleration**: Pure CUDA kernel implementation
- **Standard Bilinear Interpolation**: Manual implementation without specialized hardware

## Compared to deAttn_fp16

| Feature | deAttn_fp16 | deAttn_base |
|---------|-------------|-------------|
| Precision | FP16 (half) | FP32 (float) |
| WMMA | Yes | No |
| Tensor Core | Used | Not used |
| Hardware Accel | N/A | N/A |

## Usage

```bash
# Compile
make clean && make

# Run
./bin/test

# Run with GPGPU-Sim
./run.sh
```

## Implementation

This baseline serves as a reference for comparing optimized versions. All computations are done using standard CUDA operations without any specialized hardware features.
