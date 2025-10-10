# GCUDA - Grammar CUDA Architecture

**Version**: 1.0.0
**Status**: Planning
**Goal**: O(1) GPU acceleration for Grammar Language toolchain

## Overview

GCUDA provides deterministic, glass-box GPU acceleration for containers and Grammar Language programs. It follows the same principles as GCR:
- **O(1) operations**: Predictable performance
- **Content-addressable**: Kernels identified by hash
- **Glass-box**: Complete transparency
- **Deterministic**: Same input = same output

## Core Concepts

### 1. GPU Device
Physical GPU available on the system.

```typescript
interface GPUDevice {
  id: number;                    // Device ID (0, 1, 2, ...)
  name: string;                  // "NVIDIA RTX 4090"
  vendor: 'nvidia' | 'amd' | 'intel' | 'apple';
  compute: string;               // Compute capability (e.g., "8.9")
  memory: number;                // Total memory in bytes
  memoryFree: number;            // Free memory in bytes
  cores: number;                 // CUDA cores / Stream processors
  clockSpeed: number;            // MHz
  pcieBus: string;               // "0000:01:00.0"
}
```

### 2. GCUDA Kernel
Compiled GPU code, content-addressable.

```typescript
interface GCUDAKernel {
  hash: string;                  // sha256 of source + compiler flags
  name: string;                  // "matrix_multiply"
  version: string;               // "1.0.0"
  lang: 'cuda' | 'opencl' | 'metal' | 'webgpu';
  source: string;                // Original source code
  compiled: Buffer;              // Compiled binary (PTX/SPIR-V/etc)
  entryPoint: string;            // "matmul_kernel"
  metadata: {
    compileTime: string;
    compiler: string;            // "nvcc 12.0"
    flags: string[];
    arch: string[];              // ["sm_80", "sm_89"]
  };
}
```

### 3. Memory Buffer
GPU memory allocation, tracked and managed.

```typescript
interface MemoryBuffer {
  id: string;                    // Unique buffer ID
  device: number;                // Device ID
  size: number;                  // Size in bytes
  devicePtr: number;             // GPU memory pointer
  hostPtr?: Buffer;              // Optional host-side copy
  type: 'device' | 'host' | 'managed';
  allocated: string;             // ISO timestamp
}
```

### 4. Execution Context
Runtime context for GPU operations.

```typescript
interface GCUDAContext {
  id: string;
  device: GPUDevice;
  kernels: Map<string, GCUDAKernel>;  // hash → kernel
  buffers: Map<string, MemoryBuffer>; // id → buffer
  streams: GCUDAStream[];
  stats: {
    kernelLaunches: number;
    memoryTransfers: number;
    totalComputeTime: number;       // milliseconds
  };
}
```

### 5. Execution Stream
Asynchronous execution queue.

```typescript
interface GCUDAStream {
  id: string;
  device: number;
  operations: GCUDAOperation[];
  status: 'idle' | 'executing' | 'error';
}

interface GCUDAOperation {
  type: 'kernel' | 'memcpy_h2d' | 'memcpy_d2h' | 'memcpy_d2d';
  kernel?: string;               // Kernel hash
  grid?: [number, number, number];
  block?: [number, number, number];
  args?: any[];
  src?: string;                  // Source buffer
  dst?: string;                  // Destination buffer
  size?: number;
}
```

## File Format: .gcuda

GCUDA spec files define GPU kernels and their requirements.

```yaml
format: gcuda-v1.0
name: matrix-multiply
version: 1.0.0

# GPU requirements
gpu:
  vendor: nvidia              # nvidia, amd, intel, apple, any
  compute: 7.0                # minimum compute capability
  memory: 4GB                 # minimum GPU memory
  cores: 2560                 # minimum cores

# Kernel definitions
kernels:
  - name: matmul
    lang: cuda
    source: kernels/matmul.cu
    entry: matmul_kernel

  - name: reduce
    lang: cuda
    source: kernels/reduce.cu
    entry: reduce_kernel

# Build configuration
build:
  compiler: nvcc              # nvcc, clang, metal
  flags:
    - -O3
    - --use_fast_math
    - -lineinfo
  arch:
    - sm_70                   # Volta
    - sm_80                   # Ampere
    - sm_89                   # Ada Lovelace

# Runtime configuration
runtime:
  max_threads_per_block: 1024
  shared_memory: 48KB
  registers: 65536
  max_grid_size: [2147483647, 65535, 65535]

# Dependencies (optional)
dependencies:
  - name: cublas
    version: 12.0
  - name: cufft
    version: 11.0

# Metadata
metadata:
  author: developer@example.com
  description: High-performance matrix multiplication
  tags:
    - linear-algebra
    - blas
    - gpu
  license: MIT
```

## Architecture Components

### 1. Device Manager (`device-manager.ts`)
- Detects available GPUs
- Queries device capabilities
- Monitors device utilization
- O(1) device lookup by ID

```typescript
class DeviceManager {
  listDevices(): GPUDevice[];
  getDevice(id: number): GPUDevice | null;
  selectBestDevice(requirements: GPURequirements): GPUDevice | null;
  getDeviceUtilization(id: number): DeviceStats;
}
```

### 2. Kernel Compiler (`compiler.ts`)
- Compiles CUDA/OpenCL/Metal code
- Content-addressable compilation (hash-based caching)
- O(1) kernel lookup by hash

```typescript
class KernelCompiler {
  compile(source: string, options: CompileOptions): Promise<GCUDAKernel>;
  compileFromFile(path: string, options: CompileOptions): Promise<GCUDAKernel>;
  getKernel(hash: string): GCUDAKernel | null;
  listKernels(): GCUDAKernel[];
}
```

### 3. Memory Manager (`memory.ts`)
- Allocates GPU memory
- Transfers data (host ↔ device)
- Tracks allocations
- O(1) buffer lookup

```typescript
class MemoryManager {
  allocate(device: number, size: number, type: BufferType): MemoryBuffer;
  free(bufferId: string): void;
  copyToDevice(bufferId: string, data: Buffer): Promise<void>;
  copyFromDevice(bufferId: string): Promise<Buffer>;
  copyDeviceToDevice(src: string, dst: string, size: number): Promise<void>;
  getBuffer(id: string): MemoryBuffer | null;
  getTotalAllocated(device: number): number;
}
```

### 4. Execution Engine (`executor.ts`)
- Launches kernels
- Manages execution streams
- Synchronizes operations
- Collects performance metrics

```typescript
class ExecutionEngine {
  createContext(device: number): GCUDAContext;
  launchKernel(
    ctx: GCUDAContext,
    kernel: string,
    grid: [number, number, number],
    block: [number, number, number],
    args: any[]
  ): Promise<void>;

  createStream(ctx: GCUDAContext): GCUDAStream;
  synchronize(ctx: GCUDAContext): Promise<void>;
  getStats(ctx: GCUDAContext): ExecutionStats;
}
```

### 5. GCR Integration (`gcr-integration.ts`)
- Allows containers to access GPUs
- GPU resource allocation for containers
- Isolation and sharing policies

```typescript
class GCRGPUIntegration {
  attachGPU(container: Container, device: number): void;
  detachGPU(container: Container): void;
  listGPUContainers(): Array<{ container: Container; device: GPUDevice }>;
  getContainerGPUStats(container: Container): GPUStats;
}
```

## Storage Structure

```
.gcuda/
├── devices/
│   └── cache.json              # Device capabilities cache
├── kernels/
│   └── sha256:abc123.../
│       ├── source.cu           # Original source
│       ├── compiled.ptx        # Compiled PTX
│       ├── compiled.cubin      # Compiled binary
│       └── metadata.json       # Compilation metadata
├── specs/
│   └── matrix-multiply_1.0.0/
│       └── spec.gcuda          # GCUDA spec file
└── cache/
    └── compilation-cache.json  # Compilation cache
```

## O(1) Performance Guarantees

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Device lookup | O(1) | Direct array access |
| Kernel lookup | O(1) | Hash map |
| Buffer lookup | O(1) | Hash map |
| Memory allocation | O(1) | GPU allocator |
| Kernel launch | O(1) | GPU async |
| Device list | O(n) | n = number of GPUs (typically 1-8) |

## Example: Matrix Multiplication

**1. GCUDA Spec** (`matmul.gcuda`):
```yaml
format: gcuda-v1.0
name: matmul
version: 1.0.0

gpu:
  vendor: any
  compute: 7.0
  memory: 2GB

kernels:
  - name: matmul
    lang: cuda
    source: matmul.cu
    entry: matmul_kernel

build:
  compiler: nvcc
  flags: ['-O3', '--use_fast_math']
  arch: ['sm_70']
```

**2. CUDA Kernel** (`matmul.cu`):
```cuda
__global__ void matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

**3. TypeScript Usage**:
```typescript
import { GCUDA } from './gcuda';

// Initialize GCUDA
const gcuda = new GCUDA();

// List devices
const devices = gcuda.listDevices();
console.log(`Found ${devices.length} GPU(s)`);

// Compile kernel
const kernel = await gcuda.compileKernel('matmul.cu', {
  flags: ['-O3', '--use_fast_math'],
  arch: ['sm_70']
});

// Create context
const ctx = gcuda.createContext(0); // Use GPU 0

// Allocate memory
const N = 1024;
const A = gcuda.allocate(ctx, N * N * 4); // 4 bytes per float
const B = gcuda.allocate(ctx, N * N * 4);
const C = gcuda.allocate(ctx, N * N * 4);

// Copy data to device
await gcuda.copyToDevice(A, matrixA);
await gcuda.copyToDevice(B, matrixB);

// Launch kernel
await gcuda.launchKernel(ctx, kernel, {
  grid: [N / 16, N / 16, 1],
  block: [16, 16, 1],
  args: [A, B, C, N]
});

// Copy result back
const result = await gcuda.copyFromDevice(C);

// Cleanup
gcuda.free(A);
gcuda.free(B);
gcuda.free(C);
gcuda.destroyContext(ctx);
```

**4. GCR Integration**:
```bash
# Build GCUDA kernel
gcuda build matmul.gcuda

# Run container with GPU
gcr run myapp:1.0.0 --gpu 0 --name gpu-container

# Inside container, GPU is available
# Container can load and execute kernels
```

## Implementation Plan

### DIA 1: Types + Device Management
- Define all TypeScript interfaces
- Implement DeviceManager
- Detect NVIDIA/AMD/Intel GPUs
- Query device capabilities
- Test device enumeration

### DIA 2: Kernel Compilation + Execution
- Implement KernelCompiler
- Compile CUDA kernels (nvcc)
- Content-addressable kernel storage
- Implement ExecutionEngine
- Launch kernels
- Test with simple kernel

### DIA 3: Memory Management + Transfers
- Implement MemoryManager
- Allocate GPU memory
- Host → Device transfers
- Device → Host transfers
- Device → Device transfers
- Test memory operations

### DIA 4: GCR Integration + Testing
- Integrate with GCR containers
- GPU resource allocation
- Container GPU isolation
- End-to-end testing
- Performance benchmarks

## Dependencies

**For NVIDIA GPUs**:
- CUDA Toolkit 12.0+
- nvcc compiler
- CUDA runtime

**For AMD GPUs**:
- ROCm 5.0+
- hipcc compiler

**For Apple GPUs**:
- Metal framework
- Metal shader compiler

**Node.js bindings**:
- node-cuda (for CUDA)
- node-opencl (for OpenCL)
- OR: Custom N-API bindings

## Non-Goals

- ❌ Multi-GPU automatic load balancing (can add later)
- ❌ Custom GPU memory allocators (use default)
- ❌ GPU peer-to-peer transfers (single GPU focus)
- ❌ Distributed GPU computing (single machine only)

## Success Criteria

✅ Detect GPUs on system
✅ Compile and cache CUDA kernels
✅ Allocate GPU memory
✅ Transfer data host ↔ device
✅ Launch kernels
✅ Integrate with GCR containers
✅ O(1) kernel/buffer lookups
✅ Full TypeScript type safety
✅ Glass-box transparency (all operations visible)

---

**Status**: Architecture Complete
**Next**: Begin implementation (DIA 1)
