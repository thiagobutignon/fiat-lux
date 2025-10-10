/**
 * GCUDA Types
 *
 * Type definitions for Grammar CUDA - O(1) GPU acceleration
 */

// ============================================================================
// Device Types
// ============================================================================

export type GPUVendor = 'nvidia' | 'amd' | 'intel' | 'apple' | 'unknown';

export interface GPUDevice {
  id: number;                    // Device ID (0, 1, 2, ...)
  name: string;                  // "NVIDIA RTX 4090"
  vendor: GPUVendor;
  compute: string;               // Compute capability (e.g., "8.9")
  memory: number;                // Total memory in bytes
  memoryFree: number;            // Free memory in bytes
  cores: number;                 // CUDA cores / Stream processors
  clockSpeed: number;            // MHz
  pcieBus: string;               // "0000:01:00.0"
  uuid?: string;                 // GPU UUID
}

export interface DeviceStats {
  utilization: number;           // 0-100%
  memoryUsed: number;            // Bytes
  memoryTotal: number;           // Bytes
  temperature: number;           // Celsius
  powerUsage: number;            // Watts
  powerLimit: number;            // Watts
}

export interface GPURequirements {
  vendor?: GPUVendor | 'any';
  compute?: string;              // Minimum compute capability
  memory?: number;               // Minimum memory in bytes
  cores?: number;                // Minimum cores
}

// ============================================================================
// Kernel Types
// ============================================================================

export type KernelLang = 'cuda' | 'opencl' | 'metal' | 'webgpu';

export interface GCUDAKernel {
  hash: string;                  // sha256 of source + compiler flags
  name: string;                  // "matrix_multiply"
  version: string;               // "1.0.0"
  lang: KernelLang;
  source: string;                // Original source code
  sourcePath: string;            // Path to source file
  compiled?: Buffer;             // Compiled binary (PTX/SPIR-V/etc)
  entryPoint: string;            // "matmul_kernel"
  metadata: KernelMetadata;
}

export interface KernelMetadata {
  compileTime: string;           // ISO timestamp
  compiler: string;              // "nvcc 12.0"
  flags: string[];
  arch: string[];                // ["sm_80", "sm_89"]
  size: number;                  // Compiled binary size
  registers?: number;            // Register usage
  sharedMemory?: number;         // Shared memory usage (bytes)
}

export interface CompileOptions {
  flags?: string[];
  arch?: string[];
  optimization?: 'O0' | 'O1' | 'O2' | 'O3';
  debug?: boolean;
  verbose?: boolean;
}

// ============================================================================
// Memory Types
// ============================================================================

export type BufferType = 'device' | 'host' | 'managed';

export interface MemoryBuffer {
  id: string;                    // Unique buffer ID
  device: number;                // Device ID
  size: number;                  // Size in bytes
  devicePtr?: number;            // GPU memory pointer (if applicable)
  hostPtr?: Buffer;              // Host-side buffer
  type: BufferType;
  allocated: string;             // ISO timestamp
  freed?: string;                // ISO timestamp (if freed)
}

export interface MemoryStats {
  totalAllocated: number;        // Total bytes allocated
  totalFree: number;             // Total bytes freed
  currentUsage: number;          // Current usage in bytes
  peakUsage: number;             // Peak usage in bytes
  allocationCount: number;       // Number of allocations
  freeCount: number;             // Number of frees
}

// ============================================================================
// Execution Types
// ============================================================================

export interface GCUDAContext {
  id: string;
  device: GPUDevice;
  kernels: Map<string, GCUDAKernel>;  // hash → kernel
  buffers: Map<string, MemoryBuffer>; // id → buffer
  streams: GCUDAStream[];
  stats: ExecutionStats;
  created: string;               // ISO timestamp
}

export interface GCUDAStream {
  id: number;                    // Stream ID (0 = default stream)
  priority: number;              // Stream priority
  flags: string[];               // Stream flags
}

export interface GCUDAOperation {
  type: 'kernel' | 'memcpy_h2d' | 'memcpy_d2h' | 'memcpy_d2d';
  timestamp: string;             // ISO timestamp
  duration?: number;             // milliseconds

  // Kernel launch
  kernel?: string;               // Kernel hash
  grid?: [number, number, number];
  block?: [number, number, number];
  args?: any[];

  // Memory copy
  src?: string;                  // Source buffer ID
  dst?: string;                  // Destination buffer ID
  size?: number;                 // Bytes to copy

  // Status
  status: 'pending' | 'executing' | 'completed' | 'error';
  error?: string;
}

export interface Dim3 {
  x: number;
  y: number;
  z: number;
}

export interface LaunchConfig {
  gridDim: Dim3;
  blockDim: Dim3;
  sharedMemory?: number;         // Bytes
  stream?: number;               // Stream ID
}

export interface ExecutionStats {
  totalKernelsLaunched: number;
  totalExecutionTime: number;    // milliseconds
  averageExecutionTime: number;  // milliseconds
  failedKernels: number;
}

export interface KernelLaunchConfig {
  grid: [number, number, number];
  block: [number, number, number];
  sharedMemory?: number;         // Bytes
  stream?: string;               // Stream ID
}

// ============================================================================
// GCUDA Spec Types (.gcuda file format)
// ============================================================================

export interface GCUDASpec {
  format: string;                // "gcuda-v1.0"
  name: string;
  version: string;

  gpu: GPURequirements;

  kernels: KernelSpec[];

  build: BuildConfig;

  runtime?: RuntimeConfig;

  dependencies?: DependencySpec[];

  metadata?: SpecMetadata;
}

export interface KernelSpec {
  name: string;
  lang: KernelLang;
  source: string;                // Path to source file
  entry: string;                 // Entry point function name
}

export interface BuildConfig {
  compiler: string;              // "nvcc", "clang", "metal"
  flags?: string[];
  arch?: string[];
  defines?: { [key: string]: string };
}

export interface RuntimeConfig {
  max_threads_per_block?: number;
  shared_memory?: string;        // "48KB"
  registers?: number;
  max_grid_size?: [number, number, number];
}

export interface DependencySpec {
  name: string;
  version?: string;
  hash?: string;
}

export interface SpecMetadata {
  author?: string;
  description?: string;
  tags?: string[];
  license?: string;
  homepage?: string;
  repository?: string;
}

// ============================================================================
// Error Types
// ============================================================================

export class GCUDAError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'GCUDAError';
  }
}

export class DeviceError extends GCUDAError {
  constructor(message: string) {
    super(message);
    this.name = 'DeviceError';
  }
}

export class CompilationError extends GCUDAError {
  constructor(message: string) {
    super(message);
    this.name = 'CompilationError';
  }
}

export class MemoryError extends GCUDAError {
  constructor(message: string) {
    super(message);
    this.name = 'MemoryError';
  }
}

export class ExecutionError extends GCUDAError {
  constructor(message: string) {
    super(message);
    this.name = 'ExecutionError';
  }
}
