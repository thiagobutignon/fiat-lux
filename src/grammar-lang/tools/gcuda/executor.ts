/**
 * GCUDA Execution Engine
 *
 * Manages kernel execution on GPU devices with O(1) context lookup.
 * Glass-box: tracks all executions, provides stats.
 */

import {
  GPUDevice,
  GCUDAKernel,
  MemoryBuffer,
  LaunchConfig,
  ExecutionStats,
  ExecutionError,
  GCUDAStream,
} from './types';
import { MemoryManager } from './memory';

// ============================================================================
// Execution Context
// ============================================================================

export class GCUDAContext {
  private id: string;
  private device: GPUDevice;
  private memory: MemoryManager;
  private kernels: Map<string, GCUDAKernel>;
  private streams: GCUDAStream[];
  private executions: Map<string, ExecutionRecord>;
  private stats: ExecutionStats;

  constructor(device: GPUDevice) {
    this.id = this.generateContextId();
    this.device = device;
    this.memory = new MemoryManager(device);
    this.kernels = new Map();
    this.streams = [];
    this.executions = new Map();

    this.stats = {
      totalKernelsLaunched: 0,
      totalExecutionTime: 0,
      averageExecutionTime: 0,
      failedKernels: 0,
    };

    // Create default stream
    this.streams.push({
      id: 0,
      priority: 0,
      flags: [],
    });

    console.log(`✅ Created GCUDA context: ${this.id}`);
    console.log(`   Device: ${this.device.name}`);
  }

  /**
   * Register a kernel for execution
   */
  registerKernel(kernel: GCUDAKernel): void {
    this.kernels.set(kernel.hash, kernel);
    console.log(`📦 Registered kernel: ${kernel.name} (${kernel.hash.substring(0, 12)})`);
  }

  /**
   * Launch a kernel on the GPU
   * O(1) kernel lookup
   */
  async launchKernel(
    kernelHash: string,
    buffers: MemoryBuffer[],
    config: LaunchConfig
  ): Promise<ExecutionRecord> {
    const kernel = this.kernels.get(kernelHash);

    if (!kernel) {
      throw new ExecutionError(`Kernel not found: ${kernelHash}`);
    }

    // Validate launch configuration
    this.validateLaunchConfig(config);

    // Validate buffers
    for (const buffer of buffers) {
      if (!this.memory.getBuffer(buffer.id)) {
        throw new ExecutionError(`Buffer not found: ${buffer.id}`);
      }
      if (buffer.freed) {
        throw new ExecutionError(`Buffer already freed: ${buffer.id}`);
      }
    }

    console.log(`🚀 Launching kernel: ${kernel.name}`);
    console.log(`   Grid: (${config.gridDim.x}, ${config.gridDim.y}, ${config.gridDim.z})`);
    console.log(`   Block: (${config.blockDim.x}, ${config.blockDim.y}, ${config.blockDim.z})`);

    const executionId = this.generateExecutionId();
    const startTime = Date.now();

    try {
      // Mock execution (in reality: would call CUDA/OpenCL/Metal runtime)
      await this.mockKernelExecution(kernel, buffers, config);

      const endTime = Date.now();
      const executionTime = endTime - startTime;

      const record: ExecutionRecord = {
        id: executionId,
        kernelHash,
        kernelName: kernel.name,
        device: this.device.id,
        config,
        buffers: buffers.map(b => b.id),
        startTime: new Date(startTime).toISOString(),
        endTime: new Date(endTime).toISOString(),
        executionTime,
        status: 'completed',
      };

      this.executions.set(executionId, record);

      // Update stats
      this.stats.totalKernelsLaunched++;
      this.stats.totalExecutionTime += executionTime;
      this.stats.averageExecutionTime =
        this.stats.totalExecutionTime / this.stats.totalKernelsLaunched;

      console.log(`✅ Kernel execution complete (${executionTime}ms)`);

      return record;
    } catch (error) {
      const endTime = Date.now();
      const executionTime = endTime - startTime;

      const record: ExecutionRecord = {
        id: executionId,
        kernelHash,
        kernelName: kernel.name,
        device: this.device.id,
        config,
        buffers: buffers.map(b => b.id),
        startTime: new Date(startTime).toISOString(),
        endTime: new Date(endTime).toISOString(),
        executionTime,
        status: 'failed',
        error: error instanceof Error ? error.message : String(error),
      };

      this.executions.set(executionId, record);
      this.stats.failedKernels++;

      console.error(`❌ Kernel execution failed: ${record.error}`);

      throw new ExecutionError(`Kernel execution failed: ${record.error}`);
    }
  }

  /**
   * Synchronize device (wait for all kernels to complete)
   */
  async synchronize(): Promise<void> {
    console.log(`⏳ Synchronizing device ${this.device.id}...`);
    // Mock sync (in reality: cudaDeviceSynchronize)
    await new Promise(resolve => setTimeout(resolve, 1));
    console.log(`✅ Device synchronized`);
  }

  /**
   * Get memory manager for this context
   */
  getMemoryManager(): MemoryManager {
    return this.memory;
  }

  /**
   * Get execution statistics
   */
  getStats(): ExecutionStats {
    return { ...this.stats };
  }

  /**
   * List all executions
   */
  listExecutions(): ExecutionRecord[] {
    return Array.from(this.executions.values());
  }

  /**
   * Get specific execution record
   */
  getExecution(id: string): ExecutionRecord | null {
    return this.executions.get(id) || null;
  }

  /**
   * Destroy context and free all resources
   */
  destroy(): void {
    console.log(`🗑️  Destroying context ${this.id}...`);

    // Free all buffers
    const buffers = this.memory.listBuffers();
    for (const buffer of buffers) {
      if (!buffer.freed) {
        this.memory.free(buffer.id);
      }
    }

    this.kernels.clear();
    this.executions.clear();

    console.log(`✅ Context destroyed`);
  }

  // ==========================================================================
  // Private Methods
  // ==========================================================================

  /**
   * Validate launch configuration
   */
  private validateLaunchConfig(config: LaunchConfig): void {
    const { gridDim, blockDim } = config;

    if (gridDim.x <= 0 || gridDim.y <= 0 || gridDim.z <= 0) {
      throw new ExecutionError('Grid dimensions must be positive');
    }

    if (blockDim.x <= 0 || blockDim.y <= 0 || blockDim.z <= 0) {
      throw new ExecutionError('Block dimensions must be positive');
    }

    // Check against device limits (mock limits for now)
    const maxBlockSize = 1024;
    const totalThreadsPerBlock = blockDim.x * blockDim.y * blockDim.z;

    if (totalThreadsPerBlock > maxBlockSize) {
      throw new ExecutionError(
        `Block size too large: ${totalThreadsPerBlock} > ${maxBlockSize}`
      );
    }
  }

  /**
   * Mock kernel execution
   * In production: would use cuLaunchKernel / clEnqueueNDRangeKernel / Metal dispatch
   */
  private async mockKernelExecution(
    kernel: GCUDAKernel,
    buffers: MemoryBuffer[],
    config: LaunchConfig
  ): Promise<void> {
    const { gridDim, blockDim } = config;
    const totalThreads =
      (gridDim.x * gridDim.y * gridDim.z) *
      (blockDim.x * blockDim.y * blockDim.z);

    // Simulate execution time based on thread count
    // Assume ~1 TFLOPS GPU: 1e12 operations/second
    // Each thread does ~100 operations on average
    const operationsPerThread = 100;
    const totalOperations = totalThreads * operationsPerThread;
    const flops = 1e12; // 1 TFLOPS
    const executionTimeMs = (totalOperations / flops) * 1000;

    // Add some overhead (kernel launch latency)
    const launchOverheadMs = 0.05; // 50 microseconds
    const totalTimeMs = executionTimeMs + launchOverheadMs;

    await new Promise(resolve => setTimeout(resolve, totalTimeMs));
  }

  private generateContextId(): string {
    return `ctx_${Date.now()}_${Math.random().toString(36).substring(7)}`;
  }

  private generateExecutionId(): string {
    return `exec_${Date.now()}_${Math.random().toString(36).substring(7)}`;
  }
}

// ============================================================================
// Execution Record
// ============================================================================

export interface ExecutionRecord {
  id: string;
  kernelHash: string;
  kernelName: string;
  device: number;
  config: LaunchConfig;
  buffers: string[];
  startTime: string;
  endTime: string;
  executionTime: number; // ms
  status: 'completed' | 'failed';
  error?: string;
}

// ============================================================================
// Executor - High-level API
// ============================================================================

export class GCUDAExecutor {
  private contexts: Map<number, GCUDAContext>;

  constructor() {
    this.contexts = new Map();
  }

  /**
   * Create execution context for a device
   * O(1) context creation
   */
  createContext(device: GPUDevice): GCUDAContext {
    if (this.contexts.has(device.id)) {
      throw new ExecutionError(`Context already exists for device ${device.id}`);
    }

    const context = new GCUDAContext(device);
    this.contexts.set(device.id, context);

    return context;
  }

  /**
   * Get context for a device
   * O(1) lookup
   */
  getContext(deviceId: number): GCUDAContext | null {
    return this.contexts.get(deviceId) || null;
  }

  /**
   * Destroy context
   */
  destroyContext(deviceId: number): void {
    const context = this.contexts.get(deviceId);
    if (context) {
      context.destroy();
      this.contexts.delete(deviceId);
    }
  }

  /**
   * Destroy all contexts
   */
  destroyAllContexts(): void {
    for (const [deviceId, context] of this.contexts) {
      context.destroy();
      this.contexts.delete(deviceId);
    }
  }
}
