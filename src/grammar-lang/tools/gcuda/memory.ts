/**
 * GCUDA Memory Manager
 *
 * Manages GPU memory allocation and transfers with O(1) lookups.
 * Glass-box: tracks all allocations, provides stats.
 */

import * as crypto from 'crypto';
import {
  MemoryBuffer,
  BufferType,
  MemoryStats,
  MemoryError,
  GPUDevice,
} from './types';

// ============================================================================
// Memory Manager
// ============================================================================

export class MemoryManager {
  private buffers: Map<string, MemoryBuffer>;
  private stats: MemoryStats;
  private device: GPUDevice;

  // Mock GPU memory (in reality, this would be cudaMalloc/cuMemAlloc pointers)
  private mockDeviceMemory: Map<string, Buffer>;

  constructor(device: GPUDevice) {
    this.device = device;
    this.buffers = new Map();
    this.mockDeviceMemory = new Map();

    this.stats = {
      totalAllocated: 0,
      totalFree: 0,
      currentUsage: 0,
      peakUsage: 0,
      allocationCount: 0,
      freeCount: 0,
    };
  }

  /**
   * Allocate GPU memory
   * O(1) allocation
   */
  allocate(size: number, type: BufferType = 'device'): MemoryBuffer {
    if (size <= 0) {
      throw new MemoryError('Size must be positive');
    }

    // Only check free memory if device reports it (some devices don't expose this)
    if (this.device.memoryFree > 0 && size > this.device.memoryFree) {
      throw new MemoryError(
        `Out of memory: requested ${formatSize(size)}, available ${formatSize(this.device.memoryFree)}`
      );
    }

    // Generate unique buffer ID
    const id = this.generateBufferId();

    const buffer: MemoryBuffer = {
      id,
      device: this.device.id,
      size,
      type,
      allocated: new Date().toISOString(),
    };

    if (type === 'device') {
      // Mock device memory (in reality: cudaMalloc)
      const deviceMem = Buffer.alloc(size);
      this.mockDeviceMemory.set(id, deviceMem);
      buffer.devicePtr = parseInt(id.substring(0, 8), 16); // Mock pointer
    } else if (type === 'host') {
      // Host-pinned memory
      buffer.hostPtr = Buffer.alloc(size);
    } else if (type === 'managed') {
      // Unified memory (accessible from both host and device)
      buffer.hostPtr = Buffer.alloc(size);
      const deviceMem = Buffer.alloc(size);
      this.mockDeviceMemory.set(id, deviceMem);
      buffer.devicePtr = parseInt(id.substring(0, 8), 16);
    }

    // Track allocation
    this.buffers.set(id, buffer);
    this.stats.totalAllocated += size;
    this.stats.currentUsage += size;
    this.stats.allocationCount++;

    if (this.stats.currentUsage > this.stats.peakUsage) {
      this.stats.peakUsage = this.stats.currentUsage;
    }

    console.log(`   ✅ Allocated ${formatSize(size)} (${type})`);
    console.log(`      Buffer ID: ${id}`);

    return buffer;
  }

  /**
   * Free GPU memory
   * O(1) free
   */
  free(bufferId: string): void {
    const buffer = this.buffers.get(bufferId);

    if (!buffer) {
      throw new MemoryError(`Buffer not found: ${bufferId}`);
    }

    if (buffer.freed) {
      throw new MemoryError(`Buffer already freed: ${bufferId}`);
    }

    // Free device memory
    if (this.mockDeviceMemory.has(bufferId)) {
      this.mockDeviceMemory.delete(bufferId);
    }

    // Mark as freed
    buffer.freed = new Date().toISOString();

    // Update stats
    this.stats.totalFree += buffer.size;
    this.stats.currentUsage -= buffer.size;
    this.stats.freeCount++;

    console.log(`   ✅ Freed ${formatSize(buffer.size)}`);
    console.log(`      Buffer ID: ${bufferId}`);
  }

  /**
   * Copy data from host to device
   * O(n) where n = size
   */
  async copyToDevice(bufferId: string, data: Buffer): Promise<void> {
    const buffer = this.buffers.get(bufferId);

    if (!buffer) {
      throw new MemoryError(`Buffer not found: ${bufferId}`);
    }

    if (buffer.freed) {
      throw new MemoryError(`Buffer already freed: ${bufferId}`);
    }

    if (data.length !== buffer.size) {
      throw new MemoryError(
        `Size mismatch: buffer is ${buffer.size} bytes, data is ${data.length} bytes`
      );
    }

    if (buffer.type === 'host') {
      throw new MemoryError('Cannot copy to host buffer');
    }

    console.log(`   📤 Copying ${formatSize(data.length)} to device...`);

    // Mock transfer (in reality: cudaMemcpy H2D)
    const deviceMem = this.mockDeviceMemory.get(bufferId);
    if (deviceMem) {
      data.copy(deviceMem);
    }

    // Simulate transfer time (10 GB/s transfer rate)
    const transferTimeMs = (data.length / (10 * 1024 * 1024 * 1024)) * 1000;
    await new Promise(resolve => setTimeout(resolve, transferTimeMs));

    console.log(`   ✅ Transfer complete (${transferTimeMs.toFixed(2)}ms)`);
  }

  /**
   * Copy data from device to host
   * O(n) where n = size
   */
  async copyFromDevice(bufferId: string): Promise<Buffer> {
    const buffer = this.buffers.get(bufferId);

    if (!buffer) {
      throw new MemoryError(`Buffer not found: ${bufferId}`);
    }

    if (buffer.freed) {
      throw new MemoryError(`Buffer already freed: ${bufferId}`);
    }

    if (buffer.type === 'host') {
      // Already on host
      return buffer.hostPtr!;
    }

    console.log(`   📥 Copying ${formatSize(buffer.size)} from device...`);

    // Mock transfer (in reality: cudaMemcpy D2H)
    const deviceMem = this.mockDeviceMemory.get(bufferId);
    if (!deviceMem) {
      throw new MemoryError(`Device memory not found for buffer: ${bufferId}`);
    }

    const hostData = Buffer.alloc(buffer.size);
    deviceMem.copy(hostData);

    // Simulate transfer time (10 GB/s transfer rate)
    const transferTimeMs = (buffer.size / (10 * 1024 * 1024 * 1024)) * 1000;
    await new Promise(resolve => setTimeout(resolve, transferTimeMs));

    console.log(`   ✅ Transfer complete (${transferTimeMs.toFixed(2)}ms)`);

    return hostData;
  }

  /**
   * Copy data from device to device
   * O(n) where n = size
   */
  async copyDeviceToDevice(srcId: string, dstId: string): Promise<void> {
    const srcBuffer = this.buffers.get(srcId);
    const dstBuffer = this.buffers.get(dstId);

    if (!srcBuffer) {
      throw new MemoryError(`Source buffer not found: ${srcId}`);
    }

    if (!dstBuffer) {
      throw new MemoryError(`Destination buffer not found: ${dstId}`);
    }

    if (srcBuffer.freed) {
      throw new MemoryError(`Source buffer already freed: ${srcId}`);
    }

    if (dstBuffer.freed) {
      throw new MemoryError(`Destination buffer already freed: ${dstId}`);
    }

    if (srcBuffer.size !== dstBuffer.size) {
      throw new MemoryError(
        `Size mismatch: src=${srcBuffer.size}, dst=${dstBuffer.size}`
      );
    }

    console.log(`   🔄 Copying ${formatSize(srcBuffer.size)} device-to-device...`);

    // Mock transfer (in reality: cudaMemcpy D2D)
    const srcMem = this.mockDeviceMemory.get(srcId);
    const dstMem = this.mockDeviceMemory.get(dstId);

    if (!srcMem || !dstMem) {
      throw new MemoryError('Device memory not found');
    }

    srcMem.copy(dstMem);

    // Simulate transfer time (faster than H2D/D2H: 100 GB/s on-device bandwidth)
    const transferTimeMs = (srcBuffer.size / (100 * 1024 * 1024 * 1024)) * 1000;
    await new Promise(resolve => setTimeout(resolve, transferTimeMs));

    console.log(`   ✅ Transfer complete (${transferTimeMs.toFixed(2)}ms)`);
  }

  /**
   * Get buffer by ID
   * O(1) lookup
   */
  getBuffer(id: string): MemoryBuffer | null {
    return this.buffers.get(id) || null;
  }

  /**
   * List all buffers
   */
  listBuffers(): MemoryBuffer[] {
    return Array.from(this.buffers.values());
  }

  /**
   * Get memory statistics
   */
  getStats(): MemoryStats {
    return { ...this.stats };
  }

  /**
   * Reset all memory (for testing)
   */
  reset(): void {
    this.buffers.clear();
    this.mockDeviceMemory.clear();

    this.stats = {
      totalAllocated: 0,
      totalFree: 0,
      currentUsage: 0,
      peakUsage: 0,
      allocationCount: 0,
      freeCount: 0,
    };
  }

  // ==========================================================================
  // Private Helpers
  // ==========================================================================

  /**
   * Generate unique buffer ID
   */
  private generateBufferId(): string {
    return crypto.randomBytes(8).toString('hex');
  }
}

// ============================================================================
// Utilities
// ============================================================================

export function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)}GB`;
}
