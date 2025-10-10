#!/usr/bin/env ts-node
/**
 * GCUDA Execution Engine Test
 *
 * Demonstrates kernel compilation, memory management, and execution
 */

import { DeviceManager } from '../../src/grammar-lang/tools/gcuda/device-manager';
import { KernelCompiler } from '../../src/grammar-lang/tools/gcuda/compiler';
import { GCUDAExecutor } from '../../src/grammar-lang/tools/gcuda/executor';
import * as path from 'path';

async function main() {
  console.log('🧪 GCUDA Execution Engine Test\n');

  // ===========================================================================
  // Step 1: Initialize Device
  // ===========================================================================

  console.log('📝 Step 1: Initialize GPU device\n');

  const deviceManager = new DeviceManager();
  const devices = await deviceManager.listDevices();

  if (devices.length === 0) {
    console.error('❌ No GPU devices found');
    process.exit(1);
  }

  const device = devices[0];
  console.log(`   Using: ${device.name}\n`);

  // ===========================================================================
  // Step 2: Compile Kernel
  // ===========================================================================

  console.log('📝 Step 2: Compile GPU kernel\n');

  const compiler = new KernelCompiler();
  const kernelPath = path.join(__dirname, 'kernels', 'vecadd.cu');

  const kernel = await compiler.compileFromFile(kernelPath, {
    arch: ['sm_80'],
    optimization: 'O3',
  });

  console.log(`   Kernel: ${kernel.name}`);
  console.log(`   Hash: ${kernel.hash.substring(0, 16)}...\n`);

  // ===========================================================================
  // Step 3: Create Execution Context
  // ===========================================================================

  console.log('📝 Step 3: Create execution context\n');

  const executor = new GCUDAExecutor();
  const context = executor.createContext(device);

  // Register kernel
  context.registerKernel(kernel);
  console.log('');

  // ===========================================================================
  // Step 4: Allocate Memory
  // ===========================================================================

  console.log('📝 Step 4: Allocate GPU memory\n');

  const memory = context.getMemoryManager();
  const N = 1024 * 1024; // 1M elements
  const bytesPerElement = 4; // float32
  const totalBytes = N * bytesPerElement;

  // Allocate device buffers
  const bufferA = memory.allocate(totalBytes, 'device');
  const bufferB = memory.allocate(totalBytes, 'device');
  const bufferC = memory.allocate(totalBytes, 'device');

  console.log('');

  // ===========================================================================
  // Step 5: Prepare Data
  // ===========================================================================

  console.log('📝 Step 5: Prepare input data\n');

  // Create host buffers with test data
  const hostA = Buffer.alloc(totalBytes);
  const hostB = Buffer.alloc(totalBytes);

  // Fill with test values (as float32)
  for (let i = 0; i < N; i++) {
    hostA.writeFloatLE(i, i * 4);
    hostB.writeFloatLE(i * 2, i * 4);
  }

  console.log(`   Created ${N} elements\n`);

  // ===========================================================================
  // Step 6: Transfer to Device
  // ===========================================================================

  console.log('📝 Step 6: Transfer data to GPU\n');

  await memory.copyToDevice(bufferA.id, hostA);
  await memory.copyToDevice(bufferB.id, hostB);

  console.log('');

  // ===========================================================================
  // Step 7: Launch Kernel
  // ===========================================================================

  console.log('📝 Step 7: Launch kernel\n');

  const threadsPerBlock = 256;
  const numBlocks = Math.ceil(N / threadsPerBlock);

  const execution = await context.launchKernel(
    kernel.hash,
    [bufferA, bufferB, bufferC],
    {
      gridDim: { x: numBlocks, y: 1, z: 1 },
      blockDim: { x: threadsPerBlock, y: 1, z: 1 },
    }
  );

  console.log(`   Execution ID: ${execution.id}`);
  console.log(`   Status: ${execution.status}`);
  console.log(`   Time: ${execution.executionTime}ms\n`);

  // ===========================================================================
  // Step 8: Retrieve Results
  // ===========================================================================

  console.log('📝 Step 8: Retrieve results from GPU\n');

  const hostC = await memory.copyFromDevice(bufferC.id);

  // Verify first few results
  let correct = 0;
  let incorrect = 0;

  for (let i = 0; i < Math.min(N, 10); i++) {
    const a = hostA.readFloatLE(i * 4);
    const b = hostB.readFloatLE(i * 4);
    const c = hostC.readFloatLE(i * 4);
    const expected = a + b;

    if (Math.abs(c - expected) < 0.001) {
      correct++;
    } else {
      incorrect++;
      console.log(`   ❌ Mismatch at ${i}: ${c} !== ${expected}`);
    }
  }

  console.log(`   Verified ${correct + incorrect} elements`);
  console.log(`   Correct: ${correct}`);
  console.log(`   Incorrect: ${incorrect}\n`);

  // ===========================================================================
  // Step 9: Statistics
  // ===========================================================================

  console.log('📝 Step 9: Execution statistics\n');

  const stats = context.getStats();
  console.log(`   Total kernels launched: ${stats.totalKernelsLaunched}`);
  console.log(`   Total execution time: ${stats.totalExecutionTime.toFixed(2)}ms`);
  console.log(`   Average execution time: ${stats.averageExecutionTime.toFixed(2)}ms`);
  console.log(`   Failed kernels: ${stats.failedKernels}\n`);

  const memoryStats = memory.getStats();
  console.log(`   Memory allocated: ${formatSize(memoryStats.totalAllocated)}`);
  console.log(`   Peak usage: ${formatSize(memoryStats.peakUsage)}\n`);

  // ===========================================================================
  // Step 10: Cleanup
  // ===========================================================================

  console.log('📝 Step 10: Cleanup\n');

  memory.free(bufferA.id);
  memory.free(bufferB.id);
  memory.free(bufferC.id);

  context.destroy();

  console.log('   ✅ Resources freed\n');

  // ===========================================================================
  // Summary
  // ===========================================================================

  console.log('✅ All tests completed!\n');

  console.log('📊 Summary:');
  console.log(`   Device: ${device.name}`);
  console.log(`   Kernel: ${kernel.name}`);
  console.log(`   Elements: ${N.toLocaleString()}`);
  console.log(`   Blocks: ${numBlocks}`);
  console.log(`   Threads/block: ${threadsPerBlock}`);
  console.log(`   Execution time: ${execution.executionTime}ms`);
  console.log(`   Throughput: ${(N / (execution.executionTime / 1000) / 1e9).toFixed(2)} GFLOPS`);
  console.log(`   Correctness: ${correct}/${correct + incorrect} verified`);
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)}GB`;
}

main().catch(error => {
  console.error('❌ Error:', error.message);
  process.exit(1);
});
