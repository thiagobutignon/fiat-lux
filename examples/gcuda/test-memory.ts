#!/usr/bin/env ts-node
/**
 * GCUDA Memory Manager Test
 *
 * Demonstrates memory allocation and transfer operations
 */

import { DeviceManager } from '../../src/grammar-lang/tools/gcuda/device-manager';
import { MemoryManager, formatSize } from '../../src/grammar-lang/tools/gcuda/memory';

async function main() {
  console.log('🧪 GCUDA Memory Manager Test\n');

  // Get GPU device
  const deviceManager = new DeviceManager();
  const devices = await deviceManager.listDevices();

  if (devices.length === 0) {
    console.error('No GPU devices found');
    process.exit(1);
  }

  const device = devices[0];
  console.log(`Using device: ${device.name}\n`);

  // Create memory manager
  const memory = new MemoryManager(device);

  // Test 1: Allocate device memory
  console.log('📝 Test 1: Allocate device memory\n');

  const bufferA = memory.allocate(1024 * 1024, 'device'); // 1 MB
  const bufferB = memory.allocate(2 * 1024 * 1024, 'device'); // 2 MB
  const bufferC = memory.allocate(512 * 1024, 'device'); // 512 KB

  console.log('');

  // Test 2: Host-to-Device transfer
  console.log('📝 Test 2: Host-to-Device transfer\n');

  const hostData = Buffer.alloc(1024 * 1024);
  // Fill with test data
  for (let i = 0; i < hostData.length; i++) {
    hostData[i] = i % 256;
  }

  await memory.copyToDevice(bufferA.id, hostData);

  console.log('');

  // Test 3: Device-to-Host transfer
  console.log('📝 Test 3: Device-to-Host transfer\n');

  const deviceData = await memory.copyFromDevice(bufferA.id);
  console.log(`   Received ${formatSize(deviceData.length)} from device`);

  // Verify data integrity
  const isMatch = hostData.equals(deviceData);
  console.log(`   Data integrity: ${isMatch ? '✅ PASS' : '❌ FAIL'}`);

  console.log('');

  // Test 4: Device-to-Device transfer
  console.log('📝 Test 4: Device-to-Device transfer\n');

  // Allocate another buffer same size as bufferA
  const bufferD = memory.allocate(1024 * 1024, 'device');
  console.log('');

  await memory.copyDeviceToDevice(bufferA.id, bufferD.id);

  console.log('');

  // Test 5: Memory stats
  console.log('📝 Test 5: Memory statistics\n');

  const stats = memory.getStats();
  console.log(`   Total Allocated: ${formatSize(stats.totalAllocated)}`);
  console.log(`   Total Freed:     ${formatSize(stats.totalFree)}`);
  console.log(`   Current Usage:   ${formatSize(stats.currentUsage)}`);
  console.log(`   Peak Usage:      ${formatSize(stats.peakUsage)}`);
  console.log(`   Allocations:     ${stats.allocationCount}`);
  console.log(`   Frees:           ${stats.freeCount}`);

  console.log('');

  // Test 6: Free memory
  console.log('📝 Test 6: Free memory\n');

  memory.free(bufferA.id);
  memory.free(bufferB.id);
  memory.free(bufferC.id);
  memory.free(bufferD.id);

  console.log('');

  // Final stats
  const finalStats = memory.getStats();
  console.log('📊 Final Statistics\n');
  console.log(`   Total Allocated: ${formatSize(finalStats.totalAllocated)}`);
  console.log(`   Total Freed:     ${formatSize(finalStats.totalFree)}`);
  console.log(`   Current Usage:   ${formatSize(finalStats.currentUsage)}`);
  console.log(`   Peak Usage:      ${formatSize(finalStats.peakUsage)}`);
  console.log(`   Allocations:     ${finalStats.allocationCount}`);
  console.log(`   Frees:           ${finalStats.freeCount}`);

  console.log('');
  console.log('✅ All tests passed!');
}

main().catch(error => {
  console.error('Error:', error.message);
  process.exit(1);
});
