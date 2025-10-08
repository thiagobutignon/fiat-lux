/**
 * Verification script for ceiling division fix
 * Tests that block size calculations work correctly for edge cases
 */

import { dequantizeQ4_K, dequantizeQ6_K, dequantizeQ4_0 } from '../domain/use-cases/dequantize';

console.log('🧪 Verifying Ceiling Division Fix\n');

// Test 1: Q4_K with exactly 256 elements (1 block)
console.log('Test 1: Q4_K with 256 elements (exactly 1 block)');
try {
  const buffer1 = Buffer.alloc(144); // 1 block = 144 bytes
  const result1 = dequantizeQ4_K(buffer1, 256);
  console.log(`  ✅ Success: ${result1.length} elements\n`);
} catch (e: any) {
  console.log(`  ❌ Failed: ${e.message}\n`);
}

// Test 2: Q4_K with 257 elements (needs 2 blocks)
console.log('Test 2: Q4_K with 257 elements (needs 2 blocks)');
try {
  const buffer2 = Buffer.alloc(288); // 2 blocks = 288 bytes
  const result2 = dequantizeQ4_K(buffer2, 257);
  console.log(`  ✅ Success: ${result2.length} elements\n`);
} catch (e: any) {
  console.log(`  ❌ Failed: ${e.message}\n`);
}

// Test 3: Q4_K with 300 elements (needs 2 blocks)
console.log('Test 3: Q4_K with 300 elements (needs 2 blocks)');
try {
  const buffer3 = Buffer.alloc(288); // 2 blocks = 288 bytes
  const result3 = dequantizeQ4_K(buffer3, 300);
  console.log(`  ✅ Success: ${result3.length} elements\n`);
} catch (e: any) {
  console.log(`  ❌ Failed: ${e.message}\n`);
}

// Test 4: Q4_K with 300 elements but insufficient buffer (should fail)
console.log('Test 4: Q4_K with 300 elements but only 144 bytes (should FAIL)');
try {
  const buffer4 = Buffer.alloc(144); // Only 1 block
  const result4 = dequantizeQ4_K(buffer4, 300);
  console.log(`  ❌ Should have failed but didn't!\n`);
} catch (e: any) {
  console.log(`  ✅ Correctly rejected: ${e.message}\n`);
}

// Test 5: Q6_K with 300 elements (needs 2 blocks)
console.log('Test 5: Q6_K with 300 elements (needs 2 blocks)');
try {
  const buffer5 = Buffer.alloc(420); // 2 blocks = 420 bytes
  const result5 = dequantizeQ6_K(buffer5, 300);
  console.log(`  ✅ Success: ${result5.length} elements\n`);
} catch (e: any) {
  console.log(`  ❌ Failed: ${e.message}\n`);
}

// Test 6: Q4_0 with 50 elements (needs 2 blocks)
console.log('Test 6: Q4_0 with 50 elements (needs 2 blocks)');
try {
  const buffer6 = Buffer.alloc(36); // 2 blocks = 36 bytes
  buffer6.writeUInt16LE(0x3C00, 0);  // scale = 1.0 for block 1
  buffer6.writeUInt16LE(0x3C00, 18); // scale = 1.0 for block 2
  const result6 = dequantizeQ4_0(buffer6, 50);
  console.log(`  ✅ Success: ${result6.length} elements\n`);
} catch (e: any) {
  console.log(`  ❌ Failed: ${e.message}\n`);
}

console.log('✅ Ceiling division fix verification complete!');
