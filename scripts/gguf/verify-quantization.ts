#!/usr/bin/env tsx

/**
 * Verify all quantization types work correctly
 */

import { analyzeGGUF } from '../../src/gguf-parser/presentation';
import { GGUFTensorReader } from '../../src/gguf-parser/domain/use-cases/tensor-reader';
import { GGMLType } from '../../src/gguf-parser/domain/entities/gguf-metadata';

async function main() {
  const filePath = 'landing/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf';

  console.log('🧪 Quantization Type Verification\n');

  const { model } = await analyzeGGUF(filePath);

  const reader = new GGUFTensorReader(
    filePath,
    model.architecture.alignment || 32,
    model.tensorDataOffset || BigInt(0)
  );

  // Test one tensor of each type
  const typesToTest = [
    { type: GGMLType.F32, name: 'F32' },
    { type: GGMLType.Q4_K, name: 'Q4_K' },
    { type: GGMLType.Q6_K, name: 'Q6_K' },
  ];

  for (const { type, name } of typesToTest) {
    const tensor = model.tensors.find(t => t.type === type);

    if (!tensor) {
      console.log(`⚠️  ${name}: No tensors found`);
      continue;
    }

    try {
      console.log(`\n📦 ${name}: ${tensor.name}`);
      console.log(`   Shape: ${tensor.dimensions.join(' × ')}`);

      const data = await reader.readTensor(tensor);

      // Check for issues
      let nanCount = 0;
      let infCount = 0;
      let sum = 0;
      let sumSq = 0;

      for (let i = 0; i < data.data.length; i++) {
        const val = data.data[i];
        if (isNaN(val)) nanCount++;
        if (!isFinite(val)) infCount++;
        sum += val;
        sumSq += val * val;
      }

      const mean = sum / data.data.length;
      const variance = (sumSq / data.data.length) - (mean * mean);
      const stdDev = Math.sqrt(Math.max(0, variance));

      if (nanCount > 0 || infCount > 0) {
        console.log(`   ❌ FAILED: ${nanCount} NaN, ${infCount} Infinity values`);
      } else {
        console.log(`   ✅ PASSED`);
        console.log(`   Mean: ${mean.toFixed(6)}, Std: ${stdDev.toFixed(6)}`);
        console.log(`   Range: [${Math.min(...data.data).toFixed(6)}, ${Math.max(...data.data).toFixed(6)}]`);
        console.log(`   Sample: [${Array.from(data.data.slice(0, 5)).map(v => v.toFixed(4)).join(', ')}]`);
      }

    } catch (error: any) {
      console.log(`   ❌ ERROR: ${error.message}`);
    }
  }

  await reader.close();

  console.log('\n' + '='.repeat(80));
  console.log('✅ Verification complete!\n');
}

main();
