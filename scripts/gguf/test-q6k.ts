#!/usr/bin/env tsx

/**
 * Quick test for Q6_K dequantization
 */

import { analyzeGGUF } from '../../src/gguf-parser/presentation';
import { GGUFTensorReader } from '../../src/gguf-parser/domain/use-cases/tensor-reader';
import { WeightAnalyzer } from '../../src/gguf-parser/domain/use-cases/weight-analyzer';
import { GGMLType } from '../../src/gguf-parser/domain/entities/gguf-metadata';

async function main() {
  const filePath = 'landing/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf';

  console.log('🧪 Q6_K Dequantization Test\n');

  const { model } = await analyzeGGUF(filePath);

  // Find first Q6_K tensor
  const q6kTensor = model.tensors.find(t => t.type === GGMLType.Q6_K);

  if (!q6kTensor) {
    console.log('❌ No Q6_K tensors found');
    return;
  }

  console.log(`📦 Testing: ${q6kTensor.name}`);
  console.log(`   Shape: ${q6kTensor.dimensions.join(' × ')}`);
  console.log(`   Elements: ${q6kTensor.dimensions.reduce((a, b) => a * b, 1).toLocaleString()}\n`);

  const reader = new GGUFTensorReader(
    filePath,
    model.architecture.alignment || 32,
    model.tensorDataOffset || BigInt(0)
  );

  const analyzer = new WeightAnalyzer();

  try {
    console.log('⏳ Reading tensor data...');
    const tensorData = await reader.readTensor(q6kTensor);
    console.log('✅ Tensor loaded!\n');

    // Check for NaN/Infinity
    let nanCount = 0;
    let infCount = 0;
    for (let i = 0; i < tensorData.data.length; i++) {
      if (isNaN(tensorData.data[i])) nanCount++;
      if (!isFinite(tensorData.data[i])) infCount++;
    }

    if (nanCount > 0 || infCount > 0) {
      console.log(`❌ Found ${nanCount} NaN and ${infCount} Infinity values`);
      console.log(`   First 20 values: [${Array.from(tensorData.data.slice(0, 20)).map(v => v.toFixed(6)).join(', ')}]`);
    } else {
      console.log('✅ No NaN/Infinity values detected!');

      const stats = analyzer.analyze(tensorData);
      console.log('\n📊 STATISTICS');
      console.log('─'.repeat(80));
      console.log(analyzer.formatStats(stats));

      console.log('\n🔬 SAMPLE VALUES (first 20)');
      console.log('─'.repeat(80));
      for (let i = 0; i < 20; i++) {
        console.log(`  [${i}] = ${tensorData.data[i].toFixed(6)}`);
      }
    }

    await reader.close();

  } catch (error: any) {
    console.error(`\n❌ Error: ${error.message}`);
    if (error.stack) {
      console.error(error.stack);
    }
  }
}

main();
