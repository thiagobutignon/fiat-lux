#!/usr/bin/env tsx

/**
 * Extract sample of layer 0 tensors (F32 only)
 */

import { analyzeGGUF } from '../../src/gguf-parser/presentation';
import { GGUFTensorReader } from '../../src/gguf-parser/domain/use-cases/tensor-reader';
import { WeightAnalyzer } from '../../src/gguf-parser/domain/use-cases/weight-analyzer';
import { GGMLType } from '../../src/gguf-parser/domain/entities/gguf-metadata';

async function main() {
  const filePath = 'landing/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf';

  console.log('🔍 Layer 0 Sample Extraction (F32 Tensors Only)\n');

  const { model } = await analyzeGGUF(filePath);

  const reader = new GGUFTensorReader(
    filePath,
    model.architecture.alignment || 32,
    model.tensorDataOffset || BigInt(0)
  );

  const analyzer = new WeightAnalyzer();

  // Get layer 0 F32 tensors
  const layer0Tensors = model.tensors.filter(t =>
    t.name.startsWith('blk.0.') && t.type === GGMLType.F32
  );

  console.log(`Found ${layer0Tensors.length} F32 tensors in layer 0\n`);
  console.log('='.repeat(80));

  for (const tensorInfo of layer0Tensors) {
    console.log(`\n📦 ${tensorInfo.name}`);
    console.log(`   Shape: ${tensorInfo.dimensions.join(' × ')}`);
    console.log(`   Elements: ${tensorInfo.dimensions.reduce((a, b) => a * b, 1).toLocaleString()}`);

    try {
      const tensor = await reader.readTensor(tensorInfo);
      const stats = analyzer.analyze(tensor);

      console.log(`   Mean: ${stats.mean.toFixed(6)}, Std: ${stats.stdDev.toFixed(6)}`);
      console.log(`   Range: [${stats.min.toFixed(6)}, ${stats.max.toFixed(6)}]`);
      console.log(`   Sparsity: ${(stats.sparsity * 100).toFixed(2)}%`);

      // First 5 values
      console.log(`   First 5 values: [${Array.from(tensor.data.slice(0, 5)).map(v => v.toFixed(4)).join(', ')}]`);

    } catch (error: any) {
      console.log(`   ❌ Error: ${error.message}`);
    }
  }

  console.log('\n' + '='.repeat(80));
  console.log('✅ Extraction complete!\n');

  await reader.close();
}

main();
