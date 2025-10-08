#!/usr/bin/env tsx

/**
 * Extract and analyze a specific F32 tensor
 */

import { analyzeGGUF } from '../../src/gguf-parser/presentation';
import { GGUFTensorReader } from '../../src/gguf-parser/domain/use-cases/tensor-reader';
import { WeightAnalyzer } from '../../src/gguf-parser/domain/use-cases/weight-analyzer';
import { GGMLType } from '../../src/gguf-parser/domain/entities/gguf-metadata';

async function main() {
  const filePath = process.argv[2] || 'landing/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf';

  console.log('🔍 F32 Tensor Extractor\n');
  console.log(`📄 Model: ${filePath}\n`);

  // Parse model
  const { model } = await analyzeGGUF(filePath);

  // Find F32 tensors
  const f32Tensors = model.tensors.filter(t => t.type === GGMLType.F32);

  console.log(`Found ${f32Tensors.length} F32 tensors\n`);
  console.log('First 10 F32 tensors:');
  for (let i = 0; i < Math.min(10, f32Tensors.length); i++) {
    const t = f32Tensors[i];
    console.log(`  ${i + 1}. ${t.name} [${t.dimensions.join(' × ')}]`);
  }

  // Extract first attention norm (layer 0)
  const targetTensor = f32Tensors.find(t => t.name === 'blk.0.attn_norm.weight')!;

  console.log(`\n${'='.repeat(80)}`);
  console.log(`📦 Extracting: ${targetTensor.name}`);
  console.log('='.repeat(80));

  const reader = new GGUFTensorReader(
    filePath,
    model.architecture.alignment || 32,
    model.tensorDataOffset || BigInt(0)
  );

  const analyzer = new WeightAnalyzer();

  try {
    console.log('\n⏳ Reading tensor data...');
    const tensorData = await reader.readTensor(targetTensor);

    console.log('✅ Tensor loaded successfully!\n');

    console.log('📊 TENSOR INFO');
    console.log('─'.repeat(80));
    console.log(`Name:      ${tensorData.name}`);
    console.log(`Shape:     ${tensorData.shape.join(' × ')}`);
    console.log(`Type:      ${GGMLType[tensorData.type]}`);
    console.log(`Elements:  ${tensorData.data.length.toLocaleString()}`);
    console.log(`Memory:    ${(tensorData.data.length * 4 / 1024).toFixed(2)} KB`);

    console.log('\n📈 STATISTICS');
    console.log('─'.repeat(80));
    const stats = analyzer.analyze(tensorData);
    console.log(analyzer.formatStats(stats));

    console.log('📊 MAGNITUDE DISTRIBUTION');
    console.log('─'.repeat(80));
    const magDist = analyzer.getMagnitudeDistribution(tensorData);
    console.log(`  P50 (median): ${magDist.percentiles.p50.toFixed(6)}`);
    console.log(`  P90:          ${magDist.percentiles.p90.toFixed(6)}`);
    console.log(`  P95:          ${magDist.percentiles.p95.toFixed(6)}`);
    console.log(`  P99:          ${magDist.percentiles.p99.toFixed(6)}`);

    console.log(`\n  Top 10 magnitudes:`);
    for (let i = 0; i < 10; i++) {
      console.log(`    ${i + 1}. ${magDist.topKMagnitudes[i].toFixed(6)}`);
    }

    console.log('\n🔬 SAMPLE VALUES (first 20)');
    console.log('─'.repeat(80));
    for (let i = 0; i < 20; i++) {
      console.log(`  [${i}] = ${tensorData.data[i].toFixed(6)}`);
    }

    console.log('\n📉 HISTOGRAM (50 bins)');
    console.log('─'.repeat(80));

    // Find max count for scaling
    const maxCount = Math.max(...stats.histogram.counts);
    const barWidth = 40;

    for (let i = 0; i < stats.histogram.bins.length; i += 5) { // Show every 5th bin
      const bin = stats.histogram.bins[i];
      const count = stats.histogram.counts[i];
      const barLength = Math.round((count / maxCount) * barWidth);
      const bar = '█'.repeat(barLength);

      console.log(`  ${bin.toFixed(3)} │ ${bar} ${count}`);
    }

    console.log('\n✅ Extraction complete!');

    await reader.close();

  } catch (error: any) {
    console.error(`\n❌ Error: ${error.message}`);
    if (error.stack) {
      console.error(error.stack);
    }
    process.exit(1);
  }
}

main();
