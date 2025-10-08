#!/usr/bin/env tsx

/**
 * CLI Tool: GGUF Weight Extractor
 * Extracts and analyzes tensor weights from GGUF model files
 */

import { analyzeGGUF } from '../../src/gguf-parser/presentation';
import { GGUFTensorReader } from '../../src/gguf-parser/domain/use-cases/tensor-reader';
import { WeightAnalyzer } from '../../src/gguf-parser/domain/use-cases/weight-analyzer';
import { resolve } from 'path';
import { writeFile } from 'fs/promises';

async function main() {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    console.error('Usage: tsx scripts/gguf/extract-weights.ts <path-to-gguf-file> [options]');
    console.error('\nOptions:');
    console.error('  --layer N         Extract specific layer (default: 0)');
    console.error('  --all-layers      Extract all layers');
    console.error('  --embeddings      Extract embedding table');
    console.error('  --output FILE     Save to JSON file');
    console.error('  --stats-only      Show statistics only (no data export)');
    console.error('\nExamples:');
    console.error('  tsx scripts/gguf/extract-weights.ts model.gguf --layer 0');
    console.error('  tsx scripts/gguf/extract-weights.ts model.gguf --embeddings --output emb.json');
    console.error('  tsx scripts/gguf/extract-weights.ts model.gguf --all-layers --stats-only');
    process.exit(1);
  }

  const filePath = resolve(args[0]);
  const layerIndex = args.includes('--layer')
    ? parseInt(args[args.indexOf('--layer') + 1], 10)
    : 0;
  const allLayers = args.includes('--all-layers');
  const embeddings = args.includes('--embeddings');
  const statsOnly = args.includes('--stats-only');
  const outputFile = args.includes('--output')
    ? args[args.indexOf('--output') + 1]
    : null;

  console.log('🔍 GGUF Weight Extractor\n');
  console.log(`📄 Loading: ${filePath}\n`);

  try {
    const startTime = Date.now();

    // Step 1: Analyze model metadata
    console.log('Step 1: Analyzing model metadata...');
    const { model, analysis } = await analyzeGGUF(filePath);
    console.log(`✅ Found ${model.tensors.length} tensors\n`);

    // Step 2: Initialize tensor reader
    console.log('Step 2: Initializing tensor reader...');

    // Get tensor data offset from model
    const tensorDataOffset = model.tensorDataOffset || BigInt(0);
    const alignment = model.architecture.alignment || 32;

    console.log(`   Tensor data offset: ${tensorDataOffset}`);
    console.log(`   Alignment: ${alignment} bytes`);

    const reader = new GGUFTensorReader(filePath, alignment, tensorDataOffset);
    const analyzer = new WeightAnalyzer();
    console.log('✅ Reader initialized\n');

    // Step 3: Extract weights based on options
    if (embeddings) {
      console.log('Step 3: Extracting embedding table...');
      const embTensor = await reader.readEmbeddings(model.tensors);

      if (embTensor) {
        console.log(`✅ Extracted embeddings: ${embTensor.shape.join(' × ')}\n`);

        const stats = analyzer.analyze(embTensor);
        console.log('📊 Embedding Statistics:');
        console.log(analyzer.formatStats(stats));

        if (outputFile && !statsOnly) {
          await writeFile(
            outputFile,
            JSON.stringify({
              name: embTensor.name,
              shape: embTensor.shape,
              type: embTensor.type,
              data: Array.from(embTensor.data),
              statistics: stats,
            }, null, 2)
          );
          console.log(`\n💾 Saved to: ${outputFile}`);
        }
      } else {
        console.log('❌ Embedding table not found');
      }
    } else if (allLayers) {
      console.log(`Step 3: Extracting all ${analysis.layers} layers...\n`);

      for (let i = 0; i < analysis.layers; i++) {
        console.log(`\n${'='.repeat(80)}`);
        console.log(`📦 LAYER ${i}`);
        console.log('='.repeat(80));

        const layerWeights = await reader.readLayer(model.tensors, i);

        // Analyze each tensor in layer
        const tensorNames = [
          'attentionNorm',
          'attentionQ',
          'attentionK',
          'attentionV',
          'attentionOutput',
          'ffnNorm',
          'ffnGate',
          'ffnUp',
          'ffnDown',
        ] as const;

        for (const tensorName of tensorNames) {
          const tensor = layerWeights[tensorName];
          if (tensor) {
            console.log(`\n🔸 ${tensorName}`);
            console.log(`   Shape: ${tensor.shape.join(' × ')}`);
            console.log(`   Elements: ${tensor.data.length.toLocaleString()}`);

            const stats = analyzer.analyze(tensor);
            console.log(`   Mean: ${stats.mean.toFixed(6)}, Std: ${stats.stdDev.toFixed(6)}`);
            console.log(`   Range: [${stats.min.toFixed(6)}, ${stats.max.toFixed(6)}]`);
            console.log(`   Sparsity: ${(stats.sparsity * 100).toFixed(2)}%`);
          }
        }

        // Small delay to avoid overwhelming output
        if (i < analysis.layers - 1) {
          await new Promise(resolve => setTimeout(resolve, 100));
        }
      }
    } else {
      // Extract single layer
      console.log(`Step 3: Extracting layer ${layerIndex}...\n`);
      const layerWeights = await reader.readLayer(model.tensors, layerIndex);

      console.log(`${'='.repeat(80)}`);
      console.log(`📦 LAYER ${layerIndex} - DETAILED ANALYSIS`);
      console.log('='.repeat(80));

      const results: any = {
        layer: layerIndex,
        tensors: {},
      };

      // Analyze each tensor
      const tensorNames = [
        'attentionNorm',
        'attentionQ',
        'attentionK',
        'attentionV',
        'attentionOutput',
        'ffnNorm',
        'ffnGate',
        'ffnUp',
        'ffnDown',
      ] as const;

      for (const tensorName of tensorNames) {
        const tensor = layerWeights[tensorName];
        if (tensor) {
          console.log(`\n🔸 ${tensorName.toUpperCase()}`);
          console.log(`${'─'.repeat(80)}`);
          console.log(`Name:      ${tensor.name}`);
          console.log(`Shape:     ${tensor.shape.join(' × ')}`);
          console.log(`Type:      ${tensor.type}`);
          console.log(`Elements:  ${tensor.data.length.toLocaleString()}\n`);

          const stats = analyzer.analyze(tensor);
          console.log(analyzer.formatStats(stats));

          // Magnitude distribution
          const magDist = analyzer.getMagnitudeDistribution(tensor);
          console.log(`  Magnitude Percentiles:`);
          console.log(`    P50: ${magDist.percentiles.p50.toExponential(3)}`);
          console.log(`    P90: ${magDist.percentiles.p90.toExponential(3)}`);
          console.log(`    P95: ${magDist.percentiles.p95.toExponential(3)}`);
          console.log(`    P99: ${magDist.percentiles.p99.toExponential(3)}\n`);

          if (!statsOnly) {
            results.tensors[tensorName] = {
              name: tensor.name,
              shape: tensor.shape,
              type: tensor.type,
              statistics: stats,
              magnitudeDistribution: magDist,
              data: Array.from(tensor.data.slice(0, 1000)), // First 1000 values only
            };
          }
        }
      }

      if (outputFile && !statsOnly) {
        await writeFile(outputFile, JSON.stringify(results, null, 2));
        console.log(`\n💾 Saved to: ${outputFile}`);
      }
    }

    // Close file handle
    await reader.close();

    const elapsedTime = Date.now() - startTime;
    console.log(`\n${'='.repeat(80)}`);
    console.log(`✅ Extraction complete!`);
    console.log(`⚡ Total time: ${elapsedTime}ms`);
    console.log('='.repeat(80));

  } catch (error: any) {
    console.error(`\n❌ Error extracting weights: ${error.message}\n`);
    if (error.stack) {
      console.error(error.stack);
    }
    process.exit(1);
  }
}

main();
