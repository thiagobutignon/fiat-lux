/**
 * Phase 1: Weight Pattern Analysis Demo
 *
 * Demonstrates extraction and analysis of weight patterns from Llama 3.1
 *
 * Usage:
 *   tsx src/research/llama-hallucination/demos/phase1-weight-analysis.ts <path-to-gguf-model>
 */

import { WeightExtractor } from '../analysis/weight-extractor';
import { WeightExtractionConfig } from '../domain/weight-statistics';
import { NodeFileReader } from '../../../gguf-parser/data/use-cases/node-file-reader';
import * as fs from 'fs';
import * as path from 'path';

async function runPhase1Analysis() {
  console.log('🔬 Llama 3.1 Hallucination Research - Phase 1: Weight Pattern Analysis\n');

  // Get model path from args or use default
  const modelPath = process.argv[2] || process.env.LLAMA_MODEL_PATH;

  if (!modelPath) {
    console.error('❌ Error: No model path provided');
    console.log('\nUsage:');
    console.log('  tsx src/research/llama-hallucination/demos/phase1-weight-analysis.ts <path-to-gguf>');
    console.log('\nOr set environment variable:');
    console.log('  export LLAMA_MODEL_PATH=/path/to/llama-3.1-8b.gguf');
    process.exit(1);
  }

  if (!fs.existsSync(modelPath)) {
    console.error(`❌ Error: Model file not found: ${modelPath}`);
    process.exit(1);
  }

  console.log(`📁 Model: ${path.basename(modelPath)}`);
  console.log(`📍 Path: ${modelPath}\n`);

  // Initialize extractor
  const fileReader = new NodeFileReader();
  const extractor = new WeightExtractor(fileReader);

  try {
    // Task 1.1: Extract and analyze all layer weights
    console.log('═══════════════════════════════════════════════════════════');
    console.log('Task 1.1: Extract and analyze all layer weights');
    console.log('═══════════════════════════════════════════════════════════\n');

    // Load model
    console.log('Step 1: Loading model...');
    await extractor.loadModel(modelPath);

    // Configure extraction
    const config: WeightExtractionConfig = {
      analyzeAttention: true,
      analyzeFFN: true,
      analyzeNorms: true,
      computeSpectralNorm: false, // Expensive, skip for now
      computeHeadSimilarity: false, // Will do in Task 1.2
      batchSize: 10,
      maxMemoryMB: 8192,
      dequantize: true,
      compareWithFP16: false,
    };

    console.log('\nStep 2: Extracting weight profile...');
    console.log('Configuration:');
    console.log(`  - Analyze Attention: ${config.analyzeAttention}`);
    console.log(`  - Analyze FFN: ${config.analyzeFFN}`);
    console.log(`  - Analyze Norms: ${config.analyzeNorms}`);
    console.log(`  - Dequantize: ${config.dequantize}`);

    const profile = await extractor.extractWeightProfile(config);

    // Display results
    console.log('\n═══════════════════════════════════════════════════════════');
    console.log('RESULTS: Weight Pattern Analysis');
    console.log('═══════════════════════════════════════════════════════════\n');

    console.log(`Model: ${profile.modelName}`);
    console.log(`Quantization: ${profile.quantizationType}`);
    console.log(`Total Parameters: ${profile.totalParameters.toLocaleString()}\n`);

    // Global statistics
    console.log('Global Statistics:');
    console.log(`  Mean Weight: ${profile.globalStatistics.meanWeight.toFixed(6)}`);
    console.log(`  Std Weight: ${profile.globalStatistics.stdWeight.toFixed(6)}`);
    console.log(`  Overall Sparsity: ${(profile.globalStatistics.overallSparsity * 100).toFixed(2)}%`);
    console.log(`  Total L1 Norm: ${profile.globalStatistics.totalL1Norm.toExponential(2)}`);
    console.log(`  Total L2 Norm: ${profile.globalStatistics.totalL2Norm.toExponential(2)}\n`);

    // Layer range comparisons
    console.log('Layer Range Comparison:');
    console.log('  Early Layers (0-10):');
    console.log(`    Mean: ${profile.earlyLayers.mean.toFixed(6)}, Std: ${profile.earlyLayers.std.toFixed(6)}`);
    console.log(`    Sparsity: ${(profile.earlyLayers.sparsity * 100).toFixed(2)}%`);

    console.log('  Middle Layers (11-21):');
    console.log(`    Mean: ${profile.middleLayers.mean.toFixed(6)}, Std: ${profile.middleLayers.std.toFixed(6)}`);
    console.log(`    Sparsity: ${(profile.middleLayers.sparsity * 100).toFixed(2)}%`);

    console.log('  Late Layers (22-31):');
    console.log(`    Mean: ${profile.lateLayers.mean.toFixed(6)}, Std: ${profile.lateLayers.std.toFixed(6)}`);
    console.log(`    Sparsity: ${(profile.lateLayers.sparsity * 100).toFixed(2)}%\n`);

    // Layer-by-layer summary
    console.log('Layer-by-Layer Summary:');
    console.log('  Layer | Type      | Parameters | Memory (MB) | L2 Norm');
    console.log('  ------|-----------|------------|-------------|----------');

    for (const layer of profile.layers.slice(0, 10)) {
      // Show first 10 layers
      const memoryMB = (layer.memoryFootprint / (1024 * 1024)).toFixed(1);
      let l2Norm = 0;

      if (layer.attention) {
        l2Norm = layer.attention.query.l2Norm;
      } else if (layer.ffn) {
        l2Norm = layer.ffn.gate.l2Norm;
      } else if (layer.norm) {
        l2Norm = layer.norm.attnNorm.l2Norm;
      }

      console.log(
        `  ${layer.layerIndex.toString().padStart(5)} | ${layer.layerType.padEnd(9)} | ${layer.totalParameters.toString().padStart(10)} | ${memoryMB.padStart(11)} | ${l2Norm.toExponential(2)}`
      );
    }
    console.log(`  ... (${profile.layers.length - 10} more layers)\n`);

    // Attention head summary (if analyzed)
    if (profile.attentionHeads.length > 0) {
      console.log(`Attention Heads Analyzed: ${profile.attentionHeads.length}`);
      console.log('  (See detailed analysis in output file)\n');
    }

    // FFN gate summary
    if (profile.ffnGates.length > 0) {
      console.log(`FFN Gates Analyzed: ${profile.ffnGates.length}`);
      console.log('  (See detailed analysis in output file)\n');
    }

    // Save detailed results to JSON
    const outputDir = 'research-output/phase1';
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    const outputPath = path.join(outputDir, `weight-profile-${Date.now()}.json`);
    fs.writeFileSync(outputPath, JSON.stringify(profile, null, 2));

    console.log(`✅ Detailed results saved to: ${outputPath}\n`);

    // Success metrics
    console.log('═══════════════════════════════════════════════════════════');
    console.log('SUCCESS METRICS');
    console.log('═══════════════════════════════════════════════════════════\n');

    console.log(`✅ Task 1.1: Extract and analyze all layer weights`);
    console.log(`   ✓ Complete weight profile for ${profile.totalParameters.toLocaleString()} parameters`);
    console.log(`   ✓ Analyzed ${profile.layers.length} layer components`);
    console.log(`   ✓ Statistical distributions computed (mean, std, sparsity, norms)`);
    console.log(`   ✓ Layer range comparisons complete (early vs middle vs late)`);

    console.log('\n🎉 Phase 1 - Task 1.1 Complete!\n');

    // Next steps
    console.log('Next Steps:');
    console.log('  [ ] Task 1.2: Attention head specialization analysis');
    console.log('  [ ] Task 1.3: FFN gate analysis');
    console.log('  [ ] Task 1.4: Layer norm scale investigation');
    console.log('  [ ] Generate visualizations');
    console.log('  [ ] Write Phase 1 findings document\n');
  } catch (error) {
    console.error('\n❌ Error during analysis:', error);
    if (error instanceof Error) {
      console.error(`   ${error.message}`);
      if (error.stack) {
        console.error('\nStack trace:');
        console.error(error.stack);
      }
    }
    process.exit(1);
  }
}

// Run analysis
runPhase1Analysis().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
