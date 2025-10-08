/**
 * Phase 1 - Task 1.4: Layer Norm Investigation
 *
 * Analyzes:
 * - Layer norm scale parameters across layers
 * - Attention norm vs FFN norm patterns
 * - Correlation with attention/FFN strength
 * - Potential amplification/suppression trends
 *
 * Usage:
 *   tsx src/research/llama-hallucination/demos/phase1-task1.4-layernorm-analysis.ts <weight-profile.json>
 */

import * as fs from 'fs';
import * as path from 'path';

interface WeightStats {
  mean: number;
  std: number;
  min: number;
  max: number;
  skewness: number;
  kurtosis: number;
  sparsity: number;
  l1Norm: number;
  l2Norm: number;
}

interface NormWeights {
  attnNorm: WeightStats;
  ffnNorm: WeightStats;
}

interface FFNWeights {
  gate: WeightStats;
  up: WeightStats;
  down: WeightStats;
}

interface AttentionWeights {
  query: WeightStats;
  key: WeightStats;
  value: WeightStats;
  output: WeightStats;
}

interface LayerProfile {
  layerIndex: number;
  layerType: string;
  norm?: NormWeights;
  ffn?: FFNWeights;
  attention?: AttentionWeights;
}

interface WeightProfile {
  modelName: string;
  quantizationType: string;
  totalParameters: number;
  layers: LayerProfile[];
}

// Extract layer norm statistics
function extractLayerNormStats(layers: LayerProfile[]): {
  attnNorms: Array<{ layer: number; stats: WeightStats }>;
  ffnNorms: Array<{ layer: number; stats: WeightStats }>;
} {
  const attnNorms: Array<{ layer: number; stats: WeightStats }> = [];
  const ffnNorms: Array<{ layer: number; stats: WeightStats }> = [];

  for (const layer of layers) {
    if (layer.norm) {
      attnNorms.push({ layer: layer.layerIndex, stats: layer.norm.attnNorm });
      ffnNorms.push({ layer: layer.layerIndex, stats: layer.norm.ffnNorm });
    }
  }

  return { attnNorms, ffnNorms };
}

// Compute correlation between norm scales and component strength
function computeNormCorrelations(
  layers: LayerProfile[]
): {
  attnNormVsQueryStrength: number;
  ffnNormVsGateStrength: number;
} {
  const normLayers = layers.filter((l) => l.norm && l.attention && l.ffn);

  if (normLayers.length === 0) {
    return { attnNormVsQueryStrength: 0, ffnNormVsGateStrength: 0 };
  }

  // Extract paired values
  const attnNormScales = normLayers.map((l) => l.norm!.attnNorm.mean);
  const queryStrengths = normLayers.map((l) => l.attention!.query.l2Norm);
  const ffnNormScales = normLayers.map((l) => l.norm!.ffnNorm.mean);
  const gateStrengths = normLayers.map((l) => l.ffn!.gate.l2Norm);

  // Compute Pearson correlation
  const pearsonCorr = (x: number[], y: number[]): number => {
    const n = x.length;
    const meanX = x.reduce((a, b) => a + b, 0) / n;
    const meanY = y.reduce((a, b) => a + b, 0) / n;

    let numerator = 0;
    let denomX = 0;
    let denomY = 0;

    for (let i = 0; i < n; i++) {
      const dx = x[i] - meanX;
      const dy = y[i] - meanY;
      numerator += dx * dy;
      denomX += dx * dx;
      denomY += dy * dy;
    }

    return numerator / Math.sqrt(denomX * denomY);
  };

  return {
    attnNormVsQueryStrength: pearsonCorr(attnNormScales, queryStrengths),
    ffnNormVsGateStrength: pearsonCorr(ffnNormScales, gateStrengths),
  };
}

// Identify layers with extreme norm scales
function findExtremeNormLayers(
  norms: Array<{ layer: number; stats: WeightStats }>,
  threshold: number
): {
  highScale: number[];
  lowScale: number[];
} {
  const means = norms.map((n) => n.stats.mean);
  const avgMean = means.reduce((a, b) => a + b, 0) / means.length;
  const stdMean = Math.sqrt(means.reduce((sum, m) => sum + (m - avgMean) ** 2, 0) / means.length);

  const highScale: number[] = [];
  const lowScale: number[] = [];

  for (const norm of norms) {
    const zScore = (norm.stats.mean - avgMean) / stdMean;
    if (zScore > threshold) {
      highScale.push(norm.layer);
    } else if (zScore < -threshold) {
      lowScale.push(norm.layer);
    }
  }

  return { highScale, lowScale };
}

// Analyze norm scale trends across layer ranges
function analyzeNormTrends(norms: Array<{ layer: number; stats: WeightStats }>): {
  earlyMean: number;
  middleMean: number;
  lateMean: number;
  trend: string;
} {
  const third = Math.floor(norms.length / 3);

  const earlyNorms = norms.slice(0, third);
  const middleNorms = norms.slice(third, third * 2);
  const lateNorms = norms.slice(third * 2);

  const earlyMean = earlyNorms.reduce((sum, n) => sum + n.stats.mean, 0) / earlyNorms.length;
  const middleMean = middleNorms.reduce((sum, n) => sum + n.stats.mean, 0) / middleNorms.length;
  const lateMean = lateNorms.reduce((sum, n) => sum + n.stats.mean, 0) / lateNorms.length;

  // Determine trend
  let trend = 'Stable';
  if (lateMean > earlyMean * 1.1) {
    trend = 'Amplifying (late layers boost signals)';
  } else if (lateMean < earlyMean * 0.9) {
    trend = 'Dampening (late layers suppress signals)';
  } else if (middleMean > earlyMean * 1.05 && middleMean > lateMean * 1.05) {
    trend = 'Peak in middle (mid-layer emphasis)';
  }

  return { earlyMean, middleMean, lateMean, trend };
}

async function runTask14Analysis() {
  console.log('🔬 Task 1.4: Layer Norm Investigation\n');

  // Get weight profile path from args
  const profilePath = process.argv[2];

  if (!profilePath) {
    console.error('❌ Error: No weight profile provided');
    console.log('\nUsage:');
    console.log(
      '  tsx src/research/llama-hallucination/demos/phase1-task1.4-layernorm-analysis.ts <weight-profile.json>'
    );
    console.log('\nExample:');
    console.log(
      '  tsx src/research/llama-hallucination/demos/phase1-task1.4-layernorm-analysis.ts research-output/phase1/weight-profile-1234567890.json'
    );
    process.exit(1);
  }

  if (!fs.existsSync(profilePath)) {
    console.error(`❌ Error: Weight profile not found: ${profilePath}`);
    process.exit(1);
  }

  console.log(`📁 Loading weight profile: ${path.basename(profilePath)}\n`);

  // Load weight profile
  const profileData = fs.readFileSync(profilePath, 'utf-8');
  const profile: WeightProfile = JSON.parse(profileData);

  console.log(`Model: ${profile.modelName}`);
  console.log(`Total Layers: ${profile.layers.length}\n`);

  // =================================================================
  // Part 1: Extract Layer Norm Statistics
  // =================================================================
  console.log('═══════════════════════════════════════════════════════════');
  console.log('Part 1: Layer Norm Statistics');
  console.log('═══════════════════════════════════════════════════════════\n');

  const { attnNorms, ffnNorms } = extractLayerNormStats(profile.layers);

  console.log(`Attention Norms found: ${attnNorms.length}`);
  console.log(`FFN Norms found: ${ffnNorms.length}\n`);

  // Display first few layers
  console.log('Layer | Attn Norm Mean | Attn Norm Std | FFN Norm Mean | FFN Norm Std');
  console.log('------|----------------|---------------|---------------|-------------');

  for (let i = 0; i < Math.min(10, attnNorms.length); i++) {
    console.log(
      `  ${attnNorms[i].layer.toString().padStart(3)} | ${attnNorms[i].stats.mean.toFixed(6).padStart(14)} | ${attnNorms[i].stats.std.toFixed(6).padStart(13)} | ${ffnNorms[i].stats.mean.toFixed(6).padStart(13)} | ${ffnNorms[i].stats.std.toFixed(6).padStart(12)}`
    );
  }
  console.log(`  ... (${attnNorms.length - 10} more layers)\n`);

  // =================================================================
  // Part 2: Norm Scale Trends
  // =================================================================
  console.log('═══════════════════════════════════════════════════════════');
  console.log('Part 2: Norm Scale Trends Across Layers');
  console.log('═══════════════════════════════════════════════════════════\n');

  const attnTrend = analyzeNormTrends(attnNorms);
  const ffnTrend = analyzeNormTrends(ffnNorms);

  console.log('Attention Norm Trends:');
  console.log(`  Early layers (0-10):   ${attnTrend.earlyMean.toFixed(6)}`);
  console.log(`  Middle layers (11-21): ${attnTrend.middleMean.toFixed(6)}`);
  console.log(`  Late layers (22-31):   ${attnTrend.lateMean.toFixed(6)}`);
  console.log(`  Pattern: ${attnTrend.trend}\n`);

  console.log('FFN Norm Trends:');
  console.log(`  Early layers (0-10):   ${ffnTrend.earlyMean.toFixed(6)}`);
  console.log(`  Middle layers (11-21): ${ffnTrend.middleMean.toFixed(6)}`);
  console.log(`  Late layers (22-31):   ${ffnTrend.lateMean.toFixed(6)}`);
  console.log(`  Pattern: ${ffnTrend.trend}\n`);

  // =================================================================
  // Part 3: Extreme Norm Layers
  // =================================================================
  console.log('═══════════════════════════════════════════════════════════');
  console.log('Part 3: Layers with Extreme Norm Scales');
  console.log('═══════════════════════════════════════════════════════════\n');

  const threshold = 1.5; // 1.5 standard deviations

  const attnExtremes = findExtremeNormLayers(attnNorms, threshold);
  const ffnExtremes = findExtremeNormLayers(ffnNorms, threshold);

  console.log(`Attention Norm Extremes (threshold: ${threshold}σ):`);
  console.log(`  High scale layers: [${attnExtremes.highScale.join(', ')}]`);
  console.log(`  Low scale layers:  [${attnExtremes.lowScale.join(', ')}]\n`);

  console.log(`FFN Norm Extremes (threshold: ${threshold}σ):`);
  console.log(`  High scale layers: [${ffnExtremes.highScale.join(', ')}]`);
  console.log(`  Low scale layers:  [${ffnExtremes.lowScale.join(', ')}]\n`);

  // =================================================================
  // Part 4: Norm-Component Correlations
  // =================================================================
  console.log('═══════════════════════════════════════════════════════════');
  console.log('Part 4: Norm-Component Strength Correlations');
  console.log('═══════════════════════════════════════════════════════════\n');

  const correlations = computeNormCorrelations(profile.layers);

  console.log('Correlation Analysis:');
  console.log(
    `  Attn Norm ↔ Query Strength:  ${correlations.attnNormVsQueryStrength.toFixed(4)}`
  );
  console.log(`  FFN Norm ↔ Gate Strength:    ${correlations.ffnNormVsGateStrength.toFixed(4)}\n`);

  // Interpret correlations
  const interpretCorr = (corr: number): string => {
    if (corr > 0.7) return 'Strong positive (norm scales with strength)';
    if (corr > 0.3) return 'Moderate positive';
    if (corr < -0.7) return 'Strong negative (norm compensates for strength)';
    if (corr < -0.3) return 'Moderate negative';
    return 'Weak correlation (independent)';
  };

  console.log('Interpretation:');
  console.log(`  Attn Norm: ${interpretCorr(correlations.attnNormVsQueryStrength)}`);
  console.log(`  FFN Norm:  ${interpretCorr(correlations.ffnNormVsGateStrength)}\n`);

  // =================================================================
  // Part 5: Layer 31 Special Analysis
  // =================================================================
  console.log('═══════════════════════════════════════════════════════════');
  console.log('Part 5: Layer 31 Norm Analysis (Hallucination Layer)');
  console.log('═══════════════════════════════════════════════════════════\n');

  const layer31Norm = profile.layers.find((l) => l.layerIndex === 31 && l.norm);

  if (layer31Norm && layer31Norm.norm) {
    const avgAttnNormMean =
      attnNorms.reduce((sum, n) => sum + n.stats.mean, 0) / attnNorms.length;
    const avgFFNNormMean = ffnNorms.reduce((sum, n) => sum + n.stats.mean, 0) / ffnNorms.length;

    console.log('Layer 31 Norm Statistics:');
    console.log(`  Attn Norm Mean: ${layer31Norm.norm.attnNorm.mean.toFixed(6)}`);
    console.log(`  Attn Norm Std:  ${layer31Norm.norm.attnNorm.std.toFixed(6)}`);
    console.log(`  FFN Norm Mean:  ${layer31Norm.norm.ffnNorm.mean.toFixed(6)}`);
    console.log(`  FFN Norm Std:   ${layer31Norm.norm.ffnNorm.std.toFixed(6)}\n`);

    console.log('Comparison to Average:');
    const attnDiff = ((layer31Norm.norm.attnNorm.mean / avgAttnNormMean - 1) * 100).toFixed(2);
    const ffnDiff = ((layer31Norm.norm.ffnNorm.mean / avgFFNNormMean - 1) * 100).toFixed(2);
    console.log(`  Attn Norm: ${attnDiff}% ${parseFloat(attnDiff) > 0 ? 'above' : 'below'} avg`);
    console.log(`  FFN Norm:  ${ffnDiff}% ${parseFloat(ffnDiff) > 0 ? 'above' : 'below'} avg\n`);

    // Hypothesis test
    if (Math.abs(parseFloat(attnDiff)) > 10 || Math.abs(parseFloat(ffnDiff)) > 10) {
      console.log('⚠️  FINDING: Layer 31 has significantly different norm scales!');
      console.log('    This may contribute to hallucination-prone behavior.\n');
    } else {
      console.log('ℹ️  Layer 31 norm scales are within normal range.\n');
    }
  } else {
    console.log('⚠️  Layer 31 norm data not found\n');
  }

  // =================================================================
  // Save detailed results
  // =================================================================
  const results = {
    task: 'Task 1.4: Layer Norm Investigation',
    timestamp: new Date().toISOString(),
    model: profile.modelName,
    attnNormTrend: attnTrend,
    ffnNormTrend: ffnTrend,
    attnExtremes,
    ffnExtremes,
    correlations,
    layer31: layer31Norm
      ? {
          attnNorm: layer31Norm.norm?.attnNorm,
          ffnNorm: layer31Norm.norm?.ffnNorm,
        }
      : null,
    allAttnNorms: attnNorms.map((n) => ({ layer: n.layer, mean: n.stats.mean, std: n.stats.std })),
    allFFNNorms: ffnNorms.map((n) => ({ layer: n.layer, mean: n.stats.mean, std: n.stats.std })),
  };

  const outputDir = 'research-output/phase1';
  const outputPath = path.join(outputDir, `task1.4-layernorm-analysis-${Date.now()}.json`);
  fs.writeFileSync(outputPath, JSON.stringify(results, null, 2));

  console.log(`✅ Detailed results saved to: ${outputPath}\n`);

  // =================================================================
  // Key Findings Summary
  // =================================================================
  console.log('═══════════════════════════════════════════════════════════');
  console.log('KEY FINDINGS - Task 1.4');
  console.log('═══════════════════════════════════════════════════════════\n');

  console.log('✅ Norm Scale Trends:');
  console.log(`   • Attn Norm: ${attnTrend.trend}`);
  console.log(`   • FFN Norm:  ${ffnTrend.trend}\n`);

  console.log('✅ Extreme Norm Layers:');
  console.log(
    `   • Attn high: ${attnExtremes.highScale.length} layers, low: ${attnExtremes.lowScale.length} layers`
  );
  console.log(
    `   • FFN high: ${ffnExtremes.highScale.length} layers, low: ${ffnExtremes.lowScale.length} layers\n`
  );

  console.log('✅ Norm-Strength Correlations:');
  console.log(
    `   • Attn: ${interpretCorr(correlations.attnNormVsQueryStrength)} (r=${correlations.attnNormVsQueryStrength.toFixed(3)})`
  );
  console.log(
    `   • FFN: ${interpretCorr(correlations.ffnNormVsGateStrength)} (r=${correlations.ffnNormVsGateStrength.toFixed(3)})\n`
  );

  console.log('🎉 Task 1.4 Complete!\n');

  console.log('═══════════════════════════════════════════════════════════');
  console.log('PHASE 1 TASKS COMPLETE');
  console.log('═══════════════════════════════════════════════════════════\n');

  console.log('✅ Task 1.1: Weight statistics analysis');
  console.log('✅ Task 1.2: Attention vs FFN analysis');
  console.log('✅ Task 1.3: FFN gate specialization');
  console.log('✅ Task 1.4: Layer norm investigation\n');

  console.log('Next: Generate visualizations and consolidate findings! 📊\n');
}

runTask14Analysis().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
