/**
 * Phase 4: Combined Mitigation Strategies
 *
 * Tests combinations of multiple strategies to achieve maximum risk reduction.
 * Combines the best variants from Phase 3 plus new Norm Clipping approach.
 *
 * Target: Reduce Layer 30 risk below 20% (>40% reduction from 33.6%)
 */

import * as fs from 'fs';
import * as path from 'path';

// ============================================================================
// Type Definitions
// ============================================================================

interface WeightStats {
  mean: number;
  std: number;
  min: number;
  max: number;
  median: number;
  skewness: number;
  kurtosis: number;
  sparsity: number;
  effectiveRank: number;
  l1Norm: number;
  l2Norm: number;
  lInfNorm: number;
  frobeniusNorm: number;
  bitsPerElement: number;
}

interface LayerProfile {
  layer: number;
  attention?: {
    query: WeightStats;
    key: WeightStats;
    value: WeightStats;
    output: WeightStats;
    qkRatio: number;
    valueSparsity: number;
    attentionStrength: number;
  };
  ffn?: {
    gate: WeightStats;
    up: WeightStats;
    down: WeightStats;
    ffnStrength: number;
    gateAmplification: number;
  };
  norms?: {
    attnNorm: WeightStats;
    ffnNorm: WeightStats;
  };
}

interface HallucinationRisk {
  layer: number;
  totalRisk: number;
  components: {
    valueSparsityRisk: number;
    attentionWeakeningRisk: number;
    valueAmplificationRisk: number;
    keyMatchingRisk: number;
    normAmplificationRisk: number;
  };
  dominanceRatio: number;
  predictedAttentionStrength: number;
  predictedFFNStrength: number;
}

interface RawLayerComponent {
  layerIndex: number;
  layerType: 'attention' | 'ffn' | 'norm';
  attention?: {
    query: WeightStats;
    key: WeightStats;
    value: WeightStats;
    output: WeightStats;
  };
  ffn?: {
    gate: WeightStats;
    up: WeightStats;
    down: WeightStats;
  };
  norm?: {
    attnNorm: WeightStats;
    ffnNorm: WeightStats;
  };
}

interface RawWeightProfile {
  modelName: string;
  quantizationType: string;
  totalParameters: number;
  layers: RawLayerComponent[];
}

interface CombinedStrategyConfig {
  ffnRegularization?: {
    maxReduction: number;
    startLayer: number;
    endLayer: number;
  };
  attentionAmplification?: {
    maxBoost: number;
    startLayer: number;
    endLayer: number;
  };
  normClipping?: {
    clipRatio: number;  // e.g., 1.15 = clip at 115% of global average
    startLayer: number;
    endLayer: number;
  };
}

// ============================================================================
// Data Preprocessing
// ============================================================================

function preprocessWeightProfile(raw: RawWeightProfile): LayerProfile[] {
  const layerMap = new Map<number, LayerProfile>();

  for (const component of raw.layers) {
    const idx = component.layerIndex;

    if (!layerMap.has(idx)) {
      layerMap.set(idx, { layer: idx });
    }

    const layer = layerMap.get(idx)!;

    if (component.layerType === 'attention' && component.attention) {
      const attn = component.attention;
      const qNorm = attn.query.l2Norm;
      const kNorm = attn.key.l2Norm;

      layer.attention = {
        query: attn.query,
        key: attn.key,
        value: attn.value,
        output: attn.output,
        qkRatio: qNorm / Math.max(kNorm, 1e-8),
        valueSparsity: attn.value.sparsity,
        attentionStrength: Math.sqrt(qNorm * kNorm),
      };
    } else if (component.layerType === 'ffn' && component.ffn) {
      const ffn = component.ffn;
      const gateNorm = ffn.gate.l2Norm;
      const upNorm = ffn.up.l2Norm;
      const downNorm = ffn.down.l2Norm;

      layer.ffn = {
        gate: ffn.gate,
        up: ffn.up,
        down: ffn.down,
        ffnStrength: gateNorm * upNorm * downNorm,
        gateAmplification: upNorm * downNorm,
      };
    } else if (component.layerType === 'norm' && component.norm) {
      layer.norms = component.norm;
    }
  }

  return Array.from(layerMap.values()).sort((a, b) => a.layer - b.layer);
}

// ============================================================================
// Individual Strategy Implementations
// ============================================================================

function scaleWeightStats(stats: WeightStats, scale: number): WeightStats {
  return {
    ...stats,
    mean: stats.mean * scale,
    std: stats.std * scale,
    min: stats.min * scale,
    max: stats.max * scale,
    median: stats.median * scale,
    l1Norm: stats.l1Norm * scale,
    l2Norm: stats.l2Norm * scale,
    lInfNorm: stats.lInfNorm * scale,
    frobeniusNorm: stats.frobeniusNorm * scale,
  };
}

function applyFFNRegularization(
  layers: LayerProfile[],
  config: NonNullable<CombinedStrategyConfig['ffnRegularization']>
): LayerProfile[] {
  return layers.map((layer, idx) => {
    if (idx < config.startLayer || !layer.ffn) {
      return layer;
    }

    const progress = (idx - config.startLayer) / (config.endLayer - config.startLayer);
    const reductionFactor = 1 - (progress * config.maxReduction);

    const regularizedFFN = {
      gate: scaleWeightStats(layer.ffn.gate, reductionFactor),
      up: scaleWeightStats(layer.ffn.up, reductionFactor),
      down: scaleWeightStats(layer.ffn.down, reductionFactor),
      ffnStrength: layer.ffn.ffnStrength * Math.pow(reductionFactor, 3),
      gateAmplification: layer.ffn.gateAmplification * Math.pow(reductionFactor, 2),
    };

    return { ...layer, ffn: regularizedFFN };
  });
}

function applyAttentionAmplification(
  layers: LayerProfile[],
  config: NonNullable<CombinedStrategyConfig['attentionAmplification']>
): LayerProfile[] {
  return layers.map((layer, idx) => {
    if (idx < config.startLayer || !layer.attention) {
      return layer;
    }

    const progress = (idx - config.startLayer) / (config.endLayer - config.startLayer);
    const boostFactor = 1 + (progress * config.maxBoost);

    const amplifiedAttention = {
      ...layer.attention,
      query: scaleWeightStats(layer.attention.query, boostFactor),
      key: scaleWeightStats(layer.attention.key, boostFactor),
      value: scaleWeightStats(layer.attention.value, boostFactor),
      output: scaleWeightStats(layer.attention.output, boostFactor),
      attentionStrength: layer.attention.attentionStrength * boostFactor,
    };

    return { ...layer, attention: amplifiedAttention };
  });
}

function applyNormClipping(
  layers: LayerProfile[],
  config: NonNullable<CombinedStrategyConfig['normClipping']>,
  globalAvgFFNNorm: number
): LayerProfile[] {
  const maxNorm = globalAvgFFNNorm * config.clipRatio;

  return layers.map((layer, idx) => {
    if (idx < config.startLayer || !layer.norms) {
      return layer;
    }

    const currentNorm = layer.norms.ffnNorm.mean;
    if (currentNorm <= maxNorm) {
      return layer; // No clipping needed
    }

    // Scale down to max norm
    const clipScale = maxNorm / currentNorm;

    const clippedNorms = {
      attnNorm: layer.norms.attnNorm, // Keep attention norm unchanged
      ffnNorm: scaleWeightStats(layer.norms.ffnNorm, clipScale),
    };

    return { ...layer, norms: clippedNorms };
  });
}

// ============================================================================
// Combined Strategy Application
// ============================================================================

function applyCombinedStrategy(
  layers: LayerProfile[],
  config: CombinedStrategyConfig,
  globalAvgFFNNorm: number
): LayerProfile[] {
  let result = layers;

  // Apply strategies in order: FFN Reg → Attention Amp → Norm Clip
  // Order matters to avoid compounding side effects

  if (config.ffnRegularization) {
    result = applyFFNRegularization(result, config.ffnRegularization);
  }

  if (config.attentionAmplification) {
    result = applyAttentionAmplification(result, config.attentionAmplification);
  }

  if (config.normClipping) {
    result = applyNormClipping(result, config.normClipping, globalAvgFFNNorm);
  }

  return result;
}

// ============================================================================
// Risk Calculation
// ============================================================================

function predictAttentionActivation(attention: LayerProfile['attention']) {
  if (!attention) {
    return { predictedStrength: 0, informationRetention: 0, entropy: 0 };
  }

  const qNorm = attention.query.l2Norm;
  const kNorm = attention.key.l2Norm;
  const vSparsity = attention.value.sparsity;

  const predictedStrength = Math.sqrt(qNorm * kNorm);
  const informationRetention = 1 - vSparsity;
  const entropy = (vSparsity * attention.value.std) / Math.max(attention.value.l2Norm, 1e-8);

  return { predictedStrength, informationRetention, entropy };
}

function predictFFNActivation(ffn: LayerProfile['ffn']) {
  if (!ffn) {
    return { predictedStrength: 0, amplificationFactor: 0, nonlinearityEstimate: 0 };
  }

  const gateNorm = ffn.gate.l2Norm;
  const upNorm = ffn.up.l2Norm;
  const downNorm = ffn.down.l2Norm;

  const predictedStrength = gateNorm * upNorm * downNorm;
  const amplificationFactor = upNorm * downNorm;
  const nonlinearityEstimate = (ffn.gate.kurtosis + ffn.up.kurtosis) / 2;

  return { predictedStrength, amplificationFactor, nonlinearityEstimate };
}

function computeGlobalStatistics(layers: LayerProfile[]) {
  const validLayers = layers.filter(l => l.attention && l.ffn && l.norms);

  if (validLayers.length === 0) {
    return {
      avgAttentionStrength: 0,
      avgFFNStrength: 0,
      avgValueSparsity: 0,
      avgQKRatio: 0,
      avgFFNNorm: 0,
    };
  }

  const attnActivations = validLayers.map(l => predictAttentionActivation(l.attention));
  const ffnActivations = validLayers.map(l => predictFFNActivation(l.ffn));

  return {
    avgAttentionStrength: attnActivations.reduce((sum, a) => sum + a.predictedStrength, 0) / attnActivations.length,
    avgFFNStrength: ffnActivations.reduce((sum, a) => sum + a.predictedStrength, 0) / ffnActivations.length,
    avgValueSparsity: validLayers.reduce((sum, l) => sum + l.attention!.valueSparsity, 0) / validLayers.length,
    avgQKRatio: validLayers.reduce((sum, l) => sum + l.attention!.qkRatio, 0) / validLayers.length,
    avgFFNNorm: validLayers.reduce((sum, l) => sum + l.norms!.ffnNorm.mean, 0) / validLayers.length,
  };
}

function calculateHallucinationRisk(
  layer: LayerProfile,
  globalStats: ReturnType<typeof computeGlobalStatistics>
): HallucinationRisk {
  if (!layer.attention || !layer.ffn || !layer.norms) {
    return {
      layer: layer.layer,
      totalRisk: 0,
      components: {
        valueSparsityRisk: 0,
        attentionWeakeningRisk: 0,
        valueAmplificationRisk: 0,
        keyMatchingRisk: 0,
        normAmplificationRisk: 0,
      },
      dominanceRatio: 0,
      predictedAttentionStrength: 0,
      predictedFFNStrength: 0,
    };
  }

  const attnActivation = predictAttentionActivation(layer.attention);
  const ffnActivation = predictFFNActivation(layer.ffn);
  const dominanceRatio = ffnActivation.predictedStrength / Math.max(attnActivation.predictedStrength, 1e-8);

  const valueSparsityRisk = Math.min(100, (layer.attention.valueSparsity / 0.05) * 100);
  const attentionWeakeningRisk = Math.max(0, (1 - attnActivation.predictedStrength / globalStats.avgAttentionStrength) * 100);
  const valueAmplificationRisk = Math.min(100, (layer.attention.value.l2Norm / 100) * 100);
  const qkRatio = layer.attention.qkRatio;
  const keyMatchingRisk = Math.max(0, (1 - qkRatio / globalStats.avgQKRatio) * 100);
  const ffnNormRatio = layer.norms.ffnNorm.mean / globalStats.avgFFNNorm;
  const normAmplificationRisk = Math.max(0, (ffnNormRatio - 1) * 100);

  const totalRisk = (
    valueSparsityRisk * 0.25 +
    attentionWeakeningRisk * 0.20 +
    valueAmplificationRisk * 0.20 +
    keyMatchingRisk * 0.20 +
    normAmplificationRisk * 0.15
  );

  return {
    layer: layer.layer,
    totalRisk,
    components: {
      valueSparsityRisk,
      attentionWeakeningRisk,
      valueAmplificationRisk,
      keyMatchingRisk,
      normAmplificationRisk,
    },
    dominanceRatio,
    predictedAttentionStrength: attnActivation.predictedStrength,
    predictedFFNStrength: ffnActivation.predictedStrength,
  };
}

// ============================================================================
// Comparison
// ============================================================================

interface ComparisonResult {
  layer: number;
  baseline: { risk: number; dominance: number };
  mitigated: { risk: number; dominance: number };
  improvement: {
    riskReduction: number;
    riskReductionPercent: number;
    dominanceReduction: number;
    dominanceReductionPercent: number;
  };
}

function compareResults(
  baselineRisks: HallucinationRisk[],
  mitigatedRisks: HallucinationRisk[]
): ComparisonResult[] {
  return baselineRisks.map((baseline, idx) => {
    const mitigated = mitigatedRisks[idx];

    const riskReduction = baseline.totalRisk - mitigated.totalRisk;
    const riskReductionPercent = (riskReduction / baseline.totalRisk) * 100;
    const dominanceReduction = baseline.dominanceRatio - mitigated.dominanceRatio;
    const dominanceReductionPercent = (dominanceReduction / baseline.dominanceRatio) * 100;

    return {
      layer: baseline.layer,
      baseline: {
        risk: baseline.totalRisk,
        dominance: baseline.dominanceRatio,
      },
      mitigated: {
        risk: mitigated.totalRisk,
        dominance: mitigated.dominanceRatio,
      },
      improvement: {
        riskReduction,
        riskReductionPercent,
        dominanceReduction,
        dominanceReductionPercent,
      },
    };
  });
}

// ============================================================================
// Main Execution
// ============================================================================

async function runPhase4CombinedStrategies() {
  console.log('╔══════════════════════════════════════════════════════════════════════════════╗');
  console.log('║              PHASE 4: COMBINED MITIGATION STRATEGIES                         ║');
  console.log('╚══════════════════════════════════════════════════════════════════════════════╝\n');

  // Load baseline
  console.log('📂 Loading Phase 2A baseline results...\n');
  const baselinePath = path.join(__dirname, '../../../../research-output/phase2a/analysis-results.json');
  const baselineData = JSON.parse(fs.readFileSync(baselinePath, 'utf-8'));

  const profilePath = path.join(__dirname, '../../../../research-output/phase1/weight-profile-1759955262868.json');
  const rawProfile = JSON.parse(fs.readFileSync(profilePath, 'utf-8'));
  const baselineLayers = preprocessWeightProfile(rawProfile);

  console.log(`✓ Loaded ${baselineLayers.length} layers\n`);

  const baselineGlobalStats = computeGlobalStatistics(baselineLayers);

  // Define combined strategies
  const strategies = [
    {
      name: 'FFN Reg (70%) + Attention Amp (2×)',
      config: {
        ffnRegularization: { maxReduction: 0.7, startLayer: 24, endLayer: 31 },
        attentionAmplification: { maxBoost: 1.0, startLayer: 24, endLayer: 31 },
      },
    },
    {
      name: 'FFN Reg (70%) + Norm Clip (1.15×)',
      config: {
        ffnRegularization: { maxReduction: 0.7, startLayer: 24, endLayer: 31 },
        normClipping: { clipRatio: 1.15, startLayer: 21, endLayer: 31 },
      },
    },
    {
      name: 'FFN Reg (70%) + Attention Amp (2×) + Norm Clip (1.15×)',
      config: {
        ffnRegularization: { maxReduction: 0.7, startLayer: 24, endLayer: 31 },
        attentionAmplification: { maxBoost: 1.0, startLayer: 24, endLayer: 31 },
        normClipping: { clipRatio: 1.15, startLayer: 21, endLayer: 31 },
      },
    },
    {
      name: 'FFN Reg (50%) + Attention Amp (3×) + Norm Clip (1.10×)',
      config: {
        ffnRegularization: { maxReduction: 0.5, startLayer: 24, endLayer: 31 },
        attentionAmplification: { maxBoost: 2.0, startLayer: 24, endLayer: 31 },
        normClipping: { clipRatio: 1.10, startLayer: 21, endLayer: 31 },
      },
    },
    {
      name: 'Aggressive: FFN Reg (80%) + Attention Amp (4×) + Norm Clip (1.05×)',
      config: {
        ffnRegularization: { maxReduction: 0.8, startLayer: 24, endLayer: 31 },
        attentionAmplification: { maxBoost: 3.0, startLayer: 28, endLayer: 31 },
        normClipping: { clipRatio: 1.05, startLayer: 21, endLayer: 31 },
      },
    },
  ];

  const results: any[] = [];

  for (const strategy of strategies) {
    console.log('═'.repeat(80));
    console.log(`🎯 TESTING STRATEGY: ${strategy.name}`);
    console.log('═'.repeat(80) + '\n');

    console.log('Configuration:');
    if (strategy.config.ffnRegularization) {
      console.log(`  FFN Regularization: ${(strategy.config.ffnRegularization.maxReduction * 100).toFixed(0)}% (layers ${strategy.config.ffnRegularization.startLayer}-${strategy.config.ffnRegularization.endLayer})`);
    }
    if (strategy.config.attentionAmplification) {
      console.log(`  Attention Amplification: ${(strategy.config.attentionAmplification.maxBoost + 1).toFixed(1)}× (layers ${strategy.config.attentionAmplification.startLayer}-${strategy.config.attentionAmplification.endLayer})`);
    }
    if (strategy.config.normClipping) {
      console.log(`  Norm Clipping: ${(strategy.config.normClipping.clipRatio * 100).toFixed(0)}% of avg (layers ${strategy.config.normClipping.startLayer}-${strategy.config.normClipping.endLayer})`);
    }
    console.log('');

    // Apply combined strategy
    console.log('⚡ Applying combined mitigation...\n');
    const mitigatedLayers = applyCombinedStrategy(baselineLayers, strategy.config, baselineGlobalStats.avgFFNNorm);

    // Recalculate
    console.log('📊 Recalculating global statistics...\n');
    const mitigatedStats = computeGlobalStatistics(mitigatedLayers);

    console.log('Mitigated Global Averages:');
    console.log(`  Attention Strength: ${mitigatedStats.avgAttentionStrength.toFixed(2)} (baseline: ${baselineGlobalStats.avgAttentionStrength.toFixed(2)})`);
    console.log(`  FFN Strength: ${mitigatedStats.avgFFNStrength.toFixed(2)} (baseline: ${baselineGlobalStats.avgFFNStrength.toFixed(2)})`);
    console.log(`  Avg FFN/Attn Ratio: ${(mitigatedStats.avgFFNStrength / mitigatedStats.avgAttentionStrength).toFixed(2)}×\n`);

    console.log('🎯 Recalculating hallucination risks...\n');
    const mitigatedRisks = mitigatedLayers.map(layer => calculateHallucinationRisk(layer, mitigatedStats));

    // Compare
    console.log('📈 Comparing with baseline...\n');
    const comparison = compareResults(baselineData.hallucinationRisks, mitigatedRisks);

    // Peak layers (28-30)
    const peakLayers = comparison.filter(c => c.layer >= 28 && c.layer <= 30);

    console.log('Peak Layers (28-30) Results:\n');
    peakLayers.forEach(comp => {
      console.log(`Layer ${comp.layer}:`);
      console.log(`  Risk: ${comp.baseline.risk.toFixed(1)}% → ${comp.mitigated.risk.toFixed(1)}% (-${comp.improvement.riskReduction.toFixed(1)}pp, ${comp.improvement.riskReductionPercent.toFixed(1)}%)`);
      console.log(`  Dominance: ${comp.baseline.dominance.toFixed(0)}× → ${comp.mitigated.dominance.toFixed(0)}× (-${comp.improvement.dominanceReductionPercent.toFixed(1)}%)\n`);
    });

    // Overall stats
    const avgPeakRiskReduction = peakLayers.reduce((sum, c) => sum + c.improvement.riskReduction, 0) / peakLayers.length;
    const avgDominanceReduction = comparison.reduce((sum, c) => sum + c.improvement.dominanceReductionPercent, 0) / comparison.length;

    console.log('Overall Impact:');
    console.log(`  Average peak risk reduction (28-30): ${avgPeakRiskReduction.toFixed(2)}pp`);
    console.log(`  Average dominance reduction: ${avgDominanceReduction.toFixed(1)}%`);
    console.log(`  Layer 30 final risk: ${peakLayers[2].mitigated.risk.toFixed(1)}%\n`);

    results.push({
      strategy: strategy.name,
      config: strategy.config,
      globalStatistics: mitigatedStats,
      risks: mitigatedRisks,
      comparison,
      summary: {
        avgPeakRiskReduction,
        avgDominanceReduction,
        layer30FinalRisk: peakLayers[2].mitigated.risk,
        peakLayersImprovement: peakLayers.map(p => ({
          layer: p.layer,
          riskReduction: p.improvement.riskReduction,
          finalRisk: p.mitigated.risk,
        })),
      },
    });
  }

  // Save
  const outputPath = path.join(__dirname, '../../../../research-output/phase4/combined-strategies-results.json');
  const outputDir = path.dirname(outputPath);
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  const output = {
    timestamp: Date.now(),
    strategy: 'Combined Strategies',
    baseline: {
      source: 'phase2a/analysis-results.json',
      globalStatistics: baselineData.globalStatistics,
    },
    results,
  };

  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));

  console.log('═'.repeat(80));
  console.log('✅ PHASE 4 COMPLETE');
  console.log('═'.repeat(80) + '\n');
  console.log(`💾 Results saved to: ${outputPath}\n`);

  // Best strategy
  const bestStrategy = results.reduce((best, current) =>
    current.summary.avgPeakRiskReduction > best.summary.avgPeakRiskReduction ? current : best
  );

  console.log(`🏆 Best Combined Strategy: ${bestStrategy.strategy}`);
  console.log(`   Average Peak Risk Reduction (28-30): ${bestStrategy.summary.avgPeakRiskReduction.toFixed(2)}pp`);
  console.log(`   Layer 30 Final Risk: ${bestStrategy.summary.layer30FinalRisk.toFixed(1)}%`);
  console.log(`   Reduction from baseline: ${((1 - bestStrategy.summary.layer30FinalRisk / 33.64) * 100).toFixed(1)}%\n`);
}

// Run
if (require.main === module) {
  runPhase4CombinedStrategies().catch(console.error);
}
