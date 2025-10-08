/**
 * Phase 3.2: FFN Regularization Mitigation Strategy
 *
 * Reduces FFN strength in late layers by scaling down gate/up/down weights.
 * More direct approach than attention amplification - attacks the root cause.
 *
 * Target: Reduce FFN strength by 50-70% in layers 28-31
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

interface RegularizationParameters {
  maxReduction: number;     // Maximum reduction factor (0-1, e.g., 0.7 = 70% reduction)
  startLayer: number;        // First layer to regularize
  endLayer: number;          // Last layer to regularize
  reductionCurve: 'linear' | 'exponential' | 'step';
  componentWeights?: {       // Which components to regularize
    gate: number;
    up: number;
    down: number;
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
// FFN Regularization Implementation
// ============================================================================

/**
 * Applies FFN regularization to late layers by scaling down weights
 */
function applyFFNRegularization(
  layers: LayerProfile[],
  params: RegularizationParameters
): LayerProfile[] {
  const defaultWeights = params.componentWeights || { gate: 1.0, up: 1.0, down: 1.0 };

  return layers.map((layer, idx) => {
    // Skip early layers
    if (idx < params.startLayer) {
      return layer;
    }

    // Don't regularize if no FFN data
    if (!layer.ffn) {
      return layer;
    }

    // Calculate reduction factor based on layer depth
    const progress = (idx - params.startLayer) / (params.endLayer - params.startLayer);
    let reductionFactor: number;

    switch (params.reductionCurve) {
      case 'exponential':
        // More aggressive reduction in later layers
        reductionFactor = 1 - (Math.pow(progress, 2) * params.maxReduction);
        break;
      case 'step':
        // Sudden reduction for layers 28-31
        reductionFactor = idx >= 28 ? 1 - params.maxReduction : 1;
        break;
      case 'linear':
      default:
        // Gradual reduction
        reductionFactor = 1 - (progress * params.maxReduction);
    }

    // Apply component-specific weights
    const gateScale = reductionFactor * (1 - (1 - reductionFactor) * defaultWeights.gate);
    const upScale = reductionFactor * (1 - (1 - reductionFactor) * defaultWeights.up);
    const downScale = reductionFactor * (1 - (1 - reductionFactor) * defaultWeights.down);

    // Apply regularization to FFN weights
    const regularizedFFN = {
      gate: scaleWeightStats(layer.ffn.gate, gateScale),
      up: scaleWeightStats(layer.ffn.up, upScale),
      down: scaleWeightStats(layer.ffn.down, downScale),
      ffnStrength: layer.ffn.ffnStrength * (gateScale * upScale * downScale),
      gateAmplification: layer.ffn.gateAmplification * (upScale * downScale),
    };

    return {
      ...layer,
      ffn: regularizedFFN,
    };
  });
}

/**
 * Scales weight statistics by a given factor
 */
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
    // Sparsity, skewness, kurtosis remain unchanged (shape properties)
  };
}

// ============================================================================
// Risk Calculation (from Phase 2A/3.1)
// ============================================================================

function predictAttentionActivation(attention: LayerProfile['attention']): {
  predictedStrength: number;
  informationRetention: number;
  entropy: number;
} {
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

function predictFFNActivation(ffn: LayerProfile['ffn']): {
  predictedStrength: number;
  amplificationFactor: number;
  nonlinearityEstimate: number;
} {
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
  baseline: {
    risk: number;
    dominance: number;
    ffnStrength: number;
  };
  mitigated: {
    risk: number;
    dominance: number;
    ffnStrength: number;
  };
  improvement: {
    riskReduction: number;
    riskReductionPercent: number;
    dominanceReduction: number;
    dominanceReductionPercent: number;
    ffnReduction: number;
    ffnReductionPercent: number;
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

    const ffnReduction = baseline.predictedFFNStrength - mitigated.predictedFFNStrength;
    const ffnReductionPercent = (ffnReduction / baseline.predictedFFNStrength) * 100;

    return {
      layer: baseline.layer,
      baseline: {
        risk: baseline.totalRisk,
        dominance: baseline.dominanceRatio,
        ffnStrength: baseline.predictedFFNStrength,
      },
      mitigated: {
        risk: mitigated.totalRisk,
        dominance: mitigated.dominanceRatio,
        ffnStrength: mitigated.predictedFFNStrength,
      },
      improvement: {
        riskReduction,
        riskReductionPercent,
        dominanceReduction,
        dominanceReductionPercent,
        ffnReduction,
        ffnReductionPercent,
      },
    };
  });
}

// ============================================================================
// Main Execution
// ============================================================================

async function runPhase3FFNRegularization() {
  console.log('╔══════════════════════════════════════════════════════════════════════════════╗');
  console.log('║          PHASE 3.2: FFN REGULARIZATION MITIGATION STRATEGY                  ║');
  console.log('╚══════════════════════════════════════════════════════════════════════════════╝\n');

  // Load baseline
  console.log('📂 Loading Phase 2A baseline results...\n');
  const baselinePath = path.join(__dirname, '../../../../research-output/phase2a/analysis-results.json');
  const baselineData = JSON.parse(fs.readFileSync(baselinePath, 'utf-8'));

  // Load weight profile
  const profilePath = path.join(__dirname, '../../../../research-output/phase1/weight-profile-1759955262868.json');
  const rawProfile = JSON.parse(fs.readFileSync(profilePath, 'utf-8'));
  const baselineLayers = preprocessWeightProfile(rawProfile);

  console.log(`✓ Loaded ${baselineLayers.length} layers\n`);

  // Define strategies
  const strategies = [
    {
      name: 'Linear Reduction (30%)',
      params: { maxReduction: 0.3, startLayer: 24, endLayer: 31, reductionCurve: 'linear' as const },
    },
    {
      name: 'Linear Reduction (50%)',
      params: { maxReduction: 0.5, startLayer: 24, endLayer: 31, reductionCurve: 'linear' as const },
    },
    {
      name: 'Linear Reduction (70%)',
      params: { maxReduction: 0.7, startLayer: 24, endLayer: 31, reductionCurve: 'linear' as const },
    },
    {
      name: 'Exponential Reduction (60%)',
      params: { maxReduction: 0.6, startLayer: 24, endLayer: 31, reductionCurve: 'exponential' as const },
    },
    {
      name: 'Step Reduction (80% layers 28-31)',
      params: { maxReduction: 0.8, startLayer: 24, endLayer: 31, reductionCurve: 'step' as const },
    },
  ];

  const results: any[] = [];

  for (const strategy of strategies) {
    console.log('═'.repeat(80));
    console.log(`🎯 TESTING STRATEGY: ${strategy.name}`);
    console.log('═'.repeat(80) + '\n');

    console.log('Parameters:');
    console.log(`  Max Reduction: ${(strategy.params.maxReduction * 100).toFixed(0)}%`);
    console.log(`  Target Layers: ${strategy.params.startLayer}-${strategy.params.endLayer}`);
    console.log(`  Reduction Curve: ${strategy.params.reductionCurve}\n`);

    // Apply mitigation
    console.log('⚡ Applying FFN regularization...\n');
    const mitigatedLayers = applyFFNRegularization(baselineLayers, strategy.params);

    // Recalculate
    console.log('📊 Recalculating global statistics...\n');
    const mitigatedStats = computeGlobalStatistics(mitigatedLayers);

    console.log('Mitigated Global Averages:');
    console.log(`  Attention Strength: ${mitigatedStats.avgAttentionStrength.toFixed(2)} (baseline: ${baselineData.globalStatistics.avgAttentionStrength.toFixed(2)})`);
    console.log(`  FFN Strength: ${mitigatedStats.avgFFNStrength.toFixed(2)} (baseline: ${baselineData.globalStatistics.avgFFNStrength.toFixed(2)})`);
    console.log(`  Avg FFN/Attn Ratio: ${(mitigatedStats.avgFFNStrength / mitigatedStats.avgAttentionStrength).toFixed(2)}×\n`);

    console.log('🎯 Recalculating hallucination risks...\n');
    const mitigatedRisks = mitigatedLayers.map(layer => calculateHallucinationRisk(layer, mitigatedStats));

    // Compare
    console.log('📈 Comparing with baseline...\n');
    const comparison = compareResults(baselineData.hallucinationRisks, mitigatedRisks);

    // Top improvements
    const topImprovements = comparison
      .filter(c => c.baseline.risk > 10)
      .sort((a, b) => b.improvement.riskReduction - a.improvement.riskReduction)
      .slice(0, 5);

    console.log('Top 5 Risk Reductions:\n');
    topImprovements.forEach((comp, idx) => {
      console.log(`${idx + 1}. Layer ${comp.layer}:`);
      console.log(`   Risk: ${comp.baseline.risk.toFixed(1)}% → ${comp.mitigated.risk.toFixed(1)}% (-${comp.improvement.riskReduction.toFixed(1)}pp, ${comp.improvement.riskReductionPercent.toFixed(1)}%)`);
      console.log(`   Dominance: ${comp.baseline.dominance.toFixed(0)}× → ${comp.mitigated.dominance.toFixed(0)}× (-${comp.improvement.dominanceReductionPercent.toFixed(1)}%)`);
      console.log(`   FFN Strength: ${(comp.baseline.ffnStrength / 1000).toFixed(0)}k → ${(comp.mitigated.ffnStrength / 1000).toFixed(0)}k (-${comp.improvement.ffnReductionPercent.toFixed(1)}%)\n`);
    });

    // Overall stats
    const avgRiskReduction = comparison.reduce((sum, c) => sum + c.improvement.riskReduction, 0) / comparison.length;
    const avgDominanceReduction = comparison.reduce((sum, c) => sum + c.improvement.dominanceReductionPercent, 0) / comparison.length;
    const avgFFNReduction = comparison.reduce((sum, c) => sum + c.improvement.ffnReductionPercent, 0) / comparison.length;

    console.log('Overall Impact:');
    console.log(`  Average risk reduction: ${avgRiskReduction.toFixed(2)}pp`);
    console.log(`  Average dominance reduction: ${avgDominanceReduction.toFixed(1)}%`);
    console.log(`  Average FFN reduction: ${avgFFNReduction.toFixed(1)}%\n`);

    results.push({
      strategy: strategy.name,
      parameters: strategy.params,
      globalStatistics: mitigatedStats,
      risks: mitigatedRisks,
      comparison,
      summary: {
        avgRiskReduction,
        avgDominanceReduction,
        avgFFNReduction,
        topImprovements: topImprovements.map(t => ({
          layer: t.layer,
          riskReduction: t.improvement.riskReduction,
          dominanceReduction: t.improvement.dominanceReductionPercent,
          ffnReduction: t.improvement.ffnReductionPercent,
        })),
      },
    });
  }

  // Save
  const outputPath = path.join(__dirname, '../../../../research-output/phase3/ffn-regularization-results.json');
  const output = {
    timestamp: Date.now(),
    strategy: 'FFN Regularization',
    baseline: {
      source: 'phase2a/analysis-results.json',
      globalStatistics: baselineData.globalStatistics,
    },
    results,
  };

  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));

  console.log('═'.repeat(80));
  console.log('✅ PHASE 3.2 COMPLETE');
  console.log('═'.repeat(80) + '\n');
  console.log(`💾 Results saved to: ${outputPath}\n`);

  // Best strategy
  const bestStrategy = results.reduce((best, current) =>
    current.summary.avgRiskReduction > best.summary.avgRiskReduction ? current : best
  );

  console.log(`🏆 Best Strategy: ${bestStrategy.strategy}`);
  console.log(`   Average Risk Reduction: ${bestStrategy.summary.avgRiskReduction.toFixed(2)}pp`);
  console.log(`   Average FFN Reduction: ${bestStrategy.summary.avgFFNReduction.toFixed(1)}%\n`);
}

// Run
if (require.main === module) {
  runPhase3FFNRegularization().catch(console.error);
}
