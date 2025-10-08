/**
 * Phase 3.1: Attention Amplification Mitigation Strategy
 *
 * Boosts attention mechanism strength in late layers to counteract
 * extreme FFN dominance discovered in Phase 2A.
 *
 * Target: Reduce FFN dominance from 22,441× to < 1,000×
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

interface BaselineResults {
  globalStatistics: {
    avgAttentionStrength: number;
    avgFFNStrength: number;
    avgValueSparsity: number;
    avgQKRatio: number;
    avgFFNNorm: number;
  };
  hallucinationRisks: HallucinationRisk[];
}

// ============================================================================
// Data Preprocessing (from Phase 2A)
// ============================================================================

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
// Mitigation Strategy Implementation
// ============================================================================

interface MitigationParameters {
  maxBoost: number;        // Maximum amplification factor
  startLayer: number;      // First layer to amplify
  endLayer: number;        // Last layer to amplify
  boostCurve: 'linear' | 'exponential' | 'step';
}

/**
 * Applies attention amplification to late layers
 */
function applyAttentionAmplification(
  layers: LayerProfile[],
  params: MitigationParameters
): LayerProfile[] {
  return layers.map((layer, idx) => {
    // Skip early layers
    if (idx < params.startLayer) {
      return layer;
    }

    // Don't amplify if no attention data
    if (!layer.attention) {
      return layer;
    }

    // Calculate amplification factor based on layer depth
    const progress = (idx - params.startLayer) / (params.endLayer - params.startLayer);
    let boostFactor: number;

    switch (params.boostCurve) {
      case 'exponential':
        boostFactor = 1 + (Math.pow(progress, 2) * params.maxBoost);
        break;
      case 'step':
        boostFactor = idx >= 28 ? 1 + params.maxBoost : 1;
        break;
      case 'linear':
      default:
        boostFactor = 1 + (progress * params.maxBoost);
    }

    // Apply amplification to attention weights
    const amplifiedAttention = {
      ...layer.attention,
      query: amplifyWeightStats(layer.attention.query, boostFactor),
      key: amplifyWeightStats(layer.attention.key, boostFactor),
      value: amplifyWeightStats(layer.attention.value, boostFactor),
      output: amplifyWeightStats(layer.attention.output, boostFactor),
      attentionStrength: layer.attention.attentionStrength * boostFactor,
    };

    return {
      ...layer,
      attention: amplifiedAttention,
    };
  });
}

/**
 * Amplifies weight statistics by a given factor
 */
function amplifyWeightStats(stats: WeightStats, factor: number): WeightStats {
  return {
    ...stats,
    mean: stats.mean * factor,
    std: stats.std * factor,
    min: stats.min * factor,
    max: stats.max * factor,
    median: stats.median * factor,
    l1Norm: stats.l1Norm * factor,
    l2Norm: stats.l2Norm * factor,
    lInfNorm: stats.lInfNorm * factor,
    frobeniusNorm: stats.frobeniusNorm * factor,
    // Sparsity, skewness, kurtosis remain unchanged (shape properties)
  };
}

// ============================================================================
// Risk Recalculation (copied from Phase 2A)
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
// Comparison and Reporting
// ============================================================================

interface ComparisonResult {
  layer: number;
  baseline: {
    risk: number;
    dominance: number;
    attentionStrength: number;
  };
  mitigated: {
    risk: number;
    dominance: number;
    attentionStrength: number;
  };
  improvement: {
    riskReduction: number;       // Percentage points
    riskReductionPercent: number; // Percent of original
    dominanceReduction: number;   // Absolute reduction
    dominanceReductionPercent: number;
    attentionIncrease: number;
    attentionIncreasePercent: number;
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

    const attentionIncrease = mitigated.predictedAttentionStrength - baseline.predictedAttentionStrength;
    const attentionIncreasePercent = (attentionIncrease / baseline.predictedAttentionStrength) * 100;

    return {
      layer: baseline.layer,
      baseline: {
        risk: baseline.totalRisk,
        dominance: baseline.dominanceRatio,
        attentionStrength: baseline.predictedAttentionStrength,
      },
      mitigated: {
        risk: mitigated.totalRisk,
        dominance: mitigated.dominanceRatio,
        attentionStrength: mitigated.predictedAttentionStrength,
      },
      improvement: {
        riskReduction,
        riskReductionPercent,
        dominanceReduction,
        dominanceReductionPercent,
        attentionIncrease,
        attentionIncreasePercent,
      },
    };
  });
}

// ============================================================================
// Main Execution
// ============================================================================

async function runPhase3AttentionAmplification() {
  console.log('╔══════════════════════════════════════════════════════════════════════════════╗');
  console.log('║        PHASE 3.1: ATTENTION AMPLIFICATION MITIGATION STRATEGY                ║');
  console.log('╚══════════════════════════════════════════════════════════════════════════════╝\n');

  // Load baseline results from Phase 2A
  console.log('📂 Loading Phase 2A baseline results...\n');
  const baselinePath = path.join(__dirname, '../../../../research-output/phase2a/analysis-results.json');
  const baselineData = JSON.parse(fs.readFileSync(baselinePath, 'utf-8'));

  // Load weight profile
  const profilePath = path.join(__dirname, '../../../../research-output/phase1/weight-profile-1759955262868.json');
  const rawProfile = JSON.parse(fs.readFileSync(profilePath, 'utf-8'));

  // Preprocess layers
  const baselineLayers = preprocessWeightProfile(rawProfile);

  console.log(`✓ Loaded ${baselineLayers.length} layers\n`);

  // Define mitigation parameters
  const strategies = [
    {
      name: 'Linear Boost (2×)',
      params: { maxBoost: 1.0, startLayer: 24, endLayer: 31, boostCurve: 'linear' as const },
    },
    {
      name: 'Linear Boost (3×)',
      params: { maxBoost: 2.0, startLayer: 24, endLayer: 31, boostCurve: 'linear' as const },
    },
    {
      name: 'Linear Boost (5×)',
      params: { maxBoost: 4.0, startLayer: 24, endLayer: 31, boostCurve: 'linear' as const },
    },
    {
      name: 'Exponential Boost (3×)',
      params: { maxBoost: 2.0, startLayer: 24, endLayer: 31, boostCurve: 'exponential' as const },
    },
    {
      name: 'Step Boost (4× layers 28-31)',
      params: { maxBoost: 3.0, startLayer: 24, endLayer: 31, boostCurve: 'step' as const },
    },
  ];

  const results: any[] = [];

  for (const strategy of strategies) {
    console.log('═'.repeat(80));
    console.log(`🎯 TESTING STRATEGY: ${strategy.name}`);
    console.log('═'.repeat(80) + '\n');

    console.log('Parameters:');
    console.log(`  Max Boost: ${strategy.params.maxBoost + 1}×`);
    console.log(`  Target Layers: ${strategy.params.startLayer}-${strategy.params.endLayer}`);
    console.log(`  Boost Curve: ${strategy.params.boostCurve}\n`);

    // Apply mitigation
    console.log('⚡ Applying attention amplification...\n');
    const mitigatedLayers = applyAttentionAmplification(baselineLayers, strategy.params);

    // Recalculate statistics and risks
    console.log('📊 Recalculating global statistics...\n');
    const mitigatedStats = computeGlobalStatistics(mitigatedLayers);

    console.log('Mitigated Global Averages:');
    console.log(`  Attention Strength: ${mitigatedStats.avgAttentionStrength.toFixed(2)} (baseline: ${baselineData.globalStatistics.avgAttentionStrength.toFixed(2)})`);
    console.log(`  FFN Strength: ${mitigatedStats.avgFFNStrength.toFixed(2)} (unchanged)`);
    console.log(`  Avg FFN/Attn Ratio: ${(mitigatedStats.avgFFNStrength / mitigatedStats.avgAttentionStrength).toFixed(2)}×\n`);

    console.log('🎯 Recalculating hallucination risks...\n');
    const mitigatedRisks = mitigatedLayers.map(layer => calculateHallucinationRisk(layer, mitigatedStats));

    // Compare results
    console.log('📈 Comparing with baseline...\n');
    const comparison = compareResults(baselineData.hallucinationRisks, mitigatedRisks);

    // Find top improvements
    const topImprovements = comparison
      .filter(c => c.baseline.risk > 10) // Only consider significant baseline risks
      .sort((a, b) => b.improvement.riskReduction - a.improvement.riskReduction)
      .slice(0, 5);

    console.log('Top 5 Risk Reductions:\n');
    topImprovements.forEach((comp, idx) => {
      console.log(`${idx + 1}. Layer ${comp.layer}:`);
      console.log(`   Risk: ${comp.baseline.risk.toFixed(1)}% → ${comp.mitigated.risk.toFixed(1)}% (-${comp.improvement.riskReduction.toFixed(1)}pp, ${comp.improvement.riskReductionPercent.toFixed(1)}%)`);
      console.log(`   Dominance: ${comp.baseline.dominance.toFixed(0)}× → ${comp.mitigated.dominance.toFixed(0)}× (-${comp.improvement.dominanceReductionPercent.toFixed(1)}%)`);
      console.log(`   Attention: ${comp.baseline.attentionStrength.toFixed(1)} → ${comp.mitigated.attentionStrength.toFixed(1)} (+${comp.improvement.attentionIncreasePercent.toFixed(1)}%)\n`);
    });

    // Overall statistics
    const avgRiskReduction = comparison.reduce((sum, c) => sum + c.improvement.riskReduction, 0) / comparison.length;
    const avgDominanceReduction = comparison.reduce((sum, c) => sum + c.improvement.dominanceReductionPercent, 0) / comparison.length;

    console.log('Overall Impact:');
    console.log(`  Average risk reduction: ${avgRiskReduction.toFixed(2)}pp`);
    console.log(`  Average dominance reduction: ${avgDominanceReduction.toFixed(1)}%\n`);

    // Store results
    results.push({
      strategy: strategy.name,
      parameters: strategy.params,
      globalStatistics: mitigatedStats,
      risks: mitigatedRisks,
      comparison,
      summary: {
        avgRiskReduction,
        avgDominanceReduction,
        topImprovements: topImprovements.map(t => ({
          layer: t.layer,
          riskReduction: t.improvement.riskReduction,
          dominanceReduction: t.improvement.dominanceReductionPercent,
        })),
      },
    });
  }

  // Save results
  const outputPath = path.join(__dirname, '../../../../research-output/phase3/attention-amplification-results.json');
  const outputDir = path.dirname(outputPath);
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  const output = {
    timestamp: Date.now(),
    strategy: 'Attention Amplification',
    baseline: {
      source: 'phase2a/analysis-results.json',
      globalStatistics: baselineData.globalStatistics,
    },
    results,
  };

  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));

  console.log('═'.repeat(80));
  console.log('✅ PHASE 3.1 COMPLETE');
  console.log('═'.repeat(80) + '\n');
  console.log(`💾 Results saved to: ${outputPath}\n`);

  // Identify best strategy
  const bestStrategy = results.reduce((best, current) =>
    current.summary.avgRiskReduction > best.summary.avgRiskReduction ? current : best
  );

  console.log(`🏆 Best Strategy: ${bestStrategy.strategy}`);
  console.log(`   Average Risk Reduction: ${bestStrategy.summary.avgRiskReduction.toFixed(2)}pp`);
  console.log(`   Average Dominance Reduction: ${bestStrategy.summary.avgDominanceReduction.toFixed(1)}%\n`);
}

// Export for use in other modules
export { applyAttentionAmplification, MitigationParameters };

// Run if executed directly
if (require.main === module) {
  runPhase3AttentionAmplification().catch(console.error);
}
