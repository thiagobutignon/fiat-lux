/**
 * Phase 2A: Weight-Based Activation Analysis
 *
 * Predicts activations and validates Phase 1 hypotheses using ONLY weight statistics.
 * No runtime inference required - everything computed from weight patterns.
 *
 * Validates 5 hypotheses:
 * 1. Bimodal Value Sparsity causes information loss
 * 2. Progressive Attention Weakening reduces context integration
 * 3. Value Amplification amplifies errors
 * 4. Key Matching Deterioration weakens retrieval
 * 5. Layer Norm Amplification in FFN causes overfitting
 */

import * as fs from 'fs';
import * as path from 'path';

// ============================================================================
// Type Definitions (matching actual JSON structure)
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

// Processed layer profile
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

// ============================================================================
// Activation Prediction from Weight Statistics
// ============================================================================

/**
 * Predicts attention activation characteristics from weight statistics
 *
 * Theory: Attention activation strength ≈ sqrt(|Q| * |K|) * (1 - sparsity_V)
 * - Higher Q/K norms → stronger attention scores
 * - Higher V sparsity → more information loss
 */
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

  // Predicted attention score magnitude
  const predictedStrength = Math.sqrt(qNorm * kNorm);

  // Information retention (inverse of sparsity)
  const informationRetention = 1 - vSparsity;

  // Entropy estimation: high sparsity + high std = high uncertainty
  const entropy = (vSparsity * attention.value.std) / Math.max(attention.value.l2Norm, 1e-8);

  return { predictedStrength, informationRetention, entropy };
}

/**
 * Predicts FFN activation characteristics from weight statistics
 *
 * Theory: FFN activation strength ≈ |gate| * |up| * |down|
 * - Gate acts as gating mechanism (like sigmoid/ReLU)
 * - Up projects to intermediate dimension
 * - Down projects back to model dimension
 */
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

  // Predicted FFN strength (product of transformation magnitudes)
  const predictedStrength = gateNorm * upNorm * downNorm;

  // Amplification: how much FFN amplifies input
  const amplificationFactor = upNorm * downNorm;

  // Nonlinearity estimate: kurtosis indicates heavy tails (ReLU-like behavior)
  const nonlinearityEstimate = (ffn.gate.kurtosis + ffn.up.kurtosis) / 2;

  return { predictedStrength, amplificationFactor, nonlinearityEstimate };
}

/**
 * Computes attention-to-FFN dominance ratio
 *
 * Ratio < 1: Attention dominates (good for context integration)
 * Ratio > 1: FFN dominates (risk of overfitting/memorization)
 */
function computeDominanceRatio(
  attnActivation: ReturnType<typeof predictAttentionActivation>,
  ffnActivation: ReturnType<typeof predictFFNActivation>
): number {
  return ffnActivation.predictedStrength / Math.max(attnActivation.predictedStrength, 1e-8);
}

// ============================================================================
// Hallucination Risk Scoring
// ============================================================================

interface HallucinationRisk {
  layer: number;
  totalRisk: number;  // 0-100
  components: {
    valueSparsityRisk: number;      // Hypothesis 1
    attentionWeakeningRisk: number; // Hypothesis 2
    valueAmplificationRisk: number; // Hypothesis 3
    keyMatchingRisk: number;        // Hypothesis 4
    normAmplificationRisk: number;  // Hypothesis 5
  };
  dominanceRatio: number;
  predictedAttentionStrength: number;
  predictedFFNStrength: number;
}

/**
 * Calculates comprehensive hallucination risk score for a layer
 */
function calculateHallucinationRisk(
  layer: LayerProfile,
  globalStats: {
    avgAttentionStrength: number;
    avgFFNStrength: number;
    avgValueSparsity: number;
    avgQKRatio: number;
    avgFFNNorm: number;
  }
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
  const dominanceRatio = computeDominanceRatio(attnActivation, ffnActivation);

  // Risk Component 1: Value Sparsity (Hypothesis 1)
  // High sparsity = more information loss
  const valueSparsityRisk = Math.min(100, (layer.attention.valueSparsity / 0.05) * 100);

  // Risk Component 2: Attention Weakening (Hypothesis 2)
  // Lower than average attention strength = weaker context integration
  const attentionWeakeningRisk = Math.max(0, (1 - attnActivation.predictedStrength / globalStats.avgAttentionStrength) * 100);

  // Risk Component 3: Value Amplification (Hypothesis 3)
  // High value norms amplify errors
  const valueAmplificationRisk = Math.min(100, (layer.attention.value.l2Norm / 100) * 100);

  // Risk Component 4: Key Matching Deterioration (Hypothesis 4)
  // Low Q/K ratio = poor key-query matching
  const qkRatio = layer.attention.qkRatio;
  const keyMatchingRisk = Math.max(0, (1 - qkRatio / globalStats.avgQKRatio) * 100);

  // Risk Component 5: Norm Amplification (Hypothesis 5)
  // FFN norm above average = potential overfitting
  const ffnNormRatio = layer.norms.ffnNorm.mean / globalStats.avgFFNNorm;
  const normAmplificationRisk = Math.max(0, (ffnNormRatio - 1) * 100);

  // Total risk: weighted average of components
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
// Hypothesis Validation
// ============================================================================

interface HypothesisValidation {
  hypothesis: string;
  validated: boolean;
  confidence: number;  // 0-100%
  evidence: string[];
  quantitativeMetrics: Record<string, number>;
}

function validateHypothesis1_BimodalSparsity(layers: LayerProfile[]): HypothesisValidation {
  const valueSparsities = layers
    .filter(l => l.attention)
    .map(l => l.attention!.valueSparsity);

  // Check for bimodal distribution (two clusters)
  const sorted = [...valueSparsities].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];

  const lowCluster = sorted.filter(s => s < median);
  const highCluster = sorted.filter(s => s >= median);

  const lowMean = lowCluster.reduce((a, b) => a + b, 0) / lowCluster.length;
  const highMean = highCluster.reduce((a, b) => a + b, 0) / highCluster.length;

  // Bimodal if clusters are well-separated
  const separation = highMean - lowMean;
  const validated = separation > 0.01; // 1% separation threshold

  return {
    hypothesis: 'Bimodal Value Sparsity',
    validated,
    confidence: Math.min(100, separation * 5000), // Scale to 0-100%
    evidence: [
      `Low cluster mean: ${(lowMean * 100).toFixed(3)}%`,
      `High cluster mean: ${(highMean * 100).toFixed(3)}%`,
      `Separation: ${(separation * 100).toFixed(3)}%`,
      `Layers in low cluster: ${lowCluster.length}`,
      `Layers in high cluster: ${highCluster.length}`,
    ],
    quantitativeMetrics: {
      lowClusterMean: lowMean,
      highClusterMean: highMean,
      separation,
      lowClusterSize: lowCluster.length,
      highClusterSize: highCluster.length,
    },
  };
}

function validateHypothesis2_AttentionWeakening(layers: LayerProfile[]): HypothesisValidation {
  const qkRatios = layers
    .filter(l => l.attention)
    .map((l, idx) => ({ layer: idx, ratio: l.attention!.qkRatio }));

  // Check for declining trend in late layers
  const earlyLayers = qkRatios.slice(0, 11); // Layers 0-10
  const lateLayers = qkRatios.slice(21);      // Layers 21-31

  const earlyAvg = earlyLayers.reduce((sum, l) => sum + l.ratio, 0) / earlyLayers.length;
  const lateAvg = lateLayers.reduce((sum, l) => sum + l.ratio, 0) / lateLayers.length;

  const decline = (earlyAvg - lateAvg) / earlyAvg;
  const validated = decline > 0.1; // 10% decline threshold

  return {
    hypothesis: 'Progressive Attention Weakening',
    validated,
    confidence: Math.min(100, decline * 500), // Scale to 0-100%
    evidence: [
      `Early layers (0-10) avg Q/K ratio: ${earlyAvg.toFixed(6)}`,
      `Late layers (21-31) avg Q/K ratio: ${lateAvg.toFixed(6)}`,
      `Decline: ${(decline * 100).toFixed(2)}%`,
      `Trend: ${decline > 0 ? 'Weakening' : 'Strengthening'}`,
    ],
    quantitativeMetrics: {
      earlyAverage: earlyAvg,
      lateAverage: lateAvg,
      decline,
      percentDecline: decline * 100,
    },
  };
}

function validateHypothesis3_ValueAmplification(layers: LayerProfile[]): HypothesisValidation {
  const valueNorms = layers
    .filter(l => l.attention)
    .map((l, idx) => ({ layer: idx, norm: l.attention!.value.l2Norm }));

  const firstLayer = valueNorms[0].norm;
  const lastLayer = valueNorms[valueNorms.length - 1].norm;

  const amplification = (lastLayer - firstLayer) / firstLayer;
  const validated = amplification > 0.5; // 50% amplification threshold

  return {
    hypothesis: 'Value Amplification',
    validated,
    confidence: Math.min(100, amplification * 100), // Scale to 0-100%
    evidence: [
      `Layer 0 value norm: ${firstLayer.toFixed(2)}`,
      `Layer 31 value norm: ${lastLayer.toFixed(2)}`,
      `Amplification: ${(amplification * 100).toFixed(2)}%`,
      `Trend: ${amplification > 0 ? 'Increasing' : 'Decreasing'}`,
    ],
    quantitativeMetrics: {
      firstLayerNorm: firstLayer,
      lastLayerNorm: lastLayer,
      amplification,
      percentAmplification: amplification * 100,
    },
  };
}

function validateHypothesis4_KeyMatchingDeterioration(layers: LayerProfile[]): HypothesisValidation {
  const keyQueryAlignment = layers
    .filter(l => l.attention)
    .map((l, idx) => {
      const qNorm = l.attention!.query.l2Norm;
      const kNorm = l.attention!.key.l2Norm;
      const alignment = Math.min(qNorm, kNorm) / Math.max(qNorm, kNorm);
      return { layer: idx, alignment };
    });

  const earlyLayers = keyQueryAlignment.slice(0, 11);
  const lateLayers = keyQueryAlignment.slice(21);

  const earlyAvg = earlyLayers.reduce((sum, l) => sum + l.alignment, 0) / earlyLayers.length;
  const lateAvg = lateLayers.reduce((sum, l) => sum + l.alignment, 0) / lateLayers.length;

  const deterioration = (earlyAvg - lateAvg) / earlyAvg;
  const validated = deterioration > 0.05; // 5% deterioration threshold

  return {
    hypothesis: 'Key Matching Deterioration',
    validated,
    confidence: Math.min(100, deterioration * 1000), // Scale to 0-100%
    evidence: [
      `Early layers (0-10) avg Q-K alignment: ${earlyAvg.toFixed(6)}`,
      `Late layers (21-31) avg Q-K alignment: ${lateAvg.toFixed(6)}`,
      `Deterioration: ${(deterioration * 100).toFixed(2)}%`,
      `Trend: ${deterioration > 0 ? 'Deteriorating' : 'Improving'}`,
    ],
    quantitativeMetrics: {
      earlyAverage: earlyAvg,
      lateAverage: lateAvg,
      deterioration,
      percentDeterioration: deterioration * 100,
    },
  };
}

function validateHypothesis5_NormAmplification(layers: LayerProfile[]): HypothesisValidation {
  const ffnNorms = layers
    .filter(l => l.norms)
    .map((l, idx) => ({ layer: idx, norm: l.norms!.ffnNorm.mean }));

  const avgNorm = ffnNorms.reduce((sum, l) => sum + l.norm, 0) / ffnNorms.length;
  const layer31Norm = ffnNorms[31]?.norm || 0;

  const amplification = (layer31Norm - avgNorm) / avgNorm;
  const validated = amplification > 0.15; // 15% above average threshold

  // Check late layer trend
  const lateLayers = ffnNorms.slice(21);
  const lateAvg = lateLayers.reduce((sum, l) => sum + l.norm, 0) / lateLayers.length;
  const lateAmplification = (lateAvg - avgNorm) / avgNorm;

  return {
    hypothesis: 'Layer Norm Amplification',
    validated,
    confidence: Math.min(100, amplification * 300), // Scale to 0-100%
    evidence: [
      `Average FFN norm across all layers: ${avgNorm.toFixed(6)}`,
      `Layer 31 FFN norm: ${layer31Norm.toFixed(6)}`,
      `Layer 31 amplification: ${(amplification * 100).toFixed(2)}%`,
      `Late layers (21-31) avg amplification: ${(lateAmplification * 100).toFixed(2)}%`,
    ],
    quantitativeMetrics: {
      globalAverage: avgNorm,
      layer31Norm,
      layer31Amplification: amplification,
      lateLayersAmplification: lateAmplification,
      percentAmplification: amplification * 100,
    },
  };
}

// ============================================================================
// Data Preprocessing
// ============================================================================

/**
 * Converts raw weight profile to processed layer profiles
 * Groups components by layer index and computes derived metrics
 */
function preprocessWeightProfile(raw: RawWeightProfile): LayerProfile[] {
  // Group by layer index
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
// Main Analysis Pipeline
// ============================================================================

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

async function runPhase2A() {
  console.log('╔══════════════════════════════════════════════════════════════════════════════╗');
  console.log('║            PHASE 2A: WEIGHT-BASED ACTIVATION ANALYSIS                        ║');
  console.log('╚══════════════════════════════════════════════════════════════════════════════╝\n');

  // Load weight profile
  const profilePath = path.join(
    __dirname,
    '../../../../research-output/phase1/weight-profile-1759955262868.json'
  );

  console.log(`📂 Loading weight profile: ${path.basename(profilePath)}\n`);
  const rawProfile: RawWeightProfile = JSON.parse(fs.readFileSync(profilePath, 'utf-8'));

  console.log(`✓ Model: ${rawProfile.modelName}`);
  console.log(`✓ Quantization: ${rawProfile.quantizationType}`);
  console.log(`✓ Total parameters: ${(rawProfile.totalParameters / 1e9).toFixed(2)}B`);
  console.log(`✓ Raw components: ${rawProfile.layers.length}\n`);

  // Preprocess: group by layer and compute derived metrics
  console.log('🔄 Preprocessing: Grouping components by layer...\n');
  const layers = preprocessWeightProfile(rawProfile);
  console.log(`✓ Processed ${layers.length} layers\n`);

  // Compute global statistics
  console.log('═'.repeat(80));
  console.log('📊 COMPUTING GLOBAL STATISTICS');
  console.log('═'.repeat(80) + '\n');

  const globalStats = computeGlobalStatistics(layers);
  console.log('Global Averages:');
  console.log(`  Attention Strength: ${globalStats.avgAttentionStrength.toFixed(2)}`);
  console.log(`  FFN Strength: ${globalStats.avgFFNStrength.toFixed(2)}`);
  console.log(`  Value Sparsity: ${(globalStats.avgValueSparsity * 100).toFixed(3)}%`);
  console.log(`  Q/K Ratio: ${globalStats.avgQKRatio.toFixed(6)}`);
  console.log(`  FFN Norm Mean: ${globalStats.avgFFNNorm.toFixed(6)}\n`);

  // Calculate hallucination risks
  console.log('═'.repeat(80));
  console.log('🎯 CALCULATING HALLUCINATION RISK SCORES');
  console.log('═'.repeat(80) + '\n');

  const risks = layers.map(layer =>
    calculateHallucinationRisk(layer, globalStats)
  );

  // Find high-risk layers
  const highRiskLayers = risks
    .filter(r => r.totalRisk > 50)
    .sort((a, b) => b.totalRisk - a.totalRisk);

  console.log(`Found ${highRiskLayers.length} high-risk layers (risk > 50):\n`);

  highRiskLayers.slice(0, 5).forEach(risk => {
    console.log(`Layer ${risk.layer}: ${risk.totalRisk.toFixed(1)}% risk`);
    console.log(`  Components:`);
    console.log(`    Value Sparsity: ${risk.components.valueSparsityRisk.toFixed(1)}%`);
    console.log(`    Attention Weakening: ${risk.components.attentionWeakeningRisk.toFixed(1)}%`);
    console.log(`    Value Amplification: ${risk.components.valueAmplificationRisk.toFixed(1)}%`);
    console.log(`    Key Matching: ${risk.components.keyMatchingRisk.toFixed(1)}%`);
    console.log(`    Norm Amplification: ${risk.components.normAmplificationRisk.toFixed(1)}%`);
    console.log(`  FFN/Attention Dominance: ${risk.dominanceRatio.toFixed(2)}x\n`);
  });

  // Validate hypotheses
  console.log('═'.repeat(80));
  console.log('🔬 VALIDATING PHASE 1 HYPOTHESES');
  console.log('═'.repeat(80) + '\n');

  const validations = [
    validateHypothesis1_BimodalSparsity(layers),
    validateHypothesis2_AttentionWeakening(layers),
    validateHypothesis3_ValueAmplification(layers),
    validateHypothesis4_KeyMatchingDeterioration(layers),
    validateHypothesis5_NormAmplification(layers),
  ];

  validations.forEach((validation, idx) => {
    const status = validation.validated ? '✅ VALIDATED' : '❌ REJECTED';
    console.log(`Hypothesis ${idx + 1}: ${validation.hypothesis}`);
    console.log(`Status: ${status} (${validation.confidence.toFixed(1)}% confidence)\n`);
    console.log('Evidence:');
    validation.evidence.forEach(e => console.log(`  • ${e}`));
    console.log('');
  });

  // Summary
  console.log('═'.repeat(80));
  console.log('📈 SUMMARY');
  console.log('═'.repeat(80) + '\n');

  const validatedCount = validations.filter(v => v.validated).length;
  console.log(`Hypotheses Validated: ${validatedCount}/5 (${(validatedCount / 5 * 100).toFixed(0)}%)\n`);

  console.log('Highest Risk Layers:');
  risks
    .sort((a, b) => b.totalRisk - a.totalRisk)
    .slice(0, 3)
    .forEach((risk, idx) => {
      console.log(`  ${idx + 1}. Layer ${risk.layer}: ${risk.totalRisk.toFixed(1)}% (FFN dominance: ${risk.dominanceRatio.toFixed(2)}x)`);
    });

  console.log('\n✅ Phase 2A Complete!\n');

  // Save results
  const outputPath = path.join(
    __dirname,
    '../../../../research-output/phase2a/analysis-results.json'
  );

  const outputDir = path.dirname(outputPath);
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  const results = {
    timestamp: Date.now(),
    model: rawProfile.modelName,
    quantization: rawProfile.quantizationType,
    totalParameters: rawProfile.totalParameters,
    globalStatistics: globalStats,
    hallucinationRisks: risks,
    hypothesisValidations: validations,
    summary: {
      totalLayers: layers.length,
      highRiskLayers: highRiskLayers.length,
      validatedHypotheses: validatedCount,
      validationRate: validatedCount / 5,
    },
  };

  fs.writeFileSync(outputPath, JSON.stringify(results, null, 2));
  console.log(`💾 Results saved to: ${outputPath}\n`);
}

// Run analysis
runPhase2A().catch(console.error);
