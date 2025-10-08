/**
 * Phase 1 - Task 1.3: FFN Gate Specialization & Cross-Layer Similarity Analysis
 *
 * Analyzes:
 * - FFN gate activation patterns across layers
 * - Cross-layer similarity (cosine similarity of weight distributions)
 * - Layer clustering (which layers behave similarly)
 * - Transition points (where model behavior changes)
 *
 * Usage:
 *   tsx src/research/llama-hallucination/demos/phase1-task1.3-ffn-analysis.ts <weight-profile.json>
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
  ffn?: FFNWeights;
  attention?: AttentionWeights;
}

interface WeightProfile {
  modelName: string;
  quantizationType: string;
  totalParameters: number;
  layers: LayerProfile[];
}

// Calculate cosine similarity between two vectors (represented as WeightStats)
function cosineSimilarity(stats1: WeightStats, stats2: WeightStats): number {
  // Create feature vectors from weight statistics
  const vec1 = [
    stats1.mean,
    stats1.std,
    stats1.skewness,
    stats1.kurtosis,
    stats1.sparsity,
    stats1.l1Norm,
    stats1.l2Norm,
  ];

  const vec2 = [
    stats2.mean,
    stats2.std,
    stats2.skewness,
    stats2.kurtosis,
    stats2.sparsity,
    stats2.l1Norm,
    stats2.l2Norm,
  ];

  // Compute dot product
  let dotProduct = 0;
  let norm1 = 0;
  let norm2 = 0;

  for (let i = 0; i < vec1.length; i++) {
    dotProduct += vec1[i] * vec2[i];
    norm1 += vec1[i] * vec1[i];
    norm2 += vec2[i] * vec2[i];
  }

  norm1 = Math.sqrt(norm1);
  norm2 = Math.sqrt(norm2);

  if (norm1 === 0 || norm2 === 0) return 0;

  return dotProduct / (norm1 * norm2);
}

// Compute pairwise similarity matrix for FFN gates
function computeFFNGateSimilarityMatrix(layers: LayerProfile[]): number[][] {
  const ffnLayers = layers.filter((l) => l.ffn);
  const n = ffnLayers.length;
  const matrix: number[][] = Array(n)
    .fill(0)
    .map(() => Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const sim = cosineSimilarity(ffnLayers[i].ffn!.gate, ffnLayers[j].ffn!.gate);
      matrix[i][j] = sim;
      matrix[j][i] = sim;
    }
  }

  return matrix;
}

// Find layer clusters based on similarity threshold
function findLayerClusters(similarityMatrix: number[][], threshold: number): number[][] {
  const n = similarityMatrix.length;
  const visited = new Set<number>();
  const clusters: number[][] = [];

  for (let i = 0; i < n; i++) {
    if (visited.has(i)) continue;

    const cluster = [i];
    visited.add(i);

    for (let j = i + 1; j < n; j++) {
      if (!visited.has(j) && similarityMatrix[i][j] >= threshold) {
        cluster.push(j);
        visited.add(j);
      }
    }

    clusters.push(cluster);
  }

  return clusters;
}

// Compute gate activation specialization (diversity across layers)
function computeGateSpecialization(layers: LayerProfile[]): {
  meanDiversity: number;
  maxDiversity: number;
  transitionPoints: number[];
} {
  const ffnLayers = layers.filter((l) => l.ffn);
  const diversities: number[] = [];
  const transitionPoints: number[] = [];

  for (let i = 0; i < ffnLayers.length - 1; i++) {
    const sim = cosineSimilarity(ffnLayers[i].ffn!.gate, ffnLayers[i + 1].ffn!.gate);
    const diversity = 1 - sim; // Lower similarity = higher diversity
    diversities.push(diversity);

    // Detect transition points (sudden changes in similarity)
    if (diversity > 0.3) {
      // Threshold for "significant" transition
      transitionPoints.push(ffnLayers[i].layerIndex);
    }
  }

  return {
    meanDiversity: diversities.reduce((a, b) => a + b, 0) / diversities.length,
    maxDiversity: Math.max(...diversities),
    transitionPoints,
  };
}

// Analyze FFN gate activation patterns
function analyzeFunctionSpecialization(layers: LayerProfile[]): {
  earlyLayerPattern: string;
  middleLayerPattern: string;
  lateLayerPattern: string;
} {
  const ffnLayers = layers.filter((l) => l.ffn);
  const third = Math.floor(ffnLayers.length / 3);

  // Early layers
  const earlyGates = ffnLayers.slice(0, third).map((l) => l.ffn!.gate);
  const earlyMeanSparsity =
    earlyGates.reduce((sum, g) => sum + g.sparsity, 0) / earlyGates.length;
  const earlyMeanKurtosis =
    earlyGates.reduce((sum, g) => sum + g.kurtosis, 0) / earlyGates.length;

  // Middle layers
  const middleGates = ffnLayers.slice(third, third * 2).map((l) => l.ffn!.gate);
  const middleMeanSparsity =
    middleGates.reduce((sum, g) => sum + g.sparsity, 0) / middleGates.length;
  const middleMeanKurtosis =
    middleGates.reduce((sum, g) => sum + g.kurtosis, 0) / middleGates.length;

  // Late layers
  const lateGates = ffnLayers.slice(third * 2).map((l) => l.ffn!.gate);
  const lateMeanSparsity =
    lateGates.reduce((sum, g) => sum + g.sparsity, 0) / lateGates.length;
  const lateMeanKurtosis =
    lateGates.reduce((sum, g) => sum + g.kurtosis, 0) / lateGates.length;

  // Classify patterns based on sparsity and kurtosis
  const classifyPattern = (sparsity: number, kurtosis: number): string => {
    if (sparsity < 0.01 && kurtosis > 5) return 'Dense, Peaky (specialized neurons)';
    if (sparsity < 0.01 && kurtosis < 5) return 'Dense, Distributed (broad activation)';
    if (sparsity > 0.02 && kurtosis > 5) return 'Sparse, Peaky (highly selective)';
    if (sparsity > 0.02 && kurtosis < 5) return 'Sparse, Distributed (selective activation)';
    return 'Moderate (balanced activation)';
  };

  return {
    earlyLayerPattern: classifyPattern(earlyMeanSparsity, earlyMeanKurtosis),
    middleLayerPattern: classifyPattern(middleMeanSparsity, middleMeanKurtosis),
    lateLayerPattern: classifyPattern(lateMeanSparsity, lateMeanKurtosis),
  };
}

async function runTask13Analysis() {
  console.log('🔬 Task 1.3: FFN Gate Specialization & Cross-Layer Similarity Analysis\n');

  // Get weight profile path from args
  const profilePath = process.argv[2];

  if (!profilePath) {
    console.error('❌ Error: No weight profile provided');
    console.log('\nUsage:');
    console.log(
      '  tsx src/research/llama-hallucination/demos/phase1-task1.3-ffn-analysis.ts <weight-profile.json>'
    );
    console.log('\nExample:');
    console.log(
      '  tsx src/research/llama-hallucination/demos/phase1-task1.3-ffn-analysis.ts research-output/phase1/weight-profile-1234567890.json'
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
  // Part 1: Cross-Layer Similarity Analysis
  // =================================================================
  console.log('═══════════════════════════════════════════════════════════');
  console.log('Part 1: Cross-Layer Similarity Analysis');
  console.log('═══════════════════════════════════════════════════════════\n');

  console.log('Computing pairwise similarity matrix for FFN gates...');
  const similarityMatrix = computeFFNGateSimilarityMatrix(profile.layers);
  const ffnLayers = profile.layers.filter((l) => l.ffn);

  console.log(`✓ Computed ${similarityMatrix.length}x${similarityMatrix.length} similarity matrix\n`);

  // Display similarity heatmap (text-based)
  console.log('FFN Gate Similarity Heatmap (first 16 layers):');
  console.log('  Layer | ' + Array.from({ length: 16 }, (_, i) => i.toString().padStart(4)).join(''));
  console.log('  ------|' + Array.from({ length: 16 }, () => '----').join(''));

  for (let i = 0; i < Math.min(16, similarityMatrix.length); i++) {
    const row = similarityMatrix[i].slice(0, 16);
    const rowStr = row.map((v) => (v >= 0.9 ? ' ■■■' : v >= 0.7 ? ' ▓▓▓' : v >= 0.5 ? ' ▒▒▒' : ' ░░░')).join('');
    console.log(`  ${i.toString().padStart(5)} |${rowStr}`);
  }
  console.log('\n  Legend: ■■■ = 0.9+ (very similar)  ▓▓▓ = 0.7-0.9  ▒▒▒ = 0.5-0.7  ░░░ = <0.5\n');

  // Find most and least similar layer pairs
  let maxSim = -1;
  let maxPair = [0, 0];
  let minSim = 2;
  let minPair = [0, 0];

  for (let i = 0; i < similarityMatrix.length; i++) {
    for (let j = i + 1; j < similarityMatrix.length; j++) {
      if (similarityMatrix[i][j] > maxSim) {
        maxSim = similarityMatrix[i][j];
        maxPair = [i, j];
      }
      if (similarityMatrix[i][j] < minSim) {
        minSim = similarityMatrix[i][j];
        minPair = [i, j];
      }
    }
  }

  console.log('Most Similar Layer Pair:');
  console.log(`  Layers ${ffnLayers[maxPair[0]].layerIndex} ↔ ${ffnLayers[maxPair[1]].layerIndex}`);
  console.log(`  Similarity: ${maxSim.toFixed(4)}\n`);

  console.log('Least Similar Layer Pair:');
  console.log(`  Layers ${ffnLayers[minPair[0]].layerIndex} ↔ ${ffnLayers[minPair[1]].layerIndex}`);
  console.log(`  Similarity: ${minSim.toFixed(4)}\n`);

  // =================================================================
  // Part 2: Layer Clustering
  // =================================================================
  console.log('═══════════════════════════════════════════════════════════');
  console.log('Part 2: Layer Clustering');
  console.log('═══════════════════════════════════════════════════════════\n');

  const threshold = 0.8;
  console.log(`Clustering layers with similarity threshold ≥ ${threshold}...\n`);

  const clusters = findLayerClusters(similarityMatrix, threshold);

  console.log(`Found ${clusters.length} clusters:\n`);

  for (let i = 0; i < clusters.length; i++) {
    const clusterLayers = clusters[i].map((idx) => ffnLayers[idx].layerIndex);
    console.log(`  Cluster ${i + 1}: [${clusterLayers.join(', ')}]`);
    console.log(`    Size: ${clusterLayers.length} layers`);

    // Compute average similarity within cluster
    let avgSim = 0;
    let count = 0;
    for (let j = 0; j < clusters[i].length; j++) {
      for (let k = j + 1; k < clusters[i].length; k++) {
        avgSim += similarityMatrix[clusters[i][j]][clusters[i][k]];
        count++;
      }
    }
    if (count > 0) {
      console.log(`    Avg Internal Similarity: ${(avgSim / count).toFixed(4)}`);
    }
    console.log();
  }

  // =================================================================
  // Part 3: Gate Specialization Analysis
  // =================================================================
  console.log('═══════════════════════════════════════════════════════════');
  console.log('Part 3: Gate Specialization Analysis');
  console.log('═══════════════════════════════════════════════════════════\n');

  const specialization = computeGateSpecialization(profile.layers);

  console.log('Gate Diversity Metrics:');
  console.log(`  Mean Diversity (1 - similarity): ${specialization.meanDiversity.toFixed(4)}`);
  console.log(`  Max Diversity: ${specialization.maxDiversity.toFixed(4)}\n`);

  console.log('Transition Points (sudden changes in gate behavior):');
  if (specialization.transitionPoints.length === 0) {
    console.log('  None detected (gradual specialization)\n');
  } else {
    for (const point of specialization.transitionPoints) {
      console.log(`  Layer ${point} → ${point + 1}: Significant pattern shift detected`);
    }
    console.log();
  }

  // =================================================================
  // Part 4: Functional Specialization Patterns
  // =================================================================
  console.log('═══════════════════════════════════════════════════════════');
  console.log('Part 4: Functional Specialization Patterns');
  console.log('═══════════════════════════════════════════════════════════\n');

  const patterns = analyzeFunctionSpecialization(profile.layers);

  console.log('Layer Range Activation Patterns:\n');

  console.log('  Early Layers (0-10):');
  console.log(`    Pattern: ${patterns.earlyLayerPattern}`);
  console.log('    Interpretation: Processing low-level features\n');

  console.log('  Middle Layers (11-21):');
  console.log(`    Pattern: ${patterns.middleLayerPattern}`);
  console.log('    Interpretation: Abstract representation building\n');

  console.log('  Late Layers (22-31):');
  console.log(`    Pattern: ${patterns.lateLayerPattern}`);
  console.log('    Interpretation: High-level reasoning and output generation\n');

  // =================================================================
  // Save detailed results
  // =================================================================
  const results = {
    task: 'Task 1.3: FFN Gate Specialization & Cross-Layer Similarity',
    timestamp: new Date().toISOString(),
    model: profile.modelName,
    similarityMatrix,
    layerIndices: ffnLayers.map((l) => l.layerIndex),
    clusters: clusters.map((c) => c.map((idx) => ffnLayers[idx].layerIndex)),
    specialization,
    patterns,
    mostSimilarPair: {
      layers: [ffnLayers[maxPair[0]].layerIndex, ffnLayers[maxPair[1]].layerIndex],
      similarity: maxSim,
    },
    leastSimilarPair: {
      layers: [ffnLayers[minPair[0]].layerIndex, ffnLayers[minPair[1]].layerIndex],
      similarity: minSim,
    },
  };

  const outputDir = 'research-output/phase1';
  const outputPath = path.join(outputDir, `task1.3-ffn-specialization-${Date.now()}.json`);
  fs.writeFileSync(outputPath, JSON.stringify(results, null, 2));

  console.log(`✅ Detailed results saved to: ${outputPath}\n`);

  // =================================================================
  // Key Findings Summary
  // =================================================================
  console.log('═══════════════════════════════════════════════════════════');
  console.log('KEY FINDINGS - Task 1.3');
  console.log('═══════════════════════════════════════════════════════════\n');

  console.log('✅ Cross-Layer Similarity:');
  console.log(`   • Most similar layers: ${ffnLayers[maxPair[0]].layerIndex} ↔ ${ffnLayers[maxPair[1]].layerIndex} (${maxSim.toFixed(4)})`);
  console.log(`   • Least similar layers: ${ffnLayers[minPair[0]].layerIndex} ↔ ${ffnLayers[minPair[1]].layerIndex} (${minSim.toFixed(4)})`);
  console.log(`   • Found ${clusters.length} distinct layer clusters\n`);

  console.log('✅ Gate Specialization:');
  console.log(`   • Mean diversity: ${specialization.meanDiversity.toFixed(4)}`);
  console.log(`   • Transition points: ${specialization.transitionPoints.length}`);
  if (specialization.transitionPoints.length > 0) {
    console.log(`   • Major shifts at layers: ${specialization.transitionPoints.join(', ')}`);
  }
  console.log();

  console.log('✅ Functional Patterns:');
  console.log(`   • Early: ${patterns.earlyLayerPattern}`);
  console.log(`   • Middle: ${patterns.middleLayerPattern}`);
  console.log(`   • Late: ${patterns.lateLayerPattern}\n`);

  console.log('🎉 Task 1.3 Complete!\n');
}

runTask13Analysis().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
