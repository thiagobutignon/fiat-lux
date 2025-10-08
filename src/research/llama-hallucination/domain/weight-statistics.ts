/**
 * Weight Statistics Types
 *
 * Types for statistical analysis of weight patterns in Llama 3.1
 */

/**
 * Statistical measures for a weight tensor
 */
export interface WeightStatistics {
  // Basic statistics
  mean: number;
  std: number;
  min: number;
  max: number;
  median: number;

  // Distribution measures
  skewness: number;
  kurtosis: number;

  // Sparsity measures
  sparsity: number; // Percentage of weights near zero (< 1e-6)
  effectiveRank: number; // Estimate of effective rank

  // Norm measures
  l1Norm: number;
  l2Norm: number;
  lInfNorm: number;

  // Advanced measures
  spectralNorm?: number; // Largest singular value (if computed)
  frobeniusNorm: number; // L2 norm of entire matrix

  // Quantization info
  quantizationError?: number; // If dequantized from Q4/Q6
  bitsPerElement: number; // Effective bits per element
}

/**
 * Layer-specific weight analysis
 */
export interface LayerWeightAnalysis {
  layerIndex: number;
  layerType: 'attention' | 'ffn' | 'norm' | 'embedding' | 'output';

  // Attention components (if applicable)
  attention?: {
    query: WeightStatistics;
    key: WeightStatistics;
    value: WeightStatistics;
    output: WeightStatistics;
  };

  // FFN components (if applicable)
  ffn?: {
    gate: WeightStatistics;
    up: WeightStatistics;
    down: WeightStatistics;
  };

  // Normalization (if applicable)
  norm?: {
    attnNorm: WeightStatistics;
    ffnNorm: WeightStatistics;
  };

  // Overall layer statistics
  totalParameters: number;
  memoryFootprint: number; // Bytes

  // Comparative measures
  relativeMagnitude: number; // Compared to other layers
  layerStability: number; // Based on norm ratios
}

/**
 * Attention head-specific analysis
 */
export interface AttentionHeadAnalysis {
  layerIndex: number;
  headIndex: number;

  // Weight statistics for this head's Q, K, V projections
  queryWeights: WeightStatistics;
  keyWeights: WeightStatistics;
  valueWeights: WeightStatistics;

  // Head specialization metrics
  headEntropy: number; // Diversity of attention patterns
  headSpecialization: number; // 0-1, how specialized vs general

  // Correlation with other heads
  mostSimilarHead: { layer: number; head: number; similarity: number };

  // Behavioral classification (to be determined from activations)
  headType?: 'positional' | 'semantic' | 'syntactic' | 'mixed' | 'unknown';
}

/**
 * FFN gate activation analysis
 */
export interface FFNGateAnalysis {
  layerIndex: number;

  // Gate weight statistics
  gateWeights: WeightStatistics;
  upWeights: WeightStatistics;
  downWeights: WeightStatistics;

  // Gate activation patterns (from inference)
  averageGateActivation?: number[];
  topKActiveGates?: number[]; // Indices of most active gates

  // Specialization
  gateSpecialization: number; // How specialized are individual gates
  crossLayerSimilarity: number; // Similarity to other layers' gates
}

/**
 * Complete model weight profile
 */
export interface ModelWeightProfile {
  modelName: string;
  quantizationType: string;
  totalParameters: number;

  // Layer-wise analysis
  layers: LayerWeightAnalysis[];

  // Head-wise analysis
  attentionHeads: AttentionHeadAnalysis[];

  // FFN analysis
  ffnGates: FFNGateAnalysis[];

  // Global statistics
  globalStatistics: {
    meanWeight: number;
    stdWeight: number;
    overallSparsity: number;
    totalL1Norm: number;
    totalL2Norm: number;
  };

  // Layer comparisons
  earlyLayers: WeightStatistics; // Layers 0-10
  middleLayers: WeightStatistics; // Layers 11-21
  lateLayers: WeightStatistics; // Layers 22-31

  // Quantization impact
  quantizationImpact?: {
    averageQuantizationError: number;
    worstLayer: { index: number; error: number };
    bestLayer: { index: number; error: number };
  };
}

/**
 * Weight extraction configuration
 */
export interface WeightExtractionConfig {
  // Which components to analyze
  analyzeAttention: boolean;
  analyzeFFN: boolean;
  analyzeNorms: boolean;

  // Statistical options
  computeSpectralNorm: boolean; // Expensive, requires SVD
  computeHeadSimilarity: boolean;

  // Memory management
  batchSize: number; // Process tensors in batches
  maxMemoryMB: number; // Max memory to use

  // Quantization
  dequantize: boolean; // Convert quantized weights to FP32
  compareWithFP16: boolean; // If available, compare with FP16 version
}
