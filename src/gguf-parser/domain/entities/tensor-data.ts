/**
 * Tensor Data Types
 * Represents actual tensor weight data after dequantization
 */

import { GGMLType } from './gguf-metadata';

export interface TensorData {
  name: string;
  shape: number[];
  type: GGMLType;
  data: Float32Array;
  originalType: GGMLType; // Before dequantization
}

export interface WeightStatistics {
  mean: number;
  stdDev: number;
  variance: number;
  min: number;
  max: number;
  median: number;
  zeros: number;
  sparsity: number; // Percentage of near-zero values
  l1Norm: number;
  l2Norm: number;
  histogram: {
    bins: number[];
    counts: number[];
  };
}

export interface LayerWeights {
  layerIndex: number;

  // Normalization
  attentionNorm?: TensorData;
  ffnNorm?: TensorData;

  // Attention
  attentionQ?: TensorData;
  attentionK?: TensorData;
  attentionV?: TensorData;
  attentionOutput?: TensorData;

  // Feed-Forward Network
  ffnGate?: TensorData;
  ffnUp?: TensorData;
  ffnDown?: TensorData;
}

export interface DequantizationContext {
  blockSize: number;
  scalesCount: number;
  quantsCount: number;
}
