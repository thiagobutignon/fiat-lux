/**
 * Transformer Architecture Analyzer
 * Deep analysis of transformer model architecture from GGUF
 */

import { GGUFModel, GGUFArchitectureInfo, TensorInfo } from '../entities/gguf-metadata';
import { TensorShape, groupTensorsByLayer } from '../entities/tensor-shape';

export interface TransformerAnalysis {
  // Model identification
  modelName: string;
  architecture: string;
  totalParameters: string; // Formatted (e.g., "8.03B")
  quantization: string;
  fileSizeGB: number;

  // Transformer architecture
  layers: number;
  attentionHeads: number;
  attentionHeadsKV: number | null; // For grouped-query attention
  embeddingDimension: number;
  vocabSize: number;
  contextLength: number;
  ffnDimension: number | null;

  // Advanced features
  hasGroupedQueryAttention: boolean;
  hasRoPE: boolean;
  ropeFreqBase: number | null;
  ropeScaling: string | null;

  // Memory estimates
  memoryUsageGB: number;
  kvCacheGB: number; // At max context length

  // Tensor breakdown
  tensorsByLayer: {
    layer: number;
    tensors: string[];
    totalParams: string;
  }[];

  tensorsByType: {
    type: string;
    count: number;
    totalParams: string;
  }[];

  // Special tokens
  specialTokens: {
    bos?: number;
    eos?: number;
    pad?: number;
    unk?: number;
  };
}

export class TransformerAnalyzer {
  analyze(model: GGUFModel, fileSizeBytes: bigint): TransformerAnalysis {
    const arch = model.architecture;
    const tensors = model.tensors;

    // Basic identification
    const modelName = arch.name || arch.basename || 'Unknown';
    const architecture = arch.architecture || 'Unknown';
    const totalParameters = this.formatParameters(model.totalParameters);
    const quantization = model.quantizationType;
    const fileSizeGB = Number(fileSizeBytes) / (1024 ** 3);

    // Core architecture parameters
    const layers = arch.blockCount || 0;
    const attentionHeads = arch.headCount || 0;
    const attentionHeadsKV = arch.headCountKV || null;
    const embeddingDimension = arch.embeddingLength || 0;
    const vocabSize = arch.vocabSize || 0;
    const contextLength = arch.contextLength || 0;
    const ffnDimension = arch.feedForwardLength || null;

    // Advanced features
    const hasGroupedQueryAttention = attentionHeadsKV !== null && attentionHeadsKV < attentionHeads;
    const hasRoPE = arch.ropeFreqBase !== undefined;
    const ropeFreqBase = arch.ropeFreqBase || null;
    const ropeScaling = arch.ropeScalingType || null;

    // Memory calculations
    const memoryUsageGB = this.estimateMemoryUsage(model.totalParameters, quantization);
    const kvCacheGB = this.estimateKVCache(
      layers,
      attentionHeadsKV || attentionHeads,
      embeddingDimension / attentionHeads,
      contextLength
    );

    // Tensor analysis
    const tensorsByLayer = this.analyzeTensorsByLayer(tensors);
    const tensorsByType = this.analyzeTensorsByType(tensors);

    // Special tokens
    const specialTokens = {
      bos: arch.bosTokenId,
      eos: arch.eosTokenId,
      pad: arch.padTokenId,
      unk: arch.unkTokenId,
    };

    return {
      modelName,
      architecture,
      totalParameters,
      quantization,
      fileSizeGB,
      layers,
      attentionHeads,
      attentionHeadsKV,
      embeddingDimension,
      vocabSize,
      contextLength,
      ffnDimension,
      hasGroupedQueryAttention,
      hasRoPE,
      ropeFreqBase,
      ropeScaling,
      memoryUsageGB,
      kvCacheGB,
      tensorsByLayer,
      tensorsByType,
      specialTokens,
    };
  }

  /**
   * Format parameter count (e.g., 8030000000 -> "8.03B")
   */
  private formatParameters(params: bigint): string {
    const num = Number(params);
    if (num >= 1e9) {
      return `${(num / 1e9).toFixed(2)}B`;
    } else if (num >= 1e6) {
      return `${(num / 1e6).toFixed(2)}M`;
    } else if (num >= 1e3) {
      return `${(num / 1e3).toFixed(2)}K`;
    }
    return num.toString();
  }

  /**
   * Estimate memory usage based on parameters and quantization
   */
  private estimateMemoryUsage(params: bigint, quantization: string): number {
    const num = Number(params);

    // Estimate bits per parameter based on quantization
    let bitsPerParam = 16; // Default F16

    if (quantization.includes('Q4')) {
      bitsPerParam = 4.5;
    } else if (quantization.includes('Q5')) {
      bitsPerParam = 5.5;
    } else if (quantization.includes('Q8')) {
      bitsPerParam = 8.5;
    } else if (quantization.includes('Q2')) {
      bitsPerParam = 2.5;
    } else if (quantization.includes('Q3')) {
      bitsPerParam = 3.5;
    } else if (quantization.includes('Q6')) {
      bitsPerParam = 6.5;
    } else if (quantization.includes('F32')) {
      bitsPerParam = 32;
    }

    const bytes = (num * bitsPerParam) / 8;
    return bytes / (1024 ** 3); // Convert to GB
  }

  /**
   * Estimate KV cache size at maximum context length
   */
  private estimateKVCache(
    layers: number,
    kvHeads: number,
    headDim: number,
    contextLength: number
  ): number {
    // KV cache formula: 2 * layers * kvHeads * headDim * contextLength * sizeof(float16)
    const bytes = 2 * layers * kvHeads * headDim * contextLength * 2; // 2 bytes for FP16
    return bytes / (1024 ** 3); // Convert to GB
  }

  /**
   * Analyze tensors grouped by layer
   */
  private analyzeTensorsByLayer(tensors: TensorInfo[]): TransformerAnalysis['tensorsByLayer'] {
    const layerGroups = groupTensorsByLayer(tensors);
    const result: TransformerAnalysis['tensorsByLayer'] = [];

    for (const [layer, layerTensors] of Array.from(layerGroups.entries()).sort((a, b) => a[0] - b[0])) {
      const totalParams = layerTensors.reduce((sum, t) => {
        const count = t.dimensions.reduce((prod, dim) => prod * BigInt(dim), BigInt(1));
        return sum + count;
      }, BigInt(0));

      result.push({
        layer,
        tensors: layerTensors.map(t => t.name),
        totalParams: this.formatParameters(totalParams),
      });
    }

    return result;
  }

  /**
   * Analyze tensors grouped by type (role in architecture)
   */
  private analyzeTensorsByType(tensors: TensorInfo[]): TransformerAnalysis['tensorsByType'] {
    const typeGroups = new Map<string, TensorInfo[]>();

    for (const tensor of tensors) {
      const shape = new TensorShape(
        tensor.name,
        tensor.dimensions,
        tensor.type,
        tensor.dimensions.reduce((prod, dim) => prod * BigInt(dim), BigInt(1))
      );
      const role = shape.identifyRole();

      if (!typeGroups.has(role)) {
        typeGroups.set(role, []);
      }
      typeGroups.get(role)!.push(tensor);
    }

    const result: TransformerAnalysis['tensorsByType'] = [];

    for (const [type, typeTensors] of Array.from(typeGroups.entries()).sort()) {
      const totalParams = typeTensors.reduce((sum, t) => {
        const count = t.dimensions.reduce((prod, dim) => prod * BigInt(dim), BigInt(1));
        return sum + count;
      }, BigInt(0));

      result.push({
        type,
        count: typeTensors.length,
        totalParams: this.formatParameters(totalParams),
      });
    }

    return result;
  }
}
