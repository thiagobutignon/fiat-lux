/**
 * Weight Extractor
 *
 * Extracts and analyzes weights from GGUF model files
 */

import { GGUFParser } from '../../../gguf-parser/domain/use-cases/gguf-parser';
import { GGUFModel, TensorInfo, GGMLType } from '../../../gguf-parser/domain/entities/gguf-metadata';
import { IFileReader } from '../../../gguf-parser/data/protocols/file-reader';
import {
  WeightStatistics,
  LayerWeightAnalysis,
  ModelWeightProfile,
  WeightExtractionConfig,
  AttentionHeadAnalysis,
  FFNGateAnalysis,
} from '../domain/weight-statistics';

export class WeightExtractor {
  private parser: GGUFParser;
  private modelBuffer?: Buffer;
  private modelInfo?: GGUFModel;

  constructor(private fileReader: IFileReader) {
    this.parser = new GGUFParser(fileReader);
  }

  /**
   * Load model and prepare for weight extraction
   */
  async loadModel(filePath: string): Promise<void> {
    // Parse GGUF file
    this.modelInfo = await this.parser.parse(filePath);

    // Keep buffer for weight extraction
    this.modelBuffer = await this.fileReader.readFile(filePath);

    console.log(`✅ Loaded model: ${this.modelInfo.architecture.name}`);
    console.log(`   Total parameters: ${this.modelInfo.totalParameters.toLocaleString()}`);
    console.log(`   Quantization: ${this.modelInfo.quantizationType}`);
    console.log(`   Layers: ${this.modelInfo.architecture.blockCount}`);
  }

  /**
   * Extract complete weight profile
   */
  async extractWeightProfile(config: WeightExtractionConfig): Promise<ModelWeightProfile> {
    if (!this.modelInfo || !this.modelBuffer) {
      throw new Error('Model not loaded. Call loadModel() first.');
    }

    console.log('\n📊 Extracting weight profile...');

    const layers: LayerWeightAnalysis[] = [];
    const attentionHeads: AttentionHeadAnalysis[] = [];
    const ffnGates: FFNGateAnalysis[] = [];

    const blockCount = this.modelInfo.architecture.blockCount || 32;
    const headCount = this.modelInfo.architecture.headCount || 32;

    // Analyze each layer
    for (let layerIdx = 0; layerIdx < blockCount; layerIdx++) {
      console.log(`\n  Layer ${layerIdx}/${blockCount - 1}...`);

      // Analyze attention
      if (config.analyzeAttention) {
        const attnAnalysis = await this.analyzeLayerAttention(layerIdx);
        if (attnAnalysis) {
          layers.push({
            layerIndex: layerIdx,
            layerType: 'attention',
            attention: attnAnalysis,
            totalParameters: this.countLayerParameters(layerIdx, 'attention'),
            memoryFootprint: this.calculateLayerMemory(layerIdx, 'attention'),
            relativeMagnitude: 0, // Computed later
            layerStability: 0, // Computed later
          });

          // Extract individual head statistics
          if (config.computeHeadSimilarity) {
            const heads = await this.analyzeAttentionHeads(layerIdx, headCount);
            attentionHeads.push(...heads);
          }
        }
      }

      // Analyze FFN
      if (config.analyzeFFN) {
        const ffnAnalysis = await this.analyzeLayerFFN(layerIdx);
        if (ffnAnalysis) {
          const gateAnalysis: FFNGateAnalysis = {
            layerIndex: layerIdx,
            gateWeights: ffnAnalysis.gate,
            upWeights: ffnAnalysis.up,
            downWeights: ffnAnalysis.down,
            gateSpecialization: 0, // Computed from activations
            crossLayerSimilarity: 0, // Computed later
          };
          ffnGates.push(gateAnalysis);

          layers.push({
            layerIndex: layerIdx,
            layerType: 'ffn',
            ffn: ffnAnalysis,
            totalParameters: this.countLayerParameters(layerIdx, 'ffn'),
            memoryFootprint: this.calculateLayerMemory(layerIdx, 'ffn'),
            relativeMagnitude: 0,
            layerStability: 0,
          });
        }
      }

      // Analyze norms
      if (config.analyzeNorms) {
        const normAnalysis = await this.analyzeLayerNorms(layerIdx);
        if (normAnalysis) {
          layers.push({
            layerIndex: layerIdx,
            layerType: 'norm',
            norm: normAnalysis,
            totalParameters: this.countLayerParameters(layerIdx, 'norm'),
            memoryFootprint: this.calculateLayerMemory(layerIdx, 'norm'),
            relativeMagnitude: 0,
            layerStability: this.computeNormStability(normAnalysis),
          });
        }
      }
    }

    // Compute global statistics
    const globalStats = this.computeGlobalStatistics(layers);

    // Compute comparative statistics
    const earlyLayers = this.computeRangeStatistics(layers, 0, 10);
    const middleLayers = this.computeRangeStatistics(layers, 11, 21);
    const lateLayers = this.computeRangeStatistics(layers, 22, 31);

    return {
      modelName: this.modelInfo.architecture.name || 'unknown',
      quantizationType: this.modelInfo.quantizationType,
      totalParameters: Number(this.modelInfo.totalParameters),
      layers,
      attentionHeads,
      ffnGates,
      globalStatistics: globalStats,
      earlyLayers,
      middleLayers,
      lateLayers,
    };
  }

  /**
   * Analyze attention weights for a layer
   */
  private async analyzeLayerAttention(layerIdx: number): Promise<{
    query: WeightStatistics;
    key: WeightStatistics;
    value: WeightStatistics;
    output: WeightStatistics;
  } | null> {
    const tensors = this.findLayerTensors(layerIdx, ['attn_q', 'attn_k', 'attn_v', 'attn_output']);

    if (!tensors.query || !tensors.key || !tensors.value || !tensors.output) {
      return null;
    }

    return {
      query: await this.computeTensorStatistics(tensors.query),
      key: await this.computeTensorStatistics(tensors.key),
      value: await this.computeTensorStatistics(tensors.value),
      output: await this.computeTensorStatistics(tensors.output),
    };
  }

  /**
   * Analyze FFN weights for a layer
   */
  private async analyzeLayerFFN(layerIdx: number): Promise<{
    gate: WeightStatistics;
    up: WeightStatistics;
    down: WeightStatistics;
  } | null> {
    const tensors = this.findLayerTensors(layerIdx, ['ffn_gate', 'ffn_up', 'ffn_down']);

    if (!tensors.gate || !tensors.up || !tensors.down) {
      return null;
    }

    return {
      gate: await this.computeTensorStatistics(tensors.gate),
      up: await this.computeTensorStatistics(tensors.up),
      down: await this.computeTensorStatistics(tensors.down),
    };
  }

  /**
   * Analyze normalization weights for a layer
   */
  private async analyzeLayerNorms(layerIdx: number): Promise<{
    attnNorm: WeightStatistics;
    ffnNorm: WeightStatistics;
  } | null> {
    const tensors = this.findLayerTensors(layerIdx, ['attn_norm', 'ffn_norm']);

    if (!tensors.attnNorm || !tensors.ffnNorm) {
      return null;
    }

    return {
      attnNorm: await this.computeTensorStatistics(tensors.attnNorm),
      ffnNorm: await this.computeTensorStatistics(tensors.ffnNorm),
    };
  }

  /**
   * Analyze individual attention heads
   */
  private async analyzeAttentionHeads(
    layerIdx: number,
    headCount: number
  ): Promise<AttentionHeadAnalysis[]> {
    // This requires slicing Q/K/V tensors by head
    // For now, return empty array - will implement after basic extraction works
    return [];
  }

  /**
   * Find tensors for a specific layer
   */
  private findLayerTensors(layerIdx: number, types: string[]): Record<string, TensorInfo> {
    if (!this.modelInfo) {
      return {};
    }

    const result: Record<string, TensorInfo> = {};
    const layerPrefix = `blk.${layerIdx}.`;

    for (const tensor of this.modelInfo.tensors) {
      if (tensor.name.startsWith(layerPrefix)) {
        for (const type of types) {
          if (tensor.name.includes(type)) {
            const key = type.replace('attn_', '').replace('ffn_', '');
            result[key] = tensor;
          }
        }
      }
    }

    return result;
  }

  /**
   * Compute statistics for a tensor
   */
  private async computeTensorStatistics(tensor: TensorInfo): Promise<WeightStatistics> {
    // Extract raw weights
    const weights = await this.extractTensorWeights(tensor);

    // Compute statistics
    const n = weights.length;
    const mean = weights.reduce((sum, w) => sum + w, 0) / n;
    const variance = weights.reduce((sum, w) => sum + Math.pow(w - mean, 2), 0) / n;
    const std = Math.sqrt(variance);

    const sorted = [...weights].sort((a, b) => a - b);
    const min = sorted[0];
    const max = sorted[n - 1];
    const median = n % 2 === 0 ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2 : sorted[Math.floor(n / 2)];

    // Sparsity (percentage near zero)
    const sparsity = weights.filter(w => Math.abs(w) < 1e-6).length / n;

    // Norms
    const l1Norm = weights.reduce((sum, w) => sum + Math.abs(w), 0);
    const l2Norm = Math.sqrt(weights.reduce((sum, w) => sum + w * w, 0));
    const lInfNorm = Math.max(...weights.map(Math.abs));
    const frobeniusNorm = l2Norm; // Same for vectors

    // Skewness and kurtosis
    const m3 = weights.reduce((sum, w) => sum + Math.pow(w - mean, 3), 0) / n;
    const m4 = weights.reduce((sum, w) => sum + Math.pow(w - mean, 4), 0) / n;
    const skewness = m3 / Math.pow(std, 3);
    const kurtosis = m4 / Math.pow(std, 4) - 3; // Excess kurtosis

    // Effective rank (approximate using spectral decay)
    const effectiveRank = this.estimateEffectiveRank(weights);

    // Bits per element from tensor type
    const bitsPerElement = this.getBitsPerElement(tensor.type);

    return {
      mean,
      std,
      min,
      max,
      median,
      skewness,
      kurtosis,
      sparsity,
      effectiveRank,
      l1Norm,
      l2Norm,
      lInfNorm,
      frobeniusNorm,
      bitsPerElement,
    };
  }

  /**
   * Extract raw weights from tensor (dequantize if needed)
   */
  private async extractTensorWeights(tensor: TensorInfo): Promise<number[]> {
    if (!this.modelBuffer || !this.modelInfo) {
      throw new Error('Model not loaded');
    }

    const tensorDataOffset = Number(this.modelInfo.tensorDataOffset);
    const tensorOffset = tensorDataOffset + Number(tensor.offset);

    // For now, only handle F32 tensors
    // TODO: Implement dequantization for Q4_K, Q6_K, etc.
    if (tensor.type === GGMLType.F32) {
      const elementCount = tensor.dimensions.reduce((prod, dim) => prod * dim, 1);
      const weights: number[] = [];

      for (let i = 0; i < elementCount; i++) {
        const offset = tensorOffset + i * 4;
        weights.push(this.modelBuffer.readFloatLE(offset));
      }

      return weights;
    } else if (tensor.type === GGMLType.F16) {
      // Implement FP16 reading
      const elementCount = tensor.dimensions.reduce((prod, dim) => prod * dim, 1);
      const weights: number[] = [];

      for (let i = 0; i < elementCount; i++) {
        const offset = tensorOffset + i * 2;
        const fp16Value = this.modelBuffer.readUInt16LE(offset);
        weights.push(this.fp16ToFloat(fp16Value));
      }

      return weights;
    } else {
      // For quantized types, return empty array for now
      // TODO: Implement full dequantization
      console.warn(`  ⚠️  Skipping quantized tensor: ${tensor.name} (${GGMLType[tensor.type]})`);
      return [];
    }
  }

  /**
   * Convert FP16 to FP32
   */
  private fp16ToFloat(fp16: number): number {
    const sign = (fp16 >> 15) & 0x1;
    const exponent = (fp16 >> 10) & 0x1f;
    const fraction = fp16 & 0x3ff;

    if (exponent === 0) {
      // Subnormal or zero
      return (sign ? -1 : 1) * Math.pow(2, -14) * (fraction / 1024);
    } else if (exponent === 0x1f) {
      // Infinity or NaN
      return fraction === 0 ? (sign ? -Infinity : Infinity) : NaN;
    } else {
      // Normalized
      return (sign ? -1 : 1) * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
    }
  }

  /**
   * Estimate effective rank (simplified version)
   */
  private estimateEffectiveRank(weights: number[]): number {
    // Very rough estimate based on weight distribution
    // Proper implementation would require SVD
    const nonZero = weights.filter(w => Math.abs(w) > 1e-6).length;
    return nonZero / weights.length;
  }

  /**
   * Get bits per element for a tensor type
   */
  private getBitsPerElement(type: GGMLType): number {
    switch (type) {
      case GGMLType.F32:
        return 32;
      case GGMLType.F16:
        return 16;
      case GGMLType.Q4_0:
      case GGMLType.Q4_1:
      case GGMLType.Q4_K:
        return 4.5;
      case GGMLType.Q5_0:
      case GGMLType.Q5_1:
      case GGMLType.Q5_K:
        return 5.5;
      case GGMLType.Q6_K:
        return 6.5;
      case GGMLType.Q8_0:
      case GGMLType.Q8_1:
      case GGMLType.Q8_K:
        return 8.5;
      case GGMLType.Q2_K:
        return 2.5;
      case GGMLType.Q3_K:
        return 3.5;
      default:
        return 8;
    }
  }

  /**
   * Helper methods for layer analysis
   */

  private countLayerParameters(layerIdx: number, layerType: string): number {
    // Simplified - would need actual tensor dimensions
    return 0;
  }

  private calculateLayerMemory(layerIdx: number, layerType: string): number {
    // Simplified
    return 0;
  }

  private computeNormStability(norm: any): number {
    // Ratio of attn_norm to ffn_norm
    return norm.attnNorm.l2Norm / norm.ffnNorm.l2Norm;
  }

  private computeGlobalStatistics(layers: LayerWeightAnalysis[]): any {
    // Aggregate statistics across all layers
    return {
      meanWeight: 0,
      stdWeight: 0,
      overallSparsity: 0,
      totalL1Norm: 0,
      totalL2Norm: 0,
    };
  }

  private computeRangeStatistics(
    layers: LayerWeightAnalysis[],
    start: number,
    end: number
  ): WeightStatistics {
    // Aggregate statistics for a range of layers
    return {
      mean: 0,
      std: 0,
      min: 0,
      max: 0,
      median: 0,
      skewness: 0,
      kurtosis: 0,
      sparsity: 0,
      effectiveRank: 0,
      l1Norm: 0,
      l2Norm: 0,
      lInfNorm: 0,
      frobeniusNorm: 0,
      bitsPerElement: 0,
    };
  }
}
