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
import { Dequantizer } from './dequantizer';

export class WeightExtractor {
  private parser: GGUFParser;
  private modelBuffer?: Buffer;
  private modelInfo?: GGUFModel;
  private dequantizer: Dequantizer;

  constructor(private fileReader: IFileReader) {
    this.parser = new GGUFParser(fileReader);
    this.dequantizer = new Dequantizer();
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

    // findLayerTensors removes 'attn_' prefix, so keys are 'q', 'k', 'v', 'output'
    if (!tensors.q || !tensors.k || !tensors.v || !tensors.output) {
      return null;
    }

    return {
      query: await this.computeTensorStatistics(tensors.q),
      key: await this.computeTensorStatistics(tensors.k),
      value: await this.computeTensorStatistics(tensors.v),
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

    if (!tensors.attn_norm || !tensors.ffn_norm) {
      return null;
    }

    return {
      attnNorm: await this.computeTensorStatistics(tensors.attn_norm),
      ffnNorm: await this.computeTensorStatistics(tensors.ffn_norm),
    };
  }

  /**
   * Analyze individual attention heads
   *
   * Llama 3.1 8B architecture:
   * - 32 Q heads, 8 KV heads (Grouped Query Attention)
   * - embedding_dim = 4096, head_dim = 128
   * - Q: [4096, 4096] → 32 heads of [128, 4096]
   * - K/V: [1024, 4096] → 8 heads of [128, 4096]
   */
  private async analyzeAttentionHeads(
    layerIdx: number,
    headCount: number
  ): Promise<AttentionHeadAnalysis[]> {
    const tensors = this.findLayerTensors(layerIdx, ['attn_q', 'attn_k', 'attn_v']);

    if (!tensors.q || !tensors.k || !tensors.v) {
      return [];
    }

    const heads: AttentionHeadAnalysis[] = [];

    // Extract embedding dimensions from model metadata
    const embeddingDim = this.modelInfo?.architecture.embeddingLength || 4096;
    const headDim = Math.floor(embeddingDim / headCount);
    const kvHeadCount = this.modelInfo?.architecture.headCountKV || 8;

    // Analyze each Q head
    for (let headIdx = 0; headIdx < headCount; headIdx++) {
      try {
        // For Q: slice [headIdx * headDim : (headIdx + 1) * headDim, :] rows
        const qWeights = await this.extractHeadWeights(tensors.q, headIdx, headDim, 'row');

        // For K/V: use grouped approach (each KV head serves multiple Q heads)
        const kvHeadIdx = Math.floor(headIdx / (headCount / kvHeadCount));
        const kWeights = await this.extractHeadWeights(tensors.k, kvHeadIdx, headDim, 'row');
        const vWeights = await this.extractHeadWeights(tensors.v, kvHeadIdx, headDim, 'row');

        // Compute statistics for this head
        const qStats = await this.computeWeightStatistics(qWeights, tensors.q.type);
        const kStats = await this.computeWeightStatistics(kWeights, tensors.k.type);
        const vStats = await this.computeWeightStatistics(vWeights, tensors.v.type);

        heads.push({
          layerIndex: layerIdx,
          headIndex: headIdx,
          queryWeights: qStats,
          keyWeights: kStats,
          valueWeights: vStats,
          headEntropy: 0, // TODO: compute from activations
          headSpecialization: 0, // TODO: compute similarity to other heads
          mostSimilarHead: { layer: 0, head: 0, similarity: 0 },
        });
      } catch (error) {
        console.warn(`  ⚠️  Skipping head ${headIdx} in layer ${layerIdx}: ${error}`);
      }
    }

    return heads;
  }

  /**
   * Extract weights for a specific attention head
   */
  private async extractHeadWeights(
    tensor: TensorInfo,
    headIdx: number,
    headDim: number,
    sliceMode: 'row' | 'col'
  ): Promise<number[]> {
    // For now, return empty array and log warning
    // Full implementation requires tensor slicing which is complex
    // Will implement if needed for deeper analysis
    return [];
  }

  /**
   * Compute statistics for a weight array
   */
  private async computeWeightStatistics(weights: number[], type: GGMLType): Promise<WeightStatistics> {
    if (weights.length === 0) {
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
        bitsPerElement: this.getBitsPerElement(type),
      };
    }

    const n = weights.length;
    let sum = 0, sumSq = 0, sumAbs = 0;
    let min = weights[0], max = weights[0], maxAbs = Math.abs(weights[0]);
    let sparsityCount = 0;

    for (let i = 0; i < n; i++) {
      const w = weights[i];
      const absW = Math.abs(w);
      sum += w;
      sumSq += w * w;
      sumAbs += absW;
      if (w < min) min = w;
      if (w > max) max = w;
      if (absW > maxAbs) maxAbs = absW;
      if (absW < 1e-6) sparsityCount++;
    }

    const mean = sum / n;
    const l1Norm = sumAbs;
    const l2Norm = Math.sqrt(sumSq);
    const lInfNorm = maxAbs;
    const sparsity = sparsityCount / n;

    let variance = 0, m3 = 0, m4 = 0;
    for (let i = 0; i < n; i++) {
      const diff = weights[i] - mean;
      const diff2 = diff * diff;
      variance += diff2;
      m3 += diff2 * diff;
      m4 += diff2 * diff2;
    }

    variance /= n;
    m3 /= n;
    m4 /= n;

    const std = Math.sqrt(variance);
    const skewness = std > 0 ? m3 / Math.pow(std, 3) : 0;
    const kurtosis = std > 0 ? m4 / Math.pow(std, 4) - 3 : 0;
    const median = this.computeMedian(weights);
    const effectiveRank = sparsity < 1.0 ? 1.0 - sparsity : 0.0;

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
      frobeniusNorm: l2Norm,
      bitsPerElement: this.getBitsPerElement(type),
    };
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
            // For norms, keep full name to avoid conflicts
            const key = type.includes('_norm')
              ? type
              : type.replace('attn_', '').replace('ffn_', '');
            result[key] = tensor;
          }
        }
      }
    }

    return result;
  }

  /**
   * Compute statistics for a tensor (optimized for large arrays)
   */
  private async computeTensorStatistics(tensor: TensorInfo): Promise<WeightStatistics> {
    // Extract raw weights
    const weights = await this.extractTensorWeights(tensor);

    if (weights.length === 0) {
      // Return zero stats for empty arrays
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
        bitsPerElement: this.getBitsPerElement(tensor.type),
      };
    }

    const n = weights.length;

    // Single pass for mean, min, max, l1/l2 norms, sparsity
    let sum = 0;
    let sumSq = 0;
    let sumAbs = 0;
    let min = weights[0];
    let max = weights[0];
    let maxAbs = Math.abs(weights[0]);
    let sparsityCount = 0;

    for (let i = 0; i < n; i++) {
      const w = weights[i];
      const absW = Math.abs(w);

      sum += w;
      sumSq += w * w;
      sumAbs += absW;

      if (w < min) min = w;
      if (w > max) max = w;
      if (absW > maxAbs) maxAbs = absW;
      if (absW < 1e-6) sparsityCount++;
    }

    const mean = sum / n;
    const l1Norm = sumAbs;
    const l2Norm = Math.sqrt(sumSq);
    const lInfNorm = maxAbs;
    const frobeniusNorm = l2Norm;
    const sparsity = sparsityCount / n;

    // Second pass for variance, skewness, kurtosis
    let variance = 0;
    let m3 = 0;
    let m4 = 0;

    for (let i = 0; i < n; i++) {
      const diff = weights[i] - mean;
      const diff2 = diff * diff;
      variance += diff2;
      m3 += diff2 * diff;
      m4 += diff2 * diff2;
    }

    variance /= n;
    m3 /= n;
    m4 /= n;

    const std = Math.sqrt(variance);
    const skewness = std > 0 ? m3 / Math.pow(std, 3) : 0;
    const kurtosis = std > 0 ? m4 / Math.pow(std, 4) - 3 : 0;

    // Median requires sorting - use quickselect for large arrays
    const median = this.computeMedian(weights);

    // Effective rank (approximate)
    const effectiveRank = sparsity < 1.0 ? 1.0 - sparsity : 0.0;

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
   * Compute median efficiently (without full sort for large arrays)
   */
  private computeMedian(weights: number[]): number {
    const n = weights.length;
    if (n === 0) return 0;

    // For small arrays, just sort
    if (n < 1000) {
      const sorted = [...weights].sort((a, b) => a - b);
      return n % 2 === 0 ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2 : sorted[Math.floor(n / 2)];
    }

    // For large arrays, use sampling approximation
    const sampleSize = 1000;
    const step = Math.floor(n / sampleSize);
    const sample: number[] = [];

    for (let i = 0; i < n; i += step) {
      sample.push(weights[i]);
    }

    sample.sort((a, b) => a - b);
    const sampleN = sample.length;
    return sampleN % 2 === 0
      ? (sample[sampleN / 2 - 1] + sample[sampleN / 2]) / 2
      : sample[Math.floor(sampleN / 2)];
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
    const elementCount = tensor.dimensions.reduce((prod, dim) => prod * dim, 1);

    try {
      // Use dequantizer for all supported types
      const weights = this.dequantizer.dequantize(
        this.modelBuffer,
        tensorOffset,
        elementCount,
        tensor.type
      );

      // Debug: log first few values for Q4_K/Q6_K tensors (for debugging)
      // if ((tensor.type === GGMLType.Q4_K || tensor.type === GGMLType.Q6_K) && weights.length > 0) {
      //   const sample = weights.slice(0, Math.min(10, weights.length));
      //   console.log(`  📊 ${tensor.name} (${GGMLType[tensor.type]}): first 10 values =`, sample);
      // }

      return weights;
    } catch (error) {
      // If dequantization fails (unsupported type), warn and return empty array
      console.warn(`  ⚠️  Skipping tensor: ${tensor.name} (${GGMLType[tensor.type]}) - ${(error as Error).message}`);
      return [];
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
