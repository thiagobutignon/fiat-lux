/**
 * Tensor Shape and Parameter Calculations
 */

import { GGMLType, TensorInfo } from './gguf-metadata';

export class TensorShape {
  constructor(
    public readonly name: string,
    public readonly dimensions: number[],
    public readonly type: GGMLType,
    public readonly elementCount: bigint
  ) {}

  /**
   * Get human-readable shape string
   */
  getShapeString(): string {
    return `[${this.dimensions.join(' × ')}]`;
  }

  /**
   * Calculate number of parameters in this tensor
   */
  getParameterCount(): bigint {
    return this.elementCount;
  }

  /**
   * Get the quantization type name
   */
  getQuantizationType(): string {
    return GGMLType[this.type] || 'UNKNOWN';
  }

  /**
   * Estimate memory size based on quantization type
   * Returns size in bytes
   */
  estimateMemorySize(): bigint {
    const bitsPerElement = this.getBitsPerElement();
    return (this.elementCount * BigInt(bitsPerElement)) / BigInt(8);
  }

  /**
   * Get bits per element for different quantization types
   */
  private getBitsPerElement(): number {
    switch (this.type) {
      case GGMLType.F32:
        return 32;
      case GGMLType.F16:
        return 16;
      case GGMLType.Q4_0:
      case GGMLType.Q4_1:
      case GGMLType.Q4_K:
        return 4.5; // Approximate
      case GGMLType.Q5_0:
      case GGMLType.Q5_1:
      case GGMLType.Q5_K:
        return 5.5; // Approximate
      case GGMLType.Q8_0:
      case GGMLType.Q8_1:
      case GGMLType.Q8_K:
        return 8.5; // Approximate
      case GGMLType.Q2_K:
        return 2.5; // Approximate
      case GGMLType.Q3_K:
        return 3.5; // Approximate
      case GGMLType.Q6_K:
        return 6.5; // Approximate
      default:
        return 8; // Default fallback
    }
  }

  /**
   * Identify tensor role in transformer architecture
   */
  identifyRole(): string {
    const name = this.name.toLowerCase();

    if (name.includes('embed') || name.includes('tok_embed')) {
      return 'Token Embedding';
    }
    if (name.includes('output_norm') || name.includes('norm')) {
      return 'Layer Normalization';
    }
    if (name.includes('attn_q') || name.includes('wq')) {
      return 'Attention Query';
    }
    if (name.includes('attn_k') || name.includes('wk')) {
      return 'Attention Key';
    }
    if (name.includes('attn_v') || name.includes('wv')) {
      return 'Attention Value';
    }
    if (name.includes('attn_output') || name.includes('wo')) {
      return 'Attention Output';
    }
    if (name.includes('ffn_gate') || name.includes('w1')) {
      return 'FFN Gate';
    }
    if (name.includes('ffn_down') || name.includes('w2')) {
      return 'FFN Down';
    }
    if (name.includes('ffn_up') || name.includes('w3')) {
      return 'FFN Up';
    }
    if (name.includes('output') && !name.includes('attn')) {
      return 'Output Layer';
    }

    return 'Unknown';
  }

  /**
   * Determine if tensor is a weight or bias
   */
  isWeight(): boolean {
    return this.dimensions.length >= 2;
  }

  isBias(): boolean {
    return this.dimensions.length === 1;
  }

  /**
   * Get layer number from tensor name (if applicable)
   */
  getLayerNumber(): number | null {
    const match = this.name.match(/\.(\d+)\./);
    return match ? parseInt(match[1], 10) : null;
  }
}

/**
 * Calculate total parameters from tensor list
 */
export function calculateTotalParameters(tensors: TensorInfo[]): bigint {
  return tensors.reduce((sum, tensor) => {
    const elementCount = tensor.dimensions.reduce(
      (prod, dim) => prod * BigInt(dim),
      BigInt(1)
    );
    return sum + elementCount;
  }, BigInt(0));
}

/**
 * Group tensors by layer
 */
export function groupTensorsByLayer(tensors: TensorInfo[]): Map<number, TensorInfo[]> {
  const layers = new Map<number, TensorInfo[]>();

  for (const tensor of tensors) {
    const shape = new TensorShape(
      tensor.name,
      tensor.dimensions,
      tensor.type,
      tensor.dimensions.reduce((prod, dim) => prod * BigInt(dim), BigInt(1))
    );

    const layerNum = shape.getLayerNumber();
    if (layerNum !== null) {
      if (!layers.has(layerNum)) {
        layers.set(layerNum, []);
      }
      layers.get(layerNum)!.push(tensor);
    }
  }

  return layers;
}

/**
 * Analyze transformer architecture from tensors
 */
export function analyzeTransformerArchitecture(tensors: TensorInfo[]): {
  layers: number;
  attentionHeads: number | null;
  embeddingDim: number | null;
  vocabSize: number | null;
  ffnDim: number | null;
} {
  const layers = groupTensorsByLayer(tensors);
  const layerCount = layers.size;

  // Find embedding tensor to get vocab size and embedding dim
  const embedTensor = tensors.find(t =>
    t.name.toLowerCase().includes('embed') ||
    t.name.toLowerCase().includes('tok_embd')
  );

  const vocabSize = embedTensor?.dimensions[0] || null;
  const embeddingDim = embedTensor?.dimensions[1] || null;

  // Find attention query tensor to determine number of heads
  const attnQTensor = tensors.find(t =>
    t.name.toLowerCase().includes('attn_q') ||
    t.name.toLowerCase().includes('wq')
  );

  const attentionHeads = attnQTensor && embeddingDim
    ? Math.floor(attnQTensor.dimensions[0] / embeddingDim)
    : null;

  // Find FFN tensor to get FFN dimension
  const ffnTensor = tensors.find(t =>
    t.name.toLowerCase().includes('ffn') ||
    t.name.toLowerCase().includes('w1')
  );

  const ffnDim = ffnTensor?.dimensions[0] || null;

  return {
    layers: layerCount,
    attentionHeads,
    embeddingDim,
    vocabSize,
    ffnDim,
  };
}
