/**
 * GGUF Tensor Reader
 * Reads and dequantizes tensor data from GGUF files
 */

import { open, FileHandle } from 'fs/promises';
import { TensorInfo, GGMLType } from '../entities/gguf-metadata';
import { TensorData, LayerWeights } from '../entities/tensor-data';
import { dequantize } from './dequantize';

export class GGUFTensorReader {
  private fileHandle?: FileHandle;
  private alignment: number;
  private tensorDataOffset: bigint;

  constructor(
    private filePath: string,
    alignment: number = 32,
    tensorDataOffset: bigint = BigInt(0)
  ) {
    this.alignment = alignment;
    this.tensorDataOffset = tensorDataOffset;
  }

  /**
   * Initialize file handle
   */
  private async init(): Promise<void> {
    if (!this.fileHandle) {
      this.fileHandle = await open(this.filePath, 'r');
    }
  }

  /**
   * Close file handle
   */
  async close(): Promise<void> {
    if (this.fileHandle) {
      await this.fileHandle.close();
      this.fileHandle = undefined;
    }
  }

  /**
   * Calculate bytes needed for tensor based on quantization type
   */
  private calculateTensorBytes(tensor: TensorInfo): number {
    const elementCount = tensor.dimensions.reduce(
      (prod, dim) => prod * BigInt(dim),
      BigInt(1)
    );

    // Get bytes per element for quantization type
    const bytesPerElement = this.getBytesPerElement(tensor.type);

    return Math.ceil(Number(elementCount) * bytesPerElement);
  }

  /**
   * Get bytes per element for quantization type
   */
  private getBytesPerElement(type: GGMLType): number {
    switch (type) {
      case GGMLType.F32:
        return 4;
      case GGMLType.F16:
        return 2;
      case GGMLType.Q4_0:
        return 18 / 32; // 18 bytes per 32 elements
      case GGMLType.Q4_1:
        return 20 / 32; // 20 bytes per 32 elements
      case GGMLType.Q4_K:
        return 144 / 256; // ~144 bytes per 256 elements
      case GGMLType.Q5_0:
        return 22 / 32;
      case GGMLType.Q5_1:
        return 24 / 32;
      case GGMLType.Q6_K:
        return 210 / 256;
      case GGMLType.Q8_0:
        return 34 / 32; // 2 bytes scale + 32 bytes data
      case GGMLType.Q8_1:
        return 36 / 32;
      case GGMLType.I8:
        return 1;
      case GGMLType.I16:
        return 2;
      case GGMLType.I32:
        return 4;
      default:
        return 4; // Default to F32
    }
  }

  /**
   * Read single tensor from file
   */
  async readTensor(tensor: TensorInfo): Promise<TensorData> {
    await this.init();

    if (!this.fileHandle) {
      throw new Error('File handle not initialized');
    }

    // Calculate tensor size in bytes
    const tensorBytes = this.calculateTensorBytes(tensor);

    // Calculate actual file offset (considering alignment)
    const fileOffset = Number(this.tensorDataOffset + tensor.offset);

    // Read raw bytes
    const buffer = Buffer.allocUnsafe(tensorBytes);
    await this.fileHandle.read(buffer, 0, tensorBytes, fileOffset);

    // Calculate element count
    const elementCount = tensor.dimensions.reduce(
      (prod, dim) => prod * dim,
      1
    );

    // Dequantize
    const data = dequantize(buffer, tensor.type, elementCount);

    return {
      name: tensor.name,
      shape: tensor.dimensions,
      type: tensor.type,
      data,
      originalType: tensor.type,
    };
  }

  /**
   * Read multiple tensors
   */
  async readTensors(tensors: TensorInfo[]): Promise<TensorData[]> {
    const results: TensorData[] = [];

    for (const tensor of tensors) {
      results.push(await this.readTensor(tensor));
    }

    return results;
  }

  /**
   * Read specific layer weights
   */
  async readLayer(
    allTensors: TensorInfo[],
    layerIndex: number
  ): Promise<LayerWeights> {
    const pattern = `blk.${layerIndex}.`;
    const layerTensors = allTensors.filter((t) => t.name.startsWith(pattern));

    const weights: LayerWeights = {
      layerIndex,
    };

    // Read each tensor type for this layer
    for (const tensor of layerTensors) {
      const data = await this.readTensor(tensor);

      // Classify by name suffix
      if (tensor.name.endsWith('attn_norm.weight')) {
        weights.attentionNorm = data;
      } else if (tensor.name.endsWith('attn_q.weight')) {
        weights.attentionQ = data;
      } else if (tensor.name.endsWith('attn_k.weight')) {
        weights.attentionK = data;
      } else if (tensor.name.endsWith('attn_v.weight')) {
        weights.attentionV = data;
      } else if (tensor.name.endsWith('attn_output.weight')) {
        weights.attentionOutput = data;
      } else if (tensor.name.endsWith('ffn_norm.weight')) {
        weights.ffnNorm = data;
      } else if (tensor.name.endsWith('ffn_gate.weight')) {
        weights.ffnGate = data;
      } else if (tensor.name.endsWith('ffn_up.weight')) {
        weights.ffnUp = data;
      } else if (tensor.name.endsWith('ffn_down.weight')) {
        weights.ffnDown = data;
      }
    }

    return weights;
  }

  /**
   * Read embedding table
   */
  async readEmbeddings(allTensors: TensorInfo[]): Promise<TensorData | null> {
    const embedTensor = allTensors.find(
      (t) =>
        t.name.includes('token_embd') ||
        t.name.includes('tok_embeddings') ||
        t.name === 'token_embd.weight'
    );

    if (!embedTensor) {
      return null;
    }

    return await this.readTensor(embedTensor);
  }

  /**
   * Read output layer
   */
  async readOutputLayer(allTensors: TensorInfo[]): Promise<TensorData | null> {
    const outputTensor = allTensors.find(
      (t) => t.name === 'output.weight' || t.name.includes('output_norm')
    );

    if (!outputTensor) {
      return null;
    }

    return await this.readTensor(outputTensor);
  }
}
