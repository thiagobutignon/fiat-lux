/**
 * GGUF Binary Parser
 * Parses GGUF format according to specification v3
 */

import {
  GGUFHeader,
  GGUFMetadata,
  GGUFMetadataValue,
  GGUFValueType,
  TensorInfo,
  GGMLType,
  GGUFModel,
  GGUFArchitectureInfo,
} from '../entities/gguf-metadata';
import { calculateTotalParameters, analyzeTransformerArchitecture } from '../entities/tensor-shape';
import { IFileReader } from '../../data/protocols/file-reader';

export class GGUFParser {
  private buffer!: Buffer;
  private offset: number = 0;

  constructor(private fileReader: IFileReader) {}

  /**
   * Parse GGUF file and extract all information
   */
  async parse(filePath: string): Promise<GGUFModel> {
    // Read entire file
    this.buffer = await this.fileReader.readFile(filePath);
    this.offset = 0;

    // Parse header
    const header = this.parseHeader();

    // Parse metadata
    const metadata = this.parseMetadata(header.metadataKVCount);

    // Extract architecture information
    const architecture = this.extractArchitecture(metadata);

    // Parse tensor information
    const tensors = this.parseTensorInfo(header.tensorCount);

    // Calculate total parameters
    const totalParameters = calculateTotalParameters(tensors);

    // Determine quantization type
    const quantizationType = this.determineQuantizationType(tensors);

    // Align to boundary before tensor data
    const alignment = architecture.alignment || 32;
    const currentOffset = this.offset;
    const alignedOffset = Math.ceil(currentOffset / alignment) * alignment;
    this.offset = alignedOffset;

    return {
      header,
      metadata,
      architecture,
      tensors,
      totalParameters,
      quantizationType,
      tensorDataOffset: BigInt(this.offset), // Aligned offset where tensor data begins
    };
  }

  /**
   * Parse GGUF header
   */
  private parseHeader(): GGUFHeader {
    // Magic number (4 bytes): "GGUF"
    const magic = this.buffer.toString('utf8', 0, 4);
    if (magic !== 'GGUF') {
      throw new Error(`Invalid GGUF file: magic number is "${magic}", expected "GGUF"`);
    }
    this.offset = 4;

    // Version (4 bytes, uint32)
    const version = this.readUInt32();
    if (version !== 3) {
      throw new Error(`Unsupported GGUF version: ${version}, only version 3 is supported`);
    }

    // Tensor count (8 bytes, uint64)
    const tensorCount = this.readUInt64();

    // Metadata KV count (8 bytes, uint64)
    const metadataKVCount = this.readUInt64();

    return {
      magic,
      version,
      tensorCount,
      metadataKVCount,
    };
  }

  /**
   * Parse metadata key-value pairs
   */
  private parseMetadata(count: bigint): GGUFMetadata {
    const metadata: GGUFMetadata = {};

    for (let i = 0; i < Number(count); i++) {
      // Read key (string)
      const key = this.readString();

      // Read value type (4 bytes, uint32)
      const valueType = this.readUInt32() as GGUFValueType;

      // Read value based on type
      const value = this.readValue(valueType);

      metadata[key] = {
        type: valueType,
        value,
      };
    }

    return metadata;
  }

  /**
   * Parse tensor information
   */
  private parseTensorInfo(count: bigint): TensorInfo[] {
    const tensors: TensorInfo[] = [];

    for (let i = 0; i < Number(count); i++) {
      // Read tensor name
      const name = this.readString();

      // Read number of dimensions (4 bytes, uint32)
      const nDimensions = this.readUInt32();

      // Read dimensions
      const dimensions: number[] = [];
      for (let d = 0; d < nDimensions; d++) {
        dimensions.push(Number(this.readUInt64()));
      }

      // Read tensor type (4 bytes, uint32)
      const type = this.readUInt32() as GGMLType;

      // Read offset (8 bytes, uint64)
      const offset = this.readUInt64();

      // Calculate size (bytes)
      const elementCount = dimensions.reduce((prod, dim) => prod * BigInt(dim), BigInt(1));
      const size = this.calculateTensorSize(elementCount, type);

      tensors.push({
        name,
        dimensions,
        type,
        offset,
        size,
      });
    }

    return tensors;
  }

  /**
   * Extract architecture information from metadata
   */
  private extractArchitecture(metadata: GGUFMetadata): GGUFArchitectureInfo {
    const arch: GGUFArchitectureInfo = {};

    // Helper function to get metadata value
    const getValue = (key: string): any => {
      return metadata[key]?.value;
    };

    // General information
    arch.architecture = getValue('general.architecture');
    arch.quantizationVersion = getValue('general.quantization_version');
    arch.alignment = getValue('general.alignment');
    arch.name = getValue('general.name');
    arch.author = getValue('general.author');
    arch.version = getValue('general.version');
    arch.organization = getValue('general.organization');
    arch.basename = getValue('general.basename');
    arch.finetune = getValue('general.finetune');
    arch.description = getValue('general.description');
    arch.fileType = getValue('general.file_type');

    // Architecture-specific (llama, gpt, etc.)
    const archName = arch.architecture || 'llama';
    arch.contextLength = getValue(`${archName}.context_length`);
    arch.embeddingLength = getValue(`${archName}.embedding_length`);
    arch.blockCount = getValue(`${archName}.block_count`);
    arch.feedForwardLength = getValue(`${archName}.feed_forward_length`);
    arch.headCount = getValue(`${archName}.attention.head_count`);
    arch.headCountKV = getValue(`${archName}.attention.head_count_kv`);
    arch.attentionLayerNormRMSEpsilon = getValue(`${archName}.attention.layer_norm_rms_epsilon`);
    arch.ropeFreqBase = getValue(`${archName}.rope.freq_base`);
    arch.ropeDimensionCount = getValue(`${archName}.rope.dimension_count`);
    arch.ropeScalingType = getValue(`${archName}.rope.scaling.type`);
    arch.ropeScalingFactor = getValue(`${archName}.rope.scaling.factor`);

    // Tokenizer
    arch.tokenizerModel = getValue('tokenizer.ggml.model');
    arch.tokenizerType = getValue('tokenizer.ggml.type');
    arch.tokenizerTokens = getValue('tokenizer.ggml.tokens');
    arch.tokenizerScores = getValue('tokenizer.ggml.scores');
    arch.tokenizerTokenType = getValue('tokenizer.ggml.token_type');
    arch.tokenizerMerges = getValue('tokenizer.ggml.merges');
    arch.bosTokenId = getValue('tokenizer.ggml.bos_token_id');
    arch.eosTokenId = getValue('tokenizer.ggml.eos_token_id');
    arch.padTokenId = getValue('tokenizer.ggml.pad_token_id');
    arch.unkTokenId = getValue('tokenizer.ggml.unknown_token_id');
    arch.sepTokenId = getValue('tokenizer.ggml.separator_token_id');

    // Vocab size from tokens array or direct value
    arch.vocabSize = arch.tokenizerTokens?.length || getValue(`${archName}.vocab_size`);

    return arch;
  }

  /**
   * Determine quantization type from tensors
   */
  private determineQuantizationType(tensors: TensorInfo[]): string {
    const types = new Set(tensors.map(t => GGMLType[t.type]));
    return Array.from(types).join(', ');
  }

  // ============================================================================
  // Binary Reading Utilities
  // ============================================================================

  private readUInt32(): number {
    const value = this.buffer.readUInt32LE(this.offset);
    this.offset += 4;
    return value;
  }

  private readUInt64(): bigint {
    const value = this.buffer.readBigUInt64LE(this.offset);
    this.offset += 8;
    return value;
  }

  private readInt32(): number {
    const value = this.buffer.readInt32LE(this.offset);
    this.offset += 4;
    return value;
  }

  private readInt64(): bigint {
    const value = this.buffer.readBigInt64LE(this.offset);
    this.offset += 8;
    return value;
  }

  private readFloat32(): number {
    const value = this.buffer.readFloatLE(this.offset);
    this.offset += 4;
    return value;
  }

  private readFloat64(): number {
    const value = this.buffer.readDoubleLE(this.offset);
    this.offset += 8;
    return value;
  }

  private readUInt8(): number {
    const value = this.buffer.readUInt8(this.offset);
    this.offset += 1;
    return value;
  }

  private readInt8(): number {
    const value = this.buffer.readInt8(this.offset);
    this.offset += 1;
    return value;
  }

  private readUInt16(): number {
    const value = this.buffer.readUInt16LE(this.offset);
    this.offset += 2;
    return value;
  }

  private readInt16(): number {
    const value = this.buffer.readInt16LE(this.offset);
    this.offset += 2;
    return value;
  }

  private readBool(): boolean {
    const value = this.readUInt8();
    return value !== 0;
  }

  private readString(): string {
    const length = Number(this.readUInt64());
    const value = this.buffer.toString('utf8', this.offset, this.offset + length);
    this.offset += length;
    return value;
  }

  private readValue(type: GGUFValueType): any {
    switch (type) {
      case GGUFValueType.UINT8:
        return this.readUInt8();
      case GGUFValueType.INT8:
        return this.readInt8();
      case GGUFValueType.UINT16:
        return this.readUInt16();
      case GGUFValueType.INT16:
        return this.readInt16();
      case GGUFValueType.UINT32:
        return this.readUInt32();
      case GGUFValueType.INT32:
        return this.readInt32();
      case GGUFValueType.FLOAT32:
        return this.readFloat32();
      case GGUFValueType.BOOL:
        return this.readBool();
      case GGUFValueType.STRING:
        return this.readString();
      case GGUFValueType.ARRAY:
        return this.readArray();
      case GGUFValueType.UINT64:
        return this.readUInt64();
      case GGUFValueType.INT64:
        return this.readInt64();
      case GGUFValueType.FLOAT64:
        return this.readFloat64();
      default:
        throw new Error(`Unsupported value type: ${type}`);
    }
  }

  private readArray(): any[] {
    const elementType = this.readUInt32() as GGUFValueType;
    const length = Number(this.readUInt64());
    const array: any[] = [];

    for (let i = 0; i < length; i++) {
      array.push(this.readValue(elementType));
    }

    return array;
  }

  /**
   * Calculate tensor size using exact block sizes
   */
  private calculateTensorSize(elementCount: bigint, type: GGMLType): bigint {
    // For types with exact block sizes, use precise calculation
    switch (type) {
      case GGMLType.Q4_0:
        return (elementCount / BigInt(32)) * BigInt(18);
      case GGMLType.Q4_1:
        return (elementCount / BigInt(32)) * BigInt(20);
      case GGMLType.Q5_0:
        return (elementCount / BigInt(32)) * BigInt(22);
      case GGMLType.Q5_1:
        return (elementCount / BigInt(32)) * BigInt(24);
      case GGMLType.Q8_0:
        return (elementCount / BigInt(32)) * BigInt(34);
      case GGMLType.Q8_1:
        return (elementCount / BigInt(32)) * BigInt(36);
      case GGMLType.Q4_K:
        return (elementCount / BigInt(256)) * BigInt(144);
      case GGMLType.Q5_K:
        return (elementCount / BigInt(256)) * BigInt(176);
      case GGMLType.Q6_K:
        return (elementCount / BigInt(256)) * BigInt(210);
      case GGMLType.Q2_K:
        return (elementCount / BigInt(256)) * BigInt(80);
      case GGMLType.Q3_K:
        return (elementCount / BigInt(256)) * BigInt(112);
      case GGMLType.Q8_K:
        return (elementCount / BigInt(256)) * BigInt(272);
      // For non-quantized types, use simple multiplication
      case GGMLType.F32:
        return elementCount * BigInt(4);
      case GGMLType.F16:
        return elementCount * BigInt(2);
      case GGMLType.I8:
        return elementCount;
      case GGMLType.I16:
        return elementCount * BigInt(2);
      case GGMLType.I32:
        return elementCount * BigInt(4);
      case GGMLType.I64:
        return elementCount * BigInt(8);
      case GGMLType.F64:
        return elementCount * BigInt(8);
      default:
        // Fallback to approximate calculation
        const bytesPerElement = this.getBytesPerElement(type);
        return (elementCount * BigInt(Math.round(bytesPerElement * 1000))) / BigInt(1000);
    }
  }

  /**
   * Get approximate bytes per element for quantization types
   */
  private getBytesPerElement(type: GGMLType): number {
    switch (type) {
      case GGMLType.F32:
        return 4;
      case GGMLType.F16:
        return 2;
      case GGMLType.Q4_0:
      case GGMLType.Q4_1:
        return 0.5625; // 4.5 bits
      case GGMLType.Q5_0:
      case GGMLType.Q5_1:
        return 0.6875; // 5.5 bits
      case GGMLType.Q8_0:
      case GGMLType.Q8_1:
        return 1.0625; // 8.5 bits
      case GGMLType.Q2_K:
        return 0.3125; // 2.5 bits
      case GGMLType.Q3_K:
        return 0.4375; // 3.5 bits
      case GGMLType.Q4_K:
        return 0.5625; // 4.5 bits
      case GGMLType.Q5_K:
        return 0.6875; // 5.5 bits
      case GGMLType.Q6_K:
        return 0.8125; // 6.5 bits
      case GGMLType.Q8_K:
        return 1.0625; // 8.5 bits
      case GGMLType.I8:
        return 1;
      case GGMLType.I16:
        return 2;
      case GGMLType.I32:
        return 4;
      case GGMLType.I64:
        return 8;
      case GGMLType.F64:
        return 8;
      default:
        return 1; // Fallback
    }
  }
}
