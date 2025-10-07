/**
 * GGUF Metadata Types and Structures
 * Based on GGUF Specification v3
 */

export enum GGUFValueType {
  UINT8 = 0,
  INT8 = 1,
  UINT16 = 2,
  INT16 = 3,
  UINT32 = 4,
  INT32 = 5,
  FLOAT32 = 6,
  BOOL = 7,
  STRING = 8,
  ARRAY = 9,
  UINT64 = 10,
  INT64 = 11,
  FLOAT64 = 12,
}

export enum GGMLType {
  F32 = 0,
  F16 = 1,
  Q4_0 = 2,
  Q4_1 = 3,
  Q5_0 = 6,
  Q5_1 = 7,
  Q8_0 = 8,
  Q8_1 = 9,
  Q2_K = 10,
  Q3_K = 11,
  Q4_K = 12,
  Q5_K = 13,
  Q6_K = 14,
  Q8_K = 15,
  IQ2_XXS = 16,
  IQ2_XS = 17,
  IQ3_XXS = 18,
  IQ1_S = 19,
  IQ4_NL = 20,
  IQ3_S = 21,
  IQ2_S = 22,
  IQ4_XS = 23,
  I8 = 24,
  I16 = 25,
  I32 = 26,
  I64 = 27,
  F64 = 28,
  IQ1_M = 29,
}

export interface GGUFHeader {
  magic: string; // "GGUF"
  version: number; // Currently 3
  tensorCount: bigint;
  metadataKVCount: bigint;
}

export interface GGUFMetadataValue {
  type: GGUFValueType;
  value: any;
}

export interface GGUFMetadata {
  [key: string]: GGUFMetadataValue;
}

export interface TensorInfo {
  name: string;
  dimensions: number[];
  type: GGMLType;
  offset: bigint;
  size: bigint; // Calculated size in bytes
}

export interface GGUFArchitectureInfo {
  // General
  architecture?: string;
  quantizationVersion?: number;
  alignment?: number;

  // Model naming
  name?: string;
  author?: string;
  version?: string;
  organization?: string;
  basename?: string;
  finetune?: string;
  description?: string;

  // Vocabulary
  vocabSize?: number;
  contextLength?: number;
  embeddingLength?: number;

  // Transformer architecture
  blockCount?: number; // Number of transformer layers
  feedForwardLength?: number;
  headCount?: number;
  headCountKV?: number; // For grouped-query attention

  // Attention
  attentionLayerNormEpsilon?: number;
  attentionLayerNormRMSEpsilon?: number;

  // RoPE (Rotary Position Embedding)
  ropeFreqBase?: number;
  ropeDimensionCount?: number;
  ropeScalingType?: string;
  ropeScalingFactor?: number;

  // Tokenizer
  tokenizerModel?: string;
  tokenizerType?: string;
  tokenizerTokens?: string[];
  tokenizerScores?: number[];
  tokenizerTokenType?: number[];
  tokenizerMerges?: string[];

  // Special tokens
  bosTokenId?: number; // Beginning of sequence
  eosTokenId?: number; // End of sequence
  padTokenId?: number; // Padding
  unkTokenId?: number; // Unknown
  sepTokenId?: number; // Separator

  // Other
  fileType?: number;
}

export interface GGUFModel {
  header: GGUFHeader;
  metadata: GGUFMetadata;
  architecture: GGUFArchitectureInfo;
  tensors: TensorInfo[];
  totalParameters: bigint;
  quantizationType: string;
}
