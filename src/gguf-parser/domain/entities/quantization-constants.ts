/**
 * Quantization Constants
 *
 * This file contains all magic numbers used in GGUF quantization/dequantization.
 * Each constant is documented with:
 * - Origin (GGUF spec, llama.cpp, mathematical derivation)
 * - Purpose
 * - Valid range
 *
 * Reference: llama.cpp ggml-quants.c, ggml-common.h
 */

// ============================================================================
// Block Sizes
// ============================================================================

/**
 * Standard block size for non-K quantization types (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1)
 * @source GGUF Specification v3, llama.cpp ggml-common.h
 */
export const QK_STANDARD = 32;

/**
 * Super-block size for K-quantization types (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K)
 * Each super-block is divided into sub-blocks
 * @source GGUF Specification v3, llama.cpp ggml-common.h QK_K
 */
export const QK_K = 256;

// ============================================================================
// Bytes Per Block (from GGUF spec and llama.cpp)
// ============================================================================

/**
 * Q4_0: 2 bytes (FP16 scale) + 16 bytes (4-bit quants, 2 per byte)
 * @formula 2 + (32 * 4 / 8) = 2 + 16 = 18 bytes
 */
export const BYTES_PER_BLOCK_Q4_0 = 18;

/**
 * Q4_1: 2 bytes (FP16 scale) + 2 bytes (FP16 min) + 16 bytes (4-bit quants)
 * @formula 2 + 2 + (32 * 4 / 8) = 2 + 2 + 16 = 20 bytes
 */
export const BYTES_PER_BLOCK_Q4_1 = 20;

/**
 * Q5_0: 2 bytes (FP16 scale) + 4 bytes (high bits) + 16 bytes (low 4 bits)
 * @formula 2 + 4 + (32 * 4 / 8) = 2 + 4 + 16 = 22 bytes
 */
export const BYTES_PER_BLOCK_Q5_0 = 22;

/**
 * Q5_1: 2 bytes (FP16 scale) + 2 bytes (FP16 min) + 4 bytes (high bits) + 16 bytes (low 4 bits)
 * @formula 2 + 2 + 4 + (32 * 4 / 8) = 2 + 2 + 4 + 16 = 24 bytes
 */
export const BYTES_PER_BLOCK_Q5_1 = 24;

/**
 * Q8_0: 2 bytes (FP16 scale) + 32 bytes (8-bit quants)
 * @formula 2 + (32 * 8 / 8) = 2 + 32 = 34 bytes
 */
export const BYTES_PER_BLOCK_Q8_0 = 34;

/**
 * Q8_1: 2 bytes (FP16 scale) + 2 bytes (FP16 min) + 32 bytes (8-bit quants)
 * @formula 2 + 2 + (32 * 8 / 8) = 2 + 2 + 32 = 36 bytes
 */
export const BYTES_PER_BLOCK_Q8_1 = 36;

/**
 * Q2_K: 256 elements, 2.5 bits per element
 * @formula 256 * 2.5 / 8 = 80 bytes
 * @source llama.cpp ggml-common.h block_q2_K
 */
export const BYTES_PER_BLOCK_Q2_K = 80;

/**
 * Q3_K: 256 elements, 3.5 bits per element
 * @formula 256 * 3.5 / 8 = 112 bytes
 * @source llama.cpp ggml-common.h block_q3_K
 */
export const BYTES_PER_BLOCK_Q3_K = 112;

/**
 * Q4_K: 256 elements, 4.5 bits per element
 * Structure: 2 (d) + 2 (dmin) + 6 (8x6-bit scales) + 6 (8x6-bit mins) + 128 (4-bit quants)
 * @formula 2 + 2 + 6 + 6 + 128 = 144 bytes
 * @source llama.cpp ggml-common.h block_q4_K
 */
export const BYTES_PER_BLOCK_Q4_K = 144;

/**
 * Q5_K: 256 elements, 5.5 bits per element
 * @formula 256 * 5.5 / 8 = 176 bytes
 * @source llama.cpp ggml-common.h block_q5_K
 */
export const BYTES_PER_BLOCK_Q5_K = 176;

/**
 * Q6_K: 256 elements, 6.5625 bits per element
 * Structure: 128 (ql) + 64 (qh) + 16 (scales) + 2 (d)
 * @formula 128 + 64 + 16 + 2 = 210 bytes
 * @source llama.cpp ggml-common.h block_q6_K
 */
export const BYTES_PER_BLOCK_Q6_K = 210;

/**
 * Q8_K: 256 elements, 8.5 bits per element
 * @formula 256 * 8.5 / 8 = 272 bytes
 * @source llama.cpp ggml-common.h block_q8_K
 */
export const BYTES_PER_BLOCK_Q8_K = 272;

// ============================================================================
// Q4_K Super-block Structure
// ============================================================================

/**
 * Number of sub-blocks in a Q4_K super-block
 * Each super-block (256 elements) is divided into 8 sub-blocks of 32 elements
 * @source llama.cpp ggml-quants.c dequantize_row_q4_K
 */
export const Q4_K_SUB_BLOCKS = 8;

/**
 * Elements per sub-block in Q4_K
 * @formula QK_K / Q4_K_SUB_BLOCKS = 256 / 8 = 32
 */
export const Q4_K_ELEMENTS_PER_SUB_BLOCK = 32;

/**
 * Bytes for 6-bit packed scales in Q4_K
 * 8 scales × 6 bits = 48 bits = 6 bytes
 * @formula 8 * 6 / 8 = 6 bytes
 */
export const Q4_K_SCALES_BYTES = 6;

/**
 * Bytes for 6-bit packed mins in Q4_K
 * 8 mins × 6 bits = 48 bits = 6 bytes
 * @formula 8 * 6 / 8 = 6 bytes
 */
export const Q4_K_MINS_BYTES = 6;

/**
 * Bytes for quantized data in Q4_K
 * 256 elements × 4 bits = 1024 bits = 128 bytes
 * @formula 256 * 4 / 8 = 128 bytes
 */
export const Q4_K_QUANT_BYTES = 128;

// ============================================================================
// Q6_K Super-block Structure
// ============================================================================

/**
 * Number of sub-blocks in Q6_K
 * Each super-block (256 elements) is divided into 16 sub-blocks of 16 elements
 * @source llama.cpp ggml-quants.c dequantize_row_q6_K
 */
export const Q6_K_SUB_BLOCKS = 16;

/**
 * Elements per sub-block in Q6_K
 * @formula QK_K / Q6_K_SUB_BLOCKS = 256 / 16 = 16
 */
export const Q6_K_ELEMENTS_PER_SUB_BLOCK = 16;

/**
 * Bytes for lower 4 bits in Q6_K
 * 256 elements × 4 bits = 1024 bits = 128 bytes
 * @formula 256 * 4 / 8 = 128 bytes
 */
export const Q6_K_QL_BYTES = 128;

/**
 * Bytes for upper 2 bits in Q6_K
 * 256 elements × 2 bits = 512 bits = 64 bytes
 * @formula 256 * 2 / 8 = 64 bytes
 */
export const Q6_K_QH_BYTES = 64;

/**
 * Bytes for int8 scales in Q6_K
 * 16 sub-blocks × 1 byte (int8) = 16 bytes
 */
export const Q6_K_SCALES_BYTES = 16;

// ============================================================================
// Bit Sizes and Masks
// ============================================================================

/**
 * 4-bit nibble mask
 * Used to extract 4-bit values from bytes
 * @value 0x0F = 0b00001111 = 15
 */
export const NIBBLE_MASK = 0x0F;

/**
 * 6-bit value mask
 * Used to extract 6-bit values from packed bytes
 * @value 0x3F = 0b00111111 = 63
 */
export const SIXBIT_MASK = 0x3F;

/**
 * Maximum value for 4-bit unsigned integer
 * @value 2^4 - 1 = 15
 */
export const MAX_4BIT = 15;

/**
 * Maximum value for 6-bit unsigned integer
 * Used as denominator in some quantization schemes
 * @value 2^6 - 1 = 63
 */
export const MAX_6BIT = 63;

/**
 * Maximum value for int8 (signed byte)
 * @value 2^7 - 1 = 127
 */
export const MAX_INT8 = 127;

/**
 * Centering offset for Q4_0 and Q4_1
 * Centers 4-bit values [0,15] to signed range [-8,7]
 * @value 8
 * @note This is for Q4_0/Q4_1, NOT for Q4_K (which doesn't use centering)
 */
export const Q4_CENTER_OFFSET = 8;

/**
 * Centering offset for 6-bit values in Q6_K
 * Centers 6-bit values [0,63] to signed range [-32,31]
 * @value 32
 */
export const Q6_CENTER_OFFSET = 32;

// ============================================================================
// FP16 Constants
// ============================================================================

/**
 * FP16 exponent bias
 * Used in half-precision float conversion
 * @value 15 (for FP16)
 * @reference IEEE 754 half-precision specification
 */
export const FP16_EXPONENT_BIAS = 15;

/**
 * FP16 subnormal exponent value
 * When exponent bits are all 0, number is subnormal
 * @value -14 (effective exponent for subnormals)
 */
export const FP16_SUBNORMAL_EXPONENT = -14;

/**
 * FP16 mantissa divisor
 * Mantissa is 10 bits, normalized by dividing by 2^10
 * @value 1024 = 2^10
 */
export const FP16_MANTISSA_DIVISOR = 1024;

// ============================================================================
// Default Alignment
// ============================================================================

/**
 * Default tensor data alignment in GGUF files
 * Tensor data is aligned to 32-byte boundaries unless otherwise specified
 * @value 32 bytes
 * @source GGUF Specification v3
 */
export const DEFAULT_ALIGNMENT = 32;

// ============================================================================
// Byte Sizes for Primitive Types
// ============================================================================

/**
 * Size of FP16 (half-precision float) in bytes
 */
export const SIZEOF_FP16 = 2;

/**
 * Size of FP32 (single-precision float) in bytes
 */
export const SIZEOF_FP32 = 4;

/**
 * Size of FP64 (double-precision float) in bytes
 */
export const SIZEOF_FP64 = 8;

/**
 * Size of int8 in bytes
 */
export const SIZEOF_INT8 = 1;

/**
 * Size of int16 in bytes
 */
export const SIZEOF_INT16 = 2;

/**
 * Size of int32 in bytes
 */
export const SIZEOF_INT32 = 4;

/**
 * Size of int64 in bytes
 */
export const SIZEOF_INT64 = 8;

// ============================================================================
// Bit Operations
// ============================================================================

/**
 * Bits per byte
 */
export const BITS_PER_BYTE = 8;

/**
 * Number of 4-bit values per byte
 * @value 2 (byte has 8 bits, 8/4 = 2)
 */
export const NIBBLES_PER_BYTE = 2;

/**
 * Bit shift for extracting high nibble (upper 4 bits)
 * @value 4
 */
export const HIGH_NIBBLE_SHIFT = 4;
