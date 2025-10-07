/**
 * Dequantization Algorithms
 * Converts quantized weights back to FP32
 */

import { GGMLType } from '../entities/gguf-metadata';

/**
 * Validate buffer has enough bytes
 */
function validateBuffer(buffer: Buffer, offset: number, needed: number, context: string): void {
  if (offset + needed > buffer.length) {
    throw new Error(
      `Buffer underrun in ${context}: need ${needed} bytes at offset ${offset}, ` +
      `but buffer is only ${buffer.length} bytes (missing ${offset + needed - buffer.length} bytes)`
    );
  }
}

/**
 * Read FP16 (half precision float)
 */
function readFloat16(buffer: Buffer, offset: number): number {
  validateBuffer(buffer, offset, 2, 'readFloat16');
  const uint16 = buffer.readUInt16LE(offset);

  // Extract sign, exponent, and mantissa
  const sign = (uint16 & 0x8000) >> 15;
  const exponent = (uint16 & 0x7C00) >> 10;
  const fraction = uint16 & 0x03FF;

  // Handle special cases
  if (exponent === 0) {
    // Subnormal or zero
    return (sign ? -1 : 1) * Math.pow(2, -14) * (fraction / 1024);
  } else if (exponent === 0x1F) {
    // Infinity or NaN
    return fraction ? NaN : (sign ? -Infinity : Infinity);
  }

  // Normal number
  return (sign ? -1 : 1) * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

/**
 * Dequantize F32 (no conversion needed)
 */
export function dequantizeF32(buffer: Buffer, count: number): Float32Array {
  validateBuffer(buffer, 0, count * 4, 'dequantizeF32');
  const result = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    result[i] = buffer.readFloatLE(i * 4);
  }
  return result;
}

/**
 * Dequantize F16 to F32
 */
export function dequantizeF16(buffer: Buffer, count: number): Float32Array {
  validateBuffer(buffer, 0, count * 2, 'dequantizeF16');
  const result = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    result[i] = readFloat16(buffer, i * 2);
  }
  return result;
}

/**
 * Dequantize Q4_0 (4-bit quantization, block size 32)
 * Block format:
 * - 1 x FP16 scale (2 bytes)
 * - 16 x uint8 quantized values (16 bytes, 2 values per byte)
 * Total: 18 bytes per 32 values
 */
export function dequantizeQ4_0(buffer: Buffer, count: number): Float32Array {
  const result = new Float32Array(count);
  const blockSize = 32;
  const bytesPerBlock = 18;
  const numBlocks = Math.ceil(count / blockSize);

  validateBuffer(buffer, 0, numBlocks * bytesPerBlock, 'dequantizeQ4_0');

  let resultOffset = 0;
  let bufferOffset = 0;

  for (let block = 0; block < numBlocks; block++) {
    // Read scale (FP16)
    const scale = readFloat16(buffer, bufferOffset);
    bufferOffset += 2;

    // Read quantized values (4 bits each)
    for (let i = 0; i < blockSize && resultOffset < count; i++) {
      const byteIndex = bufferOffset + Math.floor(i / 2);
      const nibble = (i % 2 === 0)
        ? (buffer[byteIndex] & 0x0F)        // Low nibble
        : ((buffer[byteIndex] >> 4) & 0x0F); // High nibble

      // Dequantize: value = (quantized - 8) * scale
      result[resultOffset++] = (nibble - 8) * scale;
    }

    bufferOffset += 16; // 16 bytes for 32 values
  }

  return result;
}

/**
 * Dequantize Q4_1 (4-bit quantization with min, block size 32)
 * Block format:
 * - 1 x FP16 scale (2 bytes)
 * - 1 x FP16 min (2 bytes)
 * - 16 x uint8 quantized values (16 bytes)
 * Total: 20 bytes per 32 values
 */
export function dequantizeQ4_1(buffer: Buffer, count: number): Float32Array {
  const result = new Float32Array(count);
  const blockSize = 32;
  const bytesPerBlock = 20;
  const numBlocks = Math.ceil(count / blockSize);

  validateBuffer(buffer, 0, numBlocks * bytesPerBlock, 'dequantizeQ4_1');

  let resultOffset = 0;
  let bufferOffset = 0;

  for (let block = 0; block < numBlocks; block++) {
    // Read scale (FP16)
    const scale = readFloat16(buffer, bufferOffset);
    bufferOffset += 2;

    // Read min (FP16)
    const min = readFloat16(buffer, bufferOffset);
    bufferOffset += 2;

    // Read quantized values (4 bits each)
    for (let i = 0; i < blockSize && resultOffset < count; i++) {
      const byteIndex = bufferOffset + Math.floor(i / 2);
      const nibble = (i % 2 === 0)
        ? (buffer[byteIndex] & 0x0F)
        : ((buffer[byteIndex] >> 4) & 0x0F);

      // Dequantize: value = min + quantized * scale
      result[resultOffset++] = min + nibble * scale;
    }

    bufferOffset += 16;
  }

  return result;
}

/**
 * Unpack 6-bit values from packed bytes
 * Used by Q4_K for scales and mins
 *
 * Each 6-bit value can span byte boundaries, so we read 2 bytes and mask.
 * Example: 8 values * 6 bits = 48 bits = 6 bytes needed
 *
 * Edge case: When reading the last value, we might access buffer[byteOffset + 1]
 * which could be out of bounds if we're at the last byte. However, in practice
 * this is safe because:
 * - We read 2 bytes and mask with 0x3F (only keep 6 bits)
 * - The buffer is always allocated with enough space for the full block
 * - For Q4_K: 8 values * 6 bits = 48 bits = 6 bytes exactly
 *
 * We still validate the buffer has enough bytes to be safe.
 */
function unpack6bit(buffer: Buffer, offset: number, count: number): number[] {
  // Calculate bytes needed: ceil(count * 6 / 8)
  const bitsNeeded = count * 6;
  const bytesNeeded = Math.ceil(bitsNeeded / 8);

  // Add 1 extra byte because we might read byteOffset + 1 in the loop
  validateBuffer(buffer, offset, bytesNeeded + 1, 'unpack6bit');

  const result: number[] = [];
  let bitOffset = 0;

  for (let i = 0; i < count; i++) {
    const byteOffset = offset + Math.floor(bitOffset / 8);
    const bitShift = bitOffset % 8;

    // Read 2 bytes and extract 6 bits (may span byte boundary)
    // This formula works for all cases including when bitShift = 0
    const value = ((buffer[byteOffset] >> bitShift) |
                   (buffer[byteOffset + 1] << (8 - bitShift))) & 0x3F;

    result.push(value);
    bitOffset += 6;
  }

  return result;
}

/**
 * Dequantize Q4_K (4-bit k-quantization with super-blocks)
 *
 * Q4_K Block Structure (256 elements per super-block):
 * - 2 bytes: FP16 d (super-block scale for quantized scales)
 * - 2 bytes: FP16 dmin (super-block scale for quantized mins)
 * - 6 bytes: 8 x 6-bit scales (packed, integer range [0,63])
 * - 6 bytes: 8 x 6-bit mins (packed, integer range [0,63])
 * - 128 bytes: quantized data (4 bits per element, integer range [0,15])
 *
 * Total: 144 bytes per 256 elements = 4.5 bits per element
 *
 * Dequantization formula (from llama.cpp):
 *   weight = (d * scales[i]) * quant - (dmin * mins[i])
 *
 * Where:
 *   - d, dmin are FP16 super-block scales
 *   - scales[i], mins[i] are 6-bit integers [0,63] (NOT normalized)
 *   - quant is 4-bit integer [0,15] (NOT centered)
 *
 * Reference: llama.cpp ggml-quants.c dequantize_row_q4_K()
 */
export function dequantizeQ4_K(buffer: Buffer, count: number): Float32Array {
  const result = new Float32Array(count);
  const QK_K = 256; // Super-block size
  const bytesPerBlock = 144; // 2+2+6+6+128
  const numBlocks = Math.ceil(count / QK_K);

  validateBuffer(buffer, 0, numBlocks * bytesPerBlock, 'dequantizeQ4_K');

  let resultOffset = 0;
  let bufferOffset = 0;

  for (let block = 0; block < numBlocks; block++) {
    // Read main scales
    const d = readFloat16(buffer, bufferOffset);
    const dmin = readFloat16(buffer, bufferOffset + 2);
    bufferOffset += 4;

    // Unpack 8 x 6-bit scales
    const scales = unpack6bit(buffer, bufferOffset, 8);
    bufferOffset += 6;

    // Unpack 8 x 6-bit mins
    const mins = unpack6bit(buffer, bufferOffset, 8);
    bufferOffset += 6;

    // Process 8 sub-blocks
    for (let subBlock = 0; subBlock < 8 && resultOffset < count; subBlock++) {
      // llama.cpp formula: y = (d * sc) * quant - (dmin * m)
      // Where sc, m are 6-bit values [0, 63] used as integers (NOT normalized)
      // Reference: llama.cpp ggml-quants.c dequantize_row_q4_K
      const scale = d * scales[subBlock];   // 6-bit scale used directly
      const min = dmin * mins[subBlock];     // 6-bit min used directly

      // Read 32 elements (16 bytes, 4 bits per element)
      for (let i = 0; i < 32 && resultOffset < count; i++) {
        const byteIndex = bufferOffset + Math.floor(i / 2);
        const nibble = (i % 2 === 0)
          ? (buffer[byteIndex] & 0x0F)
          : ((buffer[byteIndex] >> 4) & 0x0F);

        // Dequantize: w = scale * quant - min
        // Quant is [0, 15], min term is subtracted (not added)
        // d and dmin are FP16 super-block scales, already account for 6-bit range
        result[resultOffset++] = scale * nibble - min;
      }

      bufferOffset += 16; // 16 bytes per 32 elements
    }
  }

  return result;
}

/**
 * Dequantize Q6_K (6-bit k-quantization with super-blocks)
 *
 * Q6_K Block Structure (from llama.cpp):
 * - 2 bytes: FP16 d (super-block scale)
 * - 128 bytes: ql (lower 4 bits, 2 values per byte)
 * - 64 bytes: qh (upper 2 bits, 4 values per byte)
 * - 16 bytes: int8 scales (16 sub-blocks of 16 elements)
 *
 * Total: 210 bytes per 256 elements = 6.56 bits per element
 *
 * Each 6-bit value is: (lower 4 bits) | (upper 2 bits << 4)
 * Range: [0, 63], centered at 32
 */
export function dequantizeQ6_K(buffer: Buffer, count: number): Float32Array {
  const result = new Float32Array(count);
  const QK_K = 256; // Super-block size
  const bytesPerBlock = 210; // 128+64+16+2
  const numBlocks = Math.ceil(count / QK_K);

  validateBuffer(buffer, 0, numBlocks * bytesPerBlock, 'dequantizeQ6_K');

  let resultOffset = 0;
  let bufferOffset = 0;

  for (let block = 0; block < numBlocks; block++) {
    // GGUF Q6_K layout: ql, qh, scales, d (d comes LAST, not first!)

    // Read 128 bytes of lower 4 bits (ql)
    const ql = buffer.subarray(bufferOffset, bufferOffset + 128);
    bufferOffset += 128;

    // Read 64 bytes of upper 2 bits (qh)
    const qh = buffer.subarray(bufferOffset, bufferOffset + 64);
    bufferOffset += 64;

    // Read 16 x int8 scales for sub-blocks
    const scales = new Int8Array(16);
    for (let i = 0; i < 16; i++) {
      scales[i] = buffer.readInt8(bufferOffset + i);
    }
    bufferOffset += 16;

    // Read main scale (FP16) - comes LAST in GGUF format!
    const d = readFloat16(buffer, bufferOffset);
    bufferOffset += 2;

    // Dequantize 256 elements
    for (let i = 0; i < QK_K && resultOffset < count; i++) {
      // Get lower 4 bits (2 values per byte in ql)
      const qlByteIndex = i >> 1;
      const qlValue = (i % 2 === 0)
        ? (ql[qlByteIndex] & 0x0F)        // Low nibble
        : ((ql[qlByteIndex] >> 4) & 0x0F); // High nibble

      // Get upper 2 bits (4 values per byte in qh)
      const qhByteIndex = Math.floor(i / 4);
      const qhBitShift = (i % 4) * 2;
      const qhValue = (qh[qhByteIndex] >> qhBitShift) & 0x03;

      // Combine to get 6-bit value: lower 4 bits | (upper 2 bits << 4)
      const q = qlValue | (qhValue << 4); // Range [0, 63]

      // Get sub-block scale (16 sub-blocks of 16 elements)
      const subBlockIndex = Math.floor(i / 16);
      const scale = scales[subBlockIndex];

      // Dequantize: d * scale * (q - 32)
      result[resultOffset++] = d * scale * (q - 32);
    }
  }

  return result;
}

/**
 * Dequantize Q8_0 (8-bit quantization, block size 32)
 */
export function dequantizeQ8_0(buffer: Buffer, count: number): Float32Array {
  const result = new Float32Array(count);
  const blockSize = 32;
  const bytesPerBlock = 34; // 2 + 32
  const numBlocks = Math.ceil(count / blockSize);

  validateBuffer(buffer, 0, numBlocks * bytesPerBlock, 'dequantizeQ8_0');

  let resultOffset = 0;
  let bufferOffset = 0;

  for (let block = 0; block < numBlocks; block++) {
    // Read scale (FP16)
    const scale = readFloat16(buffer, bufferOffset);
    bufferOffset += 2;

    // Read quantized values (8 bits each)
    for (let i = 0; i < blockSize && resultOffset < count; i++) {
      const value = buffer.readInt8(bufferOffset + i);
      result[resultOffset++] = value * scale;
    }

    bufferOffset += 32;
  }

  return result;
}

/**
 * Main dequantization dispatcher
 */
export function dequantize(
  buffer: Buffer,
  type: GGMLType,
  count: number
): Float32Array {
  switch (type) {
    case GGMLType.F32:
      return dequantizeF32(buffer, count);
    case GGMLType.F16:
      return dequantizeF16(buffer, count);
    case GGMLType.Q4_0:
      return dequantizeQ4_0(buffer, count);
    case GGMLType.Q4_1:
      return dequantizeQ4_1(buffer, count);
    case GGMLType.Q4_K:
      return dequantizeQ4_K(buffer, count);
    case GGMLType.Q6_K:
      return dequantizeQ6_K(buffer, count);
    case GGMLType.Q8_0:
      return dequantizeQ8_0(buffer, count);
    default:
      throw new Error(`Dequantization not implemented for type: ${GGMLType[type]}`);
  }
}
