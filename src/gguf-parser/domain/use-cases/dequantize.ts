/**
 * Dequantization Algorithms
 * Converts quantized weights back to FP32
 */

import { GGMLType } from '../entities/gguf-metadata';

/**
 * Read FP16 (half precision float)
 */
function readFloat16(buffer: Buffer, offset: number): number {
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

  let resultOffset = 0;
  let bufferOffset = 0;

  for (let block = 0; block < Math.ceil(count / blockSize); block++) {
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

  let resultOffset = 0;
  let bufferOffset = 0;

  for (let block = 0; block < Math.ceil(count / blockSize); block++) {
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
 * Dequantize Q4_K (4-bit k-quantization, block size 256)
 * More complex format with multiple scales and minimums
 */
export function dequantizeQ4_K(buffer: Buffer, count: number): Float32Array {
  const result = new Float32Array(count);
  const blockSize = 256;
  const bytesPerBlock = 144; // Approximate for Q4_K

  let resultOffset = 0;
  let bufferOffset = 0;

  for (let block = 0; block < Math.ceil(count / blockSize); block++) {
    // Q4_K has complex structure with super-blocks
    // For now, use simplified approximation

    // Read main scale (FP16)
    const scale = readFloat16(buffer, bufferOffset);
    bufferOffset += 2;

    // Read min (FP16)
    const min = readFloat16(buffer, bufferOffset);
    bufferOffset += 2;

    // Skip sub-scales (6 bytes for 6 sub-blocks)
    bufferOffset += 6;

    // Read quantized values (128 bytes for 256 values at 4 bits each)
    for (let i = 0; i < blockSize && resultOffset < count; i++) {
      const byteIndex = bufferOffset + Math.floor(i / 2);
      const nibble = (i % 2 === 0)
        ? (buffer[byteIndex] & 0x0F)
        : ((buffer[byteIndex] >> 4) & 0x0F);

      // Simplified dequantization
      result[resultOffset++] = min + (nibble / 15.0) * scale;
    }

    bufferOffset += 128; // Skip to next block
  }

  return result;
}

/**
 * Dequantize Q6_K (6-bit k-quantization, block size 256)
 */
export function dequantizeQ6_K(buffer: Buffer, count: number): Float32Array {
  const result = new Float32Array(count);
  const blockSize = 256;
  const bytesPerBlock = 210; // Approximate for Q6_K

  let resultOffset = 0;
  let bufferOffset = 0;

  for (let block = 0; block < Math.ceil(count / blockSize); block++) {
    // Read scale (FP16)
    const scale = readFloat16(buffer, bufferOffset);
    bufferOffset += 2;

    // Skip sub-scales
    bufferOffset += 16;

    // Read quantized values (6 bits each = 192 bytes for 256 values)
    for (let i = 0; i < blockSize && resultOffset < count; i++) {
      // Q6_K packs 4 values into 3 bytes
      const groupIndex = Math.floor(i / 4);
      const valueInGroup = i % 4;
      const byteOffset = bufferOffset + groupIndex * 3;

      let value: number;
      if (valueInGroup === 0) {
        // First 6 bits
        value = buffer[byteOffset] & 0x3F;
      } else if (valueInGroup === 1) {
        // Next 6 bits (spans bytes)
        value = ((buffer[byteOffset] >> 6) | ((buffer[byteOffset + 1] & 0x0F) << 2)) & 0x3F;
      } else if (valueInGroup === 2) {
        // Next 6 bits
        value = ((buffer[byteOffset + 1] >> 4) | ((buffer[byteOffset + 2] & 0x03) << 4)) & 0x3F;
      } else {
        // Last 6 bits
        value = (buffer[byteOffset + 2] >> 2) & 0x3F;
      }

      // Dequantize: value = (quantized - 32) * scale
      result[resultOffset++] = (value - 32) * scale;
    }

    bufferOffset += 192; // 192 bytes for 256 values at 6 bits each
  }

  return result;
}

/**
 * Dequantize Q8_0 (8-bit quantization, block size 32)
 */
export function dequantizeQ8_0(buffer: Buffer, count: number): Float32Array {
  const result = new Float32Array(count);
  const blockSize = 32;

  let resultOffset = 0;
  let bufferOffset = 0;

  for (let block = 0; block < Math.ceil(count / blockSize); block++) {
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
