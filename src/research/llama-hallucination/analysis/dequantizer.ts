/**
 * Dequantizer - Converts quantized tensors to FP32
 *
 * Supports:
 * - Q4_K: 4-bit quantization with K-quants (super-blocks)
 * - Q6_K: 6-bit quantization with K-quants
 *
 * References:
 * - ggml quantization: https://github.com/ggerganov/ggml/blob/master/src/ggml-quants.c
 * - GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
 */

import { GGMLType } from '../../../gguf-parser/domain/entities/gguf-metadata';

// K-quants constants
const QK_K = 256; // Super-block size for K-quants
const K_SCALE_SIZE = 12; // Number of scale bytes in Q4_K/Q6_K

/**
 * Q4_K block structure (256 elements per block)
 *
 * Block layout (~144 bytes):
 * - scales_and_mins: 12 bytes (6 scales + 6 mins, 4-bit packed)
 * - d: 2 bytes (FP16 delta)
 * - dmin: 2 bytes (FP16 min delta)
 * - qs: 128 bytes (256 * 4-bit values packed)
 *
 * Total: ~144 bytes per 256 elements
 */
interface Q4KBlock {
  scales: number[]; // 6 scales (4-bit each, unpacked)
  mins: number[]; // 6 mins (4-bit each, unpacked)
  d: number; // FP16 delta (main scale)
  dmin: number; // FP16 min delta
  qs: Uint8Array; // Quantized values (4-bit packed into bytes)
}

/**
 * Q6_K block structure (256 elements per block)
 *
 * Block layout (~210 bytes):
 * - scales: 16 bytes (16 * 8-bit scales)
 * - d: 2 bytes (FP16 delta)
 * - ql: 128 bytes (lower 4 bits of 6-bit values)
 * - qh: 64 bytes (upper 2 bits of 6-bit values)
 *
 * Total: ~210 bytes per 256 elements
 */
interface Q6KBlock {
  scales: Uint8Array; // 16 scales (8-bit each)
  d: number; // FP16 delta
  ql: Uint8Array; // Lower 4 bits (128 bytes)
  qh: Uint8Array; // Upper 2 bits (64 bytes)
}

export class Dequantizer {
  /**
   * Dequantize a tensor based on its type
   */
  dequantize(buffer: Buffer, offset: number, elementCount: number, type: GGMLType): number[] {
    switch (type) {
      case GGMLType.Q4_K:
        return this.dequantizeQ4K(buffer, offset, elementCount);
      case GGMLType.Q6_K:
        return this.dequantizeQ6K(buffer, offset, elementCount);
      case GGMLType.F16:
        return this.dequantizeFP16(buffer, offset, elementCount);
      case GGMLType.F32:
        return this.dequantizeFP32(buffer, offset, elementCount);
      default:
        throw new Error(`Unsupported quantization type: ${GGMLType[type]}`);
    }
  }

  /**
   * Dequantize Q4_K (4-bit K-quants)
   */
  private dequantizeQ4K(buffer: Buffer, offset: number, elementCount: number): number[] {
    const result: number[] = [];
    const numBlocks = Math.ceil(elementCount / QK_K);

    for (let blockIdx = 0; blockIdx < numBlocks; blockIdx++) {
      const blockOffset = offset + blockIdx * 144; // ~144 bytes per Q4_K block

      // Parse block
      const block = this.parseQ4KBlock(buffer, blockOffset);

      // Dequantize 256 elements (or fewer for last block)
      const elementsInBlock = Math.min(QK_K, elementCount - blockIdx * QK_K);

      for (let i = 0; i < elementsInBlock; i++) {
        // Determine which scale/min to use (6 groups of ~42-43 elements each)
        const scaleIdx = Math.floor((i * 6) / QK_K);
        const scale = block.scales[scaleIdx];
        const min = block.mins[scaleIdx];

        // Get quantized value (4-bit)
        const byteIdx = Math.floor(i / 2);
        const nibbleIdx = i % 2;
        const q = nibbleIdx === 0 ? block.qs[byteIdx] & 0xf : (block.qs[byteIdx] >> 4) & 0xf;

        // Dequantize: value = d * (scale * q - dmin * min)
        const dequantized = block.d * (scale * q - block.dmin * min);
        result.push(dequantized);
      }
    }

    return result;
  }

  /**
   * Parse Q4_K block from buffer
   */
  private parseQ4KBlock(buffer: Buffer, offset: number): Q4KBlock {
    let pos = offset;

    // Read scales and mins (12 bytes, 4-bit packed)
    // First 6 bytes: scales (2 per byte)
    // Next 6 bytes: mins (2 per byte)
    const scales: number[] = [];
    const mins: number[] = [];

    for (let i = 0; i < 6; i++) {
      const scaleByte = buffer.readUInt8(pos++);
      scales.push(scaleByte & 0xf); // Lower nibble
      scales.push((scaleByte >> 4) & 0xf); // Upper nibble
    }

    for (let i = 0; i < 6; i++) {
      const minByte = buffer.readUInt8(pos++);
      mins.push(minByte & 0xf);
      mins.push((minByte >> 4) & 0xf);
    }

    // Only use first 6 scales/mins (12 packed into 12 bytes)
    const finalScales = scales.slice(0, 6);
    const finalMins = mins.slice(0, 6);

    // Read d (FP16)
    const d = this.readFP16(buffer, pos);
    pos += 2;

    // Read dmin (FP16)
    const dmin = this.readFP16(buffer, pos);
    pos += 2;

    // Read quantized values (128 bytes for 256 4-bit values)
    const qs = buffer.subarray(pos, pos + 128);

    return {
      scales: finalScales,
      mins: finalMins,
      d,
      dmin,
      qs,
    };
  }

  /**
   * Dequantize Q6_K (6-bit K-quants)
   */
  private dequantizeQ6K(buffer: Buffer, offset: number, elementCount: number): number[] {
    const result: number[] = [];
    const numBlocks = Math.ceil(elementCount / QK_K);

    for (let blockIdx = 0; blockIdx < numBlocks; blockIdx++) {
      const blockOffset = offset + blockIdx * 210; // ~210 bytes per Q6_K block

      // Parse block
      const block = this.parseQ6KBlock(buffer, blockOffset);

      // Dequantize 256 elements (or fewer for last block)
      const elementsInBlock = Math.min(QK_K, elementCount - blockIdx * QK_K);

      for (let i = 0; i < elementsInBlock; i++) {
        // Determine which scale to use (16 groups of 16 elements each)
        const scaleIdx = Math.floor(i / 16);
        const scale = block.scales[scaleIdx];

        // Reconstruct 6-bit value from lower 4 bits and upper 2 bits
        const ql = block.ql[i]; // Lower 4 bits
        const qhByteIdx = Math.floor(i / 4);
        const qhBitIdx = (i % 4) * 2;
        const qh = (block.qh[qhByteIdx] >> qhBitIdx) & 0x3; // Upper 2 bits

        const q = (qh << 4) | ql; // Combine to 6-bit value (0-63)

        // Dequantize: value = d * scale * (q - 32)
        const dequantized = block.d * scale * (q - 32);
        result.push(dequantized);
      }
    }

    return result;
  }

  /**
   * Parse Q6_K block from buffer
   */
  private parseQ6KBlock(buffer: Buffer, offset: number): Q6KBlock {
    let pos = offset;

    // Read scales (16 bytes, 8-bit each)
    const scales = buffer.subarray(pos, pos + 16);
    pos += 16;

    // Read d (FP16)
    const d = this.readFP16(buffer, pos);
    pos += 2;

    // Read ql (lower 4 bits, 128 bytes)
    const ql = buffer.subarray(pos, pos + 128);
    pos += 128;

    // Read qh (upper 2 bits, 64 bytes)
    const qh = buffer.subarray(pos, pos + 64);

    return {
      scales,
      d,
      ql,
      qh,
    };
  }

  /**
   * Dequantize FP16 to FP32
   */
  private dequantizeFP16(buffer: Buffer, offset: number, elementCount: number): number[] {
    const result: number[] = [];
    for (let i = 0; i < elementCount; i++) {
      const fp16 = buffer.readUInt16LE(offset + i * 2);
      result.push(this.fp16ToFloat(fp16));
    }
    return result;
  }

  /**
   * Read FP32 values directly
   */
  private dequantizeFP32(buffer: Buffer, offset: number, elementCount: number): number[] {
    const result: number[] = [];
    for (let i = 0; i < elementCount; i++) {
      result.push(buffer.readFloatLE(offset + i * 4));
    }
    return result;
  }

  /**
   * Convert FP16 to FP32
   */
  private readFP16(buffer: Buffer, offset: number): number {
    const fp16 = buffer.readUInt16LE(offset);
    return this.fp16ToFloat(fp16);
  }

  /**
   * Convert FP16 to FP32 (IEEE 754)
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
}
