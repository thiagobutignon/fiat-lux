/**
 * Dequantizer - Converts quantized tensors to FP32
 *
 * Based on ggml reference implementation:
 * https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c
 *
 * Supports:
 * - Q4_K: 4-bit quantization with K-quants (super-blocks)
 * - Q6_K: 6-bit quantization with K-quants
 */

import { GGMLType } from '../../../gguf-parser/domain/entities/gguf-metadata';

// K-quants constants
const QK_K = 256; // Super-block size for K-quants

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
   *
   * Based on ggml dequantize_row_q4_K:
   * - Block size: 144 bytes (d + dmin + scales[12] + qs[128])
   * - 256 elements per block (QK_K)
   * - 8 sub-blocks of 32 elements each
   * - Formula: y = d * sc * q - min * m
   */
  private dequantizeQ4K(buffer: Buffer, offset: number, elementCount: number): number[] {
    const result: number[] = [];
    const numBlocks = Math.ceil(elementCount / QK_K);
    const BLOCK_SIZE = 144;

    for (let blockIdx = 0; blockIdx < numBlocks; blockIdx++) {
      const blockOffset = offset + blockIdx * BLOCK_SIZE;

      // Read block header
      const d = this.readFP16(buffer, blockOffset);
      const dmin = this.readFP16(buffer, blockOffset + 2);
      const scales = buffer.subarray(blockOffset + 4, blockOffset + 16); // 12 bytes
      const qsOffset = blockOffset + 16;

      const elementsInBlock = Math.min(QK_K, elementCount - blockIdx * QK_K);

      // Process 256 elements in groups of 64 (4 iterations)
      let qIdx = 0;
      let scaleIdx = 0;

      for (let j = 0; j < QK_K && qIdx < elementsInBlock; j += 64) {
        // Get scale/min for first 32 elements
        const [sc1, m1] = this.getScaleMinK4(scaleIdx, scales);
        const d1 = d * sc1;
        const m1val = dmin * m1;

        // Get scale/min for next 32 elements
        const [sc2, m2] = this.getScaleMinK4(scaleIdx + 1, scales);
        const d2 = d * sc2;
        const m2val = dmin * m2;

        // Dequantize 32 elements with first scale (lower 4 bits)
        for (let l = 0; l < 32 && qIdx < elementsInBlock; l++, qIdx++) {
          const qByte = buffer.readUInt8(qsOffset + Math.floor((j + l) / 2));
          const q = qByte & 0xf;
          result.push(d1 * q - m1val);
        }

        // Dequantize 32 elements with second scale (upper 4 bits)
        for (let l = 0; l < 32 && qIdx < elementsInBlock; l++, qIdx++) {
          const qByte = buffer.readUInt8(qsOffset + Math.floor((j + l) / 2));
          const q = qByte >> 4;
          result.push(d2 * q - m2val);
        }

        scaleIdx += 2;
      }
    }

    return result;
  }

  /**
   * Unpack 6-bit scale and min from Q4_K scales array
   *
   * Based on ggml get_scale_min_k4:
   * The 12 bytes contain 8 scale/min pairs, each 6-bit (0-63)
   */
  private getScaleMinK4(j: number, scales: Uint8Array): [number, number] {
    let d: number, m: number;

    if (j < 4) {
      d = scales[j] & 63;
      m = scales[j + 4] & 63;
    } else {
      d = (scales[j + 4] & 0xf) | ((scales[j - 4] >> 6) << 4);
      m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }

    return [d, m];
  }

  /**
   * Dequantize Q6_K (6-bit K-quants)
   *
   * Based on ggml dequantize_row_q6_K:
   * - Block size: 210 bytes (ql[128] + qh[64] + scales[16] + d)
   * - 256 elements per block (QK_K)
   * - 16 sub-blocks of 16 elements each
   * - Formula: y = d * scale * q (where q is signed 6-bit: -32 to 31)
   */
  private dequantizeQ6K(buffer: Buffer, offset: number, elementCount: number): number[] {
    const result: number[] = [];
    const numBlocks = Math.ceil(elementCount / QK_K);
    const BLOCK_SIZE = 210;

    for (let blockIdx = 0; blockIdx < numBlocks; blockIdx++) {
      const blockOffset = offset + blockIdx * BLOCK_SIZE;

      // Read block components
      const qlOffset = blockOffset;
      const qhOffset = blockOffset + 128;
      const scalesOffset = blockOffset + 192;
      const d = this.readFP16(buffer, scalesOffset + 16);

      const elementsInBlock = Math.min(QK_K, elementCount - blockIdx * QK_K);

      // Process 256 elements in groups of 128 (2 iterations)
      let yIdx = 0;

      for (let n = 0; n < QK_K && yIdx < elementsInBlock; n += 128) {
        const qlBase = qlOffset + (n / 2);
        const qhBase = qhOffset + (n / 4);
        const scBase = scalesOffset + (n / 16);

        for (let l = 0; l < 32 && yIdx < elementsInBlock; l++) {
          const is = Math.floor(l / 16);

          // Read quantized values and scales
          const qlByte1 = buffer.readUInt8(qlBase + l);
          const qlByte2 = buffer.readUInt8(qlBase + l + 32);
          const qhByte = buffer.readUInt8(qhBase + l);

          const sc0 = buffer.readInt8(scBase + is + 0);
          const sc2 = buffer.readInt8(scBase + is + 2);
          const sc4 = buffer.readInt8(scBase + is + 4);
          const sc6 = buffer.readInt8(scBase + is + 6);

          // Reconstruct 6-bit values (as signed: -32 to 31)
          const q1 = ((qlByte1 & 0xf) | ((qhByte & 0x3) << 4)) - 32;
          const q2 = ((qlByte2 & 0xf) | (((qhByte >> 2) & 0x3) << 4)) - 32;
          const q3 = ((qlByte1 >> 4) | (((qhByte >> 4) & 0x3) << 4)) - 32;
          const q4 = ((qlByte2 >> 4) | (((qhByte >> 6) & 0x3) << 4)) - 32;

          // Dequantize (4 elements per loop iteration)
          if (yIdx < elementsInBlock) result.push(d * sc0 * q1);
          yIdx++;
          if (yIdx < elementsInBlock) result.push(d * sc2 * q2);
          yIdx++;
          if (yIdx < elementsInBlock) result.push(d * sc4 * q3);
          yIdx++;
          if (yIdx < elementsInBlock) result.push(d * sc6 * q4);
          yIdx++;
        }
      }
    }

    return result;
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
