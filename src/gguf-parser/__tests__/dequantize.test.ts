/**
 * Dequantization Algorithm Tests
 *
 * These tests verify the correctness of our dequantization implementations
 * against known reference values and edge cases.
 */

import { describe, it, expect } from '../../shared/utils/test-runner';
import {
  dequantizeF32,
  dequantizeF16,
  dequantizeQ4_0,
  dequantizeQ4_1,
  dequantizeQ4_K,
  dequantizeQ6_K,
  dequantizeQ8_0
} from '../domain/use-cases/dequantize';

describe('Dequantization - F32', () => {
  it('should handle F32 without modification', () => {
    const buffer = Buffer.alloc(16);
    buffer.writeFloatLE(1.5, 0);
    buffer.writeFloatLE(-2.5, 4);
    buffer.writeFloatLE(0.0, 8);
    buffer.writeFloatLE(100.25, 12);

    const result = dequantizeF32(buffer, 4);

    expect.toEqual(result.length, 4);
    expect.toBeCloseTo(result[0], 1.5);
    expect.toBeCloseTo(result[1], -2.5);
    expect.toBeCloseTo(result[2], 0.0);
    expect.toBeCloseTo(result[3], 100.25);
  });

  it('should handle single element', () => {
    const buffer = Buffer.alloc(4);
    buffer.writeFloatLE(42.0, 0);

    const result = dequantizeF32(buffer, 1);

    expect.toEqual(result.length, 1);
    expect.toBeCloseTo(result[0], 42.0);
  });
});

describe('Dequantization - F16', () => {
  it('should dequantize F16 to F32', () => {
    const buffer = Buffer.alloc(8);

    // FP16 encoding of 1.0: 0x3C00
    buffer.writeUInt16LE(0x3C00, 0);
    // FP16 encoding of -1.0: 0xBC00
    buffer.writeUInt16LE(0xBC00, 2);
    // FP16 encoding of 0.0: 0x0000
    buffer.writeUInt16LE(0x0000, 4);
    // FP16 encoding of 2.0: 0x4000
    buffer.writeUInt16LE(0x4000, 6);

    const result = dequantizeF16(buffer, 4);

    expect.toEqual(result.length, 4);
    expect.toBeCloseTo(result[0], 1.0);
    expect.toBeCloseTo(result[1], -1.0);
    expect.toBeCloseTo(result[2], 0.0);
    expect.toBeCloseTo(result[3], 2.0);
  });

  it('should handle F16 special values', () => {
    const buffer = Buffer.alloc(6);

    // Infinity: 0x7C00
    buffer.writeUInt16LE(0x7C00, 0);
    // -Infinity: 0xFC00
    buffer.writeUInt16LE(0xFC00, 2);
    // NaN: 0x7C01
    buffer.writeUInt16LE(0x7C01, 4);

    const result = dequantizeF16(buffer, 3);

    expect.toEqual(result[0], Infinity);
    expect.toEqual(result[1], -Infinity);
    expect.toEqual(Number.isNaN(result[2]), true);
  });
});

describe('Dequantization - Q4_0', () => {
  it('should dequantize Q4_0 block correctly', () => {
    // Q4_0 block: 2 bytes scale + 16 bytes data (32 x 4-bit values)
    const buffer = Buffer.alloc(18);

    // Scale: 0.5 in FP16 (0x3800)
    buffer.writeUInt16LE(0x3800, 0);

    // Quantized values: all 8 (middle of [0,15] range)
    for (let i = 0; i < 16; i++) {
      buffer.writeUInt8(0x88, 2 + i); // 0x88 = nibbles (8, 8)
    }

    const result = dequantizeQ4_0(buffer, 32);

    expect.toEqual(result.length, 32);

    // Q4_0 formula: (q - 8) * scale
    // q = 8, scale = 0.5 → (8 - 8) * 0.5 = 0
    for (let i = 0; i < 32; i++) {
      expect.toBeCloseTo(result[i], 0.0);
    }
  });

  it('should handle partial block', () => {
    const buffer = Buffer.alloc(18);
    buffer.writeUInt16LE(0x3800, 0); // scale = 0.5

    // Fill with 0x00 (nibbles 0, 0)
    for (let i = 0; i < 16; i++) {
      buffer.writeUInt8(0x00, 2 + i);
    }

    const result = dequantizeQ4_0(buffer, 16); // Only 16 elements

    expect.toEqual(result.length, 16);

    // q = 0, scale = 0.5 → (0 - 8) * 0.5 = -4.0
    for (let i = 0; i < 16; i++) {
      expect.toBeCloseTo(result[i], -4.0);
    }
  });
});

describe('Dequantization - Q4_K', () => {
  it('should handle Q4_K super-block structure', () => {
    // Q4_K block: 144 bytes total
    // - 2 bytes: d (FP16)
    // - 2 bytes: dmin (FP16)
    // - 6 bytes: 8 x 6-bit scales
    // - 6 bytes: 8 x 6-bit mins
    // - 128 bytes: 256 x 4-bit values

    const buffer = Buffer.alloc(144);
    let offset = 0;

    // d = 1.0 (FP16: 0x3C00)
    buffer.writeUInt16LE(0x3C00, offset);
    offset += 2;

    // dmin = 0.0 (FP16: 0x0000)
    buffer.writeUInt16LE(0x0000, offset);
    offset += 2;

    // 8 x 6-bit scales (all 31 = middle value)
    // 6 bits per value, packed: need 48 bits = 6 bytes
    // For simplicity, set all to 0xFF (will decode as various 6-bit values)
    for (let i = 0; i < 6; i++) {
      buffer.writeUInt8(0x55, offset + i); // 0x55 = binary 01010101
    }
    offset += 6;

    // 8 x 6-bit mins (all 0)
    for (let i = 0; i < 6; i++) {
      buffer.writeUInt8(0x00, offset + i);
    }
    offset += 6;

    // 256 x 4-bit values (all 8 = middle)
    for (let i = 0; i < 128; i++) {
      buffer.writeUInt8(0x88, offset + i);
    }

    const result = dequantizeQ4_K(buffer, 256);

    expect.toEqual(result.length, 256);

    // Values should be reasonable (not NaN or Infinity)
    for (let i = 0; i < 256; i++) {
      expect.toEqual(Number.isFinite(result[i]), true);
    }
  });

  it('should produce deterministic output', () => {
    const buffer = Buffer.alloc(144);

    // Fill with predictable pattern
    for (let i = 0; i < 144; i++) {
      buffer.writeUInt8(i % 256, i);
    }

    const result1 = dequantizeQ4_K(buffer, 256);
    const result2 = dequantizeQ4_K(buffer, 256);

    expect.toEqual(result1.length, result2.length);

    for (let i = 0; i < result1.length; i++) {
      expect.toEqual(result1[i], result2[i]);
    }
  });

  it('should handle multiple blocks', () => {
    // 2 complete blocks
    const buffer = Buffer.alloc(288);

    const result = dequantizeQ4_K(buffer, 512);

    expect.toEqual(result.length, 512);
  });
});

describe('Dequantization - Q6_K', () => {
  it('should handle Q6_K super-block structure', () => {
    // Q6_K block: 210 bytes total
    // - 128 bytes: ql (lower 4 bits)
    // - 64 bytes: qh (upper 2 bits)
    // - 16 bytes: scales (int8)
    // - 2 bytes: d (FP16)

    const buffer = Buffer.alloc(210);
    let offset = 0;

    // ql: all 0x00 (lower 4 bits all 0)
    for (let i = 0; i < 128; i++) {
      buffer.writeUInt8(0x00, offset + i);
    }
    offset += 128;

    // qh: all 0x00 (upper 2 bits all 0)
    for (let i = 0; i < 64; i++) {
      buffer.writeUInt8(0x00, offset + i);
    }
    offset += 64;

    // scales: all 1
    for (let i = 0; i < 16; i++) {
      buffer.writeInt8(1, offset + i);
    }
    offset += 16;

    // d = 1.0 (FP16: 0x3C00)
    buffer.writeUInt16LE(0x3C00, offset);

    const result = dequantizeQ6_K(buffer, 256);

    expect.toEqual(result.length, 256);

    // q = 0 (from ql=0, qh=0)
    // result = d * scale * (q - 32) = 1.0 * 1 * (0 - 32) = -32
    for (let i = 0; i < 256; i++) {
      expect.toBeCloseTo(result[i], -32.0);
    }
  });

  it('should produce deterministic output', () => {
    const buffer = Buffer.alloc(210);

    for (let i = 0; i < 210; i++) {
      buffer.writeUInt8(i % 256, i);
    }

    const result1 = dequantizeQ6_K(buffer, 256);
    const result2 = dequantizeQ6_K(buffer, 256);

    expect.toEqual(result1.length, result2.length);

    for (let i = 0; i < result1.length; i++) {
      expect.toEqual(result1[i], result2[i]);
    }
  });

  it('should not produce NaN or Infinity with valid input', () => {
    const buffer = Buffer.alloc(210);

    // Fill with semi-random but valid data
    for (let i = 0; i < 210; i++) {
      buffer.writeUInt8((i * 37 + 13) % 256, i);
    }

    const result = dequantizeQ6_K(buffer, 256);

    for (let i = 0; i < 256; i++) {
      expect.toEqual(Number.isFinite(result[i]), true);
      expect.toEqual(Number.isNaN(result[i]), false);
    }
  });
});

describe('Dequantization - Edge Cases', () => {
  it('should handle zero-length arrays', () => {
    const buffer = Buffer.alloc(0);

    const result = dequantizeF32(buffer, 0);

    expect.toEqual(result.length, 0);
  });

  it('should handle count not multiple of block size', () => {
    // Q4_K block is 256 elements, test with 300
    const buffer = Buffer.alloc(288); // 2 complete blocks

    const result = dequantizeQ4_K(buffer, 300);

    expect.toEqual(result.length, 300);
  });
});

describe('Dequantization - Boundary Conditions', () => {
  it('should handle exactly one block (Q4_K)', () => {
    // Exactly 256 elements = 1 block
    const buffer = Buffer.alloc(144);

    const result = dequantizeQ4_K(buffer, 256);

    expect.toEqual(result.length, 256);
  });

  it('should handle one element over block boundary (Q4_K)', () => {
    // 257 elements = needs 2 blocks (144 * 2 = 288 bytes)
    const buffer = Buffer.alloc(288);

    const result = dequantizeQ4_K(buffer, 257);

    expect.toEqual(result.length, 257);
  });

  it('should handle partial block (Q4_K)', () => {
    // 300 elements = needs 2 blocks
    const buffer = Buffer.alloc(288);

    const result = dequantizeQ4_K(buffer, 300);

    expect.toEqual(result.length, 300);
  });

  it('should reject insufficient buffer (Q4_K)', () => {
    // 300 elements needs 288 bytes, but only provide 144
    const buffer = Buffer.alloc(144);

    let errorThrown = false;
    try {
      dequantizeQ4_K(buffer, 300);
    } catch (e: any) {
      errorThrown = true;
      expect.toEqual(e.message.includes('Buffer underrun'), true);
    }

    expect.toEqual(errorThrown, true);
  });

  it('should handle exactly one block (Q6_K)', () => {
    // Exactly 256 elements = 1 block (210 bytes)
    const buffer = Buffer.alloc(210);

    const result = dequantizeQ6_K(buffer, 256);

    expect.toEqual(result.length, 256);
  });

  it('should handle partial block (Q6_K)', () => {
    // 300 elements = needs 2 blocks (420 bytes)
    const buffer = Buffer.alloc(420);

    const result = dequantizeQ6_K(buffer, 300);

    expect.toEqual(result.length, 300);
  });

  it('should handle exactly one block (Q4_0)', () => {
    // Exactly 32 elements = 1 block (18 bytes)
    const buffer = Buffer.alloc(18);
    buffer.writeUInt16LE(0x3C00, 0); // scale = 1.0

    const result = dequantizeQ4_0(buffer, 32);

    expect.toEqual(result.length, 32);
  });

  it('should handle partial block (Q4_0)', () => {
    // 50 elements = needs 2 blocks (36 bytes)
    const buffer = Buffer.alloc(36);
    buffer.writeUInt16LE(0x3C00, 0); // scale = 1.0 for block 1
    buffer.writeUInt16LE(0x3C00, 18); // scale = 1.0 for block 2

    const result = dequantizeQ4_0(buffer, 50);

    expect.toEqual(result.length, 50);
  });
});

describe('Dequantization - Performance', () => {
  it('should dequantize large tensor efficiently', () => {
    // 1M elements
    const count = 1024 * 1024;
    const blockCount = Math.ceil(count / 256);
    const buffer = Buffer.alloc(blockCount * 144);

    const start = Date.now();
    const result = dequantizeQ4_K(buffer, count);
    const elapsed = Date.now() - start;

    expect.toEqual(result.length, count);

    // Should complete in reasonable time (<100ms for 1M elements)
    expect.toBeLessThan(elapsed, 100);
  });
});
