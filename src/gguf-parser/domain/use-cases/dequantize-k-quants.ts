/**
 * Accurate K-Quant Dequantization Algorithms
 * Based on llama.cpp implementation
 *
 * Q4_K: 4-bit k-quantization with super-blocks
 * Q6_K: 6-bit k-quantization with super-blocks
 */

/**
 * Read FP16 (half precision float)
 */
function readFloat16(buffer: Buffer, offset: number): number {
  const uint16 = buffer.readUInt16LE(offset);

  const sign = (uint16 & 0x8000) >> 15;
  const exponent = (uint16 & 0x7C00) >> 10;
  const fraction = uint16 & 0x03FF;

  if (exponent === 0) {
    return (sign ? -1 : 1) * Math.pow(2, -14) * (fraction / 1024);
  } else if (exponent === 0x1F) {
    return fraction ? NaN : (sign ? -Infinity : Infinity);
  }

  return (sign ? -1 : 1) * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

/**
 * Unpack 6-bit values from packed bytes
 * Q4_K stores scales and mins as 6-bit values packed into bytes
 */
function unpack6bit(buffer: Buffer, offset: number, count: number): number[] {
  const result: number[] = [];
  let bitOffset = 0;

  for (let i = 0; i < count; i++) {
    const byteOffset = offset + Math.floor(bitOffset / 8);
    const bitShift = bitOffset % 8;

    let value: number;

    if (bitShift <= 2) {
      // Value fits in current byte and next
      value = ((buffer[byteOffset] >> bitShift) | (buffer[byteOffset + 1] << (8 - bitShift))) & 0x3F;
    } else {
      // Value spans current and next byte
      value = ((buffer[byteOffset] >> bitShift) | (buffer[byteOffset + 1] << (8 - bitShift))) & 0x3F;
    }

    result.push(value);
    bitOffset += 6;
  }

  return result;
}

/**
 * Q4_K Block Structure (256 elements per super-block):
 *
 * Super-block contains 8 sub-blocks of 32 elements each.
 *
 * Layout:
 * - 2 bytes: FP16 d (main scale)
 * - 2 bytes: FP16 dmin (main min scale)
 * - 6 bytes: 8 x 6-bit scales (packed)
 * - 6 bytes: 8 x 6-bit mins (packed)
 * - 128 bytes: quantized data (4 bits per element, 32 elements per sub-block)
 *
 * Total: 144 bytes per 256 elements = 4.5 bits per element
 *
 * Dequantization formula for each sub-block:
 * weight = d * scale * quant + dmin * min
 */
export function dequantizeQ4_K_accurate(buffer: Buffer, count: number): Float32Array {
  const result = new Float32Array(count);
  const QK_K = 256; // Super-block size
  const numBlocks = Math.ceil(count / QK_K);

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
      const scale = d * (scales[subBlock] / 63.0); // Normalize 6-bit to [0,1]
      const min = dmin * (mins[subBlock] / 63.0);

      // Read 32 elements (16 bytes, 4 bits per element)
      for (let i = 0; i < 32 && resultOffset < count; i++) {
        const byteIndex = bufferOffset + Math.floor(i / 2);
        const nibble = (i % 2 === 0)
          ? (buffer[byteIndex] & 0x0F)
          : ((buffer[byteIndex] >> 4) & 0x0F);

        // Dequantize: w = d * scale * q + dmin * min
        // Quant is [0,15], normalize to [0,1] range
        const q = nibble / 15.0;
        result[resultOffset++] = scale * (q * 15.0 - 8.0) + min;
      }

      bufferOffset += 16; // 16 bytes per 32 elements
    }
  }

  return result;
}

/**
 * Q6_K Block Structure (256 elements per super-block):
 *
 * Layout:
 * - 2 bytes: FP16 d (scale)
 * - 128 bytes: high 4 bits of quantized values (int8)
 * - 64 bytes: low 2 bits of quantized values (packed)
 * - 16 bytes: 16 x int8 scales for sub-blocks
 *
 * Total: 210 bytes per 256 elements = 6.56 bits per element
 *
 * Each value is 6 bits: 4 high bits + 2 low bits
 * Dequantization: weight = d * scale * (high * 4 + low - 32)
 */
export function dequantizeQ6_K_accurate(buffer: Buffer, count: number): Float32Array {
  const result = new Float32Array(count);
  const QK_K = 256; // Super-block size
  const numBlocks = Math.ceil(count / QK_K);

  let resultOffset = 0;
  let bufferOffset = 0;

  for (let block = 0; block < numBlocks; block++) {
    // Read main scale
    const d = readFloat16(buffer, bufferOffset);
    bufferOffset += 2;

    // Read 128 bytes of high 4 bits (stored as int8)
    const highBits = new Int8Array(128);
    for (let i = 0; i < 128; i++) {
      highBits[i] = buffer.readInt8(bufferOffset + i);
    }
    bufferOffset += 128;

    // Read 64 bytes of low 2 bits (packed 4 per byte)
    const lowBitsBytes = Buffer.from(buffer.slice(bufferOffset, bufferOffset + 64));
    bufferOffset += 64;

    // Read 16 x int8 scales for sub-blocks (16 sub-blocks of 16 elements each)
    const scales = new Int8Array(16);
    for (let i = 0; i < 16; i++) {
      scales[i] = buffer.readInt8(bufferOffset + i);
    }
    bufferOffset += 16;

    // Dequantize 256 elements
    for (let i = 0; i < QK_K && resultOffset < count; i++) {
      // Get high 4 bits
      const high = highBits[i >> 1]; // Each byte contains 2 high-bit groups

      // Get low 2 bits (4 values per byte)
      const lowByteIndex = Math.floor(i / 4);
      const lowBitShift = (i % 4) * 2;
      const low = (lowBitsBytes[lowByteIndex] >> lowBitShift) & 0x03;

      // Combine to get 6-bit value
      const q = (high & 0x0F) * 4 + low; // Range [0, 63]

      // Get sub-block scale (16 sub-blocks of 16 elements)
      const subBlockIndex = Math.floor(i / 16);
      const scale = scales[subBlockIndex] / 127.0; // Normalize int8 scale

      // Dequantize: shift to signed range [-32, 31]
      result[resultOffset++] = d * scale * (q - 32);
    }
  }

  return result;
}

/**
 * Alternative Q4_K implementation with simpler structure
 * This matches the actual GGUF Q4_K layout more closely
 */
export function dequantizeQ4_K_v2(buffer: Buffer, count: number): Float32Array {
  const result = new Float32Array(count);
  const QK_K = 256;
  const numBlocks = Math.ceil(count / QK_K);

  let resultOffset = 0;
  let bufferOffset = 0;

  for (let block = 0; block < numBlocks; block++) {
    // Q4_K_small: simpler layout
    // 2 bytes: d (fp16)
    // 2 bytes: dmin (fp16)
    // 8 bytes: scales (8 x 8-bit values)
    // 8 bytes: mins (8 x 8-bit values)
    // 128 bytes: quants (4 bits per value)

    const d = readFloat16(buffer, bufferOffset);
    const dmin = readFloat16(buffer, bufferOffset + 2);
    bufferOffset += 4;

    // Read 8 scales (1 per sub-block)
    const scales: number[] = [];
    for (let i = 0; i < 8; i++) {
      scales.push(buffer.readUInt8(bufferOffset + i) / 255.0);
    }
    bufferOffset += 8;

    // Read 8 mins (1 per sub-block)
    const mins: number[] = [];
    for (let i = 0; i < 8; i++) {
      mins.push(buffer.readUInt8(bufferOffset + i) / 255.0);
    }
    bufferOffset += 8;

    // Process 8 sub-blocks of 32 elements each
    for (let subBlock = 0; subBlock < 8 && resultOffset < count; subBlock++) {
      const scale = d * scales[subBlock];
      const min = dmin * mins[subBlock];

      // Read 32 x 4-bit values (16 bytes)
      for (let i = 0; i < 32 && resultOffset < count; i++) {
        const byteIndex = bufferOffset + Math.floor(i / 2);
        const nibble = (i % 2 === 0)
          ? (buffer[byteIndex] & 0x0F)
          : ((buffer[byteIndex] >> 4) & 0x0F);

        // Dequantize
        result[resultOffset++] = scale * (nibble - 8) + min;
      }

      bufferOffset += 16;
    }
  }

  return result;
}
