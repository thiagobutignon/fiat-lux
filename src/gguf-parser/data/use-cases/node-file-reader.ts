/**
 * Node.js File Reader Implementation
 * Supports large files (>2GB) using file handles and chunk reading
 */

import { open, stat, access } from 'fs/promises';
import { constants } from 'fs';
import { IFileReader } from '../protocols/file-reader';

export class NodeFileReader implements IFileReader {
  /**
   * Read entire file into Buffer
   * For files >2GB, uses chunked reading to avoid Node.js limitations
   */
  async readFile(path: string): Promise<Buffer> {
    const fileSize = await this.getFileSize(path);
    const MAX_BUFFER_SIZE = 2 ** 30; // 1GB chunks for safety

    // For small files (<1GB), use direct read
    if (fileSize < MAX_BUFFER_SIZE) {
      const fileHandle = await open(path, 'r');
      try {
        const buffer = Buffer.allocUnsafe(Number(fileSize));
        await fileHandle.read(buffer, 0, Number(fileSize), 0);
        return buffer;
      } finally {
        await fileHandle.close();
      }
    }

    // For large files, read in chunks
    const buffer = Buffer.allocUnsafe(Number(fileSize));
    const fileHandle = await open(path, 'r');

    try {
      let totalBytesRead = 0;

      while (totalBytesRead < Number(fileSize)) {
        const remaining = Number(fileSize) - totalBytesRead;
        const chunkSize = Math.min(MAX_BUFFER_SIZE, remaining);

        const { bytesRead } = await fileHandle.read(
          buffer,
          totalBytesRead,
          chunkSize,
          totalBytesRead
        );

        if (bytesRead === 0) {
          throw new Error('Unexpected end of file');
        }

        totalBytesRead += bytesRead;
      }

      return buffer;
    } finally {
      await fileHandle.close();
    }
  }

  async exists(path: string): Promise<boolean> {
    try {
      await access(path, constants.F_OK);
      return true;
    } catch {
      return false;
    }
  }

  async getFileSize(path: string): Promise<bigint> {
    const stats = await stat(path);
    return BigInt(stats.size);
  }
}
