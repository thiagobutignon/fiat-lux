/**
 * File Reader Protocol
 * Abstraction for file system operations
 */

export interface IFileReader {
  /**
   * Read file and return Buffer
   */
  readFile(path: string): Promise<Buffer>;

  /**
   * Check if file exists
   */
  exists(path: string): Promise<boolean>;

  /**
   * Get file size in bytes
   */
  getFileSize(path: string): Promise<bigint>;
}
