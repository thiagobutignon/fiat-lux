/**
 * SliceRewriter - Safe slice file operations
 *
 * Handles creating, updating, and restoring knowledge slices with:
 * - Atomic writes (temp + rename)
 * - Automatic backups
 * - YAML validation
 * - Rollback capability
 * - Diff generation
 */

import fs from 'fs';
import path from 'path';
import yaml from 'yaml';
import { Observability } from './observability';

// ============================================================================
// Types
// ============================================================================

export interface SliceMetadata {
  id: string;
  title?: string;
  description?: string;
  concepts?: string[];
  domain?: string;
}

// ============================================================================
// SliceRewriter Class
// ============================================================================

export class SliceRewriter {
  constructor(
    private slicesDir: string,
    private backupDir: string,
    private observability: Observability
  ) {
    // Ensure directories exist
    this.ensureDir(slicesDir);
    this.ensureDir(backupDir);
  }

  /**
   * Create new slice
   */
  async createSlice(sliceId: string, content: string): Promise<void> {
    const span = this.observability.startSpan('slice_create');
    span.setTag('slice_id', sliceId);

    try {
      // 1. Validate YAML
      this.validateYAML(content);

      // 2. Check for conflicts
      const filePath = this.getSlicePath(sliceId);
      if (fs.existsSync(filePath)) {
        throw new Error(`Slice ${sliceId} already exists`);
      }

      // 3. Write file (atomic)
      await this.atomicWrite(filePath, content);

      // 4. Log creation
      this.observability.log('info', 'slice_created', {
        slice_id: sliceId,
        path: filePath,
      });

      span.setTag('success', true);
    } catch (error) {
      span.setTag('success', false);
      span.setTag('error', (error as Error).message);
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Update existing slice
   */
  async updateSlice(sliceId: string, newContent: string): Promise<void> {
    const span = this.observability.startSpan('slice_update');
    span.setTag('slice_id', sliceId);

    try {
      // 1. Validate YAML
      this.validateYAML(newContent);

      // 2. Backup current version
      const filePath = this.getSlicePath(sliceId);
      if (!fs.existsSync(filePath)) {
        throw new Error(`Slice ${sliceId} does not exist`);
      }

      const backupPath = await this.backup(sliceId);
      span.setTag('backup_path', backupPath);

      // 3. Atomic write
      await this.atomicWrite(filePath, newContent);

      // 4. Log update
      this.observability.log('info', 'slice_updated', {
        slice_id: sliceId,
        path: filePath,
        backup_path: backupPath,
      });

      span.setTag('success', true);
    } catch (error) {
      span.setTag('success', false);
      span.setTag('error', (error as Error).message);
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Backup slice
   */
  async backup(sliceId: string): Promise<string> {
    const span = this.observability.startSpan('slice_backup');
    span.setTag('slice_id', sliceId);

    try {
      const sourcePath = this.getSlicePath(sliceId);
      if (!fs.existsSync(sourcePath)) {
        throw new Error(`Slice ${sliceId} does not exist`);
      }

      // Generate timestamped backup filename
      const timestamp = Date.now();
      const backupFilename = `${sliceId}_${timestamp}.yml`;
      const backupPath = path.join(this.backupDir, backupFilename);

      // Copy to backup
      const content = fs.readFileSync(sourcePath, 'utf-8');
      fs.writeFileSync(backupPath, content, 'utf-8');

      this.observability.log('info', 'slice_backed_up', {
        slice_id: sliceId,
        backup_path: backupPath,
      });

      span.setTag('backup_path', backupPath);
      span.setTag('success', true);

      return backupPath;
    } catch (error) {
      span.setTag('success', false);
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Restore from backup
   */
  async restore(backupPath: string): Promise<void> {
    const span = this.observability.startSpan('slice_restore');
    span.setTag('backup_path', backupPath);

    try {
      // 1. Validate backup exists
      if (!fs.existsSync(backupPath)) {
        throw new Error(`Backup not found: ${backupPath}`);
      }

      // 2. Extract slice ID from backup filename
      const filename = path.basename(backupPath);
      const sliceId = filename.split('_')[0];

      // 3. Copy back to slices dir
      const targetPath = this.getSlicePath(sliceId);
      const content = fs.readFileSync(backupPath, 'utf-8');
      await this.atomicWrite(targetPath, content);

      // 4. Log restoration
      this.observability.log('info', 'slice_restored', {
        slice_id: sliceId,
        backup_path: backupPath,
        target_path: targetPath,
      });

      span.setTag('slice_id', sliceId);
      span.setTag('success', true);
    } catch (error) {
      span.setTag('success', false);
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Generate diff between two contents
   */
  diff(oldContent: string, newContent: string): string {
    const oldLines = oldContent.split('\n');
    const newLines = newContent.split('\n');

    const diffLines: string[] = [];
    const maxLen = Math.max(oldLines.length, newLines.length);

    for (let i = 0; i < maxLen; i++) {
      const oldLine = oldLines[i] || '';
      const newLine = newLines[i] || '';

      if (oldLine !== newLine) {
        if (oldLine) {
          diffLines.push(`- ${oldLine}`);
        }
        if (newLine) {
          diffLines.push(`+ ${newLine}`);
        }
      } else {
        diffLines.push(`  ${oldLine}`);
      }
    }

    return diffLines.join('\n');
  }

  /**
   * List backups for a slice
   */
  listBackups(sliceId: string): string[] {
    const files = fs.readdirSync(this.backupDir);

    return files
      .filter((f) => f.startsWith(`${sliceId}_`) && f.endsWith('.yml'))
      .map((f) => path.join(this.backupDir, f))
      .sort()
      .reverse(); // Most recent first
  }

  /**
   * Get slice file path
   */
  private getSlicePath(sliceId: string): string {
    return path.join(this.slicesDir, `${sliceId}.yml`);
  }

  /**
   * Validate YAML syntax
   */
  private validateYAML(content: string): void {
    try {
      yaml.parse(content);
    } catch (error) {
      throw new Error(`Invalid YAML: ${(error as Error).message}`);
    }
  }

  /**
   * Atomic write (write to temp file, then rename)
   */
  private async atomicWrite(filePath: string, content: string): Promise<void> {
    const tempPath = `${filePath}.tmp`;

    try {
      // Write to temp file
      fs.writeFileSync(tempPath, content, 'utf-8');

      // Atomic rename
      fs.renameSync(tempPath, filePath);
    } catch (error) {
      // Cleanup temp file if it exists
      if (fs.existsSync(tempPath)) {
        fs.unlinkSync(tempPath);
      }
      throw error;
    }
  }

  /**
   * Ensure directory exists
   */
  private ensureDir(dir: string): void {
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
  }

  /**
   * Read slice content
   */
  async readSlice(sliceId: string): Promise<string> {
    const filePath = this.getSlicePath(sliceId);

    if (!fs.existsSync(filePath)) {
      throw new Error(`Slice ${sliceId} does not exist`);
    }

    return fs.readFileSync(filePath, 'utf-8');
  }

  /**
   * Check if slice exists
   */
  exists(sliceId: string): boolean {
    return fs.existsSync(this.getSlicePath(sliceId));
  }

  /**
   * Delete slice (with backup)
   */
  async deleteSlice(sliceId: string): Promise<void> {
    const span = this.observability.startSpan('slice_delete');
    span.setTag('slice_id', sliceId);

    try {
      // Backup before deleting
      const backupPath = await this.backup(sliceId);

      // Delete
      const filePath = this.getSlicePath(sliceId);
      fs.unlinkSync(filePath);

      this.observability.log('info', 'slice_deleted', {
        slice_id: sliceId,
        backup_path: backupPath,
      });

      span.setTag('success', true);
    } catch (error) {
      span.setTag('success', false);
      throw error;
    } finally {
      span.end();
    }
  }
}

/**
 * Create a new SliceRewriter instance
 */
export function createSliceRewriter(
  slicesDir: string,
  backupDir: string,
  observability: Observability
): SliceRewriter {
  return new SliceRewriter(slicesDir, backupDir, observability);
}
