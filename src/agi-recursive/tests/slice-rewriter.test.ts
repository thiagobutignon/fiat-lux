/**
 * @file slice-rewriter.test.ts
 * Tests for SliceRewriter - Safe slice file operations
 *
 * Key capabilities tested:
 * - Atomic writes (temp + rename)
 * - Automatic backups
 * - YAML validation
 * - Rollback capability
 * - Diff generation
 * - Observability integration
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { SliceRewriter, createSliceRewriter } from '../core/slice-rewriter';
import { Observability } from '../core/observability';
import fs from 'fs';
import path from 'path';
import os from 'os';

describe('SliceRewriter', () => {
  let rewriter: SliceRewriter;
  let slicesDir: string;
  let backupDir: string;
  let observability: Observability;

  beforeEach(() => {
    // Create temporary directories
    slicesDir = path.join(os.tmpdir(), `test-slices-${Date.now()}`);
    backupDir = path.join(os.tmpdir(), `test-backups-${Date.now()}`);

    // Mock observability
    observability = {
      startSpan: vi.fn(() => ({
        setTag: vi.fn(),
        end: vi.fn(),
      })),
      log: vi.fn(),
    } as any;

    rewriter = new SliceRewriter(slicesDir, backupDir, observability);
  });

  afterEach(() => {
    // Cleanup test directories
    if (fs.existsSync(slicesDir)) {
      fs.rmSync(slicesDir, { recursive: true, force: true });
    }
    if (fs.existsSync(backupDir)) {
      fs.rmSync(backupDir, { recursive: true, force: true });
    }
  });

  describe('Initialization', () => {
    it('should create slices directory on initialization', () => {
      expect(fs.existsSync(slicesDir)).toBe(true);
    });

    it('should create backup directory on initialization', () => {
      expect(fs.existsSync(backupDir)).toBe(true);
    });

    it('should handle existing directories', () => {
      // Create another instance with same directories
      const rewriter2 = new SliceRewriter(slicesDir, backupDir, observability);
      expect(rewriter2).toBeDefined();
    });
  });

  describe('Create Slice', () => {
    it('should create a new slice', async () => {
      const content = `
metadata:
  id: test-slice
  title: Test Slice
  concepts: [test]
knowledge: Test knowledge
`;

      await rewriter.createSlice('test-slice', content);

      expect(rewriter.exists('test-slice')).toBe(true);
    });

    it('should validate YAML before creating', async () => {
      const invalidYAML = 'this: is: [invalid';

      await expect(rewriter.createSlice('invalid', invalidYAML)).rejects.toThrow(
        'Invalid YAML'
      );
    });

    it('should throw error if slice already exists', async () => {
      const content = 'metadata:\n  id: test\nknowledge: test';

      await rewriter.createSlice('test', content);

      await expect(rewriter.createSlice('test', content)).rejects.toThrow(
        'Slice test already exists'
      );
    });

    it('should use atomic write for slice creation', async () => {
      const content = 'metadata:\n  id: test\nknowledge: test';
      const slicePath = path.join(slicesDir, 'test.yml');
      const tempPath = `${slicePath}.tmp`;

      await rewriter.createSlice('test', content);

      // Temp file should be cleaned up
      expect(fs.existsSync(tempPath)).toBe(false);
      // Final file should exist
      expect(fs.existsSync(slicePath)).toBe(true);
    });

    it('should log slice creation', async () => {
      const content = 'metadata:\n  id: test\nknowledge: test';

      await rewriter.createSlice('test', content);

      expect(observability.log).toHaveBeenCalledWith('info', 'slice_created', expect.any(Object));
    });
  });

  describe('Update Slice', () => {
    beforeEach(async () => {
      // Create initial slice
      const content = 'metadata:\n  id: test\nknowledge: original';
      await rewriter.createSlice('test', content);
    });

    it('should update existing slice', async () => {
      const newContent = 'metadata:\n  id: test\nknowledge: updated';

      await rewriter.updateSlice('test', newContent);

      const content = await rewriter.readSlice('test');
      expect(content).toBe(newContent);
    });

    it('should create backup before updating', async () => {
      const newContent = 'metadata:\n  id: test\nknowledge: updated';

      await rewriter.updateSlice('test', newContent);

      const backups = rewriter.listBackups('test');
      expect(backups.length).toBeGreaterThan(0);
    });

    it('should throw error if slice does not exist', async () => {
      const content = 'metadata:\n  id: nonexistent\nknowledge: test';

      await expect(rewriter.updateSlice('nonexistent', content)).rejects.toThrow(
        'Slice nonexistent does not exist'
      );
    });

    it('should validate YAML before updating', async () => {
      const invalidYAML = 'this: is: [invalid';

      await expect(rewriter.updateSlice('test', invalidYAML)).rejects.toThrow('Invalid YAML');
    });

    it('should use atomic write for updates', async () => {
      const newContent = 'metadata:\n  id: test\nknowledge: updated';
      const slicePath = path.join(slicesDir, 'test.yml');
      const tempPath = `${slicePath}.tmp`;

      await rewriter.updateSlice('test', newContent);

      // Temp file should be cleaned up
      expect(fs.existsSync(tempPath)).toBe(false);
      // Final file should have new content
      expect(fs.existsSync(slicePath)).toBe(true);
    });

    it('should log slice update', async () => {
      const newContent = 'metadata:\n  id: test\nknowledge: updated';

      await rewriter.updateSlice('test', newContent);

      expect(observability.log).toHaveBeenCalledWith('info', 'slice_updated', expect.any(Object));
    });
  });

  describe('Backup', () => {
    beforeEach(async () => {
      const content = 'metadata:\n  id: test\nknowledge: original';
      await rewriter.createSlice('test', content);
    });

    it('should create backup of slice', async () => {
      const backupPath = await rewriter.backup('test');

      expect(fs.existsSync(backupPath)).toBe(true);
    });

    it('should use timestamped backup filename', async () => {
      const backupPath = await rewriter.backup('test');
      const filename = path.basename(backupPath);

      expect(filename).toMatch(/^test_\d+\.yml$/);
    });

    it('should preserve slice content in backup', async () => {
      const originalContent = await rewriter.readSlice('test');
      const backupPath = await rewriter.backup('test');
      const backupContent = fs.readFileSync(backupPath, 'utf-8');

      expect(backupContent).toBe(originalContent);
    });

    it('should throw error if slice does not exist', async () => {
      await expect(rewriter.backup('nonexistent')).rejects.toThrow(
        'Slice nonexistent does not exist'
      );
    });

    it('should log backup creation', async () => {
      await rewriter.backup('test');

      expect(observability.log).toHaveBeenCalledWith(
        'info',
        'slice_backed_up',
        expect.any(Object)
      );
    });
  });

  describe('Restore', () => {
    let backupPath: string;

    beforeEach(async () => {
      // Create and backup a slice
      const content = 'metadata:\n  id: test\nknowledge: original';
      await rewriter.createSlice('test', content);
      backupPath = await rewriter.backup('test');

      // Update the slice
      const newContent = 'metadata:\n  id: test\nknowledge: modified';
      await rewriter.updateSlice('test', newContent);
    });

    it('should restore slice from backup', async () => {
      await rewriter.restore(backupPath);

      const content = await rewriter.readSlice('test');
      expect(content).toContain('original');
      expect(content).not.toContain('modified');
    });

    it('should throw error if backup does not exist', async () => {
      const fakeBackupPath = path.join(backupDir, 'nonexistent_123.yml');

      await expect(rewriter.restore(fakeBackupPath)).rejects.toThrow('Backup not found');
    });

    it('should extract slice ID from backup filename', async () => {
      await rewriter.restore(backupPath);

      expect(rewriter.exists('test')).toBe(true);
    });

    it('should use atomic write for restoration', async () => {
      const slicePath = path.join(slicesDir, 'test.yml');
      const tempPath = `${slicePath}.tmp`;

      await rewriter.restore(backupPath);

      expect(fs.existsSync(tempPath)).toBe(false);
      expect(fs.existsSync(slicePath)).toBe(true);
    });

    it('should log restoration', async () => {
      await rewriter.restore(backupPath);

      expect(observability.log).toHaveBeenCalledWith('info', 'slice_restored', expect.any(Object));
    });
  });

  describe('Diff Generation', () => {
    it('should generate diff between two contents', () => {
      const oldContent = 'line1\nline2\nline3';
      const newContent = 'line1\nmodified\nline3\nline4';

      const diff = rewriter.diff(oldContent, newContent);

      expect(diff).toContain('- line2');
      expect(diff).toContain('+ modified');
      expect(diff).toContain('+ line4');
      expect(diff).toContain('  line1');
      expect(diff).toContain('  line3');
    });

    it('should handle identical contents', () => {
      const content = 'line1\nline2\nline3';

      const diff = rewriter.diff(content, content);

      expect(diff).toContain('  line1');
      expect(diff).toContain('  line2');
      expect(diff).toContain('  line3');
      expect(diff).not.toContain('-');
      expect(diff).not.toContain('+');
    });

    it('should handle empty old content', () => {
      const newContent = 'line1\nline2';

      const diff = rewriter.diff('', newContent);

      expect(diff).toContain('+ line1');
      expect(diff).toContain('+ line2');
    });

    it('should handle empty new content', () => {
      const oldContent = 'line1\nline2';

      const diff = rewriter.diff(oldContent, '');

      expect(diff).toContain('- line1');
      expect(diff).toContain('- line2');
    });
  });

  describe('List Backups', () => {
    beforeEach(async () => {
      const content = 'metadata:\n  id: test\nknowledge: original';
      await rewriter.createSlice('test', content);
    });

    it('should list backups for a slice', async () => {
      await rewriter.backup('test');
      // Wait a bit to ensure different timestamps
      await new Promise((resolve) => setTimeout(resolve, 10));
      await rewriter.backup('test');

      const backups = rewriter.listBackups('test');

      expect(backups.length).toBe(2);
    });

    it('should return backups in reverse chronological order', async () => {
      // Wait a bit between backups to ensure different timestamps
      await rewriter.backup('test');
      await new Promise((resolve) => setTimeout(resolve, 10));
      await rewriter.backup('test');

      const backups = rewriter.listBackups('test');

      // Most recent should be first
      expect(backups[0]).toContain(path.basename(backups[0]));
      expect(backups[1]).toContain(path.basename(backups[1]));

      // Extract timestamps
      const timestamp1 = parseInt(path.basename(backups[0]).split('_')[1]);
      const timestamp2 = parseInt(path.basename(backups[1]).split('_')[1]);

      expect(timestamp1).toBeGreaterThan(timestamp2);
    });

    it('should return empty array if no backups exist', () => {
      const backups = rewriter.listBackups('nonexistent');

      expect(backups).toEqual([]);
    });

    it('should only list backups for specified slice', async () => {
      const content1 = 'metadata:\n  id: slice1\nknowledge: test';
      const content2 = 'metadata:\n  id: slice2\nknowledge: test';

      await rewriter.createSlice('slice1', content1);
      await rewriter.createSlice('slice2', content2);

      await rewriter.backup('slice1');
      await rewriter.backup('slice2');

      const backups1 = rewriter.listBackups('slice1');
      const backups2 = rewriter.listBackups('slice2');

      expect(backups1.length).toBe(1);
      expect(backups2.length).toBe(1);
      expect(backups1[0]).toContain('slice1');
      expect(backups2[0]).toContain('slice2');
    });
  });

  describe('Read Slice', () => {
    beforeEach(async () => {
      const content = 'metadata:\n  id: test\nknowledge: content';
      await rewriter.createSlice('test', content);
    });

    it('should read slice content', async () => {
      const content = await rewriter.readSlice('test');

      expect(content).toContain('metadata:');
      expect(content).toContain('knowledge: content');
    });

    it('should throw error if slice does not exist', async () => {
      await expect(rewriter.readSlice('nonexistent')).rejects.toThrow(
        'Slice nonexistent does not exist'
      );
    });
  });

  describe('Exists Check', () => {
    it('should return true if slice exists', async () => {
      const content = 'metadata:\n  id: test\nknowledge: test';
      await rewriter.createSlice('test', content);

      expect(rewriter.exists('test')).toBe(true);
    });

    it('should return false if slice does not exist', () => {
      expect(rewriter.exists('nonexistent')).toBe(false);
    });
  });

  describe('Delete Slice', () => {
    beforeEach(async () => {
      const content = 'metadata:\n  id: test\nknowledge: content';
      await rewriter.createSlice('test', content);
    });

    it('should delete slice', async () => {
      await rewriter.deleteSlice('test');

      expect(rewriter.exists('test')).toBe(false);
    });

    it('should create backup before deleting', async () => {
      await rewriter.deleteSlice('test');

      const backups = rewriter.listBackups('test');
      expect(backups.length).toBeGreaterThan(0);
    });

    it('should throw error if slice does not exist', async () => {
      await expect(rewriter.deleteSlice('nonexistent')).rejects.toThrow(
        'Slice nonexistent does not exist'
      );
    });

    it('should log deletion', async () => {
      await rewriter.deleteSlice('test');

      expect(observability.log).toHaveBeenCalledWith('info', 'slice_deleted', expect.any(Object));
    });
  });

  describe('Observability Integration', () => {
    it('should start span for create operation', async () => {
      const content = 'metadata:\n  id: test\nknowledge: test';

      await rewriter.createSlice('test', content);

      expect(observability.startSpan).toHaveBeenCalledWith('slice_create');
    });

    it('should start span for update operation', async () => {
      const content = 'metadata:\n  id: test\nknowledge: original';
      await rewriter.createSlice('test', content);

      const newContent = 'metadata:\n  id: test\nknowledge: updated';
      await rewriter.updateSlice('test', newContent);

      expect(observability.startSpan).toHaveBeenCalledWith('slice_update');
    });

    it('should start span for backup operation', async () => {
      const content = 'metadata:\n  id: test\nknowledge: test';
      await rewriter.createSlice('test', content);

      await rewriter.backup('test');

      expect(observability.startSpan).toHaveBeenCalledWith('slice_backup');
    });

    it('should start span for restore operation', async () => {
      const content = 'metadata:\n  id: test\nknowledge: test';
      await rewriter.createSlice('test', content);
      const backupPath = await rewriter.backup('test');

      await rewriter.restore(backupPath);

      expect(observability.startSpan).toHaveBeenCalledWith('slice_restore');
    });

    it('should start span for delete operation', async () => {
      const content = 'metadata:\n  id: test\nknowledge: test';
      await rewriter.createSlice('test', content);

      await rewriter.deleteSlice('test');

      expect(observability.startSpan).toHaveBeenCalledWith('slice_delete');
    });
  });

  describe('Factory Function', () => {
    it('should create SliceRewriter instance', () => {
      const instance = createSliceRewriter(slicesDir, backupDir, observability);

      expect(instance).toBeInstanceOf(SliceRewriter);
    });
  });

  describe('Error Handling', () => {
    it('should cleanup temp file on write error', async () => {
      const content = 'metadata:\n  id: test\nknowledge: test';
      const slicePath = path.join(slicesDir, 'test.yml');
      const tempPath = `${slicePath}.tmp`;

      // Create read-only parent directory to cause write error
      fs.chmodSync(slicesDir, 0o444);

      try {
        await rewriter.createSlice('test', content);
      } catch (error) {
        // Expected to fail
      }

      // Restore permissions for cleanup
      fs.chmodSync(slicesDir, 0o755);

      // Temp file should be cleaned up
      expect(fs.existsSync(tempPath)).toBe(false);
    });
  });
});
