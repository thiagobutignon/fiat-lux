/**
 * Test Runner for SliceRewriter
 * TDD: RED phase - write tests first
 */

import { SliceRewriter } from '../core/slice-rewriter';
import { Observability, LogLevel } from '../core/observability';
import fs from 'fs';
import path from 'path';
import os from 'os';

let passed = 0;
let failed = 0;

function test(name: string, fn: () => void | Promise<void>) {
  const result = fn();
  if (result instanceof Promise) {
    result
      .then(() => {
        console.log(`✅ ${name}`);
        passed++;
      })
      .catch((error: any) => {
        console.log(`❌ ${name}`);
        console.log(`   ${error.message}`);
        failed++;
      });
  } else {
    try {
      console.log(`✅ ${name}`);
      passed++;
    } catch (error: any) {
      console.log(`❌ ${name}`);
      console.log(`   ${error.message}`);
      failed++;
    }
  }
}

function assert(condition: boolean, message: string) {
  if (!condition) {
    throw new Error(message);
  }
}

// Setup test directories
const testDir = path.join(os.tmpdir(), 'slice-rewriter-tests');
const slicesDir = path.join(testDir, 'slices');
const backupDir = path.join(testDir, 'backups');

function setupTestDirs() {
  // Clean up if exists
  if (fs.existsSync(testDir)) {
    fs.rmSync(testDir, { recursive: true });
  }

  // Create fresh directories
  fs.mkdirSync(testDir, { recursive: true });
  fs.mkdirSync(slicesDir, { recursive: true });
  fs.mkdirSync(backupDir, { recursive: true });
}

function cleanupTestDirs() {
  if (fs.existsSync(testDir)) {
    fs.rmSync(testDir, { recursive: true });
  }
}

console.log('🧪 Testing SliceRewriter\n');

// Setup
setupTestDirs();
const obs = new Observability(LogLevel.DEBUG);
const rewriter = new SliceRewriter(slicesDir, backupDir, obs);

// Test 1: Create new slice
test('should create new slice with valid YAML', async () => {
  const content = `
id: test-slice
title: Test Slice
description: A test slice
concepts:
  - test_concept_1
  - test_concept_2
content: |
  This is test content.
`;

  await rewriter.createSlice('test-slice', content);

  const filePath = path.join(slicesDir, 'test-slice.yml');
  assert(fs.existsSync(filePath), 'Slice file should exist');

  const written = fs.readFileSync(filePath, 'utf-8');
  assert(written.includes('test_concept_1'), 'Content should be written');
});

// Test 2: Reject invalid YAML
test('should reject invalid YAML', async () => {
  const invalidYaml = `
invalid: yaml: syntax: here
  - broken
`;

  try {
    await rewriter.createSlice('invalid-slice', invalidYaml);
    throw new Error('Should have thrown error for invalid YAML');
  } catch (error: any) {
    assert(
      error.message.includes('Invalid YAML'),
      'Should throw invalid YAML error'
    );
  }
});

// Test 3: Prevent duplicate slice IDs
test('should prevent duplicate slice IDs', async () => {
  const content = `
id: duplicate-test
title: Test
content: test
`;

  await rewriter.createSlice('duplicate-test', content);

  try {
    await rewriter.createSlice('duplicate-test', content);
    throw new Error('Should have thrown error for duplicate ID');
  } catch (error: any) {
    assert(
      error.message.includes('already exists'),
      'Should throw duplicate error'
    );
  }
});

// Test 4: Update existing slice
test('should update existing slice', async () => {
  const original = `
id: update-test
title: Original
content: original content
`;

  const updated = `
id: update-test
title: Updated
content: updated content
`;

  await rewriter.createSlice('update-test', original);
  await rewriter.updateSlice('update-test', updated);

  const filePath = path.join(slicesDir, 'update-test.yml');
  const written = fs.readFileSync(filePath, 'utf-8');

  assert(written.includes('Updated'), 'Should have updated title');
  assert(written.includes('updated content'), 'Should have updated content');
});

// Test 5: Backup before update
test('should create backup before updating', async () => {
  const original = `
id: backup-test
title: Original
content: original
`;

  const updated = `
id: backup-test
title: Updated
content: updated
`;

  await rewriter.createSlice('backup-test', original);
  await rewriter.updateSlice('backup-test', updated);

  // Check backup exists
  const backups = fs.readdirSync(backupDir);
  const backupFile = backups.find((f) => f.startsWith('backup-test_'));

  assert(backupFile !== undefined, 'Backup file should exist');

  const backupContent = fs.readFileSync(
    path.join(backupDir, backupFile!),
    'utf-8'
  );
  assert(backupContent.includes('Original'), 'Backup should have original content');
});

// Test 6: Atomic write (temp + rename)
test('should use atomic write pattern', async () => {
  const content = `
id: atomic-test
title: Test
content: test
`;

  await rewriter.createSlice('atomic-test', content);

  // File should exist and be complete (not partial)
  const filePath = path.join(slicesDir, 'atomic-test.yml');
  assert(fs.existsSync(filePath), 'File should exist');

  const written = fs.readFileSync(filePath, 'utf-8');
  assert(written.length > 0, 'File should not be empty');
  assert(written.includes('atomic-test'), 'File should have content');
});

// Test 7: Generate diff
test('should generate unified diff', async () => {
  const oldContent = `line 1
line 2
line 3`;

  const newContent = `line 1
line 2 modified
line 3
line 4`;

  const diff = rewriter.diff(oldContent, newContent);

  assert(diff.includes('line 2'), 'Diff should show changed line');
  assert(diff.includes('modified'), 'Diff should show modification');
});

// Test 8: Restore from backup
test('should restore from backup', async () => {
  const original = `
id: restore-test
title: Original
content: original
`;

  const modified = `
id: restore-test
title: Modified
content: modified
`;

  // Create and modify
  await rewriter.createSlice('restore-test', original);
  await rewriter.updateSlice('restore-test', modified);

  // Find backup
  const backups = fs.readdirSync(backupDir);
  const backupFile = backups.find((f) => f.startsWith('restore-test_'));
  assert(backupFile !== undefined, 'Backup should exist');

  const backupPath = path.join(backupDir, backupFile!);

  // Restore
  await rewriter.restore(backupPath);

  // Verify restoration
  const filePath = path.join(slicesDir, 'restore-test.yml');
  const restored = fs.readFileSync(filePath, 'utf-8');

  assert(restored.includes('Original'), 'Should restore original content');
  assert(!restored.includes('Modified'), 'Should not have modified content');
});

// Test 9: Error on missing backup
test('should error on missing backup', async () => {
  const fakePath = path.join(backupDir, 'nonexistent_backup.yml');

  try {
    await rewriter.restore(fakePath);
    throw new Error('Should have thrown error for missing backup');
  } catch (error: any) {
    assert(
      error.message.includes('not found') || error.message.includes('ENOENT'),
      'Should throw not found error'
    );
  }
});

// Test 10: List backups for slice
test('should list backups for a slice', async () => {
  const content = `
id: list-test
title: Test
content: test
`;

  // Create and update multiple times
  await rewriter.createSlice('list-test', content);
  await rewriter.updateSlice('list-test', content.replace('Test', 'Test 2'));
  await rewriter.updateSlice('list-test', content.replace('Test', 'Test 3'));

  const backups = rewriter.listBackups('list-test');

  assert(backups.length >= 2, `Should have at least 2 backups, got ${backups.length}`);
  assert(
    backups[0].includes('list-test'),
    'Backup filename should include slice ID'
  );
});

// Wait for async tests to complete
setTimeout(() => {
  // Cleanup
  cleanupTestDirs();

  // Summary
  console.log('\n' + '='.repeat(70));
  console.log(`Total: ${passed + failed}`);
  console.log(`✅ Passed: ${passed}`);
  console.log(`❌ Failed: ${failed}`);
  console.log('='.repeat(70));

  if (failed > 0) {
    process.exit(1);
  }
}, 1000);
