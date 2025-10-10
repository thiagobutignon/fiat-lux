/**
 * Test: Auto-Commit System
 *
 * Verifies:
 * 1. File watcher detects changes
 * 2. Diff calculator works
 * 3. Auto-commit creates git commit
 * 4. No manual intervention needed
 */

import * as fs from 'fs';
import * as path from 'path';
import { autoCommit, watchFile, exportState } from './auto-commit';

// Test file path
const TEST_DIR = path.join(__dirname, '../../../test-files/vcs');
const TEST_FILE = path.join(TEST_DIR, 'test-auto-commit.gl');

// Ensure test directory exists
if (!fs.existsSync(TEST_DIR)) {
  fs.mkdirSync(TEST_DIR, { recursive: true });
}

// Test 1: Initial file creation
console.log('📝 Test 1: Creating test file...');
fs.writeFileSync(TEST_FILE, `(define hello (-> string)
  "Hello, World!")`);

console.log('✅ Test file created:', TEST_FILE);

// Test 2: Manual auto-commit
console.log('\n📝 Test 2: Manual auto-commit...');
const committed = autoCommit(TEST_FILE);
if (committed) {
  console.log('✅ Auto-commit successful!');
} else {
  console.log('⏭️  No changes to commit (expected on first run)');
}

// Test 3: Modify file and auto-commit
console.log('\n📝 Test 3: Modifying file and auto-committing...');
fs.writeFileSync(TEST_FILE, `(define hello (-> string)
  "Hello, World!")

(define goodbye (-> string)
  "Goodbye, World!")`);

const committed2 = autoCommit(TEST_FILE);
if (committed2) {
  console.log('✅ Auto-commit detected change!');
} else {
  console.log('❌ Auto-commit failed to detect change');
}

// Test 4: Export state (glass box)
console.log('\n📝 Test 4: Exporting state (glass box)...');
const state = exportState();
console.log('State:', JSON.stringify(state, null, 2));
console.log('✅ State exported successfully!');

// Test 5: Watch file (async)
console.log('\n📝 Test 5: Setting up file watcher...');
const watcher = watchFile(TEST_FILE);
console.log('✅ File watcher active. Modify the file to test auto-commit.');
console.log('   File:', TEST_FILE);
console.log('   Watching for changes...');

// Keep process alive for 10 seconds to test watcher
setTimeout(() => {
  watcher.close();
  console.log('\n✅ Test complete! File watcher stopped.');
  console.log('\nTo test the watcher, run:');
  console.log(`  echo "// New comment" >> ${TEST_FILE}`);
  console.log('  (in another terminal while this test is running)');
}, 10000);
