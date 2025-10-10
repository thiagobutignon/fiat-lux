/**
 * GTest - Test Specification Format
 *
 * O(1) test framework for Grammar Language.
 *
 * Key innovations:
 * 1. Hash-based test discovery (O(1) file lookup)
 * 2. Incremental testing (only changed files)
 * 3. Grammar Language native (.gtest files)
 * 4. Zero external dependencies
 *
 * Test File Format (.gtest):
 * ```grammar
 * test "should calculate fibonacci":
 *   given: n = 5
 *   when: result = fibonacci(n)
 *   then: expect result equals 5
 *
 * test "should handle edge cases":
 *   given: n = 0
 *   when: result = fibonacci(n)
 *   then: expect result equals 0
 * ```
 */

import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';

// ============================================================================
// Types
// ============================================================================

/**
 * Test case format
 */
export interface GTestCase {
  id: string;              // SHA256 hash of test content
  name: string;            // Test description
  file: string;            // Source file path
  given?: string;          // Given clause (setup)
  when: string;            // When clause (action)
  then: string;            // Then clause (assertion)
  skip?: boolean;          // Skip this test
  only?: boolean;          // Run only this test
  timeout?: number;        // Max execution time (ms)
}

/**
 * Test suite format
 */
export interface GTestSuite {
  id: string;              // SHA256 hash of suite content
  name: string;            // Suite name (usually file name)
  file: string;            // Source file path
  tests: GTestCase[];      // All tests in suite
  setup?: string;          // Before all tests
  teardown?: string;       // After all tests
  beforeEach?: string;     // Before each test
  afterEach?: string;      // After each test
}

/**
 * Test index for O(1) lookup
 */
export interface GTestIndex {
  version: string;                    // Index format version
  timestamp: number;                  // When index was created
  suites: Map<string, GTestSuite>;    // Suite ID → Suite
  tests: Map<string, GTestCase>;      // Test ID → Test
  files: Map<string, string>;         // File path → Suite ID
  hashes: Map<string, string>;        // File path → Content hash
}

/**
 * Test result
 */
export interface GTestResult {
  testId: string;          // Test ID
  suiteName: string;       // Suite name
  testName: string;        // Test name
  passed: boolean;         // Pass/fail
  duration: number;        // Execution time (ms)
  error?: Error;           // Error if failed
  output?: string;         // Test output
}

/**
 * Test run summary
 */
export interface GTestSummary {
  total: number;           // Total tests run
  passed: number;          // Tests passed
  failed: number;          // Tests failed
  skipped: number;         // Tests skipped
  duration: number;        // Total duration (ms)
  results: GTestResult[];  // All results
}

// ============================================================================
// Test Specification Parser
// ============================================================================

/**
 * Parse .gtest file into test suite
 */
export function parseGTestFile(filePath: string): GTestSuite {
  const content = fs.readFileSync(filePath, 'utf-8');
  const fileName = path.basename(filePath, '.gtest');

  const suite: GTestSuite = {
    id: hashContent(content),
    name: fileName,
    file: filePath,
    tests: []
  };

  // Parse tests from content
  const testRegex = /test\s+"([^"]+)":\s*\n((?:  .*\n)*)/g;
  let match;

  while ((match = testRegex.exec(content)) !== null) {
    const testName = match[1];
    const testBody = match[2];

    // Parse given/when/then
    const givenMatch = /given:\s*(.+)/i.exec(testBody);
    const whenMatch = /when:\s*(.+)/i.exec(testBody);
    const thenMatch = /then:\s*(.+)/i.exec(testBody);

    const testCase: GTestCase = {
      id: hashContent(testName + testBody),
      name: testName,
      file: filePath,
      given: givenMatch ? givenMatch[1].trim() : undefined,
      when: whenMatch ? whenMatch[1].trim() : '',
      then: thenMatch ? thenMatch[1].trim() : ''
    };

    suite.tests.push(testCase);
  }

  return suite;
}

/**
 * Parse setup/teardown hooks
 */
export function parseHooks(content: string): {
  setup?: string;
  teardown?: string;
  beforeEach?: string;
  afterEach?: string;
} {
  const setupMatch = /setup:\s*\n((?:  .*\n)*)/i.exec(content);
  const teardownMatch = /teardown:\s*\n((?:  .*\n)*)/i.exec(content);
  const beforeEachMatch = /beforeEach:\s*\n((?:  .*\n)*)/i.exec(content);
  const afterEachMatch = /afterEach:\s*\n((?:  .*\n)*)/i.exec(content);

  return {
    setup: setupMatch ? setupMatch[1].trim() : undefined,
    teardown: teardownMatch ? teardownMatch[1].trim() : undefined,
    beforeEach: beforeEachMatch ? beforeEachMatch[1].trim() : undefined,
    afterEach: afterEachMatch ? afterEachMatch[1].trim() : undefined
  };
}

// ============================================================================
// Hash-Based Index (O(1) Operations)
// ============================================================================

/**
 * Create test index from directory
 * O(1) per file (via hash-based storage)
 */
export function createTestIndex(rootDir: string): GTestIndex {
  const index: GTestIndex = {
    version: '1.0.0',
    timestamp: Date.now(),
    suites: new Map(),
    tests: new Map(),
    files: new Map(),
    hashes: new Map()
  };

  // Find all .gtest files
  const testFiles = findTestFiles(rootDir);

  for (const filePath of testFiles) {
    const suite = parseGTestFile(filePath);
    const fileHash = hashFile(filePath);

    // O(1) insertions into maps
    index.suites.set(suite.id, suite);
    index.files.set(filePath, suite.id);
    index.hashes.set(filePath, fileHash);

    // Index each test
    for (const test of suite.tests) {
      index.tests.set(test.id, test);
    }
  }

  return index;
}

/**
 * Find tests that changed since last run
 * O(1) per file (hash comparison)
 */
export function findChangedTests(
  oldIndex: GTestIndex,
  newIndex: GTestIndex
): GTestCase[] {
  const changedTests: GTestCase[] = [];

  // Compare file hashes (O(1) per file)
  for (const [filePath, newHash] of newIndex.hashes) {
    const oldHash = oldIndex.hashes.get(filePath);

    // File changed or new file
    if (oldHash !== newHash) {
      const suiteId = newIndex.files.get(filePath);
      if (suiteId) {
        const suite = newIndex.suites.get(suiteId);
        if (suite) {
          changedTests.push(...suite.tests);
        }
      }
    }
  }

  return changedTests;
}

/**
 * Get test by ID (O(1) lookup)
 */
export function getTestById(index: GTestIndex, testId: string): GTestCase | null {
  return index.tests.get(testId) || null;
}

/**
 * Get suite by ID (O(1) lookup)
 */
export function getSuiteById(index: GTestIndex, suiteId: string): GTestSuite | null {
  return index.suites.get(suiteId) || null;
}

/**
 * Get suite by file path (O(1) lookup)
 */
export function getSuiteByFile(index: GTestIndex, filePath: string): GTestSuite | null {
  const suiteId = index.files.get(filePath);
  return suiteId ? index.suites.get(suiteId) || null : null;
}

// ============================================================================
// Helpers
// ============================================================================

/**
 * Hash content for O(1) identification
 */
export function hashContent(content: string): string {
  return crypto.createHash('sha256').update(content).digest('hex').substring(0, 16);
}

/**
 * Hash file for O(1) change detection
 */
export function hashFile(filePath: string): string {
  const content = fs.readFileSync(filePath, 'utf-8');
  return hashContent(content);
}

/**
 * Find all .gtest files in directory (recursive)
 */
export function findTestFiles(rootDir: string): string[] {
  const testFiles: string[] = [];

  function walk(dir: string) {
    const entries = fs.readdirSync(dir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);

      if (entry.isDirectory()) {
        // Skip node_modules and hidden directories
        if (!entry.name.startsWith('.') && entry.name !== 'node_modules') {
          walk(fullPath);
        }
      } else if (entry.isFile() && entry.name.endsWith('.gtest')) {
        testFiles.push(fullPath);
      }
    }
  }

  walk(rootDir);
  return testFiles;
}

/**
 * Save index to disk (for incremental testing)
 */
export function saveIndex(index: GTestIndex, outputPath: string): void {
  const serialized = {
    version: index.version,
    timestamp: index.timestamp,
    suites: Array.from(index.suites.entries()),
    tests: Array.from(index.tests.entries()),
    files: Array.from(index.files.entries()),
    hashes: Array.from(index.hashes.entries())
  };

  fs.writeFileSync(outputPath, JSON.stringify(serialized, null, 2), 'utf-8');
}

/**
 * Load index from disk
 */
export function loadIndex(inputPath: string): GTestIndex {
  const content = fs.readFileSync(inputPath, 'utf-8');
  const parsed = JSON.parse(content);

  return {
    version: parsed.version,
    timestamp: parsed.timestamp,
    suites: new Map(parsed.suites),
    tests: new Map(parsed.tests),
    files: new Map(parsed.files),
    hashes: new Map(parsed.hashes)
  };
}

// ============================================================================
// Test Generation
// ============================================================================

/**
 * Generate .gtest file template
 */
export function generateTestTemplate(name: string): string {
  return `# ${name} - Test Suite

setup:
  # Run once before all tests
  # Initialize database, create fixtures, etc.

teardown:
  # Run once after all tests
  # Clean up resources

beforeEach:
  # Run before each test
  # Reset state, clear caches

afterEach:
  # Run after each test
  # Verify cleanup

test "should handle basic case":
  given: input = "hello"
  when: result = process(input)
  then: expect result equals "HELLO"

test "should handle edge cases":
  given: input = ""
  when: result = process(input)
  then: expect result equals ""

test "should reject invalid input":
  given: input = null
  when: result = process(input)
  then: expect result throws "Invalid input"
`;
}
