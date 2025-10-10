/**
 * GTest - O(1) Testing Framework for Grammar Language
 *
 * Main entry point for GTest framework.
 *
 * Features:
 * - Hash-based test discovery (O(1) lookup)
 * - Incremental testing (only changed files)
 * - Parallel execution
 * - Code coverage tracking
 * - Integration with GLM/GSX/GLC
 *
 * Usage:
 * ```typescript
 * import { runTests, expect } from '@grammar-lang/gtest';
 *
 * // Run all tests
 * const summary = await runTests('./tests');
 *
 * // Use assertions
 * test('example', () => {
 *   expect(2 + 2).toEqual(4);
 * });
 * ```
 */

// ============================================================================
// Core Test Specification
// ============================================================================

export {
  // Types
  GTestCase,
  GTestSuite,
  GTestIndex,
  GTestResult,
  GTestSummary,

  // Parser
  parseGTestFile,
  parseHooks,

  // Index Management
  createTestIndex,
  findChangedTests,
  getTestById,
  getSuiteById,
  getSuiteByFile,

  // Helpers
  hashContent,
  hashFile,
  findTestFiles,
  saveIndex,
  loadIndex,

  // Template Generation
  generateTestTemplate
} from './spec';

// ============================================================================
// Test Runner
// ============================================================================

export {
  // Types
  GTestConfig,
  RunnerState,

  // Runner
  GTestRunner,
  createRunner,

  // Quick Run Functions
  runTests,
  runIncrementalTests,
  runParallelTests
} from './runner';

// ============================================================================
// Assertions
// ============================================================================

export {
  // Core Assertion API
  expect,
  Assertion,

  // Types
  AssertionContext,
  Matcher,
  MatcherResult,

  // Error
  AssertionError,

  // Mock/Spy
  createMock,
  createSpy,
  Mock,
  Spy
} from './assertions';

// ============================================================================
// Coverage
// ============================================================================

export {
  // Types
  FileCoverage,
  CoverageReport,
  CoverageDiff,

  // Coverage Collection
  startCoverage,
  stopCoverage,
  trackLine,
  trackBranch,
  trackFunction,

  // Reporting
  getCoverageReport,
  printCoverageReport,
  compareCoverage,
  printCoverageDiff,

  // Persistence
  saveCoverageReport,
  loadCoverageReport
} from './coverage';

// ============================================================================
// Integration
// ============================================================================

export {
  // GLM Integration
  GLMTestConfig,
  GLMTestResult,
  runGLMPackageTests,
  validateGLMPackage,

  // GSX Integration
  GSXTestContext,
  GSXTestExecutor,
  createGSXExecutor,
  runTestsWithGSX,

  // GLC Integration
  GLCTestConfig,
  GLCTestResult,
  runTestsWithGLC,
  validateGLCCompilation,

  // Unified Runner
  UnifiedTestConfig,
  UnifiedTestResult,
  runUnifiedTests,
  runTestsFromCLI,
  watchTests
} from './integration';

// ============================================================================
// Version
// ============================================================================

export const VERSION = '1.0.0';

// ============================================================================
// Default Export (Convenience API)
// ============================================================================

import { expect } from './assertions';
import { runTests, runIncrementalTests, runParallelTests } from './runner';
import { startCoverage, stopCoverage, getCoverageReport } from './coverage';
import { runUnifiedTests, runTestsFromCLI, watchTests } from './integration';

/**
 * Default export with convenience API
 */
export default {
  // Test Execution
  run: runTests,
  runIncremental: runIncrementalTests,
  runParallel: runParallelTests,
  runUnified: runUnifiedTests,
  runCLI: runTestsFromCLI,
  watch: watchTests,

  // Assertions
  expect,

  // Coverage
  coverage: {
    start: startCoverage,
    stop: stopCoverage,
    report: getCoverageReport
  },

  // Version
  version: VERSION
};
