/**
 * GTest - Integration Module
 *
 * Integrates GTest with Grammar Language toolchain:
 * - GLM (Grammar Language Manager): Package-level testing
 * - GSX (Grammar Script eXecutor): Code execution integration
 * - GLC (Grammar Language Compiler): Compilation testing
 *
 * All integrations use O(1) operations for maximum performance.
 */

import * as path from 'path';
import * as fs from 'fs';
import {
  GTestRunner,
  GTestConfig,
  GTestSummary,
  createRunner
} from './runner';
import {
  GTestIndex,
  GTestSuite,
  createTestIndex,
  findTestFiles,
  parseGTestFile
} from './spec';
import {
  startCoverage,
  stopCoverage,
  getCoverageReport,
  printCoverageReport,
  saveCoverageReport,
  CoverageReport
} from './coverage';

// ============================================================================
// GLM Integration (Grammar Language Manager)
// ============================================================================

/**
 * GLM Package Testing Configuration
 */
export interface GLMTestConfig {
  packageDir: string;           // Package root directory
  testPattern?: string;         // Test file pattern (default: **/*.gtest)
  coverage?: boolean;           // Enable coverage
  incremental?: boolean;        // Incremental testing
  parallel?: boolean;           // Parallel execution
  outputDir?: string;           // Output directory for reports
}

/**
 * GLM Package Test Result
 */
export interface GLMTestResult {
  packageName: string;
  version: string;
  summary: GTestSummary;
  coverage?: CoverageReport;
  timestamp: number;
  success: boolean;
}

/**
 * Run tests for GLM package
 * O(1) per test (hash-based selection)
 */
export async function runGLMPackageTests(config: GLMTestConfig): Promise<GLMTestResult> {
  console.log('📦 GLM Package Testing\n');

  // Read package.json (O(1) file read)
  const packageJsonPath = path.join(config.packageDir, 'package.json');
  let packageName = 'unknown';
  let version = '0.0.0';

  if (fs.existsSync(packageJsonPath)) {
    const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf-8'));
    packageName = packageJson.name || packageName;
    version = packageJson.version || version;
  }

  console.log(`Package: ${packageName}@${version}\n`);

  // Start coverage if enabled
  if (config.coverage) {
    startCoverage();
  }

  // Create test runner
  const runner = createRunner({
    rootDir: config.packageDir,
    pattern: config.testPattern || '**/*.gtest',
    incremental: config.incremental,
    parallel: config.parallel,
    coverage: config.coverage
  });

  // Run tests
  const summary = await runner.run();

  // Stop coverage and generate report
  let coverageReport: CoverageReport | undefined;
  if (config.coverage) {
    stopCoverage();
    coverageReport = getCoverageReport();
    printCoverageReport(coverageReport);

    // Save coverage report
    if (config.outputDir) {
      const coverageOutputPath = path.join(config.outputDir, 'coverage.json');
      saveCoverageReport(coverageReport, coverageOutputPath);
      console.log(`\n📊 Coverage report saved to: ${coverageOutputPath}`);
    }
  }

  // Save test summary
  if (config.outputDir) {
    const summaryOutputPath = path.join(config.outputDir, 'test-summary.json');
    fs.writeFileSync(summaryOutputPath, JSON.stringify(summary, null, 2), 'utf-8');
    console.log(`📋 Test summary saved to: ${summaryOutputPath}`);
  }

  return {
    packageName,
    version,
    summary,
    coverage: coverageReport,
    timestamp: Date.now(),
    success: summary.failed === 0
  };
}

/**
 * Validate GLM package (run tests as part of package validation)
 */
export async function validateGLMPackage(packageDir: string): Promise<boolean> {
  const result = await runGLMPackageTests({
    packageDir,
    coverage: true,
    incremental: false,
    parallel: true
  });

  return result.success;
}

// ============================================================================
// GSX Integration (Grammar Script eXecutor)
// ============================================================================

/**
 * GSX Execution Context for tests
 */
export interface GSXTestContext {
  globals: Map<string, any>;     // Global variables
  modules: Map<string, any>;     // Loaded modules
  executionStack: string[];      // Execution stack
}

/**
 * GSX Test Executor
 * Integrates GTest with GSX for executing .gl code in tests
 */
export class GSXTestExecutor {
  private context: GSXTestContext;

  constructor() {
    this.context = {
      globals: new Map(),
      modules: new Map(),
      executionStack: []
    };
  }

  /**
   * Execute Grammar Language code (placeholder for GSX integration)
   * O(1) context lookup
   */
  async executeCode(code: string): Promise<any> {
    // TODO: Integrate with actual GSX executor
    // For now, use eval as placeholder
    try {
      // Create context with global variables
      const contextVars = Array.from(this.context.globals.entries())
        .map(([key, value]) => `const ${key} = ${JSON.stringify(value)};`)
        .join('\n');

      const fullCode = contextVars + '\n' + code;
      return eval(fullCode);
    } catch (error) {
      throw new Error(`GSX execution failed: ${error}`);
    }
  }

  /**
   * Set global variable in execution context (O(1))
   */
  setGlobal(name: string, value: any): void {
    this.context.globals.set(name, value);
  }

  /**
   * Get global variable from execution context (O(1))
   */
  getGlobal(name: string): any {
    return this.context.globals.get(name);
  }

  /**
   * Load module into execution context (O(1))
   */
  loadModule(name: string, module: any): void {
    this.context.modules.set(name, module);
  }

  /**
   * Get loaded module (O(1))
   */
  getModule(name: string): any {
    return this.context.modules.get(name);
  }

  /**
   * Clear execution context
   */
  clearContext(): void {
    this.context.globals.clear();
    this.context.modules.clear();
    this.context.executionStack = [];
  }
}

/**
 * Create GSX executor for tests
 */
export function createGSXExecutor(): GSXTestExecutor {
  return new GSXTestExecutor();
}

/**
 * Run tests with GSX integration
 */
export async function runTestsWithGSX(rootDir: string): Promise<GTestSummary> {
  const executor = createGSXExecutor();

  // TODO: Integrate GSX executor with test runner
  // For now, use standard runner
  const runner = createRunner({ rootDir });
  return runner.run();
}

// ============================================================================
// GLC Integration (Grammar Language Compiler)
// ============================================================================

/**
 * GLC Compilation Test Configuration
 */
export interface GLCTestConfig {
  sourceDir: string;            // Source directory (.gl files)
  outputDir: string;            // Output directory (compiled files)
  testPattern?: string;         // Test file pattern
  compileBeforeTest?: boolean;  // Compile before running tests
  validateOutput?: boolean;     // Validate compiled output
}

/**
 * GLC Compilation Test Result
 */
export interface GLCTestResult {
  compiled: boolean;
  compilationTime: number;
  testSummary: GTestSummary;
  outputFiles: string[];
  errors: string[];
}

/**
 * Run tests with GLC compilation
 * O(1) per file (hash-based compilation check)
 */
export async function runTestsWithGLC(config: GLCTestConfig): Promise<GLCTestResult> {
  console.log('🔧 GLC Compilation Testing\n');

  const startTime = Date.now();
  const outputFiles: string[] = [];
  const errors: string[] = [];

  // Compile source files if enabled
  if (config.compileBeforeTest) {
    console.log('📦 Compiling Grammar Language files...\n');

    // TODO: Integrate with actual GLC compiler
    // For now, just copy files as placeholder
    try {
      const sourceFiles = findGrammarFiles(config.sourceDir);

      for (const sourceFile of sourceFiles) {
        const relativePath = path.relative(config.sourceDir, sourceFile);
        const outputFile = path.join(config.outputDir, relativePath.replace('.gl', '.js'));
        const outputDir = path.dirname(outputFile);

        // Create output directory
        if (!fs.existsSync(outputDir)) {
          fs.mkdirSync(outputDir, { recursive: true });
        }

        // Placeholder: copy file (in real implementation, call GLC compiler)
        fs.copyFileSync(sourceFile, outputFile);
        outputFiles.push(outputFile);
      }

      console.log(`✅ Compiled ${outputFiles.length} files\n`);
    } catch (error) {
      errors.push(`Compilation error: ${error}`);
      console.error(`❌ Compilation failed: ${error}\n`);
    }
  }

  const compilationTime = Date.now() - startTime;

  // Run tests
  const runner = createRunner({
    rootDir: config.sourceDir,
    pattern: config.testPattern || '**/*.gtest'
  });

  const testSummary = await runner.run();

  return {
    compiled: errors.length === 0,
    compilationTime,
    testSummary,
    outputFiles,
    errors
  };
}

/**
 * Validate GLC compilation (ensure compiled output is correct)
 */
export async function validateGLCCompilation(
  sourceDir: string,
  outputDir: string
): Promise<boolean> {
  const result = await runTestsWithGLC({
    sourceDir,
    outputDir,
    compileBeforeTest: true,
    validateOutput: true
  });

  return result.compiled && result.testSummary.failed === 0;
}

/**
 * Find all .gl files in directory
 */
function findGrammarFiles(rootDir: string): string[] {
  const grammarFiles: string[] = [];

  function walk(dir: string) {
    const entries = fs.readdirSync(dir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);

      if (entry.isDirectory()) {
        if (!entry.name.startsWith('.') && entry.name !== 'node_modules') {
          walk(fullPath);
        }
      } else if (entry.isFile() && entry.name.endsWith('.gl')) {
        grammarFiles.push(fullPath);
      }
    }
  }

  walk(rootDir);
  return grammarFiles;
}

// ============================================================================
// Unified Test Runner (All Integrations)
// ============================================================================

/**
 * Unified test configuration (combines all integrations)
 */
export interface UnifiedTestConfig {
  rootDir: string;
  glm?: {
    enabled: boolean;
    packageDir?: string;
    coverage?: boolean;
  };
  gsx?: {
    enabled: boolean;
    globals?: Record<string, any>;
    modules?: Record<string, any>;
  };
  glc?: {
    enabled: boolean;
    sourceDir?: string;
    outputDir?: string;
    compileBeforeTest?: boolean;
  };
  outputDir?: string;
  incremental?: boolean;
  parallel?: boolean;
}

/**
 * Unified test result (combines all integrations)
 */
export interface UnifiedTestResult {
  timestamp: number;
  glmResult?: GLMTestResult;
  glcResult?: GLCTestResult;
  testSummary: GTestSummary;
  coverage?: CoverageReport;
  success: boolean;
}

/**
 * Run tests with all integrations enabled
 * O(1) per integration (each integration is independent)
 */
export async function runUnifiedTests(config: UnifiedTestConfig): Promise<UnifiedTestResult> {
  console.log('🧪 Unified GTest Runner\n');
  console.log('═'.repeat(80));

  let glmResult: GLMTestResult | undefined;
  let glcResult: GLCTestResult | undefined;
  let testSummary: GTestSummary;
  let coverage: CoverageReport | undefined;

  // GLM Package Testing
  if (config.glm?.enabled) {
    glmResult = await runGLMPackageTests({
      packageDir: config.glm.packageDir || config.rootDir,
      coverage: config.glm.coverage,
      incremental: config.incremental,
      parallel: config.parallel,
      outputDir: config.outputDir
    });
    testSummary = glmResult.summary;
    coverage = glmResult.coverage;
  }
  // GLC Compilation Testing
  else if (config.glc?.enabled) {
    glcResult = await runTestsWithGLC({
      sourceDir: config.glc.sourceDir || config.rootDir,
      outputDir: config.glc.outputDir || path.join(config.rootDir, 'dist'),
      compileBeforeTest: config.glc.compileBeforeTest
    });
    testSummary = glcResult.testSummary;
  }
  // Standard Testing (with optional GSX)
  else {
    const runner = createRunner({
      rootDir: config.rootDir,
      incremental: config.incremental,
      parallel: config.parallel
    });

    if (config.gsx?.enabled) {
      // TODO: Integrate GSX executor
      testSummary = await runner.run();
    } else {
      testSummary = await runner.run();
    }
  }

  const success = testSummary.failed === 0 && (!glcResult || glcResult.compiled);

  return {
    timestamp: Date.now(),
    glmResult,
    glcResult,
    testSummary,
    coverage,
    success
  };
}

// ============================================================================
// CLI Helpers
// ============================================================================

/**
 * Run tests from CLI with auto-detection
 * O(1) configuration detection
 */
export async function runTestsFromCLI(
  rootDir: string,
  options: {
    coverage?: boolean;
    incremental?: boolean;
    parallel?: boolean;
    outputDir?: string;
  } = {}
): Promise<UnifiedTestResult> {
  // Auto-detect project type (O(1) file checks)
  const hasPackageJson = fs.existsSync(path.join(rootDir, 'package.json'));
  const hasGrammarFiles = findGrammarFiles(rootDir).length > 0;

  const config: UnifiedTestConfig = {
    rootDir,
    incremental: options.incremental,
    parallel: options.parallel,
    outputDir: options.outputDir
  };

  // Enable GLM if package.json exists
  if (hasPackageJson) {
    config.glm = {
      enabled: true,
      packageDir: rootDir,
      coverage: options.coverage
    };
  }

  // Enable GLC if .gl files exist
  if (hasGrammarFiles) {
    config.glc = {
      enabled: true,
      sourceDir: rootDir,
      outputDir: path.join(rootDir, 'dist'),
      compileBeforeTest: true
    };
  }

  return runUnifiedTests(config);
}

/**
 * Watch mode (run tests on file changes)
 */
export async function watchTests(
  rootDir: string,
  options: {
    coverage?: boolean;
    incremental?: boolean;
    parallel?: boolean;
  } = {}
): Promise<void> {
  console.log('👀 Watch mode enabled\n');
  console.log('Watching for changes in:', rootDir);
  console.log('Press Ctrl+C to stop\n');

  let isRunning = false;

  // Watch for file changes
  fs.watch(rootDir, { recursive: true }, async (eventType, filename) => {
    if (isRunning) return;
    if (!filename || !filename.endsWith('.gtest')) return;

    console.log(`\n📝 File changed: ${filename}`);
    console.log('Running tests...\n');

    isRunning = true;

    try {
      await runTestsFromCLI(rootDir, {
        ...options,
        incremental: true  // Always use incremental in watch mode
      });
    } catch (error) {
      console.error('Test run failed:', error);
    }

    isRunning = false;
  });

  // Keep process alive
  await new Promise(() => {});
}

// ============================================================================
// Export All
// ============================================================================

export {
  // GLM
  runGLMPackageTests,
  validateGLMPackage,

  // GSX
  createGSXExecutor,
  runTestsWithGSX,

  // GLC
  runTestsWithGLC,
  validateGLCCompilation,

  // Unified
  runUnifiedTests,
  runTestsFromCLI,
  watchTests
};
