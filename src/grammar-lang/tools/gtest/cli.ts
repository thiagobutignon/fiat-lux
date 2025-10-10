#!/usr/bin/env node
/**
 * GTest CLI - Command Line Interface
 *
 * Usage:
 *   gtest [directory] [options]
 *
 * Options:
 *   -c, --coverage        Enable code coverage
 *   -i, --incremental     Run only changed tests
 *   -p, --parallel        Run tests in parallel
 *   -w, --watch           Watch mode (re-run on changes)
 *   -o, --output <dir>    Output directory for reports
 *   -v, --verbose         Verbose output
 *   --glm                 Enable GLM package testing
 *   --gsx                 Enable GSX integration
 *   --glc                 Enable GLC compilation testing
 *   --version             Show version
 *   --help                Show help
 *
 * Examples:
 *   gtest                           # Run all tests in current directory
 *   gtest ./tests                   # Run tests in ./tests directory
 *   gtest -c -p                     # Run with coverage in parallel
 *   gtest -w -i                     # Watch mode with incremental testing
 *   gtest --glm -c                  # GLM package testing with coverage
 */

import * as path from 'path';
import * as fs from 'fs';
import {
  runTestsFromCLI,
  watchTests,
  runGLMPackageTests,
  runTestsWithGLC,
  UnifiedTestResult
} from './integration';
import { VERSION } from './index';

// ============================================================================
// CLI Arguments Parser
// ============================================================================

interface CLIArgs {
  directory: string;
  coverage: boolean;
  incremental: boolean;
  parallel: boolean;
  watch: boolean;
  output?: string;
  verbose: boolean;
  glm: boolean;
  gsx: boolean;
  glc: boolean;
  help: boolean;
  version: boolean;
}

function parseArgs(args: string[]): CLIArgs {
  const parsed: CLIArgs = {
    directory: process.cwd(),
    coverage: false,
    incremental: false,
    parallel: false,
    watch: false,
    verbose: false,
    glm: false,
    gsx: false,
    glc: false,
    help: false,
    version: false
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];

    switch (arg) {
      case '-c':
      case '--coverage':
        parsed.coverage = true;
        break;

      case '-i':
      case '--incremental':
        parsed.incremental = true;
        break;

      case '-p':
      case '--parallel':
        parsed.parallel = true;
        break;

      case '-w':
      case '--watch':
        parsed.watch = true;
        break;

      case '-o':
      case '--output':
        parsed.output = args[++i];
        break;

      case '-v':
      case '--verbose':
        parsed.verbose = true;
        break;

      case '--glm':
        parsed.glm = true;
        break;

      case '--gsx':
        parsed.gsx = true;
        break;

      case '--glc':
        parsed.glc = true;
        break;

      case '--help':
        parsed.help = true;
        break;

      case '--version':
        parsed.version = true;
        break;

      default:
        // Assume it's the directory
        if (!arg.startsWith('-')) {
          parsed.directory = path.resolve(arg);
        }
        break;
    }
  }

  return parsed;
}

// ============================================================================
// Help Text
// ============================================================================

function showHelp(): void {
  console.log(`
GTest - O(1) Testing Framework for Grammar Language

Usage:
  gtest [directory] [options]

Options:
  -c, --coverage        Enable code coverage
  -i, --incremental     Run only changed tests
  -p, --parallel        Run tests in parallel
  -w, --watch           Watch mode (re-run on changes)
  -o, --output <dir>    Output directory for reports
  -v, --verbose         Verbose output
  --glm                 Enable GLM package testing
  --gsx                 Enable GSX integration
  --glc                 Enable GLC compilation testing
  --version             Show version
  --help                Show this help

Examples:
  gtest                           # Run all tests in current directory
  gtest ./tests                   # Run tests in ./tests directory
  gtest -c -p                     # Run with coverage in parallel
  gtest -w -i                     # Watch mode with incremental testing
  gtest --glm -c                  # GLM package testing with coverage
  gtest --glc ./src ./dist        # GLC compilation testing

Test File Format (.gtest):
  # Math Tests

  test "should add two numbers":
    given: a = 2, b = 3
    when: result = add(a, b)
    then: expect result equals 5

Features:
  ✨ O(1) test discovery via hash-based indexing
  ✨ Incremental testing (only run changed files)
  ✨ Parallel execution for faster test runs
  ✨ Code coverage tracking (line/branch/function)
  ✨ Integration with GLM, GSX, and GLC tools
  ✨ Watch mode for continuous testing

Documentation:
  https://github.com/chomsky/grammar-lang/tree/main/docs/gtest

Version: ${VERSION}
`);
}

// ============================================================================
// Version
// ============================================================================

function showVersion(): void {
  console.log(`GTest v${VERSION}`);
}

// ============================================================================
// Main CLI
// ============================================================================

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));

  // Show help
  if (args.help) {
    showHelp();
    process.exit(0);
  }

  // Show version
  if (args.version) {
    showVersion();
    process.exit(0);
  }

  // Validate directory
  if (!fs.existsSync(args.directory)) {
    console.error(`❌ Directory not found: ${args.directory}`);
    process.exit(1);
  }

  // Create output directory if specified
  if (args.output && !fs.existsSync(args.output)) {
    fs.mkdirSync(args.output, { recursive: true });
  }

  try {
    // Watch mode
    if (args.watch) {
      await watchTests(args.directory, {
        coverage: args.coverage,
        incremental: args.incremental,
        parallel: args.parallel
      });
      return;
    }

    // GLM Package Testing
    if (args.glm) {
      const result = await runGLMPackageTests({
        packageDir: args.directory,
        coverage: args.coverage,
        incremental: args.incremental,
        parallel: args.parallel,
        outputDir: args.output
      });

      if (result.success) {
        console.log('\n✅ All tests passed!');
        process.exit(0);
      } else {
        console.log('\n❌ Some tests failed!');
        process.exit(1);
      }
    }

    // GLC Compilation Testing
    else if (args.glc) {
      const sourceDir = args.directory;
      const outputDir = args.output || path.join(args.directory, 'dist');

      const result = await runTestsWithGLC({
        sourceDir,
        outputDir,
        compileBeforeTest: true,
        validateOutput: true
      });

      if (result.compiled && result.testSummary.failed === 0) {
        console.log('\n✅ Compilation and tests passed!');
        process.exit(0);
      } else {
        console.log('\n❌ Compilation or tests failed!');
        process.exit(1);
      }
    }

    // Standard Testing (with auto-detection)
    else {
      const result = await runTestsFromCLI(args.directory, {
        coverage: args.coverage,
        incremental: args.incremental,
        parallel: args.parallel,
        outputDir: args.output
      });

      if (result.success) {
        console.log('\n✅ All tests passed!');
        process.exit(0);
      } else {
        console.log('\n❌ Some tests failed!');
        process.exit(1);
      }
    }
  } catch (error) {
    console.error('\n❌ Test execution failed:');
    console.error(error);
    process.exit(1);
  }
}

// Run CLI
if (require.main === module) {
  main().catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
  });
}

export { main };
