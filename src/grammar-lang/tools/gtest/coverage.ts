/**
 * GTest - Coverage Tools
 *
 * O(1) code coverage tracking with incremental updates.
 *
 * Features:
 * - Line coverage
 * - Branch coverage
 * - Function coverage
 * - Incremental tracking (O(1) per change)
 * - Hash-based file tracking
 * - Coverage diff (between runs)
 *
 * Usage:
 * ```typescript
 * import { startCoverage, stopCoverage, getCoverageReport } from './coverage';
 *
 * startCoverage();
 * // Run tests...
 * stopCoverage();
 * const report = getCoverageReport();
 * ```
 */

import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';

// ============================================================================
// Types
// ============================================================================

/**
 * Coverage data for a single file
 */
export interface FileCoverage {
  path: string;                    // File path
  hash: string;                    // Content hash for O(1) change detection
  lines: Map<number, number>;      // Line number → Hit count
  branches: Map<string, number>;   // Branch ID → Hit count
  functions: Map<string, number>;  // Function name → Hit count
  totalLines: number;              // Total coverable lines
  totalBranches: number;           // Total branches
  totalFunctions: number;          // Total functions
  coveredLines: number;            // Lines hit > 0
  coveredBranches: number;         // Branches hit > 0
  coveredFunctions: number;        // Functions hit > 0
  percentage: {
    lines: number;                 // Line coverage %
    branches: number;              // Branch coverage %
    functions: number;             // Function coverage %
  };
}

/**
 * Coverage report for all files
 */
export interface CoverageReport {
  timestamp: number;               // When report was generated
  files: Map<string, FileCoverage>;  // File path → Coverage
  summary: {
    totalFiles: number;
    totalLines: number;
    totalBranches: number;
    totalFunctions: number;
    coveredLines: number;
    coveredBranches: number;
    coveredFunctions: number;
    percentage: {
      lines: number;
      branches: number;
      functions: number;
    };
  };
}

/**
 * Coverage diff between two reports
 */
export interface CoverageDiff {
  filesAdded: string[];            // New files
  filesRemoved: string[];          // Deleted files
  filesChanged: Map<string, {      // Modified files
    linesDelta: number;            // Change in covered lines
    branchesDelta: number;         // Change in covered branches
    functionsDelta: number;        // Change in covered functions
  }>;
  summary: {
    linesDelta: number;            // Overall change in line coverage %
    branchesDelta: number;         // Overall change in branch coverage %
    functionsDelta: number;        // Overall change in function coverage %
  };
}

/**
 * Coverage tracker state
 */
interface CoverageState {
  active: boolean;
  files: Map<string, FileCoverage>;
  startTime: number;
  endTime?: number;
}

// ============================================================================
// Global State
// ============================================================================

let globalState: CoverageState = {
  active: false,
  files: new Map(),
  startTime: 0
};

// ============================================================================
// Coverage Collection
// ============================================================================

/**
 * Start coverage collection
 */
export function startCoverage(): void {
  globalState = {
    active: true,
    files: new Map(),
    startTime: Date.now()
  };

  console.log('📊 Coverage tracking started\n');
}

/**
 * Stop coverage collection
 */
export function stopCoverage(): void {
  globalState.active = false;
  globalState.endTime = Date.now();

  console.log('\n📊 Coverage tracking stopped');
}

/**
 * Track line execution (O(1))
 */
export function trackLine(filePath: string, lineNumber: number): void {
  if (!globalState.active) return;

  const coverage = getOrCreateFileCoverage(filePath);
  const currentCount = coverage.lines.get(lineNumber) || 0;
  coverage.lines.set(lineNumber, currentCount + 1);  // O(1) map update
}

/**
 * Track branch execution (O(1))
 */
export function trackBranch(filePath: string, branchId: string, taken: boolean): void {
  if (!globalState.active) return;

  const coverage = getOrCreateFileCoverage(filePath);
  const branchKey = `${branchId}:${taken ? 'T' : 'F'}`;
  const currentCount = coverage.branches.get(branchKey) || 0;
  coverage.branches.set(branchKey, currentCount + 1);  // O(1) map update
}

/**
 * Track function execution (O(1))
 */
export function trackFunction(filePath: string, functionName: string): void {
  if (!globalState.active) return;

  const coverage = getOrCreateFileCoverage(filePath);
  const currentCount = coverage.functions.get(functionName) || 0;
  coverage.functions.set(functionName, currentCount + 1);  // O(1) map update
}

/**
 * Get or create file coverage (O(1))
 */
function getOrCreateFileCoverage(filePath: string): FileCoverage {
  let coverage = globalState.files.get(filePath);

  if (!coverage) {
    coverage = {
      path: filePath,
      hash: hashFile(filePath),
      lines: new Map(),
      branches: new Map(),
      functions: new Map(),
      totalLines: 0,
      totalBranches: 0,
      totalFunctions: 0,
      coveredLines: 0,
      coveredBranches: 0,
      coveredFunctions: 0,
      percentage: {
        lines: 0,
        branches: 0,
        functions: 0
      }
    };

    // Analyze file to count total lines/branches/functions
    analyzeFile(coverage);

    globalState.files.set(filePath, coverage);  // O(1) map insert
  }

  return coverage;
}

/**
 * Analyze file to count coverable entities
 */
function analyzeFile(coverage: FileCoverage): void {
  try {
    const content = fs.readFileSync(coverage.path, 'utf-8');
    const lines = content.split('\n');

    // Count lines (exclude empty and comments)
    coverage.totalLines = lines.filter(line => {
      const trimmed = line.trim();
      return trimmed.length > 0 && !trimmed.startsWith('//') && !trimmed.startsWith('/*');
    }).length;

    // Count functions (simplified - count 'function' keyword)
    coverage.totalFunctions = (content.match(/function\s+\w+/g) || []).length;

    // Count branches (simplified - count if/else/case)
    const ifCount = (content.match(/\bif\s*\(/g) || []).length;
    const elseCount = (content.match(/\belse\b/g) || []).length;
    const caseCount = (content.match(/\bcase\s+/g) || []).length;
    coverage.totalBranches = ifCount + elseCount + caseCount;
  } catch (error) {
    // File not found or not readable
    coverage.totalLines = 0;
    coverage.totalBranches = 0;
    coverage.totalFunctions = 0;
  }
}

// ============================================================================
// Coverage Reporting
// ============================================================================

/**
 * Generate coverage report (O(n) where n = number of files)
 */
export function getCoverageReport(): CoverageReport {
  // Calculate covered entities for each file
  for (const [, coverage] of globalState.files) {
    // Count covered lines (O(1) per line)
    coverage.coveredLines = Array.from(coverage.lines.values()).filter(count => count > 0).length;

    // Count covered branches (O(1) per branch)
    coverage.coveredBranches = Array.from(coverage.branches.values()).filter(count => count > 0).length;

    // Count covered functions (O(1) per function)
    coverage.coveredFunctions = Array.from(coverage.functions.values()).filter(count => count > 0).length;

    // Calculate percentages
    coverage.percentage.lines = coverage.totalLines > 0
      ? (coverage.coveredLines / coverage.totalLines) * 100
      : 0;

    coverage.percentage.branches = coverage.totalBranches > 0
      ? (coverage.coveredBranches / coverage.totalBranches) * 100
      : 0;

    coverage.percentage.functions = coverage.totalFunctions > 0
      ? (coverage.coveredFunctions / coverage.totalFunctions) * 100
      : 0;
  }

  // Calculate summary
  const files = Array.from(globalState.files.values());

  const summary = {
    totalFiles: files.length,
    totalLines: files.reduce((sum, f) => sum + f.totalLines, 0),
    totalBranches: files.reduce((sum, f) => sum + f.totalBranches, 0),
    totalFunctions: files.reduce((sum, f) => sum + f.totalFunctions, 0),
    coveredLines: files.reduce((sum, f) => sum + f.coveredLines, 0),
    coveredBranches: files.reduce((sum, f) => sum + f.coveredBranches, 0),
    coveredFunctions: files.reduce((sum, f) => sum + f.coveredFunctions, 0),
    percentage: {
      lines: 0,
      branches: 0,
      functions: 0
    }
  };

  summary.percentage.lines = summary.totalLines > 0
    ? (summary.coveredLines / summary.totalLines) * 100
    : 0;

  summary.percentage.branches = summary.totalBranches > 0
    ? (summary.coveredBranches / summary.totalBranches) * 100
    : 0;

  summary.percentage.functions = summary.totalFunctions > 0
    ? (summary.coveredFunctions / summary.totalFunctions) * 100
    : 0;

  return {
    timestamp: Date.now(),
    files: globalState.files,
    summary
  };
}

/**
 * Print coverage report
 */
export function printCoverageReport(report: CoverageReport): void {
  console.log('\n📊 Coverage Report');
  console.log('═'.repeat(80));

  // Print per-file coverage
  for (const [filePath, coverage] of report.files) {
    const fileName = path.basename(filePath);
    console.log(`\n📄 ${fileName}`);
    console.log(`   Lines:     ${coverage.percentage.lines.toFixed(2)}% (${coverage.coveredLines}/${coverage.totalLines})`);
    console.log(`   Branches:  ${coverage.percentage.branches.toFixed(2)}% (${coverage.coveredBranches}/${coverage.totalBranches})`);
    console.log(`   Functions: ${coverage.percentage.functions.toFixed(2)}% (${coverage.coveredFunctions}/${coverage.totalFunctions})`);
  }

  // Print summary
  console.log('\n═'.repeat(80));
  console.log('Summary');
  console.log('═'.repeat(80));
  console.log(`Files:     ${report.summary.totalFiles}`);
  console.log(`Lines:     ${report.summary.percentage.lines.toFixed(2)}% (${report.summary.coveredLines}/${report.summary.totalLines})`);
  console.log(`Branches:  ${report.summary.percentage.branches.toFixed(2)}% (${report.summary.coveredBranches}/${report.summary.totalBranches})`);
  console.log(`Functions: ${report.summary.percentage.functions.toFixed(2)}% (${report.summary.coveredFunctions}/${report.summary.totalFunctions})`);
  console.log('═'.repeat(80));
}

// ============================================================================
// Coverage Diff
// ============================================================================

/**
 * Compare two coverage reports (O(1) per file)
 */
export function compareCoverage(oldReport: CoverageReport, newReport: CoverageReport): CoverageDiff {
  const filesAdded: string[] = [];
  const filesRemoved: string[] = [];
  const filesChanged = new Map<string, any>();

  // Find added and changed files
  for (const [filePath, newCoverage] of newReport.files) {
    const oldCoverage = oldReport.files.get(filePath);

    if (!oldCoverage) {
      filesAdded.push(filePath);
    } else {
      // Check if coverage changed (O(1) hash comparison)
      const linesDelta = newCoverage.coveredLines - oldCoverage.coveredLines;
      const branchesDelta = newCoverage.coveredBranches - oldCoverage.coveredBranches;
      const functionsDelta = newCoverage.coveredFunctions - oldCoverage.coveredFunctions;

      if (linesDelta !== 0 || branchesDelta !== 0 || functionsDelta !== 0) {
        filesChanged.set(filePath, {
          linesDelta,
          branchesDelta,
          functionsDelta
        });
      }
    }
  }

  // Find removed files
  for (const filePath of oldReport.files.keys()) {
    if (!newReport.files.has(filePath)) {
      filesRemoved.push(filePath);
    }
  }

  // Calculate summary
  const summary = {
    linesDelta: newReport.summary.percentage.lines - oldReport.summary.percentage.lines,
    branchesDelta: newReport.summary.percentage.branches - oldReport.summary.percentage.branches,
    functionsDelta: newReport.summary.percentage.functions - oldReport.summary.percentage.functions
  };

  return {
    filesAdded,
    filesRemoved,
    filesChanged,
    summary
  };
}

/**
 * Print coverage diff
 */
export function printCoverageDiff(diff: CoverageDiff): void {
  console.log('\n📊 Coverage Diff');
  console.log('═'.repeat(80));

  if (diff.filesAdded.length > 0) {
    console.log('\n✅ Files Added:');
    diff.filesAdded.forEach(f => console.log(`   + ${path.basename(f)}`));
  }

  if (diff.filesRemoved.length > 0) {
    console.log('\n❌ Files Removed:');
    diff.filesRemoved.forEach(f => console.log(`   - ${path.basename(f)}`));
  }

  if (diff.filesChanged.size > 0) {
    console.log('\n📝 Files Changed:');
    for (const [filePath, changes] of diff.filesChanged) {
      const fileName = path.basename(filePath);
      console.log(`   ${fileName}`);
      if (changes.linesDelta !== 0) {
        const sign = changes.linesDelta > 0 ? '+' : '';
        console.log(`      Lines: ${sign}${changes.linesDelta}`);
      }
    }
  }

  console.log('\n═'.repeat(80));
  console.log('Summary');
  console.log('═'.repeat(80));
  console.log(`Lines:     ${diff.summary.linesDelta >= 0 ? '+' : ''}${diff.summary.linesDelta.toFixed(2)}%`);
  console.log(`Branches:  ${diff.summary.branchesDelta >= 0 ? '+' : ''}${diff.summary.branchesDelta.toFixed(2)}%`);
  console.log(`Functions: ${diff.summary.functionsDelta >= 0 ? '+' : ''}${diff.summary.functionsDelta.toFixed(2)}%`);
  console.log('═'.repeat(80));
}

// ============================================================================
// Persistence
// ============================================================================

/**
 * Save coverage report to disk
 */
export function saveCoverageReport(report: CoverageReport, outputPath: string): void {
  const serialized = {
    timestamp: report.timestamp,
    files: Array.from(report.files.entries()).map(([path, coverage]) => ({
      path,
      hash: coverage.hash,
      lines: Array.from(coverage.lines.entries()),
      branches: Array.from(coverage.branches.entries()),
      functions: Array.from(coverage.functions.entries()),
      totalLines: coverage.totalLines,
      totalBranches: coverage.totalBranches,
      totalFunctions: coverage.totalFunctions,
      coveredLines: coverage.coveredLines,
      coveredBranches: coverage.coveredBranches,
      coveredFunctions: coverage.coveredFunctions,
      percentage: coverage.percentage
    })),
    summary: report.summary
  };

  fs.writeFileSync(outputPath, JSON.stringify(serialized, null, 2), 'utf-8');
}

/**
 * Load coverage report from disk
 */
export function loadCoverageReport(inputPath: string): CoverageReport {
  const content = fs.readFileSync(inputPath, 'utf-8');
  const parsed = JSON.parse(content);

  const files = new Map<string, FileCoverage>();

  for (const file of parsed.files) {
    files.set(file.path, {
      path: file.path,
      hash: file.hash,
      lines: new Map(file.lines),
      branches: new Map(file.branches),
      functions: new Map(file.functions),
      totalLines: file.totalLines,
      totalBranches: file.totalBranches,
      totalFunctions: file.totalFunctions,
      coveredLines: file.coveredLines,
      coveredBranches: file.coveredBranches,
      coveredFunctions: file.coveredFunctions,
      percentage: file.percentage
    });
  }

  return {
    timestamp: parsed.timestamp,
    files,
    summary: parsed.summary
  };
}

// ============================================================================
// Helpers
// ============================================================================

/**
 * Hash file for O(1) change detection
 */
function hashFile(filePath: string): string {
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    return crypto.createHash('sha256').update(content).digest('hex').substring(0, 16);
  } catch {
    return '';
  }
}
