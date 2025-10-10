/**
 * GBench - Comparison Tools
 *
 * Compare benchmark results across versions, implementations, or configurations.
 * Inspired by GVCS fitness comparison and genetic natural selection.
 *
 * Features:
 * - Side-by-side comparison
 * - Performance regression detection
 * - Statistical significance testing
 * - Relative performance calculation
 * - Winner selection (like genetic fitness)
 */

import { BenchmarkResult } from './suite';
import { PerformanceMetrics } from './metrics';

// ============================================================================
// Types
// ============================================================================

export interface ComparisonResult {
  baseline: string;
  candidate: string;
  metrics: MetricComparison[];
  verdict: ComparisonVerdict;
  summary: string;
}

export interface MetricComparison {
  name: string;
  baseline_value: number;
  candidate_value: number;
  diff_absolute: number;
  diff_percent: number;
  improvement: boolean;
  regression: boolean;
  significant: boolean;
}

export type ComparisonVerdict =
  | 'improvement'    // Candidate is better
  | 'regression'     // Candidate is worse
  | 'neutral'        // No significant difference
  | 'mixed';         // Some better, some worse

export interface ComparisonConfig {
  regression_threshold?: number;  // % threshold for regression (default 10%)
  improvement_threshold?: number; // % threshold for improvement (default 5%)
  significance_level?: number;    // Statistical significance (default 0.05)
}

// ============================================================================
// Benchmark Comparator
// ============================================================================

export class BenchmarkComparator {
  private config: Required<ComparisonConfig>;

  constructor(config: ComparisonConfig = {}) {
    this.config = {
      regression_threshold: config.regression_threshold ?? 10,
      improvement_threshold: config.improvement_threshold ?? 5,
      significance_level: config.significance_level ?? 0.05
    };
  }

  /**
   * Compare two benchmark results
   */
  compare(
    baselineName: string,
    baseline: BenchmarkResult,
    candidateName: string,
    candidate: BenchmarkResult
  ): ComparisonResult {
    const metrics: MetricComparison[] = [];

    // Compare avg time
    metrics.push(this.compareMetric(
      'avg_time_ms',
      baseline.avg_time_ms,
      candidate.avg_time_ms,
      true // lower is better
    ));

    // Compare min time
    metrics.push(this.compareMetric(
      'min_time_ms',
      baseline.min_time_ms,
      candidate.min_time_ms,
      true
    ));

    // Compare max time
    metrics.push(this.compareMetric(
      'max_time_ms',
      baseline.max_time_ms,
      candidate.max_time_ms,
      true
    ));

    // Compare p95
    metrics.push(this.compareMetric(
      'p95_time_ms',
      baseline.p95_time_ms,
      candidate.p95_time_ms,
      true
    ));

    // Compare ops/sec
    metrics.push(this.compareMetric(
      'ops_per_sec',
      baseline.ops_per_sec,
      candidate.ops_per_sec,
      false // higher is better
    ));

    // Compare memory (if available)
    if (baseline.memory_used_mb !== undefined && candidate.memory_used_mb !== undefined) {
      metrics.push(this.compareMetric(
        'memory_used_mb',
        baseline.memory_used_mb,
        candidate.memory_used_mb,
        true // lower is better
      ));
    }

    // Determine verdict
    const verdict = this.determineVerdict(metrics);

    // Generate summary
    const summary = this.generateSummary(baselineName, candidateName, metrics, verdict);

    return {
      baseline: baselineName,
      candidate: candidateName,
      metrics,
      verdict,
      summary
    };
  }

  /**
   * Compare multiple benchmarks (best-of-N selection)
   */
  selectBest(results: Map<string, BenchmarkResult>): { name: string; score: number } {
    let bestName = '';
    let bestScore = -Infinity;

    for (const [name, result] of results) {
      // Calculate fitness score (similar to GVCS)
      const score = this.calculateFitness(result);

      if (score > bestScore) {
        bestScore = score;
        bestName = name;
      }
    }

    return { name: bestName, score: bestScore };
  }

  /**
   * Detect regressions across multiple metrics
   */
  detectRegressions(
    baseline: BenchmarkResult,
    candidate: BenchmarkResult
  ): MetricComparison[] {
    const comparison = this.compare('baseline', baseline, 'candidate', candidate);
    return comparison.metrics.filter(m => m.regression && m.significant);
  }

  /**
   * Compare single metric
   */
  private compareMetric(
    name: string,
    baselineValue: number,
    candidateValue: number,
    lowerIsBetter: boolean
  ): MetricComparison {
    const diffAbsolute = candidateValue - baselineValue;
    const diffPercent = (diffAbsolute / baselineValue) * 100;

    // Determine if improvement or regression
    const isImprovement = lowerIsBetter
      ? candidateValue < baselineValue
      : candidateValue > baselineValue;

    const isRegression = lowerIsBetter
      ? candidateValue > baselineValue
      : candidateValue < baselineValue;

    // Check significance
    const absPercent = Math.abs(diffPercent);
    const significant = isRegression
      ? absPercent >= this.config.regression_threshold
      : absPercent >= this.config.improvement_threshold;

    return {
      name,
      baseline_value: baselineValue,
      candidate_value: candidateValue,
      diff_absolute: diffAbsolute,
      diff_percent: diffPercent,
      improvement: isImprovement && significant,
      regression: isRegression && significant,
      significant
    };
  }

  /**
   * Determine overall verdict
   */
  private determineVerdict(metrics: MetricComparison[]): ComparisonVerdict {
    const improvements = metrics.filter(m => m.improvement).length;
    const regressions = metrics.filter(m => m.regression).length;

    if (improvements > 0 && regressions === 0) {
      return 'improvement';
    }

    if (regressions > 0 && improvements === 0) {
      return 'regression';
    }

    if (improvements > 0 && regressions > 0) {
      return 'mixed';
    }

    return 'neutral';
  }

  /**
   * Generate human-readable summary
   */
  private generateSummary(
    baseline: string,
    candidate: string,
    metrics: MetricComparison[],
    verdict: ComparisonVerdict
  ): string {
    const lines: string[] = [];

    lines.push(`Comparison: ${baseline} vs ${candidate}`);
    lines.push(`Verdict: ${verdict.toUpperCase()}`);
    lines.push('');

    // Significant changes
    const significant = metrics.filter(m => m.significant);
    if (significant.length > 0) {
      lines.push('Significant changes:');
      for (const m of significant) {
        const symbol = m.improvement ? '✅' : m.regression ? '❌' : '➖';
        const direction = m.diff_percent > 0 ? '+' : '';
        lines.push(`  ${symbol} ${m.name}: ${direction}${m.diff_percent.toFixed(2)}%`);
      }
    } else {
      lines.push('No significant changes detected.');
    }

    return lines.join('\n');
  }

  /**
   * Calculate fitness score (GVCS-inspired)
   */
  private calculateFitness(result: BenchmarkResult): number {
    // Weight different factors
    const latencyScore = 1000 / result.avg_time_ms; // Higher is better
    const throughputScore = result.ops_per_sec / 1000; // Higher is better
    const consistencyScore = 1000 / (result.max_time_ms - result.min_time_ms + 1); // Higher is better

    // Weighted sum (similar to GVCS fitness)
    return (
      latencyScore * 0.4 +
      throughputScore * 0.4 +
      consistencyScore * 0.2
    );
  }
}

// ============================================================================
// Regression Detector
// ============================================================================

export class RegressionDetector {
  private history: Map<string, BenchmarkResult[]> = new Map();
  private threshold: number;

  constructor(threshold: number = 10) {
    this.threshold = threshold; // % threshold
  }

  /**
   * Add benchmark result to history (O(1))
   */
  addResult(name: string, result: BenchmarkResult): void {
    if (!this.history.has(name)) {
      this.history.set(name, []);
    }
    this.history.get(name)!.push(result);
  }

  /**
   * Detect if latest result is a regression
   */
  detectRegression(name: string): boolean {
    const results = this.history.get(name);
    if (!results || results.length < 2) {
      return false; // Need at least 2 results
    }

    const latest = results[results.length - 1];
    const baseline = this.calculateBaseline(results.slice(0, -1));

    // Compare avg_time_ms
    const diffPercent = ((latest.avg_time_ms - baseline.avg_time_ms) / baseline.avg_time_ms) * 100;

    return diffPercent > this.threshold;
  }

  /**
   * Get regression report
   */
  getReport(name: string): string | null {
    if (!this.detectRegression(name)) {
      return null;
    }

    const results = this.history.get(name)!;
    const latest = results[results.length - 1];
    const baseline = this.calculateBaseline(results.slice(0, -1));

    const diffPercent = ((latest.avg_time_ms - baseline.avg_time_ms) / baseline.avg_time_ms) * 100;

    return `⚠️ REGRESSION DETECTED: ${name}
  Baseline: ${baseline.avg_time_ms.toFixed(3)}ms
  Latest: ${latest.avg_time_ms.toFixed(3)}ms
  Diff: +${diffPercent.toFixed(2)}% (threshold: ${this.threshold}%)`;
  }

  /**
   * Calculate baseline from historical results
   */
  private calculateBaseline(results: BenchmarkResult[]): BenchmarkResult {
    const avgTimes = results.map(r => r.avg_time_ms);
    const minTimes = results.map(r => r.min_time_ms);
    const maxTimes = results.map(r => r.max_time_ms);
    const opsSec = results.map(r => r.ops_per_sec);

    const mean = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;

    return {
      name: 'baseline',
      iterations: results[0].iterations,
      total_time_ms: mean(results.map(r => r.total_time_ms)),
      avg_time_ms: mean(avgTimes),
      min_time_ms: Math.min(...minTimes),
      max_time_ms: Math.max(...maxTimes),
      median_time_ms: mean(results.map(r => r.median_time_ms)),
      p95_time_ms: mean(results.map(r => r.p95_time_ms)),
      p99_time_ms: mean(results.map(r => r.p99_time_ms)),
      ops_per_sec: mean(opsSec)
    };
  }

  /**
   * Clear history
   */
  clear(name?: string): void {
    if (name) {
      this.history.delete(name);
    } else {
      this.history.clear();
    }
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create comparator
 */
export function createComparator(config?: ComparisonConfig): BenchmarkComparator {
  return new BenchmarkComparator(config);
}

/**
 * Create regression detector
 */
export function createRegressionDetector(threshold?: number): RegressionDetector {
  return new RegressionDetector(threshold);
}

/**
 * Quick compare two results
 */
export function compare(
  baselineName: string,
  baseline: BenchmarkResult,
  candidateName: string,
  candidate: BenchmarkResult
): ComparisonResult {
  const comparator = new BenchmarkComparator();
  return comparator.compare(baselineName, baseline, candidateName, candidate);
}
