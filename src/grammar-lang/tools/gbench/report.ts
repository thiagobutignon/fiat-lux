/**
 * GBench - Reporting System
 *
 * Generate beautiful benchmark reports in multiple formats.
 * Supports console, JSON, CSV, HTML, and Markdown output.
 *
 * Features:
 * - Multi-format output (console, JSON, CSV, MD, HTML)
 * - Color-coded performance indicators
 * - Comparison tables
 * - Regression warnings
 * - Export for CI/CD integration
 */

import { BenchmarkResult } from './suite';
import { PerformanceMetrics } from './metrics';
import { ComparisonResult, MetricComparison } from './compare';

// ============================================================================
// Types
// ============================================================================

export type ReportFormat = 'console' | 'json' | 'csv' | 'markdown' | 'html';

export interface ReportConfig {
  format: ReportFormat;
  includeDetails?: boolean;
  colorize?: boolean;
  includeChart?: boolean;
}

// ============================================================================
// Report Generator
// ============================================================================

export class ReportGenerator {
  private config: Required<ReportConfig>;

  constructor(config: Partial<ReportConfig> = {}) {
    this.config = {
      format: config.format ?? 'console',
      includeDetails: config.includeDetails ?? true,
      colorize: config.colorize ?? true,
      includeChart: config.includeChart ?? false
    };
  }

  /**
   * Generate report for single benchmark
   */
  generateSingle(result: BenchmarkResult): string {
    switch (this.config.format) {
      case 'console':
        return this.formatConsole(result);
      case 'json':
        return this.formatJSON(result);
      case 'csv':
        return this.formatCSV([result]);
      case 'markdown':
        return this.formatMarkdown(result);
      case 'html':
        return this.formatHTML(result);
      default:
        return this.formatConsole(result);
    }
  }

  /**
   * Generate report for multiple benchmarks
   */
  generateMultiple(results: BenchmarkResult[]): string {
    switch (this.config.format) {
      case 'console':
        return results.map(r => this.formatConsole(r)).join('\n\n');
      case 'json':
        return JSON.stringify(results, null, 2);
      case 'csv':
        return this.formatCSV(results);
      case 'markdown':
        return this.formatMarkdownTable(results);
      case 'html':
        return this.formatHTMLTable(results);
      default:
        return results.map(r => this.formatConsole(r)).join('\n\n');
    }
  }

  /**
   * Generate comparison report
   */
  generateComparison(comparison: ComparisonResult): string {
    switch (this.config.format) {
      case 'console':
        return this.formatComparisonConsole(comparison);
      case 'json':
        return JSON.stringify(comparison, null, 2);
      case 'markdown':
        return this.formatComparisonMarkdown(comparison);
      default:
        return this.formatComparisonConsole(comparison);
    }
  }

  // =========================================================================
  // Console Format
  // =========================================================================

  private formatConsole(result: BenchmarkResult): string {
    const lines: string[] = [];
    const { colorize } = this.config;

    lines.push(this.color(`📊 ${result.name}`, 'cyan', colorize));
    lines.push(this.separator());

    lines.push(`Iterations: ${result.iterations}`);
    lines.push(`Total Time: ${result.total_time_ms.toFixed(2)}ms`);
    lines.push(``);
    lines.push(`Avg Time:   ${this.formatTime(result.avg_time_ms, colorize)}`);
    lines.push(`Min Time:   ${this.formatTime(result.min_time_ms, colorize)}`);
    lines.push(`Max Time:   ${this.formatTime(result.max_time_ms, colorize)}`);
    lines.push(`Median:     ${this.formatTime(result.median_time_ms, colorize)}`);
    lines.push(`P95:        ${this.formatTime(result.p95_time_ms, colorize)}`);
    lines.push(`P99:        ${this.formatTime(result.p99_time_ms, colorize)}`);
    lines.push(``);
    lines.push(`Throughput: ${this.formatOps(result.ops_per_sec, colorize)}`);

    if (result.memory_used_mb !== undefined) {
      lines.push(``);
      lines.push(`Memory:     ${result.memory_used_mb.toFixed(2)} MB used`);
      if (result.memory_peak_mb !== undefined) {
        lines.push(`Peak:       ${result.memory_peak_mb.toFixed(2)} MB`);
      }
    }

    if (result.warmup_time_ms !== undefined) {
      lines.push(``);
      lines.push(`Warmup:     ${result.warmup_time_ms.toFixed(2)}ms`);
    }

    lines.push(this.separator());

    return lines.join('\n');
  }

  private formatComparisonConsole(comparison: ComparisonResult): string {
    const lines: string[] = [];
    const { colorize } = this.config;

    lines.push(this.color(`🔍 Comparison: ${comparison.baseline} vs ${comparison.candidate}`, 'cyan', colorize));
    lines.push(this.separator());

    // Verdict
    const verdictColor = comparison.verdict === 'improvement' ? 'green'
      : comparison.verdict === 'regression' ? 'red'
      : 'yellow';

    lines.push(`Verdict: ${this.color(comparison.verdict.toUpperCase(), verdictColor, colorize)}`);
    lines.push(``);

    // Metrics
    for (const metric of comparison.metrics) {
      const symbol = metric.improvement ? '✅' : metric.regression ? '❌' : '➖';
      const color = metric.improvement ? 'green' : metric.regression ? 'red' : 'white';
      const sign = metric.diff_percent > 0 ? '+' : '';

      lines.push(this.color(
        `${symbol} ${metric.name}: ${sign}${metric.diff_percent.toFixed(2)}%`,
        color,
        colorize
      ));
    }

    lines.push(this.separator());

    return lines.join('\n');
  }

  // =========================================================================
  // JSON Format
  // =========================================================================

  private formatJSON(result: BenchmarkResult): string {
    return JSON.stringify(result, null, 2);
  }

  // =========================================================================
  // CSV Format
  // =========================================================================

  private formatCSV(results: BenchmarkResult[]): string {
    const headers = [
      'name',
      'iterations',
      'avg_time_ms',
      'min_time_ms',
      'max_time_ms',
      'median_time_ms',
      'p95_time_ms',
      'p99_time_ms',
      'ops_per_sec',
      'memory_used_mb'
    ];

    const lines: string[] = [headers.join(',')];

    for (const result of results) {
      const row = [
        result.name,
        result.iterations.toString(),
        result.avg_time_ms.toFixed(3),
        result.min_time_ms.toFixed(3),
        result.max_time_ms.toFixed(3),
        result.median_time_ms.toFixed(3),
        result.p95_time_ms.toFixed(3),
        result.p99_time_ms.toFixed(3),
        result.ops_per_sec.toFixed(2),
        result.memory_used_mb?.toFixed(2) ?? ''
      ];

      lines.push(row.join(','));
    }

    return lines.join('\n');
  }

  // =========================================================================
  // Markdown Format
  // =========================================================================

  private formatMarkdown(result: BenchmarkResult): string {
    const lines: string[] = [];

    lines.push(`# ${result.name}`);
    lines.push(``);
    lines.push(`| Metric | Value |`);
    lines.push(`|--------|-------|`);
    lines.push(`| Iterations | ${result.iterations} |`);
    lines.push(`| Avg Time | ${result.avg_time_ms.toFixed(3)}ms |`);
    lines.push(`| Min Time | ${result.min_time_ms.toFixed(3)}ms |`);
    lines.push(`| Max Time | ${result.max_time_ms.toFixed(3)}ms |`);
    lines.push(`| P95 | ${result.p95_time_ms.toFixed(3)}ms |`);
    lines.push(`| P99 | ${result.p99_time_ms.toFixed(3)}ms |`);
    lines.push(`| Ops/sec | ${result.ops_per_sec.toFixed(2)} |`);

    return lines.join('\n');
  }

  private formatMarkdownTable(results: BenchmarkResult[]): string {
    const lines: string[] = [];

    lines.push(`| Name | Avg Time | Min | Max | P95 | Ops/sec |`);
    lines.push(`|------|----------|-----|-----|-----|---------|`);

    for (const r of results) {
      lines.push(
        `| ${r.name} | ${r.avg_time_ms.toFixed(2)}ms | ${r.min_time_ms.toFixed(2)}ms | ${r.max_time_ms.toFixed(2)}ms | ${r.p95_time_ms.toFixed(2)}ms | ${r.ops_per_sec.toFixed(0)} |`
      );
    }

    return lines.join('\n');
  }

  private formatComparisonMarkdown(comparison: ComparisonResult): string {
    const lines: string[] = [];

    lines.push(`# Comparison: ${comparison.baseline} vs ${comparison.candidate}`);
    lines.push(``);
    lines.push(`**Verdict:** ${comparison.verdict.toUpperCase()}`);
    lines.push(``);
    lines.push(`| Metric | Baseline | Candidate | Diff | Status |`);
    lines.push(`|--------|----------|-----------|------|--------|`);

    for (const m of comparison.metrics) {
      const status = m.improvement ? '✅ Better' : m.regression ? '❌ Worse' : '➖ Same';
      const sign = m.diff_percent > 0 ? '+' : '';

      lines.push(
        `| ${m.name} | ${m.baseline_value.toFixed(2)} | ${m.candidate_value.toFixed(2)} | ${sign}${m.diff_percent.toFixed(1)}% | ${status} |`
      );
    }

    return lines.join('\n');
  }

  // =========================================================================
  // HTML Format
  // =========================================================================

  private formatHTML(result: BenchmarkResult): string {
    return `
<div class="benchmark-result">
  <h3>${result.name}</h3>
  <table>
    <tr><td>Iterations</td><td>${result.iterations}</td></tr>
    <tr><td>Avg Time</td><td>${result.avg_time_ms.toFixed(3)}ms</td></tr>
    <tr><td>Min Time</td><td>${result.min_time_ms.toFixed(3)}ms</td></tr>
    <tr><td>Max Time</td><td>${result.max_time_ms.toFixed(3)}ms</td></tr>
    <tr><td>P95</td><td>${result.p95_time_ms.toFixed(3)}ms</td></tr>
    <tr><td>Ops/sec</td><td>${result.ops_per_sec.toFixed(2)}</td></tr>
  </table>
</div>
    `.trim();
  }

  private formatHTMLTable(results: BenchmarkResult[]): string {
    const rows = results.map(r => `
      <tr>
        <td>${r.name}</td>
        <td>${r.avg_time_ms.toFixed(2)}ms</td>
        <td>${r.min_time_ms.toFixed(2)}ms</td>
        <td>${r.max_time_ms.toFixed(2)}ms</td>
        <td>${r.p95_time_ms.toFixed(2)}ms</td>
        <td>${r.ops_per_sec.toFixed(0)}</td>
      </tr>
    `).join('');

    return `
<table class="benchmark-table">
  <thead>
    <tr>
      <th>Name</th>
      <th>Avg Time</th>
      <th>Min</th>
      <th>Max</th>
      <th>P95</th>
      <th>Ops/sec</th>
    </tr>
  </thead>
  <tbody>
    ${rows}
  </tbody>
</table>
    `.trim();
  }

  // =========================================================================
  // Utilities
  // =========================================================================

  private formatTime(ms: number, colorize: boolean): string {
    const formatted = `${ms.toFixed(3)}ms`;

    if (!colorize) return formatted;

    // Color based on performance
    if (ms < 1) return this.color(formatted, 'green', true);
    if (ms < 10) return this.color(formatted, 'yellow', true);
    return this.color(formatted, 'red', true);
  }

  private formatOps(ops: number, colorize: boolean): string {
    const formatted = `${ops.toFixed(2)} ops/sec`;

    if (!colorize) return formatted;

    // Color based on throughput
    if (ops > 10000) return this.color(formatted, 'green', true);
    if (ops > 1000) return this.color(formatted, 'yellow', true);
    return this.color(formatted, 'red', true);
  }

  private color(text: string, color: string, enabled: boolean): string {
    if (!enabled) return text;

    const colors: Record<string, string> = {
      red: '\x1b[31m',
      green: '\x1b[32m',
      yellow: '\x1b[33m',
      cyan: '\x1b[36m',
      white: '\x1b[37m'
    };

    const reset = '\x1b[0m';
    return `${colors[color] || ''}${text}${reset}`;
  }

  private separator(): string {
    return '─'.repeat(60);
  }
}

// ============================================================================
// Export Utilities
// ============================================================================

export class BenchmarkExporter {
  /**
   * Export to file
   */
  static export(results: BenchmarkResult[], format: ReportFormat, filepath: string): void {
    const generator = new ReportGenerator({ format });
    const content = generator.generateMultiple(results);

    // In a real implementation, would write to file
    // fs.writeFileSync(filepath, content);
    console.log(`📝 Would export to: ${filepath}`);
    console.log(content);
  }

  /**
   * Export comparison
   */
  static exportComparison(comparison: ComparisonResult, format: ReportFormat, filepath: string): void {
    const generator = new ReportGenerator({ format });
    const content = generator.generateComparison(comparison);

    console.log(`📝 Would export to: ${filepath}`);
    console.log(content);
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create report generator
 */
export function createReporter(config?: Partial<ReportConfig>): ReportGenerator {
  return new ReportGenerator(config);
}

/**
 * Quick console report
 */
export function report(result: BenchmarkResult): void {
  const generator = new ReportGenerator({ format: 'console', colorize: true });
  console.log(generator.generateSingle(result));
}

/**
 * Quick comparison report
 */
export function reportComparison(comparison: ComparisonResult): void {
  const generator = new ReportGenerator({ format: 'console', colorize: true });
  console.log(generator.generateComparison(comparison));
}
