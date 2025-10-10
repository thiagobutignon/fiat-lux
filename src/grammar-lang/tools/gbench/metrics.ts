/**
 * GBench - Performance Metrics
 *
 * Advanced performance metrics collection and analysis.
 * Tracks CPU, memory, I/O, and custom metrics with O(1) complexity.
 *
 * Features:
 * - Real-time metrics collection
 * - Statistical analysis (mean, stddev, variance)
 * - Percentile calculations (p50, p95, p99)
 * - Memory profiling
 * - GC impact analysis
 * - Throughput calculation
 */

import { BenchmarkResult } from './suite';

// ============================================================================
// Types
// ============================================================================

export interface PerformanceMetrics {
  // Timing
  duration_ms: number;
  avg_time_ms: number;
  min_time_ms: number;
  max_time_ms: number;
  median_time_ms: number;
  p95_time_ms: number;
  p99_time_ms: number;
  stddev_ms: number;
  variance_ms: number;

  // Throughput
  ops_per_sec: number;
  items_per_sec?: number;
  bytes_per_sec?: number;

  // Memory
  memory_used_mb?: number;
  memory_peak_mb?: number;
  gc_count?: number;
  gc_duration_ms?: number;

  // I/O (if tracked)
  reads?: number;
  writes?: number;
  bytes_read?: number;
  bytes_written?: number;

  // CPU
  cpu_percent?: number;
}

export interface MetricPoint {
  timestamp: number;
  value: number;
  label?: string;
}

export interface TimeSeries {
  name: string;
  points: MetricPoint[];
  unit: string;
}

// ============================================================================
// Metrics Collector
// ============================================================================

export class MetricsCollector {
  private timings: number[] = [];
  private memorySnapshots: number[] = [];
  private gcCount: number = 0;
  private gcDuration: number = 0;
  private startTime: number = 0;
  private endTime: number = 0;

  // Custom metrics
  private customMetrics: Map<string, number[]> = new Map();

  constructor() {
    this.setupGCTracking();
  }

  /**
   * Start collection
   */
  start(): void {
    this.reset();
    this.startTime = performance.now();
    this.captureMemory();
  }

  /**
   * End collection
   */
  end(): void {
    this.endTime = performance.now();
    this.captureMemory();
  }

  /**
   * Record timing (O(1) amortized)
   */
  recordTiming(ms: number): void {
    this.timings.push(ms);
  }

  /**
   * Record custom metric (O(1) amortized)
   */
  recordMetric(name: string, value: number): void {
    if (!this.customMetrics.has(name)) {
      this.customMetrics.set(name, []);
    }
    this.customMetrics.get(name)!.push(value);
  }

  /**
   * Capture memory snapshot
   */
  private captureMemory(): void {
    if (typeof process !== 'undefined' && process.memoryUsage) {
      const usage = process.memoryUsage();
      this.memorySnapshots.push(usage.heapUsed / 1024 / 1024); // MB
    }
  }

  /**
   * Setup GC tracking (if available)
   */
  private setupGCTracking(): void {
    if (typeof global !== 'undefined' && (global as any).gc) {
      // GC tracking available in Node with --expose-gc flag
      const originalGC = (global as any).gc;
      (global as any).gc = () => {
        const start = performance.now();
        originalGC();
        this.gcDuration += performance.now() - start;
        this.gcCount++;
      };
    }
  }

  /**
   * Calculate all metrics
   */
  calculate(): PerformanceMetrics {
    if (this.timings.length === 0) {
      throw new Error('No timings recorded');
    }

    const sorted = [...this.timings].sort((a, b) => a - b);
    const duration = this.endTime - this.startTime;

    const metrics: PerformanceMetrics = {
      duration_ms: duration,
      avg_time_ms: this.mean(this.timings),
      min_time_ms: sorted[0],
      max_time_ms: sorted[sorted.length - 1],
      median_time_ms: this.percentile(sorted, 50),
      p95_time_ms: this.percentile(sorted, 95),
      p99_time_ms: this.percentile(sorted, 99),
      stddev_ms: this.stddev(this.timings),
      variance_ms: this.variance(this.timings),
      ops_per_sec: (this.timings.length / duration) * 1000
    };

    // Memory metrics
    if (this.memorySnapshots.length >= 2) {
      const memStart = this.memorySnapshots[0];
      const memEnd = this.memorySnapshots[this.memorySnapshots.length - 1];
      const memPeak = Math.max(...this.memorySnapshots);

      metrics.memory_used_mb = memEnd - memStart;
      metrics.memory_peak_mb = memPeak - memStart;
    }

    // GC metrics
    if (this.gcCount > 0) {
      metrics.gc_count = this.gcCount;
      metrics.gc_duration_ms = this.gcDuration;
    }

    return metrics;
  }

  /**
   * Get custom metric stats
   */
  getCustomMetric(name: string): { avg: number; min: number; max: number } | undefined {
    const values = this.customMetrics.get(name);
    if (!values || values.length === 0) {
      return undefined;
    }

    return {
      avg: this.mean(values),
      min: Math.min(...values),
      max: Math.max(...values)
    };
  }

  /**
   * Reset all metrics
   */
  reset(): void {
    this.timings = [];
    this.memorySnapshots = [];
    this.gcCount = 0;
    this.gcDuration = 0;
    this.customMetrics.clear();
  }

  // Statistical functions

  private mean(values: number[]): number {
    return values.reduce((a, b) => a + b, 0) / values.length;
  }

  private variance(values: number[]): number {
    const avg = this.mean(values);
    const squaredDiffs = values.map(v => Math.pow(v - avg, 2));
    return this.mean(squaredDiffs);
  }

  private stddev(values: number[]): number {
    return Math.sqrt(this.variance(values));
  }

  private percentile(sortedValues: number[], p: number): number {
    const index = Math.ceil((p / 100) * sortedValues.length) - 1;
    return sortedValues[Math.max(0, index)];
  }
}

// ============================================================================
// Metrics Aggregator
// ============================================================================

export class MetricsAggregator {
  private metrics: Map<string, PerformanceMetrics[]> = new Map();

  /**
   * Add metric result (O(1))
   */
  add(name: string, metrics: PerformanceMetrics): void {
    if (!this.metrics.has(name)) {
      this.metrics.set(name, []);
    }
    this.metrics.get(name)!.push(metrics);
  }

  /**
   * Get aggregated stats for metric
   */
  aggregate(name: string): PerformanceMetrics | undefined {
    const metricsList = this.metrics.get(name);
    if (!metricsList || metricsList.length === 0) {
      return undefined;
    }

    // Aggregate across all runs
    const avgTimes = metricsList.map(m => m.avg_time_ms);
    const minTimes = metricsList.map(m => m.min_time_ms);
    const maxTimes = metricsList.map(m => m.max_time_ms);
    const opsSec = metricsList.map(m => m.ops_per_sec);

    return {
      duration_ms: this.sum(metricsList.map(m => m.duration_ms)),
      avg_time_ms: this.mean(avgTimes),
      min_time_ms: Math.min(...minTimes),
      max_time_ms: Math.max(...maxTimes),
      median_time_ms: this.mean(metricsList.map(m => m.median_time_ms)),
      p95_time_ms: this.mean(metricsList.map(m => m.p95_time_ms)),
      p99_time_ms: this.mean(metricsList.map(m => m.p99_time_ms)),
      stddev_ms: this.stddev(avgTimes),
      variance_ms: this.variance(avgTimes),
      ops_per_sec: this.mean(opsSec)
    };
  }

  /**
   * Get all aggregated metrics
   */
  aggregateAll(): Map<string, PerformanceMetrics> {
    const result = new Map<string, PerformanceMetrics>();

    for (const [name] of this.metrics) {
      const aggregated = this.aggregate(name);
      if (aggregated) {
        result.set(name, aggregated);
      }
    }

    return result;
  }

  /**
   * Clear all metrics
   */
  clear(): void {
    this.metrics.clear();
  }

  // Utility functions

  private sum(values: number[]): number {
    return values.reduce((a, b) => a + b, 0);
  }

  private mean(values: number[]): number {
    return this.sum(values) / values.length;
  }

  private variance(values: number[]): number {
    const avg = this.mean(values);
    const squaredDiffs = values.map(v => Math.pow(v - avg, 2));
    return this.mean(squaredDiffs);
  }

  private stddev(values: number[]): number {
    return Math.sqrt(this.variance(values));
  }
}

// ============================================================================
// Time Series Tracker
// ============================================================================

export class TimeSeriesTracker {
  private series: Map<string, TimeSeries> = new Map();

  /**
   * Create new time series (O(1))
   */
  create(name: string, unit: string): void {
    if (this.series.has(name)) {
      throw new Error(`Time series "${name}" already exists`);
    }
    this.series.set(name, {
      name,
      points: [],
      unit
    });
  }

  /**
   * Add data point (O(1))
   */
  record(name: string, value: number, label?: string): void {
    const ts = this.series.get(name);
    if (!ts) {
      throw new Error(`Time series "${name}" not found`);
    }

    ts.points.push({
      timestamp: Date.now(),
      value,
      label
    });
  }

  /**
   * Get time series (O(1))
   */
  get(name: string): TimeSeries | undefined {
    return this.series.get(name);
  }

  /**
   * Get all series
   */
  getAll(): TimeSeries[] {
    return Array.from(this.series.values());
  }

  /**
   * Calculate trend (positive = increasing, negative = decreasing)
   */
  getTrend(name: string): number {
    const ts = this.series.get(name);
    if (!ts || ts.points.length < 2) {
      return 0;
    }

    const first = ts.points[0].value;
    const last = ts.points[ts.points.length - 1].value;

    return ((last - first) / first) * 100; // Percentage change
  }

  /**
   * Clear time series
   */
  clear(name?: string): void {
    if (name) {
      this.series.delete(name);
    } else {
      this.series.clear();
    }
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create metrics collector
 */
export function createCollector(): MetricsCollector {
  return new MetricsCollector();
}

/**
 * Create metrics aggregator
 */
export function createAggregator(): MetricsAggregator {
  return new MetricsAggregator();
}

/**
 * Create time series tracker
 */
export function createTimeSeries(): TimeSeriesTracker {
  return new TimeSeriesTracker();
}

// ============================================================================
// Utility: Convert BenchmarkResult to PerformanceMetrics
// ============================================================================

export function fromBenchmarkResult(result: BenchmarkResult): PerformanceMetrics {
  return {
    duration_ms: result.total_time_ms,
    avg_time_ms: result.avg_time_ms,
    min_time_ms: result.min_time_ms,
    max_time_ms: result.max_time_ms,
    median_time_ms: result.median_time_ms,
    p95_time_ms: result.p95_time_ms,
    p99_time_ms: result.p99_time_ms,
    stddev_ms: 0, // Not calculated in BenchmarkResult
    variance_ms: 0, // Not calculated in BenchmarkResult
    ops_per_sec: result.ops_per_sec,
    memory_used_mb: result.memory_used_mb,
    memory_peak_mb: result.memory_peak_mb
  };
}
