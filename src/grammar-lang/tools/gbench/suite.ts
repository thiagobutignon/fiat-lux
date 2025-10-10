/**
 * GBench - Benchmark Suite
 *
 * O(1) benchmarking system for Grammar Language tools.
 * Inspired by GVCS performance metrics and genetic fitness tracking.
 *
 * Features:
 * - O(1) hash-based benchmark registration
 * - Automated performance comparison
 * - Memory usage tracking
 * - Percentile calculations
 * - Warmup rounds for JIT optimization
 */

// ============================================================================
// Types
// ============================================================================

export interface BenchmarkConfig {
  name: string;
  iterations: number;
  warmup_iterations?: number;
  timeout_ms?: number;
  track_memory?: boolean;
}

export interface BenchmarkResult {
  name: string;
  iterations: number;
  total_time_ms: number;
  avg_time_ms: number;
  min_time_ms: number;
  max_time_ms: number;
  median_time_ms: number;
  p95_time_ms: number;
  p99_time_ms: number;
  ops_per_sec: number;
  memory_used_mb?: number;
  memory_peak_mb?: number;
  warmup_time_ms?: number;
}

export interface BenchmarkSuite {
  name: string;
  benchmarks: Map<string, BenchmarkFn>;
  results: Map<string, BenchmarkResult>;
}

export type BenchmarkFn = () => void | Promise<void>;

// ============================================================================
// Benchmark Suite
// ============================================================================

export class GBenchSuite {
  private name: string;
  private benchmarks: Map<string, BenchmarkFn>;
  private results: Map<string, BenchmarkResult>;

  constructor(name: string) {
    this.name = name;
    this.benchmarks = new Map(); // O(1) lookup
    this.results = new Map(); // O(1) lookup
  }

  /**
   * Register benchmark (O(1))
   */
  add(name: string, fn: BenchmarkFn): this {
    if (this.benchmarks.has(name)) {
      throw new Error(`Benchmark "${name}" already registered`);
    }
    this.benchmarks.set(name, fn);
    return this;
  }

  /**
   * Run single benchmark
   */
  async runBenchmark(
    name: string,
    config: Partial<BenchmarkConfig> = {}
  ): Promise<BenchmarkResult> {
    const fn = this.benchmarks.get(name);
    if (!fn) {
      throw new Error(`Benchmark "${name}" not found`);
    }

    const fullConfig: BenchmarkConfig = {
      name,
      iterations: config.iterations ?? 1000,
      warmup_iterations: config.warmup_iterations ?? 100,
      timeout_ms: config.timeout_ms ?? 30000,
      track_memory: config.track_memory ?? false
    };

    // Warmup (JIT optimization)
    const warmupStart = performance.now();
    for (let i = 0; i < fullConfig.warmup_iterations!; i++) {
      await fn();
    }
    const warmupTime = performance.now() - warmupStart;

    // Memory baseline
    const memoryBefore = fullConfig.track_memory ? this.getMemoryUsage() : undefined;
    let memoryPeak = memoryBefore;

    // Run benchmark iterations
    const timings: number[] = [];
    const start = performance.now();

    for (let i = 0; i < fullConfig.iterations; i++) {
      const iterStart = performance.now();
      await fn();
      const iterEnd = performance.now();
      timings.push(iterEnd - iterStart);

      // Track peak memory
      if (fullConfig.track_memory) {
        const current = this.getMemoryUsage();
        if (current > memoryPeak!) {
          memoryPeak = current;
        }
      }

      // Timeout check
      if (performance.now() - start > fullConfig.timeout_ms!) {
        throw new Error(`Benchmark "${name}" exceeded timeout of ${fullConfig.timeout_ms}ms`);
      }
    }

    const totalTime = performance.now() - start;

    // Memory after
    const memoryAfter = fullConfig.track_memory ? this.getMemoryUsage() : undefined;

    // Calculate statistics
    timings.sort((a, b) => a - b); // O(n log n) but only once

    const result: BenchmarkResult = {
      name,
      iterations: fullConfig.iterations,
      total_time_ms: totalTime,
      avg_time_ms: totalTime / fullConfig.iterations,
      min_time_ms: timings[0],
      max_time_ms: timings[timings.length - 1],
      median_time_ms: this.percentile(timings, 50),
      p95_time_ms: this.percentile(timings, 95),
      p99_time_ms: this.percentile(timings, 99),
      ops_per_sec: (fullConfig.iterations / totalTime) * 1000,
      warmup_time_ms: warmupTime
    };

    if (fullConfig.track_memory && memoryBefore !== undefined && memoryAfter !== undefined) {
      result.memory_used_mb = memoryAfter - memoryBefore;
      result.memory_peak_mb = memoryPeak! - memoryBefore;
    }

    this.results.set(name, result);
    return result;
  }

  /**
   * Run all benchmarks
   */
  async runAll(config: Partial<BenchmarkConfig> = {}): Promise<Map<string, BenchmarkResult>> {
    this.results.clear();

    for (const [name] of this.benchmarks) {
      console.log(`📊 Running benchmark: ${name}...`);
      await this.runBenchmark(name, config);
    }

    return this.results;
  }

  /**
   * Get result for specific benchmark (O(1))
   */
  getResult(name: string): BenchmarkResult | undefined {
    return this.results.get(name);
  }

  /**
   * Get all results
   */
  getAllResults(): BenchmarkResult[] {
    return Array.from(this.results.values());
  }

  /**
   * Get suite name
   */
  getName(): string {
    return this.name;
  }

  /**
   * Calculate percentile from sorted array
   */
  private percentile(sortedArray: number[], p: number): number {
    const index = Math.ceil((p / 100) * sortedArray.length) - 1;
    return sortedArray[index];
  }

  /**
   * Get memory usage in MB
   */
  private getMemoryUsage(): number {
    if (typeof process !== 'undefined' && process.memoryUsage) {
      return process.memoryUsage().heapUsed / 1024 / 1024;
    }
    return 0;
  }

  /**
   * Clear all results
   */
  clear(): void {
    this.results.clear();
  }

  /**
   * Remove benchmark (O(1))
   */
  remove(name: string): boolean {
    return this.benchmarks.delete(name);
  }

  /**
   * Check if benchmark exists (O(1))
   */
  has(name: string): boolean {
    return this.benchmarks.has(name);
  }

  /**
   * Get benchmark count
   */
  size(): number {
    return this.benchmarks.size;
  }
}

// ============================================================================
// Suite Registry (Global)
// ============================================================================

export class BenchmarkRegistry {
  private static suites: Map<string, GBenchSuite> = new Map();

  /**
   * Register suite (O(1))
   */
  static register(suite: GBenchSuite): void {
    const name = suite.getName();
    if (this.suites.has(name)) {
      throw new Error(`Suite "${name}" already registered`);
    }
    this.suites.set(name, suite);
  }

  /**
   * Get suite (O(1))
   */
  static get(name: string): GBenchSuite | undefined {
    return this.suites.get(name);
  }

  /**
   * Get all suites
   */
  static getAll(): GBenchSuite[] {
    return Array.from(this.suites.values());
  }

  /**
   * Run suite by name
   */
  static async run(name: string, config?: Partial<BenchmarkConfig>): Promise<Map<string, BenchmarkResult>> {
    const suite = this.suites.get(name);
    if (!suite) {
      throw new Error(`Suite "${name}" not found`);
    }
    return suite.runAll(config);
  }

  /**
   * Run all suites
   */
  static async runAll(config?: Partial<BenchmarkConfig>): Promise<Map<string, Map<string, BenchmarkResult>>> {
    const results = new Map<string, Map<string, BenchmarkResult>>();

    for (const [name, suite] of this.suites) {
      console.log(`\n🏃 Running suite: ${name}\n`);
      const suiteResults = await suite.runAll(config);
      results.set(name, suiteResults);
    }

    return results;
  }

  /**
   * Clear all suites
   */
  static clear(): void {
    this.suites.clear();
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create new benchmark suite
 */
export function createSuite(name: string): GBenchSuite {
  return new GBenchSuite(name);
}

/**
 * Create and register suite
 */
export function suite(name: string): GBenchSuite {
  const s = new GBenchSuite(name);
  BenchmarkRegistry.register(s);
  return s;
}

// ============================================================================
// Utility: Quick Benchmark
// ============================================================================

/**
 * Quick benchmark a function
 */
export async function quickBench(
  name: string,
  fn: BenchmarkFn,
  iterations: number = 1000
): Promise<BenchmarkResult> {
  const suite = new GBenchSuite('quick');
  suite.add(name, fn);
  return suite.runBenchmark(name, { iterations });
}
