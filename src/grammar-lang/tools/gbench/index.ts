/**
 * GBench - Grammar Language Benchmarking Tool
 *
 * O(1) performance benchmarking system for Grammar Language.
 *
 * Features:
 * - Benchmark suite management
 * - Performance metrics collection
 * - Result comparison
 * - Multi-format reporting (console, JSON, CSV, MD, HTML)
 * - Regression detection
 * - Statistical analysis
 *
 * Usage:
 * ```typescript
 * import { suite, report, compare } from '@/grammar-lang/tools/gbench';
 *
 * // Create benchmark suite
 * const s = suite('my-benchmarks');
 *
 * s.add('hash-lookup', () => {
 *   const map = new Map();
 *   map.set('key', 'value');
 *   map.get('key');
 * });
 *
 * // Run benchmarks
 * await s.runAll({ iterations: 10000 });
 *
 * // Get results
 * const result = s.getResult('hash-lookup');
 * report(result!);
 * ```
 */

// Suite
export {
  GBenchSuite,
  BenchmarkRegistry,
  createSuite,
  suite,
  quickBench,
  type BenchmarkConfig,
  type BenchmarkResult,
  type BenchmarkSuite,
  type BenchmarkFn
} from './suite';

// Metrics
export {
  MetricsCollector,
  MetricsAggregator,
  TimeSeriesTracker,
  createCollector,
  createAggregator,
  createTimeSeries,
  fromBenchmarkResult,
  type PerformanceMetrics,
  type MetricPoint,
  type TimeSeries
} from './metrics';

// Compare
export {
  BenchmarkComparator,
  RegressionDetector,
  createComparator,
  createRegressionDetector,
  compare,
  type ComparisonResult,
  type MetricComparison,
  type ComparisonVerdict,
  type ComparisonConfig
} from './compare';

// Report
export {
  ReportGenerator,
  BenchmarkExporter,
  createReporter,
  report,
  reportComparison,
  type ReportFormat,
  type ReportConfig
} from './report';
