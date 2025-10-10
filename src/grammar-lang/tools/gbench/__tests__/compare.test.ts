/**
 * GBench Compare - Unit Tests
 */

import { BenchmarkComparator, RegressionDetector, compare, createRegressionDetector } from '../compare';
import { BenchmarkResult } from '../suite';

describe('BenchmarkComparator', () => {
  let comparator: BenchmarkComparator;
  let baseline: BenchmarkResult;
  let candidate: BenchmarkResult;

  beforeEach(() => {
    comparator = new BenchmarkComparator();

    baseline = {
      name: 'baseline',
      iterations: 1000,
      total_time_ms: 100,
      warmup_time_ms: 10,
      metrics: {
        avg_time_ms: 0.1,
        min_time_ms: 0.05,
        max_time_ms: 0.5,
        median_time_ms: 0.09,
        p50_time_ms: 0.09,
        p95_time_ms: 0.2,
        p99_time_ms: 0.3,
        stddev_ms: 0.05,
        variance_ms: 0.0025,
        ops_per_sec: 10000
      },
      timestamp: Date.now()
    };

    candidate = {
      name: 'candidate',
      iterations: 1000,
      total_time_ms: 50,
      warmup_time_ms: 5,
      metrics: {
        avg_time_ms: 0.05,  // 50% faster
        min_time_ms: 0.025,
        max_time_ms: 0.25,
        median_time_ms: 0.045,
        p50_time_ms: 0.045,
        p95_time_ms: 0.1,
        p99_time_ms: 0.15,
        stddev_ms: 0.025,
        variance_ms: 0.000625,
        ops_per_sec: 20000  // 2x faster
      },
      timestamp: Date.now()
    };
  });

  describe('compare', () => {
    it('should detect improvement', () => {
      const result = comparator.compare('baseline', baseline, 'candidate', candidate);

      expect(result.verdict).toBe('improvement');
      expect(result.improvements.length).toBeGreaterThan(0);
      expect(result.regressions.length).toBe(0);
    });

    it('should detect regression', () => {
      const worse = {
        ...candidate,
        metrics: {
          ...candidate.metrics,
          avg_time_ms: 0.2,  // 2x slower
          ops_per_sec: 5000   // 2x slower
        }
      };

      const result = comparator.compare('baseline', baseline, 'worse', worse);

      expect(result.verdict).toBe('regression');
      expect(result.regressions.length).toBeGreaterThan(0);
      expect(result.improvements.length).toBe(0);
    });

    it('should detect mixed results', () => {
      const mixed = {
        ...candidate,
        metrics: {
          ...candidate.metrics,
          avg_time_ms: 0.05,    // Better
          max_time_ms: 1.0,     // Worse
          ops_per_sec: 20000    // Better
        }
      };

      const result = comparator.compare('baseline', baseline, 'mixed', mixed);

      expect(result.verdict).toBe('mixed');
      expect(result.improvements.length).toBeGreaterThan(0);
      expect(result.regressions.length).toBeGreaterThan(0);
    });

    it('should calculate percentage changes correctly', () => {
      const result = comparator.compare('baseline', baseline, 'candidate', candidate);

      const avgChange = result.metrics.find(m => m.metric === 'avg_time_ms');
      expect(avgChange?.percent_change).toBeCloseTo(-50, 0);

      const opsChange = result.metrics.find(m => m.metric === 'ops_per_sec');
      expect(opsChange?.percent_change).toBeCloseTo(100, 0);
    });
  });

  describe('selectBest', () => {
    it('should select result with best score', () => {
      const results = new Map<string, BenchmarkResult>([
        ['slow', { ...baseline, metrics: { ...baseline.metrics, avg_time_ms: 1.0, ops_per_sec: 1000 } }],
        ['fast', { ...baseline, metrics: { ...baseline.metrics, avg_time_ms: 0.01, ops_per_sec: 100000 } }],
        ['medium', baseline]
      ]);

      const best = comparator.selectBest(results);

      expect(best.name).toBe('fast');
    });

    it('should return null for empty results', () => {
      const best = comparator.selectBest(new Map());
      expect(best.name).toBe('');
      expect(best.score).toBe(0);
    });
  });

  describe('detectRegressions', () => {
    it('should detect all regressions', () => {
      const worse = {
        ...candidate,
        metrics: {
          ...candidate.metrics,
          avg_time_ms: 0.2,
          p95_time_ms: 0.4,
          ops_per_sec: 5000
        }
      };

      const regressions = comparator.detectRegressions(baseline, worse);

      expect(regressions.length).toBeGreaterThan(0);
      expect(regressions.every(r => r.is_regression)).toBe(true);
    });

    it('should return empty array for improvements', () => {
      const regressions = comparator.detectRegressions(baseline, candidate);
      expect(regressions.length).toBe(0);
    });
  });
});

describe('RegressionDetector', () => {
  let detector: RegressionDetector;
  let baselineResult: BenchmarkResult;

  beforeEach(() => {
    detector = new RegressionDetector(10); // 10% threshold

    baselineResult = {
      name: 'test',
      iterations: 1000,
      total_time_ms: 100,
      warmup_time_ms: 10,
      metrics: {
        avg_time_ms: 0.1,
        min_time_ms: 0.05,
        max_time_ms: 0.5,
        median_time_ms: 0.09,
        p50_time_ms: 0.09,
        p95_time_ms: 0.2,
        p99_time_ms: 0.3,
        stddev_ms: 0.05,
        variance_ms: 0.0025,
        ops_per_sec: 10000
      },
      timestamp: Date.now()
    };
  });

  describe('addResult', () => {
    it('should add result to history', () => {
      detector.addResult('test', baselineResult);
      expect(detector.getHistory('test')).toHaveLength(1);
    });

    it('should limit history to max size', () => {
      for (let i = 0; i < 150; i++) {
        detector.addResult('test', baselineResult);
      }

      expect(detector.getHistory('test')).toHaveLength(100);
    });
  });

  describe('detectRegression', () => {
    it('should detect regression above threshold', () => {
      detector.addResult('test', baselineResult);

      const worse = {
        ...baselineResult,
        metrics: {
          ...baselineResult.metrics,
          avg_time_ms: 0.15  // 50% slower (above 10% threshold)
        }
      };
      detector.addResult('test', worse);

      expect(detector.detectRegression('test')).toBe(true);
    });

    it('should not detect regression below threshold', () => {
      detector.addResult('test', baselineResult);

      const slightlyWorse = {
        ...baselineResult,
        metrics: {
          ...baselineResult.metrics,
          avg_time_ms: 0.105  // 5% slower (below 10% threshold)
        }
      };
      detector.addResult('test', slightlyWorse);

      expect(detector.detectRegression('test')).toBe(false);
    });

    it('should return false for insufficient history', () => {
      detector.addResult('test', baselineResult);
      expect(detector.detectRegression('test')).toBe(false);
    });

    it('should return false for non-existent benchmark', () => {
      expect(detector.detectRegression('nonexistent')).toBe(false);
    });
  });

  describe('getReport', () => {
    it('should generate report for regression', () => {
      detector.addResult('test', baselineResult);

      const worse = {
        ...baselineResult,
        metrics: {
          ...baselineResult.metrics,
          avg_time_ms: 0.2
        }
      };
      detector.addResult('test', worse);

      const report = detector.getReport('test');
      expect(report).toBeTruthy();
      expect(report).toContain('REGRESSION');
    });

    it('should return null for no regression', () => {
      detector.addResult('test', baselineResult);
      detector.addResult('test', baselineResult);

      expect(detector.getReport('test')).toBeNull();
    });
  });

  describe('clear', () => {
    it('should clear history for benchmark', () => {
      detector.addResult('test', baselineResult);
      detector.clear('test');

      expect(detector.getHistory('test')).toHaveLength(0);
    });

    it('should clear all history', () => {
      detector.addResult('test1', baselineResult);
      detector.addResult('test2', baselineResult);
      detector.clearAll();

      expect(detector.getHistory('test1')).toHaveLength(0);
      expect(detector.getHistory('test2')).toHaveLength(0);
    });
  });
});

describe('compare function', () => {
  it('should create comparison between two results', () => {
    const baseline: BenchmarkResult = {
      name: 'baseline',
      iterations: 1000,
      total_time_ms: 100,
      warmup_time_ms: 10,
      metrics: {
        avg_time_ms: 0.1,
        min_time_ms: 0.05,
        max_time_ms: 0.5,
        median_time_ms: 0.09,
        p50_time_ms: 0.09,
        p95_time_ms: 0.2,
        p99_time_ms: 0.3,
        stddev_ms: 0.05,
        variance_ms: 0.0025,
        ops_per_sec: 10000
      },
      timestamp: Date.now()
    };

    const candidate: BenchmarkResult = {
      ...baseline,
      name: 'candidate',
      metrics: {
        ...baseline.metrics,
        avg_time_ms: 0.05,
        ops_per_sec: 20000
      }
    };

    const result = compare('baseline', baseline, 'candidate', candidate);

    expect(result.baseline_name).toBe('baseline');
    expect(result.candidate_name).toBe('candidate');
    expect(result.verdict).toBe('improvement');
  });
});

describe('createRegressionDetector factory', () => {
  it('should create detector with threshold', () => {
    const detector = createRegressionDetector(15);
    expect(detector).toBeInstanceOf(RegressionDetector);
  });
});
