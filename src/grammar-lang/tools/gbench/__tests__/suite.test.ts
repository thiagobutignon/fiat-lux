/**
 * GBench Suite - Unit Tests
 */

import { GBenchSuite, BenchmarkRegistry, suite } from '../suite';

describe('GBenchSuite', () => {
  let benchSuite: GBenchSuite;

  beforeEach(() => {
    benchSuite = new GBenchSuite('test-suite');
  });

  describe('add', () => {
    it('should add benchmark function', () => {
      const fn = () => Math.sqrt(100);
      benchSuite.add('sqrt', fn);

      expect(benchSuite.has('sqrt')).toBe(true);
    });

    it('should throw on duplicate benchmark name', () => {
      const fn = () => Math.sqrt(100);
      benchSuite.add('sqrt', fn);

      expect(() => benchSuite.add('sqrt', fn)).toThrow();
    });

    it('should allow chaining', () => {
      const result = benchSuite
        .add('test1', () => {})
        .add('test2', () => {});

      expect(result).toBe(benchSuite);
    });
  });

  describe('remove', () => {
    it('should remove benchmark', () => {
      benchSuite.add('test', () => {});
      expect(benchSuite.remove('test')).toBe(true);
      expect(benchSuite.has('test')).toBe(false);
    });

    it('should return false for non-existent benchmark', () => {
      expect(benchSuite.remove('nonexistent')).toBe(false);
    });
  });

  describe('runBenchmark', () => {
    it('should run benchmark and return result', async () => {
      let counter = 0;
      benchSuite.add('counter', () => {
        counter++;
      });

      const result = await benchSuite.runBenchmark('counter', {
        iterations: 100,
        warmup_iterations: 10
      });

      expect(result.name).toBe('counter');
      expect(result.iterations).toBe(100);
      expect(result.metrics.avg_time_ms).toBeGreaterThanOrEqual(0);
      expect(counter).toBeGreaterThanOrEqual(110); // 100 + 10 warmup
    });

    it('should throw for non-existent benchmark', async () => {
      await expect(
        benchSuite.runBenchmark('nonexistent')
      ).rejects.toThrow();
    });

    it('should calculate percentiles correctly', async () => {
      benchSuite.add('test', () => {
        // Simple operation
        Math.sqrt(100);
      });

      const result = await benchSuite.runBenchmark('test', {
        iterations: 1000
      });

      expect(result.metrics.p50_time_ms).toBeDefined();
      expect(result.metrics.p95_time_ms).toBeDefined();
      expect(result.metrics.p99_time_ms).toBeDefined();
      expect(result.metrics.p95_time_ms).toBeGreaterThanOrEqual(result.metrics.p50_time_ms);
      expect(result.metrics.p99_time_ms).toBeGreaterThanOrEqual(result.metrics.p95_time_ms);
    });

    it('should track memory when enabled', async () => {
      benchSuite.add('alloc', () => {
        const arr = new Array(1000).fill(0);
      });

      const result = await benchSuite.runBenchmark('alloc', {
        iterations: 100,
        track_memory: true
      });

      expect(result.metrics.memory_used_mb).toBeGreaterThan(0);
      expect(result.metrics.memory_peak_mb).toBeGreaterThan(0);
    });
  });

  describe('runAll', () => {
    it('should run all benchmarks', async () => {
      benchSuite.add('test1', () => Math.sqrt(100));
      benchSuite.add('test2', () => Math.pow(2, 10));

      const results = await benchSuite.runAll({ iterations: 100 });

      expect(results.size).toBe(2);
      expect(results.has('test1')).toBe(true);
      expect(results.has('test2')).toBe(true);
    });

    it('should return empty map for no benchmarks', async () => {
      const results = await benchSuite.runAll();
      expect(results.size).toBe(0);
    });
  });

  describe('getResult', () => {
    it('should return cached result', async () => {
      benchSuite.add('test', () => {});
      await benchSuite.runBenchmark('test');

      const result = benchSuite.getResult('test');
      expect(result).toBeDefined();
      expect(result?.name).toBe('test');
    });

    it('should return undefined for non-run benchmark', () => {
      benchSuite.add('test', () => {});
      expect(benchSuite.getResult('test')).toBeUndefined();
    });
  });

  describe('clear', () => {
    it('should clear all benchmarks', () => {
      benchSuite.add('test1', () => {});
      benchSuite.add('test2', () => {});

      benchSuite.clear();

      expect(benchSuite.has('test1')).toBe(false);
      expect(benchSuite.has('test2')).toBe(false);
    });
  });
});

describe('BenchmarkRegistry', () => {
  beforeEach(() => {
    // Clear registry before each test
    BenchmarkRegistry.clear();
  });

  it('should register suite', () => {
    const s = new GBenchSuite('test');
    BenchmarkRegistry.register(s);

    expect(BenchmarkRegistry.get('test')).toBe(s);
  });

  it('should list all suites', () => {
    BenchmarkRegistry.register(new GBenchSuite('suite1'));
    BenchmarkRegistry.register(new GBenchSuite('suite2'));

    const suites = BenchmarkRegistry.list();
    expect(suites).toHaveLength(2);
  });

  it('should clear registry', () => {
    BenchmarkRegistry.register(new GBenchSuite('test'));
    BenchmarkRegistry.clear();

    expect(BenchmarkRegistry.list()).toHaveLength(0);
  });
});

describe('suite factory', () => {
  beforeEach(() => {
    BenchmarkRegistry.clear();
  });

  it('should create and register suite', () => {
    const s = suite('test');

    expect(s.getName()).toBe('test');
    expect(BenchmarkRegistry.get('test')).toBe(s);
  });

  it('should return existing suite', () => {
    const s1 = suite('test');
    const s2 = suite('test');

    expect(s1).toBe(s2);
  });
});
