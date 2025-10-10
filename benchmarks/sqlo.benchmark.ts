/**
 * SQLO Database Performance Benchmarks
 *
 * Validates O(1) performance guarantees:
 * - Database load: <100ms
 * - Query: <1ms
 * - Write: <10ms
 * - Scaling: O(1) across dataset sizes
 *
 * Run: ts-node benchmarks/sqlo.benchmark.ts
 */

import { SqloDatabase, MemoryType, Episode } from '../src/grammar-lang/database/sqlo';
import * as fs from 'fs';

// ============================================================================
// Configuration
// ============================================================================

const BENCHMARK_DB_DIR = 'benchmark_sqlo_db';
const DATASET_SIZES = [100, 500, 1000, 2000];
const ITERATIONS_PER_SIZE = 50;

// ============================================================================
// Benchmark Results
// ============================================================================

interface BenchmarkResult {
  operation: string;
  datasetSize: number;
  avgTime: number;
  minTime: number;
  maxTime: number;
  iterations: number;
  passThreshold: boolean;
  threshold: number;
}

const results: BenchmarkResult[] = [];

// ============================================================================
// Utility Functions
// ============================================================================

function createTestEpisode(index: number, memoryType: MemoryType): Omit<Episode, 'id'> {
  return {
    query: `Test query ${index}`,
    response: `Test response ${index}`,
    attention: {
      sources: [`paper${index}.pdf`],
      weights: [1.0],
      patterns: [`pattern${index}`]
    },
    outcome: 'success',
    confidence: 0.95,
    timestamp: Date.now(),
    memory_type: memoryType
  };
}

function formatTime(ms: number): string {
  if (ms < 1) {
    return `${(ms * 1000).toFixed(2)}μs`;
  }
  return `${ms.toFixed(2)}ms`;
}

function cleanupDatabase(): void {
  if (fs.existsSync(BENCHMARK_DB_DIR)) {
    fs.rmSync(BENCHMARK_DB_DIR, { recursive: true });
  }
}

function logResult(result: BenchmarkResult): void {
  const status = result.passThreshold ? '✅' : '❌';
  console.log(
    `${status} ${result.operation} [n=${result.datasetSize}]: ` +
    `avg=${formatTime(result.avgTime)}, ` +
    `min=${formatTime(result.minTime)}, ` +
    `max=${formatTime(result.maxTime)} ` +
    `(threshold: ${formatTime(result.threshold)})`
  );
}

// ============================================================================
// Benchmark: Database Load Time
// ============================================================================

async function benchmarkDatabaseLoad(): Promise<void> {
  console.log('\n📊 Benchmark: Database Load Time');
  console.log('=' .repeat(80));

  for (const size of DATASET_SIZES) {
    cleanupDatabase();

    // Pre-populate database
    let db = new SqloDatabase(BENCHMARK_DB_DIR);
    for (let i = 0; i < size; i++) {
      await db.put(createTestEpisode(i, MemoryType.LONG_TERM));
    }

    // Benchmark load time
    const times: number[] = [];
    for (let i = 0; i < 10; i++) {
      const start = performance.now();
      db = new SqloDatabase(BENCHMARK_DB_DIR);
      const end = performance.now();
      times.push(end - start);
    }

    const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
    const minTime = Math.min(...times);
    const maxTime = Math.max(...times);
    const threshold = 100; // 100ms threshold

    const result: BenchmarkResult = {
      operation: 'Database Load',
      datasetSize: size,
      avgTime,
      minTime,
      maxTime,
      iterations: 10,
      passThreshold: avgTime < threshold,
      threshold
    };

    results.push(result);
    logResult(result);
  }

  cleanupDatabase();
}

// ============================================================================
// Benchmark: Get (Read) Performance
// ============================================================================

async function benchmarkGet(): Promise<void> {
  console.log('\n📊 Benchmark: Get (Read) Performance');
  console.log('=' .repeat(80));

  for (const size of DATASET_SIZES) {
    cleanupDatabase();
    const db = new SqloDatabase(BENCHMARK_DB_DIR);

    // Pre-populate database
    const hashes: string[] = [];
    for (let i = 0; i < size; i++) {
      const hash = await db.put(createTestEpisode(i, MemoryType.LONG_TERM));
      hashes.push(hash);
    }

    // Benchmark get() operation
    const times: number[] = [];
    for (let i = 0; i < ITERATIONS_PER_SIZE; i++) {
      const randomHash = hashes[Math.floor(Math.random() * hashes.length)];
      const start = performance.now();
      db.get(randomHash);
      const end = performance.now();
      times.push(end - start);
    }

    const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
    const minTime = Math.min(...times);
    const maxTime = Math.max(...times);
    const threshold = 1; // 1ms threshold

    const result: BenchmarkResult = {
      operation: 'Get (Read)',
      datasetSize: size,
      avgTime,
      minTime,
      maxTime,
      iterations: ITERATIONS_PER_SIZE,
      passThreshold: avgTime < threshold,
      threshold
    };

    results.push(result);
    logResult(result);
  }

  cleanupDatabase();
}

// ============================================================================
// Benchmark: Put (Write) Performance
// ============================================================================

async function benchmarkPut(): Promise<void> {
  console.log('\n📊 Benchmark: Put (Write) Performance');
  console.log('=' .repeat(80));

  for (const size of DATASET_SIZES) {
    cleanupDatabase();
    const db = new SqloDatabase(BENCHMARK_DB_DIR);

    // Pre-populate database
    for (let i = 0; i < size; i++) {
      await db.put(createTestEpisode(i, MemoryType.LONG_TERM));
    }

    // Benchmark put() operation
    const times: number[] = [];
    for (let i = 0; i < ITERATIONS_PER_SIZE; i++) {
      const start = performance.now();
      await db.put(createTestEpisode(size + i, MemoryType.LONG_TERM));
      const end = performance.now();
      times.push(end - start);
    }

    const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
    const minTime = Math.min(...times);
    const maxTime = Math.max(...times);
    const threshold = 10; // 10ms threshold

    const result: BenchmarkResult = {
      operation: 'Put (Write)',
      datasetSize: size,
      avgTime,
      minTime,
      maxTime,
      iterations: ITERATIONS_PER_SIZE,
      passThreshold: avgTime < threshold,
      threshold
    };

    results.push(result);
    logResult(result);
  }

  cleanupDatabase();
}

// ============================================================================
// Benchmark: Has (Existence Check) Performance
// ============================================================================

async function benchmarkHas(): Promise<void> {
  console.log('\n📊 Benchmark: Has (Existence Check) Performance');
  console.log('=' .repeat(80));

  for (const size of DATASET_SIZES) {
    cleanupDatabase();
    const db = new SqloDatabase(BENCHMARK_DB_DIR);

    // Pre-populate database
    const hashes: string[] = [];
    for (let i = 0; i < size; i++) {
      const hash = await db.put(createTestEpisode(i, MemoryType.LONG_TERM));
      hashes.push(hash);
    }

    // Benchmark has() operation
    const times: number[] = [];
    for (let i = 0; i < ITERATIONS_PER_SIZE * 10; i++) {
      const randomHash = hashes[Math.floor(Math.random() * hashes.length)];
      const start = performance.now();
      db.has(randomHash);
      const end = performance.now();
      times.push(end - start);
    }

    const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
    const minTime = Math.min(...times);
    const maxTime = Math.max(...times);
    const threshold = 0.1; // 0.1ms threshold

    const result: BenchmarkResult = {
      operation: 'Has (Check)',
      datasetSize: size,
      avgTime,
      minTime,
      maxTime,
      iterations: ITERATIONS_PER_SIZE * 10,
      passThreshold: avgTime < threshold,
      threshold
    };

    results.push(result);
    logResult(result);
  }

  cleanupDatabase();
}

// ============================================================================
// Benchmark: Delete Performance
// ============================================================================

async function benchmarkDelete(): Promise<void> {
  console.log('\n📊 Benchmark: Delete Performance');
  console.log('=' .repeat(80));

  for (const size of DATASET_SIZES) {
    cleanupDatabase();
    const db = new SqloDatabase(BENCHMARK_DB_DIR);

    // Pre-populate database with extra episodes for deletion
    const hashes: string[] = [];
    for (let i = 0; i < size + ITERATIONS_PER_SIZE; i++) {
      const hash = await db.put(createTestEpisode(i, MemoryType.LONG_TERM));
      if (i >= size) {
        hashes.push(hash);
      }
    }

    // Benchmark delete() operation
    const times: number[] = [];
    for (const hash of hashes) {
      const start = performance.now();
      db.delete(hash);
      const end = performance.now();
      times.push(end - start);
    }

    const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
    const minTime = Math.min(...times);
    const maxTime = Math.max(...times);
    const threshold = 5; // 5ms threshold

    const result: BenchmarkResult = {
      operation: 'Delete',
      datasetSize: size,
      avgTime,
      minTime,
      maxTime,
      iterations: ITERATIONS_PER_SIZE,
      passThreshold: avgTime < threshold,
      threshold
    };

    results.push(result);
    logResult(result);
  }

  cleanupDatabase();
}

// ============================================================================
// Scaling Analysis (O(1) Verification)
// ============================================================================

function analyzeScaling(): void {
  console.log('\n📊 Scaling Analysis (O(1) Verification)');
  console.log('=' .repeat(80));

  const operations = ['Get (Read)', 'Put (Write)', 'Has (Check)', 'Delete'];

  for (const operation of operations) {
    const opResults = results.filter(r => r.operation === operation);
    if (opResults.length < 2) continue;

    const firstResult = opResults[0];
    const lastResult = opResults[opResults.length - 1];

    const sizeRatio = lastResult.datasetSize / firstResult.datasetSize;
    const timeRatio = lastResult.avgTime / firstResult.avgTime;

    // For O(1), time ratio should be close to 1 (constant time)
    // We allow up to 3x variance due to system overhead
    const isO1 = timeRatio < 3;

    const status = isO1 ? '✅' : '❌';
    console.log(
      `${status} ${operation}: ` +
      `${sizeRatio.toFixed(0)}x size increase → ${timeRatio.toFixed(2)}x time increase ` +
      `(O(1) verified: ${isO1})`
    );
  }
}

// ============================================================================
// Summary Report
// ============================================================================

function printSummary(): void {
  console.log('\n📊 Summary Report');
  console.log('=' .repeat(80));

  const passed = results.filter(r => r.passThreshold).length;
  const total = results.length;

  console.log(`\nTotal Benchmarks: ${total}`);
  console.log(`✅ Passed: ${passed}`);
  console.log(`❌ Failed: ${total - passed}`);

  if (passed === total) {
    console.log('\n🎉 All performance benchmarks passed!');
    console.log('   SQLO Database maintains O(1) performance guarantees.');
  } else {
    console.log('\n⚠️  Some benchmarks did not meet performance targets.');
  }

  console.log('\n' + '=' .repeat(80));
}

// ============================================================================
// Main Execution
// ============================================================================

async function main(): Promise<void> {
  console.log('\n🚀 SQLO Database Performance Benchmarks');
  console.log('=' .repeat(80));
  console.log(`Dataset sizes: ${DATASET_SIZES.join(', ')}`);
  console.log(`Iterations per size: ${ITERATIONS_PER_SIZE}`);
  console.log('=' .repeat(80));

  await benchmarkDatabaseLoad();
  await benchmarkGet();
  await benchmarkPut();
  await benchmarkHas();
  await benchmarkDelete();

  analyzeScaling();
  printSummary();

  cleanupDatabase();
}

// Run benchmarks
main().catch(error => {
  console.error('Benchmark failed:', error);
  cleanupDatabase();
  process.exit(1);
});
