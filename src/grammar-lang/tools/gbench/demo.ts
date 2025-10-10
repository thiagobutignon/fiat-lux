/**
 * GBench Demo - Performance Benchmarking
 *
 * Demonstrates O(1) benchmarking capabilities inspired by GVCS metrics.
 */

import { suite, report, compare, reportComparison, createRegressionDetector } from './index';

async function main() {
  console.log('🚀 GBench Demo - O(1) Benchmarking Tool\n');
  console.log('═'.repeat(70));

  // =========================================================================
  // Demo 1: Hash Map vs Array Lookup
  // =========================================================================

  console.log('\n📊 Demo 1: Hash Map vs Array Lookup (O(1) vs O(n))');
  console.log('─'.repeat(70));

  const lookupSuite = suite('lookup-comparison');

  // Prepare data
  const map = new Map<string, number>();
  const array: Array<{ key: string; value: number }> = [];

  for (let i = 0; i < 10000; i++) {
    const key = `key-${i}`;
    map.set(key, i);
    array.push({ key, value: i });
  }

  const searchKey = 'key-9999'; // Worst case for array

  // Hash map lookup (O(1))
  lookupSuite.add('hash-map-lookup', () => {
    map.get(searchKey);
  });

  // Array lookup (O(n))
  lookupSuite.add('array-lookup', () => {
    array.find(item => item.key === searchKey);
  });

  // Run benchmarks
  await lookupSuite.runAll({ iterations: 10000, warmup_iterations: 1000 });

  // Display results
  const hashResult = lookupSuite.getResult('hash-map-lookup')!;
  const arrayResult = lookupSuite.getResult('array-lookup')!;

  report(hashResult);
  console.log('');
  report(arrayResult);

  // Compare
  const comparison = compare('Array (O(n))', arrayResult, 'HashMap (O(1))', hashResult);
  reportComparison(comparison);

  // =========================================================================
  // Demo 2: GVCS-Style Fitness Comparison
  // =========================================================================

  console.log('\n📊 Demo 2: Version Fitness Comparison (GVCS-Inspired)');
  console.log('─'.repeat(70));

  const versionSuite = suite('version-comparison');

  // Simulate 3 code versions with different performance
  versionSuite.add('v1.0.0-baseline', () => {
    // Baseline: moderate performance
    let sum = 0;
    for (let i = 0; i < 100; i++) {
      sum += Math.sqrt(i);
    }
  });

  versionSuite.add('v1.0.1-optimized', () => {
    // Optimized: better performance
    let sum = 0;
    for (let i = 0; i < 100; i++) {
      sum += i ** 0.5; // Faster than Math.sqrt
    }
  });

  versionSuite.add('v1.0.2-regression', () => {
    // Regression: worse performance
    let sum = 0;
    for (let i = 0; i < 100; i++) {
      sum += Math.pow(i, 0.5); // Slower than alternatives
    }
  });

  await versionSuite.runAll({ iterations: 5000 });

  // Compare each version against baseline
  const baseline = versionSuite.getResult('v1.0.0-baseline')!;
  const optimized = versionSuite.getResult('v1.0.1-optimized')!;
  const regression = versionSuite.getResult('v1.0.2-regression')!;

  console.log('\n🔍 v1.0.1 vs Baseline:');
  const comp1 = compare('v1.0.0-baseline', baseline, 'v1.0.1-optimized', optimized);
  reportComparison(comp1);

  console.log('\n🔍 v1.0.2 vs Baseline:');
  const comp2 = compare('v1.0.0-baseline', baseline, 'v1.0.2-regression', regression);
  reportComparison(comp2);

  // =========================================================================
  // Demo 3: Regression Detection
  // =========================================================================

  console.log('\n📊 Demo 3: Regression Detection (Auto Monitoring)');
  console.log('─'.repeat(70));

  const detector = createRegressionDetector(10); // 10% threshold

  // Simulate version history
  detector.addResult('calculate-fitness', baseline);
  detector.addResult('calculate-fitness', optimized);
  detector.addResult('calculate-fitness', regression);

  const isRegression = detector.detectRegression('calculate-fitness');
  const regressionReport = detector.getReport('calculate-fitness');

  if (isRegression && regressionReport) {
    console.log(regressionReport);
  } else {
    console.log('✅ No regression detected');
  }

  // =========================================================================
  // Demo 4: Memory Tracking
  // =========================================================================

  console.log('\n📊 Demo 4: Memory Usage Tracking');
  console.log('─'.repeat(70));

  const memorySuite = suite('memory-tracking');

  // Small allocation
  memorySuite.add('small-allocation', () => {
    const arr = new Array(100).fill(0);
  });

  // Large allocation
  memorySuite.add('large-allocation', () => {
    const arr = new Array(10000).fill(0);
  });

  await memorySuite.runAll({ iterations: 1000, track_memory: true });

  const smallResult = memorySuite.getResult('small-allocation')!;
  const largeResult = memorySuite.getResult('large-allocation')!;

  report(smallResult);
  console.log('');
  report(largeResult);

  // =========================================================================
  // Summary
  // =========================================================================

  console.log('\n═'.repeat(70));
  console.log('✅ GBench Demo Complete!');
  console.log('\nKey Features Demonstrated:');
  console.log('  ✅ O(1) hash-based benchmark registration');
  console.log('  ✅ Statistical analysis (mean, p95, p99)');
  console.log('  ✅ Performance comparison (regression/improvement detection)');
  console.log('  ✅ Memory usage tracking');
  console.log('  ✅ GVCS-inspired fitness comparison');
  console.log('  ✅ Automated regression detection');
  console.log('\n🚀 Ready for production benchmarking!');
  console.log('═'.repeat(70) + '\n');
}

// Run demo
if (require.main === module) {
  main().catch(console.error);
}

export { main };
