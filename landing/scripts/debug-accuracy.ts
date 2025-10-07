#!/usr/bin/env tsx

/**
 * Debug script to understand why Grammar Engine has low accuracy
 */

import { CandlestickGenerator } from '../src/infrastructure/data-generation/CandlestickGenerator';
import { GrammarPatternDetector } from '../src/infrastructure/adapters/GrammarPatternDetector';

async function debug() {
  console.log('Debugging Grammar Engine accuracy...\n');

  const generator = new CandlestickGenerator();
  const detector = new GrammarPatternDetector();

  // Generate 1000 test cases
  const testCases = generator.generateTestCases(1000);

  let correct = 0;
  let total = testCases.length;

  const failures: any[] = [];

  for (const testCase of testCases) {
    const result = await detector.detectPatterns(testCase.sequence);
    const isCorrect = result.type === testCase.expectedSignal;

    if (!isCorrect) {
      failures.push({
        testCase,
        result,
        patterns: result.patterns,
      });

      console.log(`❌ FAILURE: ${testCase.patternType || 'NEUTRAL'}`);
      console.log(`   Expected: ${testCase.expectedSignal}, Got: ${result.type}`);
      console.log(`   Patterns detected:`);
      result.patterns.forEach(p => {
        console.log(`     - ${p.type} (${p.isBullish() ? 'BULLISH' : p.isBearish() ? 'BEARISH' : 'NEUTRAL'})`);
      });
      console.log(`   Last 3 candles:`);
      const last3 = testCase.sequence.candles.slice(-3);
      last3.forEach((c, i) => {
        const type = c.isBullish() ? 'BULL' : c.isBearish() ? 'BEAR' : 'NEUTRAL';
        console.log(`     ${i+1}. O:${c.open.toFixed(2)} H:${c.high.toFixed(2)} L:${c.low.toFixed(2)} C:${c.close.toFixed(2)} [${type}]`);
      });
      console.log();
    }

    if (isCorrect) correct++;
  }

  console.log(`\n${'='.repeat(60)}`);
  console.log(`FAILURE ANALYSIS (${failures.length} failures):`);
  console.log(`${'='.repeat(60)}\n`);

  const failuresByExpected = failures.reduce((acc, f) => {
    const expected = f.testCase.expectedSignal;
    if (!acc[expected]) acc[expected] = [];
    acc[expected].push(f);
    return acc;
  }, {} as Record<string, any[]>);

  Object.entries(failuresByExpected).forEach(([expected, fails]) => {
    console.log(`Expected ${expected}: ${fails.length} failures`);
    const gotCounts = fails.reduce((acc, f) => {
      const got = f.result.type;
      acc[got] = (acc[got] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
    Object.entries(gotCounts).forEach(([got, count]) => {
      console.log(`  → Got ${got}: ${count} times`);
    });

    // Show pattern distribution
    const patternCounts = fails.reduce((acc, f) => {
      f.patterns.forEach((p: any) => {
        acc[p.type] = (acc[p.type] || 0) + 1;
      });
      return acc;
    }, {} as Record<string, number>);
    console.log(`  Patterns detected in failures:`);
    Object.entries(patternCounts).forEach(([pattern, count]) => {
      console.log(`    - ${pattern}: ${count} times`);
    });
  });
  console.log();

  console.log(`\nAccuracy: ${correct}/${total} = ${(correct/total * 100).toFixed(1)}%`);
}

debug();
