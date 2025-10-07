import { IPatternDetector } from '../../domain/repositories/IPatternDetector';
import { CandlestickSequence } from '../../domain/entities/CandlestickSequence';
import { BenchmarkResult, BenchmarkMetrics } from '../../domain/entities/BenchmarkResult';
import { SignalType, TradingSignal } from '../../domain/entities/TradingSignal';
import { ErrorAnalysis } from '../../domain/entities/ErrorAnalysis';
import { ErrorAnalysisBuilder } from '../ErrorAnalysisBuilder';
import { TestCase } from '../../infrastructure/data-generation/CandlestickGenerator';

/**
 * Application Use Case: RunBenchmark
 * Orchestrates the execution of pattern detection benchmarks
 */

export interface BenchmarkConfig {
  detector: IPatternDetector;
  testData: CandlestickSequence[];
  groundTruth: SignalType[]; // Expected signals for each sequence
  testCases?: TestCase[]; // Optional: for detailed error analysis
}

export class RunBenchmark {
  async execute(config: BenchmarkConfig): Promise<BenchmarkResult> {
    const { detector, testData, groundTruth, testCases } = config;

    if (testData.length !== groundTruth.length) {
      throw new Error('Test data and ground truth must have the same length');
    }

    console.log(`Running benchmark for ${detector.getName()}...`);
    console.log(`Test cases: ${testData.length}`);

    const startTime = performance.now();
    const results: TradingSignal[] = [];
    const latencies: number[] = [];
    let totalCost = 0;

    // Error analysis builder
    const errorAnalysisBuilder = new ErrorAnalysisBuilder(detector.getName());

    // Run detection on each test case
    for (let i = 0; i < testData.length; i++) {
      const sequence = testData[i];

      const testStart = performance.now();
      const signal = await detector.detectPatterns(sequence);
      const testEnd = performance.now();

      results.push(signal);
      latencies.push(testEnd - testStart);

      // Calculate cost (if detector has a cost)
      if ('getCost' in detector) {
        totalCost += (detector as any).getCost();
      }

      // Add to error analysis if test cases provided
      if (testCases && testCases[i]) {
        errorAnalysisBuilder.addResult(testCases[i], signal);
      }

      // Progress logging every 100 tests
      if ((i + 1) % 100 === 0) {
        console.log(`  Progress: ${i + 1}/${testData.length} (${((i + 1) / testData.length * 100).toFixed(1)}%)`);
      }
    }

    const endTime = performance.now();
    const totalTime = endTime - startTime;

    // Calculate metrics
    const signalTypes = results.map(r => r.type);
    const metrics = this.calculateMetrics(signalTypes, groundTruth, latencies, totalCost, detector);

    console.log(`✓ Completed ${detector.getName()} in ${totalTime.toFixed(0)}ms`);
    console.log(`  Accuracy: ${(metrics.accuracy * 100).toFixed(1)}%`);
    console.log(`  Avg Latency: ${metrics.avgLatencyMs.toFixed(3)}ms`);
    console.log(`  Total Cost: $${metrics.totalCostUSD.toFixed(4)}`);

    // Build error analysis
    const errorAnalysis = errorAnalysisBuilder.build();

    const result = new BenchmarkResult(
      detector.getName(),
      metrics,
      testData.length,
      new Date()
    );

    // Attach error analysis
    (result as any).errorAnalysis = errorAnalysis;

    return result;
  }

  private calculateMetrics(
    results: SignalType[],
    groundTruth: SignalType[],
    latencies: number[],
    totalCost: number,
    detector: IPatternDetector
  ): BenchmarkMetrics {
    let truePositives = 0;
    let trueNegatives = 0;
    let falsePositives = 0;
    let falseNegatives = 0;
    let correct = 0;

    for (let i = 0; i < results.length; i++) {
      const predicted = results[i];
      const actual = groundTruth[i];

      if (predicted === actual) {
        correct++;
      }

      // For pattern detection, we consider:
      // Positive = BUY or SELL signal
      // Negative = HOLD signal
      const isPredictedPositive = predicted !== SignalType.HOLD;
      const isActualPositive = actual !== SignalType.HOLD;

      if (isPredictedPositive && isActualPositive) {
        truePositives++;
      } else if (!isPredictedPositive && !isActualPositive) {
        trueNegatives++;
      } else if (isPredictedPositive && !isActualPositive) {
        falsePositives++;
      } else if (!isPredictedPositive && isActualPositive) {
        falseNegatives++;
      }
    }

    const accuracy = correct / results.length;
    const avgLatencyMs = latencies.reduce((a, b) => a + b, 0) / latencies.length;
    const explainabilityScore = detector.isExplainable() ? 1.0 : 0.0;

    return {
      accuracy,
      avgLatencyMs,
      totalCostUSD: totalCost,
      explainabilityScore,
      truePositives,
      trueNegatives,
      falsePositives,
      falseNegatives,
    };
  }
}
