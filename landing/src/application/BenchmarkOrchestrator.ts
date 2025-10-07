import { RunBenchmark } from './use-cases/RunBenchmark';
import { BenchmarkResult } from '../domain/entities/BenchmarkResult';
import { CandlestickGenerator, TestCase } from '../infrastructure/data-generation/CandlestickGenerator';
import { GrammarPatternDetector } from '../infrastructure/adapters/GrammarPatternDetector';
import {
  createGPT4Detector,
  createClaudeDetector,
  createLlamaDetector
} from '../infrastructure/adapters/LLMPatternDetector';
import { LSTMPatternDetector } from '../infrastructure/adapters/LSTMPatternDetector';

/**
 * Application Layer: BenchmarkOrchestrator
 * Coordinates the entire benchmark execution
 */

export interface BenchmarkSummary {
  results: BenchmarkResult[];
  winner: BenchmarkResult;
  comparisons: string[];
}

export class BenchmarkOrchestrator {
  private readonly runBenchmark: RunBenchmark;
  private readonly generator: CandlestickGenerator;

  constructor() {
    this.runBenchmark = new RunBenchmark();
    this.generator = new CandlestickGenerator();
  }

  /**
   * Run the full benchmark suite across all systems
   */
  async runFullBenchmark(testCaseCount: number = 1000): Promise<BenchmarkSummary> {
    console.log('═══════════════════════════════════════════════════════');
    console.log('  DETERMINISTIC INTELLIGENCE BENCHMARK');
    console.log('  Domain: Trading Signal Generation');
    console.log('═══════════════════════════════════════════════════════\n');

    // Generate test data
    console.log(`Generating ${testCaseCount} test cases...`);
    const testCases = this.generator.generateTestCases(testCaseCount);
    const testData = testCases.map(tc => tc.sequence);
    const groundTruth = testCases.map(tc => tc.expectedSignal);
    console.log(`✓ Generated ${testCases.length} test cases\n`);

    // Initialize all detectors
    const detectors = [
      new GrammarPatternDetector(),
      createGPT4Detector(),
      createClaudeDetector(),
      createLlamaDetector(),
      new LSTMPatternDetector(),
    ];

    console.log(`Running benchmarks for ${detectors.length} systems...\n`);

    // Run benchmarks
    const results: BenchmarkResult[] = [];

    for (const detector of detectors) {
      console.log(`\n${'─'.repeat(55)}`);
      const result = await this.runBenchmark.execute({
        detector,
        testData,
        groundTruth,
      });
      results.push(result);
    }

    console.log(`\n${'═'.repeat(55)}`);
    console.log('  BENCHMARK COMPLETE');
    console.log(`${'═'.repeat(55)}\n`);

    // Find winner (highest accuracy with consideration for speed and explainability)
    const winner = this.determineWinner(results);

    // Generate comparisons
    const comparisons = this.generateComparisons(results, winner);

    return {
      results,
      winner,
      comparisons,
    };
  }

  /**
   * Determine the winner based on multiple criteria
   */
  private determineWinner(results: BenchmarkResult[]): BenchmarkResult {
    // Score each system
    const scores = results.map(result => {
      const accuracyScore = result.metrics.accuracy * 100;
      const speedScore = Math.min(100, 1000 / Math.max(0.001, result.metrics.avgLatencyMs));
      const costScore = result.metrics.totalCostUSD === 0 ? 100 : Math.min(100, 1 / result.metrics.totalCostUSD);
      const explainabilityScore = result.metrics.explainabilityScore * 100;

      // Weighted score (accuracy is most important)
      const totalScore = (
        accuracyScore * 0.4 +
        speedScore * 0.25 +
        costScore * 0.2 +
        explainabilityScore * 0.15
      );

      return { result, totalScore };
    });

    // Sort by score and return the winner
    scores.sort((a, b) => b.totalScore - a.totalScore);
    return scores[0].result;
  }

  /**
   * Generate comparison summaries
   */
  private generateComparisons(results: BenchmarkResult[], winner: BenchmarkResult): string[] {
    const comparisons: string[] = [];

    results.forEach(result => {
      if (result.systemName !== winner.systemName) {
        comparisons.push(winner.compareTo(result));
      }
    });

    return comparisons;
  }

  /**
   * Format and display results
   */
  displayResults(summary: BenchmarkSummary): void {
    console.log('\n📊 RESULTS SUMMARY\n');
    console.log('| System                      | Accuracy | Latency  | Cost/1k  | Explainable |');
    console.log('|----------------------------|----------|----------|----------|-------------|');

    summary.results.forEach(result => {
      const accuracy = (result.metrics.accuracy * 100).toFixed(0) + '%';
      const latency = result.metrics.avgLatencyMs < 1
        ? result.metrics.avgLatencyMs.toFixed(4) + 'ms'
        : result.metrics.avgLatencyMs.toFixed(1) + 'ms';
      const cost = result.metrics.totalCostUSD === 0
        ? '$0.00'
        : '$' + result.metrics.totalCostUSD.toFixed(2);
      const explainable = result.metrics.explainabilityScore === 1 ? '✅ 100%' : '❌ 0%';

      const nameColumn = result.systemName.padEnd(27);
      const accColumn = accuracy.padEnd(8);
      const latColumn = latency.padEnd(8);
      const costColumn = cost.padEnd(8);

      console.log(`| ${nameColumn} | ${accColumn} | ${latColumn} | ${costColumn} | ${explainable} |`);
    });

    console.log('\n🏆 WINNER: ' + summary.winner.systemName + '\n');

    console.log('📈 COMPARISONS:\n');
    summary.comparisons.forEach((comparison, index) => {
      console.log(`${index + 1}. ${comparison}\n`);
    });
  }

  /**
   * Export results to JSON
   */
  exportToJSON(summary: BenchmarkSummary): string {
    return JSON.stringify({
      timestamp: new Date().toISOString(),
      winner: summary.winner.systemName,
      results: summary.results.map(r => ({
        system: r.systemName,
        metrics: r.metrics,
        f1Score: r.getF1Score(),
        precision: r.getPrecision(),
        recall: r.getRecall(),
      })),
      comparisons: summary.comparisons,
    }, null, 2);
  }
}
