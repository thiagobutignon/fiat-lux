import { RunBenchmark } from './run-benchmark';
import { BenchmarkResult } from '../entities/benchmark-result';
import { CandlestickGenerator, TestCase } from '../../data/use-cases/candlestick-generator';
import { GrammarPatternDetector } from '../../infrastructure/adapters/grammar-pattern-detector';
import {
  createGPT4Detector,
  createClaudeDetector,
  createLlamaDetector
} from '../../infrastructure/adapters/llm-pattern-detector';
import { LSTMPatternDetector } from '../../infrastructure/adapters/lstm-pattern-detector';
import { GeminiPatternDetector } from '../../infrastructure/adapters/gemini-pattern-detector';
import { LocalLlamaDetector } from '../../infrastructure/adapters/local-llama-detector';
import { VllmPatternDetector } from '../../infrastructure/adapters/vllm-pattern-detector';
import { LlamaCppDetector } from '../../infrastructure/adapters/llamacpp-detector';

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
    console.log(`⚙️  Error analysis enabled - detailed metrics will be generated\n`);

    // Initialize all detectors
    const detectors = [
      new GrammarPatternDetector(),
    ];

    // Try to add Gemini detector (requires API key)
    const enableGemini = process.env.ENABLE_GEMINI !== 'false';
    if (enableGemini) {
      try {
        detectors.push(new GeminiPatternDetector());
        console.log('✓ Gemini 2.5 Flash enabled (real API integration)\n');
      } catch (error) {
        console.warn('⚠️  Skipping Gemini: GEMINI_API_KEY not configured');
        console.warn('   Get your API key from: https://aistudio.google.com/apikey\n');
      }
    } else {
      console.log('⚠️  Gemini disabled via ENABLE_GEMINI=false\n');
    }

    // Try to add llama.cpp detector (Mac M-series with Metal) - FASTEST FOR MAC
    const enableLlamaCpp = process.env.ENABLE_LLAMACPP === 'true';
    if (enableLlamaCpp) {
      try {
        const llamacppUrl = process.env.LLAMACPP_BASE_URL || 'http://localhost:8080';
        const llamacppModel = process.env.LLAMACPP_MODEL || 'Meta-Llama-3.1-8B-Instruct-Q4_K_M';
        detectors.push(new LlamaCppDetector(llamacppUrl, llamacppModel));
        console.log(`✓ llama.cpp enabled (${llamacppModel}) - Metal Acceleration\n`);
      } catch (error) {
        console.warn('⚠️  Skipping llama.cpp: Server not available');
        console.warn('   See MAC_SETUP.md for installation instructions\n');
      }
    }

    // Try to add vLLM detector (requires vLLM server) - FASTEST FOR NVIDIA
    const enableVllm = process.env.ENABLE_VLLM === 'true';
    if (enableVllm && !enableLlamaCpp) {  // Skip if llama.cpp is enabled
      try {
        const vllmUrl = process.env.VLLM_BASE_URL || 'http://localhost:8000';
        const vllmModel = process.env.VLLM_MODEL || 'meta-llama/Meta-Llama-3.1-8B-Instruct';
        detectors.push(new VllmPatternDetector(vllmUrl, vllmModel));
        console.log(`✓ vLLM enabled (${vllmModel.split('/').pop()}) - CUDA Acceleration\n`);
      } catch (error) {
        console.warn('⚠️  Skipping vLLM: Server not available');
        console.warn('   See VLLM_SETUP.md for installation instructions\n');
      }
    } else if (enableVllm && enableLlamaCpp) {
      console.log('ℹ️  Skipping vLLM (llama.cpp is enabled)\n');
    }

    // Try to add Local Llama detector (requires Ollama) - SLOWEST BUT EASIEST
    const enableLocalLlama = process.env.ENABLE_LOCAL_LLAMA === 'true';
    if (enableLocalLlama && !enableVllm && !enableLlamaCpp) {  // Skip if faster options enabled
      try {
        const ollamaUrl = process.env.OLLAMA_BASE_URL || 'http://localhost:11434';
        const ollamaModel = process.env.OLLAMA_MODEL || 'llama3.1:8b';
        detectors.push(new LocalLlamaDetector(ollamaUrl, ollamaModel));
        console.log(`✓ Local Llama enabled (${ollamaModel} via Ollama)\n`);
      } catch (error) {
        console.warn('⚠️  Skipping Local Llama: Ollama not available');
        console.warn('   Install Ollama: https://ollama.ai/\n');
      }
    } else if (enableLocalLlama && (enableVllm || enableLlamaCpp)) {
      console.log('ℹ️  Skipping Ollama (faster option enabled)\n');
    }

    // Add simulated detectors
    detectors.push(
      createGPT4Detector(),
      createClaudeDetector(),
      createLlamaDetector(),
      new LSTMPatternDetector()
    );

    console.log(`Running benchmarks for ${detectors.length} systems...\n`);

    // Run benchmarks
    const results: BenchmarkResult[] = [];

    for (const detector of detectors) {
      console.log(`\n${'─'.repeat(55)}`);
      const result = await this.runBenchmark.execute({
        detector,
        testData,
        groundTruth,
        testCases, // Include test cases for error analysis
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

    // Display error analysis for systems with < 100% accuracy
    const failingSystems = summary.results.filter(r => r.metrics.accuracy < 1.0);
    if (failingSystems.length > 0) {
      console.log('\n📉 ERROR ANALYSIS\n');
      console.log('Detailed analysis for systems with errors:\n');

      failingSystems.forEach(result => {
        const errorAnalysis = (result as any).errorAnalysis;
        if (errorAnalysis) {
          errorAnalysis.displaySummary();
        }
      });
    }
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
