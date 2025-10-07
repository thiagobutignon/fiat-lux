import { ErrorAnalysis, ErrorExample } from '../domain/entities/ErrorAnalysis';
import { SignalType, TradingSignal } from '../domain/entities/TradingSignal';
import { PatternType } from '../domain/entities/Pattern';
import { TestCase } from '../infrastructure/data-generation/CandlestickGenerator';

/**
 * Application Service: ErrorAnalysisBuilder
 * Builds error analysis from benchmark results
 */
export class ErrorAnalysisBuilder {
  private systemName: string;
  private totalCases = 0;
  private correctCases = 0;
  private falsePositives = new Map<SignalType, ErrorExample[]>();
  private falseNegatives = new Map<SignalType, ErrorExample[]>();
  private confusionMatrix = new Map<SignalType, Map<SignalType, number>>();
  private patternAccuracy = new Map<PatternType, { correct: number; total: number }>();

  constructor(systemName: string) {
    this.systemName = systemName;

    // Initialize false positives/negatives maps
    [SignalType.BUY, SignalType.SELL, SignalType.HOLD].forEach(signal => {
      this.falsePositives.set(signal, []);
      this.falseNegatives.set(signal, []);
    });

    // Initialize confusion matrix
    [SignalType.BUY, SignalType.SELL, SignalType.HOLD].forEach(expected => {
      const row = new Map<SignalType, number>();
      [SignalType.BUY, SignalType.SELL, SignalType.HOLD].forEach(predicted => {
        row.set(predicted, 0);
      });
      this.confusionMatrix.set(expected, row);
    });
  }

  /**
   * Add a test result
   */
  addResult(testCase: TestCase, predicted: TradingSignal): void {
    this.totalCases++;

    const expected = testCase.expectedSignal;
    const predictedSignal = predicted.type;

    // Update confusion matrix
    const row = this.confusionMatrix.get(expected);
    if (row) {
      row.set(predictedSignal, (row.get(predictedSignal) || 0) + 1);
    }

    // Check if correct
    const isCorrect = expected === predictedSignal;
    if (isCorrect) {
      this.correctCases++;
    }

    // Track pattern accuracy
    if (testCase.patternType) {
      const stats = this.patternAccuracy.get(testCase.patternType) || { correct: 0, total: 0 };
      stats.total++;
      if (isCorrect) stats.correct++;
      this.patternAccuracy.set(testCase.patternType, stats);
    }

    // Track false positives and false negatives
    if (!isCorrect) {
      const errorExample: ErrorExample = {
        expected,
        predicted: predictedSignal,
        candles: testCase.sequence.candles.slice(-5), // Last 5 candles
        patternType: testCase.patternType,
        explanation: predicted.explanation,
      };

      // False positive: predicted signal when should be HOLD
      if (expected === SignalType.HOLD && predictedSignal !== SignalType.HOLD) {
        this.falsePositives.get(predictedSignal)?.push(errorExample);
      }

      // False negative: predicted HOLD when should be signal
      if (expected !== SignalType.HOLD && predictedSignal === SignalType.HOLD) {
        this.falseNegatives.get(expected)?.push(errorExample);
      }
    }
  }

  /**
   * Build the error analysis
   */
  build(): ErrorAnalysis {
    return new ErrorAnalysis(
      this.systemName,
      this.totalCases,
      this.correctCases,
      this.falsePositives,
      this.falseNegatives,
      this.confusionMatrix,
      this.patternAccuracy
    );
  }

  /**
   * Export error examples to JSON for visualization
   */
  exportErrorExamples(maxPerType: number = 5): any {
    const examples: any = {
      system: this.systemName,
      accuracy: this.totalCases > 0 ? this.correctCases / this.totalCases : 0,
      falsePositives: {},
      falseNegatives: {},
    };

    // Export false positives
    this.falsePositives.forEach((errors, signal) => {
      examples.falsePositives[signal] = errors.slice(0, maxPerType).map(e => ({
        expected: e.expected,
        predicted: e.predicted,
        pattern: e.patternType,
        candles: e.candles.map(c => ({
          open: c.open,
          high: c.high,
          low: c.low,
          close: c.close,
        })),
        explanation: e.explanation,
      }));
    });

    // Export false negatives
    this.falseNegatives.forEach((errors, signal) => {
      examples.falseNegatives[signal] = errors.slice(0, maxPerType).map(e => ({
        expected: e.expected,
        predicted: e.predicted,
        pattern: e.patternType,
        candles: e.candles.map(c => ({
          open: c.open,
          high: c.high,
          low: c.low,
          close: c.close,
        })),
        explanation: e.explanation,
      }));
    });

    return examples;
  }
}
