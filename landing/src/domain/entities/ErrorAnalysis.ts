import { Candlestick } from './Candlestick';
import { SignalType } from './TradingSignal';
import { PatternType } from './Pattern';

/**
 * Domain Entity: ErrorAnalysis
 * Analyzes where a detector makes mistakes
 */

export interface ErrorExample {
  expected: SignalType;
  predicted: SignalType;
  candles: Candlestick[];
  patternType?: PatternType;
  explanation: string;
}

export class ErrorAnalysis {
  constructor(
    public readonly systemName: string,
    public readonly totalCases: number,
    public readonly correctCases: number,
    public readonly falsePositives: Map<SignalType, ErrorExample[]>, // Predicted X when should be HOLD
    public readonly falseNegatives: Map<SignalType, ErrorExample[]>, // Predicted HOLD when should be X
    public readonly confusionMatrix: Map<SignalType, Map<SignalType, number>>, // [expected][predicted] = count
    public readonly patternAccuracy: Map<PatternType, { correct: number; total: number }>
  ) {}

  /**
   * Get overall accuracy
   */
  getAccuracy(): number {
    return this.totalCases > 0 ? this.correctCases / this.totalCases : 0;
  }

  /**
   * Get precision for a signal type
   * Precision = True Positives / (True Positives + False Positives)
   */
  getPrecision(signalType: SignalType): number {
    let truePositives = 0;
    let falsePositives = 0;

    this.confusionMatrix.forEach((predictedMap, expected) => {
      predictedMap.forEach((count, predicted) => {
        if (predicted === signalType) {
          if (expected === signalType) {
            truePositives += count;
          } else {
            falsePositives += count;
          }
        }
      });
    });

    const total = truePositives + falsePositives;
    return total > 0 ? truePositives / total : 0;
  }

  /**
   * Get recall for a signal type
   * Recall = True Positives / (True Positives + False Negatives)
   */
  getRecall(signalType: SignalType): number {
    let truePositives = 0;
    let falseNegatives = 0;

    const predictedMap = this.confusionMatrix.get(signalType);
    if (!predictedMap) return 0;

    predictedMap.forEach((count, predicted) => {
      if (predicted === signalType) {
        truePositives += count;
      } else {
        falseNegatives += count;
      }
    });

    const total = truePositives + falseNegatives;
    return total > 0 ? truePositives / total : 0;
  }

  /**
   * Get F1 score for a signal type
   */
  getF1Score(signalType: SignalType): number {
    const precision = this.getPrecision(signalType);
    const recall = this.getRecall(signalType);

    if (precision + recall === 0) return 0;
    return 2 * (precision * recall) / (precision + recall);
  }

  /**
   * Get the most common error type
   */
  getMostCommonError(): { from: SignalType; to: SignalType; count: number } | null {
    let maxCount = 0;
    let maxFrom: SignalType | null = null;
    let maxTo: SignalType | null = null;

    this.confusionMatrix.forEach((predictedMap, expected) => {
      predictedMap.forEach((count, predicted) => {
        if (expected !== predicted && count > maxCount) {
          maxCount = count;
          maxFrom = expected;
          maxTo = predicted;
        }
      });
    });

    if (maxFrom && maxTo) {
      return { from: maxFrom, to: maxTo, count: maxCount };
    }

    return null;
  }

  /**
   * Get pattern with worst accuracy
   */
  getWorstPattern(): { pattern: PatternType; accuracy: number } | null {
    let worstAccuracy = 1.0;
    let worstPattern: PatternType | null = null;

    this.patternAccuracy.forEach((stats, pattern) => {
      const accuracy = stats.total > 0 ? stats.correct / stats.total : 0;
      if (accuracy < worstAccuracy) {
        worstAccuracy = accuracy;
        worstPattern = pattern;
      }
    });

    if (worstPattern) {
      return { pattern: worstPattern, accuracy: worstAccuracy };
    }

    return null;
  }

  /**
   * Display summary
   */
  displaySummary(): void {
    console.log(`\n📊 ERROR ANALYSIS: ${this.systemName}`);
    console.log(`${'='.repeat(60)}\n`);

    console.log(`Overall Accuracy: ${(this.getAccuracy() * 100).toFixed(1)}%`);
    console.log(`Correct: ${this.correctCases}/${this.totalCases}\n`);

    // Signal-level metrics
    console.log('Per-Signal Metrics:');
    [SignalType.BUY, SignalType.SELL, SignalType.HOLD].forEach(signal => {
      const precision = this.getPrecision(signal);
      const recall = this.getRecall(signal);
      const f1 = this.getF1Score(signal);

      console.log(`  ${signal}:`);
      console.log(`    Precision: ${(precision * 100).toFixed(1)}%`);
      console.log(`    Recall: ${(recall * 100).toFixed(1)}%`);
      console.log(`    F1 Score: ${(f1 * 100).toFixed(1)}%`);
    });

    // Confusion matrix
    console.log('\nConfusion Matrix:');
    console.log('             Predicted →');
    console.log('          BUY  SELL HOLD');
    [SignalType.BUY, SignalType.SELL, SignalType.HOLD].forEach(expected => {
      const row = [SignalType.BUY, SignalType.SELL, SignalType.HOLD].map(predicted => {
        const count = this.confusionMatrix.get(expected)?.get(predicted) || 0;
        return count.toString().padStart(4);
      }).join(' ');
      console.log(`${expected.padEnd(8)} ${row}`);
    });

    // Most common error
    const commonError = this.getMostCommonError();
    if (commonError) {
      console.log(`\nMost Common Error: ${commonError.from} → ${commonError.to} (${commonError.count} times)`);
    }

    // Worst pattern
    const worstPattern = this.getWorstPattern();
    if (worstPattern) {
      console.log(`Worst Pattern: ${worstPattern.pattern} (${(worstPattern.accuracy * 100).toFixed(1)}% accuracy)`);
    }

    // False positives
    console.log('\nFalse Positives (predicted signal when should be HOLD):');
    this.falsePositives.forEach((examples, signal) => {
      if (examples.length > 0) {
        console.log(`  ${signal}: ${examples.length} cases`);
      }
    });

    // False negatives
    console.log('\nFalse Negatives (predicted HOLD when should be signal):');
    this.falseNegatives.forEach((examples, signal) => {
      if (examples.length > 0) {
        console.log(`  ${signal}: ${examples.length} cases`);
      }
    });
  }
}
