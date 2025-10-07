/**
 * Domain Entity: BenchmarkResult
 * Represents the results of a benchmark test
 */

export interface BenchmarkMetrics {
  accuracy: number; // 0-1
  avgLatencyMs: number;
  totalCostUSD: number;
  explainabilityScore: number; // 0-1 (1 = fully explainable)
  truePositives: number;
  trueNegatives: number;
  falsePositives: number;
  falseNegatives: number;
}

export class BenchmarkResult {
  constructor(
    public readonly systemName: string,
    public readonly metrics: BenchmarkMetrics,
    public readonly testCount: number,
    public readonly timestamp: Date
  ) {
    this.validate();
  }

  private validate(): void {
    if (this.testCount <= 0) {
      throw new Error('Test count must be positive');
    }
    if (this.metrics.accuracy < 0 || this.metrics.accuracy > 1) {
      throw new Error('Accuracy must be between 0 and 1');
    }
    if (this.metrics.explainabilityScore < 0 || this.metrics.explainabilityScore > 1) {
      throw new Error('Explainability score must be between 0 and 1');
    }
  }

  /**
   * Calculate precision (TP / (TP + FP))
   */
  getPrecision(): number {
    const { truePositives, falsePositives } = this.metrics;
    if (truePositives + falsePositives === 0) return 0;
    return truePositives / (truePositives + falsePositives);
  }

  /**
   * Calculate recall (TP / (TP + FN))
   */
  getRecall(): number {
    const { truePositives, falseNegatives } = this.metrics;
    if (truePositives + falseNegatives === 0) return 0;
    return truePositives / (truePositives + falseNegatives);
  }

  /**
   * Calculate F1 score
   */
  getF1Score(): number {
    const precision = this.getPrecision();
    const recall = this.getRecall();
    if (precision + recall === 0) return 0;
    return 2 * (precision * recall) / (precision + recall);
  }

  /**
   * Calculate speed advantage compared to another system
   */
  getSpeedAdvantage(other: BenchmarkResult): number {
    if (this.metrics.avgLatencyMs === 0) return Infinity;
    return other.metrics.avgLatencyMs / this.metrics.avgLatencyMs;
  }

  /**
   * Calculate cost advantage compared to another system
   */
  getCostAdvantage(other: BenchmarkResult): number {
    if (this.metrics.totalCostUSD === 0) return Infinity;
    return other.metrics.totalCostUSD / this.metrics.totalCostUSD;
  }

  /**
   * Generate a comparison summary
   */
  compareTo(other: BenchmarkResult): string {
    const speedAdv = this.getSpeedAdvantage(other);
    const costAdv = this.getCostAdvantage(other);
    const accuracyDiff = ((this.metrics.accuracy - other.metrics.accuracy) * 100).toFixed(1);

    return `${this.systemName} vs ${other.systemName}:
- Speed: ${speedAdv.toFixed(0)}x faster
- Cost: ${costAdv === Infinity ? 'FREE' : costAdv.toFixed(0) + 'x cheaper'}
- Accuracy: ${accuracyDiff > 0 ? '+' : ''}${accuracyDiff}%
- Explainability: ${(this.metrics.explainabilityScore * 100).toFixed(0)}% vs ${(other.metrics.explainabilityScore * 100).toFixed(0)}%`;
  }
}
