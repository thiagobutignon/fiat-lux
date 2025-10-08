/**
 * Weight Analyzer
 * Statistical analysis of tensor weights
 */

import { TensorData, WeightStatistics } from '../entities/tensor-data';

export class WeightAnalyzer {
  /**
   * Analyze weight distribution and statistics
   */
  analyze(tensor: TensorData): WeightStatistics {
    const data = tensor.data;
    const n = data.length;

    // Basic statistics
    let sum = 0;
    let sumSq = 0;
    let min = Infinity;
    let max = -Infinity;
    let zeros = 0;
    let absSum = 0;

    for (let i = 0; i < n; i++) {
      const value = data[i];
      sum += value;
      sumSq += value * value;
      absSum += Math.abs(value);

      if (value < min) min = value;
      if (value > max) max = value;

      // Count near-zero values (threshold: 1e-7)
      if (Math.abs(value) < 1e-7) {
        zeros++;
      }
    }

    const mean = sum / n;
    const variance = (sumSq / n) - (mean * mean);
    const stdDev = Math.sqrt(Math.max(0, variance)); // Handle numerical errors

    // Calculate norms
    const l1Norm = absSum;
    const l2Norm = Math.sqrt(sumSq);

    // Calculate median (requires sorting)
    const sorted = new Float32Array(data).sort();
    const median = n % 2 === 0
      ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2
      : sorted[Math.floor(n / 2)];

    // Sparsity
    const sparsity = zeros / n;

    // Create histogram
    const histogram = this.createHistogram(data, 50);

    return {
      mean,
      stdDev,
      variance,
      min,
      max,
      median,
      zeros,
      sparsity,
      l1Norm,
      l2Norm,
      histogram,
    };
  }

  /**
   * Create histogram with specified number of bins
   */
  private createHistogram(
    data: Float32Array,
    numBins: number
  ): { bins: number[]; counts: number[] } {
    // Find range
    let min = Infinity;
    let max = -Infinity;

    for (let i = 0; i < data.length; i++) {
      if (data[i] < min) min = data[i];
      if (data[i] > max) max = data[i];
    }

    // Handle edge case: all values are the same
    if (min === max) {
      return {
        bins: [min],
        counts: [data.length],
      };
    }

    // Create bins
    const bins: number[] = [];
    const counts: number[] = new Array(numBins).fill(0);
    const binWidth = (max - min) / numBins;

    for (let i = 0; i < numBins; i++) {
      bins.push(min + (i + 0.5) * binWidth);
    }

    // Fill histogram
    for (let i = 0; i < data.length; i++) {
      const value = data[i];
      const binIndex = Math.min(
        numBins - 1,
        Math.floor((value - min) / binWidth)
      );
      counts[binIndex]++;
    }

    return { bins, counts };
  }

  /**
   * Compare two tensors (useful for quantization quality)
   */
  compareTensors(
    original: TensorData,
    quantized: TensorData
  ): {
    mse: number;
    rmse: number;
    maxError: number;
    correlation: number;
  } {
    if (original.data.length !== quantized.data.length) {
      throw new Error('Tensors must have same length for comparison');
    }

    const n = original.data.length;
    let sumSqError = 0;
    let maxError = 0;
    let sumX = 0;
    let sumY = 0;
    let sumXY = 0;
    let sumX2 = 0;
    let sumY2 = 0;

    for (let i = 0; i < n; i++) {
      const x = original.data[i];
      const y = quantized.data[i];
      const error = x - y;

      sumSqError += error * error;
      maxError = Math.max(maxError, Math.abs(error));

      // For correlation
      sumX += x;
      sumY += y;
      sumXY += x * y;
      sumX2 += x * x;
      sumY2 += y * y;
    }

    const mse = sumSqError / n;
    const rmse = Math.sqrt(mse);

    // Pearson correlation
    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt(
      (n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY)
    );
    const correlation = denominator === 0 ? 0 : numerator / denominator;

    return {
      mse,
      rmse,
      maxError,
      correlation,
    };
  }

  /**
   * Find outliers (values beyond N standard deviations)
   */
  findOutliers(
    tensor: TensorData,
    numStdDevs: number = 3
  ): {
    indices: number[];
    values: number[];
    count: number;
  } {
    const stats = this.analyze(tensor);
    const threshold = numStdDevs * stats.stdDev;
    const lowerBound = stats.mean - threshold;
    const upperBound = stats.mean + threshold;

    const indices: number[] = [];
    const values: number[] = [];

    for (let i = 0; i < tensor.data.length; i++) {
      const value = tensor.data[i];
      if (value < lowerBound || value > upperBound) {
        indices.push(i);
        values.push(value);
      }
    }

    return {
      indices,
      values,
      count: indices.length,
    };
  }

  /**
   * Calculate weight magnitude distribution
   */
  getMagnitudeDistribution(tensor: TensorData): {
    percentiles: { p50: number; p90: number; p95: number; p99: number };
    topKMagnitudes: number[];
  } {
    const magnitudes = new Float32Array(tensor.data.length);
    for (let i = 0; i < tensor.data.length; i++) {
      magnitudes[i] = Math.abs(tensor.data[i]);
    }

    magnitudes.sort((a, b) => a - b);

    const n = magnitudes.length;
    const p50 = magnitudes[Math.floor(n * 0.50)];
    const p90 = magnitudes[Math.floor(n * 0.90)];
    const p95 = magnitudes[Math.floor(n * 0.95)];
    const p99 = magnitudes[Math.floor(n * 0.99)];

    // Get top 100 magnitudes
    const topKMagnitudes = Array.from(magnitudes.slice(-100)).reverse();

    return {
      percentiles: { p50, p90, p95, p99 },
      topKMagnitudes,
    };
  }

  /**
   * Format statistics for display
   */
  formatStats(stats: WeightStatistics): string {
    let output = '';
    output += `  Mean:      ${stats.mean.toFixed(6)}\n`;
    output += `  Std Dev:   ${stats.stdDev.toFixed(6)}\n`;
    output += `  Min:       ${stats.min.toFixed(6)}\n`;
    output += `  Max:       ${stats.max.toFixed(6)}\n`;
    output += `  Median:    ${stats.median.toFixed(6)}\n`;
    output += `  L1 Norm:   ${stats.l1Norm.toExponential(3)}\n`;
    output += `  L2 Norm:   ${stats.l2Norm.toExponential(3)}\n`;
    output += `  Zeros:     ${stats.zeros.toLocaleString()} (${(stats.sparsity * 100).toFixed(2)}%)\n`;

    return output;
  }
}
