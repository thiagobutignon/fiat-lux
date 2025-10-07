import { Pattern } from './Pattern';

/**
 * Domain Entity: TradingSignal
 * Represents a trading signal generated from pattern analysis
 */

export enum SignalType {
  BUY = 'BUY',
  SELL = 'SELL',
  HOLD = 'HOLD',
}

export class TradingSignal {
  constructor(
    public readonly type: SignalType,
    public readonly patterns: Pattern[],
    public readonly timestamp: Date,
    public readonly confidence: number, // 0-1
    public readonly explanation: string
  ) {
    if (confidence < 0 || confidence > 1) {
      throw new Error('Confidence must be between 0 and 1');
    }
  }

  /**
   * Check if this signal has high confidence
   */
  hasHighConfidence(threshold: number = 0.7): boolean {
    return this.confidence >= threshold;
  }

  /**
   * Get the primary pattern (highest confidence)
   */
  getPrimaryPattern(): Pattern | undefined {
    if (this.patterns.length === 0) return undefined;
    return this.patterns.reduce((highest, current) =>
      current.confidence > highest.confidence ? current : highest
    );
  }

  /**
   * Count bullish patterns
   */
  countBullishPatterns(): number {
    return this.patterns.filter(p => p.isBullish()).length;
  }

  /**
   * Count bearish patterns
   */
  countBearishPatterns(): number {
    return this.patterns.filter(p => p.isBearish()).length;
  }
}
