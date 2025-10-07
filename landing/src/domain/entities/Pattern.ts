/**
 * Domain Entity: Pattern
 * Represents a detected candlestick pattern
 */

export enum PatternType {
  DOJI = 'DOJI',
  HAMMER = 'HAMMER',
  INVERTED_HAMMER = 'INVERTED_HAMMER',
  SHOOTING_STAR = 'SHOOTING_STAR',
  BULLISH_ENGULFING = 'BULLISH_ENGULFING',
  BEARISH_ENGULFING = 'BEARISH_ENGULFING',
  MORNING_STAR = 'MORNING_STAR',
  EVENING_STAR = 'EVENING_STAR',
  THREE_WHITE_SOLDIERS = 'THREE_WHITE_SOLDIERS',
  THREE_BLACK_CROWS = 'THREE_BLACK_CROWS',
  PIERCING_LINE = 'PIERCING_LINE',
  DARK_CLOUD_COVER = 'DARK_CLOUD_COVER',
}

export enum SignalStrength {
  WEAK = 'WEAK',
  MODERATE = 'MODERATE',
  STRONG = 'STRONG',
}

export class Pattern {
  constructor(
    public readonly type: PatternType,
    public readonly strength: SignalStrength,
    public readonly confidence: number, // 0-1
    public readonly startIndex: number,
    public readonly endIndex: number,
    public readonly explanation?: string
  ) {
    if (confidence < 0 || confidence > 1) {
      throw new Error('Confidence must be between 0 and 1');
    }
    if (startIndex > endIndex) {
      throw new Error('Start index cannot be greater than end index');
    }
  }

  /**
   * Get the number of candles involved in this pattern
   */
  getCandleCount(): number {
    return this.endIndex - this.startIndex + 1;
  }

  /**
   * Check if this pattern is bullish
   */
  isBullish(): boolean {
    return [
      PatternType.HAMMER,
      PatternType.BULLISH_ENGULFING,
      PatternType.MORNING_STAR,
      PatternType.THREE_WHITE_SOLDIERS,
      PatternType.PIERCING_LINE,
    ].includes(this.type);
  }

  /**
   * Check if this pattern is bearish
   */
  isBearish(): boolean {
    return [
      PatternType.SHOOTING_STAR,
      PatternType.INVERTED_HAMMER,
      PatternType.BEARISH_ENGULFING,
      PatternType.EVENING_STAR,
      PatternType.THREE_BLACK_CROWS,
      PatternType.DARK_CLOUD_COVER,
    ].includes(this.type);
  }
}
