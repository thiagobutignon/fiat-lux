/**
 * Domain Entity: Candlestick
 * Represents a single candlestick in a trading chart
 */
export class Candlestick {
  constructor(
    public readonly timestamp: Date,
    public readonly open: number,
    public readonly high: number,
    public readonly low: number,
    public readonly close: number,
    public readonly volume: number
  ) {
    this.validate();
  }

  private validate(): void {
    if (this.high < this.low) {
      throw new Error('High price cannot be lower than low price');
    }
    if (this.open < 0 || this.close < 0 || this.high < 0 || this.low < 0) {
      throw new Error('Prices cannot be negative');
    }
    if (this.volume < 0) {
      throw new Error('Volume cannot be negative');
    }
  }

  /**
   * Check if this is a bullish candle (close > open)
   */
  isBullish(): boolean {
    return this.close > this.open;
  }

  /**
   * Check if this is a bearish candle (close < open)
   */
  isBearish(): boolean {
    return this.close < this.open;
  }

  /**
   * Get the body size (absolute difference between open and close)
   */
  getBodySize(): number {
    return Math.abs(this.close - this.open);
  }

  /**
   * Get the upper shadow/wick size
   */
  getUpperShadow(): number {
    return this.high - Math.max(this.open, this.close);
  }

  /**
   * Get the lower shadow/wick size
   */
  getLowerShadow(): number {
    return Math.min(this.open, this.close) - this.low;
  }

  /**
   * Get the total range (high - low)
   */
  getRange(): number {
    return this.high - this.low;
  }

  /**
   * Check if this is a doji (very small body relative to range)
   */
  isDoji(threshold: number = 0.1): boolean {
    const range = this.getRange();
    if (range === 0) return true;
    return this.getBodySize() / range < threshold;
  }
}
