import { Candlestick } from './Candlestick';

/**
 * Domain Entity: CandlestickSequence
 * Represents a sequence of candlesticks for pattern analysis
 */
export class CandlestickSequence {
  constructor(
    public readonly id: string,
    public readonly candles: Candlestick[]
  ) {
    if (candles.length === 0) {
      throw new Error('Candlestick sequence cannot be empty');
    }
  }

  /**
   * Get the last N candles
   */
  getLastN(n: number): Candlestick[] {
    return this.candles.slice(-n);
  }

  /**
   * Get a candle at a specific index
   */
  getCandle(index: number): Candlestick | undefined {
    return this.candles[index];
  }

  /**
   * Get the length of the sequence
   */
  length(): number {
    return this.candles.length;
  }

  /**
   * Calculate average volume over the sequence
   */
  getAverageVolume(): number {
    const sum = this.candles.reduce((acc, candle) => acc + candle.volume, 0);
    return sum / this.candles.length;
  }

  /**
   * Calculate average range over the sequence
   */
  getAverageRange(): number {
    const sum = this.candles.reduce((acc, candle) => acc + candle.getRange(), 0);
    return sum / this.candles.length;
  }
}
