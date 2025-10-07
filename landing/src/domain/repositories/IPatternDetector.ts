import { CandlestickSequence } from '../entities/CandlestickSequence';
import { TradingSignal } from '../entities/TradingSignal';

/**
 * Domain Repository Interface: IPatternDetector
 * Defines the contract for pattern detection systems
 */
export interface IPatternDetector {
  /**
   * Detect patterns and generate trading signals from a candlestick sequence
   */
  detectPatterns(sequence: CandlestickSequence): Promise<TradingSignal>;

  /**
   * Get the name of this detector
   */
  getName(): string;

  /**
   * Check if this detector provides explainable results
   */
  isExplainable(): boolean;
}
