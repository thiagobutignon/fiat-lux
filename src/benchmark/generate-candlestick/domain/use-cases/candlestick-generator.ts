import { Candlestick } from '../../../_shared/domain/entities/candlestick';
import { CandlestickSequence } from '../../../_shared/domain/entities/candlestick-sequence';
import { SignalType } from '../../../_shared/domain/entities/trading-signal';
import { PatternType } from '../../../_shared/domain/entities/pattern';

/**
 * Infrastructure: CandlestickGenerator
 * Generates synthetic candlestick data for testing
 */

export interface TestCase {
  sequence: CandlestickSequence;
  expectedSignal: SignalType;
  patternType?: PatternType;
}

export class CandlestickGenerator {
  private sequenceCounter = 0;

  /**
   * Generate N test cases with known patterns
   */
  generateTestCases(count: number): TestCase[] {
    const testCases: TestCase[] = [];

    // Distribution of patterns
    const bullishCount = Math.floor(count * 0.35);
    const bearishCount = Math.floor(count * 0.35);
    const neutralCount = count - bullishCount - bearishCount;

    // Generate bullish patterns
    for (let i = 0; i < bullishCount; i++) {
      testCases.push(this.generateBullishPattern());
    }

    // Generate bearish patterns
    for (let i = 0; i < bearishCount; i++) {
      testCases.push(this.generateBearishPattern());
    }

    // Generate neutral/HOLD patterns
    for (let i = 0; i < neutralCount; i++) {
      testCases.push(this.generateNeutralPattern());
    }

    // Shuffle test cases
    return this.shuffle(testCases);
  }

  /**
   * Generate a bullish pattern sequence
   */
  private generateBullishPattern(): TestCase {
    const patterns = [
      () => this.generateHammerPattern(),
      () => this.generateBullishEngulfingPattern(),
      () => this.generateMorningStarPattern(),
      () => this.generateThreeWhiteSoldiersPattern(),
    ];

    const generator = patterns[Math.floor(Math.random() * patterns.length)];
    return generator();
  }

  /**
   * Generate a bearish pattern sequence
   */
  private generateBearishPattern(): TestCase {
    const patterns = [
      () => this.generateShootingStarPattern(),
      () => this.generateBearishEngulfingPattern(),
      () => this.generateEveningStarPattern(),
      () => this.generateThreeBlackCrowsPattern(),
    ];

    const generator = patterns[Math.floor(Math.random() * patterns.length)];
    return generator();
  }

  /**
   * Generate a neutral pattern (no clear signal)
   */
  private generateNeutralPattern(): TestCase {
    const basePrice = 100 + Math.random() * 100;
    const candles: Candlestick[] = [];

    // Generate 10 random candles with small movements
    for (let i = 0; i < 10; i++) {
      const open = basePrice + (Math.random() * 4 - 2);
      const close = open + (Math.random() * 2 - 1);
      const high = Math.max(open, close) + Math.random() * 0.5;
      const low = Math.min(open, close) - Math.random() * 0.5;
      const volume = 1000 + Math.random() * 5000;

      candles.push(new Candlestick(
        new Date(Date.now() + i * 3600000),
        open,
        high,
        low,
        close,
        volume
      ));
    }

    return {
      sequence: new CandlestickSequence(`seq_${this.sequenceCounter++}`, candles),
      expectedSignal: SignalType.HOLD,
    };
  }

  // Pattern generators

  private generateHammerPattern(): TestCase {
    const basePrice = 100 + Math.random() * 100;
    const candles: Candlestick[] = [];

    // Generate context (downtrend)
    for (let i = 0; i < 5; i++) {
      const open = basePrice - i * 2;
      const close = open - 1.5;
      const high = open + 0.3;
      const low = close - 0.3;
      candles.push(new Candlestick(
        new Date(Date.now() + i * 3600000),
        open,
        high,
        low,
        close,
        1000 + Math.random() * 1000
      ));
    }

    // Hammer candle
    const hammerOpen = basePrice - 10;
    const hammerClose = hammerOpen + 0.5;
    const hammerHigh = hammerClose + 0.2;
    const hammerLow = hammerOpen - 2.5; // Long lower shadow
    candles.push(new Candlestick(
      new Date(Date.now() + 5 * 3600000),
      hammerOpen,
      hammerHigh,
      hammerLow,
      hammerClose,
      2000
    ));

    return {
      sequence: new CandlestickSequence(`seq_${this.sequenceCounter++}`, candles),
      expectedSignal: SignalType.BUY,
      patternType: PatternType.HAMMER,
    };
  }

  private generateShootingStarPattern(): TestCase {
    const basePrice = 100 + Math.random() * 100;
    const candles: Candlestick[] = [];

    // Generate context (uptrend)
    for (let i = 0; i < 5; i++) {
      const open = basePrice + i * 2;
      const close = open + 1.5;
      const high = close + 0.3;
      const low = open - 0.3;
      candles.push(new Candlestick(
        new Date(Date.now() + i * 3600000),
        open,
        high,
        low,
        close,
        1000 + Math.random() * 1000
      ));
    }

    // Shooting star candle
    const starOpen = basePrice + 10;
    const starClose = starOpen - 0.5;
    const starHigh = starOpen + 2.5; // Long upper shadow
    const starLow = starClose - 0.2;
    candles.push(new Candlestick(
      new Date(Date.now() + 5 * 3600000),
      starOpen,
      starHigh,
      starLow,
      starClose,
      2000
    ));

    return {
      sequence: new CandlestickSequence(`seq_${this.sequenceCounter++}`, candles),
      expectedSignal: SignalType.SELL,
      patternType: PatternType.SHOOTING_STAR,
    };
  }

  private generateBullishEngulfingPattern(): TestCase {
    const basePrice = 100 + Math.random() * 100;
    const candles: Candlestick[] = [];

    // Context candles
    for (let i = 0; i < 4; i++) {
      const open = basePrice - i;
      const close = open - 1;
      candles.push(new Candlestick(
        new Date(Date.now() + i * 3600000),
        open,
        open + 0.3,
        close - 0.3,
        close,
        1000
      ));
    }

    // Bearish candle
    const bearishOpen = basePrice - 4;
    const bearishClose = bearishOpen - 2;
    candles.push(new Candlestick(
      new Date(Date.now() + 4 * 3600000),
      bearishOpen,
      bearishOpen + 0.2,
      bearishClose - 0.2,
      bearishClose,
      1200
    ));

    // Bullish engulfing candle
    const bullishOpen = bearishClose - 0.3;
    const bullishClose = bearishOpen + 0.5;
    candles.push(new Candlestick(
      new Date(Date.now() + 5 * 3600000),
      bullishOpen,
      bullishClose + 0.2,
      bullishOpen - 0.2,
      bullishClose,
      2500
    ));

    return {
      sequence: new CandlestickSequence(`seq_${this.sequenceCounter++}`, candles),
      expectedSignal: SignalType.BUY,
      patternType: PatternType.BULLISH_ENGULFING,
    };
  }

  private generateBearishEngulfingPattern(): TestCase {
    const basePrice = 100 + Math.random() * 100;
    const candles: Candlestick[] = [];

    // Context candles (uptrend)
    for (let i = 0; i < 4; i++) {
      const open = basePrice + i;
      const close = open + 1;
      candles.push(new Candlestick(
        new Date(Date.now() + i * 3600000),
        open,
        close + 0.3,
        open - 0.3,
        close,
        1000
      ));
    }

    // Bullish candle
    const bullishOpen = basePrice + 4;
    const bullishClose = bullishOpen + 2;
    candles.push(new Candlestick(
      new Date(Date.now() + 4 * 3600000),
      bullishOpen,
      bullishClose + 0.2,
      bullishOpen - 0.2,
      bullishClose,
      1200
    ));

    // Bearish engulfing candle
    const bearishOpen = bullishClose + 0.3;
    const bearishClose = bullishOpen - 0.5;
    candles.push(new Candlestick(
      new Date(Date.now() + 5 * 3600000),
      bearishOpen,
      bearishOpen + 0.2,
      bearishClose - 0.2,
      bearishClose,
      2500
    ));

    return {
      sequence: new CandlestickSequence(`seq_${this.sequenceCounter++}`, candles),
      expectedSignal: SignalType.SELL,
      patternType: PatternType.BEARISH_ENGULFING,
    };
  }

  private generateMorningStarPattern(): TestCase {
    const basePrice = 100 + Math.random() * 100;
    const candles: Candlestick[] = [];

    // Context (downtrend)
    for (let i = 0; i < 3; i++) {
      const open = basePrice - i * 2;
      const close = open - 1.5;
      candles.push(new Candlestick(
        new Date(Date.now() + i * 3600000),
        open,
        open + 0.3,
        close - 0.3,
        close,
        1000
      ));
    }

    // First star candle (large bearish)
    const first = basePrice - 6;
    candles.push(new Candlestick(
      new Date(Date.now() + 3 * 3600000),
      first,
      first + 0.3,
      first - 3,
      first - 2.7,
      1500
    ));

    // Second star candle (small body)
    const second = first - 3;
    candles.push(new Candlestick(
      new Date(Date.now() + 4 * 3600000),
      second,
      second + 0.3,
      second - 0.5,
      second - 0.2,
      800
    ));

    // Third star candle (large bullish)
    const third = second - 0.3;
    candles.push(new Candlestick(
      new Date(Date.now() + 5 * 3600000),
      third,
      third + 3,
      third - 0.3,
      third + 2.7,
      2000
    ));

    return {
      sequence: new CandlestickSequence(`seq_${this.sequenceCounter++}`, candles),
      expectedSignal: SignalType.BUY,
      patternType: PatternType.MORNING_STAR,
    };
  }

  private generateEveningStarPattern(): TestCase {
    const basePrice = 100 + Math.random() * 100;
    const candles: Candlestick[] = [];

    // Context (uptrend)
    for (let i = 0; i < 3; i++) {
      const open = basePrice + i * 2;
      const close = open + 1.5;
      candles.push(new Candlestick(
        new Date(Date.now() + i * 3600000),
        open,
        close + 0.3,
        open - 0.3,
        close,
        1000
      ));
    }

    // First star candle (large bullish)
    const first = basePrice + 6;
    candles.push(new Candlestick(
      new Date(Date.now() + 3 * 3600000),
      first,
      first + 3,
      first - 0.3,
      first + 2.7,
      1500
    ));

    // Second star candle (small body)
    const second = first + 3;
    candles.push(new Candlestick(
      new Date(Date.now() + 4 * 3600000),
      second,
      second + 0.5,
      second - 0.3,
      second + 0.2,
      800
    ));

    // Third star candle (large bearish)
    const third = second + 0.3;
    candles.push(new Candlestick(
      new Date(Date.now() + 5 * 3600000),
      third,
      third + 0.3,
      third - 3,
      third - 2.7,
      2000
    ));

    return {
      sequence: new CandlestickSequence(`seq_${this.sequenceCounter++}`, candles),
      expectedSignal: SignalType.SELL,
      patternType: PatternType.EVENING_STAR,
    };
  }

  private generateThreeWhiteSoldiersPattern(): TestCase {
    const basePrice = 100 + Math.random() * 100;
    const candles: Candlestick[] = [];

    // Context
    for (let i = 0; i < 3; i++) {
      const open = basePrice - i;
      const close = open - 0.5;
      candles.push(new Candlestick(
        new Date(Date.now() + i * 3600000),
        open,
        open + 0.2,
        close - 0.2,
        close,
        1000
      ));
    }

    // Three white soldiers
    for (let i = 0; i < 3; i++) {
      const open = basePrice - 3 + i * 2;
      const close = open + 2.5;
      candles.push(new Candlestick(
        new Date(Date.now() + (3 + i) * 3600000),
        open,
        close + 0.2,
        open - 0.2,
        close,
        1500 + i * 200
      ));
    }

    return {
      sequence: new CandlestickSequence(`seq_${this.sequenceCounter++}`, candles),
      expectedSignal: SignalType.BUY,
      patternType: PatternType.THREE_WHITE_SOLDIERS,
    };
  }

  private generateThreeBlackCrowsPattern(): TestCase {
    const basePrice = 100 + Math.random() * 100;
    const candles: Candlestick[] = [];

    // Context (uptrend)
    for (let i = 0; i < 3; i++) {
      const open = basePrice + i;
      const close = open + 0.5;
      candles.push(new Candlestick(
        new Date(Date.now() + i * 3600000),
        open,
        close + 0.2,
        open - 0.2,
        close,
        1000
      ));
    }

    // Three black crows
    for (let i = 0; i < 3; i++) {
      const open = basePrice + 3 - i * 2;
      const close = open - 2.5;
      candles.push(new Candlestick(
        new Date(Date.now() + (3 + i) * 3600000),
        open,
        open + 0.2,
        close - 0.2,
        close,
        1500 + i * 200
      ));
    }

    return {
      sequence: new CandlestickSequence(`seq_${this.sequenceCounter++}`, candles),
      expectedSignal: SignalType.SELL,
      patternType: PatternType.THREE_BLACK_CROWS,
    };
  }

  /**
   * Shuffle array (Fisher-Yates algorithm)
   */
  private shuffle<T>(array: T[]): T[] {
    const result = [...array];
    for (let i = result.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [result[i], result[j]] = [result[j], result[i]];
    }
    return result;
  }
}
