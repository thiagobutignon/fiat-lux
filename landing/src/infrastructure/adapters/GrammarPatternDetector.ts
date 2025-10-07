import { IPatternDetector } from '../../domain/repositories/IPatternDetector';
import { CandlestickSequence } from '../../domain/entities/CandlestickSequence';
import { TradingSignal, SignalType } from '../../domain/entities/TradingSignal';
import { Pattern, PatternType, SignalStrength } from '../../domain/entities/Pattern';
import { Candlestick } from '../../domain/entities/Candlestick';

/**
 * Infrastructure Adapter: GrammarPatternDetector
 * Deterministic pattern detection using formal grammar rules
 *
 * This is the "Universal Grammar" approach - 100% explainable,
 * deterministic, and with near-zero latency.
 */
export class GrammarPatternDetector implements IPatternDetector {
  getName(): string {
    return 'Grammar Engine (Fiat Lux)';
  }

  isExplainable(): boolean {
    return true; // 100% explainable - every decision is rule-based
  }

  async detectPatterns(sequence: CandlestickSequence): Promise<TradingSignal> {
    const patterns: Pattern[] = [];
    const candles = sequence.candles;

    // Detect all patterns
    patterns.push(...this.detectSingleCandlePatterns(candles));
    patterns.push(...this.detectTwoCandlePatterns(candles));
    patterns.push(...this.detectThreeCandlePatterns(candles));

    // Generate signal from patterns
    const signal = this.generateSignal(patterns);

    return signal;
  }

  /**
   * Detect single-candle patterns
   */
  private detectSingleCandlePatterns(candles: Candlestick[]): Pattern[] {
    const patterns: Pattern[] = [];

    candles.forEach((candle, index) => {
      // Doji
      if (candle.isDoji(0.1)) {
        patterns.push(new Pattern(
          PatternType.DOJI,
          SignalStrength.WEAK,
          0.85,
          index,
          index,
          'Doji detected: Small body indicates indecision. Open ≈ Close.'
        ));
      }

      // Hammer (bullish)
      if (this.isHammer(candle)) {
        patterns.push(new Pattern(
          PatternType.HAMMER,
          SignalStrength.STRONG,
          0.90,
          index,
          index,
          'Hammer detected: Long lower shadow (2x body), small upper shadow. Bullish reversal signal.'
        ));
      }

      // Inverted Hammer (bearish)
      if (this.isInvertedHammer(candle)) {
        patterns.push(new Pattern(
          PatternType.INVERTED_HAMMER,
          SignalStrength.MODERATE,
          0.80,
          index,
          index,
          'Inverted Hammer detected: Long upper shadow, small lower shadow. Potential bearish reversal.'
        ));
      }

      // Shooting Star (bearish)
      if (this.isShootingStar(candle)) {
        patterns.push(new Pattern(
          PatternType.SHOOTING_STAR,
          SignalStrength.STRONG,
          0.88,
          index,
          index,
          'Shooting Star detected: Small body at bottom, long upper shadow. Bearish reversal signal.'
        ));
      }
    });

    return patterns;
  }

  /**
   * Detect two-candle patterns
   */
  private detectTwoCandlePatterns(candles: Candlestick[]): Pattern[] {
    const patterns: Pattern[] = [];

    for (let i = 1; i < candles.length; i++) {
      const prev = candles[i - 1];
      const curr = candles[i];

      // Bullish Engulfing
      if (prev.isBearish() && curr.isBullish() &&
          curr.open < prev.close && curr.close > prev.open &&
          curr.getBodySize() > prev.getBodySize()) {
        patterns.push(new Pattern(
          PatternType.BULLISH_ENGULFING,
          SignalStrength.STRONG,
          0.92,
          i - 1,
          i,
          'Bullish Engulfing: Large bullish candle completely engulfs previous bearish candle. Strong buy signal.'
        ));
      }

      // Bearish Engulfing
      if (prev.isBullish() && curr.isBearish() &&
          curr.open > prev.close && curr.close < prev.open &&
          curr.getBodySize() > prev.getBodySize()) {
        patterns.push(new Pattern(
          PatternType.BEARISH_ENGULFING,
          SignalStrength.STRONG,
          0.92,
          i - 1,
          i,
          'Bearish Engulfing: Large bearish candle completely engulfs previous bullish candle. Strong sell signal.'
        ));
      }

      // Piercing Line (bullish)
      if (prev.isBearish() && curr.isBullish() &&
          curr.open < prev.low &&
          curr.close > prev.open + (prev.getBodySize() * 0.5)) {
        patterns.push(new Pattern(
          PatternType.PIERCING_LINE,
          SignalStrength.MODERATE,
          0.85,
          i - 1,
          i,
          'Piercing Line: Bullish candle opens below previous low, closes above midpoint. Bullish reversal.'
        ));
      }

      // Dark Cloud Cover (bearish)
      if (prev.isBullish() && curr.isBearish() &&
          curr.open > prev.high &&
          curr.close < prev.open + (prev.getBodySize() * 0.5)) {
        patterns.push(new Pattern(
          PatternType.DARK_CLOUD_COVER,
          SignalStrength.MODERATE,
          0.85,
          i - 1,
          i,
          'Dark Cloud Cover: Bearish candle opens above previous high, closes below midpoint. Bearish reversal.'
        ));
      }
    }

    return patterns;
  }

  /**
   * Detect three-candle patterns
   */
  private detectThreeCandlePatterns(candles: Candlestick[]): Pattern[] {
    const patterns: Pattern[] = [];

    for (let i = 2; i < candles.length; i++) {
      const first = candles[i - 2];
      const second = candles[i - 1];
      const third = candles[i];

      // Morning Star (bullish)
      if (first.isBearish() &&
          second.getBodySize() < first.getBodySize() * 0.3 &&
          third.isBullish() &&
          third.close > first.open + (first.getBodySize() * 0.5)) {
        patterns.push(new Pattern(
          PatternType.MORNING_STAR,
          SignalStrength.STRONG,
          0.93,
          i - 2,
          i,
          'Morning Star: Bearish → Small body → Large bullish. Strong bullish reversal pattern.'
        ));
      }

      // Evening Star (bearish)
      if (first.isBullish() &&
          second.getBodySize() < first.getBodySize() * 0.3 &&
          third.isBearish() &&
          third.close < first.open - (first.getBodySize() * 0.5)) {
        patterns.push(new Pattern(
          PatternType.EVENING_STAR,
          SignalStrength.STRONG,
          0.93,
          i - 2,
          i,
          'Evening Star: Bullish → Small body → Large bearish. Strong bearish reversal pattern.'
        ));
      }

      // Three White Soldiers (bullish)
      if (first.isBullish() && second.isBullish() && third.isBullish() &&
          second.open > first.open && second.close > first.close &&
          third.open > second.open && third.close > second.close) {
        patterns.push(new Pattern(
          PatternType.THREE_WHITE_SOLDIERS,
          SignalStrength.STRONG,
          0.95,
          i - 2,
          i,
          'Three White Soldiers: Three consecutive strong bullish candles. Very strong uptrend signal.'
        ));
      }

      // Three Black Crows (bearish)
      if (first.isBearish() && second.isBearish() && third.isBearish() &&
          second.open < first.open && second.close < first.close &&
          third.open < second.open && third.close < second.close) {
        patterns.push(new Pattern(
          PatternType.THREE_BLACK_CROWS,
          SignalStrength.STRONG,
          0.95,
          i - 2,
          i,
          'Three Black Crows: Three consecutive strong bearish candles. Very strong downtrend signal.'
        ));
      }
    }

    return patterns;
  }

  /**
   * Generate trading signal from detected patterns
   */
  private generateSignal(patterns: Pattern[]): TradingSignal {
    if (patterns.length === 0) {
      return new TradingSignal(
        SignalType.HOLD,
        [],
        new Date(),
        0,
        'No patterns detected. Recommend HOLD.'
      );
    }

    // Count bullish vs bearish patterns weighted by confidence
    let bullishScore = 0;
    let bearishScore = 0;

    patterns.forEach(pattern => {
      const weight = pattern.confidence * (
        pattern.strength === SignalStrength.STRONG ? 3 :
        pattern.strength === SignalStrength.MODERATE ? 2 : 1
      );

      if (pattern.isBullish()) {
        bullishScore += weight;
      } else if (pattern.isBearish()) {
        bearishScore += weight;
      }
    });

    // Determine signal
    const totalScore = bullishScore + bearishScore;
    let signalType: SignalType;
    let confidence: number;
    let explanation: string;

    if (bullishScore > bearishScore * 1.5) {
      signalType = SignalType.BUY;
      confidence = Math.min(0.98, bullishScore / (totalScore * 1.2));
      explanation = `Strong BUY signal. Detected ${patterns.filter(p => p.isBullish()).length} bullish patterns with weighted score ${bullishScore.toFixed(2)}.`;
    } else if (bearishScore > bullishScore * 1.5) {
      signalType = SignalType.SELL;
      confidence = Math.min(0.98, bearishScore / (totalScore * 1.2));
      explanation = `Strong SELL signal. Detected ${patterns.filter(p => p.isBearish()).length} bearish patterns with weighted score ${bearishScore.toFixed(2)}.`;
    } else {
      signalType = SignalType.HOLD;
      confidence = 0.6;
      explanation = `HOLD signal. Mixed patterns detected (Bullish: ${bullishScore.toFixed(2)}, Bearish: ${bearishScore.toFixed(2)}). Unclear direction.`;
    }

    return new TradingSignal(signalType, patterns, new Date(), confidence, explanation);
  }

  /**
   * Helper: Check if candle is a Hammer
   */
  private isHammer(candle: Candlestick): boolean {
    const body = candle.getBodySize();
    const lowerShadow = candle.getLowerShadow();
    const upperShadow = candle.getUpperShadow();

    return lowerShadow >= body * 2 && upperShadow <= body * 0.3;
  }

  /**
   * Helper: Check if candle is an Inverted Hammer
   */
  private isInvertedHammer(candle: Candlestick): boolean {
    const body = candle.getBodySize();
    const lowerShadow = candle.getLowerShadow();
    const upperShadow = candle.getUpperShadow();

    return upperShadow >= body * 2 && lowerShadow <= body * 0.3;
  }

  /**
   * Helper: Check if candle is a Shooting Star
   */
  private isShootingStar(candle: Candlestick): boolean {
    const body = candle.getBodySize();
    const lowerShadow = candle.getLowerShadow();
    const upperShadow = candle.getUpperShadow();
    const range = candle.getRange();

    // Small body at bottom, long upper shadow
    return upperShadow >= body * 2 &&
           lowerShadow <= body * 0.5 &&
           Math.max(candle.open, candle.close) - candle.low < range * 0.3;
  }
}
