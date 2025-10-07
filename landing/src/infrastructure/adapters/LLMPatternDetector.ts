import { IPatternDetector } from '../../domain/repositories/IPatternDetector';
import { CandlestickSequence } from '../../domain/entities/CandlestickSequence';
import { TradingSignal, SignalType } from '../../domain/entities/TradingSignal';

/**
 * Infrastructure Adapter: LLMPatternDetector
 * Simulates LLM-based pattern detection (GPT-4, Claude, Llama)
 *
 * This is a mock implementation that simulates the behavior of LLM systems:
 * - Non-deterministic (adds randomness)
 * - Higher latency
 * - Cost per API call
 * - Not explainable (black box)
 * - Lower accuracy (87-89% vs 98%)
 */
export class LLMPatternDetector implements IPatternDetector {
  constructor(
    private readonly modelName: string,
    private readonly avgLatencyMs: number,
    private readonly costPerCall: number,
    private readonly baseAccuracy: number
  ) {}

  getName(): string {
    return this.modelName;
  }

  isExplainable(): boolean {
    return false; // LLMs are black boxes
  }

  async detectPatterns(sequence: CandlestickSequence): Promise<TradingSignal> {
    // Simulate API latency
    await this.simulateLatency();

    // Simulate non-deterministic behavior with occasional errors
    const accuracy = this.baseAccuracy + (Math.random() * 0.1 - 0.05);
    const shouldError = Math.random() > accuracy;

    if (shouldError) {
      // Simulate incorrect pattern detection
      return this.generateRandomSignal(sequence);
    }

    // Generate a somewhat accurate signal (but still not perfect)
    return this.generateSemiAccurateSignal(sequence);
  }

  /**
   * Simulate network latency and LLM processing time
   */
  private async simulateLatency(): Promise<void> {
    // Add random variance (±20%)
    const variance = this.avgLatencyMs * 0.2;
    const latency = this.avgLatencyMs + (Math.random() * variance * 2 - variance);

    await new Promise(resolve => setTimeout(resolve, latency));
  }

  /**
   * Generate a random (incorrect) signal
   */
  private generateRandomSignal(_sequence: CandlestickSequence): TradingSignal {
    const signalTypes = [SignalType.BUY, SignalType.SELL, SignalType.HOLD];
    const randomSignal = signalTypes[Math.floor(Math.random() * signalTypes.length)];

    return new TradingSignal(
      randomSignal,
      [],
      new Date(),
      Math.random() * 0.5 + 0.3, // Random confidence 0.3-0.8
      `${this.modelName} analysis suggests ${randomSignal}. (Simulated - potentially incorrect)`
    );
  }

  /**
   * Generate a semi-accurate signal (not perfect, but reasonable)
   */
  private generateSemiAccurateSignal(sequence: CandlestickSequence): TradingSignal {
    const candles = sequence.candles;
    const lastThree = candles.slice(-3);

    // Simple heuristic (not as good as grammar-based)
    let bullishCount = 0;
    let bearishCount = 0;

    lastThree.forEach(candle => {
      if (candle.isBullish()) bullishCount++;
      if (candle.isBearish()) bearishCount++;
    });

    let signalType: SignalType;
    if (bullishCount > bearishCount) {
      signalType = SignalType.BUY;
    } else if (bearishCount > bullishCount) {
      signalType = SignalType.SELL;
    } else {
      signalType = SignalType.HOLD;
    }

    // LLM sometimes gets confused with weak patterns
    if (Math.random() < 0.15) {
      // 15% chance of confusion
      const types = [SignalType.BUY, SignalType.SELL, SignalType.HOLD];
      signalType = types[Math.floor(Math.random() * types.length)];
    }

    return new TradingSignal(
      signalType,
      [],
      new Date(),
      Math.random() * 0.3 + 0.6, // Confidence 0.6-0.9
      `${this.modelName} analysis. Based on recent price action and patterns observed. (Black box - not explainable)`
    );
  }

  /**
   * Get the cost of this API call
   */
  getCost(): number {
    return this.costPerCall;
  }
}

/**
 * Factory functions for creating specific LLM detectors
 */

export function createGPT4Detector(): LLMPatternDetector {
  return new LLMPatternDetector(
    'GPT-4',
    350, // 350ms avg latency
    0.0005, // $0.50 per 1000 calls
    0.87 // 87% base accuracy
  );
}

export function createClaudeDetector(): LLMPatternDetector {
  return new LLMPatternDetector(
    'Claude 3.5 Sonnet',
    280, // 280ms avg latency
    0.00045, // $0.45 per 1000 calls
    0.89 // 89% base accuracy
  );
}

export function createLlamaDetector(): LLMPatternDetector {
  return new LLMPatternDetector(
    'Fine-tuned Llama 3.1 70B',
    120, // 120ms avg latency (local)
    0.00005, // $0.05 per 1000 calls (local compute cost)
    0.82 // 82% base accuracy
  );
}
