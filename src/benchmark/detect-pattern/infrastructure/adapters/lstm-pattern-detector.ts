import { IPatternDetector } from '../../data/protocols/pattern-detector';
import { CandlestickSequence } from '../../../_shared/domain/entities/candlestick-sequence';
import { TradingSignal, SignalType } from '../../../_shared/domain/entities/trading-signal';

/**
 * Infrastructure Adapter: LSTMPatternDetector
 * Simulates traditional ML (LSTM) pattern detection
 *
 * This is a baseline ML approach:
 * - Trained on historical data
 * - Lower accuracy than LLMs (75%)
 * - Faster than LLMs but slower than grammar
 * - Not explainable (black box neural network)
 * - Low cost (local inference)
 */
export class LSTMPatternDetector implements IPatternDetector {
  private readonly avgLatencyMs = 45;
  private readonly costPerCall = 0.00001; // $0.01 per 1000 calls
  private readonly baseAccuracy = 0.75;

  getName(): string {
    return 'Custom LSTM';
  }

  isExplainable(): boolean {
    return false; // Neural networks are black boxes
  }

  async detectPatterns(sequence: CandlestickSequence): Promise<TradingSignal> {
    // Simulate inference latency
    await this.simulateLatency();

    // LSTM has the lowest accuracy of all systems
    const shouldError = Math.random() > this.baseAccuracy;

    if (shouldError) {
      return this.generateRandomSignal();
    }

    return this.generateBasicSignal(sequence);
  }

  private async simulateLatency(): Promise<void> {
    const variance = this.avgLatencyMs * 0.2;
    const latency = this.avgLatencyMs + (Math.random() * variance * 2 - variance);
    await new Promise(resolve => setTimeout(resolve, latency));
  }

  private generateRandomSignal(): TradingSignal {
    const types = [SignalType.BUY, SignalType.SELL, SignalType.HOLD];
    const randomType = types[Math.floor(Math.random() * types.length)];

    return new TradingSignal(
      randomType,
      [],
      new Date(),
      Math.random() * 0.4 + 0.3, // Confidence 0.3-0.7 (lower than LLMs)
      'LSTM prediction based on learned patterns. (Black box - not explainable)'
    );
  }

  private generateBasicSignal(sequence: CandlestickSequence): TradingSignal {
    // Very simple heuristic (LSTM learned patterns but not as sophisticated)
    const last = sequence.candles[sequence.candles.length - 1];

    let signalType: SignalType;
    if (last.isBullish() && last.getBodySize() > sequence.getAverageRange() * 0.5) {
      signalType = SignalType.BUY;
    } else if (last.isBearish() && last.getBodySize() > sequence.getAverageRange() * 0.5) {
      signalType = SignalType.SELL;
    } else {
      signalType = SignalType.HOLD;
    }

    return new TradingSignal(
      signalType,
      [],
      new Date(),
      Math.random() * 0.3 + 0.5, // Confidence 0.5-0.8
      'LSTM model prediction. Trained on historical candlestick data. (Black box)'
    );
  }

  getCost(): number {
    return this.costPerCall;
  }
}
