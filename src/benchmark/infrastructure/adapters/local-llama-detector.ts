import { IPatternDetector } from '../../data/protocols/pattern-detector';
import { CandlestickSequence } from '../../domain/entities/candlestick-sequence';
import { TradingSignal, SignalType } from '../../domain/entities/trading-signal';

/**
 * Infrastructure Adapter: LocalLlamaDetector
 * Real integration with local Llama via Ollama
 *
 * Requires Ollama running locally: https://ollama.ai/
 * Install: curl -fsSL https://ollama.com/install.sh | sh
 * Run: ollama run llama3.1:70b
 */
export class LocalLlamaDetector implements IPatternDetector {
  private readonly baseUrl: string;
  private readonly model: string;
  private totalTime: number = 0;
  private requestCount: number = 0;

  constructor(
    baseUrl: string = 'http://localhost:11434',
    model: string = 'llama3.1:8b'  // Default to 8B for speed, can use :70b for better accuracy
  ) {
    this.baseUrl = baseUrl;
    this.model = model;
  }

  getName(): string {
    return `Local Llama (${this.model})`;
  }

  isExplainable(): boolean {
    return false; // LLMs are black boxes
  }

  async detectPatterns(sequence: CandlestickSequence): Promise<TradingSignal> {
    const startTime = Date.now();

    try {
      const prompt = this.buildPrompt(sequence);

      const response = await fetch(`${this.baseUrl}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: this.model,
          prompt,
          stream: false,
          options: {
            temperature: 0.1,
            num_predict: 500,
          },
        }),
      });

      if (!response.ok) {
        throw new Error(`Ollama API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      const text = data.response;

      this.totalTime += Date.now() - startTime;
      this.requestCount++;

      return this.parseResponse(text);
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);

      // Log first error for debugging
      if (this.requestCount === 0) {
        console.error(`\n❌ Ollama Error: ${errorMsg}`);
        console.error('   Make sure Ollama is running: ollama serve');
        console.error(`   And model is available: ollama pull ${this.model}\n`);
      }

      // Return default signal on error
      return new TradingSignal(
        SignalType.HOLD,
        [],
        new Date(),
        0.0,
        `Error: ${errorMsg}`
      );
    }
  }

  /**
   * Build the prompt for Llama
   */
  private buildPrompt(sequence: CandlestickSequence): string {
    const candles = sequence.candles.slice(-5);

    const candleData = candles.map((c, i) => {
      return `Candle ${i + 1}: Open=${c.open.toFixed(2)}, High=${c.high.toFixed(2)}, Low=${c.low.toFixed(2)}, Close=${c.close.toFixed(2)}`;
    }).join('\n');

    return `You are an expert technical analyst specializing in candlestick pattern recognition.

Analyze the following candlestick sequence and generate a trading signal.

Candlestick Data:
${candleData}

Instructions:
1. Identify any candlestick patterns (e.g., Hammer, Doji, Engulfing, Morning Star, etc.)
2. Based on the patterns, determine if this is a BUY, SELL, or HOLD signal
3. Provide your confidence level (0.0 to 1.0)

Response Format (JSON):
{
  "signal": "BUY" | "SELL" | "HOLD",
  "confidence": 0.0-1.0,
  "patterns": ["pattern1", "pattern2"],
  "reasoning": "Brief explanation"
}

Respond ONLY with valid JSON, no additional text.`;
  }

  /**
   * Parse Llama's response into a TradingSignal
   */
  private parseResponse(text: string): TradingSignal {
    try {
      // Try to extract JSON from the response
      const jsonMatch = text.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        throw new Error('No JSON found in response');
      }

      const parsed = JSON.parse(jsonMatch[0]);

      const signalType = this.parseSignalType(parsed.signal);
      const confidence = Math.max(0, Math.min(1, parsed.confidence || 0.5));
      const patterns = Array.isArray(parsed.patterns) ? parsed.patterns : [];
      const reasoning = parsed.reasoning || 'No reasoning provided';

      return new TradingSignal(
        signalType,
        patterns,
        new Date(),
        confidence,
        `Llama Analysis: ${reasoning}`
      );
    } catch (error) {
      // Fallback: Try to extract signal from text
      const lowerText = text.toLowerCase();
      let signalType = SignalType.HOLD;

      if (lowerText.includes('buy') || lowerText.includes('bullish')) {
        signalType = SignalType.BUY;
      } else if (lowerText.includes('sell') || lowerText.includes('bearish')) {
        signalType = SignalType.SELL;
      }

      return new TradingSignal(
        signalType,
        [],
        new Date(),
        0.5,
        `Llama Analysis (unparsed): ${text.substring(0, 100)}...`
      );
    }
  }

  /**
   * Parse signal type string
   */
  private parseSignalType(signal: string): SignalType {
    const normalized = signal?.toUpperCase();

    switch (normalized) {
      case 'BUY':
        return SignalType.BUY;
      case 'SELL':
        return SignalType.SELL;
      case 'HOLD':
      default:
        return SignalType.HOLD;
    }
  }

  /**
   * Get average latency
   */
  getAvgLatency(): number {
    return this.requestCount > 0 ? this.totalTime / this.requestCount : 0;
  }

  /**
   * Cost is zero for local inference
   */
  getCost(): number {
    return 0;
  }
}
