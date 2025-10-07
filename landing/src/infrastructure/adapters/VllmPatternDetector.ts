import { IPatternDetector } from '../../domain/repositories/IPatternDetector';
import { CandlestickSequence } from '../../domain/entities/CandlestickSequence';
import { TradingSignal, SignalType } from '../../domain/entities/TradingSignal';

/**
 * Infrastructure Adapter: VllmPatternDetector
 * High-performance LLM inference via vLLM (10-25x faster than vanilla)
 *
 * vLLM features:
 * - PagedAttention for memory optimization
 * - Continuous batching
 * - CUDA optimizations
 * - OpenAI-compatible API
 *
 * Setup:
 * 1. Install: pip install vllm
 * 2. Start server:
 *    python -m vllm.entrypoints.openai.api_server \
 *      --model meta-llama/Meta-Llama-3.1-8B-Instruct \
 *      --port 8000
 * 3. Configure .env:
 *    ENABLE_VLLM=true
 *    VLLM_BASE_URL=http://localhost:8000
 *    VLLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
 */
export class VllmPatternDetector implements IPatternDetector {
  private readonly baseUrl: string;
  private readonly model: string;
  private totalTime: number = 0;
  private requestCount: number = 0;

  constructor(
    baseUrl: string = 'http://localhost:8000',
    model: string = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
  ) {
    this.baseUrl = baseUrl;
    this.model = model;
  }

  getName(): string {
    return `vLLM (${this.model.split('/').pop()})`;
  }

  isExplainable(): boolean {
    return false; // LLMs are black boxes
  }

  async detectPatterns(sequence: CandlestickSequence): Promise<TradingSignal> {
    const startTime = Date.now();

    try {
      const prompt = this.buildPrompt(sequence);

      // vLLM uses OpenAI-compatible API
      const response = await fetch(`${this.baseUrl}/v1/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: this.model,
          prompt,
          max_tokens: 200,
          temperature: 0.1,
          stop: ['\n\n', 'Human:', 'Assistant:'],
        }),
      });

      if (!response.ok) {
        throw new Error(`vLLM API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      const text = data.choices[0]?.text || '';

      this.totalTime += Date.now() - startTime;
      this.requestCount++;

      return this.parseResponse(text);
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);

      // Log first error for debugging
      if (this.requestCount === 0) {
        console.error(`\n❌ vLLM Error: ${errorMsg}`);
        console.error('   Make sure vLLM server is running:');
        console.error('   python -m vllm.entrypoints.openai.api_server --model MODEL_NAME --port 8000\n');
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
   * Build the prompt for vLLM
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

Respond ONLY with valid JSON, no additional text.

JSON Response:`;
  }

  /**
   * Parse vLLM's response into a TradingSignal
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
        `vLLM Analysis: ${reasoning}`
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
        `vLLM Analysis (unparsed): ${text.substring(0, 100)}...`
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
