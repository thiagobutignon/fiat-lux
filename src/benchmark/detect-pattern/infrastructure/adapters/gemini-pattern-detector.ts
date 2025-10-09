import { GoogleGenerativeAI } from '@google/generative-ai';
import { IPatternDetector } from '../../data/protocols/pattern-detector';
import { CandlestickSequence } from '../../../_shared/domain/entities/candlestick-sequence';
import { TradingSignal, SignalType } from '../../../_shared/domain/entities/trading-signal';

/**
 * Infrastructure Adapter: GeminiPatternDetector
 * Real integration with Google's Gemini 2.5 Flash API
 *
 * This adapter uses the actual Gemini API to detect candlestick patterns
 * and generate trading signals, demonstrating real-world LLM performance.
 */
/**
 * Simple rate limiter to respect API limits
 */
class RateLimiter {
  private requestTimes: number[] = [];
  private readonly maxRequests: number;
  private readonly windowMs: number;

  constructor(maxRequestsPerMinute: number = 15) {
    this.maxRequests = maxRequestsPerMinute;
    this.windowMs = 60000; // 1 minute
  }

  async waitIfNeeded(): Promise<void> {
    const now = Date.now();

    // Remove requests outside the time window
    this.requestTimes = this.requestTimes.filter(time => now - time < this.windowMs);

    // If at limit, wait
    if (this.requestTimes.length >= this.maxRequests) {
      const oldestRequest = this.requestTimes[0];
      const waitTime = this.windowMs - (now - oldestRequest) + 100; // Add 100ms buffer

      if (waitTime > 0) {
        await new Promise(resolve => setTimeout(resolve, waitTime));
      }

      // Retry after waiting
      return this.waitIfNeeded();
    }

    // Record this request
    this.requestTimes.push(now);
  }
}

export class GeminiPatternDetector implements IPatternDetector {
  private readonly client: GoogleGenerativeAI;
  private readonly model: string;
  private totalCost: number = 0;
  private readonly rateLimiter: RateLimiter;

  constructor(apiKey?: string, maxRequestsPerMinute: number = 15) {
    const key = apiKey || process.env.GEMINI_API_KEY;

    if (!key) {
      throw new Error(
        'GEMINI_API_KEY not found. Please set it in your .env file or pass it to the constructor.\n' +
        'Get your API key from: https://aistudio.google.com/apikey'
      );
    }

    this.client = new GoogleGenerativeAI(key);
    this.model = 'gemini-2.5-flash'; // Fast and cost-effective
    this.rateLimiter = new RateLimiter(maxRequestsPerMinute);
  }

  getName(): string {
    return 'Gemini 2.0 Flash';
  }

  isExplainable(): boolean {
    return false; // LLMs are black boxes
  }

  async detectPatterns(sequence: CandlestickSequence): Promise<TradingSignal> {
    // Respect rate limits
    await this.rateLimiter.waitIfNeeded();

    try {
      const prompt = this.buildPrompt(sequence);
      const model = this.client.getGenerativeModel({
        model: this.model,
        generationConfig: {
          temperature: 0.1,
          maxOutputTokens: 500,
        },
      });

      // Call with timeout
      const result = await Promise.race([
        model.generateContent(prompt),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error('Gemini API timeout (10s)')), 10000)
        )
      ]);

      const response = result.response;
      const text = response.text();

      // Track costs (approximate for Gemini 2.0 Flash)
      // Input: ~$0.075 per 1M tokens, Output: ~$0.30 per 1M tokens
      const inputTokens = prompt.split(/\s+/).length * 1.3; // Rough estimate
      const outputTokens = text.split(/\s+/).length * 1.3;

      const inputCost = (inputTokens / 1_000_000) * 0.075;
      const outputCost = (outputTokens / 1_000_000) * 0.30;
      this.totalCost += inputCost + outputCost;

      const signal = this.parseResponse(text, sequence);

      return signal;
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);

      // Log first error for debugging
      if (this.totalCost === 0) {
        console.error(`\n❌ Gemini API Error: ${errorMsg}\n`);
      }

      // Return a default signal on error (counts as incorrect)
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
   * Build the prompt for Gemini
   */
  private buildPrompt(sequence: CandlestickSequence): string {
    const candles = sequence.candles.slice(-5); // Last 5 candles for context

    const candleData = candles.map((c, i) => {
      return `Candle ${i + 1}: Open=${c.open}, High=${c.high}, Low=${c.low}, Close=${c.close}`;
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
   * Parse Gemini's response into a TradingSignal
   */
  private parseResponse(text: string, _sequence: CandlestickSequence): TradingSignal {
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
        `Gemini Analysis: ${reasoning}`
      );
    } catch (error) {
      console.warn(`Failed to parse Gemini response: ${error}`);
      console.warn(`Raw response: ${text}`);

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
        `Gemini Analysis (unparsed): ${text.substring(0, 100)}...`
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
   * Get the accumulated cost of API calls
   */
  getCost(): number {
    return this.totalCost;
  }
}
