/**
 * Anthropic LLM Adapter
 *
 * Centralized adapter for Anthropic Claude API.
 * Handles model selection, cost calculation, and response formatting.
 *
 * Supports:
 * - Claude Opus 4 (latest, most capable, expensive)
 * - Claude Sonnet 4.5 (balanced, cost-effective)
 *
 * Benefits:
 * - Single point of LLM integration
 * - Automatic cost tracking
 * - Model versioning
 * - Response normalization
 * - Error handling
 */

import Anthropic from '@anthropic-ai/sdk';

// ============================================================================
// Types
// ============================================================================

export type ClaudeModel = 'claude-opus-4' | 'claude-sonnet-4-5';

export interface LLMUsage {
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
}

export interface LLMResponse {
  text: string;
  usage: LLMUsage;
  model: string;
  stop_reason: string | null;
}

export interface LLMConfig {
  model: ClaudeModel;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
}

// ============================================================================
// Model Pricing (as of 2025-01-01)
// ============================================================================

interface ModelPricing {
  input_per_million: number;
  output_per_million: number;
}

const MODEL_PRICING: Record<ClaudeModel, ModelPricing> = {
  'claude-opus-4': {
    input_per_million: 15.0, // $15 per 1M input tokens
    output_per_million: 75.0, // $75 per 1M output tokens
  },
  'claude-sonnet-4-5': {
    input_per_million: 3.0, // $3 per 1M input tokens
    output_per_million: 15.0, // $15 per 1M output tokens
  },
};

// Model ID mapping
const MODEL_IDS: Record<ClaudeModel, string> = {
  'claude-opus-4': 'claude-opus-4-20250514',
  'claude-sonnet-4-5': 'claude-sonnet-4-5-20250929',
};

// ============================================================================
// Anthropic Adapter
// ============================================================================

export class AnthropicAdapter {
  private client: Anthropic;
  private totalCost: number = 0;
  private totalRequests: number = 0;

  constructor(apiKey: string) {
    this.client = new Anthropic({ apiKey });
  }

  /**
   * Invoke Claude with system prompt and query
   */
  async invoke(
    systemPrompt: string,
    query: string,
    config: LLMConfig = { model: 'claude-sonnet-4-5' }
  ): Promise<LLMResponse> {
    const modelId = MODEL_IDS[config.model];

    try {
      const message = await this.client.messages.create({
        model: modelId,
        max_tokens: config.max_tokens || 4000,
        temperature: config.temperature ?? 0.5,
        top_p: config.top_p,
        top_k: config.top_k,
        system: systemPrompt,
        messages: [
          {
            role: 'user',
            content: query,
          },
        ],
      });

      // Extract text from response
      const text =
        message.content[0].type === 'text' ? message.content[0].text : '';

      // Calculate cost
      const cost = this.calculateCost(
        {
          input_tokens: message.usage.input_tokens,
          output_tokens: message.usage.output_tokens,
        },
        config.model
      );

      // Track stats
      this.totalCost += cost;
      this.totalRequests++;

      return {
        text,
        usage: {
          input_tokens: message.usage.input_tokens,
          output_tokens: message.usage.output_tokens,
          cost_usd: cost,
        },
        model: modelId,
        stop_reason: message.stop_reason,
      };
    } catch (error) {
      if (error instanceof Anthropic.APIError) {
        throw new Error(
          `Anthropic API Error (${error.status}): ${error.message}`
        );
      }
      throw error;
    }
  }

  /**
   * Invoke with streaming (for real-time responses)
   */
  async *invokeStream(
    systemPrompt: string,
    query: string,
    config: LLMConfig = { model: 'claude-sonnet-4-5' }
  ): AsyncGenerator<string, LLMUsage, undefined> {
    const modelId = MODEL_IDS[config.model];

    const stream = await this.client.messages.stream({
      model: modelId,
      max_tokens: config.max_tokens || 4000,
      temperature: config.temperature ?? 0.5,
      system: systemPrompt,
      messages: [
        {
          role: 'user',
          content: query,
        },
      ],
    });

    let inputTokens = 0;
    let outputTokens = 0;

    for await (const chunk of stream) {
      if (chunk.type === 'content_block_delta') {
        if (chunk.delta.type === 'text_delta') {
          yield chunk.delta.text;
        }
      } else if (chunk.type === 'message_start') {
        inputTokens = chunk.message.usage.input_tokens;
      } else if (chunk.type === 'message_delta') {
        outputTokens = chunk.usage.output_tokens;
      }
    }

    const cost = this.calculateCost(
      { input_tokens: inputTokens, output_tokens: outputTokens },
      config.model
    );

    this.totalCost += cost;
    this.totalRequests++;

    return {
      input_tokens: inputTokens,
      output_tokens: outputTokens,
      cost_usd: cost,
    };
  }

  /**
   * Calculate cost based on token usage and model
   */
  private calculateCost(
    usage: { input_tokens: number; output_tokens: number },
    model: ClaudeModel
  ): number {
    const pricing = MODEL_PRICING[model];

    const inputCost = (usage.input_tokens / 1_000_000) * pricing.input_per_million;
    const outputCost = (usage.output_tokens / 1_000_000) * pricing.output_per_million;

    return inputCost + outputCost;
  }

  /**
   * Get total cost across all requests
   */
  getTotalCost(): number {
    return this.totalCost;
  }

  /**
   * Get total number of requests
   */
  getTotalRequests(): number {
    return this.totalRequests;
  }

  /**
   * Reset cost tracking
   */
  resetStats(): void {
    this.totalCost = 0;
    this.totalRequests = 0;
  }

  /**
   * Get cost estimate for a prompt (without making request)
   */
  estimateCost(
    systemPrompt: string,
    query: string,
    model: ClaudeModel = 'claude-sonnet-4-5'
  ): { estimated_cost: number; note: string } {
    // Rough estimate: ~4 chars per token
    const estimatedInputTokens = Math.ceil((systemPrompt.length + query.length) / 4);
    const estimatedOutputTokens = 1000; // Assume average response

    const cost = this.calculateCost(
      {
        input_tokens: estimatedInputTokens,
        output_tokens: estimatedOutputTokens,
      },
      model
    );

    return {
      estimated_cost: cost,
      note: 'Rough estimate based on ~4 chars/token, actual may vary',
    };
  }

  /**
   * Compare costs between models
   */
  compareCosts(inputTokens: number, outputTokens: number): Record<ClaudeModel, number> {
    const costs: Record<ClaudeModel, number> = {} as any;

    for (const model of Object.keys(MODEL_PRICING) as ClaudeModel[]) {
      costs[model] = this.calculateCost({ input_tokens: inputTokens, output_tokens: outputTokens }, model);
    }

    return costs;
  }
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Create adapter instance with API key from environment
 */
export function createAdapter(apiKey?: string): AnthropicAdapter {
  const key = apiKey || process.env.ANTHROPIC_API_KEY;

  if (!key) {
    throw new Error('ANTHROPIC_API_KEY not found in environment');
  }

  return new AnthropicAdapter(key);
}

/**
 * Get recommended model for task
 */
export function getRecommendedModel(task: 'reasoning' | 'creative' | 'fast' | 'cheap'): ClaudeModel {
  switch (task) {
    case 'reasoning':
      return 'claude-opus-4'; // Best reasoning capability
    case 'creative':
      return 'claude-opus-4'; // Best creative capability
    case 'fast':
      return 'claude-sonnet-4-5'; // Faster, cheaper
    case 'cheap':
      return 'claude-sonnet-4-5'; // Most cost-effective
    default:
      return 'claude-sonnet-4-5';
  }
}
