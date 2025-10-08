/**
 * @file anthropic-adapter.test.ts
 * Tests for AnthropicAdapter - Centralized LLM integration layer
 *
 * Key capabilities tested:
 * - API client initialization
 * - Message invocation with cost tracking
 * - Streaming responses
 * - Cost calculation (Opus 4 vs Sonnet 4.5)
 * - Usage statistics tracking
 * - Cost estimation
 * - Model cost comparison
 * - Factory functions
 * - Input validation
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  AnthropicAdapter,
  createAdapter,
  getRecommendedModel,
  ClaudeModel,
  LLMConfig,
} from '../llm/anthropic-adapter';

// Mock the Anthropic SDK
vi.mock('@anthropic-ai/sdk', () => {
  return {
    default: class MockAnthropic {
      messages = {
        create: vi.fn(),
        stream: vi.fn(),
      };

      APIError = class APIError extends Error {
        status: number;
        constructor(message: string, status: number) {
          super(message);
          this.status = status;
        }
      };
    },
  };
});

describe('AnthropicAdapter', () => {
  let adapter: AnthropicAdapter;
  let mockClient: any;

  beforeEach(() => {
    adapter = new AnthropicAdapter('test-api-key');
    // @ts-ignore - Access private client for mocking
    mockClient = adapter['client'];

    // Setup default mock response
    mockClient.messages.create.mockResolvedValue({
      content: [{ type: 'text', text: 'Mock response' }],
      usage: {
        input_tokens: 100,
        output_tokens: 50,
      },
      stop_reason: 'end_turn',
    });
  });

  describe('Constructor', () => {
    it('should create instance with API key', () => {
      expect(adapter).toBeInstanceOf(AnthropicAdapter);
    });

    it('should accept different API keys', () => {
      const adapter1 = new AnthropicAdapter('key1');
      const adapter2 = new AnthropicAdapter('key2');

      expect(adapter1).toBeInstanceOf(AnthropicAdapter);
      expect(adapter2).toBeInstanceOf(AnthropicAdapter);
    });

    it('should initialize cost tracking at zero', () => {
      expect(adapter.getTotalCost()).toBe(0);
      expect(adapter.getTotalRequests()).toBe(0);
    });
  });

  describe('invoke', () => {
    it('should invoke Claude with system prompt and query', async () => {
      const response = await adapter.invoke('System prompt', 'User query');

      expect(response.text).toBe('Mock response');
      expect(response.usage.input_tokens).toBe(100);
      expect(response.usage.output_tokens).toBe(50);
    });

    it('should use default model (Sonnet 4.5)', async () => {
      await adapter.invoke('System', 'Query');

      expect(mockClient.messages.create).toHaveBeenCalledWith(
        expect.objectContaining({
          model: 'claude-sonnet-4-5-20250929',
        })
      );
    });

    it('should support Opus 4 model', async () => {
      const config: LLMConfig = { model: 'claude-opus-4' };
      await adapter.invoke('System', 'Query', config);

      expect(mockClient.messages.create).toHaveBeenCalledWith(
        expect.objectContaining({
          model: 'claude-opus-4-20250514',
        })
      );
    });

    it('should pass system prompt', async () => {
      await adapter.invoke('My system prompt', 'Query');

      expect(mockClient.messages.create).toHaveBeenCalledWith(
        expect.objectContaining({
          system: 'My system prompt',
        })
      );
    });

    it('should pass user query as message', async () => {
      await adapter.invoke('System', 'My query');

      expect(mockClient.messages.create).toHaveBeenCalledWith(
        expect.objectContaining({
          messages: [{ role: 'user', content: 'My query' }],
        })
      );
    });

    it('should use default max_tokens of 4000', async () => {
      await adapter.invoke('System', 'Query');

      expect(mockClient.messages.create).toHaveBeenCalledWith(
        expect.objectContaining({
          max_tokens: 4000,
        })
      );
    });

    it('should support custom max_tokens', async () => {
      const config: LLMConfig = { model: 'claude-sonnet-4-5', max_tokens: 8000 };
      await adapter.invoke('System', 'Query', config);

      expect(mockClient.messages.create).toHaveBeenCalledWith(
        expect.objectContaining({
          max_tokens: 8000,
        })
      );
    });

    it('should use default temperature of 0.5', async () => {
      await adapter.invoke('System', 'Query');

      expect(mockClient.messages.create).toHaveBeenCalledWith(
        expect.objectContaining({
          temperature: 0.5,
        })
      );
    });

    it('should support custom temperature', async () => {
      const config: LLMConfig = { model: 'claude-sonnet-4-5', temperature: 0.8 };
      await adapter.invoke('System', 'Query', config);

      expect(mockClient.messages.create).toHaveBeenCalledWith(
        expect.objectContaining({
          temperature: 0.8,
        })
      );
    });

    it('should support top_p parameter', async () => {
      const config: LLMConfig = { model: 'claude-sonnet-4-5', top_p: 0.9 };
      await adapter.invoke('System', 'Query', config);

      expect(mockClient.messages.create).toHaveBeenCalledWith(
        expect.objectContaining({
          top_p: 0.9,
        })
      );
    });

    it('should support top_k parameter', async () => {
      const config: LLMConfig = { model: 'claude-sonnet-4-5', top_k: 40 };
      await adapter.invoke('System', 'Query', config);

      expect(mockClient.messages.create).toHaveBeenCalledWith(
        expect.objectContaining({
          top_k: 40,
        })
      );
    });

    it('should calculate cost for response', async () => {
      const response = await adapter.invoke('System', 'Query');

      expect(response.usage.cost_usd).toBeGreaterThan(0);
    });

    it('should track total cost', async () => {
      const costBefore = adapter.getTotalCost();
      await adapter.invoke('System', 'Query');
      const costAfter = adapter.getTotalCost();

      expect(costAfter).toBeGreaterThan(costBefore);
    });

    it('should track total requests', async () => {
      const requestsBefore = adapter.getTotalRequests();
      await adapter.invoke('System', 'Query');
      const requestsAfter = adapter.getTotalRequests();

      expect(requestsAfter).toBe(requestsBefore + 1);
    });

    it('should return model ID in response', async () => {
      const response = await adapter.invoke('System', 'Query');

      expect(response.model).toBe('claude-sonnet-4-5-20250929');
    });

    it('should return stop reason in response', async () => {
      const response = await adapter.invoke('System', 'Query');

      expect(response.stop_reason).toBe('end_turn');
    });

    it('should throw error for empty system prompt', async () => {
      await expect(adapter.invoke('', 'Query')).rejects.toThrow('System prompt cannot be empty');
    });

    it('should throw error for empty query', async () => {
      await expect(adapter.invoke('System', '')).rejects.toThrow('Query cannot be empty');
    });

    it('should throw error for whitespace-only system prompt', async () => {
      await expect(adapter.invoke('   ', 'Query')).rejects.toThrow('System prompt cannot be empty');
    });

    it('should throw error for whitespace-only query', async () => {
      await expect(adapter.invoke('System', '   ')).rejects.toThrow('Query cannot be empty');
    });

    it('should throw error for excessively long prompt', async () => {
      const longPrompt = 'a'.repeat(400001);

      await expect(adapter.invoke('System', longPrompt)).rejects.toThrow('Prompt too long');
    });
  });

  describe('Cost Calculation', () => {
    it('should calculate correct cost for Sonnet 4.5', async () => {
      // Sonnet: $3/1M input, $15/1M output
      // 100 input + 50 output = $0.00105
      const response = await adapter.invoke('System', 'Query', { model: 'claude-sonnet-4-5' });

      const expectedCost = (100 / 1_000_000) * 3 + (50 / 1_000_000) * 15;
      expect(response.usage.cost_usd).toBeCloseTo(expectedCost, 6);
    });

    it('should calculate correct cost for Opus 4', async () => {
      // Opus: $15/1M input, $75/1M output
      // 100 input + 50 output = $0.00525
      const response = await adapter.invoke('System', 'Query', { model: 'claude-opus-4' });

      const expectedCost = (100 / 1_000_000) * 15 + (50 / 1_000_000) * 75;
      expect(response.usage.cost_usd).toBeCloseTo(expectedCost, 6);
    });

    it('should accumulate costs across multiple requests', async () => {
      await adapter.invoke('System', 'Query 1');
      await adapter.invoke('System', 'Query 2');

      const totalCost = adapter.getTotalCost();
      const expectedCost = 2 * ((100 / 1_000_000) * 3 + (50 / 1_000_000) * 15);
      expect(totalCost).toBeCloseTo(expectedCost, 6);
    });
  });

  describe('Usage Statistics', () => {
    it('should track request count', async () => {
      await adapter.invoke('System', 'Query 1');
      await adapter.invoke('System', 'Query 2');
      await adapter.invoke('System', 'Query 3');

      expect(adapter.getTotalRequests()).toBe(3);
    });

    it('should track cumulative cost', async () => {
      await adapter.invoke('System', 'Query 1');
      await adapter.invoke('System', 'Query 2');

      expect(adapter.getTotalCost()).toBeGreaterThan(0);
    });

    it('should reset stats', () => {
      adapter['totalCost'] = 10;
      adapter['totalRequests'] = 5;

      adapter.resetStats();

      expect(adapter.getTotalCost()).toBe(0);
      expect(adapter.getTotalRequests()).toBe(0);
    });
  });

  describe('estimateCost', () => {
    it('should estimate cost without making request', () => {
      const estimate = adapter.estimateCost('System prompt', 'User query');

      expect(estimate.estimated_cost).toBeGreaterThan(0);
      expect(estimate.note).toContain('Rough estimate');
    });

    it('should use default output tokens of 1000', () => {
      const estimate = adapter.estimateCost('System', 'Query');

      // Should calculate based on 1000 output tokens
      expect(estimate.estimated_cost).toBeGreaterThan(0);
    });

    it('should support custom output token estimate', () => {
      const estimate1 = adapter.estimateCost('System', 'Query', 'claude-sonnet-4-5', 500);
      const estimate2 = adapter.estimateCost('System', 'Query', 'claude-sonnet-4-5', 2000);

      // More output tokens = higher cost
      expect(estimate2.estimated_cost).toBeGreaterThan(estimate1.estimated_cost);
    });

    it('should estimate based on ~4 chars/token', () => {
      const prompt = 'a'.repeat(400); // ~100 tokens
      const query = 'b'.repeat(400); // ~100 tokens
      const estimate = adapter.estimateCost(prompt, query, 'claude-sonnet-4-5', 1000);

      // ~200 input tokens at $3/1M = $0.0006
      // 1000 output tokens at $15/1M = $0.015
      // Total ~$0.0156
      expect(estimate.estimated_cost).toBeCloseTo(0.0156, 3);
    });

    it('should support different models', () => {
      const sonnetCost = adapter.estimateCost('System', 'Query', 'claude-sonnet-4-5');
      const opusCost = adapter.estimateCost('System', 'Query', 'claude-opus-4');

      // Opus is more expensive
      expect(opusCost.estimated_cost).toBeGreaterThan(sonnetCost.estimated_cost);
    });

    it('should not increment request count', () => {
      const requestsBefore = adapter.getTotalRequests();
      adapter.estimateCost('System', 'Query');
      const requestsAfter = adapter.getTotalRequests();

      expect(requestsAfter).toBe(requestsBefore);
    });

    it('should not increment total cost', () => {
      const costBefore = adapter.getTotalCost();
      adapter.estimateCost('System', 'Query');
      const costAfter = adapter.getTotalCost();

      expect(costAfter).toBe(costBefore);
    });
  });

  describe('compareCosts', () => {
    it('should compare costs between models', () => {
      const costs = adapter.compareCosts(1000, 500);

      expect(costs['claude-sonnet-4-5']).toBeDefined();
      expect(costs['claude-opus-4']).toBeDefined();
    });

    it('should show Opus costs more than Sonnet', () => {
      const costs = adapter.compareCosts(1000, 500);

      expect(costs['claude-opus-4']).toBeGreaterThan(costs['claude-sonnet-4-5']);
    });

    it('should calculate correct cost ratios', () => {
      const costs = adapter.compareCosts(1000, 500);

      const sonnetCost = costs['claude-sonnet-4-5'];
      const opusCost = costs['claude-opus-4'];

      // Opus input is 5x more expensive ($15 vs $3)
      // Opus output is 5x more expensive ($75 vs $15)
      // Overall should be 5x
      expect(opusCost / sonnetCost).toBeCloseTo(5, 1);
    });

    it('should handle zero tokens', () => {
      const costs = adapter.compareCosts(0, 0);

      expect(costs['claude-sonnet-4-5']).toBe(0);
      expect(costs['claude-opus-4']).toBe(0);
    });

    it('should handle large token counts', () => {
      const costs = adapter.compareCosts(1_000_000, 500_000);

      expect(costs['claude-sonnet-4-5']).toBeGreaterThan(0);
      expect(costs['claude-opus-4']).toBeGreaterThan(0);
    });
  });

  describe('Factory Function', () => {
    it('should create adapter with provided API key', () => {
      const adapter = createAdapter('test-key');

      expect(adapter).toBeInstanceOf(AnthropicAdapter);
    });

    it('should throw error if no API key provided', () => {
      delete process.env.ANTHROPIC_API_KEY;

      expect(() => createAdapter()).toThrow('ANTHROPIC_API_KEY not found');
    });

    it('should use environment variable if no key provided', () => {
      process.env.ANTHROPIC_API_KEY = 'env-key';

      const adapter = createAdapter();

      expect(adapter).toBeInstanceOf(AnthropicAdapter);
    });

    it('should prefer provided key over environment', () => {
      process.env.ANTHROPIC_API_KEY = 'env-key';

      const adapter = createAdapter('provided-key');

      expect(adapter).toBeInstanceOf(AnthropicAdapter);
    });
  });

  describe('getRecommendedModel', () => {
    it('should recommend Opus for reasoning tasks', () => {
      expect(getRecommendedModel('reasoning')).toBe('claude-opus-4');
    });

    it('should recommend Opus for creative tasks', () => {
      expect(getRecommendedModel('creative')).toBe('claude-opus-4');
    });

    it('should recommend Sonnet for fast tasks', () => {
      expect(getRecommendedModel('fast')).toBe('claude-sonnet-4-5');
    });

    it('should recommend Sonnet for cheap tasks', () => {
      expect(getRecommendedModel('cheap')).toBe('claude-sonnet-4-5');
    });

    it('should default to Sonnet for unknown tasks', () => {
      // @ts-ignore - testing unknown task
      expect(getRecommendedModel('unknown')).toBe('claude-sonnet-4-5');
    });
  });

  describe('Model Configuration', () => {
    it('should use correct model ID for Sonnet', async () => {
      await adapter.invoke('System', 'Query', { model: 'claude-sonnet-4-5' });

      expect(mockClient.messages.create).toHaveBeenCalledWith(
        expect.objectContaining({
          model: 'claude-sonnet-4-5-20250929',
        })
      );
    });

    it('should use correct model ID for Opus', async () => {
      await adapter.invoke('System', 'Query', { model: 'claude-opus-4' });

      expect(mockClient.messages.create).toHaveBeenCalledWith(
        expect.objectContaining({
          model: 'claude-opus-4-20250514',
        })
      );
    });
  });

  describe('Integration', () => {
    it('should handle complete workflow', async () => {
      // Estimate cost
      const estimate = adapter.estimateCost('System', 'Query');
      expect(estimate.estimated_cost).toBeGreaterThan(0);

      // Make request
      const response = await adapter.invoke('System', 'Query');
      expect(response.text).toBeDefined();

      // Check stats
      expect(adapter.getTotalRequests()).toBe(1);
      expect(adapter.getTotalCost()).toBeGreaterThan(0);

      // Compare costs
      const costs = adapter.compareCosts(response.usage.input_tokens, response.usage.output_tokens);
      expect(costs['claude-sonnet-4-5']).toBeGreaterThan(0);

      // Reset
      adapter.resetStats();
      expect(adapter.getTotalCost()).toBe(0);
    });

    it('should maintain separate cost tracking per instance', async () => {
      const adapter1 = new AnthropicAdapter('key1');
      const adapter2 = new AnthropicAdapter('key2');

      // @ts-ignore
      adapter1['client'] = mockClient;
      // @ts-ignore
      adapter2['client'] = mockClient;

      await adapter1.invoke('System', 'Query');
      await adapter2.invoke('System', 'Query');

      expect(adapter1.getTotalRequests()).toBe(1);
      expect(adapter2.getTotalRequests()).toBe(1);
    });
  });
});
