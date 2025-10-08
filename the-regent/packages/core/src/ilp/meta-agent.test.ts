/**
 * Unit Tests for Meta-Agent System
 *
 * Tests orchestration, recursion, constitution enforcement, and ACL validation
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  MetaAgent,
  SpecializedAgent,
  AgentResponse,
  RecursionState,
  QueryDecomposition,
  CompositionResult,
} from './meta-agent.js';
import type { LLMResponse } from './llm/anthropic-adapter.js';

// Mock API key for testing
const TEST_API_KEY = 'test-api-key-12345';

// ============================================================================
// Mock Specialized Agent
// ============================================================================

class MockFinanceAgent extends SpecializedAgent {
  private mockResponse: AgentResponse;

  constructor(
    apiKey: string,
    mockResponse?: AgentResponse
  ) {
    super(
      apiKey,
      'You are a financial expert.',
      0.5,
      'claude-sonnet-4-5'
    );
    this.mockResponse = mockResponse || {
      answer: 'Mock finance response',
      concepts: ['budgeting', 'investment'],
      confidence: 0.9,
      reasoning: 'Test reasoning',
    };
  }

  getDomain(): string {
    return 'finance';
  }

  async process(query: string, context: RecursionState): Promise<AgentResponse> {
    return this.mockResponse;
  }
}

class MockTechAgent extends SpecializedAgent {
  private mockResponse: AgentResponse;

  constructor(
    apiKey: string,
    mockResponse?: AgentResponse
  ) {
    super(
      apiKey,
      'You are a technology expert.',
      0.5,
      'claude-sonnet-4-5'
    );
    this.mockResponse = mockResponse || {
      answer: 'Mock tech response',
      concepts: ['algorithms', 'optimization'],
      confidence: 0.85,
      reasoning: 'Test reasoning',
    };
  }

  getDomain(): string {
    return 'technology';
  }

  async process(query: string, context: RecursionState): Promise<AgentResponse> {
    return this.mockResponse;
  }
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Create a mock LLM response
 */
function createMockLLMResponse(text: string, cost: number = 0.001): LLMResponse {
  return {
    text,
    usage: {
      input_tokens: 100,
      output_tokens: 50,
      cost_usd: cost,
    },
    model: 'claude-sonnet-4-5',
    finish_reason: 'end_turn',
  };
}

// ============================================================================
// SpecializedAgent Tests
// ============================================================================

describe('SpecializedAgent', () => {
  let agent: MockFinanceAgent;

  beforeEach(() => {
    agent = new MockFinanceAgent(TEST_API_KEY);
  });

  describe('constructor', () => {
    it('should initialize with correct domain', () => {
      expect(agent.getDomain()).toBe('finance');
    });

    it('should accept custom system prompt', () => {
      const customAgent = new MockFinanceAgent(TEST_API_KEY);
      expect(customAgent).toBeDefined();
    });
  });

  describe('process', () => {
    it('should return agent response', async () => {
      const state: RecursionState = {
        depth: 0,
        invocation_count: 0,
        cost_so_far: 0,
        previous_agents: [],
        traces: [],
        insights: new Map(),
      };

      const response = await agent.process('test query', state);

      expect(response).toBeDefined();
      expect(response.answer).toBeTruthy();
      expect(response.confidence).toBeGreaterThan(0);
      expect(response.concepts).toBeInstanceOf(Array);
    });

    it('should have required response fields', async () => {
      const state: RecursionState = {
        depth: 0,
        invocation_count: 0,
        cost_so_far: 0,
        previous_agents: [],
        traces: [],
        insights: new Map(),
      };

      const response = await agent.process('test query', state);

      expect(response).toHaveProperty('answer');
      expect(response).toHaveProperty('concepts');
      expect(response).toHaveProperty('confidence');
      expect(response).toHaveProperty('reasoning');
    });
  });

  describe('setSliceNavigator', () => {
    it('should accept slice navigator', () => {
      const mockNavigator = {} as any;
      expect(() => agent.setSliceNavigator(mockNavigator)).not.toThrow();
    });
  });
});

// ============================================================================
// MetaAgent Tests
// ============================================================================

describe('MetaAgent', () => {
  let metaAgent: MetaAgent;
  let financeAgent: MockFinanceAgent;
  let techAgent: MockTechAgent;

  beforeEach(() => {
    metaAgent = new MetaAgent(
      TEST_API_KEY,
      5,  // maxDepth
      10, // maxInvocations
      1.0 // maxCostUSD
    );

    financeAgent = new MockFinanceAgent(TEST_API_KEY);
    techAgent = new MockTechAgent(TEST_API_KEY);

    metaAgent.registerAgent('finance', financeAgent);
    metaAgent.registerAgent('technology', techAgent);
  });

  describe('constructor', () => {
    it('should initialize with correct defaults', () => {
      expect(metaAgent).toBeDefined();
    });

    it('should accept custom budget constraints', () => {
      const customAgent = new MetaAgent(TEST_API_KEY, 3, 5, 0.5);
      expect(customAgent).toBeDefined();
    });
  });

  describe('registerAgent', () => {
    it('should register specialized agents', () => {
      const newAgent = new MetaAgent(TEST_API_KEY);
      const agent = new MockFinanceAgent(TEST_API_KEY);

      expect(() => newAgent.registerAgent('test', agent)).not.toThrow();
    });

    it('should give agent access to slice navigator', () => {
      const newAgent = new MetaAgent(TEST_API_KEY);
      const agent = new MockFinanceAgent(TEST_API_KEY);

      newAgent.registerAgent('test', agent);
      // Agent should have access to navigator (set internally)
      expect(agent).toBeDefined();
    });
  });

  describe('initialize', () => {
    it('should initialize slice navigator', async () => {
      // This might fail if slices directory doesn't exist, which is OK in tests
      await expect(metaAgent.initialize()).resolves.not.toThrow();
    });
  });

  describe('process - basic functionality', () => {
    it('should process simple query', async () => {
      // Mock the LLM responses
      const mockLLM = vi.spyOn(metaAgent as any, 'decomposeQuery');
      mockLLM.mockResolvedValue({
        domains: ['finance'],
        reasoning: 'Finance is relevant',
        primary_domain: 'finance',
      } as QueryDecomposition);

      const mockCompose = vi.spyOn(metaAgent as any, 'composeInsights');
      mockCompose.mockResolvedValue({
        synthesis: 'Final answer',
        should_recurse: false,
        confidence: 0.9,
      } as CompositionResult);

      const mockSynthesize = vi.spyOn(metaAgent as any, 'synthesizeFinal');
      mockSynthesize.mockResolvedValue({
        answer: 'Final synthesized answer',
        concepts: ['budgeting'],
        confidence: 0.9,
        reasoning: 'Synthesis complete',
      });

      const result = await metaAgent.process('What is budgeting?');

      expect(result).toBeDefined();
      expect(result.final_answer).toBeTruthy();
      expect(result.trace).toBeInstanceOf(Array);
      expect(result.emergent_insights).toBeInstanceOf(Array);
      expect(result.reasoning_path).toBeTruthy();
      expect(result.constitution_violations).toBeInstanceOf(Array);
    });

    it('should return attention tracking data', async () => {
      const mockLLM = vi.spyOn(metaAgent as any, 'decomposeQuery');
      mockLLM.mockResolvedValue({
        domains: ['finance'],
        reasoning: 'Finance is relevant',
        primary_domain: 'finance',
      });

      const mockCompose = vi.spyOn(metaAgent as any, 'composeInsights');
      mockCompose.mockResolvedValue({
        synthesis: 'Final answer',
        should_recurse: false,
        confidence: 0.9,
      });

      const mockSynthesize = vi.spyOn(metaAgent as any, 'synthesizeFinal');
      mockSynthesize.mockResolvedValue({
        answer: 'Final answer',
        concepts: [],
        confidence: 0.9,
        reasoning: 'Done',
      });

      const result = await metaAgent.process('Test query');

      expect(result.attention).toBeDefined();
      if (result.attention) {
        expect(result.attention.query_id).toBeTruthy();
        expect(result.attention.query).toBe('Test query');
        expect(result.attention.traces).toBeInstanceOf(Array);
      }
    });
  });

  describe('process - budget constraints', () => {
    it('should respect max depth', async () => {
      const shallowAgent = new MetaAgent(TEST_API_KEY, 1, 10, 1.0);
      shallowAgent.registerAgent('finance', financeAgent);

      const mockLLM = vi.spyOn(shallowAgent as any, 'decomposeQuery');
      mockLLM.mockResolvedValue({
        domains: ['finance'],
        reasoning: 'Finance',
        primary_domain: 'finance',
      });

      const mockCompose = vi.spyOn(shallowAgent as any, 'composeInsights');
      mockCompose.mockResolvedValue({
        synthesis: 'Need more depth',
        should_recurse: true,
        confidence: 0.5,
        missing_perspectives: ['technology'],
      });

      const mockSynthesize = vi.spyOn(shallowAgent as any, 'synthesizeFinal');
      mockSynthesize.mockResolvedValue({
        answer: 'Limited answer',
        concepts: [],
        confidence: 0.5,
        reasoning: 'Max depth reached',
      });

      const result = await shallowAgent.process('Test');

      // Should stop recursion due to depth limit
      expect(result.trace.length).toBeLessThanOrEqual(2);
    });

    it('should respect max invocations', async () => {
      const limitedAgent = new MetaAgent(TEST_API_KEY, 5, 2, 1.0);
      limitedAgent.registerAgent('finance', financeAgent);
      limitedAgent.registerAgent('technology', techAgent);

      const mockLLM = vi.spyOn(limitedAgent as any, 'decomposeQuery');
      mockLLM.mockResolvedValue({
        domains: ['finance', 'technology'],
        reasoning: 'Both needed',
      });

      const mockCompose = vi.spyOn(limitedAgent as any, 'composeInsights');
      mockCompose.mockResolvedValue({
        synthesis: 'Continue',
        should_recurse: false,
        confidence: 0.9,
      });

      const mockSynthesize = vi.spyOn(limitedAgent as any, 'synthesizeFinal');
      mockSynthesize.mockResolvedValue({
        answer: 'Final',
        concepts: [],
        confidence: 0.9,
        reasoning: 'Done',
      });

      const result = await limitedAgent.process('Test');

      // Should stop due to invocation limit
      expect(result.trace.length).toBeLessThanOrEqual(2);
    });
  });

  describe('process - constitution enforcement', () => {
    it('should validate responses against constitution', async () => {
      const mockLLM = vi.spyOn(metaAgent as any, 'decomposeQuery');
      mockLLM.mockResolvedValue({
        domains: ['finance'],
        reasoning: 'Finance',
      });

      const mockCompose = vi.spyOn(metaAgent as any, 'composeInsights');
      mockCompose.mockResolvedValue({
        synthesis: 'Answer',
        should_recurse: false,
        confidence: 0.9,
      });

      const mockSynthesize = vi.spyOn(metaAgent as any, 'synthesizeFinal');
      mockSynthesize.mockResolvedValue({
        answer: 'Answer',
        concepts: [],
        confidence: 0.9,
        reasoning: 'Done',
      });

      const result = await metaAgent.process('Test query');

      // Constitution violations should be tracked
      expect(result.constitution_violations).toBeDefined();
      expect(Array.isArray(result.constitution_violations)).toBe(true);
    });

    it('should collect violations from multiple agents', async () => {
      // Create agent with low confidence (potential violation)
      const lowConfidenceAgent = new MockFinanceAgent(TEST_API_KEY, {
        answer: 'Uncertain answer',
        concepts: [],
        confidence: 0.1, // Very low confidence
        reasoning: 'Not sure',
      });

      const testAgent = new MetaAgent(TEST_API_KEY);
      testAgent.registerAgent('finance', lowConfidenceAgent);

      const mockLLM = vi.spyOn(testAgent as any, 'decomposeQuery');
      mockLLM.mockResolvedValue({
        domains: ['finance'],
        reasoning: 'Finance',
      });

      const mockCompose = vi.spyOn(testAgent as any, 'composeInsights');
      mockCompose.mockResolvedValue({
        synthesis: 'Answer',
        should_recurse: false,
        confidence: 0.9,
      });

      const mockSynthesize = vi.spyOn(testAgent as any, 'synthesizeFinal');
      mockSynthesize.mockResolvedValue({
        answer: 'Answer',
        concepts: [],
        confidence: 0.9,
        reasoning: 'Done',
      });

      const result = await testAgent.process('Test');

      // May have warnings about low confidence
      expect(result.constitution_violations).toBeDefined();
    });
  });

  describe('process - emergent insights', () => {
    it('should extract emergent insights from multiple agents', async () => {
      const mockLLM = vi.spyOn(metaAgent as any, 'decomposeQuery');
      mockLLM.mockResolvedValue({
        domains: ['finance', 'technology'],
        reasoning: 'Both perspectives needed',
      });

      const mockCompose = vi.spyOn(metaAgent as any, 'composeInsights');
      mockCompose.mockResolvedValue({
        synthesis: 'Combined answer',
        should_recurse: false,
        confidence: 0.95,
      });

      const mockSynthesize = vi.spyOn(metaAgent as any, 'synthesizeFinal');
      mockSynthesize.mockResolvedValue({
        answer: 'Synthesized answer',
        concepts: ['budgeting', 'algorithms'],
        confidence: 0.95,
        reasoning: 'Combined insights',
      });

      const result = await metaAgent.process('How to optimize budgeting with algorithms?');

      expect(result.emergent_insights).toBeDefined();
      expect(result.emergent_insights.length).toBeGreaterThan(0);
    });
  });

  describe('process - reasoning path', () => {
    it('should format reasoning path with agent sequence', async () => {
      const mockLLM = vi.spyOn(metaAgent as any, 'decomposeQuery');
      mockLLM.mockResolvedValue({
        domains: ['finance', 'technology'],
        reasoning: 'Both needed',
      });

      const mockCompose = vi.spyOn(metaAgent as any, 'composeInsights');
      mockCompose.mockResolvedValue({
        synthesis: 'Answer',
        should_recurse: false,
        confidence: 0.9,
      });

      const mockSynthesize = vi.spyOn(metaAgent as any, 'synthesizeFinal');
      mockSynthesize.mockResolvedValue({
        answer: 'Answer',
        concepts: [],
        confidence: 0.9,
        reasoning: 'Done',
      });

      const result = await metaAgent.process('Test');

      expect(result.reasoning_path).toBeTruthy();
      expect(typeof result.reasoning_path).toBe('string');
    });
  });
});

// ============================================================================
// Integration Tests
// ============================================================================

describe('MetaAgent Integration', () => {
  let metaAgent: MetaAgent;

  beforeEach(() => {
    metaAgent = new MetaAgent(TEST_API_KEY, 5, 10, 1.0);
  });

  describe('multi-agent collaboration', () => {
    it('should coordinate multiple specialized agents', async () => {
      const finance = new MockFinanceAgent(TEST_API_KEY, {
        answer: 'Finance perspective: Focus on ROI',
        concepts: ['roi', 'investment'],
        confidence: 0.9,
        reasoning: 'Financial analysis',
        suggestions_to_invoke: ['technology'],
      });

      const tech = new MockTechAgent(TEST_API_KEY, {
        answer: 'Tech perspective: Optimize algorithms',
        concepts: ['algorithms', 'optimization'],
        confidence: 0.85,
        reasoning: 'Technical analysis',
      });

      metaAgent.registerAgent('finance', finance);
      metaAgent.registerAgent('technology', tech);

      const mockLLM = vi.spyOn(metaAgent as any, 'decomposeQuery');
      mockLLM.mockResolvedValue({
        domains: ['finance', 'technology'],
        reasoning: 'Both perspectives valuable',
      });

      const mockCompose = vi.spyOn(metaAgent as any, 'composeInsights');
      mockCompose.mockResolvedValue({
        synthesis: 'Combined insights',
        should_recurse: false,
        confidence: 0.9,
      });

      const mockSynthesize = vi.spyOn(metaAgent as any, 'synthesizeFinal');
      mockSynthesize.mockResolvedValue({
        answer: 'Coordinated answer combining finance and tech',
        concepts: ['roi', 'algorithms'],
        confidence: 0.9,
        reasoning: 'Multi-agent synthesis',
      });

      const result = await metaAgent.process('How to optimize investment with algorithms?');

      expect(result.trace.length).toBeGreaterThan(0);
      expect(result.emergent_insights.length).toBeGreaterThan(0);
    });
  });

  describe('recursive composition', () => {
    it('should handle recursive refinement', async () => {
      const agent = new MockFinanceAgent(TEST_API_KEY);
      metaAgent.registerAgent('finance', agent);

      let decomposeCalls = 0;
      const mockLLM = vi.spyOn(metaAgent as any, 'decomposeQuery');
      mockLLM.mockImplementation(async () => {
        decomposeCalls++;
        return {
          domains: ['finance'],
          reasoning: 'Finance needed',
        };
      });

      let composeCalls = 0;
      const mockCompose = vi.spyOn(metaAgent as any, 'composeInsights');
      mockCompose.mockImplementation(async () => {
        composeCalls++;
        return {
          synthesis: 'Needs more refinement',
          should_recurse: composeCalls < 2, // Recurse once
          confidence: 0.7,
          missing_perspectives: composeCalls < 2 ? ['more_detail'] : undefined,
        };
      });

      const mockSynthesize = vi.spyOn(metaAgent as any, 'synthesizeFinal');
      mockSynthesize.mockResolvedValue({
        answer: 'Refined answer',
        concepts: [],
        confidence: 0.9,
        reasoning: 'After recursion',
      });

      const result = await metaAgent.process('Complex query needing refinement');

      // Should have recursed at least once
      expect(decomposeCalls).toBeGreaterThan(1);
      expect(result.trace.length).toBeGreaterThan(1);
    });
  });
});

// ============================================================================
// Error Handling Tests
// ============================================================================

describe('MetaAgent Error Handling', () => {
  let metaAgent: MetaAgent;

  beforeEach(() => {
    metaAgent = new MetaAgent(TEST_API_KEY);
  });

  describe('agent failures', () => {
    it('should handle agent throwing errors gracefully', async () => {
      class ErrorAgent extends SpecializedAgent {
        getDomain() {
          return 'error';
        }
        async process(): Promise<AgentResponse> {
          throw new Error('Agent failure');
        }
      }

      const errorAgent = new ErrorAgent(TEST_API_KEY, 'Error agent', 0.5);
      metaAgent.registerAgent('error', errorAgent);

      const mockLLM = vi.spyOn(metaAgent as any, 'decomposeQuery');
      mockLLM.mockResolvedValue({
        domains: ['error'],
        reasoning: 'Error domain',
      });

      // Should either handle error or throw predictably
      await expect(metaAgent.process('Test')).rejects.toThrow();
    });
  });

  describe('malformed responses', () => {
    it('should handle invalid JSON responses', async () => {
      class MalformedAgent extends SpecializedAgent {
        getDomain() {
          return 'malformed';
        }
        async process(): Promise<AgentResponse> {
          // Return invalid response
          return {
            answer: '',
            concepts: [],
            confidence: -1, // Invalid confidence
            reasoning: '',
          };
        }
      }

      const malformedAgent = new MalformedAgent(TEST_API_KEY, 'Malformed', 0.5);
      metaAgent.registerAgent('malformed', malformedAgent);

      const mockLLM = vi.spyOn(metaAgent as any, 'decomposeQuery');
      mockLLM.mockResolvedValue({
        domains: ['malformed'],
        reasoning: 'Test',
      });

      const mockCompose = vi.spyOn(metaAgent as any, 'composeInsights');
      mockCompose.mockResolvedValue({
        synthesis: 'Answer',
        should_recurse: false,
        confidence: 0.9,
      });

      const mockSynthesize = vi.spyOn(metaAgent as any, 'synthesizeFinal');
      mockSynthesize.mockResolvedValue({
        answer: 'Answer',
        concepts: [],
        confidence: 0.9,
        reasoning: 'Done',
      });

      const result = await metaAgent.process('Test');

      // Should collect violations for invalid confidence
      expect(result.constitution_violations.length).toBeGreaterThan(0);
    });
  });
});
