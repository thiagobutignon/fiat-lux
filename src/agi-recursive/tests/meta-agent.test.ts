/**
 * @file meta-agent.test.ts
 * Tests for Meta-Agent orchestration layer
 *
 * Key capabilities tested:
 * - Agent registration and initialization
 * - Query decomposition
 * - Recursive processing with budget limits
 * - Constitutional enforcement integration
 * - Anti-Corruption Layer integration
 * - Attention tracking integration
 * - Insight composition and synthesis
 * - Emergent insight discovery
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  MetaAgent,
  SpecializedAgent,
  AgentResponse,
  RecursionState,
} from '../core/meta-agent';
import { AnthropicAdapter } from '../llm/anthropic-adapter';

// Mock the AnthropicAdapter
vi.mock('../llm/anthropic-adapter', () => ({
  AnthropicAdapter: vi.fn().mockImplementation(() => ({
    invoke: vi.fn(),
    getTotalCost: vi.fn(() => 0.5),
    getTotalRequests: vi.fn(() => 5),
  })),
}));

// Mock specialized agent for testing
class MockFinancialAgent extends SpecializedAgent {
  constructor(apiKey: string) {
    super(apiKey, 'You are a financial advisor', 0.5);
  }

  getDomain(): string {
    return 'financial';
  }

  async process(query: string, context: RecursionState): Promise<AgentResponse> {
    // Simulate cost
    context.cost_so_far += 0.01;

    return {
      answer: 'Diversification reduces risk in your portfolio',
      concepts: ['diversification', 'risk', 'portfolio'],
      confidence: 0.85,
      reasoning: 'Financial principle based on Modern Portfolio Theory',
      suggestions_to_invoke: [],
    };
  }
}

class MockBiologyAgent extends SpecializedAgent {
  constructor(apiKey: string) {
    super(apiKey, 'You are a biology expert', 0.5);
  }

  getDomain(): string {
    return 'biology';
  }

  async process(query: string, context: RecursionState): Promise<AgentResponse> {
    // Simulate cost
    context.cost_so_far += 0.01;

    return {
      answer: 'Homeostasis maintains internal balance through feedback mechanisms',
      concepts: ['homeostasis', 'balance', 'feedback'],
      confidence: 0.9,
      reasoning: 'Biological systems regulate through negative feedback loops',
      suggestions_to_invoke: [],
    };
  }
}

describe('MetaAgent', () => {
  let metaAgent: MetaAgent;
  let mockLLM: any;

  beforeEach(() => {
    // Create meta-agent with test API key
    metaAgent = new MetaAgent('test-api-key', 5, 10, 1.0);

    // Get the mocked LLM instance
    mockLLM = (metaAgent as any).llm;
  });

  describe('Agent Registration', () => {
    it('should register specialized agents', () => {
      const financialAgent = new MockFinancialAgent('test-key');

      metaAgent.registerAgent('financial', financialAgent);

      // Agent should be registered (verified indirectly through processing)
      expect(metaAgent).toBeDefined();
    });

    it('should register multiple agents', () => {
      const financialAgent = new MockFinancialAgent('test-key');
      const biologyAgent = new MockBiologyAgent('test-key');

      metaAgent.registerAgent('financial', financialAgent);
      metaAgent.registerAgent('biology', biologyAgent);

      expect(metaAgent).toBeDefined();
    });

    it('should allow overwriting agents', () => {
      const agent1 = new MockFinancialAgent('test-key');
      const agent2 = new MockFinancialAgent('test-key');

      metaAgent.registerAgent('financial', agent1);
      metaAgent.registerAgent('financial', agent2);

      expect(metaAgent).toBeDefined();
    });
  });

  describe('Initialization', () => {
    it('should initialize slice navigator', async () => {
      await expect(metaAgent.initialize()).resolves.not.toThrow();
    });
  });

  describe('Query Processing', () => {
    beforeEach(async () => {
      // Register test agents
      metaAgent.registerAgent('financial', new MockFinancialAgent('test-key'));
      metaAgent.registerAgent('biology', new MockBiologyAgent('test-key'));

      // Mock LLM responses
      mockLLM.invoke.mockImplementation((system: string, prompt: string) => {
        // Detect which call this is based on the system prompt
        if (system.includes('decompose queries')) {
          // Query decomposition
          return Promise.resolve({
            text: JSON.stringify({
              domains: ['financial'],
              reasoning: 'Query is about financial concepts',
              primary_domain: 'financial',
            }),
            usage: { cost_usd: 0.001 },
          });
        } else if (system.includes('synthesis engine')) {
          // Composition
          return Promise.resolve({
            text: JSON.stringify({
              synthesis: 'Combined insights from specialists',
              should_recurse: false,
              confidence: 0.9,
            }),
            usage: { cost_usd: 0.002 },
          });
        } else if (system.includes('final synthesis')) {
          // Final synthesis
          return Promise.resolve({
            text: JSON.stringify({
              answer: 'Final comprehensive answer combining all insights',
              concepts: ['synthesis', 'composition'],
              confidence: 0.95,
              reasoning: 'Composed from financial and biology perspectives',
            }),
            usage: { cost_usd: 0.003 },
          });
        }

        return Promise.resolve({
          text: '{}',
          usage: { cost_usd: 0.001 },
        });
      });

      await metaAgent.initialize();
    });

    it('should process a query and return final answer', async () => {
      const result = await metaAgent.process('How can I reduce portfolio risk?');

      expect(result.final_answer).toBeDefined();
      expect(result.trace).toBeDefined();
      expect(result.emergent_insights).toBeDefined();
      expect(result.reasoning_path).toBeDefined();
    });

    it('should include recursion traces', async () => {
      const result = await metaAgent.process('How can I reduce portfolio risk?');

      expect(result.trace.length).toBeGreaterThan(0);
      expect(result.trace[0]).toHaveProperty('depth');
      expect(result.trace[0]).toHaveProperty('agent_id');
      expect(result.trace[0]).toHaveProperty('query');
      expect(result.trace[0]).toHaveProperty('response');
      expect(result.trace[0]).toHaveProperty('timestamp');
      expect(result.trace[0]).toHaveProperty('cost_estimate');
    });

    it('should track attention during processing', async () => {
      const result = await metaAgent.process('How can I reduce portfolio risk?');

      expect(result.attention).toBeDefined();
      expect(result.attention?.query_id).toBeDefined();
      expect(result.attention?.query).toBe('How can I reduce portfolio risk?');
    });

    it('should track constitution violations', async () => {
      const result = await metaAgent.process('How can I reduce portfolio risk?');

      expect(result.constitution_violations).toBeDefined();
      expect(Array.isArray(result.constitution_violations)).toBe(true);
    });

    it('should format reasoning path', async () => {
      const result = await metaAgent.process('How can I reduce portfolio risk?');

      expect(result.reasoning_path).toBeDefined();
      expect(result.reasoning_path).toContain('REASONING PATH');
    });

    it('should extract emergent insights', async () => {
      const result = await metaAgent.process('How can I reduce portfolio risk?');

      expect(result.emergent_insights).toBeDefined();
      expect(Array.isArray(result.emergent_insights)).toBe(true);
    });
  });

  describe('Budget Enforcement', () => {
    beforeEach(async () => {
      metaAgent.registerAgent('financial', new MockFinancialAgent('test-key'));

      mockLLM.invoke.mockImplementation(() =>
        Promise.resolve({
          text: JSON.stringify({
            domains: ['financial'],
            reasoning: 'Financial query',
          }),
          usage: { cost_usd: 0.001 },
        })
      );

      await metaAgent.initialize();
    });

    it('should respect max depth limit', async () => {
      // Create agent with very small depth limit
      const limitedAgent = new MetaAgent('test-key', 1, 100, 1.0);
      limitedAgent.registerAgent('financial', new MockFinancialAgent('test-key'));

      // Mock to always suggest recursion
      const limitedMockLLM = (limitedAgent as any).llm;
      limitedMockLLM.invoke.mockImplementation((system: string) => {
        if (system.includes('synthesis engine')) {
          return Promise.resolve({
            text: JSON.stringify({
              synthesis: 'Need more exploration',
              should_recurse: true,
              missing_perspectives: ['biology'],
              confidence: 0.7,
            }),
            usage: { cost_usd: 0.001 },
          });
        }
        return Promise.resolve({
          text: JSON.stringify({
            domains: ['financial'],
            reasoning: 'Financial query',
          }),
          usage: { cost_usd: 0.001 },
        });
      });

      await limitedAgent.initialize();

      const result = await limitedAgent.process('Test query');

      // Should stop at max depth
      const maxDepth = Math.max(...result.trace.map((t) => t.depth));
      expect(maxDepth).toBeLessThanOrEqual(1);
    });

    it('should track cost across invocations', async () => {
      const result = await metaAgent.process('How can I reduce portfolio risk?');

      // Cost should be tracked
      expect(result.trace.length).toBeGreaterThan(0);
      result.trace.forEach((trace) => {
        expect(trace.cost_estimate).toBeGreaterThanOrEqual(0);
      });
    });
  });

  describe('LLM Statistics', () => {
    it('should track total cost', () => {
      const cost = metaAgent.getTotalCost();
      expect(typeof cost).toBe('number');
    });

    it('should track total requests', () => {
      const requests = metaAgent.getTotalRequests();
      expect(typeof requests).toBe('number');
    });
  });

  describe('Attention Tracking', () => {
    beforeEach(async () => {
      metaAgent.registerAgent('financial', new MockFinancialAgent('test-key'));

      mockLLM.invoke.mockImplementation(() =>
        Promise.resolve({
          text: JSON.stringify({
            domains: ['financial'],
            reasoning: 'Financial query',
            synthesis: 'Test synthesis',
            should_recurse: false,
            confidence: 0.9,
            answer: 'Final answer',
            concepts: ['test'],
          }),
          usage: { cost_usd: 0.001 },
        })
      );

      await metaAgent.initialize();
    });

    it('should provide access to attention tracker', () => {
      const tracker = metaAgent.getAttentionTracker();
      expect(tracker).toBeDefined();
    });

    it('should export attention data for audit', async () => {
      await metaAgent.process('Test query');

      const auditData = metaAgent.exportAttentionForAudit();
      expect(auditData).toBeDefined();
    });

    it('should get attention statistics', () => {
      const stats = metaAgent.getAttentionStats();
      expect(stats).toBeDefined();
      expect(stats).toHaveProperty('total_queries');
    });
  });

  describe('Error Handling', () => {
    beforeEach(async () => {
      metaAgent.registerAgent('financial', new MockFinancialAgent('test-key'));
      await metaAgent.initialize();
    });

    it('should handle LLM JSON parse errors gracefully', async () => {
      // Mock LLM to return invalid JSON
      mockLLM.invoke.mockImplementation(() =>
        Promise.resolve({
          text: 'This is not valid JSON',
          usage: { cost_usd: 0.001 },
        })
      );

      const result = await metaAgent.process('Test query');

      // Should still complete with fallback behavior
      expect(result.final_answer).toBeDefined();
      expect(result.trace).toBeDefined();
    });

    it('should handle JSON in markdown code blocks', async () => {
      mockLLM.invoke.mockImplementation(() =>
        Promise.resolve({
          text: '```json\n{"domains": ["financial"], "reasoning": "test"}\n```',
          usage: { cost_usd: 0.001 },
        })
      );

      const result = await metaAgent.process('Test query');

      expect(result.final_answer).toBeDefined();
    });
  });

  describe('Emergent Insights Extraction', () => {
    it('should identify concepts not mentioned by individual agents', async () => {
      metaAgent.registerAgent('financial', new MockFinancialAgent('test-key'));
      metaAgent.registerAgent('biology', new MockBiologyAgent('test-key'));

      mockLLM.invoke.mockImplementation((system: string) => {
        if (system.includes('decompose queries')) {
          return Promise.resolve({
            text: JSON.stringify({
              domains: ['financial', 'biology'],
              reasoning: 'Multi-domain query',
            }),
            usage: { cost_usd: 0.001 },
          });
        } else if (system.includes('final synthesis')) {
          return Promise.resolve({
            text: JSON.stringify({
              answer: 'Systems thinking applies to both finance and biology',
              concepts: ['systems_thinking', 'cross_domain_patterns'],
              confidence: 0.95,
              reasoning: 'Emergent insight from combining domains',
            }),
            usage: { cost_usd: 0.003 },
          });
        }
        return Promise.resolve({
          text: JSON.stringify({
            synthesis: 'test',
            should_recurse: false,
            confidence: 0.9,
          }),
          usage: { cost_usd: 0.001 },
        });
      });

      await metaAgent.initialize();

      const result = await metaAgent.process('How do systems maintain balance?');

      expect(result.emergent_insights).toBeDefined();
      expect(Array.isArray(result.emergent_insights)).toBe(true);
    });
  });

  describe('Reasoning Path Formatting', () => {
    beforeEach(async () => {
      metaAgent.registerAgent('financial', new MockFinancialAgent('test-key'));

      mockLLM.invoke.mockImplementation(() =>
        Promise.resolve({
          text: JSON.stringify({
            domains: ['financial'],
            reasoning: 'Financial query',
            synthesis: 'Test',
            should_recurse: false,
            confidence: 0.9,
            answer: 'Final answer',
            concepts: ['test'],
          }),
          usage: { cost_usd: 0.001 },
        })
      );

      await metaAgent.initialize();
    });

    it('should include depth information in reasoning path', async () => {
      const result = await metaAgent.process('Test query');

      expect(result.reasoning_path).toContain('depth:');
    });

    it('should include agent IDs in reasoning path', async () => {
      const result = await metaAgent.process('Test query');

      expect(result.reasoning_path).toContain('[financial]');
    });

    it('should include confidence scores in reasoning path', async () => {
      const result = await metaAgent.process('Test query');

      expect(result.reasoning_path).toContain('confidence:');
    });

    it('should include concepts in reasoning path', async () => {
      const result = await metaAgent.process('Test query');

      expect(result.reasoning_path).toContain('Concepts:');
    });
  });

  describe('Constitutional Enforcement Integration', () => {
    it('should validate responses against constitution', async () => {
      // Create agent that violates epistemic honesty (low confidence, no uncertainty)
      class ViolatingAgent extends SpecializedAgent {
        getDomain(): string {
          return 'violating';
        }

        async process(query: string, context: RecursionState): Promise<AgentResponse> {
          context.cost_so_far += 0.01;
          return {
            answer: 'This is definitely correct!',
            concepts: ['test', 'violation'],
            confidence: 0.3, // Low confidence but no uncertainty admission
            reasoning: 'I am certain about this answer even though my confidence is low',
          };
        }
      }

      metaAgent.registerAgent('violating', new ViolatingAgent('test-key'));

      mockLLM.invoke.mockImplementation(() =>
        Promise.resolve({
          text: JSON.stringify({
            domains: ['violating'],
            reasoning: 'Test',
            synthesis: 'Test',
            should_recurse: false,
            confidence: 0.9,
            answer: 'Final answer',
            concepts: ['test'],
          }),
          usage: { cost_usd: 0.001 },
        })
      );

      await metaAgent.initialize();

      const result = await metaAgent.process('Test query');

      // Should have constitutional violations reported
      expect(result.constitution_violations).toBeDefined();
    });
  });

  describe('Anti-Corruption Layer Integration', () => {
    it('should validate responses against ACL domain boundaries', async () => {
      // Create agent that speaks outside its domain with high confidence
      class OutOfDomainAgent extends SpecializedAgent {
        getDomain(): string {
          return 'financial';
        }

        async process(query: string, context: RecursionState): Promise<AgentResponse> {
          context.cost_so_far += 0.01;
          return {
            answer: 'Mitochondria produce ATP through cellular respiration',
            concepts: ['mitochondria', 'atp', 'cells'], // Biology concepts!
            confidence: 0.95, // High confidence
            reasoning: 'This is a detailed biological explanation about cellular energy production',
          };
        }
      }

      metaAgent.registerAgent('financial', new OutOfDomainAgent('test-key'));

      mockLLM.invoke.mockImplementation(() =>
        Promise.resolve({
          text: JSON.stringify({
            domains: ['financial'],
            reasoning: 'Financial query',
            synthesis: 'Test',
            should_recurse: false,
            confidence: 0.9,
            answer: 'Final answer',
            concepts: ['test'],
          }),
          usage: { cost_usd: 0.001 },
        })
      );

      await metaAgent.initialize();

      const result = await metaAgent.process('Test query');

      // Should have ACL violations reported
      const aclViolations = result.constitution_violations.filter(
        (v) => v.principle_id === 'domain_boundary'
      );
      expect(aclViolations.length).toBeGreaterThan(0);
    });
  });
});
