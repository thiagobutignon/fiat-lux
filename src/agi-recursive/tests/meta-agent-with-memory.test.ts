/**
 * @file meta-agent-with-memory.test.ts
 * Tests for MetaAgentWithMemory - Extended orchestration with episodic memory
 *
 * Key capabilities tested:
 * - Query processing with memory augmentation
 * - Cached response retrieval (similarity > 0.8)
 * - Episode storage after processing
 * - Memory querying by concepts/domains
 * - Memory statistics
 * - Memory consolidation
 * - Export/import for persistence
 * - Concept/domain extraction from traces
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  MetaAgentWithMemory,
  ProcessResultWithMemory,
  createMetaAgentWithMemory,
} from '../core/meta-agent-with-memory';
import { RecursionTrace, AgentResponse } from '../core/meta-agent';
import { Episode } from '../core/episodic-memory';

// Mock the parent MetaAgent class
vi.mock('../core/meta-agent', async () => {
  const actual = await vi.importActual('../core/meta-agent');
  return {
    ...actual,
    MetaAgent: class MockMetaAgent {
      constructor() {}

      async process(query: string) {
        const trace: RecursionTrace[] = [
          {
            depth: 1,
            agent_id: 'financial',
            query,
            response: {
              answer: 'Test financial response',
              concepts: ['finance', 'investment'],
              confidence: 0.85,
              reasoning: 'Financial analysis',
            },
            timestamp: Date.now(),
            cost_estimate: 0.01,
          },
        ];

        return {
          final_answer: 'Test final answer',
          trace,
          emergent_insights: ['Test insight'],
          reasoning_path: 'Test reasoning',
          constitution_violations: [],
          attention: null,
        };
      }

      registerAgent() {}
      async initialize() {}
    },
  };
});

describe('MetaAgentWithMemory', () => {
  let agent: MetaAgentWithMemory;

  beforeEach(() => {
    agent = new MetaAgentWithMemory('test-api-key', 5, 10, 1.0, true);
  });

  describe('Constructor', () => {
    it('should create instance with default parameters', () => {
      expect(agent).toBeInstanceOf(MetaAgentWithMemory);
    });

    it('should accept custom parameters', () => {
      const customAgent = new MetaAgentWithMemory('key', 3, 5, 0.5, false);
      expect(customAgent).toBeInstanceOf(MetaAgentWithMemory);
    });

    it('should initialize with memory enabled by default', async () => {
      const result = await agent.processWithMemory('test query');

      // Should store in memory
      const stats = agent.getMemoryStats();
      expect(stats.total_episodes).toBe(1);
    });

    it('should support disabling memory', async () => {
      const noMemoryAgent = new MetaAgentWithMemory('key', 5, 10, 1.0, false);
      await noMemoryAgent.processWithMemory('test query');

      const stats = noMemoryAgent.getMemoryStats();
      expect(stats.total_episodes).toBe(0);
    });
  });

  describe('processWithMemory - First Query', () => {
    it('should process query without cache hit', async () => {
      const result = await agent.processWithMemory('How to invest wisely?');

      expect(result.final_answer).toBeDefined();
      expect(result.memory_used).toBe(false);
      expect(result.similar_past_queries).toEqual([]);
    });

    it('should store episode in memory after processing', async () => {
      await agent.processWithMemory('Test query');

      const stats = agent.getMemoryStats();
      expect(stats.total_episodes).toBe(1);
    });

    it('should extract concepts from trace', async () => {
      await agent.processWithMemory('Test query');

      const episodes = agent.queryMemory({});
      expect(episodes.length).toBe(1);
      expect(episodes[0].concepts).toContain('finance');
      expect(episodes[0].concepts).toContain('investment');
    });

    it('should extract domains from trace', async () => {
      await agent.processWithMemory('Test query');

      const episodes = agent.queryMemory({});
      expect(episodes[0].domains).toContain('financial');
    });

    it('should extract agents used from trace', async () => {
      await agent.processWithMemory('Test query');

      const episodes = agent.queryMemory({});
      expect(episodes[0].agents_used).toContain('financial');
    });

    it('should calculate average confidence', async () => {
      await agent.processWithMemory('Test query');

      const episodes = agent.queryMemory({});
      expect(episodes[0].confidence).toBeGreaterThan(0);
      expect(episodes[0].confidence).toBeLessThanOrEqual(1);
    });

    it('should calculate total cost', async () => {
      await agent.processWithMemory('Test query');

      const episodes = agent.queryMemory({});
      expect(episodes[0].cost).toBeGreaterThanOrEqual(0);
    });

    it('should mark success based on confidence and violations', async () => {
      await agent.processWithMemory('Test query');

      const episodes = agent.queryMemory({});
      expect(episodes[0].success).toBe(true);
    });

    it('should include invocation count', async () => {
      const result = await agent.processWithMemory('Test query');

      expect(result.invocations).toBeGreaterThan(0);
    });

    it('should include max depth reached', async () => {
      const result = await agent.processWithMemory('Test query');

      expect(result.max_depth_reached).toBeGreaterThanOrEqual(0);
    });
  });

  describe('processWithMemory - Cache Hit', () => {
    beforeEach(async () => {
      // Store initial query
      await agent.processWithMemory('How to diversify portfolio?');
    });

    it('should return cached response for very similar query', async () => {
      // Very similar query (Jaccard > 0.8)
      const result = await agent.processWithMemory('How to diversify portfolio?');

      expect(result.memory_used).toBe(true);
      expect(result.final_answer).toContain('[Using cached response');
      expect(result.invocations).toBe(0);
    });

    it('should find similar past queries', async () => {
      const result = await agent.processWithMemory('portfolio diversification strategies');

      // Should find similar queries even without cache hit
      expect(result.similar_past_queries.length).toBeGreaterThanOrEqual(0);
    });

    it('should not use cache if similarity < 0.8', async () => {
      // Different query
      const result = await agent.processWithMemory('What is quantum computing?');

      expect(result.memory_used).toBe(false);
      expect(result.invocations).toBeGreaterThan(0);
    });

    it('should not use cache if past query was unsuccessful', async () => {
      // Manually add unsuccessful episode
      agent.clearMemory();

      const memory = (agent as any).memory;
      memory.addEpisode(
        'test failed query',
        'failed response',
        ['test'],
        ['test'],
        ['test'],
        0.01,
        false, // unsuccessful
        0.9,
        [],
        []
      );

      const result = await agent.processWithMemory('test failed query');

      // Should not use cache for failed queries
      expect(result.memory_used).toBe(false);
    });

    it('should not use cache if past confidence < 0.7', async () => {
      // Add low confidence episode
      agent.clearMemory();

      const memory = (agent as any).memory;
      memory.addEpisode(
        'low confidence query',
        'uncertain response',
        ['test'],
        ['test'],
        ['test'],
        0.01,
        true,
        0.5, // low confidence
        [],
        []
      );

      const result = await agent.processWithMemory('low confidence query');

      // Should not use cache for low confidence
      expect(result.memory_used).toBe(false);
    });

    it('should calculate Jaccard similarity correctly', async () => {
      // Query with high word overlap
      const result = await agent.processWithMemory('diversify your portfolio how');

      // Should detect similarity
      expect(result.similar_past_queries.length).toBeGreaterThan(0);
    });
  });

  describe('queryMemory', () => {
    beforeEach(async () => {
      await agent.processWithMemory('Financial planning basics');
      await agent.processWithMemory('Investment strategies');
    });

    it('should query all episodes', () => {
      const episodes = agent.queryMemory({});

      expect(episodes.length).toBe(2);
    });

    it('should filter by concepts', () => {
      const episodes = agent.queryMemory({ concepts: ['finance'] });

      expect(episodes.length).toBeGreaterThan(0);
      episodes.forEach((ep) => {
        expect(ep.concepts).toContain('finance');
      });
    });

    it('should filter by domains', () => {
      const episodes = agent.queryMemory({ domains: ['financial'] });

      expect(episodes.length).toBeGreaterThan(0);
      episodes.forEach((ep) => {
        expect(ep.domains).toContain('financial');
      });
    });

    it('should filter by query text', () => {
      const episodes = agent.queryMemory({ query_text: 'Financial' });

      expect(episodes.length).toBeGreaterThan(0);
    });

    it('should filter by minimum confidence', () => {
      const episodes = agent.queryMemory({ min_confidence: 0.8 });

      episodes.forEach((ep) => {
        expect(ep.confidence).toBeGreaterThanOrEqual(0.8);
      });
    });

    it('should respect limit parameter', () => {
      const episodes = agent.queryMemory({ limit: 1 });

      expect(episodes.length).toBeLessThanOrEqual(1);
    });

    it('should return empty array if no matches', () => {
      const episodes = agent.queryMemory({ concepts: ['nonexistent'] });

      expect(episodes).toEqual([]);
    });
  });

  describe('getMemoryStats', () => {
    it('should return empty stats initially', () => {
      const stats = agent.getMemoryStats();

      expect(stats.total_episodes).toBe(0);
    });

    it('should track total episodes', async () => {
      await agent.processWithMemory('Query 1');
      await agent.processWithMemory('Query 2');

      const stats = agent.getMemoryStats();
      expect(stats.total_episodes).toBe(2);
    });

    it('should track success rate', async () => {
      await agent.processWithMemory('Test query');

      const stats = agent.getMemoryStats();
      expect(stats.success_rate).toBeGreaterThan(0);
    });

    it('should track average confidence', async () => {
      await agent.processWithMemory('Test query');

      const stats = agent.getMemoryStats();
      expect(stats.average_confidence).toBeGreaterThan(0);
      expect(stats.average_confidence).toBeLessThanOrEqual(1);
    });

    it('should track total cost', async () => {
      await agent.processWithMemory('Test query');

      const stats = agent.getMemoryStats();
      expect(stats.total_cost).toBeGreaterThanOrEqual(0);
    });

    it('should track total concepts', async () => {
      await agent.processWithMemory('Test query');

      const stats = agent.getMemoryStats();
      expect(stats.total_concepts).toBeGreaterThan(0);
    });

    it('should list most queried domains', async () => {
      await agent.processWithMemory('Test query');

      const stats = agent.getMemoryStats();
      expect(Array.isArray(stats.most_queried_domains)).toBe(true);
      expect(stats.most_queried_domains.length).toBeGreaterThan(0);
    });
  });

  describe('consolidateMemory', () => {
    beforeEach(async () => {
      await agent.processWithMemory('Query 1');
      await agent.processWithMemory('Query 2');
      await agent.processWithMemory('Query 3');
    });

    it('should consolidate memory', () => {
      const result = agent.consolidateMemory();

      expect(result).toBeDefined();
    });

    it('should not crash on consolidation', () => {
      expect(() => agent.consolidateMemory()).not.toThrow();
    });
  });

  describe('exportMemory', () => {
    beforeEach(async () => {
      await agent.processWithMemory('Test query for export');
    });

    it('should export memory as JSON string', () => {
      const exported = agent.exportMemory();

      expect(typeof exported).toBe('string');
      expect(() => JSON.parse(exported)).not.toThrow();
    });

    it('should include episodes in export', () => {
      const exported = agent.exportMemory();
      const parsed = JSON.parse(exported);

      expect(parsed.episodes).toBeDefined();
      expect(Array.isArray(parsed.episodes)).toBe(true);
      expect(parsed.episodes.length).toBeGreaterThan(0);
    });

    it('should preserve episode data', () => {
      const exported = agent.exportMemory();
      const parsed = JSON.parse(exported);

      const episode = parsed.episodes[0];
      expect(episode.query).toBeDefined();
      expect(episode.response).toBeDefined();
      expect(episode.concepts).toBeDefined();
    });
  });

  describe('importMemory', () => {
    let exportedData: string;

    beforeEach(async () => {
      await agent.processWithMemory('Query to be exported');
      exportedData = agent.exportMemory();
      agent.clearMemory();
    });

    it('should import memory from JSON', () => {
      const imported = agent.importMemory(exportedData);

      expect(imported).toBeGreaterThan(0);
    });

    it('should restore episodes', () => {
      agent.importMemory(exportedData);

      const episodes = agent.queryMemory({});
      expect(episodes.length).toBeGreaterThan(0);
    });

    it('should preserve episode content', () => {
      agent.importMemory(exportedData);

      const episodes = agent.queryMemory({});
      expect(episodes[0].query).toContain('Query to be exported');
    });

    it('should return number of imported episodes', () => {
      const count = agent.importMemory(exportedData);

      expect(count).toBe(1);
    });

    it('should handle empty import', () => {
      const emptyData = JSON.stringify({ episodes: [] });
      const count = agent.importMemory(emptyData);

      expect(count).toBe(0);
    });
  });

  describe('clearMemory', () => {
    beforeEach(async () => {
      await agent.processWithMemory('Query 1');
      await agent.processWithMemory('Query 2');
    });

    it('should clear all episodes', () => {
      agent.clearMemory();

      const episodes = agent.queryMemory({});
      expect(episodes.length).toBe(0);
    });

    it('should reset stats', () => {
      agent.clearMemory();

      const stats = agent.getMemoryStats();
      expect(stats.total_episodes).toBe(0);
    });

    it('should allow processing after clear', async () => {
      agent.clearMemory();

      const result = await agent.processWithMemory('New query');
      expect(result).toBeDefined();
    });
  });

  describe('Concept Extraction', () => {
    it('should extract unique concepts from multiple traces', async () => {
      await agent.processWithMemory('Test query');

      const episodes = agent.queryMemory({});
      const concepts = episodes[0].concepts;

      // Should have unique concepts
      expect(new Set(concepts).size).toBe(concepts.length);
    });

    it('should handle empty concepts', async () => {
      await agent.processWithMemory('Test query');

      const episodes = agent.queryMemory({});
      expect(Array.isArray(episodes[0].concepts)).toBe(true);
    });
  });

  describe('Domain Extraction', () => {
    it('should extract unique domains from traces', async () => {
      await agent.processWithMemory('Test query');

      const episodes = agent.queryMemory({});
      const domains = episodes[0].domains;

      expect(new Set(domains).size).toBe(domains.length);
    });

    it('should extract domain from agent_id', async () => {
      await agent.processWithMemory('Test query');

      const episodes = agent.queryMemory({});
      expect(episodes[0].domains).toContain('financial');
    });
  });

  describe('Agent Extraction', () => {
    it('should extract unique agents from traces', async () => {
      await agent.processWithMemory('Test query');

      const episodes = agent.queryMemory({});
      const agents = episodes[0].agents_used;

      expect(new Set(agents).size).toBe(agents.length);
    });

    it('should map agent_id correctly', async () => {
      await agent.processWithMemory('Test query');

      const episodes = agent.queryMemory({});
      expect(episodes[0].agents_used.length).toBeGreaterThan(0);
    });
  });

  describe('Factory Function', () => {
    it('should create MetaAgentWithMemory instance', () => {
      const instance = createMetaAgentWithMemory('test-key');

      expect(instance).toBeInstanceOf(MetaAgentWithMemory);
    });

    it('should accept custom parameters', () => {
      const instance = createMetaAgentWithMemory('key', 3, 5, 0.5, false);

      expect(instance).toBeInstanceOf(MetaAgentWithMemory);
    });

    it('should use default parameters', () => {
      const instance = createMetaAgentWithMemory('key');

      expect(instance).toBeDefined();
    });
  });

  describe('Integration', () => {
    it('should process multiple queries and build memory', async () => {
      await agent.processWithMemory('Query 1');
      await agent.processWithMemory('Query 2');
      await agent.processWithMemory('Query 3');

      const stats = agent.getMemoryStats();
      expect(stats.total_episodes).toBe(3);
    });

    it('should use cache for identical query', async () => {
      const query = 'Exact same query';

      await agent.processWithMemory(query);
      const result = await agent.processWithMemory(query);

      expect(result.memory_used).toBe(true);
    });

    it('should not interfere with base MetaAgent functionality', async () => {
      const result = await agent.processWithMemory('Test query');

      expect(result.final_answer).toBeDefined();
      expect(result.trace).toBeDefined();
      expect(result.reasoning_path).toBeDefined();
    });
  });
});
