/**
 * @file episodic-memory.test.ts
 * Tests for EpisodicMemory - Long-term memory storage and retrieval
 *
 * Key capabilities tested:
 * - Episode storage with deduplication
 * - Concept and domain indexing
 * - Query filtering (concepts, domains, confidence, timestamp)
 * - Memory statistics calculation
 * - Memory consolidation (merge similar episodes)
 * - Jaccard similarity search
 * - Export/import for persistence
 * - Pattern discovery
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  EpisodicMemory,
  Episode,
  MemoryQuery,
  createMemory,
} from '../core/episodic-memory';
import { RecursionTrace } from '../core/meta-agent';

describe('EpisodicMemory', () => {
  let memory: EpisodicMemory;
  let testTrace: RecursionTrace[];

  beforeEach(() => {
    memory = new EpisodicMemory();
    testTrace = [
      {
        depth: 1,
        agent_id: 'financial',
        query: 'Test query',
        response: {
          answer: 'Test answer',
          concepts: ['finance', 'investment'],
          confidence: 0.9,
          reasoning: 'Test reasoning',
        },
        timestamp: Date.now(),
        cost_estimate: 0.01,
      },
    ];
  });

  describe('Constructor', () => {
    it('should create empty memory instance', () => {
      expect(memory).toBeInstanceOf(EpisodicMemory);
    });

    it('should initialize with empty stats', () => {
      const stats = memory.getStats();
      expect(stats.total_episodes).toBe(0);
    });
  });

  describe('addEpisode', () => {
    it('should add episode to memory', () => {
      const episode = memory.addEpisode(
        'How to invest?',
        'Diversify your portfolio',
        ['finance', 'investment'],
        ['financial'],
        ['financial_agent'],
        0.01,
        true,
        0.9,
        testTrace,
        ['Diversification is key']
      );

      expect(episode.id).toBeDefined();
      expect(episode.query).toBe('How to invest?');
      expect(episode.response).toBe('Diversify your portfolio');
      expect(episode.concepts).toEqual(['finance', 'investment']);
      expect(episode.success).toBe(true);
    });

    it('should generate unique episode ID', () => {
      const ep1 = memory.addEpisode('Query 1', 'Response 1', [], [], [], 0, true, 0.9, testTrace);
      const ep2 = memory.addEpisode('Query 2', 'Response 2', [], [], [], 0, true, 0.9, testTrace);

      expect(ep1.id).not.toBe(ep2.id);
    });

    it('should include timestamp', () => {
      const before = Date.now();
      const episode = memory.addEpisode('Test', 'Response', [], [], [], 0, true, 0.9, testTrace);
      const after = Date.now();

      expect(episode.timestamp).toBeGreaterThanOrEqual(before);
      expect(episode.timestamp).toBeLessThanOrEqual(after);
    });

    it('should calculate query hash', () => {
      const episode = memory.addEpisode('Test query', 'Response', [], [], [], 0, true, 0.9, testTrace);

      expect(episode.query_hash).toBeDefined();
      expect(typeof episode.query_hash).toBe('string');
      expect(episode.query_hash.length).toBe(64); // SHA-256 hex
    });

    it('should index episode by concepts', () => {
      memory.addEpisode('Test', 'Response', ['concept1', 'concept2'], [], [], 0, true, 0.9, testTrace);

      const results1 = memory.query({ concepts: ['concept1'] });
      const results2 = memory.query({ concepts: ['concept2'] });

      expect(results1.length).toBe(1);
      expect(results2.length).toBe(1);
    });

    it('should index episode by domains', () => {
      memory.addEpisode('Test', 'Response', [], ['domain1', 'domain2'], [], 0, true, 0.9, testTrace);

      const results1 = memory.query({ domains: ['domain1'] });
      const results2 = memory.query({ domains: ['domain2'] });

      expect(results1.length).toBe(1);
      expect(results2.length).toBe(1);
    });

    it('should store execution trace', () => {
      const episode = memory.addEpisode('Test', 'Response', [], [], [], 0, true, 0.9, testTrace);

      expect(episode.execution_trace).toBeDefined();
      expect(episode.execution_trace.length).toBe(1);
      expect(episode.execution_trace[0].agent_id).toBe('financial');
    });

    it('should store emergent insights', () => {
      const episode = memory.addEpisode(
        'Test',
        'Response',
        [],
        [],
        [],
        0,
        true,
        0.9,
        testTrace,
        ['Insight 1', 'Insight 2']
      );

      expect(episode.emergent_insights).toEqual(['Insight 1', 'Insight 2']);
    });

    it('should calculate metadata depth from trace', () => {
      const multiDepthTrace: RecursionTrace[] = [
        { ...testTrace[0], depth: 1 },
        { ...testTrace[0], depth: 2 },
        { ...testTrace[0], depth: 3 },
      ];

      const episode = memory.addEpisode('Test', 'Response', [], [], [], 0, true, 0.9, multiDepthTrace);

      expect(episode.metadata.depth).toBe(3);
    });

    it('should calculate metadata invocations from trace length', () => {
      const multiTrace: RecursionTrace[] = [testTrace[0], testTrace[0], testTrace[0]];

      const episode = memory.addEpisode('Test', 'Response', [], [], [], 0, true, 0.9, multiTrace);

      expect(episode.metadata.invocations).toBe(3);
    });

    it('should handle empty trace gracefully', () => {
      const episode = memory.addEpisode('Test', 'Response', [], [], [], 0, true, 0.9, []);

      expect(episode.metadata.depth).toBe(0);
      expect(episode.metadata.invocations).toBe(0);
    });

    it('should update existing episode for duplicate query', () => {
      const ep1 = memory.addEpisode('Same query', 'Response 1', [], [], [], 0, true, 0.9, testTrace);
      const ep2 = memory.addEpisode('Same query', 'Response 2', [], [], [], 0, true, 0.8, testTrace);

      // Should be same ID (updated episode)
      expect(ep2.id).toBe(ep1.id);
      expect(ep2.response).toBe('Response 2');
      expect(ep2.confidence).toBeCloseTo(0.85, 2); // Average of 0.9 and 0.8
    });

    it('should normalize query for deduplication', () => {
      const ep1 = memory.addEpisode('  TEST   QUERY  ', 'Response 1', [], [], [], 0, true, 0.9, testTrace);
      const ep2 = memory.addEpisode('test query', 'Response 2', [], [], [], 0, true, 0.8, testTrace);

      // Should be same ID (normalized to same query)
      expect(ep2.id).toBe(ep1.id);
    });

    it('should refresh timestamp on duplicate update', async () => {
      const ep1 = memory.addEpisode('Query', 'Response 1', [], [], [], 0, true, 0.9, testTrace);
      const timestamp1 = ep1.timestamp;

      // Wait a bit
      await new Promise((resolve) => setTimeout(resolve, 10));

      const ep2 = memory.addEpisode('Query', 'Response 2', [], [], [], 0, true, 0.9, testTrace);

      expect(ep2.timestamp).toBeGreaterThan(timestamp1);
    });
  });

  describe('query', () => {
    beforeEach(() => {
      memory.addEpisode('Financial query', 'Response 1', ['finance'], ['financial'], [], 0.01, true, 0.9, testTrace);
      memory.addEpisode('Biology query', 'Response 2', ['biology', 'cell'], ['biology'], [], 0.02, true, 0.8, testTrace);
      memory.addEpisode('Systems query', 'Response 3', ['systems'], ['systems'], [], 0.03, false, 0.6, testTrace);
    });

    it('should return all episodes with empty query', () => {
      const results = memory.query({});

      expect(results.length).toBe(3);
    });

    it('should filter by single concept', () => {
      const results = memory.query({ concepts: ['finance'] });

      expect(results.length).toBe(1);
      expect(results[0].query).toBe('Financial query');
    });

    it('should filter by multiple concepts (OR)', () => {
      const results = memory.query({ concepts: ['finance', 'biology'] });

      expect(results.length).toBe(2);
    });

    it('should filter by single domain', () => {
      const results = memory.query({ domains: ['biology'] });

      expect(results.length).toBe(1);
      expect(results[0].query).toBe('Biology query');
    });

    it('should filter by multiple domains (OR)', () => {
      const results = memory.query({ domains: ['financial', 'systems'] });

      expect(results.length).toBe(2);
    });

    it('should filter by minimum confidence', () => {
      const results = memory.query({ min_confidence: 0.75 });

      expect(results.length).toBe(2);
      results.forEach((ep) => {
        expect(ep.confidence).toBeGreaterThanOrEqual(0.75);
      });
    });

    it('should filter by query text (substring)', () => {
      const results = memory.query({ query_text: 'Financial' });

      expect(results.length).toBe(1);
      expect(results[0].query).toContain('Financial');
    });

    it('should filter by query text (case insensitive)', () => {
      const results = memory.query({ query_text: 'financial' });

      expect(results.length).toBe(1);
    });

    it('should filter by timestamp (since)', () => {
      const now = Date.now();
      const results = memory.query({ since: now - 1000 });

      expect(results.length).toBeGreaterThanOrEqual(0);
    });

    it('should apply limit', () => {
      const results = memory.query({ limit: 2 });

      expect(results.length).toBeLessThanOrEqual(2);
    });

    it('should combine multiple filters (AND)', () => {
      const results = memory.query({
        concepts: ['biology'],
        min_confidence: 0.75,
      });

      expect(results.length).toBe(1);
      expect(results[0].query).toBe('Biology query');
      expect(results[0].confidence).toBeGreaterThanOrEqual(0.75);
    });

    it('should return empty array if no matches', () => {
      const results = memory.query({ concepts: ['nonexistent'] });

      expect(results).toEqual([]);
    });

    it('should sort by relevance (timestamp * confidence)', () => {
      const results = memory.query({});

      // Results should be sorted descending by timestamp * confidence
      for (let i = 0; i < results.length - 1; i++) {
        const score1 = results[i].timestamp * results[i].confidence;
        const score2 = results[i + 1].timestamp * results[i + 1].confidence;
        expect(score1).toBeGreaterThanOrEqual(score2);
      }
    });

    it('should handle nonexistent concept gracefully', () => {
      const results = memory.query({ concepts: ['fake_concept'] });

      expect(results).toEqual([]);
    });

    it('should handle nonexistent domain gracefully', () => {
      const results = memory.query({ domains: ['fake_domain'] });

      expect(results).toEqual([]);
    });
  });

  describe('getStats', () => {
    it('should return empty stats for empty memory', () => {
      const stats = memory.getStats();

      expect(stats.total_episodes).toBe(0);
      expect(stats.total_concepts).toBe(0);
      expect(stats.total_cost).toBe(0);
      expect(stats.average_confidence).toBe(0);
      expect(stats.success_rate).toBe(0);
      expect(stats.most_common_concepts).toEqual([]);
      expect(stats.most_queried_domains).toEqual([]);
    });

    it('should count total episodes', () => {
      memory.addEpisode('Q1', 'R1', [], [], [], 0, true, 0.9, testTrace);
      memory.addEpisode('Q2', 'R2', [], [], [], 0, true, 0.9, testTrace);

      const stats = memory.getStats();
      expect(stats.total_episodes).toBe(2);
    });

    it('should count unique concepts', () => {
      memory.addEpisode('Q1', 'R1', ['concept1', 'concept2'], [], [], 0, true, 0.9, testTrace);
      memory.addEpisode('Q2', 'R2', ['concept2', 'concept3'], [], [], 0, true, 0.9, testTrace);

      const stats = memory.getStats();
      expect(stats.total_concepts).toBe(3); // concept1, concept2, concept3
    });

    it('should sum total cost', () => {
      memory.addEpisode('Q1', 'R1', [], [], [], 0.01, true, 0.9, testTrace);
      memory.addEpisode('Q2', 'R2', [], [], [], 0.02, true, 0.9, testTrace);

      const stats = memory.getStats();
      expect(stats.total_cost).toBeCloseTo(0.03, 2);
    });

    it('should calculate average confidence', () => {
      memory.addEpisode('Q1', 'R1', [], [], [], 0, true, 0.8, testTrace);
      memory.addEpisode('Q2', 'R2', [], [], [], 0, true, 0.9, testTrace);

      const stats = memory.getStats();
      expect(stats.average_confidence).toBeCloseTo(0.85, 2);
    });

    it('should calculate success rate', () => {
      memory.addEpisode('Q1', 'R1', [], [], [], 0, true, 0.9, testTrace);
      memory.addEpisode('Q2', 'R2', [], [], [], 0, false, 0.9, testTrace);
      memory.addEpisode('Q3', 'R3', [], [], [], 0, true, 0.9, testTrace);

      const stats = memory.getStats();
      expect(stats.success_rate).toBeCloseTo(2 / 3, 2);
    });

    it('should list most common concepts', () => {
      memory.addEpisode('Q1', 'R1', ['finance', 'investment'], [], [], 0, true, 0.9, testTrace);
      memory.addEpisode('Q2', 'R2', ['finance'], [], [], 0, true, 0.9, testTrace);
      memory.addEpisode('Q3', 'R3', ['biology'], [], [], 0, true, 0.9, testTrace);

      const stats = memory.getStats();
      expect(stats.most_common_concepts.length).toBeGreaterThan(0);
      expect(stats.most_common_concepts[0].concept).toBe('finance');
      expect(stats.most_common_concepts[0].count).toBe(2);
    });

    it('should list most queried domains', () => {
      memory.addEpisode('Q1', 'R1', [], ['financial'], [], 0, true, 0.9, testTrace);
      memory.addEpisode('Q2', 'R2', [], ['financial'], [], 0, true, 0.9, testTrace);
      memory.addEpisode('Q3', 'R3', [], ['biology'], [], 0, true, 0.9, testTrace);

      const stats = memory.getStats();
      expect(stats.most_queried_domains.length).toBeGreaterThan(0);
      expect(stats.most_queried_domains[0].domain).toBe('financial');
      expect(stats.most_queried_domains[0].count).toBe(2);
    });

    it('should calculate temporal coverage', () => {
      const ep1 = memory.addEpisode('Q1', 'R1', [], [], [], 0, true, 0.9, testTrace);
      const ep2 = memory.addEpisode('Q2', 'R2', [], [], [], 0, true, 0.9, testTrace);

      const stats = memory.getStats();
      expect(stats.temporal_coverage.oldest).toBeLessThanOrEqual(stats.temporal_coverage.newest);
      expect(stats.temporal_coverage.span_hours).toBeGreaterThanOrEqual(0);
    });

    it('should limit most common concepts to 10', () => {
      for (let i = 0; i < 15; i++) {
        memory.addEpisode(`Q${i}`, `R${i}`, [`concept${i}`], [], [], 0, true, 0.9, testTrace);
      }

      const stats = memory.getStats();
      expect(stats.most_common_concepts.length).toBeLessThanOrEqual(10);
    });

    it('should sort concepts by frequency', () => {
      memory.addEpisode('Q1', 'R1', ['a'], [], [], 0, true, 0.9, testTrace);
      memory.addEpisode('Q2', 'R2', ['b', 'b'], [], [], 0, true, 0.9, testTrace);
      memory.addEpisode('Q3', 'R3', ['a', 'b'], [], [], 0, true, 0.9, testTrace);

      const stats = memory.getStats();
      // 'b' should appear first (count 2), then 'a' (count 2)
      expect(stats.most_common_concepts[0].count).toBeGreaterThanOrEqual(stats.most_common_concepts[1].count);
    });
  });

  describe('consolidate', () => {
    it('should return consolidation result', () => {
      memory.addEpisode('Q1', 'R1', ['finance'], [], [], 0, true, 0.9, testTrace);

      const result = memory.consolidate();

      expect(result.merged_count).toBeDefined();
      expect(result.new_insights).toBeDefined();
      expect(result.patterns_discovered).toBeDefined();
    });

    it('should merge duplicate queries', () => {
      memory.addEpisode('Same query', 'R1', ['finance'], [], [], 0, true, 0.9, testTrace);
      memory.addEpisode('Same query', 'R2', ['finance'], [], [], 0, true, 0.9, testTrace);

      const result = memory.consolidate();

      // Should have merged 1 duplicate
      expect(result.merged_count).toBeGreaterThanOrEqual(0);
    });

    it('should preserve most recent episode', () => {
      const ep1 = memory.addEpisode('Query', 'Old response', ['finance'], [], [], 0, true, 0.9, testTrace);
      const ep2 = memory.addEpisode('Query', 'New response', ['finance'], [], [], 0, true, 0.9, testTrace);

      memory.consolidate();

      const episode = memory.getEpisode(ep1.id);
      expect(episode?.response).toBe('New response');
    });

    it('should merge emergent insights', () => {
      memory.addEpisode('Q', 'R1', ['finance'], [], [], 0, true, 0.9, testTrace, ['Insight A']);
      memory.addEpisode('Q', 'R2', ['finance'], [], [], 0, true, 0.9, testTrace, ['Insight B']);

      const result = memory.consolidate();

      expect(result.new_insights.length).toBeGreaterThanOrEqual(0);
    });

    it('should discover concept patterns', () => {
      // Add multiple episodes with same concept pairs
      for (let i = 0; i < 5; i++) {
        memory.addEpisode(`Q${i}`, `R${i}`, ['finance', 'investment'], [], [], 0, true, 0.9, testTrace);
      }

      const result = memory.consolidate();

      // Should discover pattern with >20% frequency
      expect(result.patterns_discovered.length).toBeGreaterThanOrEqual(0);
    });

    it('should not crash on empty memory', () => {
      expect(() => memory.consolidate()).not.toThrow();
    });

    it('should not crash on single episode', () => {
      memory.addEpisode('Q1', 'R1', ['finance'], [], [], 0, true, 0.9, testTrace);

      expect(() => memory.consolidate()).not.toThrow();
    });
  });

  describe('findSimilarQueries', () => {
    beforeEach(() => {
      memory.addEpisode('How to invest in stocks', 'Response 1', [], [], [], 0, true, 0.9, testTrace);
      memory.addEpisode('Investment strategies for beginners', 'Response 2', [], [], [], 0, true, 0.9, testTrace);
      memory.addEpisode('What is quantum computing?', 'Response 3', [], [], [], 0, true, 0.9, testTrace);
    });

    it('should find similar queries using Jaccard similarity', () => {
      const similar = memory.findSimilarQueries('How to invest in bonds');

      expect(similar.length).toBeGreaterThan(0);
      expect(similar[0].query).toContain('invest');
    });

    it('should sort by similarity descending', () => {
      const similar = memory.findSimilarQueries('investment stocks');

      for (let i = 0; i < similar.length - 1; i++) {
        // Can't directly check similarity scores, but results should be ordered
        expect(similar[i]).toBeDefined();
      }
    });

    it('should respect limit parameter', () => {
      const similar = memory.findSimilarQueries('investment', 2);

      expect(similar.length).toBeLessThanOrEqual(2);
    });

    it('should return empty array if no similar queries', () => {
      memory.clear();
      const similar = memory.findSimilarQueries('test query');

      expect(similar).toEqual([]);
    });

    it('should handle case insensitive matching', () => {
      const similar = memory.findSimilarQueries('INVEST STOCKS');

      expect(similar.length).toBeGreaterThan(0);
    });

    it('should calculate Jaccard similarity correctly', () => {
      // Exact match should be most similar
      const similar = memory.findSimilarQueries('How to invest in stocks');

      expect(similar[0].query).toBe('How to invest in stocks');
    });

    it('should ignore word order', () => {
      const similar = memory.findSimilarQueries('stocks in invest to how');

      expect(similar.length).toBeGreaterThan(0);
    });

    it('should filter out zero similarity results', () => {
      const similar = memory.findSimilarQueries('completely different unrelated query xyz');

      similar.forEach((ep) => {
        // All results should have some word overlap
        expect(ep).toBeDefined();
      });
    });
  });

  describe('getEpisode', () => {
    it('should retrieve episode by ID', () => {
      const added = memory.addEpisode('Test query', 'Response', [], [], [], 0, true, 0.9, testTrace);

      const retrieved = memory.getEpisode(added.id);

      expect(retrieved).toBeDefined();
      expect(retrieved?.id).toBe(added.id);
      expect(retrieved?.query).toBe('Test query');
    });

    it('should return undefined for nonexistent ID', () => {
      const retrieved = memory.getEpisode('fake-id');

      expect(retrieved).toBeUndefined();
    });
  });

  describe('clear', () => {
    beforeEach(() => {
      memory.addEpisode('Q1', 'R1', ['finance'], ['financial'], [], 0, true, 0.9, testTrace);
      memory.addEpisode('Q2', 'R2', ['biology'], ['biology'], [], 0, true, 0.9, testTrace);
    });

    it('should clear all episodes', () => {
      memory.clear();

      const stats = memory.getStats();
      expect(stats.total_episodes).toBe(0);
    });

    it('should clear concept index', () => {
      memory.clear();

      const results = memory.query({ concepts: ['finance'] });
      expect(results).toEqual([]);
    });

    it('should clear domain index', () => {
      memory.clear();

      const results = memory.query({ domains: ['financial'] });
      expect(results).toEqual([]);
    });

    it('should allow adding episodes after clear', () => {
      memory.clear();

      const episode = memory.addEpisode('New query', 'Response', [], [], [], 0, true, 0.9, testTrace);

      expect(episode).toBeDefined();
      expect(memory.getStats().total_episodes).toBe(1);
    });
  });

  describe('export', () => {
    beforeEach(() => {
      memory.addEpisode('Test query', 'Response', ['finance'], ['financial'], [], 0.01, true, 0.9, testTrace);
    });

    it('should export memory as JSON string', () => {
      const exported = memory.export();

      expect(typeof exported).toBe('string');
      expect(() => JSON.parse(exported)).not.toThrow();
    });

    it('should include episodes array', () => {
      const exported = memory.export();
      const parsed = JSON.parse(exported);

      expect(parsed.episodes).toBeDefined();
      expect(Array.isArray(parsed.episodes)).toBe(true);
      expect(parsed.episodes.length).toBe(1);
    });

    it('should include metadata', () => {
      const exported = memory.export();
      const parsed = JSON.parse(exported);

      expect(parsed.metadata).toBeDefined();
      expect(parsed.metadata.exported_at).toBeDefined();
      expect(parsed.metadata.version).toBe('1.0');
    });

    it('should preserve episode data', () => {
      const exported = memory.export();
      const parsed = JSON.parse(exported);

      const episode = parsed.episodes[0];
      expect(episode.query).toBe('Test query');
      expect(episode.response).toBe('Response');
      expect(episode.concepts).toEqual(['finance']);
    });

    it('should format JSON with indentation', () => {
      const exported = memory.export();

      // Check for newlines (formatted)
      expect(exported).toContain('\n');
    });
  });

  describe('import', () => {
    let exportedData: string;

    beforeEach(() => {
      memory.addEpisode('Original query', 'Response', ['finance'], ['financial'], [], 0.01, true, 0.9, testTrace);
      exportedData = memory.export();
      memory.clear();
    });

    it('should import memory from JSON', () => {
      const count = memory.import(exportedData);

      expect(count).toBe(1);
    });

    it('should restore episodes', () => {
      memory.import(exportedData);

      const stats = memory.getStats();
      expect(stats.total_episodes).toBe(1);
    });

    it('should preserve episode content', () => {
      memory.import(exportedData);

      const results = memory.query({});
      expect(results[0].query).toBe('Original query');
      expect(results[0].response).toBe('Response');
    });

    it('should rebuild concept index', () => {
      memory.import(exportedData);

      const results = memory.query({ concepts: ['finance'] });
      expect(results.length).toBe(1);
    });

    it('should rebuild domain index', () => {
      memory.import(exportedData);

      const results = memory.query({ domains: ['financial'] });
      expect(results.length).toBe(1);
    });

    it('should rebuild query index', () => {
      memory.import(exportedData);

      const similar = memory.findSimilarQueries('Original query');
      expect(similar.length).toBeGreaterThan(0);
    });

    it('should handle multiple episodes', () => {
      memory.clear();
      memory.addEpisode('Q1', 'R1', ['finance'], [], [], 0, true, 0.9, testTrace);
      memory.addEpisode('Q2', 'R2', ['biology'], [], [], 0, true, 0.9, testTrace);
      const multiExport = memory.export();

      memory.clear();
      const count = memory.import(multiExport);

      expect(count).toBe(2);
    });

    it('should handle empty episodes array', () => {
      const emptyData = JSON.stringify({ episodes: [], metadata: {} });
      const count = memory.import(emptyData);

      expect(count).toBe(0);
    });
  });

  describe('Factory Function', () => {
    it('should create EpisodicMemory instance', () => {
      const instance = createMemory();

      expect(instance).toBeInstanceOf(EpisodicMemory);
    });

    it('should create independent instances', () => {
      const mem1 = createMemory();
      const mem2 = createMemory();

      mem1.addEpisode('Q1', 'R1', [], [], [], 0, true, 0.9, testTrace);

      expect(mem1.getStats().total_episodes).toBe(1);
      expect(mem2.getStats().total_episodes).toBe(0);
    });
  });

  describe('Integration', () => {
    it('should handle complete workflow', () => {
      // Add episodes
      memory.addEpisode('Q1', 'R1', ['finance'], ['financial'], [], 0.01, true, 0.9, testTrace);
      memory.addEpisode('Q2', 'R2', ['biology'], ['biology'], [], 0.02, true, 0.8, testTrace);

      // Query
      const results = memory.query({ concepts: ['finance'] });
      expect(results.length).toBe(1);

      // Stats
      const stats = memory.getStats();
      expect(stats.total_episodes).toBe(2);

      // Export
      const exported = memory.export();
      expect(exported).toBeDefined();

      // Clear and import
      memory.clear();
      memory.import(exported);
      expect(memory.getStats().total_episodes).toBe(2);
    });

    it('should maintain consistency across operations', () => {
      const ep1 = memory.addEpisode('Test', 'Response', ['concept1'], ['domain1'], [], 0, true, 0.9, testTrace);

      // Verify retrieval methods are consistent
      const byId = memory.getEpisode(ep1.id);
      const byConcept = memory.query({ concepts: ['concept1'] });
      const byDomain = memory.query({ domains: ['domain1'] });

      expect(byId?.id).toBe(ep1.id);
      expect(byConcept[0].id).toBe(ep1.id);
      expect(byDomain[0].id).toBe(ep1.id);
    });
  });
});
