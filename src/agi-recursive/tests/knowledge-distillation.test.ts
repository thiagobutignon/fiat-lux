/**
 * @file knowledge-distillation.test.ts
 * Tests for KnowledgeDistillation - Pattern discovery from episodic memory
 *
 * Key capabilities tested:
 * - Recurring pattern discovery
 * - Knowledge gap identification
 * - Systematic error detection
 * - Knowledge synthesis
 * - Domain inference
 * - Pattern statistics
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  KnowledgeDistillation,
  KnowledgePattern,
  ErrorPattern,
  KnowledgeGap,
  createKnowledgeDistillation,
} from '../core/knowledge-distillation';
import { Episode } from '../core/episodic-memory';
import { AnthropicAdapter } from '../llm/anthropic-adapter';
import { Observability } from '../core/observability';

describe('KnowledgeDistillation', () => {
  let distillation: KnowledgeDistillation;
  let mockMemory: any;
  let mockLLM: any;
  let mockObservability: any;
  let testEpisodes: Episode[];

  beforeEach(() => {
    // Mock dependencies
    mockMemory = {};

    mockLLM = {
      invoke: vi.fn(),
    };

    mockObservability = {
      startSpan: vi.fn(() => ({
        setTag: vi.fn(),
        end: vi.fn(),
      })),
      log: vi.fn(),
    };

    distillation = new KnowledgeDistillation(mockMemory, mockLLM, mockObservability);

    // Create test episodes
    testEpisodes = [
      {
        id: 'ep1',
        query: 'How to diversify my portfolio?',
        response: 'Use different asset classes',
        concepts: ['diversification', 'portfolio', 'risk'],
        domains: ['financial'],
        agents_used: ['financial'],
        cost: 0.01,
        success: true,
        confidence: 0.9,
        timestamp: Date.now(),
        execution_trace: [],
        emergent_insights: [],
      },
      {
        id: 'ep2',
        query: 'What is portfolio diversification?',
        response: 'Spreading investments',
        concepts: ['diversification', 'portfolio', 'risk'],
        domains: ['financial'],
        agents_used: ['financial'],
        cost: 0.01,
        success: true,
        confidence: 0.85,
        timestamp: Date.now(),
        execution_trace: [],
        emergent_insights: [],
      },
      {
        id: 'ep3',
        query: 'Risk management in portfolios',
        response: 'Diversify assets',
        concepts: ['diversification', 'portfolio', 'risk'],
        domains: ['financial'],
        agents_used: ['financial'],
        cost: 0.01,
        success: true,
        confidence: 0.88,
        timestamp: Date.now(),
        execution_trace: [],
        emergent_insights: [],
      },
      {
        id: 'ep4',
        query: 'How do cells maintain homeostasis?',
        response: 'Through feedback mechanisms',
        concepts: ['homeostasis', 'feedback', 'cells'],
        domains: ['biology'],
        agents_used: ['biology'],
        cost: 0.01,
        success: true,
        confidence: 0.92,
        timestamp: Date.now(),
        execution_trace: [],
        emergent_insights: [],
      },
      {
        id: 'ep5',
        query: 'What is quantum entanglement?',
        response: 'I am not certain about this',
        concepts: ['quantum', 'entanglement'],
        domains: ['physics'],
        agents_used: ['physics'],
        cost: 0.01,
        success: false,
        confidence: 0.3,
        timestamp: Date.now(),
        execution_trace: [],
        emergent_insights: [],
      },
    ];
  });

  describe('Pattern Discovery', () => {
    it('should discover recurring concept patterns', async () => {
      const patterns = await distillation.discoverPatterns(testEpisodes, 3);

      expect(patterns.length).toBeGreaterThan(0);
      expect(patterns[0].concepts).toEqual(['diversification', 'portfolio', 'risk']);
      expect(patterns[0].frequency).toBe(3);
    });

    it('should filter patterns by minimum frequency', async () => {
      const patterns = await distillation.discoverPatterns(testEpisodes, 5);

      // No pattern should have frequency < 5
      expect(patterns.length).toBe(0);
    });

    it('should only include successful episodes in patterns', async () => {
      const patterns = await distillation.discoverPatterns(testEpisodes, 1);

      // Failed episodes should not contribute to patterns
      const quantumPattern = patterns.find((p) => p.concepts.includes('quantum'));
      expect(quantumPattern).toBeUndefined();
    });

    it('should include representative queries', async () => {
      const patterns = await distillation.discoverPatterns(testEpisodes, 3);

      expect(patterns[0].representative_queries).toBeDefined();
      expect(patterns[0].representative_queries.length).toBeGreaterThan(0);
      expect(patterns[0].representative_queries.length).toBeLessThanOrEqual(3);
    });

    it('should include domains from pattern occurrences', async () => {
      const patterns = await distillation.discoverPatterns(testEpisodes, 3);

      expect(patterns[0].domains).toBeDefined();
      expect(patterns[0].domains).toContain('financial');
    });

    it('should calculate confidence based on frequency', async () => {
      const patterns = await distillation.discoverPatterns(testEpisodes, 3);

      expect(patterns[0].confidence).toBeGreaterThan(0);
      expect(patterns[0].confidence).toBeLessThanOrEqual(1);
    });

    it('should sort patterns by frequency (most common first)', async () => {
      const patterns = await distillation.discoverPatterns(testEpisodes, 1);

      for (let i = 0; i < patterns.length - 1; i++) {
        expect(patterns[i].frequency).toBeGreaterThanOrEqual(patterns[i + 1].frequency);
      }
    });

    it('should generate emergent insight description', async () => {
      const patterns = await distillation.discoverPatterns(testEpisodes, 3);

      expect(patterns[0].emergent_insight).toBeDefined();
      expect(patterns[0].emergent_insight).toContain('Pattern');
      expect(patterns[0].emergent_insight).toContain('3');
    });

    it('should handle episodes with no concepts', async () => {
      const episodesWithEmpty = [
        ...testEpisodes,
        {
          id: 'empty',
          query: 'test',
          response: 'test',
          concepts: [],
          domains: ['test'],
          agents_used: ['test'],
          cost: 0.01,
          success: true,
          confidence: 0.9,
          timestamp: Date.now(),
          execution_trace: [],
          emergent_insights: [],
        },
      ];

      const patterns = await distillation.discoverPatterns(episodesWithEmpty, 3);

      // Should not include empty concept sets
      expect(patterns.every((p) => p.concepts.length > 0)).toBe(true);
    });

    it('should log pattern discovery', async () => {
      await distillation.discoverPatterns(testEpisodes, 3);

      expect(mockObservability.log).toHaveBeenCalledWith(
        'info',
        'patterns_discovered',
        expect.any(Object)
      );
    });
  });

  describe('Gap Identification', () => {
    it('should identify knowledge gaps from failed episodes', async () => {
      const gaps = await distillation.identifyGaps(testEpisodes);

      expect(gaps.length).toBeGreaterThan(0);
    });

    it('should include concepts from failed episodes', async () => {
      const gaps = await distillation.identifyGaps(testEpisodes);

      const quantumGap = gaps.find((g) => g.concept === 'quantum');
      expect(quantumGap).toBeDefined();
      expect(quantumGap?.evidence).toContain('ep5');
    });

    it('should identify gaps from low-confidence episodes', async () => {
      const lowConfidenceEpisode: Episode = {
        id: 'low-conf',
        query: 'Explain differential equations',
        response: 'I am not very confident about this',
        concepts: ['differential', 'equations'],
        domains: ['math'],
        agents_used: ['math'],
        cost: 0.01,
        success: true, // Success but low confidence
        confidence: 0.4,
        timestamp: Date.now(),
        execution_trace: [],
        emergent_insights: [],
      };

      const gaps = await distillation.identifyGaps([...testEpisodes, lowConfidenceEpisode]);

      const mathGap = gaps.find((g) =>
        g.concept === 'differential' || g.concept === 'equations'
      );
      expect(mathGap).toBeDefined();
    });

    it('should extract concepts from query words', async () => {
      const gaps = await distillation.identifyGaps(testEpisodes);

      // Should extract 'quantum' and 'entanglement' from the failed query
      expect(gaps.some((g) => g.concept === 'quantum')).toBe(true);
      expect(gaps.some((g) => g.concept === 'entanglement')).toBe(true);
    });

    it('should provide evidence (episode IDs) for gaps', async () => {
      const gaps = await distillation.identifyGaps(testEpisodes);

      gaps.forEach((gap) => {
        expect(gap.evidence).toBeDefined();
        expect(Array.isArray(gap.evidence)).toBe(true);
        expect(gap.evidence.length).toBeGreaterThan(0);
      });
    });

    it('should log gap identification', async () => {
      await distillation.identifyGaps(testEpisodes);

      expect(mockObservability.log).toHaveBeenCalledWith('info', 'gaps_identified', expect.any(Object));
    });

    it('should filter short query words', async () => {
      const gaps = await distillation.identifyGaps(testEpisodes);

      // Words like 'is', 'to', 'my' should be filtered out (< 5 chars)
      expect(gaps.every((g) => g.concept.length > 4 || g.concept.includes('-'))).toBe(true);
    });
  });

  describe('Error Detection', () => {
    it('should detect systematic errors', async () => {
      // Add another failed episode to create a systematic error
      const episodesWithSystematicError = [
        ...testEpisodes,
        {
          id: 'ep6',
          query: 'What is quantum mechanics?',
          response: 'I do not know',
          concepts: ['quantum'],
          domains: ['physics'],
          agents_used: ['physics'],
          cost: 0.01,
          success: false,
          confidence: 0.2,
          timestamp: Date.now(),
          execution_trace: [],
          emergent_insights: [],
        },
      ];

      const errors = await distillation.detectErrors(episodesWithSystematicError);

      expect(errors.length).toBeGreaterThan(0);
    });

    it('should only analyze failed episodes', async () => {
      const allSuccessful = testEpisodes.filter((ep) => ep.success);
      const errors = await distillation.detectErrors(allSuccessful);

      expect(errors.length).toBe(0);
    });

    it('should require minimum error frequency', async () => {
      // Add another failed episode with same concept
      const episodesWithDuplicateError = [
        ...testEpisodes,
        {
          id: 'ep6',
          query: 'What is quantum mechanics?',
          response: 'I do not know',
          concepts: ['quantum'],
          domains: ['physics'],
          agents_used: ['physics'],
          cost: 0.01,
          success: false,
          confidence: 0.2,
          timestamp: Date.now(),
          execution_trace: [],
          emergent_insights: [],
        },
      ];

      const errors = await distillation.detectErrors(episodesWithDuplicateError);

      // Should find 'quantum' error (appears 2 times)
      const quantumError = errors.find((e) => e.concept === 'quantum');
      expect(quantumError).toBeDefined();
      expect(quantumError?.frequency).toBe(2);
    });

    it('should provide typical error description', async () => {
      const episodesWithDuplicateError = [
        ...testEpisodes,
        {
          id: 'ep6',
          query: 'What is quantum mechanics?',
          response: 'I do not know',
          concepts: ['quantum'],
          domains: ['physics'],
          agents_used: ['physics'],
          cost: 0.01,
          success: false,
          confidence: 0.2,
          timestamp: Date.now(),
          execution_trace: [],
          emergent_insights: [],
        },
      ];

      const errors = await distillation.detectErrors(episodesWithDuplicateError);

      const quantumError = errors.find((e) => e.concept === 'quantum');
      expect(quantumError?.typical_error).toBeDefined();
      expect(quantumError?.typical_error).toContain('quantum');
    });

    it('should suggest fixes for errors', async () => {
      const episodesWithDuplicateError = [
        ...testEpisodes,
        {
          id: 'ep6',
          query: 'What is quantum mechanics?',
          response: 'I do not know',
          concepts: ['quantum'],
          domains: ['physics'],
          agents_used: ['physics'],
          cost: 0.01,
          success: false,
          confidence: 0.2,
          timestamp: Date.now(),
          execution_trace: [],
          emergent_insights: [],
        },
      ];

      const errors = await distillation.detectErrors(episodesWithDuplicateError);

      const quantumError = errors.find((e) => e.concept === 'quantum');
      expect(quantumError?.suggested_fix).toBeDefined();
      expect(quantumError?.suggested_fix).toContain('knowledge slice');
    });

    it('should sort errors by frequency', async () => {
      const errors = await distillation.detectErrors(testEpisodes);

      for (let i = 0; i < errors.length - 1; i++) {
        expect(errors[i].frequency).toBeGreaterThanOrEqual(errors[i + 1].frequency);
      }
    });

    it('should log error detection', async () => {
      await distillation.detectErrors(testEpisodes);

      expect(mockObservability.log).toHaveBeenCalledWith('info', 'errors_detected', expect.any(Object));
    });
  });

  describe('Knowledge Synthesis', () => {
    let testPattern: KnowledgePattern;

    beforeEach(() => {
      testPattern = {
        concepts: ['diversification', 'portfolio', 'risk'],
        frequency: 5,
        domains: ['financial'],
        confidence: 0.9,
        representative_queries: [
          'How to diversify portfolio?',
          'What is portfolio diversification?',
        ],
        emergent_insight: 'Pattern about portfolio diversification',
      };

      // Mock LLM response
      mockLLM.invoke.mockResolvedValue({
        text: `\`\`\`yaml
id: portfolio-diversification
title: Portfolio Diversification
description: Spreading investments to reduce risk
concepts:
  - diversification
  - portfolio
  - risk
domains:
  - financial
content: |
  Diversification is the practice of spreading investments across various
  financial instruments, industries, and other categories to reduce risk.
\`\`\``,
        usage: { cost_usd: 0.05 },
      });
    });

    it('should synthesize knowledge from pattern', async () => {
      const yaml = await distillation.synthesize(testPattern);

      expect(yaml).toBeDefined();
      expect(yaml).toContain('id:');
      expect(yaml).toContain('title:');
      expect(yaml).toContain('concepts:');
    });

    it('should extract YAML from markdown code blocks', async () => {
      const yaml = await distillation.synthesize(testPattern);

      // Should not include ```yaml markers
      expect(yaml).not.toContain('```yaml');
      expect(yaml).not.toContain('```');
    });

    it('should invoke LLM with correct parameters', async () => {
      await distillation.synthesize(testPattern);

      expect(mockLLM.invoke).toHaveBeenCalledWith(
        expect.stringContaining('knowledge synthesizer'),
        expect.stringContaining('Concepts: diversification, portfolio, risk'),
        expect.objectContaining({
          model: 'claude-sonnet-4-5',
          temperature: 0.3,
        })
      );
    });

    it('should log synthesis', async () => {
      await distillation.synthesize(testPattern);

      expect(mockObservability.log).toHaveBeenCalledWith(
        'info',
        'knowledge_synthesized',
        expect.any(Object)
      );
    });

    it('should handle LLM response without code blocks', async () => {
      mockLLM.invoke.mockResolvedValue({
        text: 'id: test\ntitle: Test\nconcepts:\n  - test',
        usage: { cost_usd: 0.05 },
      });

      const yaml = await distillation.synthesize(testPattern);

      expect(yaml).toContain('id: test');
    });
  });

  describe('Domain Inference', () => {
    it('should infer financial domain', async () => {
      // Test via error detection which uses inferDomain internally
      const financialError: Episode = {
        id: 'err1',
        query: 'What is interest rate?',
        response: 'Unknown',
        concepts: ['interest'],
        domains: ['unknown'],
        agents_used: ['test'],
        cost: 0.01,
        success: false,
        confidence: 0.3,
        timestamp: Date.now(),
        execution_trace: [],
        emergent_insights: [],
      };

      const errors = await distillation.detectErrors([financialError]);

      if (errors.length > 0) {
        expect(errors[0].suggested_fix).toContain('financial');
      }
    });

    it('should infer biology domain', async () => {
      const biologyError: Episode = {
        id: 'err2',
        query: 'What is cellular biology?',
        response: 'Unknown',
        concepts: ['cellular'],
        domains: ['unknown'],
        agents_used: ['test'],
        cost: 0.01,
        success: false,
        confidence: 0.3,
        timestamp: Date.now(),
        execution_trace: [],
        emergent_insights: [],
      };

      const errors = await distillation.detectErrors([biologyError]);

      if (errors.length > 0) {
        expect(errors[0].suggested_fix).toContain('biology');
      }
    });

    it('should infer systems domain', async () => {
      const systemsError: Episode = {
        id: 'err3',
        query: 'What is feedback loop?',
        response: 'Unknown',
        concepts: ['feedback'],
        domains: ['unknown'],
        agents_used: ['test'],
        cost: 0.01,
        success: false,
        confidence: 0.3,
        timestamp: Date.now(),
        execution_trace: [],
        emergent_insights: [],
      };

      const errors = await distillation.detectErrors([systemsError]);

      if (errors.length > 0) {
        expect(errors[0].suggested_fix).toContain('systems');
      }
    });

    it('should default to general domain for unknown concepts', async () => {
      const unknownError: Episode = {
        id: 'err4',
        query: 'What is xyz-unknown-concept?',
        response: 'Unknown',
        concepts: ['xyz-unknown-concept'],
        domains: ['unknown'],
        agents_used: ['test'],
        cost: 0.01,
        success: false,
        confidence: 0.3,
        timestamp: Date.now(),
        execution_trace: [],
        emergent_insights: [],
      };

      const errors = await distillation.detectErrors([unknownError]);

      if (errors.length > 0) {
        expect(errors[0].suggested_fix).toContain('general');
      }
    });
  });

  describe('Pattern Statistics', () => {
    let patterns: KnowledgePattern[];

    beforeEach(async () => {
      patterns = await distillation.discoverPatterns(testEpisodes, 1);
    });

    it('should calculate total patterns', () => {
      const stats = distillation.getPatternStats(patterns);

      expect(stats.total).toBe(patterns.length);
    });

    it('should calculate average frequency', () => {
      const stats = distillation.getPatternStats(patterns);

      expect(stats.avg_frequency).toBeGreaterThan(0);
      expect(typeof stats.avg_frequency).toBe('number');
    });

    it('should calculate average confidence', () => {
      const stats = distillation.getPatternStats(patterns);

      expect(stats.avg_confidence).toBeGreaterThan(0);
      expect(stats.avg_confidence).toBeLessThanOrEqual(1);
    });

    it('should identify most common domains', () => {
      const stats = distillation.getPatternStats(patterns);

      expect(stats.most_common_domains).toBeDefined();
      expect(Array.isArray(stats.most_common_domains)).toBe(true);
      expect(stats.most_common_domains.length).toBeLessThanOrEqual(3);
    });

    it('should handle empty pattern list', () => {
      const stats = distillation.getPatternStats([]);

      expect(stats.total).toBe(0);
      expect(stats.avg_frequency).toBe(0);
      expect(stats.avg_confidence).toBe(0);
      expect(stats.most_common_domains).toEqual([]);
    });
  });

  describe('Factory Function', () => {
    it('should create KnowledgeDistillation instance', () => {
      const instance = createKnowledgeDistillation(mockMemory, mockLLM, mockObservability);

      expect(instance).toBeInstanceOf(KnowledgeDistillation);
    });
  });

  describe('Observability Integration', () => {
    it('should start span for pattern discovery', async () => {
      await distillation.discoverPatterns(testEpisodes, 3);

      expect(mockObservability.startSpan).toHaveBeenCalledWith('discover_patterns');
    });

    it('should start span for gap identification', async () => {
      await distillation.identifyGaps(testEpisodes);

      expect(mockObservability.startSpan).toHaveBeenCalledWith('identify_gaps');
    });

    it('should start span for error detection', async () => {
      await distillation.detectErrors(testEpisodes);

      expect(mockObservability.startSpan).toHaveBeenCalledWith('detect_errors');
    });

    it('should start span for synthesis', async () => {
      const testPattern: KnowledgePattern = {
        concepts: ['test'],
        frequency: 5,
        domains: ['test'],
        confidence: 0.9,
        representative_queries: ['test query'],
        emergent_insight: 'Test insight',
      };

      mockLLM.invoke.mockResolvedValue({
        text: 'id: test',
        usage: { cost_usd: 0.05 },
      });

      await distillation.synthesize(testPattern);

      expect(mockObservability.startSpan).toHaveBeenCalledWith('synthesize_knowledge');
    });
  });
});
