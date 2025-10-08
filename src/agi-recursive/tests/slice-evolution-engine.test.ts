/**
 * @file slice-evolution-engine.test.ts
 * Tests for SliceEvolutionEngine - Self-evolution orchestrator
 *
 * Key capabilities tested:
 * - Pattern analysis and candidate proposal
 * - Evolution deployment (create/update slices)
 * - Rollback capability
 * - Constitutional validation
 * - Evolution history tracking
 * - Metrics calculation
 * - Backup management
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  SliceEvolutionEngine,
  EvolutionType,
  EvolutionTrigger,
  SliceCandidate,
  SliceEvolution,
  createSliceEvolutionEngine,
} from '../core/slice-evolution-engine';
import { Episode } from '../core/episodic-memory';
import { KnowledgePattern } from '../core/knowledge-distillation';

describe('SliceEvolutionEngine', () => {
  let engine: SliceEvolutionEngine;
  let mockEpisodicMemory: any;
  let mockKnowledgeDistillation: any;
  let mockSliceRewriter: any;
  let mockSliceNavigator: any;
  let mockConstitutionEnforcer: any;
  let mockObservability: any;

  let testPattern: KnowledgePattern;
  let testEpisodes: Episode[];

  beforeEach(() => {
    // Create test data
    testPattern = {
      concepts: ['test', 'pattern'],
      frequency: 5,
      domains: ['test'],
      confidence: 0.9,
      representative_queries: ['test query 1', 'test query 2'],
      emergent_insight: 'Test insight',
    };

    testEpisodes = [
      {
        id: 'ep1',
        query: 'test query',
        response: 'test response',
        concepts: ['test', 'pattern'],
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

    // Mock dependencies
    mockEpisodicMemory = {
      query: vi.fn(() => testEpisodes),
    };

    mockKnowledgeDistillation = {
      discoverPatterns: vi.fn(async () => [testPattern]),
      identifyGaps: vi.fn(async () => []),
      detectErrors: vi.fn(async () => []),
      synthesize: vi.fn(async (pattern: KnowledgePattern) => {
        return `id: ${pattern.concepts.join('-')}\ntitle: Test Slice\nconcepts:\n  - ${pattern.concepts[0]}\n  - ${pattern.concepts[1]}`;
      }),
    };

    mockSliceRewriter = {
      exists: vi.fn(() => false),
      createSlice: vi.fn(async () => {}),
      updateSlice: vi.fn(async () => {}),
      backup: vi.fn(async () => '/backup/test_123.yml'),
      restore: vi.fn(async () => {}),
      deleteSlice: vi.fn(async () => {}),
    };

    mockSliceNavigator = {};

    mockConstitutionEnforcer = {};

    mockObservability = {
      startSpan: vi.fn(() => ({
        setTag: vi.fn(),
        end: vi.fn(),
      })),
      log: vi.fn(),
    };

    engine = new SliceEvolutionEngine(
      mockEpisodicMemory,
      mockKnowledgeDistillation,
      mockSliceRewriter,
      mockSliceNavigator,
      mockConstitutionEnforcer,
      mockObservability
    );
  });

  describe('analyzeAndPropose', () => {
    it('should analyze episodes and propose candidates', async () => {
      const candidates = await engine.analyzeAndPropose(EvolutionTrigger.MANUAL, 3);

      expect(candidates.length).toBeGreaterThan(0);
      expect(mockEpisodicMemory.query).toHaveBeenCalled();
      expect(mockKnowledgeDistillation.discoverPatterns).toHaveBeenCalled();
    });

    it('should only propose candidates for high-confidence patterns', async () => {
      // Pattern with high confidence should be proposed
      const highConfPattern = { ...testPattern, confidence: 0.8 };
      mockKnowledgeDistillation.discoverPatterns.mockResolvedValue([highConfPattern]);

      const candidates = await engine.analyzeAndPropose();

      expect(candidates.length).toBeGreaterThan(0);
      expect(candidates[0].pattern.confidence).toBeGreaterThanOrEqual(0.7);
    });

    it('should filter out low-confidence patterns', async () => {
      const lowConfPattern = { ...testPattern, confidence: 0.5 };
      mockKnowledgeDistillation.discoverPatterns.mockResolvedValue([lowConfPattern]);

      const candidates = await engine.analyzeAndPropose();

      expect(candidates.length).toBe(0);
    });

    it('should return empty array when no episodes exist', async () => {
      mockEpisodicMemory.query.mockReturnValue([]);

      const candidates = await engine.analyzeAndPropose();

      expect(candidates).toEqual([]);
    });

    it('should synthesize content for candidates', async () => {
      const candidates = await engine.analyzeAndPropose();

      expect(mockKnowledgeDistillation.synthesize).toHaveBeenCalled();
      expect(candidates[0].content).toBeDefined();
      expect(candidates[0].content).toContain('id:');
    });

    it('should generate proper slice ID from concepts', async () => {
      const candidates = await engine.analyzeAndPropose();

      expect(candidates[0].id).toContain('test');
      expect(candidates[0].id).toContain('pattern');
    });

    it('should determine candidate type based on slice existence', async () => {
      mockSliceRewriter.exists.mockReturnValue(false);
      const candidates = await engine.analyzeAndPropose();
      expect(candidates[0].type).toBe('new');

      mockSliceRewriter.exists.mockReturnValue(true);
      const candidates2 = await engine.analyzeAndPropose();
      expect(candidates2[0].type).toBe('update');
    });

    it('should calculate constitutional score', async () => {
      const candidates = await engine.analyzeAndPropose();

      expect(candidates[0].constitutional_score).toBeDefined();
      expect(candidates[0].constitutional_score).toBeGreaterThan(0);
      expect(candidates[0].constitutional_score).toBeLessThanOrEqual(1);
    });

    it('should set should_deploy based on scores', async () => {
      const highConfPattern = { ...testPattern, confidence: 0.9 };
      mockKnowledgeDistillation.discoverPatterns.mockResolvedValue([highConfPattern]);

      const candidates = await engine.analyzeAndPropose();

      // Should deploy when confidence >= 0.7 and constitutional >= 0.7
      expect(candidates[0].should_deploy).toBe(true);
    });

    it('should include supporting episodes', async () => {
      const candidates = await engine.analyzeAndPropose();

      expect(candidates[0].supporting_episodes).toBeDefined();
      expect(Array.isArray(candidates[0].supporting_episodes)).toBe(true);
    });

    it('should log analysis results', async () => {
      await engine.analyzeAndPropose();

      expect(mockObservability.log).toHaveBeenCalledWith(
        'info',
        expect.stringContaining('candidates'),
        expect.any(Object)
      );
    });
  });

  describe('deployEvolution', () => {
    let testCandidate: SliceCandidate;

    beforeEach(() => {
      testCandidate = {
        id: 'test-slice',
        type: 'new',
        title: 'Test Slice',
        description: 'Test description',
        concepts: ['test'],
        content: 'id: test-slice\ntitle: Test',
        supporting_episodes: ['ep1'],
        pattern: testPattern,
        constitutional_score: 0.9,
        test_performance: {
          queries_tested: 10,
          accuracy_improvement: 0.15,
          cost_delta: -0.05,
        },
        should_deploy: true,
        reasoning: 'Test reasoning',
      };
    });

    it('should deploy new slice candidate', async () => {
      const evolution = await engine.deployEvolution(testCandidate);

      expect(mockSliceRewriter.createSlice).toHaveBeenCalledWith(
        'test-slice',
        testCandidate.content
      );
      expect(evolution.approved).toBe(true);
      expect(evolution.deployed_at).toBeDefined();
    });

    it('should deploy update slice candidate with backup', async () => {
      testCandidate.type = 'update';
      mockSliceRewriter.exists.mockReturnValue(true);

      const evolution = await engine.deployEvolution(testCandidate);

      expect(mockSliceRewriter.backup).toHaveBeenCalledWith('test-slice');
      expect(mockSliceRewriter.updateSlice).toHaveBeenCalled();
      expect(evolution.backup_path).toBe('/backup/test_123.yml');
    });

    it('should throw error if should_deploy is false', async () => {
      testCandidate.should_deploy = false;

      await expect(engine.deployEvolution(testCandidate)).rejects.toThrow('should_deploy=false');
    });

    it('should create evolution record', async () => {
      const evolution = await engine.deployEvolution(testCandidate);

      expect(evolution.id).toBeDefined();
      expect(evolution.evolution_type).toBe(EvolutionType.CREATED);
      expect(evolution.slice_id).toBe('test-slice');
      expect(evolution.candidate).toBe(testCandidate);
    });

    it('should track performance delta', async () => {
      const evolution = await engine.deployEvolution(testCandidate);

      expect(evolution.performance_delta).toBeDefined();
      expect(evolution.performance_delta?.accuracy_change).toBe(0.15);
      expect(evolution.performance_delta?.cost_change).toBe(-0.05);
    });

    it('should log deployment', async () => {
      await engine.deployEvolution(testCandidate);

      expect(mockObservability.log).toHaveBeenCalledWith(
        'info',
        'evolution_deployed',
        expect.any(Object)
      );
    });

    it('should use correct evolution type for updates', async () => {
      testCandidate.type = 'update';
      mockSliceRewriter.exists.mockReturnValue(true);

      const evolution = await engine.deployEvolution(testCandidate);

      expect(evolution.evolution_type).toBe(EvolutionType.UPDATED);
    });

    it('should store evolution in history', async () => {
      await engine.deployEvolution(testCandidate);

      const history = engine.getEvolutionHistory();
      expect(history.length).toBe(1);
      expect(history[0].slice_id).toBe('test-slice');
    });
  });

  describe('rollback', () => {
    let evolution: SliceEvolution;

    beforeEach(async () => {
      const candidate: SliceCandidate = {
        id: 'test-slice',
        type: 'update',
        title: 'Test',
        description: 'Test',
        concepts: ['test'],
        content: 'test content',
        supporting_episodes: [],
        pattern: testPattern,
        constitutional_score: 0.9,
        test_performance: {
          queries_tested: 0,
          accuracy_improvement: 0,
          cost_delta: 0,
        },
        should_deploy: true,
        reasoning: 'test',
      };

      mockSliceRewriter.exists.mockReturnValue(true);
      evolution = await engine.deployEvolution(candidate);
    });

    it('should rollback evolution with backup', async () => {
      await engine.rollback(evolution.id);

      expect(mockSliceRewriter.restore).toHaveBeenCalledWith('/backup/test_123.yml');
    });

    it('should mark evolution as rolled back', async () => {
      await engine.rollback(evolution.id);

      const history = engine.getEvolutionHistory();
      expect(history[0].rolled_back).toBe(true);
      expect(history[0].rollback_reason).toBeDefined();
    });

    it('should throw error for non-existent evolution', async () => {
      await expect(engine.rollback('nonexistent')).rejects.toThrow('not found');
    });

    it('should throw error if already rolled back', async () => {
      await engine.rollback(evolution.id);

      await expect(engine.rollback(evolution.id)).rejects.toThrow('already rolled back');
    });

    it('should delete slice for CREATED evolution without backup', async () => {
      const newCandidate: SliceCandidate = {
        id: 'new-slice',
        type: 'new',
        title: 'New',
        description: 'New',
        concepts: ['new'],
        content: 'new content',
        supporting_episodes: [],
        pattern: testPattern,
        constitutional_score: 0.9,
        test_performance: {
          queries_tested: 0,
          accuracy_improvement: 0,
          cost_delta: 0,
        },
        should_deploy: true,
        reasoning: 'test',
      };

      mockSliceRewriter.exists.mockReturnValue(false);
      const newEvolution = await engine.deployEvolution(newCandidate);

      // Clear backup_path to test deletion path
      newEvolution.backup_path = undefined;

      await engine.rollback(newEvolution.id);

      expect(mockSliceRewriter.deleteSlice).toHaveBeenCalledWith('new-slice');
    });

    it('should log rollback', async () => {
      await engine.rollback(evolution.id);

      expect(mockObservability.log).toHaveBeenCalledWith(
        'info',
        'evolution_rolled_back',
        expect.any(Object)
      );
    });
  });

  describe('getEvolutionHistory', () => {
    it('should return empty array initially', () => {
      const history = engine.getEvolutionHistory();

      expect(history).toEqual([]);
    });

    it('should return copy of evolutions', async () => {
      const candidate: SliceCandidate = {
        id: 'test',
        type: 'new',
        title: 'Test',
        description: 'Test',
        concepts: ['test'],
        content: 'content',
        supporting_episodes: [],
        pattern: testPattern,
        constitutional_score: 0.9,
        test_performance: {
          queries_tested: 0,
          accuracy_improvement: 0,
          cost_delta: 0,
        },
        should_deploy: true,
        reasoning: 'test',
      };

      await engine.deployEvolution(candidate);

      const history1 = engine.getEvolutionHistory();
      const history2 = engine.getEvolutionHistory();

      expect(history1).not.toBe(history2); // Different arrays
      expect(history1).toEqual(history2); // Same content
    });

    it('should include all deployed evolutions', async () => {
      const candidate1: SliceCandidate = {
        id: 'slice1',
        type: 'new',
        title: 'Slice 1',
        description: 'Test',
        concepts: ['test'],
        content: 'content',
        supporting_episodes: [],
        pattern: testPattern,
        constitutional_score: 0.9,
        test_performance: {
          queries_tested: 0,
          accuracy_improvement: 0,
          cost_delta: 0,
        },
        should_deploy: true,
        reasoning: 'test',
      };

      const candidate2: SliceCandidate = {
        ...candidate1,
        id: 'slice2',
        title: 'Slice 2',
      };

      await engine.deployEvolution(candidate1);
      await engine.deployEvolution(candidate2);

      const history = engine.getEvolutionHistory();
      expect(history.length).toBe(2);
    });
  });

  describe('getMetrics', () => {
    beforeEach(async () => {
      // Deploy multiple evolutions with different types and triggers
      const candidate1: SliceCandidate = {
        id: 'slice1',
        type: 'new',
        title: 'Slice 1',
        description: 'Test',
        concepts: ['test'],
        content: 'content',
        supporting_episodes: [],
        pattern: testPattern,
        constitutional_score: 0.8,
        test_performance: {
          queries_tested: 0,
          accuracy_improvement: 0,
          cost_delta: 0,
        },
        should_deploy: true,
        reasoning: 'test',
      };

      const candidate2: SliceCandidate = {
        ...candidate1,
        id: 'slice2',
        type: 'update',
        constitutional_score: 0.9,
      };

      mockSliceRewriter.exists.mockReturnValue(false);
      await engine.deployEvolution(candidate1, EvolutionTrigger.MANUAL);

      mockSliceRewriter.exists.mockReturnValue(true);
      await engine.deployEvolution(candidate2, EvolutionTrigger.THRESHOLD);
    });

    it('should count total evolutions', () => {
      const metrics = engine.getMetrics();

      expect(metrics.total_evolutions).toBe(2);
    });

    it('should count successful deployments', () => {
      const metrics = engine.getMetrics();

      expect(metrics.successful_deployments).toBe(2);
    });

    it('should count evolutions by type', () => {
      const metrics = engine.getMetrics();

      expect(metrics.by_type[EvolutionType.CREATED]).toBe(1);
      expect(metrics.by_type[EvolutionType.UPDATED]).toBe(1);
    });

    it('should count evolutions by trigger', () => {
      const metrics = engine.getMetrics();

      expect(metrics.by_trigger[EvolutionTrigger.MANUAL]).toBe(1);
      expect(metrics.by_trigger[EvolutionTrigger.THRESHOLD]).toBe(1);
    });

    it('should calculate average constitutional score', () => {
      const metrics = engine.getMetrics();

      expect(metrics.avg_constitutional_score).toBe(0.85); // (0.8 + 0.9) / 2
    });

    it('should track knowledge growth', () => {
      const metrics = engine.getMetrics();

      expect(metrics.knowledge_growth.slices_created).toBe(1);
      expect(metrics.knowledge_growth.slices_updated).toBe(1);
    });

    it('should count rollbacks', async () => {
      const history = engine.getEvolutionHistory();
      await engine.rollback(history[0].id);

      const metrics = engine.getMetrics();
      expect(metrics.rollbacks).toBe(1);
    });

    it('should handle empty metrics', () => {
      const emptyEngine = new SliceEvolutionEngine(
        mockEpisodicMemory,
        mockKnowledgeDistillation,
        mockSliceRewriter,
        mockSliceNavigator,
        mockConstitutionEnforcer,
        mockObservability
      );

      const metrics = emptyEngine.getMetrics();

      expect(metrics.total_evolutions).toBe(0);
      expect(metrics.avg_constitutional_score).toBe(0);
    });
  });

  describe('Observability Integration', () => {
    it('should create span for analysis', async () => {
      await engine.analyzeAndPropose();

      expect(mockObservability.startSpan).toHaveBeenCalledWith('analyze_and_propose');
    });

    it('should create span for deployment', async () => {
      const candidate: SliceCandidate = {
        id: 'test',
        type: 'new',
        title: 'Test',
        description: 'Test',
        concepts: ['test'],
        content: 'content',
        supporting_episodes: [],
        pattern: testPattern,
        constitutional_score: 0.9,
        test_performance: {
          queries_tested: 0,
          accuracy_improvement: 0,
          cost_delta: 0,
        },
        should_deploy: true,
        reasoning: 'test',
      };

      await engine.deployEvolution(candidate);

      expect(mockObservability.startSpan).toHaveBeenCalledWith('deploy_evolution');
    });

    it('should create span for rollback', async () => {
      const candidate: SliceCandidate = {
        id: 'test',
        type: 'new',
        title: 'Test',
        description: 'Test',
        concepts: ['test'],
        content: 'content',
        supporting_episodes: [],
        pattern: testPattern,
        constitutional_score: 0.9,
        test_performance: {
          queries_tested: 0,
          accuracy_improvement: 0,
          cost_delta: 0,
        },
        should_deploy: true,
        reasoning: 'test',
      };

      const evolution = await engine.deployEvolution(candidate);
      await engine.rollback(evolution.id);

      expect(mockObservability.startSpan).toHaveBeenCalledWith('rollback_evolution');
    });
  });

  describe('Factory Function', () => {
    it('should create SliceEvolutionEngine instance', () => {
      const instance = createSliceEvolutionEngine(
        mockEpisodicMemory,
        mockKnowledgeDistillation,
        mockSliceRewriter,
        mockSliceNavigator,
        mockConstitutionEnforcer,
        mockObservability
      );

      expect(instance).toBeInstanceOf(SliceEvolutionEngine);
    });
  });
});
