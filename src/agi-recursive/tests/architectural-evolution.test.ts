/**
 * Tests for Architectural Evolution System
 *
 * Validates meta-emergence of second order: system redesigning itself
 * based on discovered principles.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  ArchitecturalEvolution,
  createArchitecturalEvolution,
  type DiscoveredPrinciple,
  type ArchitecturalProposal,
} from '../core/architectural-evolution';
import { EpisodicMemory } from '../core/episodic-memory';
import { Constitution } from '../core/constitution';

describe('ArchitecturalEvolution', () => {
  let evolution: ArchitecturalEvolution;
  let memory: EpisodicMemory;
  let constitution: Constitution;

  beforeEach(() => {
    memory = new EpisodicMemory();
    constitution = new Constitution();
    evolution = createArchitecturalEvolution(memory, constitution);
  });

  // =========================================================================
  // Principle Discovery
  // =========================================================================

  describe('Principle Discovery', () => {
    it('should discover new principle from evidence', () => {
      const principle = evolution.discoverPrinciple(
        'Composition Is All You Need',
        'Intelligence emerges from composition of simple agents, not from model size',
        [
          'AGI system with 4 agents outperforms GPT-4 in specific domains',
          '80% cost reduction through composition',
          'Novel insights impossible for individual agents',
          'Empirical validation across 48 production queries',
        ],
        'outcomes'
      );

      expect(principle.name).toBe('Composition Is All You Need');
      expect(principle.confidence).toBeGreaterThan(0.7);
      expect(principle.evidence).toHaveLength(4);
      expect(principle.source).toBe('outcomes');
    });

    it('should calculate higher confidence with more evidence', () => {
      const principle1 = evolution.discoverPrinciple(
        'Test 1',
        'Statement 1',
        ['Evidence 1', 'Evidence 2'],
        'architecture'
      );

      const principle2 = evolution.discoverPrinciple(
        'Test 2',
        'Statement 2',
        ['Evidence 1', 'Evidence 2', 'Evidence 3', 'Evidence 4', 'Evidence 5'],
        'architecture'
      );

      expect(principle2.confidence).toBeGreaterThan(principle1.confidence);
    });

    it('should give bonus for empirical evidence', () => {
      const qualitative = evolution.discoverPrinciple(
        'Qualitative',
        'Statement',
        ['It works well', 'Users like it'],
        'behavior'
      );

      const quantitative = evolution.discoverPrinciple(
        'Quantitative',
        'Statement',
        ['97.3% success rate', '80% cost reduction'],
        'behavior'
      );

      expect(quantitative.confidence).toBeGreaterThan(qualitative.confidence);
    });

    it('should have 3 pre-initialized principles', () => {
      const principles = evolution.getPrinciples();
      expect(principles).toHaveLength(3);

      const names = principles.map((p) => p.name);
      expect(names).toContain("You Don't Know Is All You Need");
      expect(names).toContain('Idleness Is All You Need');
      expect(names).toContain('Continuous Evolution Is All You Need');
    });
  });

  // =========================================================================
  // Implication Analysis
  // =========================================================================

  describe('Implication Analysis', () => {
    it('should derive architectural implications from principles', () => {
      const implications_map = evolution.analyzeImplications();

      expect(implications_map.size).toBeGreaterThan(0);

      const epistemic_implications = implications_map.get('epistemic-honesty');
      expect(epistemic_implications).toBeDefined();
      expect(epistemic_implications!.length).toBeGreaterThan(0);
    });

    it('should generate Uncertainty Propagation Layer from epistemic honesty', () => {
      const implications_map = evolution.analyzeImplications();
      const epistemic = implications_map.get('epistemic-honesty')!;

      const uncertainty_layer = epistemic.find((i) => i.pattern_name.includes('Uncertainty'));
      expect(uncertainty_layer).toBeDefined();
      expect(uncertainty_layer!.description).toContain('uncertainty tracking');
      expect(uncertainty_layer!.rationale).toContain('first-class citizen');
    });

    it('should generate Cache-First Architecture from lazy efficiency', () => {
      const implications_map = evolution.analyzeImplications();
      const lazy = implications_map.get('lazy-efficiency')!;

      const cache_first = lazy.find((i) => i.pattern_name.includes('Cache'));
      expect(cache_first).toBeDefined();
      expect(cache_first!.description).toContain('cache before computation');
    });

    it('should generate Self-Modifying Architecture from continuous evolution', () => {
      const implications_map = evolution.analyzeImplications();
      const evolution_impl = implications_map.get('continuous-evolution')!;

      const self_modifying = evolution_impl.find((i) => i.pattern_name.includes('Self-Modifying'));
      expect(self_modifying).toBeDefined();
      expect(self_modifying!.implementation_complexity).toBe('high');
    });

    it('should include examples for each implication', () => {
      const implications_map = evolution.analyzeImplications();

      for (const implications of implications_map.values()) {
        for (const impl of implications) {
          expect(impl.examples.length).toBeGreaterThan(0);
        }
      }
    });

    it('should filter out low-confidence principles', () => {
      // Add low-confidence principle
      evolution.discoverPrinciple('Low Confidence Test', 'Statement', ['Only one piece'], 'behavior');

      const implications_map = evolution.analyzeImplications();

      // Should not include implications for low-confidence principles
      expect(implications_map.has('low-confidence-test')).toBe(false);
    });
  });

  // =========================================================================
  // Proposal Generation
  // =========================================================================

  describe('Proposal Generation', () => {
    it('should generate proposal from implications', () => {
      const implications_map = evolution.analyzeImplications();
      const implications = implications_map.get('epistemic-honesty')!;

      const proposal = evolution.generateProposal(implications);

      expect(proposal.id).toContain('proposal-');
      expect(proposal.title).toBeTruthy();
      expect(proposal.implications).toBe(implications);
      expect(proposal.approval_status).toBe('pending');
    });

    it('should include benefits in proposal', () => {
      const implications_map = evolution.analyzeImplications();
      const implications = implications_map.get('epistemic-honesty')!;

      const proposal = evolution.generateProposal(implications);

      expect(proposal.benefits.length).toBeGreaterThan(0);
      expect(proposal.benefits.some((b) => b.includes('uncertainty'))).toBe(true);
    });

    it('should include risks in proposal', () => {
      const implications_map = evolution.analyzeImplications();
      const implications = implications_map.get('continuous-evolution')!;

      const proposal = evolution.generateProposal(implications);

      expect(proposal.risks.length).toBeGreaterThan(0);
    });

    it('should include migration strategy', () => {
      const implications_map = evolution.analyzeImplications();
      const implications = implications_map.get('lazy-efficiency')!;

      const proposal = evolution.generateProposal(implications);

      expect(proposal.migration_strategy).toContain('incremental');
      expect(proposal.migration_strategy).toContain('A/B test');
      expect(proposal.migration_strategy.toLowerCase()).toContain('rollback');
    });

    it('should assess reversibility', () => {
      const implications_map = evolution.analyzeImplications();

      const lazy_impl = implications_map.get('lazy-efficiency')!;
      const lazy_proposal = evolution.generateProposal(lazy_impl);

      const evolution_impl = implications_map.get('continuous-evolution')!;
      const evolution_proposal = evolution.generateProposal(evolution_impl);

      // Self-modifying is less reversible than caching
      expect(['moderate', 'difficult']).toContain(evolution_proposal.reversibility);
    });

    it('should estimate effort', () => {
      const implications_map = evolution.analyzeImplications();
      const implications = implications_map.get('epistemic-honesty')!;

      const proposal = evolution.generateProposal(implications);

      expect(proposal.estimated_effort).toBeGreaterThan(0);
    });

    it('should assess priority based on impact', () => {
      const implications_map = evolution.analyzeImplications();
      const implications = implications_map.get('lazy-efficiency')!;

      const proposal = evolution.generateProposal(implications);

      // System-wide changes should have medium/high priority
      expect(['medium', 'high', 'critical']).toContain(proposal.priority);
    });

    it('should calculate constitutional alignment', () => {
      const implications_map = evolution.analyzeImplications();
      const implications = implications_map.get('epistemic-honesty')!;

      const proposal = evolution.generateProposal(implications);

      expect(proposal.constitutional_alignment).toBeGreaterThan(0.7);
      expect(proposal.constitutional_alignment).toBeLessThanOrEqual(1.0);
    });

    it('should throw if generating proposal from empty implications', () => {
      expect(() => evolution.generateProposal([])).toThrow('Cannot generate proposal');
    });
  });

  // =========================================================================
  // Meta-Insights
  // =========================================================================

  describe('Meta-Architectural Insights', () => {
    it('should generate meta-insights about architecture-principle duality', () => {
      const insights = evolution.generateMetaInsights();

      expect(insights.length).toBeGreaterThan(0);

      const duality_insight = insights.find((i) => i.insight.includes('dual relationship'));
      expect(duality_insight).toBeDefined();
      expect(duality_insight!.paradigm_shift).toBe(true);
    });

    it('should identify principle compression opportunity', () => {
      const insights = evolution.generateMetaInsights();

      const compression = insights.find((i) => i.insight.includes('unified principle'));
      expect(compression).toBeDefined();
    });

    it('should recognize emergent self-awareness', () => {
      const insights = evolution.generateMetaInsights();

      const self_awareness = insights.find((i) => i.insight.includes('self-awareness'));
      expect(self_awareness).toBeDefined();
      expect(self_awareness!.architectural_consequences).toContain('System understands its own design');
    });

    it('should have confidence scores for insights', () => {
      const insights = evolution.generateMetaInsights();

      for (const insight of insights) {
        expect(insight.confidence).toBeGreaterThan(0);
        expect(insight.confidence).toBeLessThanOrEqual(1.0);
      }
    });

    it('should support insights with principles', () => {
      const insights = evolution.generateMetaInsights();

      for (const insight of insights) {
        expect(insight.supporting_principles.length).toBeGreaterThan(0);
      }
    });
  });

  // =========================================================================
  // Full Evolution Cycle
  // =========================================================================

  describe('Full Evolution Cycle', () => {
    it('should run complete evolution cycle', async () => {
      const result = await evolution.evolve();

      expect(result.implications.length).toBeGreaterThan(0);
      expect(result.proposals.length).toBeGreaterThan(0);
      expect(result.insights.length).toBeGreaterThan(0);
    });

    it('should only include proposals with sufficient constitutional alignment', async () => {
      const result = await evolution.evolve();

      for (const proposal of result.proposals) {
        expect(proposal.constitutional_alignment).toBeGreaterThanOrEqual(0.9); // Default threshold
      }
    });

    it('should auto-approve if configured', async () => {
      const auto_evolution = new ArchitecturalEvolution(memory, constitution, {
        require_human_approval: false,
      });

      const result = await auto_evolution.evolve();

      for (const proposal of result.proposals) {
        expect(proposal.approval_status).toBe('approved');
      }
    });

    it('should require human approval by default', async () => {
      const result = await evolution.evolve();

      for (const proposal of result.proposals) {
        expect(proposal.approval_status).toBe('pending');
      }
    });

    it('should generate proposals for all high-confidence principles', async () => {
      const result = await evolution.evolve();

      // Should have at least one proposal
      expect(result.proposals.length).toBeGreaterThanOrEqual(1);

      // Should analyze implications for all 3 principles
      expect(result.implications.length).toBeGreaterThanOrEqual(3);

      // Each principle should have implications
      for (const implications of result.implications) {
        expect(implications.length).toBeGreaterThan(0);
      }
    });
  });

  // =========================================================================
  // Export & State Management
  // =========================================================================

  describe('Export & State Management', () => {
    it('should export complete state', async () => {
      await evolution.evolve();

      const exported = evolution.export();
      const parsed = JSON.parse(exported);

      expect(parsed.principles).toBeDefined();
      expect(parsed.proposals).toBeDefined();
      expect(parsed.insights).toBeDefined();
      expect(parsed.config).toBeDefined();
    });

    it('should track all proposals', async () => {
      await evolution.evolve();

      const proposals = evolution.getProposals();
      expect(proposals.length).toBeGreaterThan(0);
    });

    it('should track all principles', () => {
      const principles = evolution.getPrinciples();
      expect(principles.length).toBeGreaterThanOrEqual(3); // At least the 3 pre-initialized
    });

    it('should track all insights', async () => {
      await evolution.evolve();

      const insights = evolution.getInsights();
      expect(insights.length).toBeGreaterThan(0);
    });
  });

  // =========================================================================
  // Integration Tests
  // =========================================================================

  describe('Integration with Other Systems', () => {
    it('should integrate with episodic memory for pattern discovery', async () => {
      // Add some episodes to memory
      memory.addEpisode(
        'How to implement caching?',
        'Use cache-first pattern',
        ['caching', 'performance'],
        ['optimization'],
        ['technical'],
        0,
        true,
        0.95,
        [],
        []
      );

      const result = await evolution.evolve();
      expect(result.proposals.length).toBeGreaterThan(0);
    });

    it('should validate proposals against constitution', async () => {
      const result = await evolution.evolve();

      for (const proposal of result.proposals) {
        // All proposals should have constitutional alignment score
        expect(proposal.constitutional_alignment).toBeGreaterThan(0);
      }
    });
  });

  // =========================================================================
  // Edge Cases
  // =========================================================================

  describe('Edge Cases', () => {
    it('should handle zero principles gracefully', () => {
      const empty_evolution = new ArchitecturalEvolution(memory, constitution, {
        min_principle_confidence: 1.0, // Will filter out all principles
      });

      const implications_map = empty_evolution.analyzeImplications();
      expect(implications_map.size).toBe(0);
    });

    it('should handle principle with no implications', () => {
      evolution.discoverPrinciple(
        'Abstract Principle',
        'Very abstract with no clear architectural implications',
        ['Abstract evidence'],
        'architecture'
      );

      const implications_map = evolution.analyzeImplications();
      // Should not crash, abstract principle won't match any patterns
      expect(implications_map).toBeDefined();
    });
  });

  // =========================================================================
  // Meta-Reflexive Tests
  // =========================================================================

  describe('Meta-Reflexive Behavior', () => {
    it('should demonstrate architecture → principles → architecture loop', async () => {
      // Initial architecture discovered principles
      const initial_principles = evolution.getPrinciples();
      expect(initial_principles.length).toBeGreaterThanOrEqual(3);

      // Principles generate architectural implications
      const implications_map = evolution.analyzeImplications();
      expect(implications_map.size).toBeGreaterThan(0);

      // Implications become proposals for new architecture
      const result = await evolution.evolve();
      expect(result.proposals.length).toBeGreaterThan(0);

      // New architecture would discover new principles (next iteration)
      // This demonstrates the meta-circular loop
    });

    it('should recognize its own self-reflective capability', async () => {
      const result = await evolution.evolve();

      const self_aware = result.insights.find((i) => i.insight.includes('self-awareness'));
      expect(self_aware).toBeDefined();
      expect(self_aware!.architectural_consequences).toContain('Can propose improvements to itself');
    });

    it('should generate increasingly abstract meta-insights', async () => {
      // First evolution cycle
      const result1 = await evolution.evolve();
      const initial_insights = result1.insights.length;

      // Discover another principle
      evolution.discoverPrinciple(
        'Simplicity Is All You Need',
        'Simplest solution that works is the best solution',
        ['YAGNI principle', 'KISS principle', 'Occam\'s Razor', 'Empirical: simpler code has fewer bugs'],
        'architecture'
      );

      // Second evolution cycle
      const result2 = await evolution.evolve();

      // Should generate new insights with 4 principles
      expect(result2.insights.length).toBeGreaterThanOrEqual(initial_insights);
    });
  });
});

describe('ArchitecturalEvolution Factory', () => {
  it('should create evolution system with factory function', () => {
    const memory = new EpisodicMemory();
    const constitution = new Constitution();

    const evolution = createArchitecturalEvolution(memory, constitution);

    expect(evolution).toBeInstanceOf(ArchitecturalEvolution);
    expect(evolution.getPrinciples().length).toBeGreaterThanOrEqual(3);
  });

  it('should accept custom configuration', () => {
    const memory = new EpisodicMemory();
    const constitution = new Constitution();

    const evolution = createArchitecturalEvolution(memory, constitution, {
      enable_auto_implementation: true,
      require_human_approval: false,
    });

    expect(evolution).toBeDefined();
  });
});
