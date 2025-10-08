/**
 * @file attention-tracker.test.ts
 * Tests for Attention Tracker - Interpretability Layer
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  AttentionTracker,
  AttentionTrace,
  QueryAttention,
  AttentionStats,
  computeInfluenceWeight,
  extractInfluentialConcepts,
} from '../core/attention-tracker';

describe('AttentionTracker', () => {
  let tracker: AttentionTracker;

  beforeEach(() => {
    tracker = new AttentionTracker();
  });

  describe('Query Management', () => {
    it('should start tracking a new query', () => {
      tracker.startQuery('query_1', 'How to optimize budget?');

      const attention = tracker.getQueryAttention('query_1');
      expect(attention).toBeDefined();
      expect(attention?.query).toBe('How to optimize budget?');
      expect(attention?.traces).toEqual([]);
      expect(attention?.total_concepts).toBe(0);
    });

    it('should require startQuery before adding traces', () => {
      expect(() => {
        tracker.addTrace('homeostasis', 'biology/cells.md', 0.91, 'Test reasoning');
      }).toThrow('No active query');
    });

    it('should end query and return attention data', () => {
      tracker.startQuery('query_1', 'Test query');
      tracker.addTrace('concept1', 'slice1.md', 0.8, 'Reasoning 1');
      tracker.addTrace('concept2', 'slice2.md', 0.9, 'Reasoning 2');

      const attention = tracker.endQuery();

      expect(attention).toBeDefined();
      expect(attention?.traces.length).toBe(2);
      expect(attention?.total_concepts).toBe(2);
      expect(attention?.top_influencers.length).toBeGreaterThan(0);
    });
  });

  describe('Trace Recording', () => {
    beforeEach(() => {
      tracker.startQuery('query_1', 'Test query');
    });

    it('should add attention trace', () => {
      tracker.addTrace(
        'homeostasis',
        'biology/cells.md',
        0.91,
        'Biological self-regulation mechanism'
      );

      const traces = tracker.getCurrentTraces();
      expect(traces.length).toBe(1);
      expect(traces[0].concept).toBe('homeostasis');
      expect(traces[0].slice).toBe('biology/cells.md');
      expect(traces[0].weight).toBe(0.91);
      expect(traces[0].reasoning).toBe('Biological self-regulation mechanism');
    });

    it('should validate weight range', () => {
      expect(() => {
        tracker.addTrace('concept', 'slice', 1.5, 'Invalid weight');
      }).toThrow('Invalid weight');

      expect(() => {
        tracker.addTrace('concept', 'slice', -0.1, 'Negative weight');
      }).toThrow('Invalid weight');
    });

    it('should accept valid weights 0-1', () => {
      tracker.addTrace('concept1', 'slice', 0.0, 'Min weight');
      tracker.addTrace('concept2', 'slice', 0.5, 'Mid weight');
      tracker.addTrace('concept3', 'slice', 1.0, 'Max weight');

      const traces = tracker.getCurrentTraces();
      expect(traces.length).toBe(3);
    });

    it('should add multiple traces at once', () => {
      const traces = [
        { concept: 'c1', slice: 's1', weight: 0.8, reasoning: 'r1' },
        { concept: 'c2', slice: 's2', weight: 0.9, reasoning: 'r2' },
        { concept: 'c3', slice: 's3', weight: 0.7, reasoning: 'r3' },
      ];

      tracker.addTraces(traces);

      const currentTraces = tracker.getCurrentTraces();
      expect(currentTraces.length).toBe(3);
    });

    it('should include timestamp in traces', () => {
      const beforeTime = Date.now();
      tracker.addTrace('concept', 'slice', 0.8, 'reasoning');
      const afterTime = Date.now();

      const traces = tracker.getCurrentTraces();
      expect(traces[0].timestamp).toBeGreaterThanOrEqual(beforeTime);
      expect(traces[0].timestamp).toBeLessThanOrEqual(afterTime);
    });
  });

  describe('Decision Path', () => {
    beforeEach(() => {
      tracker.startQuery('query_1', 'Test query');
    });

    it('should record decision points', () => {
      tracker.addDecisionPoint('Decomposed query into domains');
      tracker.addDecisionPoint('Invoked financial agent');
      tracker.addDecisionPoint('Composed final answer');

      const attention = tracker.endQuery();
      expect(attention?.decision_path).toEqual([
        'Decomposed query into domains',
        'Invoked financial agent',
        'Composed final answer',
      ]);
    });

    it('should handle empty decision path', () => {
      const attention = tracker.endQuery();
      expect(attention?.decision_path).toEqual([]);
    });
  });

  describe('Top Influencers', () => {
    beforeEach(() => {
      tracker.startQuery('query_1', 'Test query');
    });

    it('should calculate top 5 influencers by weight', () => {
      tracker.addTrace('concept1', 'slice1', 0.5, 'Low influence');
      tracker.addTrace('concept2', 'slice2', 0.91, 'High influence');
      tracker.addTrace('concept3', 'slice3', 0.84, 'Medium-high');
      tracker.addTrace('concept4', 'slice4', 0.77, 'Medium');
      tracker.addTrace('concept5', 'slice5', 0.3, 'Very low');
      tracker.addTrace('concept6', 'slice6', 0.95, 'Highest');

      const attention = tracker.endQuery();
      const topInfluencers = attention?.top_influencers || [];

      expect(topInfluencers.length).toBeLessThanOrEqual(5);
      expect(topInfluencers[0].concept).toBe('concept6'); // Highest weight
      expect(topInfluencers[0].weight).toBe(0.95);

      // Should be sorted descending
      for (let i = 1; i < topInfluencers.length; i++) {
        expect(topInfluencers[i - 1].weight).toBeGreaterThanOrEqual(topInfluencers[i].weight);
      }
    });

    it('should return all traces if less than 5', () => {
      tracker.addTrace('concept1', 'slice1', 0.8, 'reasoning 1');
      tracker.addTrace('concept2', 'slice2', 0.9, 'reasoning 2');

      const attention = tracker.endQuery();
      expect(attention?.top_influencers.length).toBe(2);
    });
  });

  describe('Statistics', () => {
    it('should calculate statistics across all queries', () => {
      // Query 1
      tracker.startQuery('query_1', 'Query 1');
      tracker.addTrace('homeostasis', 'biology/cells.md', 0.91, 'Bio concept');
      tracker.addTrace('feedback_loop', 'systems/control.md', 0.84, 'Systems concept');
      tracker.endQuery();

      // Query 2
      tracker.startQuery('query_2', 'Query 2');
      tracker.addTrace('homeostasis', 'biology/cells.md', 0.88, 'Bio again');
      tracker.addTrace('diversification', 'finance/risk.md', 0.77, 'Finance concept');
      tracker.endQuery();

      const stats = tracker.getStatistics();

      expect(stats.total_queries).toBe(2);
      expect(stats.total_traces).toBe(4);
      expect(stats.average_traces_per_query).toBe(2);
      expect(stats.most_influential_concepts.length).toBeGreaterThan(0);
      expect(stats.most_used_slices.length).toBeGreaterThan(0);
    });

    it('should identify most influential concepts', () => {
      tracker.startQuery('query_1', 'Query 1');
      tracker.addTrace('homeostasis', 'slice1', 0.9, 'High weight');
      tracker.endQuery();

      tracker.startQuery('query_2', 'Query 2');
      tracker.addTrace('homeostasis', 'slice1', 0.95, 'Even higher');
      tracker.endQuery();

      const stats = tracker.getStatistics();
      const topConcept = stats.most_influential_concepts[0];

      expect(topConcept.concept).toBe('homeostasis');
      expect(topConcept.count).toBe(2);
      expect(topConcept.average_weight).toBeCloseTo(0.925, 2);
    });

    it('should identify most used slices', () => {
      tracker.startQuery('query_1', 'Query 1');
      tracker.addTrace('c1', 'biology/cells.md', 0.8, 'r1');
      tracker.addTrace('c2', 'biology/cells.md', 0.9, 'r2');
      tracker.endQuery();

      const stats = tracker.getStatistics();
      const topSlice = stats.most_used_slices[0];

      expect(topSlice.slice).toBe('biology/cells.md');
      expect(topSlice.count).toBe(2);
    });

    it('should find high-confidence patterns', () => {
      // Same concepts appearing together
      tracker.startQuery('query_1', 'Query 1');
      tracker.addTrace('homeostasis', 'slice1', 0.9, 'r1');
      tracker.addTrace('feedback_loop', 'slice2', 0.85, 'r2');
      tracker.endQuery();

      tracker.startQuery('query_2', 'Query 2');
      tracker.addTrace('homeostasis', 'slice1', 0.88, 'r3');
      tracker.addTrace('feedback_loop', 'slice2', 0.86, 'r4');
      tracker.endQuery();

      const stats = tracker.getStatistics();
      const pattern = stats.high_confidence_patterns[0];

      expect(pattern).toBeDefined();
      expect(pattern.concepts).toContain('homeostasis');
      expect(pattern.concepts).toContain('feedback_loop');
      expect(pattern.frequency).toBe(2);
    });
  });

  describe('Audit Export', () => {
    it('should export data for regulatory compliance', () => {
      tracker.startQuery('query_1', 'Compliance test query');
      tracker.addTrace('concept1', 'slice1.md', 0.8, 'Reasoning 1');
      tracker.addDecisionPoint('Decision 1');
      tracker.endQuery();

      const audit = tracker.exportForAudit();

      expect(audit.export_timestamp).toBeDefined();
      expect(audit.total_queries).toBe(1);
      expect(audit.queries).toHaveLength(1);

      const query = audit.queries[0];
      expect(query.query_id).toBe('query_1');
      expect(query.query).toBe('Compliance test query');
      expect(query.decision_path).toContain('Decision 1');
      expect(query.traces).toHaveLength(1);
      expect(query.traces[0].concept).toBe('concept1');
    });

    it('should include all required audit fields', () => {
      tracker.startQuery('query_1', 'Test');
      tracker.addTrace('c1', 's1', 0.9, 'r1');
      tracker.endQuery();

      const audit = tracker.exportForAudit();
      const query = audit.queries[0];

      expect(query).toHaveProperty('query_id');
      expect(query).toHaveProperty('query');
      expect(query).toHaveProperty('timestamp');
      expect(query).toHaveProperty('decision_path');
      expect(query).toHaveProperty('traces');

      const trace = query.traces[0];
      expect(trace).toHaveProperty('concept');
      expect(trace).toHaveProperty('slice');
      expect(trace).toHaveProperty('weight');
      expect(trace).toHaveProperty('reasoning');
    });
  });

  describe('Query Explanation', () => {
    it('should generate human-readable explanation', () => {
      tracker.startQuery('query_1', 'How to stabilize spending?');
      tracker.addDecisionPoint('Decomposed into finance + biology');
      tracker.addTrace('homeostasis', 'biology/cells.md', 0.91, 'Bio stabilization');
      tracker.addTrace('feedback_loop', 'systems/control.md', 0.84, 'Control mechanism');
      tracker.endQuery();

      const explanation = tracker.explainQuery('query_1');

      expect(explanation).toContain('REASONING EXPLANATION');
      expect(explanation).toContain('How to stabilize spending?');
      expect(explanation).toContain('DECISION PATH');
      expect(explanation).toContain('Decomposed into finance + biology');
      expect(explanation).toContain('TOP 5 INFLUENCES');
      expect(explanation).toContain('homeostasis');
      expect(explanation).toContain('91.0%');
    });

    it('should handle non-existent query', () => {
      const explanation = tracker.explainQuery('nonexistent');
      expect(explanation).toContain('No attention data found');
    });
  });

  describe('Memory Management', () => {
    it('should clear all tracking data', () => {
      tracker.startQuery('query_1', 'Test');
      tracker.addTrace('c1', 's1', 0.8, 'r1');
      tracker.endQuery();

      tracker.clear();

      const attentions = tracker.getAllAttentions();
      expect(attentions.length).toBe(0);
    });

    it('should get memory statistics', () => {
      tracker.startQuery('query_1', 'Test 1');
      tracker.addTrace('c1', 's1', 0.8, 'r1');
      tracker.addTrace('c2', 's2', 0.9, 'r2');
      tracker.endQuery();

      tracker.startQuery('query_2', 'Test 2');
      tracker.addTrace('c3', 's3', 0.7, 'r3');
      tracker.endQuery();

      const stats = tracker.getMemoryStats();

      expect(stats.total_queries).toBe(2);
      expect(stats.total_traces).toBe(3);
      expect(stats.estimated_bytes).toBeGreaterThan(0);
    });

    it('should estimate memory usage reasonably', () => {
      tracker.startQuery('query_1', 'Test');
      tracker.addTrace('concept', 'slice', 0.8, 'reasoning');
      tracker.endQuery();

      const stats = tracker.getMemoryStats();
      // Each trace ~200 bytes, query ~500 bytes
      expect(stats.estimated_bytes).toBeGreaterThanOrEqual(500);
      expect(stats.estimated_bytes).toBeLessThanOrEqual(1000);
    });
  });

  describe('Multiple Queries', () => {
    it('should handle multiple concurrent queries', () => {
      tracker.startQuery('query_1', 'First query');
      tracker.addTrace('c1', 's1', 0.8, 'r1');
      const attention1 = tracker.endQuery();

      tracker.startQuery('query_2', 'Second query');
      tracker.addTrace('c2', 's2', 0.9, 'r2');
      const attention2 = tracker.endQuery();

      expect(attention1?.query).toBe('First query');
      expect(attention2?.query).toBe('Second query');

      const allAttentions = tracker.getAllAttentions();
      expect(allAttentions.length).toBe(2);
    });

    it('should maintain separate traces per query', () => {
      tracker.startQuery('query_1', 'Query 1');
      tracker.addTrace('c1', 's1', 0.8, 'r1');
      tracker.endQuery();

      tracker.startQuery('query_2', 'Query 2');
      tracker.addTrace('c2', 's2', 0.9, 'r2');
      tracker.endQuery();

      const q1 = tracker.getQueryAttention('query_1');
      const q2 = tracker.getQueryAttention('query_2');

      expect(q1?.traces[0].concept).toBe('c1');
      expect(q2?.traces[0].concept).toBe('c2');
    });
  });
});

describe('Utility Functions', () => {
  describe('computeInfluenceWeight', () => {
    it('should compute geometric mean of confidence and relevance', () => {
      const weight = computeInfluenceWeight(0.9, 0.8);
      expect(weight).toBeCloseTo(Math.sqrt(0.9 * 0.8), 5);
    });

    it('should return 0 if either input is 0', () => {
      expect(computeInfluenceWeight(0, 0.8)).toBe(0);
      expect(computeInfluenceWeight(0.9, 0)).toBe(0);
    });

    it('should return 1 if both inputs are 1', () => {
      expect(computeInfluenceWeight(1, 1)).toBe(1);
    });

    it('should balance confidence and relevance', () => {
      const highConfLowRel = computeInfluenceWeight(0.9, 0.1);
      const lowConfHighRel = computeInfluenceWeight(0.1, 0.9);
      const balanced = computeInfluenceWeight(0.5, 0.5);

      expect(highConfLowRel).toBeCloseTo(lowConfHighRel, 5);
      expect(balanced).toBe(0.5);
    });
  });

  describe('extractInfluentialConcepts', () => {
    it('should extract concepts mentioned in response', () => {
      const response = 'The homeostasis mechanism and feedback_loop are key';
      const availableConcepts = ['homeostasis', 'feedback_loop', 'diversification'];

      const influential = extractInfluentialConcepts(response, availableConcepts);

      expect(influential).toContain('homeostasis');
      expect(influential).toContain('feedback_loop');
      expect(influential).not.toContain('diversification');
    });

    it('should be case-insensitive', () => {
      const response = 'Homeostasis and FEEDBACK_LOOP are important';
      const availableConcepts = ['homeostasis', 'feedback_loop'];

      const influential = extractInfluentialConcepts(response, availableConcepts);

      expect(influential).toContain('homeostasis');
      expect(influential).toContain('feedback_loop');
    });

    it('should return empty array if no concepts found', () => {
      const response = 'This text has no matching concepts';
      const availableConcepts = ['homeostasis', 'feedback_loop'];

      const influential = extractInfluentialConcepts(response, availableConcepts);

      expect(influential).toEqual([]);
    });

    it('should handle empty inputs', () => {
      expect(extractInfluentialConcepts('', ['concept'])).toEqual([]);
      expect(extractInfluentialConcepts('text', [])).toEqual([]);
    });
  });
});
