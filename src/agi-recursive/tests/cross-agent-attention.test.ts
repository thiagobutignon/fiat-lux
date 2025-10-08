/**
 * @file cross-agent-attention.test.ts
 * Tests for Cross-Agent Attention mechanism
 *
 * Key capabilities tested:
 * - Multi-head attention initialization
 * - Embedding creation from agent outputs
 * - Attention weight calculation
 * - Weight normalization (softmax)
 * - Output blending based on attention
 * - Cross-domain insight generation
 * - Weight learning from history
 * - Attention visualization
 * - Configuration export/import
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  CrossAgentAttention,
  AgentOutput,
  createCrossAgentAttention,
  extractAttentionMatrix,
} from '../core/cross-agent-attention';

describe('CrossAgentAttention', () => {
  let attention: CrossAgentAttention;
  let testOutputs: AgentOutput[];

  beforeEach(() => {
    attention = new CrossAgentAttention();

    testOutputs = [
      {
        agent_id: 'financial',
        domain: 'finance',
        answer: 'Financial analysis',
        concepts: ['budget', 'spending', 'investment'],
        confidence: 0.9,
        reasoning: 'Based on financial data',
      },
      {
        agent_id: 'biology',
        domain: 'biology',
        answer: 'Biological perspective',
        concepts: ['homeostasis', 'feedback', 'regulation'],
        confidence: 0.85,
        reasoning: 'Based on biological principles',
      },
      {
        agent_id: 'systems',
        domain: 'systems',
        answer: 'Systems analysis',
        concepts: ['feedback', 'leverage', 'interconnection'],
        confidence: 0.8,
        reasoning: 'Based on systems thinking',
      },
    ];
  });

  describe('Constructor', () => {
    it('should create instance with default config', () => {
      expect(attention).toBeInstanceOf(CrossAgentAttention);
    });

    it('should accept custom config', () => {
      const customAttention = new CrossAgentAttention({
        num_heads: 8,
        embedding_dim: 256,
        temperature: 0.5,
      });

      expect(customAttention).toBeInstanceOf(CrossAgentAttention);
    });

    it('should initialize with default 4 heads', () => {
      const config = attention.exportConfig();
      expect(config.num_heads).toBe(4);
    });

    it('should initialize with default embedding dim 512', () => {
      const config = attention.exportConfig();
      expect(config.embedding_dim).toBe(512);
    });

    it('should initialize with learning enabled by default', () => {
      const config = attention.exportConfig();
      expect(config.learn_weights).toBe(true);
    });
  });

  describe('applyAttention', () => {
    it('should process agent outputs with attention', () => {
      const attended = attention.applyAttention(testOutputs);

      expect(attended.length).toBe(3);
      attended.forEach((output) => {
        expect(output.agent_id).toBeDefined();
        expect(output.attention_weights).toBeDefined();
        expect(output.attended_concepts).toBeDefined();
      });
    });

    it('should return empty array for empty input', () => {
      const attended = attention.applyAttention([]);

      expect(attended).toEqual([]);
    });

    it('should calculate attention weights between agents', () => {
      const attended = attention.applyAttention(testOutputs);

      attended.forEach((output) => {
        expect(output.attention_weights.length).toBeGreaterThan(0);
        output.attention_weights.forEach((weight) => {
          expect(weight.from_agent).toBe(output.agent_id);
          expect(weight.to_agent).toBeDefined();
          expect(weight.weight).toBeGreaterThanOrEqual(0);
          expect(weight.weight).toBeLessThanOrEqual(1);
        });
      });
    });

    it('should normalize attention weights to sum to 1', () => {
      const attended = attention.applyAttention(testOutputs);

      attended.forEach((output) => {
        const sum = output.attention_weights.reduce((s, w) => s + w.weight, 0);
        expect(sum).toBeCloseTo(1.0, 5);
      });
    });

    it('should blend concepts from attended agents', () => {
      const attended = attention.applyAttention(testOutputs);

      attended.forEach((output) => {
        expect(output.attended_concepts.length).toBeGreaterThanOrEqual(output.original_output.concepts.length);
      });
    });

    it('should generate cross-domain insights', () => {
      const attended = attention.applyAttention(testOutputs);

      const hasInsights = attended.some((output) => output.cross_domain_insights.length > 0);
      expect(hasInsights).toBe(true);
    });

    it('should include multi-head contributions', () => {
      const attended = attention.applyAttention(testOutputs);

      attended.forEach((output) => {
        output.attention_weights.forEach((weight) => {
          expect(Array.isArray(weight.head_contributions)).toBe(true);
          expect(weight.head_contributions.length).toBe(4); // Default num_heads
        });
      });
    });

    it('should blend confidence scores', () => {
      const attended = attention.applyAttention(testOutputs);

      attended.forEach((output) => {
        expect(output.blended_confidence).toBeGreaterThan(0);
        expect(output.blended_confidence).toBeLessThanOrEqual(1);
      });
    });
  });

  describe('Self-Attention', () => {
    it('should exclude self-attention by default', () => {
      const attended = attention.applyAttention(testOutputs);

      attended.forEach((output) => {
        const self_weights = output.attention_weights.filter((w) => w.to_agent === output.agent_id);
        expect(self_weights.length).toBe(0);
      });
    });

    it('should include self-attention when enabled', () => {
      const attentionWithSelf = new CrossAgentAttention({ enable_self_attention: true });
      const attended = attentionWithSelf.applyAttention(testOutputs);

      attended.forEach((output) => {
        const total_weights = output.attention_weights.length;
        expect(total_weights).toBe(testOutputs.length); // Including self
      });
    });
  });

  describe('Weight Learning', () => {
    it('should learn weights from history when enabled', () => {
      // First application
      attention.applyAttention(testOutputs);

      // Get statistics
      const stats = attention.getWeightStatistics();
      expect(stats.size).toBeGreaterThan(0);
    });

    it('should accumulate statistics over multiple applications', () => {
      attention.applyAttention(testOutputs);
      attention.applyAttention(testOutputs);
      attention.applyAttention(testOutputs);

      const stats = attention.getWeightStatistics();
      for (const [key, stat] of stats.entries()) {
        expect(stat.count).toBe(3);
        expect(stat.avg).toBeGreaterThan(0);
      }
    });

    it('should not learn when disabled', () => {
      const noLearnAttention = new CrossAgentAttention({ learn_weights: false });
      noLearnAttention.applyAttention(testOutputs);

      const stats = noLearnAttention.getWeightStatistics();
      expect(stats.size).toBe(0);
    });

    it('should bias towards historically useful connections', () => {
      // Apply multiple times to establish history
      for (let i = 0; i < 5; i++) {
        attention.applyAttention(testOutputs);
      }

      // Weights should stabilize
      const attended = attention.applyAttention(testOutputs);

      attended.forEach((output) => {
        output.attention_weights.forEach((weight) => {
          expect(weight.weight).toBeGreaterThan(0);
        });
      });
    });
  });

  describe('Attention History', () => {
    it('should store attention history per agent', () => {
      attention.applyAttention(testOutputs);

      const history = attention.getAttentionHistory('financial');
      expect(history.length).toBeGreaterThan(0);
    });

    it('should return empty array for unknown agent', () => {
      const history = attention.getAttentionHistory('unknown');
      expect(history).toEqual([]);
    });

    it('should update history on each application', () => {
      attention.applyAttention(testOutputs);
      const history1 = attention.getAttentionHistory('financial');

      attention.applyAttention(testOutputs);
      const history2 = attention.getAttentionHistory('financial');

      // History is overwritten, not accumulated (per-query basis)
      expect(history2.length).toBeGreaterThan(0);
    });
  });

  describe('Embedding Creation', () => {
    it('should create embeddings for agent outputs', () => {
      const attended = attention.applyAttention(testOutputs);

      // Embeddings are created internally, verify through attention weights
      expect(attended.length).toBe(testOutputs.length);
    });

    it('should use provided embeddings when available', () => {
      const outputsWithEmbeddings = testOutputs.map((output) => ({
        ...output,
        embedding: new Array(512).fill(0.1),
      }));

      const attended = attention.applyAttention(outputsWithEmbeddings);
      expect(attended.length).toBe(testOutputs.length);
    });

    it('should handle outputs with different concepts', () => {
      const diverseOutputs = [
        { ...testOutputs[0], concepts: ['a', 'b', 'c'] },
        { ...testOutputs[1], concepts: ['d', 'e', 'f'] },
        { ...testOutputs[2], concepts: ['g', 'h', 'i'] },
      ];

      const attended = attention.applyAttention(diverseOutputs);
      expect(attended.length).toBe(3);
    });
  });

  describe('Cross-Domain Insights', () => {
    it('should identify high-attention connections', () => {
      const attended = attention.applyAttention(testOutputs);

      const withInsights = attended.filter((output) => output.cross_domain_insights.length > 0);
      expect(withInsights.length).toBeGreaterThan(0);
    });

    it('should highlight cross-domain connections', () => {
      const attended = attention.applyAttention(testOutputs);

      // Should have insights with cross-domain indicators
      const crossDomainInsights = attended.flatMap((output) => output.cross_domain_insights).filter((insight) => insight.includes('←→'));

      expect(crossDomainInsights.length).toBeGreaterThan(0);
    });

    it('should identify shared concepts', () => {
      // Biology and Systems both have 'feedback'
      const attended = attention.applyAttention(testOutputs);

      const biologyOutput = attended.find((a) => a.agent_id === 'biology');
      const systemsOutput = attended.find((a) => a.agent_id === 'systems');

      // They should have high attention to each other due to shared concept
      const biologyToSystems = biologyOutput?.attention_weights.find((w) => w.to_agent === 'systems');
      const systemsToBiology = systemsOutput?.attention_weights.find((w) => w.to_agent === 'biology');

      expect(biologyToSystems).toBeDefined();
      expect(systemsToBiology).toBeDefined();
    });
  });

  describe('Attention Visualization', () => {
    it('should generate ASCII attention matrix', () => {
      const attended = attention.applyAttention(testOutputs);
      const viz = attention.visualizeAttention(attended);

      expect(viz).toContain('Attention Matrix');
      expect(viz).toContain('finan'); // Truncated names
      expect(viz).toContain('biolo');
      expect(viz).toContain('syste');
    });

    it('should use bars to represent weights', () => {
      const attended = attention.applyAttention(testOutputs);
      const viz = attention.visualizeAttention(attended);

      expect(viz).toContain('█');
    });

    it('should handle single agent', () => {
      const attended = attention.applyAttention([testOutputs[0]]);
      const viz = attention.visualizeAttention(attended);

      expect(viz).toContain('Attention Matrix');
    });
  });

  describe('Configuration', () => {
    it('should export configuration', () => {
      const config = attention.exportConfig();

      expect(config.num_heads).toBeDefined();
      expect(config.embedding_dim).toBeDefined();
      expect(config.head_dim).toBeDefined();
      expect(config.temperature).toBeDefined();
    });

    it('should export custom configuration', () => {
      const custom = new CrossAgentAttention({
        num_heads: 8,
        temperature: 0.5,
      });

      const config = custom.exportConfig();
      expect(config.num_heads).toBe(8);
      expect(config.temperature).toBe(0.5);
    });

    it('should support different head dimensions', () => {
      const custom = new CrossAgentAttention({
        num_heads: 4,
        embedding_dim: 256,
        head_dim: 32,
      });

      const attended = custom.applyAttention(testOutputs);
      expect(attended.length).toBe(3);
    });

    it('should support temperature scaling', () => {
      const lowTemp = new CrossAgentAttention({ temperature: 0.1 });
      const highTemp = new CrossAgentAttention({ temperature: 2.0 });

      const attendedLow = lowTemp.applyAttention(testOutputs);
      const attendedHigh = highTemp.applyAttention(testOutputs);

      // Low temperature should create sharper distributions
      const lowMaxWeight = Math.max(...attendedLow[0].attention_weights.map((w) => w.weight));
      const highMaxWeight = Math.max(...attendedHigh[0].attention_weights.map((w) => w.weight));

      expect(lowMaxWeight).toBeGreaterThan(highMaxWeight);
    });
  });

  describe('Weight Export', () => {
    it('should export learned weights as JSON', () => {
      attention.applyAttention(testOutputs);
      const exported = attention.exportWeights();

      expect(typeof exported).toBe('string');
      expect(() => JSON.parse(exported)).not.toThrow();
    });

    it('should include statistics in export', () => {
      attention.applyAttention(testOutputs);
      const exported = JSON.parse(attention.exportWeights());

      expect(exported.statistics).toBeDefined();
      expect(Array.isArray(exported.statistics)).toBe(true);
    });

    it('should include config in export', () => {
      attention.applyAttention(testOutputs);
      const exported = JSON.parse(attention.exportWeights());

      expect(exported.config).toBeDefined();
      expect(exported.config.num_heads).toBe(4);
    });

    it('should include attention head metadata', () => {
      attention.applyAttention(testOutputs);
      const exported = JSON.parse(attention.exportWeights());

      expect(exported.attention_heads).toBeDefined();
      expect(Array.isArray(exported.attention_heads)).toBe(true);
    });
  });

  describe('Clear', () => {
    beforeEach(() => {
      attention.applyAttention(testOutputs);
      attention.applyAttention(testOutputs);
    });

    it('should clear attention history', () => {
      attention.clear();

      const history = attention.getAttentionHistory('financial');
      expect(history).toEqual([]);
    });

    it('should clear weight statistics', () => {
      attention.clear();

      const stats = attention.getWeightStatistics();
      expect(stats.size).toBe(0);
    });

    it('should allow processing after clear', () => {
      attention.clear();

      const attended = attention.applyAttention(testOutputs);
      expect(attended.length).toBe(3);
    });
  });

  describe('Factory Function', () => {
    it('should create attention instance', () => {
      const instance = createCrossAgentAttention();

      expect(instance).toBeInstanceOf(CrossAgentAttention);
    });

    it('should accept config', () => {
      const instance = createCrossAgentAttention({ num_heads: 8 });

      const config = instance.exportConfig();
      expect(config.num_heads).toBe(8);
    });
  });

  describe('Utility Functions', () => {
    describe('extractAttentionMatrix', () => {
      it('should extract attention matrix', () => {
        const attended = attention.applyAttention(testOutputs);
        const matrix = extractAttentionMatrix(attended);

        expect(matrix.length).toBe(testOutputs.length);
        expect(matrix[0].length).toBe(testOutputs.length);
      });

      it('should have normalized rows', () => {
        const attended = attention.applyAttention(testOutputs);
        const matrix = extractAttentionMatrix(attended);

        matrix.forEach((row) => {
          const sum = row.reduce((s, val) => s + val, 0);
          expect(sum).toBeCloseTo(1.0, 5);
        });
      });

      it('should handle single agent', () => {
        const attended = attention.applyAttention([testOutputs[0]]);
        const matrix = extractAttentionMatrix(attended);

        expect(matrix.length).toBe(1);
        expect(matrix[0].length).toBe(1);
      });
    });
  });

  describe('Integration', () => {
    it('should handle complete workflow', () => {
      // Apply attention
      const attended = attention.applyAttention(testOutputs);

      // Verify outputs
      expect(attended.length).toBe(3);
      attended.forEach((output) => {
        expect(output.attention_weights.length).toBeGreaterThan(0);
        expect(output.attended_concepts.length).toBeGreaterThan(0);
        expect(output.blended_confidence).toBeGreaterThan(0);
      });

      // Get history
      const history = attention.getAttentionHistory('financial');
      expect(history.length).toBeGreaterThan(0);

      // Get statistics
      const stats = attention.getWeightStatistics();
      expect(stats.size).toBeGreaterThan(0);

      // Visualize
      const viz = attention.visualizeAttention(attended);
      expect(viz).toContain('Attention Matrix');

      // Export
      const exported = attention.exportWeights();
      expect(exported).toBeDefined();
    });

    it('should handle multiple rounds of attention', () => {
      const rounds = 5;
      for (let i = 0; i < rounds; i++) {
        const attended = attention.applyAttention(testOutputs);
        expect(attended.length).toBe(3);
      }

      const stats = attention.getWeightStatistics();
      for (const [key, stat] of stats.entries()) {
        expect(stat.count).toBe(rounds);
      }
    });

    it('should maintain consistency across applications', () => {
      const attended1 = attention.applyAttention(testOutputs);
      const attended2 = attention.applyAttention(testOutputs);

      // Structure should be consistent
      expect(attended1.length).toBe(attended2.length);
      expect(attended1[0].attention_weights.length).toBe(attended2[0].attention_weights.length);
    });
  });
});
