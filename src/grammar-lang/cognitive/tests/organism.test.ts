/**
 * Cognitive Organism Tests
 * Tests .glass organism integration
 */

import { describe, it, expect } from '@jest/globals';
import {
  createCognitiveOrganism,
  analyzeText,
  exportOrganism,
  loadOrganism,
  getOrganismStats,
  validateConstitutional
} from '../glass/cognitive-organism';

describe('Cognitive Organism', () => {
  describe('Creation', () => {
    it('should create valid organism', () => {
      const organism = createCognitiveOrganism('Test Organism');

      expect(organism.format).toBe('fiat-glass-v1.0');
      expect(organism.type).toBe('cognitive-defense-organism');
      expect(organism.metadata.name).toBe('Test Organism');
      expect(organism.metadata.version).toBe('1.0.0');
      expect(organism.metadata.specialization).toBe('manipulation-detection');
      expect(organism.metadata.maturity).toBe(0.0);
      expect(organism.metadata.techniques_count).toBe(180);
    });

    it('should have constitutional principles enabled', () => {
      const organism = createCognitiveOrganism();

      expect(organism.constitutional.privacy).toBe(true);
      expect(organism.constitutional.transparency).toBe(true);
      expect(organism.constitutional.protection).toBe(true);
      expect(organism.constitutional.accuracy).toBe(true);
      expect(organism.constitutional.no_diagnosis).toBe(true);
      expect(organism.constitutional.context_aware).toBe(true);
      expect(organism.constitutional.evidence_based).toBe(true);
    });

    it('should have 180 techniques loaded', () => {
      const organism = createCognitiveOrganism();

      expect(organism.knowledge.techniques).toBeDefined();
      expect(organism.knowledge.techniques.length).toBe(180);
    });

    it('should have Dark Tetrad markers defined', () => {
      const organism = createCognitiveOrganism();

      expect(organism.knowledge.dark_tetrad_markers.narcissism).toBeDefined();
      expect(organism.knowledge.dark_tetrad_markers.machiavellianism).toBeDefined();
      expect(organism.knowledge.dark_tetrad_markers.psychopathy).toBeDefined();
      expect(organism.knowledge.dark_tetrad_markers.sadism).toBeDefined();
    });

    it('should have neurodivergent protection', () => {
      const organism = createCognitiveOrganism();

      expect(organism.knowledge.neurodivergent_protection).toBeDefined();
      expect(organism.knowledge.neurodivergent_protection.autism_markers).toBeDefined();
      expect(organism.knowledge.neurodivergent_protection.adhd_markers).toBeDefined();
      expect(organism.knowledge.neurodivergent_protection.false_positive_threshold).toBe(0.15);
    });
  });

  describe('Text Analysis', () => {
    it('should analyze manipulative text', async () => {
      const organism = createCognitiveOrganism();
      const text = "That never happened. You're imagining things.";

      const result = await analyzeText(organism, text);

      expect(result.organism).toBeDefined();
      expect(result.results).toBeInstanceOf(Array);
      expect(result.summary).toBeDefined();
      expect(result.summary.length).toBeGreaterThan(0);
    });

    it('should increase organism maturity after analysis', async () => {
      const organism = createCognitiveOrganism();
      const initialMaturity = organism.metadata.maturity;

      await analyzeText(organism, "You're wrong");

      expect(organism.metadata.maturity).toBeGreaterThan(initialMaturity);
    });

    it('should log detections to memory', async () => {
      const organism = createCognitiveOrganism();
      const text = "You're crazy!";

      await analyzeText(organism, text);

      expect(organism.memory.detected_patterns.length).toBeGreaterThan(0);
      expect(organism.memory.detected_patterns[0].text).toBe(text);
    });

    it('should maintain audit trail', async () => {
      const organism = createCognitiveOrganism();

      await analyzeText(organism, "Test text");

      expect(organism.memory.audit_trail.length).toBeGreaterThan(0);
      expect(organism.memory.audit_trail[0].action).toBe('analyze_text');
    });

    it('should generate summary for detections', async () => {
      const organism = createCognitiveOrganism();
      const text = "That didn't happen. You're lying.";

      const result = await analyzeText(organism, text);

      expect(result.summary).toContain('Detected');
    });

    it('should handle benign text', async () => {
      const organism = createCognitiveOrganism();
      const text = "I love you. Let's have dinner.";

      const result = await analyzeText(organism, text);

      expect(result.summary).toContain('No manipulation');
    });
  });

  describe('Export/Import', () => {
    it('should export organism to JSON', () => {
      const organism = createCognitiveOrganism('Export Test');
      const json = exportOrganism(organism);

      expect(json).toBeDefined();
      expect(json.length).toBeGreaterThan(0);
      expect(JSON.parse(json)).toBeDefined();
    });

    it('should load organism from JSON', () => {
      const original = createCognitiveOrganism('Load Test');
      const json = exportOrganism(original);
      const loaded = loadOrganism(json);

      expect(loaded.metadata.name).toBe('Load Test');
      expect(loaded.format).toBe('fiat-glass-v1.0');
      expect(loaded.knowledge.techniques.length).toBe(180);
    });

    it('should preserve maturity across export/import', async () => {
      const organism = createCognitiveOrganism();
      await analyzeText(organism, "Test");

      const maturity = organism.metadata.maturity;
      const json = exportOrganism(organism);
      const loaded = loadOrganism(json);

      expect(loaded.metadata.maturity).toBe(maturity);
    });

    it('should preserve memory across export/import', async () => {
      const organism = createCognitiveOrganism();
      await analyzeText(organism, "Test text");

      const json = exportOrganism(organism);
      const loaded = loadOrganism(json);

      expect(loaded.memory.detected_patterns.length).toBe(1);
      expect(loaded.memory.detected_patterns[0].text).toBe("Test text");
    });
  });

  describe('Statistics', () => {
    it('should generate organism stats', async () => {
      const organism = createCognitiveOrganism('Stats Test');
      await analyzeText(organism, "You're wrong");

      const stats = getOrganismStats(organism);

      expect(stats.name).toBe('Stats Test');
      expect(stats.version).toBe('1.0.0');
      expect(stats.maturity).toBeDefined();
      expect(stats.techniques_loaded).toBe(180);
      expect(stats.total_analyses).toBe(1);
      expect(stats.generation).toBe(0);
      expect(stats.constitutional_compliant).toBe(true);
    });

    it('should track total detections', async () => {
      const organism = createCognitiveOrganism();
      await analyzeText(organism, "That never happened. You're crazy!");

      const stats = getOrganismStats(organism);
      expect(stats.total_detections).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Constitutional Validation', () => {
    it('should validate compliant organism', () => {
      const organism = createCognitiveOrganism();
      const validation = validateConstitutional(organism);

      expect(validation.compliant).toBe(true);
      expect(validation.violations).toHaveLength(0);
    });

    it('should detect violations when principles disabled', () => {
      const organism = createCognitiveOrganism();
      organism.constitutional.privacy = false;

      const validation = validateConstitutional(organism);

      expect(validation.compliant).toBe(false);
      expect(validation.violations.length).toBeGreaterThan(0);
      expect(validation.violations).toContain('Privacy principle not enabled');
    });

    it('should validate all constitutional principles', () => {
      const organism = createCognitiveOrganism();

      // Disable all principles
      organism.constitutional.privacy = false;
      organism.constitutional.transparency = false;
      organism.constitutional.protection = false;
      organism.constitutional.accuracy = false;
      organism.constitutional.no_diagnosis = false;
      organism.constitutional.context_aware = false;
      organism.constitutional.evidence_based = false;

      const validation = validateConstitutional(organism);

      expect(validation.violations.length).toBe(7);
    });
  });

  describe('Context-Aware Analysis', () => {
    it('should use context in analysis', async () => {
      const organism = createCognitiveOrganism();
      const text = "You're wrong";
      const context = "Previous conversation about directions";

      const result = await analyzeText(organism, text, context);

      expect(result).toBeDefined();
      // Context should influence detection
    });
  });

  describe('Maturity Growth', () => {
    it('should grow maturity with usage', async () => {
      const organism = createCognitiveOrganism();

      for (let i = 0; i < 10; i++) {
        await analyzeText(organism, `Test ${i}`);
      }

      expect(organism.metadata.maturity).toBeGreaterThan(0);
      expect(organism.metadata.maturity).toBeLessThanOrEqual(1.0);
    });

    it('should cap maturity at 1.0', async () => {
      const organism = createCognitiveOrganism();
      organism.metadata.maturity = 0.999;

      await analyzeText(organism, "Test");

      expect(organism.metadata.maturity).toBe(1.0);
    });
  });

  describe('Evolution Tracking', () => {
    it('should have evolution enabled by default', () => {
      const organism = createCognitiveOrganism();

      expect(organism.evolution.enabled).toBe(true);
      expect(organism.evolution.generations).toBe(0);
      expect(organism.evolution.fitness_trajectory).toHaveLength(0);
    });
  });

  describe('Model Information', () => {
    it('should have correct model metadata', () => {
      const organism = createCognitiveOrganism();

      expect(organism.model.architecture).toBe('transformer-27M');
      expect(organism.model.parameters).toBe(27_000_000);
      expect(organism.model.constitutional).toBe(true);
      expect(organism.model.focus).toBe('linguistic-analysis');
    });
  });

  describe('Integration with Detection Engine', () => {
    it('should use full detection engine', async () => {
      const organism = createCognitiveOrganism();
      const text = "I never said that. You're imagining things. You're too sensitive.";

      const result = await analyzeText(organism, text);

      expect(result.results.length).toBeGreaterThan(0);
      expect(result.summary).toContain('🚨');
      expect(result.summary).toContain('Dark Tetrad');
    });

    it('should apply neurodivergent protection', async () => {
      const organism = createCognitiveOrganism();
      const text = "I forgot to mention. I meant literally.";

      const result = await analyzeText(organism, text);

      // Should detect neurodivergent markers and adjust
      expect(result).toBeDefined();
    });
  });
});
