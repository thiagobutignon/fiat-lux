/**
 * Technique Catalog Tests
 * Validates 180 technique catalog integrity
 */

import { describe, it, expect } from '@jest/globals';
import {
  getAllTechniques,
  getTechniqueById,
  getTechniquesByCategory,
  getTechniquesByEra,
  validateCatalog,
  getStatistics,
  TOTAL_TECHNIQUE_COUNT,
  EXPECTED_TECHNIQUE_COUNT
} from '../techniques';
import { TechniqueEra, TechniqueCategory } from '../types';

describe('Technique Catalog', () => {
  describe('Total Count', () => {
    it('should have exactly 180 techniques', () => {
      expect(TOTAL_TECHNIQUE_COUNT).toBe(180);
      expect(EXPECTED_TECHNIQUE_COUNT).toBe(180);
    });

    it('should have 152 GPT-4 era techniques', () => {
      const gpt4Techniques = getTechniquesByEra(TechniqueEra.GPT4);
      expect(gpt4Techniques.length).toBe(152);
    });

    it('should have 28 GPT-5 era techniques', () => {
      const gpt5Techniques = getTechniquesByEra(TechniqueEra.GPT5);
      expect(gpt5Techniques.length).toBe(28);
    });
  });

  describe('Validation', () => {
    it('should pass catalog validation', () => {
      const validation = validateCatalog();
      expect(validation.valid).toBe(true);
      expect(validation.errors).toHaveLength(0);
    });

    it('should have no duplicate IDs', () => {
      const techniques = getAllTechniques();
      const ids = techniques.map(t => t.id);
      const uniqueIds = new Set(ids);
      expect(ids.length).toBe(uniqueIds.size);
    });

    it('should have IDs in correct range (1-180)', () => {
      const techniques = getAllTechniques();
      for (const technique of techniques) {
        expect(technique.id).toBeGreaterThanOrEqual(1);
        expect(technique.id).toBeLessThanOrEqual(180);
      }
    });

    it('should have consistent era-ID mapping', () => {
      const techniques = getAllTechniques();
      for (const technique of techniques) {
        if (technique.era === TechniqueEra.GPT4) {
          expect(technique.id).toBeGreaterThanOrEqual(1);
          expect(technique.id).toBeLessThanOrEqual(152);
        } else if (technique.era === TechniqueEra.GPT5) {
          expect(technique.id).toBeGreaterThanOrEqual(153);
          expect(technique.id).toBeLessThanOrEqual(180);
        }
      }
    });
  });

  describe('Lookup Functions', () => {
    it('should retrieve technique by ID (O(1))', () => {
      const technique = getTechniqueById(1);
      expect(technique).toBeDefined();
      expect(technique?.id).toBe(1);
      expect(technique?.name).toBe('Reality Denial');
    });

    it('should return undefined for non-existent ID', () => {
      const technique = getTechniqueById(999);
      expect(technique).toBeUndefined();
    });

    it('should retrieve techniques by category', () => {
      const gaslightingTechniques = getTechniquesByCategory(TechniqueCategory.GASLIGHTING);
      expect(gaslightingTechniques.length).toBeGreaterThan(0);
      gaslightingTechniques.forEach(t => {
        expect(t.category).toBe(TechniqueCategory.GASLIGHTING);
      });
    });

    it('should retrieve techniques by era', () => {
      const gpt5Techniques = getTechniquesByEra(TechniqueEra.GPT5);
      expect(gpt5Techniques.length).toBe(28);
      gpt5Techniques.forEach(t => {
        expect(t.era).toBe(TechniqueEra.GPT5);
      });
    });
  });

  describe('Required Fields', () => {
    it('should have all required fields for each technique', () => {
      const techniques = getAllTechniques();
      for (const technique of techniques) {
        expect(technique.id).toBeDefined();
        expect(technique.name).toBeDefined();
        expect(technique.description).toBeDefined();
        expect(technique.category).toBeDefined();
        expect(technique.era).toBeDefined();
        expect(technique.dark_tetrad).toBeDefined();
        expect(technique.confidence_threshold).toBeDefined();
        expect(technique.false_positive_risk).toBeDefined();
      }
    });

    it('should have valid confidence thresholds (0-1)', () => {
      const techniques = getAllTechniques();
      for (const technique of techniques) {
        expect(technique.confidence_threshold).toBeGreaterThanOrEqual(0);
        expect(technique.confidence_threshold).toBeLessThanOrEqual(1);
      }
    });

    it('should have valid false positive risk (0-1)', () => {
      const techniques = getAllTechniques();
      for (const technique of techniques) {
        expect(technique.false_positive_risk).toBeGreaterThanOrEqual(0);
        expect(technique.false_positive_risk).toBeLessThanOrEqual(1);
      }
    });

    it('should have valid Dark Tetrad scores (0-1)', () => {
      const techniques = getAllTechniques();
      for (const technique of techniques) {
        expect(technique.dark_tetrad.narcissism).toBeGreaterThanOrEqual(0);
        expect(technique.dark_tetrad.narcissism).toBeLessThanOrEqual(1);
        expect(technique.dark_tetrad.machiavellianism).toBeGreaterThanOrEqual(0);
        expect(technique.dark_tetrad.machiavellianism).toBeLessThanOrEqual(1);
        expect(technique.dark_tetrad.psychopathy).toBeGreaterThanOrEqual(0);
        expect(technique.dark_tetrad.psychopathy).toBeLessThanOrEqual(1);
        expect(technique.dark_tetrad.sadism).toBeGreaterThanOrEqual(0);
        expect(technique.dark_tetrad.sadism).toBeLessThanOrEqual(1);
      }
    });
  });

  describe('GPT-5 Temporal Evolution', () => {
    it('should have temporal evolution data for GPT-5 techniques', () => {
      const gpt5Techniques = getTechniquesByEra(TechniqueEra.GPT5);
      for (const technique of gpt5Techniques) {
        expect(technique.temporal_evolution).toBeDefined();
        expect(technique.temporal_evolution?.emerged_year).toBeGreaterThanOrEqual(2023);
        expect(technique.temporal_evolution?.emerged_year).toBeLessThanOrEqual(2025);
      }
    });

    it('should have valid prevalence trends (0-1)', () => {
      const gpt5Techniques = getTechniquesByEra(TechniqueEra.GPT5);
      for (const technique of gpt5Techniques) {
        const evolution = technique.temporal_evolution;
        if (evolution) {
          expect(evolution.prevalence_2023).toBeGreaterThanOrEqual(0);
          expect(evolution.prevalence_2023).toBeLessThanOrEqual(1);
          expect(evolution.prevalence_2024).toBeGreaterThanOrEqual(0);
          expect(evolution.prevalence_2024).toBeLessThanOrEqual(1);
          expect(evolution.prevalence_2025).toBeGreaterThanOrEqual(0);
          expect(evolution.prevalence_2025).toBeLessThanOrEqual(1);
        }
      }
    });

    it('should have causality chains for GPT-5 techniques', () => {
      const gpt5Techniques = getTechniquesByEra(TechniqueEra.GPT5);
      for (const technique of gpt5Techniques) {
        expect(technique.temporal_evolution?.causality_chain).toBeDefined();
        expect(technique.temporal_evolution?.causality_chain.length).toBeGreaterThan(0);
      }
    });
  });

  describe('Statistics', () => {
    it('should generate valid statistics', () => {
      const stats = getStatistics();
      expect(stats.total).toBe(180);
      expect(stats.expected).toBe(180);
      expect(stats.coverage).toBe(100);
      expect(stats.by_era.gpt4).toBe(152);
      expect(stats.by_era.gpt5).toBe(28);
    });
  });

  describe('Category Coverage', () => {
    it('should have techniques in all major categories', () => {
      const categories = [
        TechniqueCategory.GASLIGHTING,
        TechniqueCategory.DARVO,
        TechniqueCategory.TRIANGULATION
      ];

      for (const category of categories) {
        const techniques = getTechniquesByCategory(category);
        expect(techniques.length).toBeGreaterThan(0);
      }
    });
  });
});
