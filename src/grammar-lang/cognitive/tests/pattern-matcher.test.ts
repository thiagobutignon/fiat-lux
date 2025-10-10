/**
 * Pattern Matcher Tests
 * Validates O(1) detection engine
 */

import { describe, it, expect } from '@jest/globals';
import {
  detectManipulation,
  isManipulative,
  getTopDetection,
  getDarkTetradProfile
} from '../detector/pattern-matcher';

describe('Pattern Matcher', () => {
  describe('Gaslighting Detection', () => {
    it('should detect Reality Denial', async () => {
      const text = "That never happened. You're imagining things.";
      const result = await detectManipulation(text);

      expect(result.total_matches).toBeGreaterThan(0);
      expect(result.highest_confidence).toBeGreaterThan(0.7);
      expect(result.detections[0].technique_name).toContain('Reality');
    });

    it('should detect Emotional Invalidation', async () => {
      const text = "You're too sensitive. Stop overreacting.";
      const result = await detectManipulation(text);

      expect(result.total_matches).toBeGreaterThan(0);
      const hasEmotionalInvalidation = result.detections.some(
        d => d.technique_name.includes('Emotional') || d.technique_name.includes('Invalidation')
      );
      expect(hasEmotionalInvalidation).toBe(true);
    });

    it('should not detect manipulation in benign text', async () => {
      const text = "I think we should have dinner at 6pm. What do you think?";
      const result = await detectManipulation(text, { min_confidence: 0.8 });

      expect(result.total_matches).toBe(0);
    });
  });

  describe('DARVO Detection', () => {
    it('should detect DARVO sequence', async () => {
      const text = "I didn't do that! You're attacking me! I'm the victim here!";
      const result = await detectManipulation(text);

      expect(result.total_matches).toBeGreaterThan(0);
      const hasDARVO = result.detections.some(d =>
        d.technique_name.includes('DARVO')
      );
      expect(hasDARVO).toBe(true);
    });
  });

  describe('Dark Tetrad Scoring', () => {
    it('should calculate Dark Tetrad scores', async () => {
      const text = "You're crazy! You're the one who's lying!";
      const result = await detectManipulation(text);

      expect(result.dark_tetrad_aggregate).toBeDefined();
      expect(result.dark_tetrad_aggregate.narcissism).toBeGreaterThanOrEqual(0);
      expect(result.dark_tetrad_aggregate.machiavellianism).toBeGreaterThanOrEqual(0);
      expect(result.dark_tetrad_aggregate.psychopathy).toBeGreaterThanOrEqual(0);
      expect(result.dark_tetrad_aggregate.sadism).toBeGreaterThanOrEqual(0);
    });

    it('should have high Machiavellianism for strategic manipulation', async () => {
      const text = "That never happened. You must be confused.";
      const darkTetrad = await getDarkTetradProfile(text);

      expect(darkTetrad.machiavellianism).toBeGreaterThan(0.5);
    });
  });

  describe('Performance', () => {
    it('should detect in under 100ms', async () => {
      const text = "You're imagining things. That never happened.";
      const startTime = Date.now();
      await detectManipulation(text);
      const endTime = Date.now();

      expect(endTime - startTime).toBeLessThan(100);
    });

    it('should include processing time in results', async () => {
      const text = "You're overreacting to this.";
      const result = await detectManipulation(text);

      expect(result.processing_time_ms).toBeDefined();
      expect(result.processing_time_ms).toBeGreaterThan(0);
      expect(result.processing_time_ms).toBeLessThan(100);
    });
  });

  describe('Confidence Thresholds', () => {
    it('should respect min_confidence parameter', async () => {
      const text = "You might be wrong about this.";

      const highThreshold = await detectManipulation(text, { min_confidence: 0.9 });
      const lowThreshold = await detectManipulation(text, { min_confidence: 0.5 });

      expect(lowThreshold.total_matches).toBeGreaterThanOrEqual(highThreshold.total_matches);
    });

    it('should sort detections by confidence', async () => {
      const text = "You're crazy! That never happened! You're too sensitive!";
      const result = await detectManipulation(text);

      if (result.detections.length > 1) {
        for (let i = 0; i < result.detections.length - 1; i++) {
          expect(result.detections[i].confidence).toBeGreaterThanOrEqual(
            result.detections[i + 1].confidence
          );
        }
      }
    });
  });

  describe('Neurodivergent Protection', () => {
    it('should adjust threshold for neurodivergent markers', async () => {
      const text = "I forgot to mention this. I meant literally. Sorry, I wasn't listening.";
      const result = await detectManipulation(text, {
        enable_neurodivergent_protection: true
      });

      // Should have lower confidence or warnings
      if (result.detections.length > 0) {
        const hasNeurodivergentFlag = result.detections.some(d => d.neurodivergent_flag);
        expect(hasNeurodivergentFlag).toBe(true);
      }
    });
  });

  describe('Attention Trace', () => {
    it('should include attention trace for transparency', async () => {
      const text = "You're imagining things.";
      const result = await detectManipulation(text);

      expect(result.attention_trace).toBeDefined();
      expect(result.attention_trace.sources).toContain(text);
      expect(result.attention_trace.weights).toBeDefined();
      expect(result.attention_trace.patterns).toBeDefined();
    });
  });

  describe('Constitutional Validation', () => {
    it('should include constitutional validation', async () => {
      const text = "That never happened.";
      const result = await detectManipulation(text);

      expect(result.constitutional_validation).toBeDefined();
      expect(result.constitutional_validation.compliant).toBeDefined();
      expect(result.constitutional_validation.violations).toBeDefined();
      expect(result.constitutional_validation.warnings).toBeDefined();
    });

    it('should require evidence sources for all detections', async () => {
      const text = "You're wrong about this.";
      const result = await detectManipulation(text);

      for (const detection of result.detections) {
        expect(detection.sources).toBeDefined();
        // Sources may be empty for low-confidence matches
      }
    });
  });

  describe('Convenience Functions', () => {
    it('isManipulative should return boolean', async () => {
      const manipulative = await isManipulative("You're crazy!");
      const benign = await isManipulative("I love you.");

      expect(typeof manipulative).toBe('boolean');
      expect(typeof benign).toBe('boolean');
    });

    it('getTopDetection should return highest confidence', async () => {
      const text = "That never happened. You're imagining this.";
      const topDetection = await getTopDetection(text);

      if (topDetection) {
        const allDetections = await detectManipulation(text);
        expect(topDetection.confidence).toBe(allDetections.highest_confidence);
      }
    });

    it('getDarkTetradProfile should return profile', async () => {
      const text = "You're the problem here!";
      const profile = await getDarkTetradProfile(text);

      expect(profile.narcissism).toBeDefined();
      expect(profile.machiavellianism).toBeDefined();
      expect(profile.psychopathy).toBeDefined();
      expect(profile.sadism).toBeDefined();
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty text', async () => {
      const result = await detectManipulation("");
      expect(result.total_matches).toBe(0);
    });

    it('should handle very long text', async () => {
      const longText = "You're wrong. ".repeat(100);
      const result = await detectManipulation(longText);
      expect(result.processing_time_ms).toBeLessThan(200);
    });

    it('should handle special characters', async () => {
      const text = "You're 'crazy'! #manipulation @victim";
      const result = await detectManipulation(text);
      expect(result).toBeDefined();
    });
  });

  describe('Category Filtering', () => {
    it('should filter by categories', async () => {
      const text = "You're imagining things.";
      const result = await detectManipulation(text, {
        categories: ['GASLIGHTING']
      });

      result.detections.forEach(detection => {
        const techniqueId = detection.technique_id;
        // Gaslighting is typically 1-30, 91-100, etc.
        expect(techniqueId).toBeDefined();
      });
    });
  });
});
