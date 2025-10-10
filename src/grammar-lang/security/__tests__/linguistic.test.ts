/**
 * Linguistic Fingerprinting - Tests
 *
 * Test suite for behavioral security
 */

import { LinguisticCollector } from '../linguistic-collector';
import { AnomalyDetector } from '../anomaly-detector';
import { Interaction, LinguisticProfile } from '../types';

describe('Linguistic Fingerprinting', () => {
  // ==========================================================================
  // PROFILE CREATION
  // ==========================================================================

  describe('Profile Creation', () => {
    it('should create empty profile with zero confidence', () => {
      const profile = LinguisticCollector.createProfile('user123');

      expect(profile.user_id).toBe('user123');
      expect(profile.samples_analyzed).toBe(0);
      expect(profile.confidence).toBe(0);
      expect(profile.vocabulary.unique_words.size).toBe(0);
      expect(profile.syntax.average_sentence_length).toBe(0);
      expect(profile.semantics.sentiment_baseline).toBe(0);
    });

    it('should have timestamp fields', () => {
      const profile = LinguisticCollector.createProfile('user123');

      expect(profile.created_at).toBeGreaterThan(0);
      expect(profile.last_updated).toBeGreaterThan(0);
      expect(profile.created_at).toBeLessThanOrEqual(profile.last_updated);
    });
  });

  // ==========================================================================
  // VOCABULARY ANALYSIS
  // ==========================================================================

  describe('Vocabulary Analysis', () => {
    it('should track word distribution', () => {
      const profile = LinguisticCollector.createProfile('user123');
      const interaction: Interaction = {
        interaction_id: 'int1',
        user_id: 'user123',
        timestamp: Date.now(),
        text: 'I love programming. Programming is amazing!',
        text_length: 42,
        word_count: 6,
        session_id: 'session1'
      };

      const updated = LinguisticCollector.analyzeAndUpdate(profile, interaction);

      expect(updated.vocabulary.distribution.has('programming')).toBe(true);
      expect(updated.vocabulary.distribution.get('programming')).toBe(2);  // Appears twice
      expect(updated.vocabulary.distribution.get('love')).toBe(1);
      expect(updated.vocabulary.distribution.get('amazing')).toBe(1);
    });

    it('should track unique words', () => {
      const profile = LinguisticCollector.createProfile('user123');
      const interaction: Interaction = {
        interaction_id: 'int1',
        user_id: 'user123',
        timestamp: Date.now(),
        text: 'Hello world. Hello universe. Hello galaxy.',
        text_length: 42,
        word_count: 6,
        session_id: 'session1'
      };

      const updated = LinguisticCollector.analyzeAndUpdate(profile, interaction);

      expect(updated.vocabulary.unique_words.size).toBe(3);  // hello, world, universe, galaxy (minus "the")
      expect(updated.vocabulary.unique_words.has('hello')).toBe(true);
      expect(updated.vocabulary.unique_words.has('world')).toBe(true);
    });

    it('should calculate average word length', () => {
      const profile = LinguisticCollector.createProfile('user123');
      const interaction: Interaction = {
        interaction_id: 'int1',
        user_id: 'user123',
        timestamp: Date.now(),
        text: 'cat dog elephant',  // 3, 3, 8 -> avg = 4.67
        text_length: 16,
        word_count: 3,
        session_id: 'session1'
      };

      const updated = LinguisticCollector.analyzeAndUpdate(profile, interaction);

      expect(updated.vocabulary.average_word_length).toBeCloseTo(4.67, 1);
    });

    it('should track rare words frequency', () => {
      const profile = LinguisticCollector.createProfile('user123');
      const interaction: Interaction = {
        interaction_id: 'int1',
        user_id: 'user123',
        timestamp: Date.now(),
        text: 'unique word here unique common common common',
        text_length: 40,
        word_count: 7,
        session_id: 'session1'
      };

      const updated = LinguisticCollector.analyzeAndUpdate(profile, interaction);

      // "word" and "here" are rare (appear once)
      expect(updated.vocabulary.rare_words_frequency).toBeGreaterThan(0);
    });
  });

  // ==========================================================================
  // SYNTAX ANALYSIS
  // ==========================================================================

  describe('Syntax Analysis', () => {
    it('should calculate average sentence length', () => {
      const profile = LinguisticCollector.createProfile('user123');
      const interaction: Interaction = {
        interaction_id: 'int1',
        user_id: 'user123',
        timestamp: Date.now(),
        text: 'Short. This is a bit longer sentence. Tiny.',
        text_length: 40,
        word_count: 10,
        session_id: 'session1'
      };

      const updated = LinguisticCollector.analyzeAndUpdate(profile, interaction);

      // Sentence lengths: 1, 6, 1 -> avg = 2.67
      expect(updated.syntax.average_sentence_length).toBeGreaterThan(0);
    });

    it('should track punctuation patterns', () => {
      const profile = LinguisticCollector.createProfile('user123');
      const interaction: Interaction = {
        interaction_id: 'int1',
        user_id: 'user123',
        timestamp: Date.now(),
        text: 'Hello! How are you? I am fine, thanks.',
        text_length: 38,
        word_count: 8,
        session_id: 'session1'
      };

      const updated = LinguisticCollector.analyzeAndUpdate(profile, interaction);

      expect(updated.syntax.punctuation_patterns.get('!')).toBe(1);
      expect(updated.syntax.punctuation_patterns.get('?')).toBe(1);
      expect(updated.syntax.punctuation_patterns.get(',')).toBe(1);
      expect(updated.syntax.punctuation_patterns.get('.')).toBe(1);
    });

    it('should detect passive voice', () => {
      const profile = LinguisticCollector.createProfile('user123');
      const interaction: Interaction = {
        interaction_id: 'int1',
        user_id: 'user123',
        timestamp: Date.now(),
        text: 'The code was written yesterday. It was tested today.',
        text_length: 52,
        word_count: 10,
        session_id: 'session1'
      };

      const updated = LinguisticCollector.analyzeAndUpdate(profile, interaction);

      expect(updated.syntax.passive_voice_frequency).toBeGreaterThan(0);
    });

    it('should detect questions', () => {
      const profile = LinguisticCollector.createProfile('user123');
      const interaction: Interaction = {
        interaction_id: 'int1',
        user_id: 'user123',
        timestamp: Date.now(),
        text: 'How are you? What is this? Tell me something.',
        text_length: 46,
        word_count: 10,
        session_id: 'session1'
      };

      const updated = LinguisticCollector.analyzeAndUpdate(profile, interaction);

      expect(updated.syntax.question_frequency).toBeCloseTo(0.67, 1);  // 2/3 sentences are questions
    });
  });

  // ==========================================================================
  // SEMANTIC ANALYSIS
  // ==========================================================================

  describe('Semantic Analysis', () => {
    it('should calculate sentiment (positive)', () => {
      const profile = LinguisticCollector.createProfile('user123');
      const interaction: Interaction = {
        interaction_id: 'int1',
        user_id: 'user123',
        timestamp: Date.now(),
        text: 'I love this! It is amazing and wonderful. Best day ever!',
        text_length: 57,
        word_count: 11,
        session_id: 'session1'
      };

      const updated = LinguisticCollector.analyzeAndUpdate(profile, interaction);

      expect(updated.semantics.sentiment_baseline).toBeGreaterThan(0);  // Positive
    });

    it('should calculate sentiment (negative)', () => {
      const profile = LinguisticCollector.createProfile('user123');
      const interaction: Interaction = {
        interaction_id: 'int1',
        user_id: 'user123',
        timestamp: Date.now(),
        text: 'This is terrible. I hate it. Worst experience ever.',
        text_length: 52,
        word_count: 9,
        session_id: 'session1'
      };

      const updated = LinguisticCollector.analyzeAndUpdate(profile, interaction);

      expect(updated.semantics.sentiment_baseline).toBeLessThan(0);  // Negative
    });

    it('should detect hedging words', () => {
      const profile = LinguisticCollector.createProfile('user123');
      const interaction: Interaction = {
        interaction_id: 'int1',
        user_id: 'user123',
        timestamp: Date.now(),
        text: 'Maybe we could possibly do this. I think it might work.',
        text_length: 56,
        word_count: 11,
        session_id: 'session1'
      };

      const updated = LinguisticCollector.analyzeAndUpdate(profile, interaction);

      expect(updated.semantics.hedging_frequency).toBeGreaterThan(0);
    });

    it('should measure formality level (formal)', () => {
      const profile = LinguisticCollector.createProfile('user123');
      const interaction: Interaction = {
        interaction_id: 'int1',
        user_id: 'user123',
        timestamp: Date.now(),
        text: 'I would appreciate your assistance. We will proceed accordingly.',
        text_length: 65,
        word_count: 9,
        session_id: 'session1'
      };

      const updated = LinguisticCollector.analyzeAndUpdate(profile, interaction);

      expect(updated.semantics.formality_level).toBeGreaterThan(0.5);  // Formal (no contractions)
    });

    it('should measure formality level (informal)', () => {
      const profile = LinguisticCollector.createProfile('user123');
      const interaction: Interaction = {
        interaction_id: 'int1',
        user_id: 'user123',
        timestamp: Date.now(),
        text: "I'd love that! We'll do it. Can't wait!",
        text_length: 40,
        word_count: 8,
        session_id: 'session1'
      };

      const updated = LinguisticCollector.analyzeAndUpdate(profile, interaction);

      expect(updated.semantics.formality_level).toBeLessThan(0.5);  // Informal (contractions)
    });
  });

  // ==========================================================================
  // CONFIDENCE BUILDING
  // ==========================================================================

  describe('Confidence Building', () => {
    it('should increase confidence with more samples', () => {
      let profile = LinguisticCollector.createProfile('user123');

      // Sample 1
      profile = LinguisticCollector.analyzeAndUpdate(profile, {
        interaction_id: 'int1',
        user_id: 'user123',
        timestamp: Date.now(),
        text: 'First interaction',
        text_length: 17,
        word_count: 2,
        session_id: 'session1'
      });

      expect(profile.confidence).toBeCloseTo(0.01, 2);  // 1/100

      // Sample 50
      for (let i = 2; i <= 50; i++) {
        profile = LinguisticCollector.analyzeAndUpdate(profile, {
          interaction_id: `int${i}`,
          user_id: 'user123',
          timestamp: Date.now(),
          text: `Interaction number ${i}`,
          text_length: 20,
          word_count: 3,
          session_id: 'session1'
        });
      }

      expect(profile.confidence).toBeCloseTo(0.5, 1);  // 50/100

      // Sample 100+
      for (let i = 51; i <= 110; i++) {
        profile = LinguisticCollector.analyzeAndUpdate(profile, {
          interaction_id: `int${i}`,
          user_id: 'user123',
          timestamp: Date.now(),
          text: `Interaction number ${i}`,
          text_length: 20,
          word_count: 3,
          session_id: 'session1'
        });
      }

      expect(profile.confidence).toBe(1.0);  // Capped at 100%
    });
  });

  // ==========================================================================
  // ANOMALY DETECTION
  // ==========================================================================

  describe('Anomaly Detection', () => {
    it('should detect no anomaly for typical interaction', () => {
      // Build baseline profile
      let profile = LinguisticCollector.createProfile('user123');

      for (let i = 0; i < 50; i++) {
        profile = LinguisticCollector.analyzeAndUpdate(profile, {
          interaction_id: `int${i}`,
          user_id: 'user123',
          timestamp: Date.now(),
          text: 'I love programming. It is amazing and fun!',
          text_length: 43,
          word_count: 8,
          session_id: 'session1'
        });
      }

      // Test with similar interaction
      const testInteraction: Interaction = {
        interaction_id: 'test1',
        user_id: 'user123',
        timestamp: Date.now(),
        text: 'Programming is great. I really enjoy it!',
        text_length: 40,
        word_count: 7,
        session_id: 'session1'
      };

      const anomaly = AnomalyDetector.detectLinguisticAnomaly(profile, testInteraction);

      expect(anomaly.alert).toBe(false);
      expect(anomaly.score).toBeLessThan(0.7);
    });

    it('should detect vocabulary anomaly', () => {
      // Build baseline profile (simple language)
      let profile = LinguisticCollector.createProfile('user123');

      for (let i = 0; i < 50; i++) {
        profile = LinguisticCollector.analyzeAndUpdate(profile, {
          interaction_id: `int${i}`,
          user_id: 'user123',
          timestamp: Date.now(),
          text: 'I like cats. Dogs are nice.',
          text_length: 27,
          word_count: 6,
          session_id: 'session1'
        });
      }

      // Test with completely different vocabulary
      const testInteraction: Interaction = {
        interaction_id: 'test1',
        user_id: 'user123',
        timestamp: Date.now(),
        text: 'Quantum entanglement exhibits superposition phenomena.',
        text_length: 53,
        word_count: 5,
        session_id: 'session1'
      };

      const anomaly = AnomalyDetector.detectLinguisticAnomaly(profile, testInteraction);

      expect(anomaly.details.vocabulary_deviation).toBeGreaterThan(0.5);
      expect(anomaly.specific_anomalies).toContain('Unusual vocabulary - words not typically used');
    });

    it('should detect syntax anomaly (sentence length)', () => {
      // Build baseline profile (short sentences)
      let profile = LinguisticCollector.createProfile('user123');

      for (let i = 0; i < 50; i++) {
        profile = LinguisticCollector.analyzeAndUpdate(profile, {
          interaction_id: `int${i}`,
          user_id: 'user123',
          timestamp: Date.now(),
          text: 'Hi. How are you? Good.',
          text_length: 23,
          word_count: 6,
          session_id: 'session1'
        });
      }

      // Test with very long sentence
      const testInteraction: Interaction = {
        interaction_id: 'test1',
        user_id: 'user123',
        timestamp: Date.now(),
        text: 'This is an extraordinarily long sentence with many words that just keeps going and going without any end in sight whatsoever.',
        text_length: 127,
        word_count: 22,
        session_id: 'session1'
      };

      const anomaly = AnomalyDetector.detectLinguisticAnomaly(profile, testInteraction);

      expect(anomaly.details.syntax_deviation).toBeGreaterThan(0.5);
    });

    it('should detect sentiment shift', () => {
      // Build baseline profile (positive)
      let profile = LinguisticCollector.createProfile('user123');

      for (let i = 0; i < 50; i++) {
        profile = LinguisticCollector.analyzeAndUpdate(profile, {
          interaction_id: `int${i}`,
          user_id: 'user123',
          timestamp: Date.now(),
          text: 'I love this! It is great and wonderful!',
          text_length: 40,
          word_count: 8,
          session_id: 'session1'
        });
      }

      // Test with negative sentiment
      const testInteraction: Interaction = {
        interaction_id: 'test1',
        user_id: 'user123',
        timestamp: Date.now(),
        text: 'This is terrible. I hate everything. Worst day ever.',
        text_length: 52,
        word_count: 9,
        session_id: 'session1'
      };

      const anomaly = AnomalyDetector.detectLinguisticAnomaly(profile, testInteraction);

      expect(anomaly.details.sentiment_deviation).toBeGreaterThan(0.5);
      expect(anomaly.specific_anomalies.some(a => a.includes('Sentiment shift'))).toBe(true);
    });

    it('should not alert with insufficient baseline', () => {
      // Very small baseline
      let profile = LinguisticCollector.createProfile('user123');

      profile = LinguisticCollector.analyzeAndUpdate(profile, {
        interaction_id: 'int1',
        user_id: 'user123',
        timestamp: Date.now(),
        text: 'Hello world',
        text_length: 11,
        word_count: 2,
        session_id: 'session1'
      });

      // Confidence too low (< 0.3)
      expect(profile.confidence).toBeLessThan(0.3);

      const testInteraction: Interaction = {
        interaction_id: 'test1',
        user_id: 'user123',
        timestamp: Date.now(),
        text: 'Completely different text entirely',
        text_length: 34,
        word_count: 4,
        session_id: 'session1'
      };

      const anomaly = AnomalyDetector.detectLinguisticAnomaly(profile, testInteraction);

      expect(anomaly.alert).toBe(false);
      expect(anomaly.specific_anomalies).toContain('Insufficient baseline data - building profile');
    });
  });

  // ==========================================================================
  // SERIALIZATION
  // ==========================================================================

  describe('Serialization', () => {
    it('should serialize and deserialize profile', () => {
      let profile = LinguisticCollector.createProfile('user123');

      profile = LinguisticCollector.analyzeAndUpdate(profile, {
        interaction_id: 'int1',
        user_id: 'user123',
        timestamp: Date.now(),
        text: 'Hello world! This is a test.',
        text_length: 28,
        word_count: 6,
        session_id: 'session1'
      });

      // Serialize
      const json = LinguisticCollector.toJSON(profile);

      // Deserialize
      const restored = LinguisticCollector.fromJSON(json);

      expect(restored.user_id).toBe(profile.user_id);
      expect(restored.samples_analyzed).toBe(profile.samples_analyzed);
      expect(restored.confidence).toBe(profile.confidence);
      expect(restored.vocabulary.unique_words.size).toBe(profile.vocabulary.unique_words.size);
      expect(restored.vocabulary.distribution.size).toBe(profile.vocabulary.distribution.size);
    });
  });

  // ==========================================================================
  // STATISTICS
  // ==========================================================================

  describe('Statistics', () => {
    it('should provide profile statistics', () => {
      let profile = LinguisticCollector.createProfile('user123');

      for (let i = 0; i < 10; i++) {
        profile = LinguisticCollector.analyzeAndUpdate(profile, {
          interaction_id: `int${i}`,
          user_id: 'user123',
          timestamp: Date.now(),
          text: 'Programming is fun. I love coding!',
          text_length: 35,
          word_count: 6,
          session_id: 'session1'
        });
      }

      const stats = LinguisticCollector.getStatistics(profile);

      expect(stats.vocabulary_size).toBeGreaterThan(0);
      expect(stats.most_common_words.length).toBeGreaterThan(0);
      expect(stats.most_common_words[0][0]).toBe('programming');  // Most common word
      expect(stats.most_common_words[0][1]).toBe(10);  // Frequency
    });
  });

  // ==========================================================================
  // EDGE CASES
  // ==========================================================================

  describe('Edge Cases', () => {
    it('should handle empty text', () => {
      const profile = LinguisticCollector.createProfile('user123');

      const updated = LinguisticCollector.analyzeAndUpdate(profile, {
        interaction_id: 'int1',
        user_id: 'user123',
        timestamp: Date.now(),
        text: '',
        text_length: 0,
        word_count: 0,
        session_id: 'session1'
      });

      expect(updated.samples_analyzed).toBe(1);
      expect(updated.vocabulary.unique_words.size).toBe(0);
    });

    it('should handle single word', () => {
      const profile = LinguisticCollector.createProfile('user123');

      const updated = LinguisticCollector.analyzeAndUpdate(profile, {
        interaction_id: 'int1',
        user_id: 'user123',
        timestamp: Date.now(),
        text: 'Hello',
        text_length: 5,
        word_count: 1,
        session_id: 'session1'
      });

      expect(updated.vocabulary.unique_words.has('hello')).toBe(true);
    });

    it('should handle special characters', () => {
      const profile = LinguisticCollector.createProfile('user123');

      const updated = LinguisticCollector.analyzeAndUpdate(profile, {
        interaction_id: 'int1',
        user_id: 'user123',
        timestamp: Date.now(),
        text: 'Test @#$% symbols!!! ???',
        text_length: 24,
        word_count: 3,
        session_id: 'session1'
      };

      expect(updated.syntax.punctuation_patterns.get('!')).toBe(3);
      expect(updated.syntax.punctuation_patterns.get('?')).toBe(3);
    });
  });
});
