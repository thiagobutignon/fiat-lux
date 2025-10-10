/**
 * Typing Pattern Tests
 *
 * Test typing profile collection and anomaly detection
 */

import { TypingCollector } from '../typing-collector';
import { TypingAnomalyDetector } from '../typing-anomaly-detector';
import { Interaction, TypingProfile } from '../types';

describe('TypingCollector', () => {
  describe('Profile Creation', () => {
    test('should create empty typing profile', () => {
      const profile = TypingCollector.createProfile('alice');

      expect(profile.user_id).toBe('alice');
      expect(profile.samples_analyzed).toBe(0);
      expect(profile.confidence).toBe(0);
      expect(profile.timing.keystroke_interval_avg).toBe(0);
      expect(profile.errors.typo_rate).toBe(0);
    });
  });

  describe('Timing Pattern Analysis', () => {
    test('should analyze keystroke timing', () => {
      let profile = TypingCollector.createProfile('alice');

      const interaction: Interaction = {
        interaction_id: 'test_1',
        user_id: 'alice',
        timestamp: Date.now(),
        text: 'Hello world',
        text_length: 11,
        word_count: 2,
        session_id: 'session_1',
        typing_data: {
          keystroke_intervals: [100, 120, 110, 105, 115, 100, 120, 110, 105, 115, 100],
          total_typing_time: 1200,
          pauses: [300, 250],
          backspaces: 2,
          corrections: 1,
        },
      };

      profile = TypingCollector.analyzeAndUpdate(profile, interaction);

      expect(profile.samples_analyzed).toBe(1);
      expect(profile.timing.keystroke_interval_avg).toBeGreaterThan(0);
      expect(profile.timing.keystroke_intervals.length).toBe(11);
    });

    test('should calculate pause durations correctly', () => {
      let profile = TypingCollector.createProfile('alice');

      const interaction: Interaction = {
        interaction_id: 'test_1',
        user_id: 'alice',
        timestamp: Date.now(),
        text: 'Test',
        text_length: 4,
        word_count: 1,
        session_id: 'session_1',
        typing_data: {
          keystroke_intervals: [100, 100, 100],
          total_typing_time: 2000,
          pauses: [200, 300, 1000, 1500, 3000], // short, word, thinking, thinking, sentence
          backspaces: 0,
          corrections: 0,
        },
      };

      profile = TypingCollector.analyzeAndUpdate(profile, interaction);

      // Word pause should be < 500ms average
      expect(profile.timing.word_pause_duration).toBeLessThan(500);

      // Thinking pause should be 500-2000ms
      expect(profile.timing.thinking_pause_duration).toBeGreaterThan(500);
      expect(profile.timing.thinking_pause_duration).toBeLessThan(2000);

      // Sentence pause should be > 2000ms
      expect(profile.timing.sentence_pause_duration).toBeGreaterThan(2000);
    });

    test('should build confidence with more samples', () => {
      let profile = TypingCollector.createProfile('alice');

      // Add 10 samples
      for (let i = 0; i < 10; i++) {
        const interaction: Interaction = {
          interaction_id: `test_${i}`,
          user_id: 'alice',
          timestamp: Date.now(),
          text: 'Sample text',
          text_length: 11,
          word_count: 2,
          session_id: 'session_1',
          typing_data: {
            keystroke_intervals: [100, 110, 105, 115, 100, 120, 110, 105, 115, 100, 120],
            total_typing_time: 1200,
            pauses: [300],
            backspaces: 1,
            corrections: 0,
          },
        };
        profile = TypingCollector.analyzeAndUpdate(profile, interaction);
      }

      expect(profile.samples_analyzed).toBe(10);
      expect(profile.confidence).toBe(0.1); // 10/100 = 10%

      // Add 90 more samples
      for (let i = 10; i < 100; i++) {
        const interaction: Interaction = {
          interaction_id: `test_${i}`,
          user_id: 'alice',
          timestamp: Date.now(),
          text: 'Sample',
          text_length: 6,
          word_count: 1,
          session_id: 'session_1',
          typing_data: {
            keystroke_intervals: [100, 110, 105, 115, 100, 120],
            total_typing_time: 600,
            pauses: [],
            backspaces: 0,
            corrections: 0,
          },
        };
        profile = TypingCollector.analyzeAndUpdate(profile, interaction);
      }

      expect(profile.samples_analyzed).toBe(100);
      expect(profile.confidence).toBe(1.0); // 100/100 = 100%
    });
  });

  describe('Error Pattern Analysis', () => {
    test('should track typo rate', () => {
      let profile = TypingCollector.createProfile('alice');

      const interaction: Interaction = {
        interaction_id: 'test_1',
        user_id: 'alice',
        timestamp: Date.now(),
        text: 'Test with errors',
        text_length: 16,
        word_count: 3,
        session_id: 'session_1',
        typing_data: {
          keystroke_intervals: [100, 100, 100],
          total_typing_time: 1000,
          pauses: [],
          backspaces: 5,
          corrections: 3,
        },
      };

      profile = TypingCollector.analyzeAndUpdate(profile, interaction);

      expect(profile.errors.typo_rate).toBeGreaterThan(0);
      expect(profile.errors.backspace_frequency).toBe(5);
    });

    test('should track low error rate', () => {
      let profile = TypingCollector.createProfile('alice');

      const interaction: Interaction = {
        interaction_id: 'test_1',
        user_id: 'alice',
        timestamp: Date.now(),
        text: 'Perfect typing',
        text_length: 14,
        word_count: 2,
        session_id: 'session_1',
        typing_data: {
          keystroke_intervals: [100, 100, 100],
          total_typing_time: 1000,
          pauses: [],
          backspaces: 0,
          corrections: 0,
        },
      };

      profile = TypingCollector.analyzeAndUpdate(profile, interaction);

      expect(profile.errors.typo_rate).toBe(0);
      expect(profile.errors.backspace_frequency).toBe(0);
    });
  });

  describe('Input Behavior Detection', () => {
    test('should detect input burst (paste)', () => {
      let profile = TypingCollector.createProfile('alice');

      const longText = 'This is a very long text that was pasted from somewhere else';
      const interaction: Interaction = {
        interaction_id: 'test_1',
        user_id: 'alice',
        timestamp: Date.now(),
        text: longText,
        text_length: longText.length,
        word_count: 12,
        session_id: 'session_1',
        typing_data: {
          keystroke_intervals: [5, 5, 5, 5, 5], // Very fast (paste)
          total_typing_time: 25,
          pauses: [],
          backspaces: 0,
          corrections: 0,
        },
      };

      profile = TypingCollector.analyzeAndUpdate(profile, interaction);

      expect(profile.input.input_burst_detected).toBe(true);
      expect(profile.input.copy_paste_frequency).toBe(1);
    });

    test('should NOT detect burst for normal fast typing', () => {
      let profile = TypingCollector.createProfile('alice');

      const interaction: Interaction = {
        interaction_id: 'test_1',
        user_id: 'alice',
        timestamp: Date.now(),
        text: 'Short', // Short text, even if fast
        text_length: 5,
        word_count: 1,
        session_id: 'session_1',
        typing_data: {
          keystroke_intervals: [8, 8, 8, 8],
          total_typing_time: 32,
          pauses: [],
          backspaces: 0,
          corrections: 0,
        },
      };

      profile = TypingCollector.analyzeAndUpdate(profile, interaction);

      expect(profile.input.input_burst_detected).toBe(false);
    });
  });

  describe('Serialization', () => {
    test('should serialize and deserialize profile', () => {
      let profile = TypingCollector.createProfile('alice');

      // Add some data
      profile.timing.keystroke_intervals = [100, 110, 105];
      profile.timing.keystroke_interval_avg = 105;
      profile.errors.common_typos.set('teh', 'the');
      profile.errors.common_typos.set('recieve', 'receive');

      const json = TypingCollector.toJSON(profile);
      const restored = TypingCollector.fromJSON(json);

      expect(restored.user_id).toBe(profile.user_id);
      expect(restored.timing.keystroke_interval_avg).toBe(105);
      expect(restored.errors.common_typos.get('teh')).toBe('the');
      expect(restored.errors.common_typos.get('recieve')).toBe('receive');
    });
  });
});

describe('TypingAnomalyDetector', () => {
  describe('Anomaly Detection', () => {
    test('should detect no anomaly for consistent typing', () => {
      // Build baseline
      let profile = TypingCollector.createProfile('alice');
      for (let i = 0; i < 50; i++) {
        const interaction: Interaction = {
          interaction_id: `baseline_${i}`,
          user_id: 'alice',
          timestamp: Date.now(),
          text: 'Consistent typing pattern',
          text_length: 25,
          word_count: 3,
          session_id: 'session_1',
          typing_data: {
            keystroke_intervals: Array(25).fill(110), // Consistent 110ms
            total_typing_time: 2750,
            pauses: [300, 250],
            backspaces: 1,
            corrections: 0,
          },
        };
        profile = TypingCollector.analyzeAndUpdate(profile, interaction);
      }

      // Test with similar interaction
      const testInteraction: Interaction = {
        interaction_id: 'test',
        user_id: 'alice',
        timestamp: Date.now(),
        text: 'Still consistent',
        text_length: 16,
        word_count: 2,
        session_id: 'session_2',
        typing_data: {
          keystroke_intervals: Array(16).fill(112), // Very similar
          total_typing_time: 1792,
          pauses: [280],
          backspaces: 1,
          corrections: 0,
        },
      };

      const anomaly = TypingAnomalyDetector.detectTypingAnomaly(profile, testInteraction);

      expect(anomaly.alert).toBe(false);
      expect(anomaly.score).toBeLessThan(0.5);
    });

    test('should detect speed deviation (too fast)', () => {
      // Build baseline (normal speed: 110ms/keystroke)
      let profile = TypingCollector.createProfile('alice');
      for (let i = 0; i < 50; i++) {
        const interaction: Interaction = {
          interaction_id: `baseline_${i}`,
          user_id: 'alice',
          timestamp: Date.now(),
          text: 'Normal speed',
          text_length: 12,
          word_count: 2,
          session_id: 'session_1',
          typing_data: {
            keystroke_intervals: Array(12).fill(110),
            total_typing_time: 1320,
            pauses: [300],
            backspaces: 1,
            corrections: 0,
          },
        };
        profile = TypingCollector.analyzeAndUpdate(profile, interaction);
      }

      // Test with very fast typing (rushed/duress)
      const testInteraction: Interaction = {
        interaction_id: 'test',
        user_id: 'alice',
        timestamp: Date.now(),
        text: 'Rushed typing',
        text_length: 13,
        word_count: 2,
        session_id: 'session_2',
        typing_data: {
          keystroke_intervals: Array(13).fill(50), // 2x faster!
          total_typing_time: 650,
          pauses: [100],
          backspaces: 3,
          corrections: 2,
        },
      };

      const anomaly = TypingAnomalyDetector.detectTypingAnomaly(profile, testInteraction);

      expect(anomaly.details.speed_deviation).toBeGreaterThan(0.5);
      expect(anomaly.specific_anomalies.length).toBeGreaterThan(0);
    });

    test('should detect error rate spike (stress)', () => {
      // Build baseline (low error rate)
      let profile = TypingCollector.createProfile('alice');
      for (let i = 0; i < 50; i++) {
        const interaction: Interaction = {
          interaction_id: `baseline_${i}`,
          user_id: 'alice',
          timestamp: Date.now(),
          text: 'Clean typing',
          text_length: 12,
          word_count: 2,
          session_id: 'session_1',
          typing_data: {
            keystroke_intervals: Array(12).fill(110),
            total_typing_time: 1320,
            pauses: [300],
            backspaces: 1,
            corrections: 0,
          },
        };
        profile = TypingCollector.analyzeAndUpdate(profile, interaction);
      }

      // Test with many errors (stress/duress)
      const testInteraction: Interaction = {
        interaction_id: 'test',
        user_id: 'alice',
        timestamp: Date.now(),
        text: 'Error prone',
        text_length: 11,
        word_count: 2,
        session_id: 'session_2',
        typing_data: {
          keystroke_intervals: Array(11).fill(110),
          total_typing_time: 1210,
          pauses: [300],
          backspaces: 10,
          corrections: 8,
        },
      };

      const anomaly = TypingAnomalyDetector.detectTypingAnomaly(profile, testInteraction);

      expect(anomaly.details.error_rate_change).toBeGreaterThan(0.5);
    });

    test('should detect input burst (paste attack)', () => {
      let profile = TypingCollector.createProfile('alice');

      // Baseline doesn't matter much for burst detection
      for (let i = 0; i < 50; i++) {
        const interaction: Interaction = {
          interaction_id: `baseline_${i}`,
          user_id: 'alice',
          timestamp: Date.now(),
          text: 'Normal',
          text_length: 6,
          word_count: 1,
          session_id: 'session_1',
          typing_data: {
            keystroke_intervals: Array(6).fill(110),
            total_typing_time: 660,
            pauses: [],
            backspaces: 0,
            corrections: 0,
          },
        };
        profile = TypingCollector.analyzeAndUpdate(profile, interaction);
      }

      // Test with paste (very fast + long text)
      const longText = 'This is a very long text that was obviously pasted from somewhere else';
      const testInteraction: Interaction = {
        interaction_id: 'test',
        user_id: 'alice',
        timestamp: Date.now(),
        text: longText,
        text_length: longText.length,
        word_count: 13,
        session_id: 'session_2',
        typing_data: {
          keystroke_intervals: Array(longText.length).fill(5), // Impossibly fast
          total_typing_time: longText.length * 5,
          pauses: [],
          backspaces: 0,
          corrections: 0,
        },
      };

      const anomaly = TypingAnomalyDetector.detectTypingAnomaly(profile, testInteraction);

      expect(anomaly.details.input_burst).toBe(true);
      expect(anomaly.alert).toBe(true); // This is VERY suspicious
    });
  });

  describe('Duress Detection', () => {
    test('should detect duress from rushed typing', () => {
      // Build baseline
      let profile = TypingCollector.createProfile('alice');
      for (let i = 0; i < 50; i++) {
        const interaction: Interaction = {
          interaction_id: `baseline_${i}`,
          user_id: 'alice',
          timestamp: Date.now(),
          text: 'Normal',
          text_length: 6,
          word_count: 1,
          session_id: 'session_1',
          typing_data: {
            keystroke_intervals: Array(6).fill(110),
            total_typing_time: 660,
            pauses: [300],
            backspaces: 1,
            corrections: 0,
          },
        };
        profile = TypingCollector.analyzeAndUpdate(profile, interaction);
      }

      // Test duress: very fast + high errors
      const testInteraction: Interaction = {
        interaction_id: 'test',
        user_id: 'alice',
        timestamp: Date.now(),
        text: 'Rushed',
        text_length: 6,
        word_count: 1,
        session_id: 'session_2',
        typing_data: {
          keystroke_intervals: Array(6).fill(40), // 3x faster (rushed!)
          total_typing_time: 240,
          pauses: [100],
          backspaces: 5,
          corrections: 4,
        },
      };

      const duress = TypingAnomalyDetector.detectDuressFromTyping(profile, testInteraction);

      expect(duress.duress_detected).toBe(true);
      expect(duress.confidence).toBeGreaterThan(0.6);
      expect(duress.indicators).toContain('Typing significantly faster (rushed under duress)');
    });

    test('should NOT detect duress for normal typing', () => {
      // Build baseline
      let profile = TypingCollector.createProfile('alice');
      for (let i = 0; i < 50; i++) {
        const interaction: Interaction = {
          interaction_id: `baseline_${i}`,
          user_id: 'alice',
          timestamp: Date.now(),
          text: 'Normal',
          text_length: 6,
          word_count: 1,
          session_id: 'session_1',
          typing_data: {
            keystroke_intervals: Array(6).fill(110),
            total_typing_time: 660,
            pauses: [300],
            backspaces: 1,
            corrections: 0,
          },
        };
        profile = TypingCollector.analyzeAndUpdate(profile, interaction);
      }

      // Test normal typing
      const testInteraction: Interaction = {
        interaction_id: 'test',
        user_id: 'alice',
        timestamp: Date.now(),
        text: 'Normal',
        text_length: 6,
        word_count: 1,
        session_id: 'session_2',
        typing_data: {
          keystroke_intervals: Array(6).fill(112), // Very similar
          total_typing_time: 672,
          pauses: [290],
          backspaces: 1,
          corrections: 0,
        },
      };

      const duress = TypingAnomalyDetector.detectDuressFromTyping(profile, testInteraction);

      expect(duress.duress_detected).toBe(false);
      expect(duress.confidence).toBeLessThan(0.5);
    });
  });

  describe('Edge Cases', () => {
    test('should handle insufficient baseline', () => {
      const profile = TypingCollector.createProfile('alice'); // No samples

      const testInteraction: Interaction = {
        interaction_id: 'test',
        user_id: 'alice',
        timestamp: Date.now(),
        text: 'Test',
        text_length: 4,
        word_count: 1,
        session_id: 'session_1',
        typing_data: {
          keystroke_intervals: [100, 100, 100],
          total_typing_time: 300,
          pauses: [],
          backspaces: 0,
          corrections: 0,
        },
      };

      const anomaly = TypingAnomalyDetector.detectTypingAnomaly(profile, testInteraction);

      expect(anomaly.alert).toBe(false);
      expect(anomaly.specific_anomalies).toContain('Insufficient baseline data - building typing profile');
    });

    test('should handle missing typing data', () => {
      let profile = TypingCollector.createProfile('alice');

      // Build some baseline
      for (let i = 0; i < 50; i++) {
        const interaction: Interaction = {
          interaction_id: `baseline_${i}`,
          user_id: 'alice',
          timestamp: Date.now(),
          text: 'Normal',
          text_length: 6,
          word_count: 1,
          session_id: 'session_1',
          typing_data: {
            keystroke_intervals: Array(6).fill(110),
            total_typing_time: 660,
            pauses: [],
            backspaces: 0,
            corrections: 0,
          },
        };
        profile = TypingCollector.analyzeAndUpdate(profile, interaction);
      }

      // Test without typing data
      const testInteraction: Interaction = {
        interaction_id: 'test',
        user_id: 'alice',
        timestamp: Date.now(),
        text: 'No typing data',
        text_length: 15,
        word_count: 3,
        session_id: 'session_2',
        // No typing_data
      };

      const anomaly = TypingAnomalyDetector.detectTypingAnomaly(profile, testInteraction);

      expect(anomaly.alert).toBe(false);
      expect(anomaly.specific_anomalies).toContain('No typing data available in current interaction');
    });
  });
});
