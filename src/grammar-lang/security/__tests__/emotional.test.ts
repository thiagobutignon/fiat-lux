/**
 * Emotional Signature - Test Suite
 *
 * Tests for EmotionalCollector and EmotionalAnomalyDetector
 */

import { describe, test, expect } from '@jest/globals';
import { EmotionalCollector } from '../emotional-collector';
import { EmotionalAnomalyDetector } from '../emotional-anomaly-detector';
import { Interaction } from '../types';

describe('EmotionalCollector', () => {
  // ==========================================================================
  // PROFILE CREATION
  // ==========================================================================

  test('should create emotional profile', () => {
    const profile = EmotionalCollector.createProfile('alice');

    expect(profile.user_id).toBe('alice');
    expect(profile.samples_analyzed).toBe(0);
    expect(profile.confidence).toBe(0);
    expect(profile.baseline.valence).toBe(0);
    expect(profile.baseline.arousal).toBe(0.5);
    expect(profile.baseline.dominance).toBe(0.5);
  });

  // ==========================================================================
  // VAD ANALYSIS
  // ==========================================================================

  test('should analyze positive valence (sentiment)', () => {
    let profile = EmotionalCollector.createProfile('alice');

    const interaction: Interaction = {
      interaction_id: 'test_1',
      user_id: 'alice',
      timestamp: Date.now(),
      text: 'I am so happy and excited! This is wonderful and amazing!',
      text_length: 55,
      word_count: 10,
      session_id: 'session_1',
    };

    profile = EmotionalCollector.analyzeAndUpdate(profile, interaction);

    expect(profile.baseline.valence).toBeGreaterThan(0); // Positive sentiment
    expect(profile.samples_analyzed).toBe(1);
  });

  test('should analyze negative valence (sentiment)', () => {
    let profile = EmotionalCollector.createProfile('alice');

    const interaction: Interaction = {
      interaction_id: 'test_2',
      user_id: 'alice',
      timestamp: Date.now(),
      text: 'I am sad and worried. This is terrible and awful.',
      text_length: 50,
      word_count: 10,
      session_id: 'session_1',
    };

    profile = EmotionalCollector.analyzeAndUpdate(profile, interaction);

    expect(profile.baseline.valence).toBeLessThan(0); // Negative sentiment
    expect(profile.samples_analyzed).toBe(1);
  });

  test('should analyze high arousal (stress)', () => {
    let profile = EmotionalCollector.createProfile('alice');

    const interaction: Interaction = {
      interaction_id: 'test_3',
      user_id: 'alice',
      timestamp: Date.now(),
      text: 'URGENT! Need this IMMEDIATELY! Very anxious and stressed!!!',
      text_length: 60,
      word_count: 8,
      session_id: 'session_1',
    };

    profile = EmotionalCollector.analyzeAndUpdate(profile, interaction);

    expect(profile.baseline.arousal).toBeGreaterThan(0.6); // High arousal
    expect(profile.samples_analyzed).toBe(1);
  });

  test('should analyze low dominance (submission)', () => {
    let profile = EmotionalCollector.createProfile('alice');

    const interaction: Interaction = {
      interaction_id: 'test_4',
      user_id: 'alice',
      timestamp: Date.now(),
      text: 'Maybe? I guess so... Not sure. Please help. Sorry.',
      text_length: 50,
      word_count: 10,
      session_id: 'session_1',
    };

    profile = EmotionalCollector.analyzeAndUpdate(profile, interaction);

    expect(profile.baseline.dominance).toBeLessThan(0.5); // Low dominance
    expect(profile.samples_analyzed).toBe(1);
  });

  // ==========================================================================
  // EMOTION MARKERS
  // ==========================================================================

  test('should detect joy markers', () => {
    let profile = EmotionalCollector.createProfile('alice');

    const interaction: Interaction = {
      interaction_id: 'test_5',
      user_id: 'alice',
      timestamp: Date.now(),
      text: 'Haha! I am so happy :) This is awesome!',
      text_length: 40,
      word_count: 8,
      session_id: 'session_1',
    };

    profile = EmotionalCollector.analyzeAndUpdate(profile, interaction);

    expect(profile.markers.joy_markers.length).toBeGreaterThan(0);
    expect(profile.markers.joy_markers).toContain('haha');
    expect(profile.markers.joy_markers).toContain('happy');
  });

  test('should detect fear markers', () => {
    let profile = EmotionalCollector.createProfile('alice');

    const interaction: Interaction = {
      interaction_id: 'test_6',
      user_id: 'alice',
      timestamp: Date.now(),
      text: 'I am afraid and worried. Very anxious about this.',
      text_length: 50,
      word_count: 10,
      session_id: 'session_1',
    };

    profile = EmotionalCollector.analyzeAndUpdate(profile, interaction);

    expect(profile.markers.fear_markers.length).toBeGreaterThan(0);
    expect(profile.markers.fear_markers).toContain('afraid');
    expect(profile.markers.fear_markers).toContain('worried');
  });

  test('should detect anger markers', () => {
    let profile = EmotionalCollector.createProfile('alice');

    const interaction: Interaction = {
      interaction_id: 'test_7',
      user_id: 'alice',
      timestamp: Date.now(),
      text: 'I am so angry and frustrated! This is annoying!',
      text_length: 50,
      word_count: 9,
      session_id: 'session_1',
    };

    profile = EmotionalCollector.analyzeAndUpdate(profile, interaction);

    expect(profile.markers.anger_markers.length).toBeGreaterThan(0);
    expect(profile.markers.anger_markers).toContain('angry');
    expect(profile.markers.anger_markers).toContain('frustrated');
  });

  test('should detect sadness markers', () => {
    let profile = EmotionalCollector.createProfile('alice');

    const interaction: Interaction = {
      interaction_id: 'test_8',
      user_id: 'alice',
      timestamp: Date.now(),
      text: 'I am sad and disappointed :( Feeling down.',
      text_length: 45,
      word_count: 8,
      session_id: 'session_1',
    };

    profile = EmotionalCollector.analyzeAndUpdate(profile, interaction);

    expect(profile.markers.sadness_markers.length).toBeGreaterThan(0);
    expect(profile.markers.sadness_markers).toContain('sad');
    expect(profile.markers.sadness_markers).toContain('disappointed');
  });

  // ==========================================================================
  // BASELINE BUILDING
  // ==========================================================================

  test('should build baseline from multiple samples', () => {
    let profile = EmotionalCollector.createProfile('alice');

    // Simulate 50 normal interactions (consistent positive, mid arousal/dominance)
    for (let i = 0; i < 50; i++) {
      const interaction: Interaction = {
        interaction_id: `baseline_${i}`,
        user_id: 'alice',
        timestamp: Date.now(),
        text: 'I am doing well today. Everything is good.',
        text_length: 42,
        word_count: 8,
        session_id: 'session_baseline',
      };

      profile = EmotionalCollector.analyzeAndUpdate(profile, interaction);
    }

    expect(profile.samples_analyzed).toBe(50);
    expect(profile.confidence).toBe(0.5); // 50/100 = 50%
    expect(profile.baseline.valence).toBeGreaterThan(0); // Should be positive
    expect(profile.baseline.arousal).toBeGreaterThan(0.3); // Mid arousal
    expect(profile.baseline.arousal).toBeLessThan(0.7);
  });

  test('should increase confidence with more samples', () => {
    let profile = EmotionalCollector.createProfile('alice');

    expect(profile.confidence).toBe(0);

    // Add 10 samples
    for (let i = 0; i < 10; i++) {
      profile = EmotionalCollector.analyzeAndUpdate(profile, {
        interaction_id: `test_${i}`,
        user_id: 'alice',
        timestamp: Date.now(),
        text: 'Normal text here.',
        text_length: 17,
        word_count: 3,
        session_id: 'session_1',
      });
    }

    expect(profile.confidence).toBe(0.1); // 10/100 = 10%

    // Add 90 more samples
    for (let i = 10; i < 100; i++) {
      profile = EmotionalCollector.analyzeAndUpdate(profile, {
        interaction_id: `test_${i}`,
        user_id: 'alice',
        timestamp: Date.now(),
        text: 'Normal text here.',
        text_length: 17,
        word_count: 3,
        session_id: 'session_1',
      });
    }

    expect(profile.confidence).toBe(1.0); // 100/100 = 100%
  });

  // ==========================================================================
  // SERIALIZATION
  // ==========================================================================

  test('should serialize and deserialize profile', () => {
    let profile = EmotionalCollector.createProfile('alice');

    // Build some baseline
    for (let i = 0; i < 10; i++) {
      profile = EmotionalCollector.analyzeAndUpdate(profile, {
        interaction_id: `test_${i}`,
        user_id: 'alice',
        timestamp: Date.now(),
        text: 'I am happy :)',
        text_length: 12,
        word_count: 3,
        session_id: 'session_1',
      });
    }

    const json = EmotionalCollector.toJSON(profile);
    const restored = EmotionalCollector.fromJSON(json);

    expect(restored.user_id).toBe(profile.user_id);
    expect(restored.samples_analyzed).toBe(profile.samples_analyzed);
    expect(restored.confidence).toBe(profile.confidence);
    expect(restored.baseline.valence).toBe(profile.baseline.valence);
    expect(restored.markers.joy_markers.length).toBe(profile.markers.joy_markers.length);
  });
});

// =============================================================================
// EMOTIONAL ANOMALY DETECTOR
// =============================================================================

describe('EmotionalAnomalyDetector', () => {
  // ==========================================================================
  // NORMAL DETECTION
  // ==========================================================================

  test('should NOT detect anomaly for normal emotional state', () => {
    // Build baseline (positive, mid arousal/dominance)
    let profile = EmotionalCollector.createProfile('alice');

    for (let i = 0; i < 50; i++) {
      profile = EmotionalCollector.analyzeAndUpdate(profile, {
        interaction_id: `baseline_${i}`,
        user_id: 'alice',
        timestamp: Date.now(),
        text: 'I am doing well. Everything is good.',
        text_length: 37,
        word_count: 7,
        session_id: 'session_baseline',
      });
    }

    // Test normal interaction
    const testInteraction: Interaction = {
      interaction_id: 'test_normal',
      user_id: 'alice',
      timestamp: Date.now(),
      text: 'Things are going well today.',
      text_length: 28,
      word_count: 5,
      session_id: 'session_test',
    };

    const anomaly = EmotionalAnomalyDetector.detectEmotionalAnomaly(profile, testInteraction);

    expect(anomaly.alert).toBe(false);
    expect(anomaly.score).toBeLessThan(0.7);
  });

  // ==========================================================================
  // ANOMALY DETECTION
  // ==========================================================================

  test('should detect valence anomaly (sudden negativity)', () => {
    // Build positive baseline
    let profile = EmotionalCollector.createProfile('alice');

    for (let i = 0; i < 50; i++) {
      profile = EmotionalCollector.analyzeAndUpdate(profile, {
        interaction_id: `baseline_${i}`,
        user_id: 'alice',
        timestamp: Date.now(),
        text: 'I am happy and excited! This is wonderful!',
        text_length: 43,
        word_count: 8,
        session_id: 'session_baseline',
      });
    }

    // Test sudden negative interaction
    const testInteraction: Interaction = {
      interaction_id: 'test_negative',
      user_id: 'alice',
      timestamp: Date.now(),
      text: 'This is terrible and awful. I am sad and worried.',
      text_length: 50,
      word_count: 10,
      session_id: 'session_test',
    };

    const anomaly = EmotionalAnomalyDetector.detectEmotionalAnomaly(profile, testInteraction);

    expect(anomaly.details.valence_deviation).toBeGreaterThan(0.5);
  });

  test('should detect arousal anomaly (sudden stress)', () => {
    // Build calm baseline
    let profile = EmotionalCollector.createProfile('alice');

    for (let i = 0; i < 50; i++) {
      profile = EmotionalCollector.analyzeAndUpdate(profile, {
        interaction_id: `baseline_${i}`,
        user_id: 'alice',
        timestamp: Date.now(),
        text: 'Everything is calm and peaceful.',
        text_length: 32,
        word_count: 5,
        session_id: 'session_baseline',
      });
    }

    // Test sudden high arousal
    const testInteraction: Interaction = {
      interaction_id: 'test_stressed',
      user_id: 'alice',
      timestamp: Date.now(),
      text: 'URGENT URGENT URGENT!!! Need this NOW!!! Very anxious!!!',
      text_length: 57,
      word_count: 8,
      session_id: 'session_test',
    };

    const anomaly = EmotionalAnomalyDetector.detectEmotionalAnomaly(profile, testInteraction);

    expect(anomaly.details.arousal_deviation).toBeGreaterThan(0.5);
  });

  // ==========================================================================
  // COERCION DETECTION
  // ==========================================================================

  test('should detect coercion from emotional pattern', () => {
    // Build normal baseline (neutral/positive)
    let profile = EmotionalCollector.createProfile('alice');

    for (let i = 0; i < 50; i++) {
      profile = EmotionalCollector.analyzeAndUpdate(profile, {
        interaction_id: `baseline_${i}`,
        user_id: 'alice',
        timestamp: Date.now(),
        text: 'I will do that. No problem.',
        text_length: 28,
        word_count: 6,
        session_id: 'session_baseline',
      });
    }

    // Test coercion pattern: negative + stressed + submissive
    const testInteraction: Interaction = {
      interaction_id: 'test_coercion',
      user_id: 'alice',
      timestamp: Date.now(),
      text: 'I am afraid and worried. Please, I have to do this. Sorry. Very anxious.',
      text_length: 72,
      word_count: 14,
      session_id: 'session_test',
    };

    const coercion = EmotionalAnomalyDetector.detectCoercion(profile, testInteraction);

    expect(coercion.coercion_detected).toBe(true);
    expect(coercion.confidence).toBeGreaterThan(0.6);
    expect(coercion.indicators.length).toBeGreaterThan(0);
  });

  test('should NOT detect coercion for normal assertive interaction', () => {
    // Build baseline
    let profile = EmotionalCollector.createProfile('alice');

    for (let i = 0; i < 50; i++) {
      profile = EmotionalCollector.analyzeAndUpdate(profile, {
        interaction_id: `baseline_${i}`,
        user_id: 'alice',
        timestamp: Date.now(),
        text: 'I will handle this. No worries.',
        text_length: 32,
        word_count: 6,
        session_id: 'session_baseline',
      });
    }

    // Test normal assertive interaction
    const testInteraction: Interaction = {
      interaction_id: 'test_assertive',
      user_id: 'alice',
      timestamp: Date.now(),
      text: 'I will take care of it. Everything is under control.',
      text_length: 53,
      word_count: 11,
      session_id: 'session_test',
    };

    const coercion = EmotionalAnomalyDetector.detectCoercion(profile, testInteraction);

    expect(coercion.coercion_detected).toBe(false);
  });

  // ==========================================================================
  // EDGE CASES
  // ==========================================================================

  test('should handle insufficient baseline data', () => {
    const profile = EmotionalCollector.createProfile('alice');

    // Only 1 sample (confidence very low)
    const testInteraction: Interaction = {
      interaction_id: 'test_1',
      user_id: 'alice',
      timestamp: Date.now(),
      text: 'Test text here.',
      text_length: 15,
      word_count: 3,
      session_id: 'session_test',
    };

    const anomaly = EmotionalAnomalyDetector.detectEmotionalAnomaly(profile, testInteraction);

    expect(anomaly.alert).toBe(false);
    expect(anomaly.specific_anomalies).toContain('Insufficient baseline data - building emotional profile');
  });

  test('should handle empty text', () => {
    let profile = EmotionalCollector.createProfile('alice');

    const interaction: Interaction = {
      interaction_id: 'test_empty',
      user_id: 'alice',
      timestamp: Date.now(),
      text: '',
      text_length: 0,
      word_count: 0,
      session_id: 'session_test',
    };

    profile = EmotionalCollector.analyzeAndUpdate(profile, interaction);

    expect(profile.samples_analyzed).toBe(1);
    expect(profile.baseline.valence).toBeDefined();
  });
});
