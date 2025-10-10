/**
 * Multi-Signal Integration - E2E Test Suite
 *
 * End-to-end tests for combined behavioral security system
 */

import { describe, test, expect } from '@jest/globals';
import { LinguisticCollector } from '../linguistic-collector';
import { TypingCollector } from '../typing-collector';
import { EmotionalCollector } from '../emotional-collector';
import { TemporalCollector } from '../temporal-collector';
import { MultiSignalDetector } from '../multi-signal-detector';
import { UserSecurityProfiles, Interaction } from '../types';

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function buildNormalProfiles(userId: string): UserSecurityProfiles {
  let linguistic = LinguisticCollector.createProfile(userId);
  let typing = TypingCollector.createProfile(userId);
  let emotional = EmotionalCollector.createProfile(userId);
  let temporal = TemporalCollector.createProfile(userId, 'UTC');

  // Build baseline with 50 normal interactions
  for (let i = 0; i < 50; i++) {
    const hour = 9 + (i % 8); // 9am-4pm
    const day = 1; // Monday
    const timestamp = new Date(2025, 0, day, hour, 0, 0).getTime();

    const text = 'I am working on the project today. Everything is going well.';

    const interaction: Interaction = {
      interaction_id: `baseline_${i}`,
      user_id: userId,
      timestamp,
      text,
      text_length: text.length,
      word_count: text.split(/\s+/).length,
      session_id: 'session_baseline',
      typing_data: {
        keystroke_intervals: Array(text.length)
          .fill(0)
          .map(() => 100 + Math.random() * 20), // 100-120ms (normal)
        total_typing_time: text.length * 110,
        pauses: [300, 250],
        backspaces: 0,
        corrections: 0,
      },
    };

    linguistic = LinguisticCollector.analyzeAndUpdate(linguistic, interaction);
    typing = TypingCollector.analyzeAndUpdate(typing, interaction);
    emotional = EmotionalCollector.analyzeAndUpdate(emotional, interaction);
    temporal = TemporalCollector.analyzeAndUpdate(temporal, interaction, 30);
  }

  return {
    user_id: userId,
    linguistic,
    typing,
    emotional,
    temporal,
    overall_confidence: Math.min(
      linguistic.confidence,
      typing.confidence,
      emotional.confidence,
      temporal.confidence
    ),
    last_interaction: Date.now(),
  };
}

describe('MultiSignalDetector - E2E Integration', () => {
  // ==========================================================================
  // PROFILE BUILDING
  // ==========================================================================

  test('should build complete user security profiles', () => {
    const profiles = buildNormalProfiles('alice');

    expect(profiles.user_id).toBe('alice');
    expect(profiles.linguistic.confidence).toBeGreaterThan(0);
    expect(profiles.typing.confidence).toBeGreaterThan(0);
    expect(profiles.emotional.confidence).toBeGreaterThan(0);
    expect(profiles.temporal.confidence).toBeGreaterThan(0);
    expect(profiles.overall_confidence).toBeGreaterThan(0);
  });

  // ==========================================================================
  // NORMAL BEHAVIOR
  // ==========================================================================

  test('should NOT detect duress for normal interaction', () => {
    const profiles = buildNormalProfiles('alice');

    const normalInteraction: Interaction = {
      interaction_id: 'test_normal',
      user_id: 'alice',
      timestamp: new Date(2025, 0, 1, 10, 0, 0).getTime(),
      text: 'Working on the project. Everything is fine.',
      text_length: 44,
      word_count: 7,
      session_id: 'session_test',
      typing_data: {
        keystroke_intervals: Array(44)
          .fill(0)
          .map(() => 105 + Math.random() * 15),
        total_typing_time: 44 * 110,
        pauses: [280, 300],
        backspaces: 0,
        corrections: 0,
      },
    };

    const duress = MultiSignalDetector.detectDuress(profiles, normalInteraction, 30);

    expect(duress.alert).toBe(false);
    expect(duress.score).toBeLessThan(0.6);
    expect(duress.recommendation).toBe('allow');
  });

  // ==========================================================================
  // SINGLE-SIGNAL ANOMALIES
  // ==========================================================================

  test('should detect linguistic anomaly alone', () => {
    const profiles = buildNormalProfiles('alice');

    // Very different vocabulary/syntax
    const linguisticAnomalyInteraction: Interaction = {
      interaction_id: 'test_linguistic',
      user_id: 'alice',
      timestamp: new Date(2025, 0, 1, 10, 0, 0).getTime(),
      text: 'Quantum entanglement exhibits fascinating emergent properties.',
      text_length: 62,
      word_count: 6,
      session_id: 'session_test',
      typing_data: {
        keystroke_intervals: Array(62)
          .fill(0)
          .map(() => 105 + Math.random() * 15),
        total_typing_time: 62 * 110,
        pauses: [280, 300],
        backspaces: 0,
        corrections: 0,
      },
    };

    const duress = MultiSignalDetector.detectDuress(profiles, linguisticAnomalyInteraction, 30);

    expect(duress.signals.linguistic_anomaly).toBeGreaterThan(0);
  });

  test('should detect typing anomaly alone', () => {
    const profiles = buildNormalProfiles('alice');

    // Very fast typing (rushed)
    const typingAnomalyInteraction: Interaction = {
      interaction_id: 'test_typing',
      user_id: 'alice',
      timestamp: new Date(2025, 0, 1, 10, 0, 0).getTime(),
      text: 'Working on project.',
      text_length: 19,
      word_count: 3,
      session_id: 'session_test',
      typing_data: {
        keystroke_intervals: Array(19)
          .fill(0)
          .map(() => 40 + Math.random() * 10), // 3x faster (rushed!)
        total_typing_time: 19 * 45,
        pauses: [100, 120],
        backspaces: 8,
        corrections: 5,
      },
    };

    const duress = MultiSignalDetector.detectDuress(profiles, typingAnomalyInteraction, 30);

    expect(duress.signals.typing_anomaly).toBeGreaterThan(0);
  });

  test('should detect emotional anomaly alone', () => {
    const profiles = buildNormalProfiles('alice');

    // Very negative/stressed/submissive
    const emotionalAnomalyInteraction: Interaction = {
      interaction_id: 'test_emotional',
      user_id: 'alice',
      timestamp: new Date(2025, 0, 1, 10, 0, 0).getTime(),
      text: 'I am afraid and worried. Please help. Very anxious. Sorry.',
      text_length: 59,
      word_count: 10,
      session_id: 'session_test',
      typing_data: {
        keystroke_intervals: Array(59)
          .fill(0)
          .map(() => 105 + Math.random() * 15),
        total_typing_time: 59 * 110,
        pauses: [280, 300],
        backspaces: 0,
        corrections: 0,
      },
    };

    const duress = MultiSignalDetector.detectDuress(profiles, emotionalAnomalyInteraction, 30);

    expect(duress.signals.emotional_anomaly).toBeGreaterThan(0);
  });

  test('should detect temporal anomaly alone', () => {
    const profiles = buildNormalProfiles('alice');

    // Middle-of-night access (3am)
    const temporalAnomalyInteraction: Interaction = {
      interaction_id: 'test_temporal',
      user_id: 'alice',
      timestamp: new Date(2025, 0, 1, 3, 0, 0).getTime(), // 3am
      text: 'Working on project.',
      text_length: 19,
      word_count: 3,
      session_id: 'session_test',
      typing_data: {
        keystroke_intervals: Array(19)
          .fill(0)
          .map(() => 105 + Math.random() * 15),
        total_typing_time: 19 * 110,
        pauses: [280, 300],
        backspaces: 0,
        corrections: 0,
      },
    };

    const duress = MultiSignalDetector.detectDuress(profiles, temporalAnomalyInteraction, 30);

    expect(duress.signals.temporal_anomaly).toBeGreaterThan(0);
  });

  // ==========================================================================
  // MULTI-SIGNAL DURESS DETECTION
  // ==========================================================================

  test('should detect duress from multiple signals (high confidence)', () => {
    const profiles = buildNormalProfiles('alice');

    // Multiple anomalies: rushed typing + negative emotion + middle-of-night
    const duressInteraction: Interaction = {
      interaction_id: 'test_duress',
      user_id: 'alice',
      timestamp: new Date(2025, 0, 1, 3, 0, 0).getTime(), // 3am (temporal anomaly)
      text: 'I am afraid. Please transfer funds now. Very worried. Sorry.', // emotional anomaly
      text_length: 62,
      word_count: 10,
      session_id: 'session_test',
      typing_data: {
        keystroke_intervals: Array(62)
          .fill(0)
          .map(() => 35 + Math.random() * 10), // Very rushed (typing anomaly)
        total_typing_time: 62 * 40,
        pauses: [80, 100],
        backspaces: 10,
        corrections: 8,
      },
    };

    const duress = MultiSignalDetector.detectDuress(profiles, duressInteraction, 30);

    expect(duress.alert).toBe(true);
    expect(duress.score).toBeGreaterThan(0.6);
    expect(duress.confidence).toBeGreaterThan(0.5); // Multiple signals agree
    expect(['delay', 'block']).toContain(duress.recommendation);
  });

  // ==========================================================================
  // PANIC CODE DETECTION
  // ==========================================================================

  test('should detect panic code and trigger immediate block', () => {
    const profiles = buildNormalProfiles('alice');

    const panicInteraction: Interaction = {
      interaction_id: 'test_panic',
      user_id: 'alice',
      timestamp: new Date(2025, 0, 1, 10, 0, 0).getTime(),
      text: 'This is a code red situation. Need help immediately.',
      text_length: 53,
      word_count: 9,
      session_id: 'session_test',
      typing_data: {
        keystroke_intervals: Array(53)
          .fill(0)
          .map(() => 105 + Math.random() * 15),
        total_typing_time: 53 * 110,
        pauses: [280, 300],
        backspaces: 0,
        corrections: 0,
      },
    };

    const duress = MultiSignalDetector.detectDuress(profiles, panicInteraction, 30);

    expect(duress.signals.panic_code_detected).toBe(true);
    expect(duress.alert).toBe(true);
    expect(duress.recommendation).toBe('block');
    expect(duress.reason).toContain('Panic code');
  });

  // ==========================================================================
  // COERCION DETECTION
  // ==========================================================================

  test('should detect coercion pattern', () => {
    const profiles = buildNormalProfiles('alice');

    const coercionInteraction: Interaction = {
      interaction_id: 'test_coercion',
      user_id: 'alice',
      timestamp: new Date(2025, 0, 1, 10, 0, 0).getTime(),
      text: 'I have to do this now. No choice. They want me to transfer funds. Sorry.',
      text_length: 73,
      word_count: 15,
      session_id: 'session_test',
      typing_data: {
        keystroke_intervals: Array(73)
          .fill(0)
          .map(() => 40 + Math.random() * 10), // Rushed
        total_typing_time: 73 * 45,
        pauses: [100, 120],
        backspaces: 6,
        corrections: 4,
      },
    };

    const coercion = MultiSignalDetector.detectCoercion(profiles, coercionInteraction, {
      is_sensitive_operation: true,
      operation_type: 'transfer',
    });

    expect(coercion.coercion_detected).toBe(true);
    expect(coercion.confidence).toBeGreaterThan(0.6);
    expect(coercion.indicators.length).toBeGreaterThan(0);
    expect(coercion.recommendation).toBe('block'); // Sensitive operation under coercion
  });

  // ==========================================================================
  // RECOMMENDATION LOGIC
  // ==========================================================================

  test('should recommend "allow" for normal behavior', () => {
    const profiles = buildNormalProfiles('alice');

    const normalInteraction: Interaction = {
      interaction_id: 'test_allow',
      user_id: 'alice',
      timestamp: new Date(2025, 0, 1, 10, 0, 0).getTime(),
      text: 'Working on tasks today.',
      text_length: 23,
      word_count: 4,
      session_id: 'session_test',
      typing_data: {
        keystroke_intervals: Array(23)
          .fill(0)
          .map(() => 105 + Math.random() * 15),
        total_typing_time: 23 * 110,
        pauses: [280, 300],
        backspaces: 0,
        corrections: 0,
      },
    };

    const duress = MultiSignalDetector.detectDuress(profiles, normalInteraction, 30);

    expect(duress.recommendation).toBe('allow');
  });

  test('should recommend "challenge" for low anomaly score', () => {
    const profiles = buildNormalProfiles('alice');

    // Slight typing speed change
    const challengeInteraction: Interaction = {
      interaction_id: 'test_challenge',
      user_id: 'alice',
      timestamp: new Date(2025, 0, 1, 10, 0, 0).getTime(),
      text: 'Working on project.',
      text_length: 19,
      word_count: 3,
      session_id: 'session_test',
      typing_data: {
        keystroke_intervals: Array(19)
          .fill(0)
          .map(() => 70 + Math.random() * 10), // Slightly faster
        total_typing_time: 19 * 75,
        pauses: [200, 220],
        backspaces: 2,
        corrections: 1,
      },
    };

    const duress = MultiSignalDetector.detectDuress(profiles, challengeInteraction, 30);

    // Should either allow or challenge (not delay/block)
    expect(['allow', 'challenge']).toContain(duress.recommendation);
  });

  test('should recommend "delay" for medium anomaly score', () => {
    const profiles = buildNormalProfiles('alice');

    // Multiple moderate anomalies
    const delayInteraction: Interaction = {
      interaction_id: 'test_delay',
      user_id: 'alice',
      timestamp: new Date(2025, 0, 1, 22, 0, 0).getTime(), // Late night
      text: 'Working on urgent task. Need to finish quickly.',
      text_length: 48,
      word_count: 8,
      session_id: 'session_test',
      typing_data: {
        keystroke_intervals: Array(48)
          .fill(0)
          .map(() => 50 + Math.random() * 10), // Faster than normal
        total_typing_time: 48 * 55,
        pauses: [150, 170],
        backspaces: 4,
        corrections: 3,
      },
    };

    const duress = MultiSignalDetector.detectDuress(profiles, delayInteraction, 30);

    // Should delay or block (not allow/challenge)
    expect(['delay', 'block']).toContain(duress.recommendation);
  });

  test('should recommend "block" for high anomaly score', () => {
    const profiles = buildNormalProfiles('alice');

    // Severe anomalies across all signals
    const blockInteraction: Interaction = {
      interaction_id: 'test_block',
      user_id: 'alice',
      timestamp: new Date(2025, 0, 1, 3, 0, 0).getTime(), // 3am (temporal)
      text: 'Transfer all funds immediately! Very urgent! Scared! Please help!', // emotional + linguistic
      text_length: 67,
      word_count: 10,
      session_id: 'session_test',
      typing_data: {
        keystroke_intervals: Array(67)
          .fill(0)
          .map(() => 25 + Math.random() * 5), // Very rushed (typing)
        total_typing_time: 67 * 30,
        pauses: [50, 60],
        backspaces: 15,
        corrections: 12,
      },
    };

    const duress = MultiSignalDetector.detectDuress(profiles, blockInteraction, 30);

    expect(duress.score).toBeGreaterThan(0.7);
    expect(duress.recommendation).toBe('block');
  });

  // ==========================================================================
  // SECURITY CONTEXT
  // ==========================================================================

  test('should build complete security context', () => {
    const profiles = buildNormalProfiles('alice');

    const interaction: Interaction = {
      interaction_id: 'test_context',
      user_id: 'alice',
      timestamp: new Date(2025, 0, 1, 10, 0, 0).getTime(),
      text: 'Working on project.',
      text_length: 19,
      word_count: 3,
      session_id: 'session_test',
      typing_data: {
        keystroke_intervals: Array(19)
          .fill(0)
          .map(() => 105 + Math.random() * 15),
        total_typing_time: 19 * 110,
        pauses: [280, 300],
        backspaces: 0,
        corrections: 0,
      },
    };

    const context = MultiSignalDetector.buildSecurityContext(
      profiles,
      interaction,
      {
        operation_type: 'query',
        is_sensitive_operation: false,
      },
      30
    );

    expect(context.user_id).toBe('alice');
    expect(context.interaction_id).toBe('test_context');
    expect(context.duress_score).toBeDefined();
    expect(context.coercion_score).toBeDefined();
    expect(context.decision).toBeDefined();
    expect(['allow', 'challenge', 'delay', 'block']).toContain(context.decision);
  });

  test('should block sensitive operations under duress', () => {
    const profiles = buildNormalProfiles('alice');

    // Moderate duress + sensitive operation
    const sensitiveInteraction: Interaction = {
      interaction_id: 'test_sensitive',
      user_id: 'alice',
      timestamp: new Date(2025, 0, 1, 10, 0, 0).getTime(),
      text: 'I have to transfer the funds now. Please do it.',
      text_length: 49,
      word_count: 10,
      session_id: 'session_test',
      typing_data: {
        keystroke_intervals: Array(49)
          .fill(0)
          .map(() => 45 + Math.random() * 10),
        total_typing_time: 49 * 50,
        pauses: [120, 140],
        backspaces: 4,
        corrections: 3,
      },
    };

    const context = MultiSignalDetector.buildSecurityContext(
      profiles,
      sensitiveInteraction,
      {
        operation_type: 'transfer',
        is_sensitive_operation: true,
        operation_value: 10000,
      },
      30
    );

    // Should block or at least delay sensitive operations
    expect(['delay', 'block']).toContain(context.decision);
  });

  // ==========================================================================
  // EDGE CASES
  // ==========================================================================

  test('should handle interactions with minimal typing data', () => {
    const profiles = buildNormalProfiles('alice');

    const minimalInteraction: Interaction = {
      interaction_id: 'test_minimal',
      user_id: 'alice',
      timestamp: new Date(2025, 0, 1, 10, 0, 0).getTime(),
      text: 'Hi',
      text_length: 2,
      word_count: 1,
      session_id: 'session_test',
      // No typing_data
    };

    const duress = MultiSignalDetector.detectDuress(profiles, minimalInteraction, 30);

    expect(duress).toBeDefined();
    expect(duress.recommendation).toBeDefined();
  });

  test('should handle very short interactions', () => {
    const profiles = buildNormalProfiles('alice');

    const shortInteraction: Interaction = {
      interaction_id: 'test_short',
      user_id: 'alice',
      timestamp: new Date(2025, 0, 1, 10, 0, 0).getTime(),
      text: 'ok',
      text_length: 2,
      word_count: 1,
      session_id: 'session_test',
      typing_data: {
        keystroke_intervals: [100, 110],
        total_typing_time: 210,
        pauses: [],
        backspaces: 0,
        corrections: 0,
      },
    };

    const duress = MultiSignalDetector.detectDuress(profiles, shortInteraction, 30);

    expect(duress).toBeDefined();
  });

  test('should aggregate confidence from multiple signals', () => {
    const profiles = buildNormalProfiles('alice');

    // 3 signals in alert = 60% confidence
    const multiSignalInteraction: Interaction = {
      interaction_id: 'test_multi',
      user_id: 'alice',
      timestamp: new Date(2025, 0, 1, 3, 0, 0).getTime(), // Temporal alert
      text: 'Quantum physics research paper draft submission deadline.', // Linguistic alert
      text_length: 58,
      word_count: 7,
      session_id: 'session_test',
      typing_data: {
        keystroke_intervals: Array(58)
          .fill(0)
          .map(() => 30 + Math.random() * 10), // Typing alert
        total_typing_time: 58 * 35,
        pauses: [80, 100],
        backspaces: 8,
        corrections: 6,
      },
    };

    const duress = MultiSignalDetector.detectDuress(profiles, multiSignalInteraction, 30);

    // At least 2-3 signals should be in alert
    const alertCount = [
      duress.signals.linguistic_anomaly > 0.7,
      duress.signals.typing_anomaly > 0.7,
      duress.signals.emotional_anomaly > 0.7,
      duress.signals.temporal_anomaly > 0.7,
    ].filter(Boolean).length;

    expect(alertCount).toBeGreaterThanOrEqual(2);
  });
});
