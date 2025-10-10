/**
 * Temporal Patterns - Test Suite
 *
 * Tests for TemporalCollector and TemporalAnomalyDetector
 */

import { describe, test, expect } from '@jest/globals';
import { TemporalCollector } from '../temporal-collector';
import { TemporalAnomalyDetector } from '../temporal-anomaly-detector';
import { Interaction } from '../types';

describe('TemporalCollector', () => {
  // ==========================================================================
  // PROFILE CREATION
  // ==========================================================================

  test('should create temporal profile', () => {
    const profile = TemporalCollector.createProfile('alice', 'America/New_York');

    expect(profile.user_id).toBe('alice');
    expect(profile.timezone).toBe('America/New_York');
    expect(profile.samples_analyzed).toBe(0);
    expect(profile.confidence).toBe(0);
    expect(profile.hourly.typical_hours.size).toBe(0);
    expect(profile.daily.typical_days.size).toBe(0);
  });

  // ==========================================================================
  // HOURLY PATTERNS
  // ==========================================================================

  test('should analyze hourly patterns', () => {
    let profile = TemporalCollector.createProfile('alice');

    // Simulate interactions at 9am, 10am, 11am (work hours)
    for (let i = 0; i < 30; i++) {
      const hour = 9 + (i % 3); // Rotate between 9, 10, 11
      const timestamp = new Date(2025, 0, 1, hour, 0, 0).getTime();

      const interaction: Interaction = {
        interaction_id: `test_${i}`,
        user_id: 'alice',
        timestamp,
        text: 'Working on task',
        text_length: 15,
        word_count: 3,
        session_id: 'session_1',
      };

      profile = TemporalCollector.analyzeAndUpdate(profile, interaction, 30);
    }

    expect(profile.samples_analyzed).toBe(30);
    expect(profile.hourly.hour_distribution.get(9)).toBe(10);
    expect(profile.hourly.hour_distribution.get(10)).toBe(10);
    expect(profile.hourly.hour_distribution.get(11)).toBe(10);
    expect(profile.hourly.typical_hours.size).toBeGreaterThan(0);
  });

  test('should identify typical hours', () => {
    let profile = TemporalCollector.createProfile('alice');

    // 50 interactions at 10am, 5 interactions at 3am
    for (let i = 0; i < 50; i++) {
      const hour = 10;
      const timestamp = new Date(2025, 0, 1, hour, 0, 0).getTime();

      profile = TemporalCollector.analyzeAndUpdate(profile, {
        interaction_id: `test_10am_${i}`,
        user_id: 'alice',
        timestamp,
        text: 'Work',
        text_length: 4,
        word_count: 1,
        session_id: 'session_1',
      });
    }

    for (let i = 0; i < 5; i++) {
      const hour = 3;
      const timestamp = new Date(2025, 0, 1, hour, 0, 0).getTime();

      profile = TemporalCollector.analyzeAndUpdate(profile, {
        interaction_id: `test_3am_${i}`,
        user_id: 'alice',
        timestamp,
        text: 'Work',
        text_length: 4,
        word_count: 1,
        session_id: 'session_1',
      });
    }

    // 10am should be typical (50/55 = 91%), 3am should not (5/55 = 9%)
    expect(profile.hourly.typical_hours.has(10)).toBe(true);
    expect(profile.hourly.typical_hours.has(3)).toBe(false);
  });

  // ==========================================================================
  // DAILY PATTERNS
  // ==========================================================================

  test('should analyze daily patterns', () => {
    let profile = TemporalCollector.createProfile('alice');

    // Simulate interactions on Monday (1), Tuesday (2), Wednesday (3)
    for (let i = 0; i < 30; i++) {
      const day = 1 + (i % 3); // Rotate between Mon, Tue, Wed
      const timestamp = new Date(2025, 0, day, 10, 0, 0).getTime(); // Same hour, different days

      const interaction: Interaction = {
        interaction_id: `test_${i}`,
        user_id: 'alice',
        timestamp,
        text: 'Working on task',
        text_length: 15,
        word_count: 3,
        session_id: 'session_1',
      };

      profile = TemporalCollector.analyzeAndUpdate(profile, interaction);
    }

    expect(profile.daily.day_distribution.get(1)).toBe(10); // Monday
    expect(profile.daily.day_distribution.get(2)).toBe(10); // Tuesday
    expect(profile.daily.day_distribution.get(3)).toBe(10); // Wednesday
    expect(profile.daily.typical_days.size).toBeGreaterThan(0);
  });

  // ==========================================================================
  // SESSION PATTERNS
  // ==========================================================================

  test('should track session duration', () => {
    let profile = TemporalCollector.createProfile('alice');

    // Sessions of 30 minutes each
    for (let i = 0; i < 20; i++) {
      const timestamp = Date.now();

      profile = TemporalCollector.analyzeAndUpdate(
        profile,
        {
          interaction_id: `test_${i}`,
          user_id: 'alice',
          timestamp,
          text: 'Work',
          text_length: 4,
          word_count: 1,
          session_id: 'session_1',
        },
        30 // 30 minutes session
      );
    }

    expect(profile.sessions.session_duration_avg).toBeCloseTo(30, 1);
    expect(profile.samples_analyzed).toBe(20);
  });

  test('should calculate session duration variance', () => {
    let profile = TemporalCollector.createProfile('alice');

    // Mix of 20min, 30min, 40min sessions
    const durations = [20, 30, 40];

    for (let i = 0; i < 30; i++) {
      const duration = durations[i % 3];
      const timestamp = Date.now();

      profile = TemporalCollector.analyzeAndUpdate(
        profile,
        {
          interaction_id: `test_${i}`,
          user_id: 'alice',
          timestamp,
          text: 'Work',
          text_length: 4,
          word_count: 1,
          session_id: 'session_1',
        },
        duration
      );
    }

    expect(profile.sessions.session_duration_avg).toBeCloseTo(30, 1);
    expect(profile.sessions.session_duration_variance).toBeGreaterThan(0);
  });

  // ==========================================================================
  // CONFIDENCE BUILDING
  // ==========================================================================

  test('should increase confidence with more samples', () => {
    let profile = TemporalCollector.createProfile('alice');

    expect(profile.confidence).toBe(0);

    // Add 50 samples
    for (let i = 0; i < 50; i++) {
      profile = TemporalCollector.analyzeAndUpdate(profile, {
        interaction_id: `test_${i}`,
        user_id: 'alice',
        timestamp: Date.now(),
        text: 'Work',
        text_length: 4,
        word_count: 1,
        session_id: 'session_1',
      });
    }

    expect(profile.confidence).toBe(0.5); // 50/100 = 50%

    // Add 50 more samples
    for (let i = 50; i < 100; i++) {
      profile = TemporalCollector.analyzeAndUpdate(profile, {
        interaction_id: `test_${i}`,
        user_id: 'alice',
        timestamp: Date.now(),
        text: 'Work',
        text_length: 4,
        word_count: 1,
        session_id: 'session_1',
      });
    }

    expect(profile.confidence).toBe(1.0); // 100/100 = 100%
  });

  // ==========================================================================
  // STATISTICS
  // ==========================================================================

  test('should generate temporal statistics', () => {
    let profile = TemporalCollector.createProfile('alice');

    // Build baseline: 9am-11am on weekdays
    for (let i = 0; i < 50; i++) {
      const hour = 9 + (i % 3);
      const day = 1 + (i % 5); // Mon-Fri
      const timestamp = new Date(2025, 0, day, hour, 0, 0).getTime();

      profile = TemporalCollector.analyzeAndUpdate(profile, {
        interaction_id: `test_${i}`,
        user_id: 'alice',
        timestamp,
        text: 'Work',
        text_length: 4,
        word_count: 1,
        session_id: 'session_1',
      });
    }

    const stats = TemporalCollector.getStatistics(profile);

    expect(stats.most_active_hours.length).toBeGreaterThan(0);
    expect(stats.most_active_days.length).toBeGreaterThan(0);
  });

  // ==========================================================================
  // SERIALIZATION
  // ==========================================================================

  test('should serialize and deserialize profile', () => {
    let profile = TemporalCollector.createProfile('alice', 'America/New_York');

    // Build some baseline
    for (let i = 0; i < 20; i++) {
      const timestamp = new Date(2025, 0, 1, 10, 0, 0).getTime();

      profile = TemporalCollector.analyzeAndUpdate(profile, {
        interaction_id: `test_${i}`,
        user_id: 'alice',
        timestamp,
        text: 'Work',
        text_length: 4,
        word_count: 1,
        session_id: 'session_1',
      });
    }

    const json = TemporalCollector.toJSON(profile);
    const restored = TemporalCollector.fromJSON(json);

    expect(restored.user_id).toBe(profile.user_id);
    expect(restored.timezone).toBe(profile.timezone);
    expect(restored.samples_analyzed).toBe(profile.samples_analyzed);
    expect(restored.confidence).toBe(profile.confidence);
    expect(restored.hourly.typical_hours.size).toBe(profile.hourly.typical_hours.size);
  });
});

// =============================================================================
// TEMPORAL ANOMALY DETECTOR
// =============================================================================

describe('TemporalAnomalyDetector', () => {
  // ==========================================================================
  // NORMAL DETECTION
  // ==========================================================================

  test('should NOT detect anomaly for normal time access', () => {
    // Build baseline: 9am-11am weekdays
    let profile = TemporalCollector.createProfile('alice');

    for (let i = 0; i < 50; i++) {
      const hour = 9 + (i % 3); // 9am, 10am, 11am
      const day = 1; // Monday
      const timestamp = new Date(2025, 0, day, hour, 0, 0).getTime();

      profile = TemporalCollector.analyzeAndUpdate(profile, {
        interaction_id: `baseline_${i}`,
        user_id: 'alice',
        timestamp,
        text: 'Work',
        text_length: 4,
        word_count: 1,
        session_id: 'session_baseline',
      });
    }

    // Test normal access (10am Monday)
    const testTimestamp = new Date(2025, 0, 1, 10, 0, 0).getTime();
    const testInteraction: Interaction = {
      interaction_id: 'test_normal',
      user_id: 'alice',
      timestamp: testTimestamp,
      text: 'Work',
      text_length: 4,
      word_count: 1,
      session_id: 'session_test',
    };

    const anomaly = TemporalAnomalyDetector.detectTemporalAnomaly(profile, testInteraction);

    expect(anomaly.alert).toBe(false);
    expect(anomaly.details.unusual_hour).toBe(false);
  });

  // ==========================================================================
  // ANOMALY DETECTION
  // ==========================================================================

  test('should detect unusual hour anomaly', () => {
    // Build baseline: 9am-11am only
    let profile = TemporalCollector.createProfile('alice');

    for (let i = 0; i < 50; i++) {
      const hour = 9 + (i % 3); // 9am, 10am, 11am
      const timestamp = new Date(2025, 0, 1, hour, 0, 0).getTime();

      profile = TemporalCollector.analyzeAndUpdate(profile, {
        interaction_id: `baseline_${i}`,
        user_id: 'alice',
        timestamp,
        text: 'Work',
        text_length: 4,
        word_count: 1,
        session_id: 'session_baseline',
      });
    }

    // Test access at 3am (unusual)
    const testTimestamp = new Date(2025, 0, 1, 3, 0, 0).getTime();
    const testInteraction: Interaction = {
      interaction_id: 'test_3am',
      user_id: 'alice',
      timestamp: testTimestamp,
      text: 'Work',
      text_length: 4,
      word_count: 1,
      session_id: 'session_test',
    };

    const anomaly = TemporalAnomalyDetector.detectTemporalAnomaly(profile, testInteraction);

    expect(anomaly.details.unusual_hour).toBe(true);
    expect(anomaly.score).toBeGreaterThan(0);
  });

  test('should detect unusual day anomaly', () => {
    // Build baseline: Weekdays only (Mon-Fri)
    let profile = TemporalCollector.createProfile('alice');

    for (let i = 0; i < 50; i++) {
      const day = 1 + (i % 5); // Mon-Fri
      const timestamp = new Date(2025, 0, day, 10, 0, 0).getTime();

      profile = TemporalCollector.analyzeAndUpdate(profile, {
        interaction_id: `baseline_${i}`,
        user_id: 'alice',
        timestamp,
        text: 'Work',
        text_length: 4,
        word_count: 1,
        session_id: 'session_baseline',
      });
    }

    // Test access on Sunday (unusual)
    const testTimestamp = new Date(2025, 0, 7, 10, 0, 0).getTime(); // Sunday
    const testInteraction: Interaction = {
      interaction_id: 'test_sunday',
      user_id: 'alice',
      timestamp: testTimestamp,
      text: 'Work',
      text_length: 4,
      word_count: 1,
      session_id: 'session_test',
    };

    const anomaly = TemporalAnomalyDetector.detectTemporalAnomaly(profile, testInteraction);

    expect(anomaly.details.unusual_day).toBe(true);
    expect(anomaly.score).toBeGreaterThan(0);
  });

  // ==========================================================================
  // IMPERSONATION DETECTION
  // ==========================================================================

  test('should detect impersonation from temporal pattern', () => {
    // Build baseline: 9am-5pm weekdays
    let profile = TemporalCollector.createProfile('alice');

    for (let i = 0; i < 50; i++) {
      const hour = 9 + (i % 8); // 9am-4pm
      const day = 1 + (i % 5); // Mon-Fri
      const timestamp = new Date(2025, 0, day, hour, 0, 0).getTime();

      profile = TemporalCollector.analyzeAndUpdate(profile, {
        interaction_id: `baseline_${i}`,
        user_id: 'alice',
        timestamp,
        text: 'Work',
        text_length: 4,
        word_count: 1,
        session_id: 'session_baseline',
      });
    }

    // Test middle-of-night access (3am Sunday)
    const testTimestamp = new Date(2025, 0, 7, 3, 0, 0).getTime();
    const testInteraction: Interaction = {
      interaction_id: 'test_impersonation',
      user_id: 'alice',
      timestamp: testTimestamp,
      text: 'Work',
      text_length: 4,
      word_count: 1,
      session_id: 'session_test',
    };

    const impersonation = TemporalAnomalyDetector.detectImpersonation(profile, testInteraction);

    expect(impersonation.impersonation_detected).toBe(true);
    expect(impersonation.confidence).toBeGreaterThan(0.6);
    expect(impersonation.indicators.length).toBeGreaterThan(0);
  });

  // ==========================================================================
  // EDGE CASES
  // ==========================================================================

  test('should handle insufficient baseline data', () => {
    const profile = TemporalCollector.createProfile('alice');

    // Only 1 sample (confidence very low)
    const testInteraction: Interaction = {
      interaction_id: 'test_1',
      user_id: 'alice',
      timestamp: Date.now(),
      text: 'Work',
      text_length: 4,
      word_count: 1,
      session_id: 'session_test',
    };

    const anomaly = TemporalAnomalyDetector.detectTemporalAnomaly(profile, testInteraction);

    expect(anomaly.alert).toBe(false);
    expect(anomaly.specific_anomalies).toContain('Insufficient baseline data - building temporal profile');
  });
});
