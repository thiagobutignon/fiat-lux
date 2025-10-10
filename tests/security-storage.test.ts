/**
 * Security Storage Integration Tests
 *
 * Tests the integration between VERMELHO (Security) and LARANJA (Storage)
 * Covers profile persistence, event logging, and O(1) operations
 */

import { describe, it, expect, beforeEach, afterEach, runTests } from '../src/shared/utils/test-runner';
import * as fs from 'fs';
import * as path from 'path';
import { SecurityStorage } from '../src/grammar-lang/security/security-storage';
import { LinguisticCollector } from '../src/grammar-lang/security/linguistic-collector';
import { TypingCollector } from '../src/grammar-lang/security/typing-collector';
import { EmotionalCollector } from '../src/grammar-lang/security/emotional-collector';
import { TemporalCollector } from '../src/grammar-lang/security/temporal-collector';
import { CognitiveChallenge } from '../src/grammar-lang/security/cognitive-challenge';
import { UserSecurityProfiles, Interaction } from '../src/grammar-lang/security/types';

// Test storage directory (separate from production)
const TEST_STORAGE_DIR = 'test_sqlo_security';

// Global storage instance
let storage: SecurityStorage;

// Helper function
function createTestProfiles(userId: string): UserSecurityProfiles {
  return {
    user_id: userId,
    linguistic: LinguisticCollector.createProfile(userId),
    typing: TypingCollector.createProfile(userId),
    emotional: EmotionalCollector.createProfile(userId),
    temporal: TemporalCollector.createProfile(userId, 'UTC'),
    overall_confidence: 0.5,
    last_interaction: Date.now(),
  };
}

// ===========================================================================
// PROFILE PERSISTENCE TESTS
// ===========================================================================

describe('Profile Persistence', () => {
  beforeEach(() => {
    storage = new SecurityStorage(TEST_STORAGE_DIR);
  });

  afterEach(() => {
    if (fs.existsSync(TEST_STORAGE_DIR)) {
      fs.rmSync(TEST_STORAGE_DIR, { recursive: true, force: true });
    }
  });

  it('should save and load a complete user profile', () => {
    const profiles = createTestProfiles('alice');
    const hash = storage.saveProfile(profiles);

    expect.toBeTruthy(hash);
    expect.toEqual(hash.length, 64); // SHA-256 hash

    const loaded = storage.loadProfile('alice');
    expect.toBeTruthy(loaded);
    expect.toEqual(loaded!.user_id, 'alice');
    expect.toEqual(loaded!.overall_confidence, profiles.overall_confidence);
  });

  it('should preserve Map and Set data structures', () => {
    const linguistic = LinguisticCollector.createProfile('bob');
    const interaction: Interaction = {
      interaction_id: 'test_1',
      user_id: 'bob',
      timestamp: Date.now(),
      text: 'Testing vocabulary preservation.',
      text_length: 30,
      word_count: 3,
      session_id: 'session_test',
    };

    const updatedLinguistic = LinguisticCollector.analyzeAndUpdate(linguistic, interaction);
    const profiles: UserSecurityProfiles = {
      user_id: 'bob',
      linguistic: updatedLinguistic,
      typing: TypingCollector.createProfile('bob'),
      emotional: EmotionalCollector.createProfile('bob'),
      temporal: TemporalCollector.createProfile('bob', 'UTC'),
      overall_confidence: 0.3,
      last_interaction: Date.now(),
    };

    storage.saveProfile(profiles);
    const loaded = storage.loadProfile('bob');

    expect.toBeTruthy(loaded!.linguistic.vocabulary.distribution instanceof Map);
    expect.toBeGreaterThan(loaded!.linguistic.vocabulary.unique_words.size, 0);
  });

  it('should update a profile incrementally', () => {
    const profiles = createTestProfiles('alice');
    storage.saveProfile(profiles);

    const success = storage.updateProfile('alice', {
      overall_confidence: 0.8,
    });

    expect.toEqual(success, true);

    const loaded = storage.loadProfile('alice');
    expect.toEqual(loaded!.overall_confidence, 0.8);
  });

  it('should delete a profile', () => {
    const profiles = createTestProfiles('alice');
    storage.saveProfile(profiles);

    const success = storage.deleteProfile('alice');
    expect.toEqual(success, true);

    const loaded = storage.loadProfile('alice');
    expect.toEqual(loaded, null);
  });
});

// ===========================================================================
// COGNITIVE CHALLENGES TESTS
// ===========================================================================

describe('Cognitive Challenges', () => {
  beforeEach(() => {
    storage = new SecurityStorage(TEST_STORAGE_DIR);
  });

  afterEach(() => {
    if (fs.existsSync(TEST_STORAGE_DIR)) {
      fs.rmSync(TEST_STORAGE_DIR, { recursive: true, force: true });
    }
  });

  it('should save and load cognitive challenges', () => {
    // First save a profile (required for challenges)
    const profiles = createTestProfiles('alice');
    storage.saveProfile(profiles);

    const challengeSet = CognitiveChallenge.generateChallengeSet('alice');

    storage.saveChallenges('alice', challengeSet);
    const loaded = storage.loadChallenges('alice');

    expect.toBeTruthy(loaded);
    expect.toEqual(loaded!.user_id, 'alice');
    expect.toEqual(loaded!.challenges.length, challengeSet.challenges.length);
  });
});

// ===========================================================================
// SECURITY EVENTS TESTS
// ===========================================================================

describe('Security Events', () => {
  beforeEach(() => {
    storage = new SecurityStorage(TEST_STORAGE_DIR);
  });

  afterEach(() => {
    if (fs.existsSync(TEST_STORAGE_DIR)) {
      fs.rmSync(TEST_STORAGE_DIR, { recursive: true, force: true });
    }
  });

  it('should log a security event', () => {
    const eventHash = storage.logEvent({
      user_id: 'alice',
      timestamp: Date.now(),
      event_type: 'duress_detected',
      duress_score: 0.8,
      confidence: 0.85,
      decision: 'block',
      reason: 'High duress score detected',
    });

    expect.toBeTruthy(eventHash);
    expect.toEqual(eventHash.length, 64);
  });

  it('should retrieve user events', () => {
    const now = Date.now();

    storage.logEvent({
      user_id: 'alice',
      timestamp: now - 3000,
      event_type: 'duress_detected',
      confidence: 0.8,
      decision: 'challenge',
      reason: 'Duress detected',
    });

    storage.logEvent({
      user_id: 'alice',
      timestamp: now - 2000,
      event_type: 'coercion_detected',
      coercion_score: 0.9,
      confidence: 0.85,
      decision: 'block',
      reason: 'Coercion detected',
    });

    const events = storage.getUserEvents('alice', 10);

    expect.toEqual(events.length, 2);
    expect.toEqual(events[0].event_type, 'coercion_detected'); // Newest first
  });

  it('should retrieve recent alerts', () => {
    const now = Date.now();

    storage.logEvent({
      user_id: 'alice',
      timestamp: now - 1000,
      event_type: 'duress_detected',
      confidence: 0.8,
      decision: 'block',
      reason: 'Alert',
    });

    const alerts = storage.getRecentAlerts(24);

    expect.toEqual(alerts.length, 1);
    expect.toEqual(alerts[0].user_id, 'alice');
  });
});

// ===========================================================================
// STATISTICS TESTS
// ===========================================================================

describe('Statistics', () => {
  beforeEach(() => {
    storage = new SecurityStorage(TEST_STORAGE_DIR);
  });

  afterEach(() => {
    if (fs.existsSync(TEST_STORAGE_DIR)) {
      fs.rmSync(TEST_STORAGE_DIR, { recursive: true, force: true });
    }
  });

  it('should track total profiles', () => {
    const stats1 = storage.getStatistics();
    expect.toEqual(stats1.total_profiles, 0);

    storage.saveProfile(createTestProfiles('alice'));
    storage.saveProfile(createTestProfiles('bob'));

    const stats2 = storage.getStatistics();
    expect.toEqual(stats2.total_profiles, 2);
  });

  it('should track total events', () => {
    const stats1 = storage.getStatistics();
    expect.toEqual(stats1.total_events, 0);

    storage.logEvent({
      user_id: 'alice',
      timestamp: Date.now(),
      event_type: 'profile_updated',
      confidence: 0.5,
      decision: 'allow',
      reason: 'Test',
    });

    const stats2 = storage.getStatistics();
    expect.toEqual(stats2.total_events, 1);
  });
});

// Run tests
runTests();
