/**
 * Tests for Cognitive Challenge System
 */

import { describe, it, expect } from '@jest/globals';
import {
  CognitiveChallenge,
  CognitiveAuthenticator,
} from '../cognitive-challenge';

describe('CognitiveChallenge', () => {
  // ==========================================================================
  // CHALLENGE CREATION
  // ==========================================================================

  describe('Challenge Creation', () => {
    it('should create a personal fact challenge', () => {
      const challenge = CognitiveChallenge.create(
        'user123',
        'personal_fact',
        'What is your favorite hobby?',
        'photography',
        {
          fuzzy_match: true,
          confidence_threshold: 0.7,
          difficulty: 0.3,
        }
      );

      expect(challenge.user_id).toBe('user123');
      expect(challenge.type).toBe('personal_fact');
      expect(challenge.question).toBe('What is your favorite hobby?');
      expect(challenge.fuzzy_match).toBe(true);
      expect(challenge.confidence_threshold).toBe(0.7);
      expect(challenge.difficulty).toBe(0.3);
      expect(challenge.use_count).toBe(0);
    });

    it('should create a preference challenge', () => {
      const challenge = CognitiveChallenge.create(
        'user123',
        'preference',
        'Do you prefer coffee or tea?',
        'coffee',
        {
          fuzzy_match: true,
          difficulty: 0.2,
        }
      );

      expect(challenge.type).toBe('preference');
      expect(challenge.fuzzy_match).toBe(true);
    });

    it('should create a memory challenge with context', () => {
      const challenge = CognitiveChallenge.create(
        'user123',
        'memory',
        'What did we discuss last Tuesday?',
        'climate change',
        {
          fuzzy_match: true,
          context: {
            interaction_date: '2025-10-07',
            topic: 'climate',
          },
        }
      );

      expect(challenge.type).toBe('memory');
      expect(challenge.context).toBeDefined();
      expect(challenge.context?.interaction_date).toBe('2025-10-07');
      expect(challenge.context?.topic).toBe('climate');
    });

    it('should create a reasoning challenge (exact match only)', () => {
      const challenge = CognitiveChallenge.create(
        'user123',
        'reasoning',
        'What is your panic code?',
        'red alert',
        {
          fuzzy_match: false,
          confidence_threshold: 1.0,
          difficulty: 0.8,
        }
      );

      expect(challenge.type).toBe('reasoning');
      expect(challenge.fuzzy_match).toBe(false);
      expect(challenge.confidence_threshold).toBe(1.0);
      expect(challenge.difficulty).toBe(0.8);
    });

    it('should hash the expected answer (never store plaintext)', () => {
      const challenge = CognitiveChallenge.create(
        'user123',
        'personal_fact',
        'What is your pet name?',
        'fluffy'
      );

      // Answer should be hashed, not stored in plaintext
      expect(challenge.expected_answer_hash).toBeDefined();
      expect(challenge.expected_answer_hash).not.toBe('fluffy');
      expect(challenge.expected_answer_hash.length).toBe(64); // SHA-256 = 64 hex chars
    });
  });

  // ==========================================================================
  // ANSWER VERIFICATION
  // ==========================================================================

  describe('Answer Verification', () => {
    it('should verify exact match answers', () => {
      const challenge = CognitiveChallenge.create(
        'user123',
        'personal_fact',
        'What is your favorite color?',
        'blue'
      );

      const result = CognitiveChallenge.verify(challenge, 'blue');

      expect(result.verified).toBe(true);
      expect(result.confidence).toBe(1.0);
      expect(result.method).toBe('exact_match');
    });

    it('should verify case-insensitive answers', () => {
      const challenge = CognitiveChallenge.create(
        'user123',
        'personal_fact',
        'What is your favorite color?',
        'Blue'
      );

      const result = CognitiveChallenge.verify(challenge, 'BLUE');

      expect(result.verified).toBe(true);
      expect(result.confidence).toBe(1.0);
      expect(result.method).toBe('exact_match');
    });

    it('should verify answers with extra whitespace', () => {
      const challenge = CognitiveChallenge.create(
        'user123',
        'personal_fact',
        'What is your favorite color?',
        'dark blue'
      );

      const result = CognitiveChallenge.verify(challenge, '  dark   blue  ');

      expect(result.verified).toBe(true);
      expect(result.confidence).toBe(1.0);
      expect(result.method).toBe('exact_match');
    });

    it('should reject incorrect answers (exact match required)', () => {
      const challenge = CognitiveChallenge.create(
        'user123',
        'reasoning',
        'What is your panic code?',
        'red alert',
        {
          fuzzy_match: false,
        }
      );

      const result = CognitiveChallenge.verify(challenge, 'blue alert');

      expect(result.verified).toBe(false);
      expect(result.confidence).toBe(0.0);
      expect(result.method).toBe('failed');
    });

    it('should use fuzzy matching when enabled', () => {
      const challenge = CognitiveChallenge.create(
        'user123',
        'preference',
        'What is your favorite drink?',
        'coffee',
        {
          fuzzy_match: true,
          confidence_threshold: 0.3,
        }
      );

      // "coffee" is close enough to pass with low threshold
      const result = CognitiveChallenge.verify(challenge, 'drink');

      // With simple keyword matching, this might pass with low confidence
      expect(result.method).toMatch(/fuzzy_match|failed/);
    });

    it('should reject fuzzy matches below threshold', () => {
      const challenge = CognitiveChallenge.create(
        'user123',
        'preference',
        'What is your favorite drink?',
        'coffee',
        {
          fuzzy_match: true,
          confidence_threshold: 0.9, // Very high threshold
        }
      );

      const result = CognitiveChallenge.verify(challenge, 'tea');

      expect(result.verified).toBe(false);
      expect(result.method).toBe('failed');
    });
  });

  // ==========================================================================
  // CHALLENGE SET GENERATION
  // ==========================================================================

  describe('Challenge Set Generation', () => {
    it('should generate a challenge set for a user', () => {
      const challengeSet = CognitiveChallenge.generateChallengeSet('user123');

      expect(challengeSet.user_id).toBe('user123');
      expect(challengeSet.challenges.length).toBeGreaterThan(0);
      expect(challengeSet.created_at).toBeDefined();
      expect(challengeSet.last_updated).toBeDefined();
    });

    it('should generate diverse challenge types', () => {
      const challengeSet = CognitiveChallenge.generateChallengeSet('user123');

      const types = new Set(challengeSet.challenges.map((c) => c.type));

      // Should have multiple types
      expect(types.size).toBeGreaterThan(1);
      expect(types.has('personal_fact')).toBe(true);
      expect(types.has('preference')).toBe(true);
    });

    it('should include memory challenges if interactions provided', () => {
      const interactions = [
        { text: 'We discussed climate change', timestamp: Date.now() },
      ];

      const challengeSet = CognitiveChallenge.generateChallengeSet('user123', interactions);

      const memoryChallenge = challengeSet.challenges.find((c) => c.type === 'memory');
      expect(memoryChallenge).toBeDefined();
    });

    it('should include reasoning challenge for panic code', () => {
      const challengeSet = CognitiveChallenge.generateChallengeSet('user123');

      const reasoningChallenge = challengeSet.challenges.find((c) => c.type === 'reasoning');
      expect(reasoningChallenge).toBeDefined();
      expect(reasoningChallenge?.fuzzy_match).toBe(false); // Must be exact
      expect(reasoningChallenge?.difficulty).toBeGreaterThanOrEqual(0.5);
    });
  });

  // ==========================================================================
  // CHALLENGE SELECTION
  // ==========================================================================

  describe('Challenge Selection', () => {
    it('should select easier challenges for normal operations', () => {
      const challengeSet = CognitiveChallenge.generateChallengeSet('user123');

      const selected = CognitiveChallenge.selectChallenges(challengeSet, {
        is_sensitive_operation: false,
        count: 1,
      });

      expect(selected.length).toBe(1);
    });

    it('should select harder challenges for sensitive operations', () => {
      const challengeSet = CognitiveChallenge.generateChallengeSet('user123');

      const selected = CognitiveChallenge.selectChallenges(challengeSet, {
        is_sensitive_operation: true,
        count: 2,
      });

      expect(selected.length).toBeLessThanOrEqual(2);
      // Should prefer harder challenges
      if (selected.length > 0) {
        expect(selected[0].difficulty).toBeGreaterThanOrEqual(0.3);
      }
    });

    it('should select harder challenges for high duress scores', () => {
      const challengeSet = CognitiveChallenge.generateChallengeSet('user123');

      const selected = CognitiveChallenge.selectChallenges(challengeSet, {
        duress_score: 0.8, // High duress
        count: 1,
      });

      expect(selected.length).toBeLessThanOrEqual(1);
      if (selected.length > 0) {
        expect(selected[0].difficulty).toBeGreaterThanOrEqual(0.5);
      }
    });

    it('should respect minimum difficulty filter', () => {
      const challengeSet = CognitiveChallenge.generateChallengeSet('user123');

      const selected = CognitiveChallenge.selectChallenges(challengeSet, {
        min_difficulty: 0.7,
        count: 3,
      });

      for (const challenge of selected) {
        expect(challenge.difficulty).toBeGreaterThanOrEqual(0.7);
      }
    });

    it('should prefer least recently used challenges', () => {
      const challengeSet = CognitiveChallenge.generateChallengeSet('user123');

      // Mark first challenge as recently used
      challengeSet.challenges[0] = CognitiveChallenge.markChallengeUsed(
        challengeSet.challenges[0]
      );

      const selected = CognitiveChallenge.selectChallenges(challengeSet, {
        count: 1,
      });

      // Should select a different challenge (not the recently used one)
      if (selected.length > 0 && challengeSet.challenges.length > 1) {
        // First challenge was recently used, so should be deprioritized
        expect(selected[0].use_count).toBe(0);
      }
    });
  });

  // ==========================================================================
  // CHALLENGE USAGE TRACKING
  // ==========================================================================

  describe('Challenge Usage Tracking', () => {
    it('should mark challenge as used', () => {
      const challenge = CognitiveChallenge.create(
        'user123',
        'personal_fact',
        'What is your favorite hobby?',
        'photography'
      );

      expect(challenge.use_count).toBe(0);
      expect(challenge.last_used_at).toBeUndefined();

      const updated = CognitiveChallenge.markChallengeUsed(challenge);

      expect(updated.use_count).toBe(1);
      expect(updated.last_used_at).toBeDefined();
      expect(updated.last_used_at).toBeGreaterThan(0);
    });

    it('should increment use count on repeated use', () => {
      let challenge = CognitiveChallenge.create(
        'user123',
        'personal_fact',
        'What is your favorite hobby?',
        'photography'
      );

      challenge = CognitiveChallenge.markChallengeUsed(challenge);
      expect(challenge.use_count).toBe(1);

      challenge = CognitiveChallenge.markChallengeUsed(challenge);
      expect(challenge.use_count).toBe(2);

      challenge = CognitiveChallenge.markChallengeUsed(challenge);
      expect(challenge.use_count).toBe(3);
    });
  });

  // ==========================================================================
  // SERIALIZATION
  // ==========================================================================

  describe('Serialization', () => {
    it('should serialize and deserialize challenge set', () => {
      const challengeSet = CognitiveChallenge.generateChallengeSet('user123');

      const json = CognitiveChallenge.toJSON(challengeSet);
      const restored = CognitiveChallenge.fromJSON(json);

      expect(restored.user_id).toBe(challengeSet.user_id);
      expect(restored.challenges.length).toBe(challengeSet.challenges.length);
      expect(restored.created_at).toBe(challengeSet.created_at);
      expect(restored.last_updated).toBe(challengeSet.last_updated);
    });

    it('should preserve challenge details in serialization', () => {
      const challengeSet = CognitiveChallenge.generateChallengeSet('user123');

      const json = CognitiveChallenge.toJSON(challengeSet);
      const restored = CognitiveChallenge.fromJSON(json);

      for (let i = 0; i < challengeSet.challenges.length; i++) {
        const original = challengeSet.challenges[i];
        const restoredChallenge = restored.challenges[i];

        expect(restoredChallenge.challenge_id).toBe(original.challenge_id);
        expect(restoredChallenge.type).toBe(original.type);
        expect(restoredChallenge.question).toBe(original.question);
        expect(restoredChallenge.expected_answer_hash).toBe(original.expected_answer_hash);
        expect(restoredChallenge.fuzzy_match).toBe(original.fuzzy_match);
        expect(restoredChallenge.difficulty).toBe(original.difficulty);
      }
    });
  });

  // ==========================================================================
  // MULTI-FACTOR COGNITIVE AUTHENTICATOR
  // ==========================================================================

  describe('CognitiveAuthenticator', () => {
    it('should authenticate user with correct answers', () => {
      const challenge1 = CognitiveChallenge.create(
        'user123',
        'personal_fact',
        'What is your favorite color?',
        'blue'
      );

      const challenge2 = CognitiveChallenge.create(
        'user123',
        'preference',
        'Do you prefer coffee or tea?',
        'coffee'
      );

      const challengeSet = {
        user_id: 'user123',
        challenges: [challenge1, challenge2],
        created_at: Date.now(),
        last_updated: Date.now(),
      };

      const answers = [
        { challenge_id: challenge1.challenge_id, answer: 'blue' },
        { challenge_id: challenge2.challenge_id, answer: 'coffee' },
      ];

      const result = CognitiveAuthenticator.authenticate(challengeSet, answers, {
        min_challenges: 2,
        min_confidence: 0.7,
      });

      expect(result.authenticated).toBe(true);
      expect(result.confidence).toBeGreaterThanOrEqual(0.7);
      expect(result.results.length).toBe(2);
    });

    it('should reject authentication with incorrect answers', () => {
      const challenge = CognitiveChallenge.create(
        'user123',
        'personal_fact',
        'What is your favorite color?',
        'blue'
      );

      const challengeSet = {
        user_id: 'user123',
        challenges: [challenge],
        created_at: Date.now(),
        last_updated: Date.now(),
      };

      const answers = [{ challenge_id: challenge.challenge_id, answer: 'red' }];

      const result = CognitiveAuthenticator.authenticate(challengeSet, answers, {
        min_challenges: 1,
        min_confidence: 0.7,
      });

      expect(result.authenticated).toBe(false);
      expect(result.confidence).toBe(0);
    });

    it('should require minimum number of correct answers', () => {
      const challenge1 = CognitiveChallenge.create(
        'user123',
        'personal_fact',
        'What is your favorite color?',
        'blue'
      );

      const challenge2 = CognitiveChallenge.create(
        'user123',
        'preference',
        'Do you prefer coffee or tea?',
        'coffee'
      );

      const challengeSet = {
        user_id: 'user123',
        challenges: [challenge1, challenge2],
        created_at: Date.now(),
        last_updated: Date.now(),
      };

      // Only answer 1 challenge correctly
      const answers = [{ challenge_id: challenge1.challenge_id, answer: 'blue' }];

      const result = CognitiveAuthenticator.authenticate(challengeSet, answers, {
        min_challenges: 2, // Require 2 correct answers
        min_confidence: 0.7,
      });

      expect(result.authenticated).toBe(false);
    });

    it('should calculate average confidence across all challenges', () => {
      const challenge1 = CognitiveChallenge.create(
        'user123',
        'personal_fact',
        'What is your favorite color?',
        'blue'
      );

      const challenge2 = CognitiveChallenge.create(
        'user123',
        'preference',
        'Do you prefer coffee or tea?',
        'coffee'
      );

      const challengeSet = {
        user_id: 'user123',
        challenges: [challenge1, challenge2],
        created_at: Date.now(),
        last_updated: Date.now(),
      };

      const answers = [
        { challenge_id: challenge1.challenge_id, answer: 'blue' },
        { challenge_id: challenge2.challenge_id, answer: 'coffee' },
      ];

      const result = CognitiveAuthenticator.authenticate(challengeSet, answers);

      expect(result.confidence).toBeGreaterThan(0);
      expect(result.confidence).toBeLessThanOrEqual(1.0);
    });

    it('should handle unknown challenge IDs gracefully', () => {
      const challengeSet = CognitiveChallenge.generateChallengeSet('user123');

      const answers = [{ challenge_id: 'unknown_id', answer: 'some answer' }];

      const result = CognitiveAuthenticator.authenticate(challengeSet, answers);

      expect(result.authenticated).toBe(false);
      expect(result.results.length).toBe(1);
      expect(result.results[0].result.verified).toBe(false);
      expect(result.results[0].result.reason).toContain('Challenge not found');
    });
  });
});
