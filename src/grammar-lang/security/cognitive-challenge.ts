/**
 * Multi-Factor Cognitive Authentication
 *
 * Verification based on personal knowledge, preferences, and memories
 * More secure than traditional security questions (not publicly discoverable)
 */

import { createHash } from 'crypto';

// =============================================================================
// TYPES
// =============================================================================

export type ChallengeType = 'personal_fact' | 'preference' | 'memory' | 'reasoning';

export interface CognitiveChallenge {
  // Identity
  challenge_id: string;
  user_id: string;

  // Challenge details
  type: ChallengeType;
  question: string;
  expected_answer_hash: string; // Never store plaintext answer

  // Matching configuration
  fuzzy_match: boolean; // Allow semantic similarity
  confidence_threshold: number; // Minimum similarity for fuzzy match (0-1)

  // Metadata
  difficulty: number; // 0-1 (0 = easy, 1 = hard)
  created_at: number;
  last_used_at?: number;
  use_count: number;

  // Context (for memory-based challenges)
  context?: {
    interaction_date?: string;
    topic?: string;
    related_entities?: string[];
  };
}

export interface VerificationResult {
  verified: boolean;
  confidence: number; // 0-1
  method: 'exact_match' | 'fuzzy_match' | 'failed';
  reason?: string;
}

export interface ChallengeSet {
  user_id: string;
  challenges: CognitiveChallenge[];
  created_at: number;
  last_updated: number;
}

// =============================================================================
// COGNITIVE CHALLENGE MANAGER
// =============================================================================

export class CognitiveChallenge {
  /**
   * Create a new cognitive challenge
   */
  static create(
    userId: string,
    type: ChallengeType,
    question: string,
    expectedAnswer: string,
    options: {
      fuzzy_match?: boolean;
      confidence_threshold?: number;
      difficulty?: number;
      context?: CognitiveChallenge['context'];
    } = {}
  ): CognitiveChallenge {
    const answerHash = this.hashAnswer(expectedAnswer);

    return {
      challenge_id: this.generateChallengeId(),
      user_id: userId,
      type,
      question,
      expected_answer_hash: answerHash,
      fuzzy_match: options.fuzzy_match ?? true,
      confidence_threshold: options.confidence_threshold ?? 0.7,
      difficulty: options.difficulty ?? 0.5,
      created_at: Date.now(),
      use_count: 0,
      context: options.context,
    };
  }

  /**
   * Verify an answer to a cognitive challenge
   */
  static verify(
    challenge: CognitiveChallenge,
    answer: string,
    episodicMemory?: any[] // Optional episodic memory for semantic matching
  ): VerificationResult {
    // Normalize and hash the answer
    const normalizedAnswer = this.normalizeAnswer(answer);
    const answerHash = this.hashAnswer(normalizedAnswer);

    // Try exact match first
    const exactMatch = answerHash === challenge.expected_answer_hash;

    if (exactMatch) {
      return {
        verified: true,
        confidence: 1.0,
        method: 'exact_match',
      };
    }

    // If fuzzy match is allowed, try semantic similarity
    if (challenge.fuzzy_match) {
      const similarity = this.calculateSemanticSimilarity(
        normalizedAnswer,
        challenge,
        episodicMemory
      );

      if (similarity >= challenge.confidence_threshold) {
        return {
          verified: true,
          confidence: similarity,
          method: 'fuzzy_match',
          reason: `Semantic similarity: ${(similarity * 100).toFixed(0)}%`,
        };
      }

      return {
        verified: false,
        confidence: similarity,
        method: 'failed',
        reason: `Insufficient similarity: ${(similarity * 100).toFixed(0)}% < ${(challenge.confidence_threshold * 100).toFixed(0)}%`,
      };
    }

    // Fuzzy match not allowed, exact match failed
    return {
      verified: false,
      confidence: 0.0,
      method: 'failed',
      reason: 'Answer does not match expected answer',
    };
  }

  /**
   * Generate a set of cognitive challenges for a user
   * Typically used during account setup or when updating security
   */
  static generateChallengeSet(
    userId: string,
    interactions?: any[] // User's past interactions for memory-based challenges
  ): ChallengeSet {
    const challenges: CognitiveChallenge[] = [];

    // Generate diverse challenge types
    // In production, these would be generated from actual user data

    // Personal fact (from profile/conversations)
    challenges.push(
      this.create(userId, 'personal_fact', 'What is your favorite hobby?', '', {
        fuzzy_match: true,
        confidence_threshold: 0.7,
        difficulty: 0.3,
      })
    );

    // Preference (from observed patterns)
    challenges.push(
      this.create(userId, 'preference', 'Do you prefer coffee or tea in the morning?', '', {
        fuzzy_match: true,
        confidence_threshold: 0.8,
        difficulty: 0.2,
      })
    );

    // Memory (from episodic memory)
    if (interactions && interactions.length > 0) {
      challenges.push(
        this.create(
          userId,
          'memory',
          'What topic did we discuss in our last conversation?',
          '',
          {
            fuzzy_match: true,
            confidence_threshold: 0.6,
            difficulty: 0.5,
            context: {
              interaction_date: new Date().toISOString().split('T')[0],
              topic: 'to_be_filled',
            },
          }
        )
      );
    }

    // Reasoning (protocol-based)
    challenges.push(
      this.create(
        userId,
        'reasoning',
        'If you were under duress, what would be your safe word to alert me?',
        '',
        {
          fuzzy_match: false, // Must be exact for safety
          confidence_threshold: 1.0,
          difficulty: 0.8,
        }
      )
    );

    return {
      user_id: userId,
      challenges,
      created_at: Date.now(),
      last_updated: Date.now(),
    };
  }

  /**
   * Select appropriate challenge(s) for verification
   * Returns challenges based on context (e.g., sensitive operation = harder challenges)
   */
  static selectChallenges(
    challengeSet: ChallengeSet,
    context: {
      is_sensitive_operation?: boolean;
      duress_score?: number;
      min_difficulty?: number;
      count?: number;
    }
  ): CognitiveChallenge[] {
    const { is_sensitive_operation = false, duress_score = 0, min_difficulty = 0, count = 1 } = context;

    let filteredChallenges = [...challengeSet.challenges];

    // If sensitive operation or high duress score, use harder challenges
    if (is_sensitive_operation || duress_score > 0.6) {
      filteredChallenges = filteredChallenges.filter((c) => c.difficulty >= 0.5);
    } else if (min_difficulty > 0) {
      filteredChallenges = filteredChallenges.filter((c) => c.difficulty >= min_difficulty);
    }

    // Sort by least recently used
    filteredChallenges.sort((a, b) => {
      const aLastUsed = a.last_used_at ?? 0;
      const bLastUsed = b.last_used_at ?? 0;
      return aLastUsed - bLastUsed;
    });

    // Return requested number of challenges
    return filteredChallenges.slice(0, count);
  }

  /**
   * Update challenge after use
   */
  static markChallengeUsed(challenge: CognitiveChallenge): CognitiveChallenge {
    return {
      ...challenge,
      last_used_at: Date.now(),
      use_count: challenge.use_count + 1,
    };
  }

  // ===========================================================================
  // HELPER METHODS
  // ===========================================================================

  /**
   * Generate unique challenge ID
   */
  private static generateChallengeId(): string {
    return `challenge_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
  }

  /**
   * Hash an answer (one-way, secure)
   */
  private static hashAnswer(answer: string): string {
    return createHash('sha256').update(answer.toLowerCase().trim()).digest('hex');
  }

  /**
   * Normalize answer (lowercase, trim, remove extra spaces)
   */
  private static normalizeAnswer(answer: string): string {
    return answer.toLowerCase().trim().replace(/\s+/g, ' ');
  }

  /**
   * Calculate semantic similarity between answer and expected answer
   * In production, this would use LLM or semantic embedding
   * For now, we use simple keyword matching
   */
  private static calculateSemanticSimilarity(
    answer: string,
    challenge: CognitiveChallenge,
    episodicMemory?: any[]
  ): number {
    // This is a simplified version
    // In production, use LLM or semantic embeddings

    // For memory-based challenges, check episodic memory
    if (challenge.type === 'memory' && episodicMemory) {
      // Check if answer appears in recent episodic memory
      const recentMemory = episodicMemory.slice(-10);
      const answerWords = answer.toLowerCase().split(/\s+/);

      let matchCount = 0;
      for (const memory of recentMemory) {
        const memoryText = JSON.stringify(memory).toLowerCase();
        for (const word of answerWords) {
          if (word.length > 3 && memoryText.includes(word)) {
            matchCount++;
          }
        }
      }

      // Similarity based on word matches
      const similarity = Math.min(matchCount / answerWords.length, 1.0);
      return similarity;
    }

    // For other types, use simple keyword similarity
    // This is a placeholder - in production, use proper semantic similarity
    const answerWords = answer.toLowerCase().split(/\s+/);
    const questionWords = challenge.question.toLowerCase().split(/\s+/);

    // Simple overlap measure
    let overlap = 0;
    for (const word of answerWords) {
      if (questionWords.includes(word) && word.length > 3) {
        overlap++;
      }
    }

    // Basic similarity score
    // In production, this would be much more sophisticated
    return Math.min(overlap / Math.max(answerWords.length, 1), 0.5);
  }

  // ===========================================================================
  // SERIALIZATION
  // ===========================================================================

  /**
   * Convert challenge set to JSON
   */
  static toJSON(challengeSet: ChallengeSet): string {
    return JSON.stringify(challengeSet, null, 2);
  }

  /**
   * Restore challenge set from JSON
   */
  static fromJSON(json: string): ChallengeSet {
    return JSON.parse(json);
  }
}

// =============================================================================
// MULTI-FACTOR COGNITIVE AUTHENTICATOR
// =============================================================================

export class CognitiveAuthenticator {
  /**
   * Perform multi-factor cognitive authentication
   * Returns true if user passes all required challenges
   */
  static authenticate(
    challengeSet: ChallengeSet,
    answers: { challenge_id: string; answer: string }[],
    context: {
      is_sensitive_operation?: boolean;
      duress_score?: number;
      min_challenges?: number;
      min_confidence?: number;
    } = {}
  ): {
    authenticated: boolean;
    confidence: number;
    results: { challenge_id: string; result: VerificationResult }[];
  } {
    const { min_challenges = 1, min_confidence = 0.7 } = context;

    // Verify each answer
    const results: { challenge_id: string; result: VerificationResult }[] = [];

    for (const { challenge_id, answer } of answers) {
      const challenge = challengeSet.challenges.find((c) => c.challenge_id === challenge_id);

      if (!challenge) {
        results.push({
          challenge_id,
          result: {
            verified: false,
            confidence: 0,
            method: 'failed',
            reason: 'Challenge not found',
          },
        });
        continue;
      }

      const result = CognitiveChallenge.verify(challenge, answer);
      results.push({ challenge_id, result });
    }

    // Calculate overall authentication
    const verifiedChallenges = results.filter((r) => r.result.verified);
    const authenticated = verifiedChallenges.length >= min_challenges;

    // Calculate average confidence
    const totalConfidence = verifiedChallenges.reduce((sum, r) => sum + r.result.confidence, 0);
    const avgConfidence = verifiedChallenges.length > 0 ? totalConfidence / verifiedChallenges.length : 0;

    // Check if confidence meets threshold
    const confidenceMet = avgConfidence >= min_confidence;

    return {
      authenticated: authenticated && confidenceMet,
      confidence: avgConfidence,
      results,
    };
  }
}
