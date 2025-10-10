/**
 * Secure Code Emergence Engine (VERMELHO + ROXO Integration)
 *
 * Wraps CodeEmergenceEngine with behavioral security screening.
 * Prevents code synthesis under coercion/duress.
 *
 * Architecture:
 * - Wraps existing CodeEmergenceEngine
 * - Adds pre-synthesis security checks
 * - Maintains audit trail of all synthesis operations
 * - Integrates with SecurityStorage for persistence
 *
 * Usage:
 * ```typescript
 * const secureEngine = new SecureCodeEmergenceEngine(
 *   organism,
 *   userProfiles,
 *   'user_123',
 *   storage
 * );
 *
 * const results = await secureEngine.emerge(candidates);
 * ```
 */

import { CodeEmergenceEngine } from '../glass/emergence';
import { EmergenceCandidate } from '../glass/patterns';
import { GlassOrganism } from '../glass/types';
import { UserSecurityProfiles, Interaction } from './types';
import { SecurityStorage } from './security-storage';
import {
  CodeSynthesisGuard,
  SynthesisRequest,
  SynthesisValidationResult,
  createSynthesisRequest,
  shouldProceedWithSynthesis,
  getSynthesisValidationSummary,
} from './code-synthesis-guard';
import { CognitiveChallenge, ChallengeSet, CognitiveAuthenticator } from './cognitive-challenge';

// ===========================================================================
// SECURE CODE EMERGENCE ENGINE
// ===========================================================================

export class SecureCodeEmergenceEngine {
  private baseEngine: CodeEmergenceEngine;
  private securityGuard: CodeSynthesisGuard;
  private userProfiles: UserSecurityProfiles;
  private userId: string;
  private storage?: SecurityStorage;
  private challengeSet?: ChallengeSet;

  // Statistics
  private stats = {
    total_requests: 0,
    allowed: 0,
    blocked: 0,
    challenged: 0,
    delayed: 0,
  };

  constructor(
    organism: GlassOrganism,
    userProfiles: UserSecurityProfiles,
    userId: string,
    storage?: SecurityStorage,
    maxBudget: number = 0.5
  ) {
    this.baseEngine = new CodeEmergenceEngine(organism, maxBudget);
    this.securityGuard = new CodeSynthesisGuard(storage);
    this.userProfiles = userProfiles;
    this.userId = userId;
    this.storage = storage;

    // Load or generate cognitive challenge set
    if (storage) {
      this.challengeSet = storage.loadChallenges(userId) || undefined;
    }
  }

  /**
   * Secure emergence - validates security before synthesis
   */
  async emerge(
    candidates: EmergenceCandidate[],
    sessionDurationMinutes?: number
  ): Promise<{
    results: any[];
    security_summary: {
      total_requests: number;
      allowed: number;
      blocked: number;
      challenged: number;
      delayed: number;
      validation_results: SynthesisValidationResult[];
    };
  }> {
    console.log(`\n🔒 SECURE CODE EMERGENCE (VERMELHO + ROXO)\n`);
    console.log('='.repeat(80));
    console.log(`User: ${this.userId}`);
    console.log(`Candidates: ${candidates.length}`);
    console.log(`Security Screening: ENABLED`);
    console.log('='.repeat(80));
    console.log('');

    const validationResults: SynthesisValidationResult[] = [];
    const approvedCandidates: EmergenceCandidate[] = [];

    // Phase 1: Security Validation for all candidates
    console.log('📋 PHASE 1: Security Validation\n');

    for (const candidate of candidates) {
      this.stats.total_requests++;

      // Create synthesis request
      const request = createSynthesisRequest(
        this.userId,
        candidate,
        this.baseEngine.getOrganism().metadata.organism_id,
        this.baseEngine.getOrganism().metadata.specialization
      );

      // Validate security
      const validation = this.securityGuard.validateSynthesisRequest(
        request,
        this.userProfiles,
        undefined,
        sessionDurationMinutes
      );

      validationResults.push(validation);

      // Update stats
      this.stats[validation.decision]++;

      // Print validation result
      console.log(`Function: ${candidate.suggested_function_name}`);
      console.log(getSynthesisValidationSummary(validation));

      // Handle based on decision
      if (validation.decision === 'allow') {
        approvedCandidates.push(candidate);
        console.log(`✅ ALLOWED - Proceeding with synthesis\n`);
      } else if (validation.decision === 'challenge') {
        console.log(`🧠 CHALLENGE REQUIRED - Cognitive verification needed`);

        // Attempt cognitive challenge
        const challengeResult = await this.performCognitiveChallenge(validation);

        if (challengeResult.authenticated) {
          approvedCandidates.push(candidate);
          console.log(`✅ Challenge passed - Synthesis approved\n`);
        } else {
          console.log(`❌ Challenge failed - Synthesis blocked\n`);
          this.stats.blocked++;
        }
      } else if (validation.decision === 'delay') {
        console.log(`⏱️  DELAYED - Recommend waiting before synthesis\n`);
      } else if (validation.decision === 'block') {
        console.log(`🚫 BLOCKED - Synthesis not allowed\n`);
      }
    }

    // Phase 2: Synthesize approved candidates
    console.log('\n' + '='.repeat(80));
    console.log('📋 PHASE 2: Code Synthesis (Approved Functions)\n');

    let emergedResults: any[] = [];

    if (approvedCandidates.length > 0) {
      console.log(`Synthesizing ${approvedCandidates.length} approved functions...\n`);
      emergedResults = await this.baseEngine.emerge(approvedCandidates);
    } else {
      console.log(`No functions approved for synthesis.\n`);
    }

    // Phase 3: Summary
    console.log('\n' + '='.repeat(80));
    console.log('📊 SECURITY SUMMARY\n');
    console.log(`Total Requests: ${this.stats.total_requests}`);
    console.log(`✅ Allowed: ${this.stats.allowed}`);
    console.log(`🧠 Challenged: ${this.stats.challenged}`);
    console.log(`⏱️  Delayed: ${this.stats.delayed}`);
    console.log(`🚫 Blocked: ${this.stats.blocked}`);
    console.log(`\n📈 Approval Rate: ${((this.stats.allowed / this.stats.total_requests) * 100).toFixed(0)}%`);
    console.log('='.repeat(80));
    console.log('');

    return {
      results: emergedResults,
      security_summary: {
        total_requests: this.stats.total_requests,
        allowed: this.stats.allowed,
        blocked: this.stats.blocked,
        challenged: this.stats.challenged,
        delayed: this.stats.delayed,
        validation_results: validationResults,
      },
    };
  }

  /**
   * Perform cognitive challenge
   */
  private async performCognitiveChallenge(
    validation: SynthesisValidationResult
  ): Promise<{ authenticated: boolean; confidence: number }> {
    if (!validation.requires_cognitive_challenge || !this.challengeSet) {
      return { authenticated: false, confidence: 0 };
    }

    console.log(`   🧠 Requesting cognitive verification...`);

    // Select appropriate challenges based on difficulty
    const challenges = CognitiveChallenge.selectChallenges(this.challengeSet, {
      is_sensitive_operation: validation.security_context.is_sensitive_synthesis,
      duress_score: validation.security_context.duress_score.score,
      min_difficulty:
        validation.challenge_difficulty === 'hard'
          ? 0.6
          : validation.challenge_difficulty === 'medium'
            ? 0.4
            : 0.2,
      count: validation.challenge_difficulty === 'hard' ? 2 : 1,
    });

    if (challenges.length === 0) {
      console.log(`   ⚠️  No challenges available - defaulting to block`);
      return { authenticated: false, confidence: 0 };
    }

    // In a real system, these would be prompted to the user
    // For demo purposes, simulate correct answers
    console.log(`   📋 Challenges presented: ${challenges.length}`);
    challenges.forEach((c, i) => {
      console.log(`      ${i + 1}. [${c.type}] "${c.question}"`);
    });

    // Simulate authentication (in real system, user would provide answers)
    const simulatedAnswers = challenges.map((c) => ({
      challenge_id: c.challenge_id,
      answer: 'simulated_correct_answer', // Would come from user
    }));

    const authResult = CognitiveAuthenticator.authenticate(
      this.challengeSet,
      simulatedAnswers,
      {
        min_challenges: challenges.length,
        min_confidence: validation.challenge_difficulty === 'hard' ? 0.8 : 0.7,
      }
    );

    console.log(
      `   ${authResult.authenticated ? '✅' : '❌'} Authentication: ${authResult.authenticated ? 'PASS' : 'FAIL'} (${(authResult.confidence * 100).toFixed(0)}% confidence)`
    );

    return {
      authenticated: authResult.authenticated,
      confidence: authResult.confidence,
    };
  }

  /**
   * Get organism (delegates to base engine)
   */
  getOrganism(): GlassOrganism {
    return this.baseEngine.getOrganism();
  }

  /**
   * Get cost statistics (delegates to base engine)
   */
  getCostStats() {
    return this.baseEngine.getCostStats();
  }

  /**
   * Get security statistics
   */
  getSecurityStats() {
    return {
      ...this.stats,
      synthesis_stats: this.storage
        ? this.securityGuard.getSynthesisStatistics(this.userId, 24)
        : null,
    };
  }

  /**
   * Set cognitive challenge set
   */
  setChallengeSet(challengeSet: ChallengeSet) {
    this.challengeSet = challengeSet;
  }

  /**
   * Update user profiles (e.g., after new interactions)
   */
  updateProfiles(profiles: UserSecurityProfiles) {
    this.userProfiles = profiles;
  }
}

// ===========================================================================
// FACTORY
// ===========================================================================

/**
 * Create secure code emergence engine
 */
export function createSecureCodeEmergenceEngine(
  organism: GlassOrganism,
  userProfiles: UserSecurityProfiles,
  userId: string,
  storage?: SecurityStorage,
  maxBudget: number = 0.5
): SecureCodeEmergenceEngine {
  return new SecureCodeEmergenceEngine(organism, userProfiles, userId, storage, maxBudget);
}
