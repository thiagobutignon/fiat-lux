/**
 * Multi-Signal Duress Detection
 *
 * Combines all 4 behavioral signals for high-confidence duress detection:
 * - Linguistic fingerprinting
 * - Typing patterns
 * - Emotional state
 * - Temporal patterns
 */

import { AnomalyDetector } from './anomaly-detector';
import { TypingAnomalyDetector } from './typing-anomaly-detector';
import { EmotionalAnomalyDetector } from './emotional-anomaly-detector';
import { TemporalAnomalyDetector } from './temporal-anomaly-detector';
import { CognitiveChallenge, ChallengeSet } from './cognitive-challenge';
import {
  UserSecurityProfiles,
  Interaction,
  DuressScore,
  SecurityContext,
} from './types';

// =============================================================================
// MULTI-SIGNAL DETECTOR
// =============================================================================

export class MultiSignalDetector {
  /**
   * Detect duress using all available behavioral signals
   * Combines linguistic, typing, emotional, and temporal analysis
   */
  static detectDuress(
    profiles: UserSecurityProfiles,
    interaction: Interaction,
    sessionDurationMinutes?: number
  ): DuressScore {
    // Run all detectors
    const linguisticAnomaly = AnomalyDetector.detectLinguisticAnomaly(
      profiles.linguistic,
      interaction
    );

    const typingAnomaly = TypingAnomalyDetector.detectTypingAnomaly(
      profiles.typing,
      interaction
    );

    const emotionalAnomaly = EmotionalAnomalyDetector.detectEmotionalAnomaly(
      profiles.emotional,
      interaction
    );

    const temporalAnomaly = TemporalAnomalyDetector.detectTemporalAnomaly(
      profiles.temporal,
      interaction,
      sessionDurationMinutes
    );

    // Check for panic code
    const panicCodeDetected = this.detectPanicCode(interaction.text);

    // Weighted combination of all signals
    // Linguistic: 25% (vocabulary/syntax changes)
    // Typing: 25% (speed/error changes)
    // Emotional: 25% (VAD changes)
    // Temporal: 15% (unusual access time)
    // Panic code: 10% bonus (if detected)
    const duressScore =
      linguisticAnomaly.score * 0.25 +
      typingAnomaly.score * 0.25 +
      emotionalAnomaly.score * 0.25 +
      temporalAnomaly.score * 0.15 +
      (panicCodeDetected ? 0.10 : 0);

    // Calculate confidence based on how many signals agree
    const signalsInAlert = [
      linguisticAnomaly.alert,
      typingAnomaly.alert,
      emotionalAnomaly.alert,
      temporalAnomaly.alert,
      panicCodeDetected,
    ].filter(Boolean).length;

    const confidence = signalsInAlert / 5; // 0-1 based on agreement

    // Determine threshold and alert
    const threshold = 0.6; // 60% threshold for duress
    const alert = duressScore > threshold || panicCodeDetected;

    // Determine recommendation
    const recommendation = this.determineRecommendation(
      duressScore,
      signalsInAlert,
      panicCodeDetected,
      interaction.operation_type
    );

    // Build reason
    const reason = this.buildReason(
      duressScore,
      signalsInAlert,
      panicCodeDetected,
      linguisticAnomaly.alert,
      typingAnomaly.alert,
      emotionalAnomaly.alert,
      temporalAnomaly.alert
    );

    return {
      score: Math.min(duressScore, 1.0),
      threshold,
      alert,
      confidence,
      signals: {
        linguistic_anomaly: linguisticAnomaly.score,
        typing_anomaly: typingAnomaly.score,
        emotional_anomaly: emotionalAnomaly.score,
        temporal_anomaly: temporalAnomaly.score,
        panic_code_detected: panicCodeDetected,
      },
      recommendation,
      reason,
    };
  }

  // ===========================================================================
  // COERCION DETECTION
  // ===========================================================================

  /**
   * Detect coercion using multiple behavioral signals
   * Higher confidence than single-signal detection
   */
  static detectCoercion(
    profiles: UserSecurityProfiles,
    interaction: Interaction,
    context: { is_sensitive_operation: boolean; operation_type?: string }
  ): {
    coercion_detected: boolean;
    confidence: number;
    indicators: string[];
    recommendation: 'allow' | 'challenge' | 'delay' | 'block';
  } {
    // Get emotional coercion indicators
    const emotionalCoercion = EmotionalAnomalyDetector.detectCoercion(
      profiles.emotional,
      interaction
    );

    // Get typing duress indicators
    const typingDuress = TypingAnomalyDetector.detectDuressFromTyping(
      profiles.typing,
      interaction
    );

    // Combine indicators
    const indicators: string[] = [];
    let coercionScore = 0;

    // Emotional indicators (40% weight)
    if (emotionalCoercion.coercion_detected) {
      indicators.push(...emotionalCoercion.indicators);
      coercionScore += emotionalCoercion.confidence * 0.4;
    }

    // Typing indicators (40% weight)
    if (typingDuress.duress_detected) {
      indicators.push(...typingDuress.indicators);
      coercionScore += typingDuress.confidence * 0.4;
    }

    // Check for submission language (20% weight)
    const submissionDetected = this.detectSubmissionLanguage(interaction.text);
    if (submissionDetected) {
      indicators.push('Submission/compliance language detected');
      coercionScore += 0.2;
    }

    // If sensitive operation AND showing coercion signs, increase score
    if (context.is_sensitive_operation && coercionScore > 0.3) {
      indicators.push('Coercion indicators during sensitive operation');
      coercionScore += 0.3; // Strong boost
    }

    const coercionDetected = coercionScore > 0.6;
    const confidence = Math.min(coercionScore, 1.0);

    // Determine recommendation
    let recommendation: 'allow' | 'challenge' | 'delay' | 'block';
    if (context.is_sensitive_operation && coercionDetected) {
      recommendation = 'block'; // Block sensitive ops under coercion
    } else if (coercionScore > 0.8) {
      recommendation = 'block';
    } else if (coercionScore > 0.6) {
      recommendation = 'delay';
    } else if (coercionScore > 0.4) {
      recommendation = 'challenge';
    } else {
      recommendation = 'allow';
    }

    return {
      coercion_detected: coercionDetected,
      confidence,
      indicators,
      recommendation,
    };
  }

  // ===========================================================================
  // PANIC CODE DETECTION
  // ===========================================================================

  /**
   * Detect panic codes (pre-established phrases)
   * In production, these would be user-configured
   */
  private static detectPanicCode(text: string): boolean {
    const lowerText = text.toLowerCase();

    // Example panic codes (in production, these would be user-defined)
    const panicCodes = [
      'code red',
      'emergency situation',
      'under duress',
      'help me please',
      'call guardian',
    ];

    return panicCodes.some((code) => lowerText.includes(code));
  }

  /**
   * Detect submission/compliance language
   */
  private static detectSubmissionLanguage(text: string): boolean {
    const lowerText = text.toLowerCase();

    const submissionPhrases = [
      'i have to',
      'i must',
      'no choice',
      'forced to',
      'they want me to',
      'being made to',
      'okay fine',
      'whatever',
      'just do it',
    ];

    return submissionPhrases.some((phrase) => lowerText.includes(phrase));
  }

  // ===========================================================================
  // RECOMMENDATION LOGIC
  // ===========================================================================

  /**
   * Determine recommendation based on duress score and signals
   */
  private static determineRecommendation(
    score: number,
    signalsInAlert: number,
    panicCodeDetected: boolean,
    operationType?: string
  ): 'allow' | 'challenge' | 'delay' | 'block' {
    // Panic code = immediate block
    if (panicCodeDetected) {
      return 'block';
    }

    // High score OR multiple signals = block
    if (score > 0.8 || signalsInAlert >= 4) {
      return 'block';
    }

    // Medium-high score OR 3 signals = delay
    if (score > 0.6 || signalsInAlert >= 3) {
      return 'delay';
    }

    // Medium score OR 2 signals = challenge
    if (score > 0.4 || signalsInAlert >= 2) {
      return 'challenge';
    }

    // Low score = allow
    return 'allow';
  }

  /**
   * Build human-readable reason
   */
  private static buildReason(
    score: number,
    signalsInAlert: number,
    panicCode: boolean,
    linguistic: boolean,
    typing: boolean,
    emotional: boolean,
    temporal: boolean
  ): string {
    if (panicCode) {
      return 'Panic code detected - immediate intervention required';
    }

    if (signalsInAlert >= 4) {
      return 'Multiple behavioral anomalies detected (4+ signals) - high confidence duress';
    }

    if (score > 0.8) {
      return 'Very high anomaly score - strong duress indication';
    }

    const signals: string[] = [];
    if (linguistic) signals.push('linguistic');
    if (typing) signals.push('typing');
    if (emotional) signals.push('emotional');
    if (temporal) signals.push('temporal');

    if (signals.length >= 2) {
      return `Behavioral anomalies detected: ${signals.join(', ')}`;
    }

    if (signals.length === 1) {
      return `Minor anomaly in ${signals[0]} pattern`;
    }

    return 'Normal behavioral pattern';
  }

  // ===========================================================================
  // SECURITY CONTEXT BUILDER
  // ===========================================================================

  /**
   * Build complete security context for decision making
   */
  static buildSecurityContext(
    profiles: UserSecurityProfiles,
    interaction: Interaction,
    operationContext: {
      operation_type?: string;
      is_sensitive_operation: boolean;
      operation_value?: number;
    },
    sessionDurationMinutes?: number
  ): SecurityContext {
    // Detect duress
    const duressScore = this.detectDuress(profiles, interaction, sessionDurationMinutes);

    // Detect coercion
    const coercionScore = this.detectCoercion(profiles, interaction, {
      is_sensitive_operation: operationContext.is_sensitive_operation,
      operation_type: operationContext.operation_type,
    });

    // Determine final decision (most conservative wins)
    let decision: 'allow' | 'challenge' | 'delay' | 'block';
    const recommendations = [duressScore.recommendation, coercionScore.recommendation];

    if (recommendations.includes('block')) {
      decision = 'block';
    } else if (recommendations.includes('delay')) {
      decision = 'delay';
    } else if (recommendations.includes('challenge')) {
      decision = 'challenge';
    } else {
      decision = 'allow';
    }

    // Build decision reason
    let decisionReason = duressScore.reason;
    if (coercionScore.coercion_detected) {
      decisionReason += ` | Coercion detected (${(coercionScore.confidence * 100).toFixed(0)}% confidence)`;
    }

    return {
      user_id: profiles.user_id,
      interaction_id: interaction.interaction_id,
      timestamp: interaction.timestamp,
      profiles,
      duress_score: duressScore,
      coercion_score: {
        score: coercionScore.confidence,
        threshold: 0.6,
        alert: coercionScore.coercion_detected,
        confidence: coercionScore.confidence,
        indicators: {
          compliance_language: coercionScore.indicators.some((i) => i.includes('compliance')),
          passive_voice_excessive: false, // Would need linguistic analysis
          hedging_excessive: false, // Would need linguistic analysis
          fear_markers: coercionScore.indicators.some((i) => i.includes('fear')),
          submission_markers: coercionScore.indicators.some((i) => i.includes('submission')),
          rushed_responses: coercionScore.indicators.some((i) => i.includes('rushed')),
          unusual_requests: operationContext.is_sensitive_operation,
        },
        recommendation: coercionScore.recommendation,
        reason: coercionScore.indicators.join('; '),
      },
      operation_type: operationContext.operation_type,
      is_sensitive_operation: operationContext.is_sensitive_operation,
      operation_value: operationContext.operation_value,
      decision,
      decision_reason: decisionReason,
    };
  }

  // ===========================================================================
  // COGNITIVE CHALLENGE INTEGRATION
  // ===========================================================================

  /**
   * Determine if cognitive challenge is required based on security context
   * Returns true if additional verification is needed
   */
  static requiresCognitiveChallenge(context: SecurityContext): {
    required: boolean;
    reason: string;
    difficulty_level: 'easy' | 'medium' | 'hard';
  } {
    const duressScore = context.duress_score.score;
    const coercionScore = context.coercion_score.score;
    const isSensitive = context.is_sensitive_operation;

    // High duress/coercion score = require challenge
    if (duressScore > 0.6 || coercionScore > 0.6) {
      return {
        required: true,
        reason: 'High anomaly detected - additional verification required',
        difficulty_level: 'hard',
      };
    }

    // Sensitive operation with medium duress = require challenge
    if (isSensitive && (duressScore > 0.4 || coercionScore > 0.4)) {
      return {
        required: true,
        reason: 'Sensitive operation with behavioral anomaly - verification required',
        difficulty_level: 'medium',
      };
    }

    // Challenge recommendation = require challenge
    if (context.decision === 'challenge') {
      return {
        required: true,
        reason: 'Behavioral anomaly requires verification',
        difficulty_level: 'medium',
      };
    }

    // Panic code detected = require challenge (to confirm it's not attacker)
    if (context.duress_score.signals.panic_code_detected) {
      return {
        required: true,
        reason: 'Panic code detected - verifying user identity',
        difficulty_level: 'hard',
      };
    }

    // No challenge required
    return {
      required: false,
      reason: 'Normal behavioral pattern',
      difficulty_level: 'easy',
    };
  }

  /**
   * Request cognitive challenges appropriate for the security context
   * Returns challenges that should be presented to the user
   */
  static requestCognitiveVerification(
    challengeSet: ChallengeSet,
    context: SecurityContext
  ): {
    challenges: ReturnType<typeof CognitiveChallenge.selectChallenges>;
    required_count: number;
    min_confidence: number;
    reason: string;
  } {
    const challengeRequirement = this.requiresCognitiveChallenge(context);

    if (!challengeRequirement.required) {
      return {
        challenges: [],
        required_count: 0,
        min_confidence: 0,
        reason: 'No cognitive challenge required',
      };
    }

    // Determine challenge parameters based on difficulty level
    let minDifficulty: number;
    let requiredCount: number;
    let minConfidence: number;

    switch (challengeRequirement.difficulty_level) {
      case 'hard':
        minDifficulty = 0.6;
        requiredCount = 2;
        minConfidence = 0.8;
        break;
      case 'medium':
        minDifficulty = 0.4;
        requiredCount = 1;
        minConfidence = 0.7;
        break;
      case 'easy':
      default:
        minDifficulty = 0.2;
        requiredCount = 1;
        minConfidence = 0.6;
        break;
    }

    // Select appropriate challenges
    const challenges = CognitiveChallenge.selectChallenges(challengeSet, {
      is_sensitive_operation: context.is_sensitive_operation,
      duress_score: context.duress_score.score,
      min_difficulty: minDifficulty,
      count: requiredCount,
    });

    return {
      challenges,
      required_count: requiredCount,
      min_confidence: minConfidence,
      reason: challengeRequirement.reason,
    };
  }
}
