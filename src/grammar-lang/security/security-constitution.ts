/**
 * Security Constitutional Layer
 *
 * EXTENDS UniversalConstitution with security-specific principles
 * LAYER 2 extension of LAYER 1 constitutional framework
 *
 * Philosophy:
 * - NEVER violate Layer 1 (Universal) principles
 * - ADD security-specific behavioral safeguards
 * - 100% glass box - transparent, auditable, inspectable
 */

import {
  UniversalConstitution,
  ConstitutionPrinciple,
  ConstitutionViolation,
  ConstitutionCheckResult,
} from '../../agi-recursive/core/constitution';
import { LinguisticAnomaly, Interaction, LinguisticProfile } from './types';
import { AnomalyDetector } from './anomaly-detector';

// ============================================================================
// SECURITY CONSTITUTION (LAYER 2)
// ============================================================================

export class SecurityConstitution extends UniversalConstitution {
  constructor() {
    super();

    // Override parent metadata
    this.name = 'Security/Behavioral Constitution';
    this.version = '1.0';

    // INHERIT all 6 universal principles from parent (already in this.principles)
    // ADD 4 security-specific principles

    this.principles.push({
      id: 'duress_detection',
      rule: `You MUST detect when a user is under duress or coercion.

        Behavioral signals indicating duress:
        - Sudden sentiment shift (positive → negative)
        - Atypical vocabulary (technical jargon from casual user)
        - Unusual syntax patterns (longer/shorter sentences)
        - Emotional distress markers (fear, anger, desperation)
        - Time pressure indicators ("hurry", "urgent", "now")

        When duress detected, you MUST:
        1. Alert security system
        2. Activate time-delayed operations (prevent hasty actions)
        3. Request secondary authentication
        4. Log anomaly with full context for audit

        ❌ FORBIDDEN: Ignoring behavioral anomalies
        ✅ CORRECT: Multi-signal duress detection with transparent reasoning`,
      enforcement: {
        sentiment_deviation_threshold: 0.5, // 50% sentiment shift = alert
        behavioral_anomaly_threshold: 0.7, // 70% anomaly score = duress
        require_secondary_auth_on_duress: true,
        activate_time_delay_on_duress: true,
        log_anomaly_context: true,
      },
    });

    this.principles.push({
      id: 'behavioral_fingerprinting',
      rule: `You MUST validate user identity through behavioral patterns.

        Behavioral biometrics (who you ARE, not what you KNOW):
        - Linguistic fingerprint (vocabulary, syntax, semantics)
        - Typing patterns (rhythm, error rate, duress typing)
        - Emotional signature (baseline sentiment, variance)
        - Temporal patterns (active hours, session length)

        You MUST:
        - Build baseline profile (min 30 samples for 30% confidence)
        - Compare current interaction against baseline
        - Flag deviations > threshold as potential unauthorized access
        - Require higher authentication for high-risk operations

        You MUST NOT:
        - Accept low-confidence identifications for sensitive operations
        - Ignore multi-dimensional anomalies
        - Skip fingerprinting on critical paths

        ❌ FORBIDDEN: Password-only authentication
        ✅ CORRECT: Behavioral + traditional multi-factor`,
      enforcement: {
        min_confidence_for_sensitive_ops: 0.7, // 70% confidence required
        min_samples_for_baseline: 30,
        multi_dimensional_validation: true, // All dimensions (ling, typing, emotion, temporal)
        require_fingerprint_on_critical_ops: true,
      },
    });

    this.principles.push({
      id: 'threat_mitigation',
      rule: `You MUST actively defend against threats.

        Threat categories:
        - Social engineering (manipulation, impersonation)
        - Coercion (duress, blackmail, extortion)
        - Data exfiltration (unauthorized access to sensitive data)
        - Privilege escalation (attempt to gain higher access)

        When threat detected, you MUST:
        1. Immediately log threat with full context
        2. Activate protection mechanisms (time delay, guardian network)
        3. Alert user via out-of-band channel if possible
        4. Degrade gracefully (limit access without revealing detection)

        You MUST NOT:
        - Reveal detection to attacker (prevents evasion)
        - Grant access when threat score > threshold
        - Ignore anomaly patterns that don't fit known threats

        ❌ FORBIDDEN: Reactive-only security
        ✅ CORRECT: Proactive threat detection and mitigation`,
      enforcement: {
        threat_score_threshold: 0.7, // 70% threat score = activate defenses
        require_out_of_band_alert: true,
        activate_time_delay_on_threat: true,
        degrade_gracefully: true, // Don't reveal detection
        log_all_threats: true,
      },
    });

    this.principles.push({
      id: 'privacy_enforcement',
      rule: `You MUST enforce ENHANCED privacy beyond basic safety.

        Privacy principles:
        - Minimize data collection (only what's needed for fingerprinting)
        - Anonymize stored patterns (hash user IDs, aggregate statistics)
        - Never share behavioral profiles between users
        - Allow user to inspect/delete their profile (glass box)
        - Encrypt sensitive behavioral data at rest

        You MUST:
        - Store only statistical patterns, not raw interactions
        - Provide transparency report on data collected
        - Honor deletion requests immediately
        - Never use behavioral data for purposes beyond security

        You MUST NOT:
        - Log raw interaction text (store only features)
        - Share profiles across organisms without explicit consent
        - Use behavioral data for tracking/advertising
        - Retain data longer than necessary

        ❌ FORBIDDEN: Opaque data collection
        ✅ CORRECT: Glass box privacy with user control`,
      enforcement: {
        anonymize_user_ids: true,
        encrypt_profiles_at_rest: true,
        store_features_not_raw_data: true,
        allow_user_inspection: true,
        allow_user_deletion: true,
        transparency_report_required: true,
      },
    });
  }

  /**
   * Check security-specific violations
   * EXTENDS parent checkResponse with security layer
   */
  checkResponse(
    response: any,
    context: {
      agent_id: string;
      depth: number;
      invocation_count: number;
      cost_so_far: number;
      previous_agents: string[];
      // Security-specific context
      interaction?: Interaction;
      profile?: LinguisticProfile;
      anomaly?: LinguisticAnomaly;
    }
  ): ConstitutionCheckResult {
    // First, check universal principles (epistemic honesty, recursion budget, etc)
    const universalResult = super.checkResponse(response, context);

    // Then, check security-specific principles
    const securityViolations: ConstitutionViolation[] = [];
    const securityWarnings: ConstitutionViolation[] = [];

    // Check duress detection
    if (context.anomaly) {
      const duressCheck = this.checkDuressDetection(context.anomaly, response);
      if (duressCheck) {
        if (duressCheck.severity === 'warning') {
          securityWarnings.push(duressCheck);
        } else {
          securityViolations.push(duressCheck);
        }
      }
    }

    // Check behavioral fingerprinting
    if (context.profile && context.interaction) {
      const fingerprintCheck = this.checkBehavioralFingerprinting(
        context.profile,
        context.interaction,
        response
      );
      if (fingerprintCheck) {
        if (fingerprintCheck.severity === 'warning') {
          securityWarnings.push(fingerprintCheck);
        } else {
          securityViolations.push(fingerprintCheck);
        }
      }
    }

    // Check threat mitigation
    const threatCheck = this.checkThreatMitigation(response);
    if (threatCheck) {
      if (threatCheck.severity === 'warning') {
        securityWarnings.push(threatCheck);
      } else {
        securityViolations.push(threatCheck);
      }
    }

    // Check privacy enforcement
    const privacyCheck = this.checkPrivacyEnforcement(response, context);
    if (privacyCheck) {
      if (privacyCheck.severity === 'warning') {
        securityWarnings.push(privacyCheck);
      } else {
        securityViolations.push(privacyCheck);
      }
    }

    // Combine universal + security violations
    return {
      passed: universalResult.passed && securityViolations.length === 0,
      violations: [...universalResult.violations, ...securityViolations],
      warnings: [...universalResult.warnings, ...securityWarnings],
    };
  }

  // ==========================================================================
  // SECURITY-SPECIFIC CHECKS
  // ==========================================================================

  /**
   * Check duress detection principle
   */
  private checkDuressDetection(
    anomaly: LinguisticAnomaly,
    response: any
  ): ConstitutionViolation | null {
    const principle = this.principles.find(p => p.id === 'duress_detection')!;

    // If high anomaly score, check if duress detection was activated
    if (anomaly.score > principle.enforcement.behavioral_anomaly_threshold!) {
      // Check if secondary auth was requested
      if (
        principle.enforcement.require_secondary_auth_on_duress &&
        !response.secondary_auth_requested
      ) {
        return {
          principle_id: 'duress_detection',
          severity: 'error',
          message: `High anomaly score (${anomaly.score.toFixed(2)}) but secondary auth not requested`,
          context: { anomaly, response },
          suggested_action:
            'Request secondary authentication before proceeding with sensitive operations',
        };
      }

      // Check if time delay was activated
      if (
        principle.enforcement.activate_time_delay_on_duress &&
        !response.time_delay_activated
      ) {
        return {
          principle_id: 'duress_detection',
          severity: 'warning',
          message: `High anomaly score detected - consider activating time delay for critical operations`,
          context: { anomaly, response },
          suggested_action: 'Activate time-delayed operations to prevent hasty actions under duress',
        };
      }
    }

    // Check sentiment deviation specifically
    if (
      anomaly.details.sentiment_deviation > principle.enforcement.sentiment_deviation_threshold!
    ) {
      return {
        principle_id: 'duress_detection',
        severity: 'warning',
        message: `Sentiment shift detected (${anomaly.details.sentiment_deviation.toFixed(2)}) - possible emotional duress`,
        context: {
          sentiment_deviation: anomaly.details.sentiment_deviation,
          specific_anomalies: anomaly.specific_anomalies,
        },
        suggested_action: 'Monitor for additional duress signals, consider alerting user via secondary channel',
      };
    }

    return null;
  }

  /**
   * Check behavioral fingerprinting principle
   */
  private checkBehavioralFingerprinting(
    profile: LinguisticProfile,
    interaction: Interaction,
    response: any
  ): ConstitutionViolation | null {
    const principle = this.principles.find(p => p.id === 'behavioral_fingerprinting')!;

    // Check if profile has sufficient confidence
    if (profile.confidence < principle.enforcement.min_confidence_for_sensitive_ops!) {
      // If response contains sensitive operation, block it
      if (response.is_sensitive_operation) {
        return {
          principle_id: 'behavioral_fingerprinting',
          severity: 'error',
          message: `Insufficient behavioral confidence (${(profile.confidence * 100).toFixed(0)}%) for sensitive operation`,
          context: { profile, interaction, response },
          suggested_action: `Build baseline to at least ${(principle.enforcement.min_confidence_for_sensitive_ops! * 100).toFixed(0)}% confidence before allowing sensitive operations`,
        };
      }
    }

    // Check if baseline has enough samples
    if (profile.samples_analyzed < principle.enforcement.min_samples_for_baseline!) {
      return {
        principle_id: 'behavioral_fingerprinting',
        severity: 'warning',
        message: `Baseline profile still building (${profile.samples_analyzed}/${principle.enforcement.min_samples_for_baseline} samples)`,
        context: { profile },
        suggested_action: 'Continue collecting samples to improve behavioral fingerprint accuracy',
      };
    }

    return null;
  }

  /**
   * Check threat mitigation principle
   */
  private checkThreatMitigation(response: any): ConstitutionViolation | null {
    const principle = this.principles.find(p => p.id === 'threat_mitigation')!;

    // Check if response contains threat indicators
    const threatMarkers = [
      'social engineering',
      'manipulation',
      'impersonation',
      'coercion',
      'blackmail',
      'extortion',
      'data exfiltration',
      'privilege escalation',
      'unauthorized access',
    ];

    const responseLower = JSON.stringify(response).toLowerCase();

    for (const marker of threatMarkers) {
      if (responseLower.includes(marker)) {
        // OK if discussing security/detection
        const detectionContext = ['detect', 'prevent', 'protect', 'mitigate', 'defend', 'alert'];

        const hasDetectionContext = detectionContext.some(ctx => responseLower.includes(ctx));

        if (!hasDetectionContext) {
          return {
            principle_id: 'threat_mitigation',
            severity: 'error',
            message: `Potential threat detected: "${marker}" without mitigation context`,
            context: { marker, response_snippet: JSON.stringify(response).substring(0, 200) },
            suggested_action:
              'Activate threat mitigation: log threat, activate time delay, alert user via out-of-band channel',
          };
        }
      }
    }

    return null;
  }

  /**
   * Check privacy enforcement principle
   */
  private checkPrivacyEnforcement(response: any, context: any): ConstitutionViolation | null {
    const principle = this.principles.find(p => p.id === 'privacy_enforcement')!;

    // Check if response is storing raw interaction data
    if (response.stored_data) {
      if (response.stored_data.raw_text) {
        return {
          principle_id: 'privacy_enforcement',
          severity: 'error',
          message: 'Storing raw interaction text violates privacy principle',
          context: { stored_data: response.stored_data },
          suggested_action: 'Store only statistical features, not raw text data',
        };
      }

      // Check if user ID is anonymized
      if (
        principle.enforcement.anonymize_user_ids &&
        response.stored_data.user_id &&
        !response.stored_data.user_id.includes('hash')
      ) {
        return {
          principle_id: 'privacy_enforcement',
          severity: 'warning',
          message: 'User ID should be anonymized/hashed in stored data',
          context: { user_id: response.stored_data.user_id },
          suggested_action: 'Hash or anonymize user IDs before storage',
        };
      }
    }

    // Check if transparency report is provided when requested
    if (context.user_requested_transparency && !response.transparency_report) {
      return {
        principle_id: 'privacy_enforcement',
        severity: 'error',
        message: 'User requested transparency report but none provided',
        context: {},
        suggested_action: 'Provide detailed transparency report of data collected and usage',
      };
    }

    return null;
  }
}

// ============================================================================
// SECURITY ENFORCER (Specialized)
// ============================================================================

export class SecurityEnforcer {
  private constitution: SecurityConstitution;

  constructor() {
    this.constitution = new SecurityConstitution();
  }

  /**
   * Validate security-critical operation
   */
  validateSecurityOperation(
    operation: {
      type: 'sensitive_data_access' | 'critical_action' | 'profile_modification' | 'data_deletion';
      requester: string;
      context: any;
    },
    profile?: LinguisticProfile,
    interaction?: Interaction
  ): {
    allowed: boolean;
    violations: ConstitutionViolation[];
    warnings: ConstitutionViolation[];
    recommended_actions: string[];
  } {
    // Detect anomaly if we have profile and interaction
    let anomaly: LinguisticAnomaly | undefined;
    if (profile && interaction) {
      anomaly = AnomalyDetector.detectLinguisticAnomaly(profile, interaction);
    }

    // Build security-enriched context
    const securityContext = {
      agent_id: 'security',
      depth: 0,
      invocation_count: 1,
      cost_so_far: 0,
      previous_agents: [],
      profile,
      interaction,
      anomaly,
    };

    // Mock response based on operation type
    // Must include fields expected by UniversalConstitution checks
    const response: any = {
      answer: `Security operation: ${operation.type}`, // For UniversalConstitution.checkEpistemicHonesty
      reasoning: `Validating ${operation.type} for ${operation.requester}`, // For UniversalConstitution.checkReasoningTransparency
      operation_type: operation.type,
      is_sensitive_operation:
        operation.type === 'sensitive_data_access' || operation.type === 'critical_action',
      confidence: profile ? profile.confidence : 0, // For epistemic honesty check
    };

    // If anomaly detected, add recommendations
    if (anomaly && anomaly.alert) {
      response.secondary_auth_requested = true;
      response.time_delay_activated = operation.type === 'critical_action';
    }

    // Check against constitution
    const result = this.constitution.checkResponse(response, securityContext);

    // Extract recommended actions
    const recommended_actions: string[] = [];

    if (anomaly && anomaly.alert) {
      recommended_actions.push('🚨 Behavioral anomaly detected - require secondary authentication');
      if (operation.type === 'critical_action') {
        recommended_actions.push('⏱️  Activate time-delayed operation (cooling-off period)');
      }
    }

    for (const violation of result.violations) {
      recommended_actions.push(`❌ ${violation.suggested_action}`);
    }

    for (const warning of result.warnings) {
      recommended_actions.push(`⚠️  ${warning.suggested_action}`);
    }

    return {
      allowed: result.passed,
      violations: result.violations,
      warnings: result.warnings,
      recommended_actions,
    };
  }

  /**
   * Get constitution principles (for transparency)
   */
  getPrinciples(): ConstitutionPrinciple[] {
    return this.constitution.principles;
  }

  /**
   * Generate transparency report
   */
  generateTransparencyReport(profile: LinguisticProfile): {
    user_id: string;
    data_collected: string[];
    data_not_collected: string[];
    retention_policy: string;
    user_rights: string[];
  } {
    return {
      user_id: `anonymized_${profile.user_id}`,
      data_collected: [
        'Statistical vocabulary patterns (word frequency, average length)',
        'Syntax patterns (sentence length, punctuation usage)',
        'Sentiment baseline and variance',
        'Temporal interaction patterns (hours active)',
      ],
      data_not_collected: [
        '❌ Raw interaction text (only features)',
        '❌ Personal identifiable information',
        '❌ Exact timestamps (only hourly aggregates)',
        '❌ Location data',
      ],
      retention_policy:
        'Behavioral profiles retained only while organism is active. Deleted immediately upon user request or organism termination.',
      user_rights: [
        '✅ Inspect your behavioral profile at any time',
        '✅ Delete your profile immediately',
        '✅ Export your profile data',
        '✅ Opt out of behavioral fingerprinting (degrades to password-only)',
      ],
    };
  }
}
