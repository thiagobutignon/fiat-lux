/**
 * Cognitive Constitution
 * EXTENDS UniversalConstitution with cognitive-specific principles
 *
 * Layer 1: UniversalConstitution (6 base principles)
 * Layer 2: CognitiveConstitution (4 additional cognitive principles)
 *
 * NEVER violates Layer 1, only adds cognitive-specific safeguards
 */

import {
  UniversalConstitution,
  ConstitutionPrinciple,
  ConstitutionViolation
} from '../../../agi-recursive/core/constitution';

// ============================================================================
// Cognitive Constitution (Layer 2)
// ============================================================================

export class CognitiveConstitution extends UniversalConstitution {
  constructor() {
    super();
    this.name = 'Cognitive OS Constitution';
    this.version = '2.0';

    // Add cognitive-specific principles (extends 6 base principles)
    this.principles.push(
      this.manipulationDetectionPrinciple(),
      this.darkTetradProtectionPrinciple(),
      this.neurodivergentSafeguardsPrinciple(),
      this.intentTransparencyPrinciple()
    );
  }

  /**
   * Principle 1: Manipulation Detection
   * 180 techniques must be applied consistently
   */
  private manipulationDetectionPrinciple(): ConstitutionPrinciple {
    return {
      id: 'manipulation_detection',
      rule: `You MUST detect manipulation techniques consistently:
        - All 180 techniques (1-152 GPT-4, 153-180 GPT-5)
        - Multi-layer analysis (MORPHEMES → SYNTAX → SEMANTICS → PRAGMATICS)
        - O(1) detection performance (<1ms per technique)
        - Confidence threshold: 0.8 minimum

        You MUST NOT:
        - Skip detection for performance (O(1) is mandatory)
        - Detect without explanation (glass box transparency)
        - Classify without linguistic evidence (cite sources)
        - Ignore temporal evolution (GPT-5 techniques track 2023-2025)

        If detection confidence < 0.8:
        1. Include warning in results
        2. Explain uncertainty in reasoning
        3. Suggest context for higher confidence`,
      enforcement: {
        require_source_citation: true,
        confidence_threshold: 0.8,
        require_reasoning_trace: true,
        min_explanation_length: 100,
        detect_performance_regression: true,
        max_detection_time_ms: 1
      }
    };
  }

  /**
   * Principle 2: Dark Tetrad Protection
   * 80+ behavioral markers across 4 dimensions
   */
  private darkTetradProtectionPrinciple(): ConstitutionPrinciple {
    return {
      id: 'dark_tetrad_protection',
      rule: `You MUST profile Dark Tetrad behaviors (Narcissism, Machiavellianism, Psychopathy, Sadism):
        - 80+ markers across 4 dimensions
        - Aggregate scoring (weighted by confidence)
        - Never diagnose individuals (detect patterns, not label people)
        - Always provide behavioral evidence

        You MUST NOT:
        - Label someone as "narcissist" or "psychopath" (no diagnosis principle)
        - Detect without behavioral markers (cite linguistic evidence)
        - Ignore context (relationship power dynamics matter)
        - Use Dark Tetrad for profiling without consent

        Safeguards:
        - All scores include confidence levels
        - All markers cite specific text sources
        - Context-aware adjustments applied
        - No personal data stored (privacy principle)`,
      enforcement: {
        no_diagnosis: true,
        require_behavioral_evidence: true,
        context_aware: true,
        privacy_check: true,
        min_markers_for_detection: 3
      }
    };
  }

  /**
   * Principle 3: Neurodivergent Safeguards
   * 10+ vulnerabilities protected
   */
  private neurodivergentSafeguardsPrinciple(): ConstitutionPrinciple {
    return {
      id: 'neurodivergent_safeguards',
      rule: `You MUST protect neurodivergent communication from false positives:
        - Autism markers: literal interpretation, direct communication, technical accuracy
        - ADHD markers: memory gaps, topic jumping, impulsive responses
        - Threshold adjustment: +15% when neurodivergent markers detected
        - False positive rate: <1% target

        You MUST NOT:
        - Classify neurodivergent communication as manipulation
        - Ignore neurodivergent context in detection
        - Apply same threshold to all communication styles
        - Discriminate based on communication differences

        Protection mechanisms:
        - Automatic marker detection (autism + ADHD)
        - Confidence threshold increased by 15%
        - Warnings when neurodivergent patterns detected
        - Cultural sensitivity applied (9 cultures)`,
      enforcement: {
        detect_neurodivergent_markers: true,
        threshold_adjustment: 0.15,
        max_false_positive_rate: 0.01,
        cultural_sensitivity: true,
        require_context_awareness: true
      }
    };
  }

  /**
   * Principle 4: Intent Transparency
   * Cognitive layer must explain intent detection
   */
  private intentTransparencyPrinciple(): ConstitutionPrinciple {
    return {
      id: 'intent_transparency',
      rule: `You MUST explain intent detection transparently:
        - Primary intent + secondary intents
        - Confidence (linguistic) vs context-adjusted confidence
        - Reasoning chain (why this intent was detected)
        - Context factors (relationship, history, power dynamics)

        You MUST NOT:
        - Detect intent without explaining linguistic basis
        - Apply context adjustment without justification
        - Predict escalation without evidence
        - Classify manipulation strategy without markers

        Glass box requirements:
        - All intent detections cite linguistic evidence
        - All context adjustments explained
        - All risk scores show calculation
        - All intervention urgency levels justified`,
      enforcement: {
        require_reasoning_trace: true,
        min_explanation_length: 150,
        cite_linguistic_evidence: true,
        explain_context_adjustments: true,
        transparency_score_minimum: 1.0
      }
    };
  }

  /**
   * Check cognitive-specific principles
   * Extends base checkResponse with cognitive validation
   */
  checkResponse(response: any, context: any): any {
    // First check base constitutional principles (Layer 1)
    const baseCheck = super.checkResponse(response, context);

    // Then check cognitive-specific principles (Layer 2)
    const cognitiveViolations: ConstitutionViolation[] = [];

    // Check manipulation detection principle
    if (response.detections) {
      const manipulationCheck = this.checkManipulationDetection(response);
      if (manipulationCheck) cognitiveViolations.push(manipulationCheck);
    }

    // Check Dark Tetrad protection principle
    if (response.dark_tetrad_aggregate) {
      const darkTetradCheck = this.checkDarkTetradProtection(response);
      if (darkTetradCheck) cognitiveViolations.push(darkTetradCheck);
    }

    // Check neurodivergent safeguards principle
    if (response.detections && response.detections.length > 0) {
      const neurodivergentCheck = this.checkNeurodivergentSafeguards(response);
      if (neurodivergentCheck) cognitiveViolations.push(neurodivergentCheck);
    }

    // Check intent transparency principle
    if (response.intent || response.analysis) {
      const intentCheck = this.checkIntentTransparency(response);
      if (intentCheck) cognitiveViolations.push(intentCheck);
    }

    // Merge base + cognitive violations
    return {
      passed: baseCheck.passed && cognitiveViolations.length === 0,
      violations: [...baseCheck.violations, ...cognitiveViolations],
      warnings: baseCheck.warnings
    };
  }

  /**
   * Check manipulation detection compliance
   */
  private checkManipulationDetection(response: any): ConstitutionViolation | null {
    // Check if detections have sources
    for (const detection of response.detections) {
      if (!detection.sources || detection.sources.length === 0) {
        return {
          principle_id: 'manipulation_detection',
          severity: 'error',
          message: `Detection "${detection.technique_name}" lacks source citation`,
          context: { detection },
          suggested_action: 'Add linguistic sources to all detections (morphemes, syntax, semantics)'
        };
      }

      // Check if confidence is below threshold
      if (detection.confidence < 0.8 && !detection.explanation.includes('uncertainty')) {
        return {
          principle_id: 'manipulation_detection',
          severity: 'warning',
          message: `Low confidence (${detection.confidence.toFixed(2)}) without uncertainty disclaimer`,
          context: { detection },
          suggested_action: 'Include warning about detection uncertainty'
        };
      }
    }

    // Check processing time (O(1) requirement)
    if (response.processing_time_ms && response.processing_time_ms > 1) {
      return {
        principle_id: 'manipulation_detection',
        severity: 'warning',
        message: `Detection time ${response.processing_time_ms}ms exceeds O(1) target (1ms)`,
        context: { processing_time: response.processing_time_ms },
        suggested_action: 'Optimize pattern matching for O(1) performance'
      };
    }

    return null;
  }

  /**
   * Check Dark Tetrad protection compliance
   */
  private checkDarkTetradProtection(response: any): ConstitutionViolation | null {
    // Check for diagnosis language (forbidden)
    const diagnosisMarkers = [
      'is a narcissist',
      'is a psychopath',
      'diagnosed with',
      'has NPD',
      'is manipulative'
    ];

    const summary = response.summary || response.explanation || '';
    for (const marker of diagnosisMarkers) {
      if (summary.toLowerCase().includes(marker)) {
        return {
          principle_id: 'dark_tetrad_protection',
          severity: 'fatal',
          message: `Diagnosis language detected: "${marker}" (violates no-diagnosis principle)`,
          context: { marker, summary },
          suggested_action: 'Rephrase to describe patterns, not label individuals'
        };
      }
    }

    // Check if Dark Tetrad scores have evidence
    if (!response.detections || response.detections.length < 3) {
      return {
        principle_id: 'dark_tetrad_protection',
        severity: 'warning',
        message: `Dark Tetrad profile with insufficient evidence (${response.detections?.length || 0} detections)`,
        context: { detection_count: response.detections?.length || 0 },
        suggested_action: 'Require minimum 3 behavioral markers for Dark Tetrad profiling'
      };
    }

    return null;
  }

  /**
   * Check neurodivergent safeguards compliance
   */
  private checkNeurodivergentSafeguards(response: any): ConstitutionViolation | null {
    // Check if neurodivergent markers were detected but not acknowledged
    const hasNeurodivergentFlag = response.detections.some((d: any) => d.neurodivergent_flag);

    if (hasNeurodivergentFlag) {
      // Check if threshold adjustment was applied
      const hasWarning = response.constitutional_validation?.warnings?.some((w: string) =>
        w.includes('Neurodivergent')
      );

      if (!hasWarning) {
        return {
          principle_id: 'neurodivergent_safeguards',
          severity: 'warning',
          message: 'Neurodivergent markers detected but not acknowledged in warnings',
          context: { detections: response.detections.filter((d: any) => d.neurodivergent_flag) },
          suggested_action: 'Add neurodivergent protection warning to constitutional_validation'
        };
      }
    }

    return null;
  }

  /**
   * Check intent transparency compliance
   */
  private checkIntentTransparency(response: any): ConstitutionViolation | null {
    const intent = response.intent || response.analysis?.intent;

    if (!intent) {
      return null; // No intent to validate
    }

    // Check if reasoning is provided
    if (!intent.reasoning || intent.reasoning.length === 0) {
      return {
        principle_id: 'intent_transparency',
        severity: 'error',
        message: 'Intent detected without reasoning explanation',
        context: { intent },
        suggested_action: 'Provide reasoning chain explaining why this intent was detected'
      };
    }

    // Check if reasoning is detailed enough
    const reasoningText = intent.reasoning.join(' ');
    if (reasoningText.length < 150) {
      return {
        principle_id: 'intent_transparency',
        severity: 'warning',
        message: `Intent reasoning too short (${reasoningText.length} chars, minimum 150)`,
        context: { reasoning: intent.reasoning },
        suggested_action: 'Provide more detailed explanation of intent detection process'
      };
    }

    return null;
  }
}

/**
 * Register CognitiveConstitution with ConstitutionEnforcer
 */
export function registerCognitiveConstitution(enforcer: any) {
  enforcer.constitutions.set('cognitive', new CognitiveConstitution());
}
