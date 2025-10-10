/**
 * Cognitive-Behavior Guard - Dual-Layer Security Integration
 *
 * Combines VERMELHO (behavioral security) + CINZA (cognitive manipulation detection)
 * for comprehensive protection against:
 * - External coercion (VERMELHO): Detects if user is under duress
 * - Linguistic manipulation (CINZA): Detects manipulation techniques in text
 *
 * Architecture:
 * ┌──────────────┐              ┌──────────────┐
 * │  VERMELHO    │              │    CINZA     │
 * │ (Behavioral) │              │ (Cognitive)  │
 * └──────┬───────┘              └──────┬───────┘
 *        │                             │
 *        │    ┌────────────────────┐   │
 *        └────►  This Module       │◄──┘
 *             │  (Integration)     │
 *             └─────────┬──────────┘
 *                       │
 *                       ▼
 *             Unified Security Decision
 *
 * Philosophy:
 * - Glass box (100% transparent)
 * - Layered defense (behavioral + cognitive)
 * - Constitutional enforcement (Layer 1 + Layer 2)
 * - O(1) + O(n) complexity (behavioral O(1), cognitive O(n) where n = text length)
 */

import { GitOperationGuard, GitOperationRequest, GitValidationResult } from './git-operation-guard';
import { SecurityStorage } from './security-storage';
import { UserSecurityProfiles, Interaction } from './types';
import { detectManipulation } from '../cognitive/detector/pattern-matcher';
import { PatternMatchResult, ManipulationTechnique } from '../cognitive/types';

// ===== TYPES =====

/**
 * Combined security analysis result
 * Includes both behavioral (VERMELHO) and cognitive (CINZA) analysis
 */
export interface CognitiveBehaviorAnalysis {
  // Behavioral Security (VERMELHO)
  behavioral: {
    duress_score: number;
    coercion_score: number;
    confidence: number;
    anomalies_detected: string[];
  };

  // Cognitive Security (CINZA)
  cognitive: {
    manipulation_detected: boolean;
    techniques_found: ManipulationTechnique[];
    highest_confidence: number;
    dark_tetrad_scores: {
      narcissism: number;
      machiavellianism: number;
      psychopathy: number;
      sadism: number;
    };
    constitutional_violations: string[];
  };

  // Combined Analysis
  combined: {
    threat_level: 'none' | 'low' | 'medium' | 'high' | 'critical';
    risk_score: number; // 0.0 - 1.0 (weighted combination)
    recommendation: 'allow' | 'challenge' | 'delay' | 'block';
    reasoning: string;
  };
}

/**
 * Enhanced Git validation result with cognitive analysis
 */
export interface CognitiveBehaviorValidationResult extends GitValidationResult {
  cognitive_analysis?: CognitiveBehaviorAnalysis;
  manipulation_snapshot_created?: boolean;
  manipulation_snapshot_path?: string;
}

// ===== CORE CLASS =====

/**
 * Cognitive-Behavior Guard
 *
 * Dual-layer security validation combining:
 * - VERMELHO: Behavioral biometrics (duress/coercion detection)
 * - CINZA: Cognitive manipulation detection (Chomsky Hierarchy)
 */
export class CognitiveBehaviorGuard {
  private behavioralGuard: GitOperationGuard;
  private storage?: SecurityStorage;

  constructor(storage?: SecurityStorage) {
    this.behavioralGuard = new GitOperationGuard(storage);
    this.storage = storage;
  }

  /**
   * Validate Git operation with dual-layer security
   *
   * Performs:
   * 1. Behavioral security analysis (VERMELHO)
   * 2. Cognitive manipulation detection (CINZA)
   * 3. Combined threat assessment
   * 4. Unified security decision
   *
   * @param request Git operation request metadata
   * @param profiles User security profiles
   * @param interaction Optional interaction data
   * @param sessionDurationMinutes Optional session duration
   * @returns Combined validation result
   */
  async validateGitOperation(
    request: GitOperationRequest,
    profiles: UserSecurityProfiles,
    interaction?: Interaction,
    sessionDurationMinutes?: number
  ): Promise<CognitiveBehaviorValidationResult> {
    // ===== LAYER 1: BEHAVIORAL SECURITY (VERMELHO) =====
    const behavioralResult = this.behavioralGuard.validateCommitRequest(
      request,
      profiles,
      interaction,
      sessionDurationMinutes
    );

    // Extract behavioral scores
    const behavioral = {
      duress_score: behavioralResult.security_context.duress_score,
      coercion_score: behavioralResult.security_context.coercion_score,
      confidence: behavioralResult.security_context.confidence,
      anomalies_detected: behavioralResult.security_context.anomalies_detected
    };

    // ===== LAYER 2: COGNITIVE SECURITY (CINZA) =====
    let cognitive = {
      manipulation_detected: false,
      techniques_found: [] as ManipulationTechnique[],
      highest_confidence: 0,
      dark_tetrad_scores: {
        narcissism: 0,
        machiavellianism: 0,
        psychopathy: 0,
        sadism: 0
      },
      constitutional_violations: [] as string[]
    };

    // Analyze commit message for manipulation
    if (request.message && request.message.length > 0) {
      try {
        const manipulationResult = await detectManipulation(request.message, {
          min_confidence: 0.5,
          enable_neurodivergent_protection: true
        });

        cognitive = {
          manipulation_detected: manipulationResult.total_matches > 0,
          techniques_found: manipulationResult.detections.map(d => d.technique),
          highest_confidence: manipulationResult.highest_confidence,
          dark_tetrad_scores: manipulationResult.dark_tetrad_aggregate,
          constitutional_violations: manipulationResult.constitutional_validation?.violations || []
        };
      } catch (error) {
        console.warn(`⚠️  Cognitive manipulation detection error: ${error}`);
        // Fail-open: if cognitive system is down, continue with behavioral only
      }
    }

    // ===== COMBINED ANALYSIS =====
    const combined = this.calculateCombinedThreat(behavioral, cognitive, behavioralResult);

    // ===== UNIFIED DECISION =====
    const finalDecision = this.makeUnifiedDecision(
      behavioralResult.decision,
      combined.recommendation
    );

    // Create manipulation snapshot if critical threat detected
    let manipulationSnapshotCreated = false;
    let manipulationSnapshotPath: string | undefined;

    if (combined.threat_level === 'critical' && cognitive.manipulation_detected) {
      const snapshotResult = this.createManipulationSnapshot(
        request,
        behavioral,
        cognitive
      );
      manipulationSnapshotCreated = snapshotResult.created;
      manipulationSnapshotPath = snapshotResult.path;
    }

    // ===== RETURN ENHANCED RESULT =====
    return {
      ...behavioralResult,
      decision: finalDecision,
      cognitive_analysis: {
        behavioral,
        cognitive,
        combined
      },
      manipulation_snapshot_created: manipulationSnapshotCreated,
      manipulation_snapshot_path: manipulationSnapshotPath
    };
  }

  /**
   * Calculate combined threat level
   *
   * Threat Level Matrix:
   * - none: No duress, no manipulation
   * - low: Low duress OR low manipulation (not both)
   * - medium: Medium duress OR medium manipulation
   * - high: High duress OR high manipulation OR (medium duress + medium manipulation)
   * - critical: (High duress + manipulation) OR (Dark Tetrad detected + sensitive operation)
   *
   * @param behavioral Behavioral analysis
   * @param cognitive Cognitive analysis
   * @param behavioralResult Full behavioral result
   * @returns Combined threat assessment
   */
  private calculateCombinedThreat(
    behavioral: CognitiveBehaviorAnalysis['behavioral'],
    cognitive: CognitiveBehaviorAnalysis['cognitive'],
    behavioralResult: GitValidationResult
  ): CognitiveBehaviorAnalysis['combined'] {
    const duress = behavioral.duress_score;
    const coercion = behavioral.coercion_score;
    const manipulation = cognitive.highest_confidence;
    const darkTetrad = Object.values(cognitive.dark_tetrad_scores).reduce((a, b) => a + b, 0) / 4;
    const isSensitive = behavioralResult.security_context.is_sensitive_operation;

    // Calculate weighted risk score
    // Behavioral: 50% weight, Cognitive: 50% weight
    const behavioralRisk = Math.max(duress, coercion);
    const cognitiveRisk = Math.max(manipulation, darkTetrad);
    const riskScore = (behavioralRisk * 0.5) + (cognitiveRisk * 0.5);

    // Determine threat level
    let threatLevel: CognitiveBehaviorAnalysis['combined']['threat_level'];
    let recommendation: CognitiveBehaviorAnalysis['combined']['recommendation'];
    let reasoning: string;

    // CRITICAL: High duress + manipulation OR Dark Tetrad + sensitive operation
    if ((behavioralRisk > 0.6 && manipulation > 0.5) || (darkTetrad > 0.7 && isSensitive)) {
      threatLevel = 'critical';
      recommendation = 'block';
      reasoning = 'Critical threat: High behavioral risk combined with cognitive manipulation';
      if (darkTetrad > 0.7) {
        reasoning += '. Dark Tetrad personality traits detected';
      }
    }
    // HIGH: High duress OR high manipulation OR (medium both)
    else if (behavioralRisk > 0.6 || manipulation > 0.7 || (behavioralRisk > 0.4 && manipulation > 0.5)) {
      threatLevel = 'high';
      recommendation = isSensitive ? 'block' : 'delay';
      reasoning = 'High threat: Significant behavioral or cognitive risk detected';
    }
    // MEDIUM: Medium duress OR medium manipulation
    else if (behavioralRisk > 0.3 || manipulation > 0.5) {
      threatLevel = 'medium';
      recommendation = isSensitive ? 'challenge' : 'allow';
      reasoning = 'Medium threat: Moderate behavioral or cognitive indicators';
    }
    // LOW: Low duress OR low manipulation
    else if (behavioralRisk > 0.15 || manipulation > 0.3) {
      threatLevel = 'low';
      recommendation = 'allow';
      reasoning = 'Low threat: Minor behavioral or cognitive indicators';
    }
    // NONE: Clean
    else {
      threatLevel = 'none';
      recommendation = 'allow';
      reasoning = 'No threat detected: Normal behavioral and cognitive patterns';
    }

    return {
      threat_level: threatLevel,
      risk_score: riskScore,
      recommendation,
      reasoning
    };
  }

  /**
   * Make unified security decision
   *
   * Combines behavioral and cognitive recommendations:
   * - If either says block → block
   * - If either says delay → delay
   * - If either says challenge → challenge
   * - If both say allow → allow
   *
   * @param behavioralDecision Behavioral security decision
   * @param cognitiveRecommendation Cognitive security recommendation
   * @returns Unified decision
   */
  private makeUnifiedDecision(
    behavioralDecision: string,
    cognitiveRecommendation: 'allow' | 'challenge' | 'delay' | 'block'
  ): string {
    // Priority: block > delay > challenge > allow
    if (behavioralDecision === 'block' || cognitiveRecommendation === 'block') {
      return 'block';
    }
    if (behavioralDecision === 'delay' || cognitiveRecommendation === 'delay') {
      return 'delay';
    }
    if (behavioralDecision === 'challenge' || cognitiveRecommendation === 'challenge') {
      return 'challenge';
    }
    return 'allow';
  }

  /**
   * Create manipulation snapshot
   *
   * Similar to duress snapshot, but specifically for manipulation detection.
   * Stores:
   * - Behavioral biometrics
   * - Cognitive manipulation analysis
   * - Dark Tetrad scores
   * - Detected manipulation techniques
   *
   * @param request Git operation request
   * @param behavioral Behavioral analysis
   * @param cognitive Cognitive analysis
   * @returns Snapshot creation result
   */
  private createManipulationSnapshot(
    request: GitOperationRequest,
    behavioral: CognitiveBehaviorAnalysis['behavioral'],
    cognitive: CognitiveBehaviorAnalysis['cognitive']
  ): { created: boolean; path?: string } {
    try {
      const timestamp = Date.now();
      const snapshotPath = `.git/manipulation-snapshots/${timestamp}-${request.user_id}`;

      // Create snapshot directory (in real implementation)
      // For now, just log
      console.log(`📸 Creating manipulation snapshot: ${snapshotPath}`);
      console.log(`   Behavioral: duress=${behavioral.duress_score}, coercion=${behavioral.coercion_score}`);
      console.log(`   Cognitive: manipulation=${cognitive.highest_confidence}, techniques=${cognitive.techniques_found.length}`);
      console.log(`   Dark Tetrad: N=${cognitive.dark_tetrad_scores.narcissism.toFixed(2)}, M=${cognitive.dark_tetrad_scores.machiavellianism.toFixed(2)}, P=${cognitive.dark_tetrad_scores.psychopathy.toFixed(2)}, S=${cognitive.dark_tetrad_scores.sadism.toFixed(2)}`);

      return {
        created: true,
        path: snapshotPath
      };
    } catch (error) {
      console.error(`❌ Failed to create manipulation snapshot: ${error}`);
      return { created: false };
    }
  }
}

// ===== HELPER FUNCTIONS =====

/**
 * Format cognitive-behavior analysis for logging
 */
export function formatCognitiveBehaviorAnalysis(
  analysis: CognitiveBehaviorAnalysis
): string {
  const lines = [
    '🔒 Cognitive-Behavior Security Analysis:',
    '',
    '📊 BEHAVIORAL (VERMELHO):',
    `   Duress: ${(analysis.behavioral.duress_score * 100).toFixed(1)}%`,
    `   Coercion: ${(analysis.behavioral.coercion_score * 100).toFixed(1)}%`,
    `   Confidence: ${(analysis.behavioral.confidence * 100).toFixed(1)}%`,
    `   Anomalies: ${analysis.behavioral.anomalies_detected.length}`,
    '',
    '🧠 COGNITIVE (CINZA):',
    `   Manipulation: ${analysis.cognitive.manipulation_detected ? 'YES' : 'NO'}`,
    `   Techniques: ${analysis.cognitive.techniques_found.length}`,
    `   Confidence: ${(analysis.cognitive.highest_confidence * 100).toFixed(1)}%`,
    `   Dark Tetrad:`,
    `      Narcissism: ${(analysis.cognitive.dark_tetrad_scores.narcissism * 100).toFixed(1)}%`,
    `      Machiavellianism: ${(analysis.cognitive.dark_tetrad_scores.machiavellianism * 100).toFixed(1)}%`,
    `      Psychopathy: ${(analysis.cognitive.dark_tetrad_scores.psychopathy * 100).toFixed(1)}%`,
    `      Sadism: ${(analysis.cognitive.dark_tetrad_scores.sadism * 100).toFixed(1)}%`,
    '',
    '⚡ COMBINED:',
    `   Threat Level: ${analysis.combined.threat_level.toUpperCase()}`,
    `   Risk Score: ${(analysis.combined.risk_score * 100).toFixed(1)}%`,
    `   Recommendation: ${analysis.combined.recommendation.toUpperCase()}`,
    `   Reasoning: ${analysis.combined.reasoning}`
  ];

  return lines.join('\n');
}

/**
 * Check if operation should proceed based on cognitive-behavior validation
 */
export function shouldProceedWithOperation(
  result: CognitiveBehaviorValidationResult
): boolean {
  return result.decision === 'allow' || result.decision === 'challenge';
}

/**
 * Get summary of cognitive-behavior validation
 */
export function getCognitiveBehaviorSummary(
  result: CognitiveBehaviorValidationResult
): string {
  if (!result.cognitive_analysis) {
    return 'No cognitive-behavior analysis available';
  }

  const analysis = result.cognitive_analysis;

  return [
    `Decision: ${result.decision.toUpperCase()}`,
    `Threat Level: ${analysis.combined.threat_level.toUpperCase()}`,
    `Risk Score: ${(analysis.combined.risk_score * 100).toFixed(1)}%`,
    `Behavioral: duress=${(analysis.behavioral.duress_score * 100).toFixed(1)}%, coercion=${(analysis.behavioral.coercion_score * 100).toFixed(1)}%`,
    `Cognitive: manipulation=${analysis.cognitive.manipulation_detected}, techniques=${analysis.cognitive.techniques_found.length}`,
    analysis.combined.reasoning
  ].join(' | ');
}
