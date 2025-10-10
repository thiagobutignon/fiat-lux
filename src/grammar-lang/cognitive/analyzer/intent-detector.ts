/**
 * Enhanced Intent Detector - Context-Aware Analysis
 * Sprint 2: Analysis Layer
 *
 * Extends basic intent detection with:
 * - Contextual awareness (relationship history, power dynamics)
 * - Multi-turn conversation tracking
 * - Intent evolution over time
 * - Confidence calibration based on context
 */

import { Morphemes, Syntax, Semantics, Pragmatics, DetectionResult } from '../types';

// ============================================================
// TYPES
// ============================================================

export interface ConversationContext {
  relationship_type: 'intimate' | 'professional' | 'familial' | 'casual' | 'unknown';
  relationship_duration_years?: number;
  power_dynamic: 'equal' | 'superior' | 'subordinate' | 'unclear';
  history_of_manipulation: boolean;
  previous_detections: DetectionResult[];
  cultural_context?: string;
  conversation_length: number;  // Number of turns
}

export interface EnhancedIntent {
  primary_intent: 'manipulate' | 'control' | 'confuse' | 'dominate' | 'deceive' | 'harm' | 'gaslight' | 'isolate';
  secondary_intents: string[];
  confidence: number;
  context_adjusted_confidence: number;
  reasoning: string[];
  escalation_pattern?: 'increasing' | 'stable' | 'decreasing';
  manipulation_strategy?: 'subtle' | 'overt' | 'intermittent' | 'systematic';
}

export interface ContextualAnalysis {
  intent: EnhancedIntent;
  relationship_risk_score: number;  // 0-1
  temporal_pattern: string;
  predicted_escalation: boolean;
  intervention_urgency: 'low' | 'medium' | 'high' | 'critical';
  context_factors: string[];
}

// ============================================================
// ENHANCED INTENT DETECTION
// ============================================================

/**
 * Detect intent with full contextual awareness
 * Extends basic pragmatics with relationship and temporal context
 */
export function detectEnhancedIntent(
  morphemes: Morphemes,
  syntax: Syntax,
  semantics: Semantics,
  pragmatics: Pragmatics,
  context: ConversationContext
): EnhancedIntent {
  // Base intent from pragmatics
  const baseIntent = pragmatics.intent;

  // Calculate confidence from linguistic markers
  const linguisticConfidence = calculateLinguisticConfidence(
    morphemes,
    syntax,
    semantics,
    pragmatics
  );

  // Adjust confidence based on context
  const contextAdjustment = calculateContextAdjustment(context);
  const contextAdjustedConfidence = Math.min(
    1.0,
    Math.max(0.0, linguisticConfidence * contextAdjustment)
  );

  // Detect secondary intents
  const secondaryIntents = detectSecondaryIntents(
    morphemes,
    syntax,
    semantics,
    baseIntent
  );

  // Build reasoning chain
  const reasoning = buildReasoningChain(
    morphemes,
    syntax,
    semantics,
    pragmatics,
    context
  );

  // Detect escalation pattern
  const escalationPattern = detectEscalationPattern(context);

  // Classify manipulation strategy
  const manipulationStrategy = classifyManipulationStrategy(
    morphemes,
    syntax,
    context
  );

  return {
    primary_intent: baseIntent as any,
    secondary_intents: secondaryIntents,
    confidence: linguisticConfidence,
    context_adjusted_confidence: contextAdjustedConfidence,
    reasoning,
    escalation_pattern: escalationPattern,
    manipulation_strategy: manipulationStrategy
  };
}

/**
 * Calculate linguistic confidence from all 4 layers
 */
function calculateLinguisticConfidence(
  morphemes: Morphemes,
  syntax: Syntax,
  semantics: Semantics,
  pragmatics: Pragmatics
): number {
  let score = 0.0;
  let factors = 0;

  // Morpheme indicators (weight: 0.3)
  if (morphemes.keywords.length > 0) {
    score += 0.3 * Math.min(1.0, morphemes.keywords.length / 3);
    factors++;
  }

  if (morphemes.negations.length > 2) {
    score += 0.1;
    factors++;
  }

  if (morphemes.intensifiers.length > 0) {
    score += 0.05 * morphemes.intensifiers.length;
    factors++;
  }

  // Syntax indicators (weight: 0.2)
  if (syntax.pronoun_reversal) {
    score += 0.15;
    factors++;
  }

  if (syntax.temporal_distortion) {
    score += 0.15;
    factors++;
  }

  if (syntax.modal_manipulation) {
    score += 0.1;
    factors++;
  }

  // Semantics indicators (weight: 0.3)
  if (semantics.reality_denial) {
    score += 0.2;
    factors++;
  }

  if (semantics.memory_invalidation) {
    score += 0.2;
    factors++;
  }

  if (semantics.emotional_dismissal) {
    score += 0.1;
    factors++;
  }

  if (semantics.blame_shifting) {
    score += 0.15;
    factors++;
  }

  if (semantics.projection) {
    score += 0.15;
    factors++;
  }

  // Pragmatics indicators (weight: 0.2)
  if (pragmatics.context_awareness < 0.3) {
    score += 0.2;  // Low context awareness = manipulation indicator
    factors++;
  }

  if (pragmatics.power_dynamic === 'exploit') {
    score += 0.15;
    factors++;
  }

  if (pragmatics.social_impact === 'isolate') {
    score += 0.15;
    factors++;
  }

  return Math.min(1.0, score);
}

/**
 * Calculate context adjustment multiplier
 * Increases confidence if context supports manipulation hypothesis
 * Decreases confidence if context suggests benign interaction
 */
function calculateContextAdjustment(context: ConversationContext): number {
  let adjustment = 1.0;

  // History of manipulation increases confidence
  if (context.history_of_manipulation) {
    adjustment *= 1.3;
  }

  // Previous detections in conversation
  if (context.previous_detections.length > 0) {
    adjustment *= 1.0 + (context.previous_detections.length * 0.1);
  }

  // Power dynamic
  if (context.power_dynamic === 'superior') {
    adjustment *= 1.2;  // Abuser often holds power
  } else if (context.power_dynamic === 'equal') {
    adjustment *= 0.9;  // Less likely in equal relationships
  }

  // Relationship type
  if (context.relationship_type === 'intimate') {
    adjustment *= 1.1;  // Intimate partner abuse is common
  } else if (context.relationship_type === 'casual') {
    adjustment *= 0.8;  // Less likely in casual interactions
  }

  // Conversation length (longer = more data = higher confidence)
  if (context.conversation_length > 5) {
    adjustment *= 1.1;
  } else if (context.conversation_length < 2) {
    adjustment *= 0.7;  // Not enough data
  }

  return adjustment;
}

/**
 * Detect secondary intents beyond primary
 */
function detectSecondaryIntents(
  morphemes: Morphemes,
  syntax: Syntax,
  semantics: Semantics,
  primaryIntent: string
): string[] {
  const secondaryIntents: string[] = [];

  // If primary is manipulate, check for gaslight
  if (primaryIntent === 'manipulate') {
    if (semantics.reality_denial || semantics.memory_invalidation) {
      secondaryIntents.push('gaslight');
    }
  }

  // If primary is control, check for isolate
  if (primaryIntent === 'control') {
    if (morphemes.keywords.some(k => k.includes('no one else') || k.includes('only i'))) {
      secondaryIntents.push('isolate');
    }
  }

  // If primary is confuse, check for deceive
  if (primaryIntent === 'confuse') {
    if (semantics.projection || syntax.temporal_distortion) {
      secondaryIntents.push('deceive');
    }
  }

  // Check for harm intent
  if (morphemes.intensifiers.length > 2 && semantics.emotional_dismissal) {
    secondaryIntents.push('harm');
  }

  // Check for dominate intent
  if (syntax.pronoun_reversal && semantics.blame_shifting) {
    secondaryIntents.push('dominate');
  }

  return secondaryIntents;
}

/**
 * Build reasoning chain explaining intent detection
 * Glass box transparency
 */
function buildReasoningChain(
  morphemes: Morphemes,
  syntax: Syntax,
  semantics: Semantics,
  pragmatics: Pragmatics,
  context: ConversationContext
): string[] {
  const reasoning: string[] = [];

  // Morpheme evidence
  if (morphemes.keywords.length > 0) {
    reasoning.push(`Detected ${morphemes.keywords.length} manipulation keywords: ${morphemes.keywords.slice(0, 3).join(', ')}`);
  }

  if (morphemes.negations.length > 2) {
    reasoning.push(`Excessive negations (${morphemes.negations.length}) suggest reality denial`);
  }

  // Syntax evidence
  if (syntax.pronoun_reversal) {
    reasoning.push('Pronoun reversal detected - shifting blame from self to target');
  }

  if (syntax.temporal_distortion) {
    reasoning.push('Temporal distortion detected - manipulating timeline of events');
  }

  // Semantics evidence
  if (semantics.reality_denial) {
    reasoning.push('Reality denial pattern - denying objective facts');
  }

  if (semantics.memory_invalidation) {
    reasoning.push('Memory invalidation - questioning target\'s recollection');
  }

  if (semantics.projection) {
    reasoning.push('Projection detected - attributing own behavior to target');
  }

  // Pragmatics evidence
  if (pragmatics.power_dynamic === 'exploit') {
    reasoning.push('Power exploitation - leveraging authority/influence');
  }

  if (pragmatics.social_impact === 'isolate') {
    reasoning.push('Isolation intent - separating target from support network');
  }

  // Context evidence
  if (context.history_of_manipulation) {
    reasoning.push('Prior manipulation history increases likelihood');
  }

  if (context.previous_detections.length > 0) {
    reasoning.push(`${context.previous_detections.length} previous techniques detected in conversation`);
  }

  return reasoning;
}

/**
 * Detect escalation pattern from conversation history
 */
function detectEscalationPattern(context: ConversationContext): 'increasing' | 'stable' | 'decreasing' | undefined {
  if (context.previous_detections.length < 2) {
    return undefined;  // Not enough data
  }

  // Calculate average confidence over time
  const firstHalf = context.previous_detections.slice(0, Math.floor(context.previous_detections.length / 2));
  const secondHalf = context.previous_detections.slice(Math.floor(context.previous_detections.length / 2));

  const firstAvg = firstHalf.reduce((sum, d) => sum + d.confidence, 0) / firstHalf.length;
  const secondAvg = secondHalf.reduce((sum, d) => sum + d.confidence, 0) / secondHalf.length;

  const difference = secondAvg - firstAvg;

  if (difference > 0.1) {
    return 'increasing';
  } else if (difference < -0.1) {
    return 'decreasing';
  } else {
    return 'stable';
  }
}

/**
 * Classify manipulation strategy
 */
function classifyManipulationStrategy(
  morphemes: Morphemes,
  syntax: Syntax,
  context: ConversationContext
): 'subtle' | 'overt' | 'intermittent' | 'systematic' {
  // Overt: High linguistic markers, direct aggression
  if (morphemes.intensifiers.length > 3 && morphemes.keywords.length > 5) {
    return 'overt';
  }

  // Systematic: Pattern across multiple turns
  if (context.previous_detections.length > 3) {
    return 'systematic';
  }

  // Intermittent: Sporadic detections
  if (context.conversation_length > 5 && context.previous_detections.length > 0 && context.previous_detections.length < 3) {
    return 'intermittent';
  }

  // Subtle: Low markers but still detected
  return 'subtle';
}

// ============================================================
// CONTEXTUAL ANALYSIS
// ============================================================

/**
 * Full contextual analysis combining intent with risk assessment
 */
export function analyzeWithContext(
  morphemes: Morphemes,
  syntax: Syntax,
  semantics: Semantics,
  pragmatics: Pragmatics,
  context: ConversationContext
): ContextualAnalysis {
  // Enhanced intent detection
  const intent = detectEnhancedIntent(morphemes, syntax, semantics, pragmatics, context);

  // Calculate relationship risk score
  const relationshipRiskScore = calculateRelationshipRisk(intent, context);

  // Detect temporal pattern
  const temporalPattern = detectTemporalPattern(context);

  // Predict escalation
  const predictedEscalation = predictEscalation(intent, context);

  // Determine intervention urgency
  const interventionUrgency = determineInterventionUrgency(
    intent,
    relationshipRiskScore,
    predictedEscalation
  );

  // Identify context factors
  const contextFactors = identifyContextFactors(context);

  return {
    intent,
    relationship_risk_score: relationshipRiskScore,
    temporal_pattern: temporalPattern,
    predicted_escalation: predictedEscalation,
    intervention_urgency: interventionUrgency,
    context_factors: contextFactors
  };
}

/**
 * Calculate relationship risk score (0-1)
 */
function calculateRelationshipRisk(intent: EnhancedIntent, context: ConversationContext): number {
  let risk = 0.0;

  // Base risk from intent confidence
  risk += intent.context_adjusted_confidence * 0.4;

  // History of manipulation
  if (context.history_of_manipulation) {
    risk += 0.3;
  }

  // Power dynamic
  if (context.power_dynamic === 'superior') {
    risk += 0.2;
  }

  // Relationship type (intimate = higher risk)
  if (context.relationship_type === 'intimate') {
    risk += 0.2;
  } else if (context.relationship_type === 'familial') {
    risk += 0.15;
  }

  // Escalation pattern
  if (intent.escalation_pattern === 'increasing') {
    risk += 0.25;
  }

  // Manipulation strategy
  if (intent.manipulation_strategy === 'systematic') {
    risk += 0.2;
  }

  return Math.min(1.0, risk);
}

/**
 * Detect temporal pattern in manipulation
 */
function detectTemporalPattern(context: ConversationContext): string {
  if (context.previous_detections.length === 0) {
    return 'No prior detections';
  }

  if (context.previous_detections.length === 1) {
    return 'Single incident';
  }

  const escalation = detectEscalationPattern(context);

  if (escalation === 'increasing') {
    return 'Escalating pattern - manipulation intensity increasing over time';
  } else if (escalation === 'stable') {
    return 'Consistent pattern - stable manipulation intensity';
  } else if (escalation === 'decreasing') {
    return 'De-escalating pattern - manipulation intensity decreasing';
  }

  return 'Insufficient data for pattern detection';
}

/**
 * Predict if escalation is likely
 */
function predictEscalation(intent: EnhancedIntent, context: ConversationContext): boolean {
  // Already escalating
  if (intent.escalation_pattern === 'increasing') {
    return true;
  }

  // Systematic manipulation with high confidence
  if (intent.manipulation_strategy === 'systematic' && intent.context_adjusted_confidence > 0.8) {
    return true;
  }

  // Multiple secondary intents
  if (intent.secondary_intents.length > 2) {
    return true;
  }

  // History of manipulation + current detection
  if (context.history_of_manipulation && intent.context_adjusted_confidence > 0.7) {
    return true;
  }

  return false;
}

/**
 * Determine intervention urgency level
 */
function determineInterventionUrgency(
  intent: EnhancedIntent,
  relationshipRiskScore: number,
  predictedEscalation: boolean
): 'low' | 'medium' | 'high' | 'critical' {
  // Critical: High confidence, high risk, escalating
  if (intent.context_adjusted_confidence > 0.9 && relationshipRiskScore > 0.8 && predictedEscalation) {
    return 'critical';
  }

  // High: Any 2 of the 3 high-risk factors
  const highRiskFactors = [
    intent.context_adjusted_confidence > 0.8,
    relationshipRiskScore > 0.7,
    predictedEscalation
  ].filter(Boolean).length;

  if (highRiskFactors >= 2) {
    return 'high';
  }

  // Medium: Moderate confidence and risk
  if (intent.context_adjusted_confidence > 0.6 || relationshipRiskScore > 0.5) {
    return 'medium';
  }

  // Low: Low confidence and risk
  return 'low';
}

/**
 * Identify contextual factors influencing analysis
 */
function identifyContextFactors(context: ConversationContext): string[] {
  const factors: string[] = [];

  if (context.history_of_manipulation) {
    factors.push('Prior manipulation history documented');
  }

  if (context.relationship_type === 'intimate') {
    factors.push('Intimate partner relationship (higher risk context)');
  }

  if (context.power_dynamic === 'superior') {
    factors.push('Manipulator holds power advantage');
  }

  if (context.previous_detections.length > 3) {
    factors.push(`${context.previous_detections.length} techniques detected in conversation`);
  }

  if (context.cultural_context) {
    factors.push(`Cultural context: ${context.cultural_context}`);
  }

  if (context.conversation_length > 10) {
    factors.push('Extended conversation provides high confidence');
  } else if (context.conversation_length < 3) {
    factors.push('Limited conversation data - lower confidence');
  }

  return factors;
}
