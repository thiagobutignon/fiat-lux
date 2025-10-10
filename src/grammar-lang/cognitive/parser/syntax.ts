/**
 * SYNTAX Parser
 * Analyzes grammatical structure patterns
 * Detects pronoun reversal, temporal distortion, passive voice, etc.
 */

import { Syntax } from '../types';

// ============================================================
// SYNTAX PATTERN DETECTION
// ============================================================

/**
 * Detect pronoun reversal (shifting blame via pronouns)
 * "I didn't" → "You did"
 */
function detectPronounReversal(text: string): boolean {
  const lowerText = text.toLowerCase();

  // Pattern: Accusatory "you" followed by action verb
  const youAccusations = /you (always|never|constantly|are|were|did|do|have|had)/i;

  // Pattern: "I'm the victim" reversal
  const victimReversal = /i('m| am) (the victim|being hurt|being attacked)/i;

  // Pattern: Deflection from "I" to "you"
  const deflection = /(i didn't|i never).*(you did|you always|you were)/i;

  return (
    youAccusations.test(text) ||
    victimReversal.test(text) ||
    deflection.test(text)
  );
}

/**
 * Detect temporal distortion (manipulating timeline)
 * Changing when events occurred or denying timing
 */
function detectTemporalDistortion(text: string): boolean {
  const lowerText = text.toLowerCase();

  // Denial of timeline
  const timelineDenial = /(that (never|didn't) happen|that wasn't (yesterday|today|last week))/i;

  // Vague temporal references to create confusion
  const vagueTime = /(i think it was|maybe it was|i don't remember when|that was a long time ago)/i;

  // Changing sequence of events
  const sequenceChange = /(actually, (first|before that)|no, that happened (after|before))/i;

  return (
    timelineDenial.test(text) ||
    vagueTime.test(text) ||
    sequenceChange.test(text)
  );
}

/**
 * Detect modal manipulation
 * Misuse of "could", "should", "would", "might"
 */
function detectModalManipulation(text: string): boolean {
  const lowerText = text.toLowerCase();

  // Deflective modals
  const deflectiveModals = /(you (should|could|would|might) have|you (should|could|would) be)/i;

  // Hypothetical distortion
  const hypothetical = /(i (would never|could never)|if i (did|had|were))/i;

  // Obligation shifting
  const obligationShift = /(you (should|need to|have to|must))/i;

  return (
    deflectiveModals.test(text) ||
    hypothetical.test(text) ||
    obligationShift.test(text)
  );
}

/**
 * Detect passive voice (avoiding responsibility)
 * "Mistakes were made" instead of "I made mistakes"
 */
function detectPassiveVoice(text: string): boolean {
  // Pattern: [be verb] + [past participle] without clear subject
  const passivePatterns = [
    /mistakes were made/i,
    /things were said/i,
    /it was done/i,
    /(feelings|people) were hurt/i,
    /that was said/i,
    /(something|that) happened/i
  ];

  return passivePatterns.some(pattern => pattern.test(text));
}

/**
 * Extract leading/rhetorical question patterns
 */
function extractQuestionPatterns(text: string): string[] {
  const patterns: string[] = [];

  // Leading questions
  const leadingQuestions = [
    /don't you think/i,
    /wouldn't you say/i,
    /isn't it (true|obvious|clear)/i,
    /don't you agree/i,
    /how could you/i,
    /why would (i|you)/i,
    /are you (sure|certain)/i,
    /did that really happen/i
  ];

  for (const pattern of leadingQuestions) {
    const match = text.match(pattern);
    if (match) {
      patterns.push(match[0]);
    }
  }

  return patterns;
}

// ============================================================
// MAIN SYNTAX PARSER
// ============================================================

/**
 * Parse text for syntactic patterns
 * O(n) where n = text length
 */
export function parseSyntax(text: string): Syntax {
  return {
    pronoun_reversal: detectPronounReversal(text),
    temporal_distortion: detectTemporalDistortion(text),
    modal_manipulation: detectModalManipulation(text),
    passive_voice: detectPassiveVoice(text),
    question_patterns: extractQuestionPatterns(text)
  };
}

/**
 * Calculate syntax match score
 * Returns 0-1 indicating how well syntax matches a pattern
 */
export function calculateSyntaxScore(
  detected: Syntax,
  pattern: Syntax
): number {
  let score = 0;
  let checks = 0;

  // Check boolean patterns (20% each)
  if (pattern.pronoun_reversal === detected.pronoun_reversal && pattern.pronoun_reversal) {
    score += 0.2;
  }
  checks += 0.2;

  if (pattern.temporal_distortion === detected.temporal_distortion && pattern.temporal_distortion) {
    score += 0.2;
  }
  checks += 0.2;

  if (pattern.modal_manipulation === detected.modal_manipulation && pattern.modal_manipulation) {
    score += 0.2;
  }
  checks += 0.2;

  if (pattern.passive_voice === detected.passive_voice && pattern.passive_voice) {
    score += 0.2;
  }
  checks += 0.2;

  // Check question patterns (20%)
  if (pattern.question_patterns.length > 0 && detected.question_patterns.length > 0) {
    score += 0.2;
  }
  checks += 0.2;

  return checks > 0 ? score / checks : 0;
}

/**
 * Get syntax statistics
 */
export function getSyntaxStats(syntax: Syntax) {
  return {
    pronoun_reversal: syntax.pronoun_reversal,
    temporal_distortion: syntax.temporal_distortion,
    modal_manipulation: syntax.modal_manipulation,
    passive_voice: syntax.passive_voice,
    question_pattern_count: syntax.question_patterns.length,
    manipulation_indicators:
      (syntax.pronoun_reversal ? 1 : 0) +
      (syntax.temporal_distortion ? 1 : 0) +
      (syntax.modal_manipulation ? 1 : 0) +
      (syntax.passive_voice ? 1 : 0) +
      (syntax.question_patterns.length > 0 ? 1 : 0)
  };
}
