/**
 * SEMANTICS Parser
 * Analyzes meaning and intent
 * Detects reality denial, memory invalidation, emotional dismissal, etc.
 *
 * Now supports LLM-powered deep semantic analysis for nuanced understanding.
 */

import { Semantics } from '../types';
import { createGlassLLM, GlassLLM } from '../../glass/llm-adapter';

// ============================================================
// SEMANTIC PATTERN DETECTION
// ============================================================

/**
 * Detect reality denial
 * Denying factual events or objective reality
 */
function detectRealityDenial(text: string): boolean {
  const lowerText = text.toLowerCase();

  const denialPatterns = [
    /that (never|didn't) happen/i,
    /that('s| is) not true/i,
    /you('re| are) (making|imagining) (this|that) up/i,
    /that('s| is) (ridiculous|absurd|nonsense)/i,
    /(i|that) never (said|did|happened)/i,
    /you('re| are) lying/i,
    /that didn't exist/i
  ];

  return denialPatterns.some(pattern => pattern.test(text));
}

/**
 * Detect memory invalidation
 * Attacking victim's memory or perception
 */
function detectMemoryInvalidation(text: string): boolean {
  const lowerText = text.toLowerCase();

  const memoryPatterns = [
    /you('re| are) remembering (wrong|incorrectly)/i,
    /that('s| is) not how it happened/i,
    /you have a bad memory/i,
    /you('re| are) (confusing|mixing) (this|things) up/i,
    /i never said that/i,
    /you must be (confused|mistaken)/i,
    /that('s| is) not what (happened|i said|occurred)/i,
    /you don't remember (correctly|right)/i
  ];

  return memoryPatterns.some(pattern => pattern.test(text));
}

/**
 * Detect emotional dismissal
 * Minimizing or invalidating emotions
 */
function detectEmotionalDismissal(text: string): boolean {
  const lowerText = text.toLowerCase();

  const dismissalPatterns = [
    /you('re| are) (too|being|so) sensitive/i,
    /you('re| are) overreacting/i,
    /(stop|don't) be(ing)? so dramatic/i,
    /(calm|settle) down/i,
    /it('s| is) not (that|a) big (deal|thing)/i,
    /you('re| are) being (emotional|irrational)/i,
    /you('re| are) (too|so) emotional/i,
    /why are you so upset/i,
    /there('s| is) no reason to (be|feel)/i
  ];

  return dismissalPatterns.some(pattern => pattern.test(text));
}

/**
 * Detect blame shifting
 * Transferring responsibility to victim
 */
function detectBlameShifting(text: string): boolean {
  const lowerText = text.toLowerCase();

  const blamePatterns = [
    /you (made|forced|caused) me (to|do)/i,
    /this is your fault/i,
    /you('re| are) (the|to) blame/i,
    /if you (hadn't|didn't|wouldn't have)/i,
    /you brought this on yourself/i,
    /you('re| are) (the one|responsible)/i,
    /this wouldn't have happened if you/i,
    /you (always|never) do this/i
  ];

  return blamePatterns.some(pattern => pattern.test(text));
}

/**
 * Detect projection
 * Attributing own behavior/feelings to victim
 */
function detectProjection(text: string): boolean {
  const lowerText = text.toLowerCase();

  const projectionPatterns = [
    /you('re| are) the (one|person) who/i,
    /you('re| are) (doing|being) exactly what/i,
    /you('re| are) the (abusive|controlling|manipulative) one/i,
    /you('re| are) (gaslighting|manipulating) me/i,
    /you('re| are) attacking me/i,
    /you('re| are) (being|the) (aggressive|hostile|mean)/i,
    /you need (help|therapy)/i
  ];

  return projectionPatterns.some(pattern => pattern.test(text));
}

// ============================================================
// MAIN SEMANTICS PARSER
// ============================================================

/**
 * Parse text for semantic patterns
 * O(n) where n = text length
 */
export function parseSemantics(text: string): Semantics {
  return {
    reality_denial: detectRealityDenial(text),
    memory_invalidation: detectMemoryInvalidation(text),
    emotional_dismissal: detectEmotionalDismissal(text),
    blame_shifting: detectBlameShifting(text),
    projection: detectProjection(text)
  };
}

/**
 * Parse semantics using LLM deep understanding
 * Goes beyond pattern matching to analyze true meaning and implicit messages
 */
export async function parseSemanticsWithLLM(
  text: string,
  llm: GlassLLM
): Promise<{
  semantics: Semantics;
  confidence: number;
  implicit_meanings: string[];
  reasoning: string;
}> {
  const prompt = `Perform deep semantic analysis of this text:

**Text**: "${text}"

**Task**: Analyze the semantic meaning beyond surface patterns. Detect:

1. **Reality Denial**: Denying factual events or objective reality
   - Examples: "That never happened", "You're imagining things"

2. **Memory Invalidation**: Attacking victim's memory or perception
   - Examples: "You're remembering wrong", "That's not what I said"

3. **Emotional Dismissal**: Minimizing or invalidating emotions
   - Examples: "You're too sensitive", "You're overreacting"

4. **Blame Shifting**: Transferring responsibility to victim
   - Examples: "You made me do it", "This is your fault"

5. **Projection**: Attributing own behavior/feelings to victim
   - Examples: "You're the manipulative one", "You need therapy"

**Important**: Look beyond exact phrases - analyze implicit meaning, context, and subtext.

Return JSON:
\`\`\`json
{
  "reality_denial": true/false,
  "memory_invalidation": true/false,
  "emotional_dismissal": true/false,
  "blame_shifting": true/false,
  "projection": true/false,
  "confidence": 0.85,
  "implicit_meanings": [
    "What the text implicitly communicates",
    "Subtext and hidden messages"
  ],
  "reasoning": "Brief explanation of semantic analysis"
}
\`\`\``;

  try {
    const response = await llm.invoke(prompt, {
      task: 'semantic-analysis',
      max_tokens: 800,
      enable_constitutional: true
    });

    // Parse LLM response
    const jsonMatch = response.text.match(/```(?:json)?\n([\s\S]*?)\n```/);
    if (jsonMatch) {
      const data = JSON.parse(jsonMatch[1]);

      const semantics: Semantics = {
        reality_denial: data.reality_denial || false,
        memory_invalidation: data.memory_invalidation || false,
        emotional_dismissal: data.emotional_dismissal || false,
        blame_shifting: data.blame_shifting || false,
        projection: data.projection || false
      };

      return {
        semantics,
        confidence: data.confidence || 0.7,
        implicit_meanings: data.implicit_meanings || [],
        reasoning: data.reasoning || 'LLM semantic analysis'
      };
    }
  } catch (error) {
    console.warn('⚠️  LLM semantic analysis failed:', error);
  }

  // Fallback to rule-based
  const fallbackSemantics = parseSemantics(text);
  return {
    semantics: fallbackSemantics,
    confidence: 0.6,
    implicit_meanings: [],
    reasoning: 'Fallback to regex pattern matching'
  };
}

/**
 * Calculate semantics match score
 * Returns 0-1 indicating how well semantics match a pattern
 */
export function calculateSemanticsScore(
  detected: Semantics,
  pattern: Semantics
): number {
  let score = 0;
  let checks = 0;

  // Each semantic marker carries equal weight (20% each)
  if (pattern.reality_denial === detected.reality_denial && pattern.reality_denial) {
    score += 0.2;
  }
  checks += 0.2;

  if (pattern.memory_invalidation === detected.memory_invalidation && pattern.memory_invalidation) {
    score += 0.2;
  }
  checks += 0.2;

  if (pattern.emotional_dismissal === detected.emotional_dismissal && pattern.emotional_dismissal) {
    score += 0.2;
  }
  checks += 0.2;

  if (pattern.blame_shifting === detected.blame_shifting && pattern.blame_shifting) {
    score += 0.2;
  }
  checks += 0.2;

  if (pattern.projection === detected.projection && pattern.projection) {
    score += 0.2;
  }
  checks += 0.2;

  return checks > 0 ? score / checks : 0;
}

/**
 * Get semantics statistics
 */
export function getSemanticsStats(semantics: Semantics) {
  return {
    reality_denial: semantics.reality_denial,
    memory_invalidation: semantics.memory_invalidation,
    emotional_dismissal: semantics.emotional_dismissal,
    blame_shifting: semantics.blame_shifting,
    projection: semantics.projection,
    manipulation_indicators:
      (semantics.reality_denial ? 1 : 0) +
      (semantics.memory_invalidation ? 1 : 0) +
      (semantics.emotional_dismissal ? 1 : 0) +
      (semantics.blame_shifting ? 1 : 0) +
      (semantics.projection ? 1 : 0),
    severity: calculateSeverity(semantics)
  };
}

/**
 * Calculate severity of semantic manipulation
 * 0-1 scale
 */
function calculateSeverity(semantics: Semantics): number {
  // Weight different semantic markers
  const weights = {
    reality_denial: 0.25,        // Severe
    memory_invalidation: 0.25,   // Severe
    emotional_dismissal: 0.15,   // Moderate
    blame_shifting: 0.2,         // Moderate-severe
    projection: 0.15             // Moderate
  };

  let severity = 0;

  if (semantics.reality_denial) severity += weights.reality_denial;
  if (semantics.memory_invalidation) severity += weights.memory_invalidation;
  if (semantics.emotional_dismissal) severity += weights.emotional_dismissal;
  if (semantics.blame_shifting) severity += weights.blame_shifting;
  if (semantics.projection) severity += weights.projection;

  return severity;
}

/**
 * Determine manipulation category from semantics
 */
export function categorizeSemantics(semantics: Semantics): string[] {
  const categories: string[] = [];

  if (semantics.reality_denial || semantics.memory_invalidation) {
    categories.push('gaslighting');
  }

  if (semantics.blame_shifting && semantics.projection) {
    categories.push('darvo');
  }

  if (semantics.emotional_dismissal) {
    categories.push('invalidation');
  }

  return categories;
}
