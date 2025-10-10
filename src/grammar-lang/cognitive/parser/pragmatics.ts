/**
 * PRAGMATICS Parser
 * Analyzes intent, context, power dynamics, social impact
 * Highest level of linguistic analysis - determines manipulation intent
 *
 * Now supports LLM-powered intent analysis for deeper semantic understanding.
 */

import { Pragmatics, Morphemes, Syntax, Semantics } from '../types';
import { createGlassLLM, GlassLLM } from '../../glass/llm-adapter';

// ============================================================
// INTENT DETECTION
// ============================================================

/**
 * Detect manipulation intent from linguistic patterns
 * Combines morphemes, syntax, semantics to infer intent
 */
export function detectIntent(
  morphemes: Morphemes,
  syntax: Syntax,
  semantics: Semantics
): Pragmatics['intent'] {
  // High semantic severity + denial patterns = manipulate
  if ((semantics.reality_denial || semantics.memory_invalidation) &&
      morphemes.negations.length > 2) {
    return 'manipulate';
  }

  // Projection + blame shifting = control
  if (semantics.projection && semantics.blame_shifting) {
    return 'control';
  }

  // Question patterns + qualifiers = confuse
  if (syntax.question_patterns.length > 2 && morphemes.qualifiers.length > 2) {
    return 'confuse';
  }

  // Intensifiers + emotional dismissal = dominate
  if (morphemes.intensifiers.length > 2 && semantics.emotional_dismissal) {
    return 'dominate';
  }

  // Temporal distortion + modal manipulation = deceive
  if (syntax.temporal_distortion && syntax.modal_manipulation) {
    return 'deceive';
  }

  // Default to manipulate if multiple semantic flags
  const semanticCount = [
    semantics.reality_denial,
    semantics.memory_invalidation,
    semantics.emotional_dismissal,
    semantics.blame_shifting,
    semantics.projection
  ].filter(Boolean).length;

  if (semanticCount >= 2) {
    return 'manipulate';
  }

  return 'manipulate';  // Default for unclassified patterns
}

/**
 * Detect intent using LLM semantic analysis
 * Provides deeper understanding of communicative intent
 */
export async function detectIntentWithLLM(
  text: string,
  morphemes: Morphemes,
  syntax: Syntax,
  semantics: Semantics,
  llm: GlassLLM
): Promise<{
  intent: Pragmatics['intent'];
  confidence: number;
  reasoning: string[];
}> {
  const prompt = `Analyze the communicative intent of this text using pragmatic analysis:

**Text**: "${text}"

**Linguistic Context**:
- Negations: ${morphemes.negations.length}
- Intensifiers: ${morphemes.intensifiers.length}
- Qualifiers: ${morphemes.qualifiers.length}
- Question patterns: ${syntax.question_patterns.length}
- Temporal distortion: ${syntax.temporal_distortion ? 'yes' : 'no'}
- Reality denial: ${semantics.reality_denial ? 'yes' : 'no'}
- Memory invalidation: ${semantics.memory_invalidation ? 'yes' : 'no'}
- Emotional dismissal: ${semantics.emotional_dismissal ? 'yes' : 'no'}
- Blame shifting: ${semantics.blame_shifting ? 'yes' : 'no'}
- Projection: ${semantics.projection ? 'yes' : 'no'}

**Task**: Determine the primary communicative intent:
- **manipulate**: Distorting reality to control someone's perception
- **control**: Exercising power to restrict autonomy
- **deceive**: Intentionally misleading about facts or events
- **confuse**: Creating uncertainty to undermine clarity
- **dominate**: Asserting superiority through dismissal
- **harm**: Causing psychological damage

Consider:
1. Power dynamics (who has power, who is being controlled)
2. Context dependencies (relationship history, social context)
3. Gricean maxims violations (quality, quantity, relation, manner)
4. Speech act theory (illocutionary force vs perlocutionary effect)

Return JSON:
\`\`\`json
{
  "intent": "manipulate" | "control" | "deceive" | "confuse" | "dominate" | "harm",
  "confidence": 0.85,
  "reasoning": [
    "Step 1: Analysis of linguistic markers",
    "Step 2: Context and power dynamics",
    "Step 3: Final intent classification"
  ]
}
\`\`\``;

  try {
    const response = await llm.invoke(prompt, {
      task: 'intent-analysis',
      max_tokens: 800,
      enable_constitutional: true
    });

    // Parse LLM response
    const jsonMatch = response.text.match(/```(?:json)?\n([\s\S]*?)\n```/);
    if (jsonMatch) {
      const data = JSON.parse(jsonMatch[1]);

      // Validate intent value
      const validIntents: Pragmatics['intent'][] = [
        'manipulate', 'control', 'deceive', 'confuse', 'dominate', 'harm'
      ];

      if (validIntents.includes(data.intent)) {
        return {
          intent: data.intent,
          confidence: data.confidence || 0.7,
          reasoning: data.reasoning || []
        };
      }
    }
  } catch (error) {
    console.warn('⚠️  LLM intent detection failed:', error);
  }

  // Fallback to rule-based
  const fallbackIntent = detectIntent(morphemes, syntax, semantics);
  return {
    intent: fallbackIntent,
    confidence: 0.6,
    reasoning: ['Fallback to rule-based detection']
  };
}

/**
 * Calculate context awareness
 * How dependent is the manipulation on specific context?
 * 0 = works anywhere, 1 = highly context-specific
 */
export function calculateContextAwareness(
  morphemes: Morphemes,
  syntax: Syntax
): number {
  let contextScore = 0;

  // Generic manipulation phrases = low context dependency
  const genericKeywords = [
    'that never happened',
    'you\'re crazy',
    'you\'re too sensitive'
  ];

  const hasGeneric = morphemes.keywords.some(k =>
    genericKeywords.includes(k)
  );

  if (hasGeneric) {
    contextScore -= 0.3;  // Less context-dependent
  }

  // Specific references require context
  if (syntax.temporal_distortion) {
    contextScore += 0.3;  // References specific events
  }

  // Leading questions are context-dependent
  if (syntax.question_patterns.length > 0) {
    contextScore += 0.2;
  }

  // Normalize to 0-1
  return Math.max(0, Math.min(1, 0.5 + contextScore));
}

/**
 * Detect power dynamic manipulation
 */
export function detectPowerDynamic(
  syntax: Syntax,
  semantics: Semantics
): Pragmatics['power_dynamic'] {
  // DARVO pattern = reverse power dynamic
  if (semantics.projection && semantics.blame_shifting) {
    return 'reverse';
  }

  // Pronoun reversal = exploit existing power
  if (syntax.pronoun_reversal) {
    return 'exploit';
  }

  // Default to exploit
  return 'exploit';
}

/**
 * Detect social impact pattern
 */
export function detectSocialImpact(
  morphemes: Morphemes,
  semantics: Semantics
): Pragmatics['social_impact'] {
  // Triangulation keywords = triangulate
  const triangulationMarkers = ['unlike you', 'understands me better', 'at least they'];
  if (morphemes.keywords.some(k => triangulationMarkers.includes(k))) {
    return 'triangulate';
  }

  // DARVO pattern = recruit sympathy
  if (semantics.projection && semantics.blame_shifting) {
    return 'recruit';
  }

  // Reality denial + memory invalidation = isolate victim
  if (semantics.reality_denial || semantics.memory_invalidation) {
    return 'isolate';
  }

  // Blame shifting = divide
  if (semantics.blame_shifting) {
    return 'divide';
  }

  // Default to isolate
  return 'isolate';
}

// ============================================================
// MAIN PRAGMATICS PARSER
// ============================================================

/**
 * Parse pragmatics from linguistic analysis
 * Combines all lower levels to determine intent
 */
export function parsePragmatics(
  morphemes: Morphemes,
  syntax: Syntax,
  semantics: Semantics
): Pragmatics {
  return {
    intent: detectIntent(morphemes, syntax, semantics),
    context_awareness: calculateContextAwareness(morphemes, syntax),
    power_dynamic: detectPowerDynamic(syntax, semantics),
    social_impact: detectSocialImpact(morphemes, semantics)
  };
}

/**
 * Parse pragmatics using LLM-powered analysis
 * Provides deeper semantic understanding of intent
 */
export async function parsePragmaticsWithLLM(
  text: string,
  morphemes: Morphemes,
  syntax: Syntax,
  semantics: Semantics,
  llm: GlassLLM
): Promise<{
  pragmatics: Pragmatics;
  confidence: number;
  reasoning: string[];
}> {
  // Use LLM for intent detection
  const intentResult = await detectIntentWithLLM(text, morphemes, syntax, semantics, llm);

  // Use rule-based for other components (fast and accurate)
  const pragmatics: Pragmatics = {
    intent: intentResult.intent,
    context_awareness: calculateContextAwareness(morphemes, syntax),
    power_dynamic: detectPowerDynamic(syntax, semantics),
    social_impact: detectSocialImpact(morphemes, semantics)
  };

  return {
    pragmatics,
    confidence: intentResult.confidence,
    reasoning: intentResult.reasoning
  };
}

/**
 * Calculate pragmatics match score
 * Returns 0-1 indicating how well pragmatics match a pattern
 */
export function calculatePragmaticsScore(
  detected: Pragmatics,
  pattern: Pragmatics
): number {
  let score = 0;

  // Intent match (40% weight)
  if (detected.intent === pattern.intent) {
    score += 0.4;
  }

  // Context awareness match (20% weight)
  const contextDiff = Math.abs(detected.context_awareness - pattern.context_awareness);
  score += (1 - contextDiff) * 0.2;

  // Power dynamic match (20% weight)
  if (detected.power_dynamic === pattern.power_dynamic) {
    score += 0.2;
  }

  // Social impact match (20% weight)
  if (detected.social_impact === pattern.social_impact) {
    score += 0.2;
  }

  return score;
}

/**
 * Get pragmatics statistics
 */
export function getPragmaticsStats(pragmatics: Pragmatics) {
  return {
    intent: pragmatics.intent,
    context_awareness: pragmatics.context_awareness,
    context_dependency: pragmatics.context_awareness > 0.7 ? 'high' : 'low',
    power_dynamic: pragmatics.power_dynamic,
    social_impact: pragmatics.social_impact,
    manipulation_severity: calculateManipulationSeverity(pragmatics)
  };
}

/**
 * Calculate overall manipulation severity from pragmatics
 * 0-1 scale
 */
function calculateManipulationSeverity(pragmatics: Pragmatics): number {
  let severity = 0;

  // Intent severity
  const intentSeverity: Record<Pragmatics['intent'], number> = {
    'manipulate': 0.9,
    'control': 0.8,
    'deceive': 0.7,
    'confuse': 0.6,
    'dominate': 0.9,
    'harm': 1.0
  };
  severity += intentSeverity[pragmatics.intent] * 0.4;

  // Power dynamic severity
  if (pragmatics.power_dynamic === 'exploit') {
    severity += 0.3;
  } else if (pragmatics.power_dynamic === 'reverse') {
    severity += 0.4;  // DARVO is more severe
  }

  // Social impact severity
  const socialSeverity: Record<Pragmatics['social_impact'], number> = {
    'isolate': 0.3,
    'triangulate': 0.25,
    'recruit': 0.2,
    'divide': 0.25
  };
  severity += socialSeverity[pragmatics.social_impact];

  return Math.min(1, severity);
}

/**
 * Determine manipulation category from pragmatics
 */
export function categorizePragmatics(pragmatics: Pragmatics): string[] {
  const categories: string[] = [];

  if (pragmatics.intent === 'manipulate' && pragmatics.social_impact === 'isolate') {
    categories.push('gaslighting');
  }

  if (pragmatics.power_dynamic === 'reverse') {
    categories.push('darvo');
  }

  if (pragmatics.social_impact === 'triangulate') {
    categories.push('triangulation');
  }

  if (pragmatics.intent === 'control') {
    categories.push('control');
  }

  return categories;
}
