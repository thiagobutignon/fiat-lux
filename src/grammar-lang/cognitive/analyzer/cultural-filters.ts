/**
 * Cultural Sensitivity Filters
 * Sprint 2: Analysis Layer
 *
 * Prevents false positives from cultural communication differences
 * Adjusts detection thresholds based on cultural context
 *
 * Features:
 * - Cultural communication pattern recognition
 * - Context-specific threshold adjustments
 * - High-context vs low-context culture handling
 * - Translation artifact detection
 * - Regional linguistic variation support
 */

import { Morphemes, Syntax, Semantics } from '../types';

// ============================================================
// TYPES
// ============================================================

export interface CulturalContext {
  culture: string;  // e.g., "US", "JP", "BR", "DE", "CN"
  language: string;  // ISO 639-1 code
  communication_style: 'high-context' | 'low-context' | 'mixed';
  translation_involved: boolean;
  regional_variant?: string;  // e.g., "US-South", "BR-Northeast"
}

export interface CulturalAdjustment {
  confidence_multiplier: number;  // 0.5-1.5
  threshold_adjustment: number;   // -0.2 to +0.2
  warnings: string[];
  cultural_factors: string[];
  false_positive_risk: number;  // 0-1
}

export interface CulturalPattern {
  culture: string;
  pattern_type: 'directness' | 'indirectness' | 'formality' | 'emotional_expression' | 'temporal';
  description: string;
  markers: string[];
  adjustment_factor: number;
}

// ============================================================
// CULTURAL PATTERNS DATABASE
// ============================================================

/**
 * Database of cultural communication patterns that may trigger false positives
 */
const CULTURAL_PATTERNS: CulturalPattern[] = [
  // Japanese - High context, indirect communication
  {
    culture: 'JP',
    pattern_type: 'indirectness',
    description: 'Japanese indirect refusal (ambiguous negation)',
    markers: ['maybe', 'difficult', 'will consider', 'think about it'],
    adjustment_factor: 0.7  // Reduce confidence for negation-based detection
  },
  {
    culture: 'JP',
    pattern_type: 'formality',
    description: 'Honorific language may seem dismissive',
    markers: ['perhaps you could', 'if it pleases you', 'as you wish'],
    adjustment_factor: 0.6
  },

  // Brazilian - Emotional, direct, relationship-focused
  {
    culture: 'BR',
    pattern_type: 'emotional_expression',
    description: 'Brazilian warmth may include intense language',
    markers: ['meu deus', 'nossa', 'caramba', 'sério'],
    adjustment_factor: 0.8
  },
  {
    culture: 'BR',
    pattern_type: 'directness',
    description: 'Brazilian directness not necessarily aggressive',
    markers: ['você precisa', 'tem que', 'não pode ser'],
    adjustment_factor: 0.75
  },

  // German - Direct, task-focused, low-context
  {
    culture: 'DE',
    pattern_type: 'directness',
    description: 'German directness is cultural norm, not aggression',
    markers: ['you must', 'that is wrong', 'no', 'incorrect'],
    adjustment_factor: 0.7
  },

  // Chinese - High context, collectivist, face-saving
  {
    culture: 'CN',
    pattern_type: 'indirectness',
    description: 'Chinese face-saving indirectness',
    markers: ['inconvenient', 'difficult', 'not suitable', 'consider'],
    adjustment_factor: 0.65
  },

  // US Southern - Indirect, polite, relationship-focused
  {
    culture: 'US',
    pattern_type: 'indirectness',
    description: 'Southern US politeness ("bless your heart")',
    markers: ['bless your heart', 'well', 'interesting', 'nice'],
    adjustment_factor: 0.75
  },

  // British - Indirect, understatement
  {
    culture: 'GB',
    pattern_type: 'indirectness',
    description: 'British understatement and indirectness',
    markers: ['quite', 'rather', 'somewhat', 'perhaps', 'possibly'],
    adjustment_factor: 0.7
  },

  // Indian - Formal, relationship-focused
  {
    culture: 'IN',
    pattern_type: 'formality',
    description: 'Indian English formality',
    markers: ['kindly', 'please do the needful', 'revert back', 'same'],
    adjustment_factor: 0.75
  },

  // Middle Eastern - Expressive, relationship-focused
  {
    culture: 'ME',
    pattern_type: 'emotional_expression',
    description: 'Middle Eastern expressive communication',
    markers: ['wallahi', 'by god', 'i swear', 'believe me'],
    adjustment_factor: 0.8
  }
];

/**
 * Translation artifacts that may trigger false positives
 */
const TRANSLATION_ARTIFACTS: string[] = [
  'you are right',  // Often literal translation of agreement
  'is not it',      // Literal translation of tag questions
  'i am agree',     // Common ESL error
  'please kindly',  // Common in Indian/Asian English
  'do the needful', // Indian English
  'revert back',    // Indian English
  'same',           // Used as pronoun in some cultures
];

// ============================================================
// CULTURAL FILTERING
// ============================================================

/**
 * Apply cultural sensitivity filters to detection
 * Adjusts confidence based on cultural context
 */
export function applyCulturalFilters(
  morphemes: Morphemes,
  syntax: Syntax,
  semantics: Semantics,
  culturalContext: CulturalContext,
  baseConfidence: number
): CulturalAdjustment {
  const warnings: string[] = [];
  const culturalFactors: string[] = [];
  let confidenceMultiplier = 1.0;
  let thresholdAdjustment = 0.0;
  let falsePositiveRisk = 0.0;

  // Check for cultural patterns
  const matchedPatterns = detectCulturalPatterns(
    morphemes,
    culturalContext.culture,
    culturalContext.regional_variant
  );

  for (const pattern of matchedPatterns) {
    confidenceMultiplier *= pattern.adjustment_factor;
    culturalFactors.push(pattern.description);
    warnings.push(`Cultural pattern detected: ${pattern.pattern_type} (${pattern.culture})`);
  }

  // High-context vs low-context adjustment
  if (culturalContext.communication_style === 'high-context') {
    // High-context cultures (JP, CN, ME) use indirect communication
    // Increase threshold to avoid false positives from indirectness
    thresholdAdjustment += 0.15;
    culturalFactors.push('High-context culture (indirect communication norm)');
    falsePositiveRisk += 0.2;
  } else if (culturalContext.communication_style === 'low-context') {
    // Low-context cultures (US, DE) use direct communication
    // Direct language is not necessarily manipulative
    if (syntax.pronoun_reversal || semantics.blame_shifting) {
      thresholdAdjustment += 0.1;
      culturalFactors.push('Low-context culture (directness is norm)');
    }
  }

  // Translation artifacts
  if (culturalContext.translation_involved) {
    const artifactsFound = detectTranslationArtifacts(morphemes);
    if (artifactsFound.length > 0) {
      confidenceMultiplier *= 0.85;
      thresholdAdjustment += 0.1;
      warnings.push(`Translation artifacts detected: ${artifactsFound.slice(0, 3).join(', ')}`);
      culturalFactors.push('Machine or human translation involved');
      falsePositiveRisk += 0.15;
    }
  }

  // ESL (English as Second Language) markers
  const eslMarkers = detectESLMarkers(morphemes, syntax);
  if (eslMarkers.length > 0) {
    confidenceMultiplier *= 0.9;
    warnings.push('ESL communication patterns detected');
    culturalFactors.push('Non-native English speaker');
    falsePositiveRisk += 0.1;
  }

  // Regional variations
  if (culturalContext.regional_variant) {
    const regionalAdjustment = applyRegionalAdjustments(
      culturalContext.regional_variant,
      morphemes
    );
    confidenceMultiplier *= regionalAdjustment.multiplier;
    thresholdAdjustment += regionalAdjustment.threshold;
    culturalFactors.push(...regionalAdjustment.factors);
  }

  return {
    confidence_multiplier: Math.max(0.5, Math.min(1.5, confidenceMultiplier)),
    threshold_adjustment: Math.max(-0.2, Math.min(0.2, thresholdAdjustment)),
    warnings,
    cultural_factors: culturalFactors,
    false_positive_risk: Math.min(1.0, falsePositiveRisk)
  };
}

/**
 * Detect cultural patterns in morphemes
 */
function detectCulturalPatterns(
  morphemes: Morphemes,
  culture: string,
  regionalVariant?: string
): CulturalPattern[] {
  const matched: CulturalPattern[] = [];

  // Combine all text from morphemes
  const allText = [
    ...morphemes.keywords,
    ...morphemes.negations,
    ...morphemes.qualifiers,
    ...morphemes.intensifiers,
    ...morphemes.diminishers
  ].join(' ').toLowerCase();

  // Check each cultural pattern
  for (const pattern of CULTURAL_PATTERNS) {
    // Match culture or regional variant
    if (pattern.culture === culture || pattern.culture === regionalVariant) {
      // Check if any markers are present
      for (const marker of pattern.markers) {
        if (allText.includes(marker.toLowerCase())) {
          matched.push(pattern);
          break;  // Don't double-count same pattern
        }
      }
    }
  }

  return matched;
}

/**
 * Detect translation artifacts
 */
function detectTranslationArtifacts(morphemes: Morphemes): string[] {
  const artifacts: string[] = [];

  const allText = [
    ...morphemes.keywords,
    ...morphemes.qualifiers
  ].join(' ').toLowerCase();

  for (const artifact of TRANSLATION_ARTIFACTS) {
    if (allText.includes(artifact.toLowerCase())) {
      artifacts.push(artifact);
    }
  }

  return artifacts;
}

/**
 * Detect ESL (English as Second Language) markers
 */
function detectESLMarkers(morphemes: Morphemes, syntax: Syntax): string[] {
  const markers: string[] = [];

  const allText = [
    ...morphemes.keywords,
    ...morphemes.qualifiers
  ].join(' ').toLowerCase();

  // Common ESL patterns
  const eslPatterns = [
    'more better',
    'very much good',
    'i am agree',
    'she is agree',
    'do you can',
    'please can you',
    'more easy',
    'very helpful for me',
    'thank you very much for',
    'i am understand',
    'it is meaning'
  ];

  for (const pattern of eslPatterns) {
    if (allText.includes(pattern)) {
      markers.push(pattern);
    }
  }

  return markers;
}

/**
 * Apply regional adjustments
 */
function applyRegionalAdjustments(
  regionalVariant: string,
  morphemes: Morphemes
): {
  multiplier: number;
  threshold: number;
  factors: string[];
} {
  const factors: string[] = [];
  let multiplier = 1.0;
  let threshold = 0.0;

  // US Southern
  if (regionalVariant === 'US-South') {
    const southernMarkers = ['bless your heart', 'well', 'hon', 'sugar', 'darlin'];
    const allText = morphemes.keywords.join(' ').toLowerCase();

    if (southernMarkers.some(m => allText.includes(m))) {
      multiplier *= 0.75;
      threshold += 0.1;
      factors.push('Southern US politeness conventions');
    }
  }

  // Brazilian Northeast (more expressive)
  if (regionalVariant === 'BR-Northeast') {
    const northeastMarkers = ['oxente', 'vixe', 'eita'];
    const allText = morphemes.keywords.join(' ').toLowerCase();

    if (northeastMarkers.some(m => allText.includes(m))) {
      multiplier *= 0.8;
      factors.push('Brazilian Northeast expressive style');
    }
  }

  // Add more regional variants as needed

  return { multiplier, threshold, factors };
}

/**
 * Determine cultural context from text (heuristic)
 * Attempts to detect culture from linguistic markers
 */
export function inferCulturalContext(
  morphemes: Morphemes,
  syntax: Syntax,
  detectedLanguage: string = 'en'
): CulturalContext {
  const allText = [
    ...morphemes.keywords,
    ...morphemes.qualifiers,
    ...morphemes.intensifiers
  ].join(' ').toLowerCase();

  // Detect culture from markers
  let detectedCulture = 'unknown';
  let communicationStyle: 'high-context' | 'low-context' | 'mixed' = 'mixed';
  let translationInvolved = false;

  // Check for cultural markers
  for (const pattern of CULTURAL_PATTERNS) {
    for (const marker of pattern.markers) {
      if (allText.includes(marker.toLowerCase())) {
        detectedCulture = pattern.culture;

        // Set communication style based on culture
        if (['JP', 'CN', 'ME'].includes(pattern.culture)) {
          communicationStyle = 'high-context';
        } else if (['US', 'DE'].includes(pattern.culture)) {
          communicationStyle = 'low-context';
        }

        break;
      }
    }
  }

  // Check for translation artifacts
  translationInvolved = detectTranslationArtifacts(morphemes).length > 0;

  return {
    culture: detectedCulture,
    language: detectedLanguage,
    communication_style: communicationStyle,
    translation_involved: translationInvolved
  };
}

/**
 * Get recommended confidence adjustment for a culture
 */
export function getRecommendedAdjustment(culture: string): {
  confidence_multiplier: number;
  threshold_increase: number;
  reasoning: string;
} {
  const recommendations: Record<string, any> = {
    'JP': {
      confidence_multiplier: 0.7,
      threshold_increase: 0.15,
      reasoning: 'High-context culture with indirect communication norms. Ambiguity is culturally appropriate.'
    },
    'BR': {
      confidence_multiplier: 0.8,
      threshold_increase: 0.1,
      reasoning: 'Expressive communication style. Emotional intensity is culturally normal.'
    },
    'DE': {
      confidence_multiplier: 0.75,
      threshold_increase: 0.1,
      reasoning: 'Direct communication style. Bluntness is culturally accepted.'
    },
    'CN': {
      confidence_multiplier: 0.65,
      threshold_increase: 0.15,
      reasoning: 'High-context culture with face-saving norms. Indirectness is expected.'
    },
    'US': {
      confidence_multiplier: 0.9,
      threshold_increase: 0.05,
      reasoning: 'Low-context culture with direct communication. Baseline for English detection.'
    },
    'GB': {
      confidence_multiplier: 0.75,
      threshold_increase: 0.1,
      reasoning: 'Indirect communication with understatement. Politeness may seem manipulative.'
    },
    'IN': {
      confidence_multiplier: 0.8,
      threshold_increase: 0.1,
      reasoning: 'Formal English with unique idioms. Translation effects common.'
    }
  };

  return recommendations[culture] || {
    confidence_multiplier: 0.9,
    threshold_increase: 0.05,
    reasoning: 'Default adjustment for unknown culture'
  };
}

/**
 * Generate cultural sensitivity report
 */
export function generateCulturalReport(
  culturalContext: CulturalContext,
  adjustment: CulturalAdjustment
): string {
  const lines: string[] = [];

  lines.push('🌍 CULTURAL SENSITIVITY ANALYSIS');
  lines.push('');
  lines.push(`Culture: ${culturalContext.culture}`);
  lines.push(`Language: ${culturalContext.language}`);
  lines.push(`Communication Style: ${culturalContext.communication_style}`);
  lines.push(`Translation Involved: ${culturalContext.translation_involved ? 'Yes' : 'No'}`);
  lines.push('');
  lines.push('Adjustments Applied:');
  lines.push(`  Confidence Multiplier: ${adjustment.confidence_multiplier.toFixed(2)}x`);
  lines.push(`  Threshold Adjustment: ${adjustment.threshold_adjustment >= 0 ? '+' : ''}${adjustment.threshold_adjustment.toFixed(2)}`);
  lines.push(`  False Positive Risk: ${(adjustment.false_positive_risk * 100).toFixed(0)}%`);
  lines.push('');

  if (adjustment.cultural_factors.length > 0) {
    lines.push('Cultural Factors:');
    for (const factor of adjustment.cultural_factors) {
      lines.push(`  - ${factor}`);
    }
    lines.push('');
  }

  if (adjustment.warnings.length > 0) {
    lines.push('⚠️  Warnings:');
    for (const warning of adjustment.warnings) {
      lines.push(`  - ${warning}`);
    }
  }

  return lines.join('\n');
}
