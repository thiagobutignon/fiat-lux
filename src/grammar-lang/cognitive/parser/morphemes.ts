/**
 * MORPHEMES Parser
 * Analyzes minimal units of meaning in text
 * Detects keywords, negations, qualifiers, intensifiers, diminishers
 */

import { Morphemes } from '../types';

// ============================================================
// MORPHEME DICTIONARIES
// ============================================================

/**
 * Common manipulation keywords
 * Pre-compiled for O(1) lookup
 */
const KEYWORD_SETS = {
  gaslighting: new Set([
    'never happened',
    'imagining things',
    'making this up',
    'never said',
    'remembering wrong',
    'too sensitive',
    'overreacting',
    'being dramatic',
    'calm down',
    'you\'re crazy'
  ]),

  denial: new Set([
    'didn\'t do',
    'never did',
    'that\'s not true',
    'didn\'t happen',
    'you\'re lying',
    'that\'s ridiculous'
  ]),

  darvo: new Set([
    'i\'m the victim',
    'you\'re hurting me',
    'you\'re attacking me',
    'you\'re abusing me',
    'i can\'t believe you',
    'how could you'
  ]),

  triangulation: new Set([
    'unlike you',
    'never does this',
    'understands me better',
    'be more like',
    'at least they'
  ])
};

/**
 * Negation words
 */
const NEGATIONS = new Set([
  'no', 'not', 'never', 'none', 'nobody', 'nothing', 'nowhere',
  'neither', 'nor', 'hardly', 'scarcely', 'barely',
  'didn\'t', 'don\'t', 'doesn\'t', 'won\'t', 'wouldn\'t',
  'can\'t', 'couldn\'t', 'shouldn\'t', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t'
]);

/**
 * Modal qualifiers (uncertainty, hedging)
 */
const QUALIFIERS = new Set([
  'maybe', 'perhaps', 'possibly', 'probably', 'might', 'may',
  'could', 'would', 'should', 'seems', 'appears',
  'i think', 'i believe', 'i guess', 'sort of', 'kind of'
]);

/**
 * Intensifiers (strengthening)
 */
const INTENSIFIERS = new Set([
  'always', 'never', 'absolutely', 'completely', 'totally',
  'utterly', 'entirely', 'extremely', 'very', 'so',
  'really', 'definitely', 'certainly', 'clearly', 'obviously'
]);

/**
 * Diminishers (weakening)
 */
const DIMINISHERS = new Set([
  'just', 'only', 'merely', 'simply', 'barely',
  'slightly', 'somewhat', 'a bit', 'a little', 'kinda'
]);

// ============================================================
// MORPHEME PARSER
// ============================================================

/**
 * Parse text for morphemes
 * O(n) where n = text length, but with O(1) lookups
 */
export function parseMorphemes(text: string): Morphemes {
  const lowerText = text.toLowerCase();
  const words = lowerText.split(/\s+/);

  // Extract morphemes
  const keywords: string[] = [];
  const negations: string[] = [];
  const qualifiers: string[] = [];
  const intensifiers: string[] = [];
  const diminishers: string[] = [];

  // Check for multi-word keywords first
  for (const [category, keywordSet] of Object.entries(KEYWORD_SETS)) {
    for (const keyword of keywordSet) {
      if (lowerText.includes(keyword)) {
        keywords.push(keyword);
      }
    }
  }

  // Check individual words
  for (const word of words) {
    if (NEGATIONS.has(word)) {
      negations.push(word);
    }
    if (QUALIFIERS.has(word)) {
      qualifiers.push(word);
    }
    if (INTENSIFIERS.has(word)) {
      intensifiers.push(word);
    }
    if (DIMINISHERS.has(word)) {
      diminishers.push(word);
    }
  }

  return {
    keywords: Array.from(new Set(keywords)),  // Deduplicate
    negations: Array.from(new Set(negations)),
    qualifiers: Array.from(new Set(qualifiers)),
    intensifiers: Array.from(new Set(intensifiers)),
    diminishers: Array.from(new Set(diminishers))
  };
}

/**
 * Calculate morpheme match score
 * Returns 0-1 indicating how well morphemes match a pattern
 */
export function calculateMorphemeScore(
  detected: Morphemes,
  pattern: Morphemes
): number {
  let score = 0;
  let totalChecks = 0;

  // Keywords (most important, 50% weight)
  if (pattern.keywords.length > 0) {
    const matchedKeywords = pattern.keywords.filter(k =>
      detected.keywords.includes(k)
    );
    score += (matchedKeywords.length / pattern.keywords.length) * 0.5;
    totalChecks += 0.5;
  }

  // Negations (20% weight)
  if (pattern.negations.length > 0) {
    const matchedNegations = pattern.negations.filter(n =>
      detected.negations.includes(n)
    );
    score += (matchedNegations.length / pattern.negations.length) * 0.2;
    totalChecks += 0.2;
  }

  // Qualifiers (10% weight)
  if (pattern.qualifiers.length > 0) {
    const hasQualifier = pattern.qualifiers.some(q =>
      detected.qualifiers.includes(q)
    );
    score += hasQualifier ? 0.1 : 0;
    totalChecks += 0.1;
  }

  // Intensifiers (10% weight)
  if (pattern.intensifiers.length > 0) {
    const hasIntensifier = pattern.intensifiers.some(i =>
      detected.intensifiers.includes(i)
    );
    score += hasIntensifier ? 0.1 : 0;
    totalChecks += 0.1;
  }

  // Diminishers (10% weight)
  if (pattern.diminishers.length > 0) {
    const hasDiminisher = pattern.diminishers.some(d =>
      detected.diminishers.includes(d)
    );
    score += hasDiminisher ? 0.1 : 0;
    totalChecks += 0.1;
  }

  // Normalize score (if not all checks were used)
  return totalChecks > 0 ? score / totalChecks : 0;
}

/**
 * Extract specific manipulation morphemes for neurodivergent protection
 */
export function hasNeurodivergentMarkers(morphemes: Morphemes): boolean {
  // Neurodivergent communication patterns that might trigger false positives
  const autismMarkers = [
    'i meant literally',
    'to be precise',
    'technically speaking',
    'actually',
    'in fact'
  ];

  const adhdMarkers = [
    'i forgot',
    'sorry, i wasn\'t listening',
    'wait, what were we talking about',
    'i got distracted'
  ];

  const text = morphemes.keywords.join(' ');

  const hasAutismMarkers = autismMarkers.some(marker =>
    text.includes(marker)
  );

  const hasAdhdMarkers = adhdMarkers.some(marker =>
    text.includes(marker)
  );

  return hasAutismMarkers || hasAdhdMarkers;
}

/**
 * Get morpheme statistics
 */
export function getMorphemeStats(morphemes: Morphemes) {
  return {
    keyword_count: morphemes.keywords.length,
    negation_count: morphemes.negations.length,
    qualifier_count: morphemes.qualifiers.length,
    intensifier_count: morphemes.intensifiers.length,
    diminisher_count: morphemes.diminishers.length,
    total_morphemes:
      morphemes.keywords.length +
      morphemes.negations.length +
      morphemes.qualifiers.length +
      morphemes.intensifiers.length +
      morphemes.diminishers.length
  };
}
