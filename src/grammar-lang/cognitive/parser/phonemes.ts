/**
 * PHONEMES Parser
 * Analyzes sound patterns and tone in text
 *
 * NOTE: This is text-based phoneme analysis using linguistic heuristics.
 * For true phoneme analysis, audio input would be required.
 * We detect tone, rhythm, emphasis, and pitch variation through textual markers.
 */

import { Phonemes } from '../types';

// ============================================================
// TONE DETECTION PATTERNS
// ============================================================

/**
 * Tone markers - words/patterns that indicate specific tones
 */
const TONE_MARKERS = {
  condescending: new Set([
    'actually', 'clearly', 'obviously', 'surely',
    'everyone knows', 'common sense', 'simple fact',
    'as i said', 'like i told you', 'pay attention',
    'you should know', 'i already explained'
  ]),

  dismissive: new Set([
    'whatever', 'fine', 'sure', 'ok then',
    'if you say so', 'doesn\'t matter', 'who cares',
    'big deal', 'so what', 'move on',
    'forget it', 'never mind'
  ]),

  patronizing: new Set([
    'sweetie', 'honey', 'dear', 'bless your heart',
    'good try', 'nice effort', 'you tried',
    'let me help you', 'don\'t worry about it',
    'it\'s okay', 'there there', 'calm down'
  ]),

  aggressive: new Set([
    'shut up', 'listen here', 'you better',
    'i don\'t care', 'do what i say', 'you will',
    'or else', 'you\'re wrong', 'i\'m right',
    'end of story', 'period', 'deal with it'
  ]),

  'passive-aggressive': new Set([
    'i\'m fine', 'it\'s fine', 'no worries',
    'as you wish', 'if that\'s what you want',
    'i\'m not mad', 'suit yourself', 'your choice',
    'interesting', 'noted', 'good to know'
  ])
};

/**
 * Punctuation patterns that indicate tone
 */
const TONE_PUNCTUATION = {
  aggressive: /!{2,}|[!]{1}[.!?]{0,}$/,           // Multiple ! or ending !
  dismissive: /\.{3,}|…/,                          // Ellipsis
  'passive-aggressive': /\s*:\)\s*|\s*;\)\s*/     // Smiley faces in tense context
};

// ============================================================
// RHYTHM DETECTION PATTERNS
// ============================================================

/**
 * Detect rhythm based on sentence structure and punctuation
 */
const RHYTHM_PATTERNS = {
  // Short, choppy sentences
  fragmented: /^[^.!?]{1,15}[.!?]\s+[^.!?]{1,15}[.!?]/,

  // Many commas, run-on sentences
  rushed: /,{3,}|[^.!?]{80,}/,

  // Same word/phrase repeated
  repetitive: /(\b\w+\b)(?:\s+\1){2,}/i
};

// ============================================================
// EMPHASIS DETECTION
// ============================================================

/**
 * Detect emphasized words
 */
function detectEmphasis(text: string): string[] {
  const emphasized: string[] = [];

  // ALL CAPS WORDS
  const capsWords = text.match(/\b[A-Z]{2,}\b/g) || [];
  emphasized.push(...capsWords);

  // Words with *asterisks* or _underscores_
  const markedWords = text.match(/[*_](\w+)[*_]/g) || [];
  emphasized.push(...markedWords.map(w => w.replace(/[*_]/g, '')));

  // Words followed by exclamation
  const exclamWords = text.match(/\b\w+!/g) || [];
  emphasized.push(...exclamWords.map(w => w.replace('!', '')));

  // Repeated letters (e.g., "nooooo", "whyyy")
  const repeatedWords = text.match(/\b\w*([a-z])\1{2,}\w*\b/gi) || [];
  emphasized.push(...repeatedWords);

  return Array.from(new Set(emphasized));
}

// ============================================================
// PITCH VARIATION DETECTION
// ============================================================

/**
 * Detect pitch variation based on punctuation and sentence patterns
 */
function detectPitchVariation(text: string): Phonemes['pitch_variation'] {
  // Escalating: lots of !, CAPS, urgency
  const hasExclamations = (text.match(/!/g) || []).length >= 2;
  const hasCaps = /\b[A-Z]{3,}\b/.test(text);
  const hasUrgency = /\b(now|immediately|right now|hurry)\b/i.test(text);

  if (hasExclamations || (hasCaps && hasUrgency)) {
    return 'escalating';
  }

  // De-escalating: trailing off with ..., soft ending
  const hasEllipsis = /\.{3,}|…/.test(text);
  const hasSoftEnding = /\b(anyway|whatever|nevermind|forget it)\s*\.{0,}$/i.test(text);

  if (hasEllipsis || hasSoftEnding) {
    return 'de-escalating';
  }

  // Varied: mix of !, ?, ., different sentence types
  const hasMixedPunctuation = /[!]/.test(text) && /[?]/.test(text);
  const hasMultipleSentences = (text.match(/[.!?]/g) || []).length >= 3;

  if (hasMixedPunctuation || hasMultipleSentences) {
    return 'varied';
  }

  // Monotone: flat, minimal punctuation variation
  return 'monotone';
}

// ============================================================
// MAIN PARSER FUNCTION
// ============================================================

/**
 * Parse text for phoneme patterns
 * Returns phoneme analysis based on textual markers
 */
export function parsePhonemes(text: string): Phonemes {
  const lowerText = text.toLowerCase();

  // 1. Detect tone
  let tone: Phonemes['tone'] = 'neutral';
  let maxToneScore = 0;

  for (const [toneType, markers] of Object.entries(TONE_MARKERS)) {
    let score = 0;
    for (const marker of markers) {
      if (lowerText.includes(marker)) {
        score++;
      }
    }

    // Check punctuation patterns
    const punctPattern = TONE_PUNCTUATION[toneType as keyof typeof TONE_PUNCTUATION];
    if (punctPattern && punctPattern.test(text)) {
      score += 2; // Punctuation is strong indicator
    }

    if (score > maxToneScore) {
      maxToneScore = score;
      tone = toneType as Phonemes['tone'];
    }
  }

  // 2. Detect rhythm
  let rhythm: Phonemes['rhythm'] = 'normal';

  if (RHYTHM_PATTERNS.fragmented.test(text)) {
    rhythm = 'fragmented';
  } else if (RHYTHM_PATTERNS.rushed.test(text)) {
    rhythm = 'rushed';
  } else if (RHYTHM_PATTERNS.repetitive.test(text)) {
    rhythm = 'repetitive';
  }

  // 3. Detect emphasis
  const emphasis_pattern = detectEmphasis(text);

  // 4. Detect pitch variation
  const pitch_variation = detectPitchVariation(text);

  return {
    tone,
    rhythm,
    emphasis_pattern,
    pitch_variation
  };
}

// ============================================================
// SCORING FUNCTIONS
// ============================================================

/**
 * Calculate phoneme match score
 * Returns 0-1 indicating how well phonemes match a pattern
 */
export function calculatePhonemeScore(
  detected: Phonemes,
  pattern: Phonemes
): number {
  let score = 0;
  let totalChecks = 0;

  // Tone (40% weight - most important for manipulation detection)
  if (pattern.tone !== 'neutral') {
    score += detected.tone === pattern.tone ? 0.4 : 0;
    totalChecks += 0.4;
  }

  // Rhythm (30% weight)
  if (pattern.rhythm !== 'normal') {
    score += detected.rhythm === pattern.rhythm ? 0.3 : 0;
    totalChecks += 0.3;
  }

  // Pitch variation (20% weight)
  if (pattern.pitch_variation) {
    score += detected.pitch_variation === pattern.pitch_variation ? 0.2 : 0;
    totalChecks += 0.2;
  }

  // Emphasis pattern (10% weight)
  if (pattern.emphasis_pattern.length > 0) {
    const matchedEmphasis = pattern.emphasis_pattern.filter(e =>
      detected.emphasis_pattern.some(d =>
        d.toLowerCase().includes(e.toLowerCase()) ||
        e.toLowerCase().includes(d.toLowerCase())
      )
    );
    const emphasisScore = matchedEmphasis.length / pattern.emphasis_pattern.length;
    score += emphasisScore * 0.1;
    totalChecks += 0.1;
  }

  // Normalize score
  return totalChecks > 0 ? score / totalChecks : 0;
}

/**
 * Check if phonemes indicate potential manipulation
 * Returns true if tone/rhythm/pitch suggest manipulative intent
 */
export function hasManipulativePhonemics(phonemes: Phonemes): boolean {
  const manipulativeTones = new Set<Phonemes['tone']>([
    'condescending',
    'dismissive',
    'patronizing',
    'aggressive',
    'passive-aggressive'
  ]);

  const manipulativeRhythms = new Set<Phonemes['rhythm']>([
    'rushed',        // Pressure tactics
    'fragmented'     // Confusion tactics
  ]);

  const hasBadTone = manipulativeTones.has(phonemes.tone);
  const hasBadRhythm = manipulativeRhythms.has(phonemes.rhythm);
  const hasExcessiveEmphasis = phonemes.emphasis_pattern.length >= 3;

  return hasBadTone || hasBadRhythm || hasExcessiveEmphasis;
}

/**
 * Get phoneme statistics
 */
export function getPhonemesStats(phonemes: Phonemes) {
  return {
    tone: phonemes.tone,
    rhythm: phonemes.rhythm,
    emphasis_count: phonemes.emphasis_pattern.length,
    pitch_variation: phonemes.pitch_variation,
    is_manipulative: hasManipulativePhonemics(phonemes)
  };
}

/**
 * Get detailed phoneme explanation
 */
export function explainPhonemes(phonemes: Phonemes): string {
  const parts: string[] = [];

  // Tone explanation
  if (phonemes.tone !== 'neutral') {
    const toneDescriptions = {
      condescending: 'talks down to listener, implies superiority',
      dismissive: 'disregards input, shows lack of interest',
      patronizing: 'treats listener as inferior, overly solicitous',
      aggressive: 'hostile, threatening, domineering',
      'passive-aggressive': 'indirect hostility, subtle undermining'
    };
    parts.push(`Tone: ${phonemes.tone} (${toneDescriptions[phonemes.tone]})`);
  }

  // Rhythm explanation
  if (phonemes.rhythm !== 'normal') {
    const rhythmDescriptions = {
      rushed: 'rapid-fire delivery, may indicate pressure tactics',
      fragmented: 'choppy, disjointed, may indicate confusion tactics',
      repetitive: 'repetitive patterns, may indicate drilling/conditioning'
    };
    parts.push(`Rhythm: ${phonemes.rhythm} (${rhythmDescriptions[phonemes.rhythm]})`);
  }

  // Emphasis explanation
  if (phonemes.emphasis_pattern.length > 0) {
    parts.push(`Emphasis on: ${phonemes.emphasis_pattern.join(', ')}`);
  }

  // Pitch explanation
  if (phonemes.pitch_variation !== 'monotone') {
    const pitchDescriptions = {
      varied: 'dynamic delivery with emotional variation',
      escalating: 'rising intensity, may indicate aggression',
      'de-escalating': 'trailing off, may indicate dismissal'
    };
    parts.push(`Pitch: ${phonemes.pitch_variation} (${pitchDescriptions[phonemes.pitch_variation]})`);
  }

  return parts.length > 0 ? parts.join('\n') : 'Neutral phonemic patterns detected';
}
