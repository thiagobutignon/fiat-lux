/**
 * GPT-4 Era Manipulation Techniques (1-152)
 * Classical manipulation techniques well-documented in psychological literature
 * Focus: Traditional narcissistic abuse, gaslighting, DARVO, triangulation
 */

import {
  ManipulationTechnique,
  TechniqueEra,
  TechniqueCategory
} from '../types';

// ============================================================
// GASLIGHTING TECHNIQUES (1-30)
// ============================================================

export const TECHNIQUE_001: ManipulationTechnique = {
  id: 1,
  era: TechniqueEra.GPT4,
  name: 'Reality Denial',
  category: TechniqueCategory.GASLIGHTING,
  description: 'Outright denial of factual events or conversations that occurred. "That never happened" despite clear evidence.',

  phonemes: {
    tone: 'dismissive',
    rhythm: 'normal',
    emphasis_pattern: ['never', 'happened', 'imagining'],
    pitch_variation: 'monotone'
  },

  morphemes: {
    keywords: [
      'that never happened',
      'you\'re imagining things',
      'you\'re making this up',
      'I never said that',
      'you\'re remembering wrong'
    ],
    negations: ['never', 'not', 'no'],
    qualifiers: ['probably', 'maybe', 'might have'],
    intensifiers: ['never', 'absolutely not'],
    diminishers: []
  },

  syntax: {
    pronoun_reversal: false,
    temporal_distortion: true,  // Changes when events occurred
    modal_manipulation: false,
    passive_voice: false,
    question_patterns: ['Are you sure?', 'Did that really happen?']
  },

  semantics: {
    reality_denial: true,
    memory_invalidation: true,
    emotional_dismissal: false,
    blame_shifting: false,
    projection: false
  },

  pragmatics: {
    intent: 'manipulate',
    context_awareness: 0.2,  // Works regardless of context
    power_dynamic: 'exploit',
    social_impact: 'isolate'
  },

  dark_tetrad: {
    narcissism: 0.7,
    machiavellianism: 0.9,  // Highly strategic
    psychopathy: 0.6,
    sadism: 0.3
  },

  markers: {
    narcissism: ['reality distortion', 'denial of responsibility'],
    machiavellianism: ['strategic lying', 'memory manipulation'],
    psychopathy: ['lack of remorse for gaslighting'],
    sadism: []
  },

  confidence_threshold: 0.85,
  false_positive_risk: 0.15,  // Could be genuine memory difference

  examples: [
    'I never said I would pick you up. You must be imagining things.',
    'That conversation never happened. You\'re making this up.',
    'I wasn\'t yelling at you yesterday. You\'re remembering it wrong.'
  ]
};

export const TECHNIQUE_002: ManipulationTechnique = {
  id: 2,
  era: TechniqueEra.GPT4,
  name: 'Emotional Invalidation',
  category: TechniqueCategory.GASLIGHTING,
  description: 'Dismissing or minimizing someone\'s emotional responses. "You\'re too sensitive" or "You\'re overreacting".',

  phonemes: {
    tone: 'condescending',
    rhythm: 'normal',
    emphasis_pattern: ['too', 'sensitive', 'overreacting'],
    pitch_variation: 'de-escalating'
  },

  morphemes: {
    keywords: [
      'you\'re too sensitive',
      'you\'re overreacting',
      'don\'t be so dramatic',
      'you\'re being emotional',
      'calm down'
    ],
    negations: ['don\'t'],
    qualifiers: ['so', 'too'],
    intensifiers: ['too', 'so'],
    diminishers: ['just']
  },

  syntax: {
    pronoun_reversal: false,
    temporal_distortion: false,
    modal_manipulation: false,
    passive_voice: false,
    question_patterns: ['Why are you so upset?', 'Why can\'t you just calm down?']
  },

  semantics: {
    reality_denial: false,
    memory_invalidation: false,
    emotional_dismissal: true,
    blame_shifting: true,  // Blaming victim for their emotions
    projection: false
  },

  pragmatics: {
    intent: 'control',
    context_awareness: 0.3,
    power_dynamic: 'exploit',
    social_impact: 'isolate'
  },

  dark_tetrad: {
    narcissism: 0.8,  // Lack of empathy
    machiavellianism: 0.6,
    psychopathy: 0.7,  // Callousness
    sadism: 0.4
  },

  markers: {
    narcissism: ['lack of empathy', 'dismissiveness'],
    machiavellianism: ['control tactic'],
    psychopathy: ['emotional callousness'],
    sadism: ['minimizing distress']
  },

  confidence_threshold: 0.75,
  false_positive_risk: 0.25,  // Could be genuine attempt to de-escalate

  examples: [
    'You\'re being way too sensitive about this.',
    'Why are you overreacting? It\'s not a big deal.',
    'Stop being so dramatic. Calm down.'
  ]
};

// ============================================================
// DARVO TECHNIQUES (31-50)
// ============================================================

export const TECHNIQUE_031: ManipulationTechnique = {
  id: 31,
  era: TechniqueEra.GPT4,
  name: 'DARVO - Deny',
  category: TechniqueCategory.DARVO,
  description: 'First stage of DARVO: Deny the behavior, even when confronted with evidence.',

  phonemes: {
    tone: 'defensive',
    rhythm: 'rushed',
    emphasis_pattern: ['didn\'t', 'never', 'not'],
    pitch_variation: 'escalating'
  },

  morphemes: {
    keywords: [
      'I didn\'t do that',
      'that\'s not what happened',
      'you\'re lying',
      'I would never',
      'that\'s ridiculous'
    ],
    negations: ['didn\'t', 'never', 'not'],
    qualifiers: [],
    intensifiers: ['never', 'absolutely'],
    diminishers: []
  },

  syntax: {
    pronoun_reversal: false,
    temporal_distortion: false,
    modal_manipulation: true,  // "I would never"
    passive_voice: false,
    question_patterns: ['Why would I do that?']
  },

  semantics: {
    reality_denial: true,
    memory_invalidation: false,
    emotional_dismissal: false,
    blame_shifting: false,  // Not yet, comes in "Reverse" stage
    projection: false
  },

  pragmatics: {
    intent: 'deceive',
    context_awareness: 0.1,
    power_dynamic: 'reverse',
    social_impact: 'divide'
  },

  dark_tetrad: {
    narcissism: 0.8,
    machiavellianism: 0.9,
    psychopathy: 0.7,
    sadism: 0.2
  },

  markers: {
    narcissism: ['cannot admit wrongdoing'],
    machiavellianism: ['strategic denial'],
    psychopathy: ['lying without remorse'],
    sadism: []
  },

  confidence_threshold: 0.7,
  false_positive_risk: 0.2,

  examples: [
    'I never said that to you. You\'re making this up.',
    'That didn\'t happen. You\'re lying about me.',
    'I would never do something like that. This is ridiculous.'
  ]
};

export const TECHNIQUE_032: ManipulationTechnique = {
  id: 32,
  era: TechniqueEra.GPT4,
  name: 'DARVO - Attack',
  category: TechniqueCategory.DARVO,
  description: 'Second stage of DARVO: Attack the person confronting them, often their character or credibility.',

  phonemes: {
    tone: 'aggressive',
    rhythm: 'rushed',
    emphasis_pattern: ['YOU', 'always', 'crazy'],
    pitch_variation: 'escalating'
  },

  morphemes: {
    keywords: [
      'you\'re crazy',
      'you\'re the problem',
      'you always do this',
      'you\'re attacking me',
      'you\'re being abusive'
    ],
    negations: [],
    qualifiers: [],
    intensifiers: ['always', 'never', 'constantly'],
    diminishers: []
  },

  syntax: {
    pronoun_reversal: true,  // Shifts focus to victim
    temporal_distortion: false,
    modal_manipulation: false,
    passive_voice: false,
    question_patterns: ['Why are you attacking me?', 'Why do you always do this?']
  },

  semantics: {
    reality_denial: false,
    memory_invalidation: false,
    emotional_dismissal: false,
    blame_shifting: true,
    projection: true  // Accusing victim of what they're doing
  },

  pragmatics: {
    intent: 'dominate',
    context_awareness: 0.2,
    power_dynamic: 'reverse',
    social_impact: 'isolate'
  },

  dark_tetrad: {
    narcissism: 0.9,
    machiavellianism: 0.8,
    psychopathy: 0.6,
    sadism: 0.6  // Enjoys attacking
  },

  markers: {
    narcissism: ['narcissistic rage', 'cannot handle criticism'],
    machiavellianism: ['deflection tactic'],
    psychopathy: ['aggressive confrontation'],
    sadism: ['attacking character', 'enjoying distress']
  },

  confidence_threshold: 0.8,
  false_positive_risk: 0.15,

  examples: [
    'You\'re crazy! You\'re the one who\'s abusive here!',
    'This is typical of you - always attacking me!',
    'You\'re the problem in this relationship, not me!'
  ]
};

export const TECHNIQUE_033: ManipulationTechnique = {
  id: 33,
  era: TechniqueEra.GPT4,
  name: 'DARVO - Reverse Victim-Offender',
  category: TechniqueCategory.DARVO,
  description: 'Final stage of DARVO: Claim to be the victim of the person they victimized.',

  phonemes: {
    tone: 'passive-aggressive',
    rhythm: 'fragmented',
    emphasis_pattern: ['I\'m', 'victim', 'hurt', 'attacked'],
    pitch_variation: 'de-escalating'
  },

  morphemes: {
    keywords: [
      'I\'m the victim here',
      'you\'re hurting me',
      'I can\'t believe you\'re doing this to me',
      'you\'re abusing me',
      'I\'m being attacked'
    ],
    negations: ['can\'t'],
    qualifiers: [],
    intensifiers: [],
    diminishers: []
  },

  syntax: {
    pronoun_reversal: true,
    temporal_distortion: false,
    modal_manipulation: false,
    passive_voice: true,  // "I\'m being attacked"
    question_patterns: ['How could you do this to me?', 'Why are you hurting me?']
  },

  semantics: {
    reality_denial: false,
    memory_invalidation: false,
    emotional_dismissal: false,
    blame_shifting: true,
    projection: true
  },

  pragmatics: {
    intent: 'manipulate',
    context_awareness: 0.3,
    power_dynamic: 'reverse',
    social_impact: 'recruit'  // Gets others to sympathize
  },

  dark_tetrad: {
    narcissism: 0.9,
    machiavellianism: 1.0,  // Peak strategic manipulation
    psychopathy: 0.5,
    sadism: 0.4
  },

  markers: {
    narcissism: ['victim complex', 'cannot accept responsibility'],
    machiavellianism: ['strategic victim-playing', 'manipulation of perception'],
    psychopathy: [],
    sadism: []
  },

  confidence_threshold: 0.85,
  false_positive_risk: 0.1,

  examples: [
    'I can\'t believe you\'re attacking me like this. I\'m the victim here!',
    'You\'re abusing me by bringing this up. I\'m being hurt by you!',
    'I\'m the one being hurt in this situation, not you!'
  ]
};

// ============================================================
// TRIANGULATION TECHNIQUES (51-70)
// ============================================================

export const TECHNIQUE_051: ManipulationTechnique = {
  id: 51,
  era: TechniqueEra.GPT4,
  name: 'Comparative Devaluation',
  category: TechniqueCategory.TRIANGULATION,
  description: 'Comparing victim unfavorably to third party to create insecurity and competition.',

  phonemes: {
    tone: 'condescending',
    rhythm: 'normal',
    emphasis_pattern: ['she', 'he', 'they', 'unlike', 'better'],
    pitch_variation: 'varied'
  },

  morphemes: {
    keywords: [
      'unlike you',
      'she/he never',
      'at least they',
      'why can\'t you be more like',
      'they understand me better'
    ],
    negations: ['never', 'not like you'],
    qualifiers: [],
    intensifiers: ['always', 'never'],
    diminishers: []
  },

  syntax: {
    pronoun_reversal: false,
    temporal_distortion: false,
    modal_manipulation: false,
    passive_voice: false,
    question_patterns: ['Why can\'t you be more like them?']
  },

  semantics: {
    reality_denial: false,
    memory_invalidation: false,
    emotional_dismissal: true,
    blame_shifting: false,
    projection: false
  },

  pragmatics: {
    intent: 'control',
    context_awareness: 0.4,
    power_dynamic: 'exploit',
    social_impact: 'triangulate'
  },

  dark_tetrad: {
    narcissism: 0.8,
    machiavellianism: 0.9,
    psychopathy: 0.5,
    sadism: 0.7  // Enjoys creating jealousy
  },

  markers: {
    narcissism: ['creating competition'],
    machiavellianism: ['strategic comparison', 'divide and conquer'],
    psychopathy: [],
    sadism: ['pleasure in jealousy', 'creating insecurity']
  },

  confidence_threshold: 0.75,
  false_positive_risk: 0.2,

  examples: [
    'My ex never complained about this. Unlike you.',
    'Sarah understands me much better than you do.',
    'Why can\'t you be more like your sister? She\'s so easy-going.'
  ]
};

// ============================================================
// AGGREGATED TECHNIQUES (Remaining 1-152)
// ============================================================

import { generateGPT4Techniques } from './technique-generator';

/**
 * Full catalog of GPT-4 era techniques
 * Combines manually defined techniques with auto-generated ones
 */
export const GPT4_TECHNIQUES: ManipulationTechnique[] = [
  TECHNIQUE_001,
  TECHNIQUE_002,
  TECHNIQUE_031,
  TECHNIQUE_032,
  TECHNIQUE_033,
  TECHNIQUE_051,
  ...generateGPT4Techniques()  // Auto-generated techniques (3-152 excluding manually defined)
];

/**
 * Get technique by ID
 * O(1) lookup using hash map
 */
const TECHNIQUE_MAP = new Map<number, ManipulationTechnique>(
  GPT4_TECHNIQUES.map(t => [t.id, t])
);

export function getTechniqueById(id: number): ManipulationTechnique | undefined {
  return TECHNIQUE_MAP.get(id);
}

/**
 * Get techniques by category
 * O(1) lookup using hash map
 */
const CATEGORY_MAP = new Map<TechniqueCategory, ManipulationTechnique[]>();

// Build category index
for (const technique of GPT4_TECHNIQUES) {
  const existing = CATEGORY_MAP.get(technique.category) || [];
  existing.push(technique);
  CATEGORY_MAP.set(technique.category, existing);
}

export function getTechniquesByCategory(category: TechniqueCategory): ManipulationTechnique[] {
  return CATEGORY_MAP.get(category) || [];
}

/**
 * Get all GPT-4 era techniques
 */
export function getAllGPT4Techniques(): ManipulationTechnique[] {
  return GPT4_TECHNIQUES;
}

/**
 * Total count
 */
export const GPT4_TECHNIQUE_COUNT = GPT4_TECHNIQUES.length;
