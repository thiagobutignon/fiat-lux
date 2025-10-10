/**
 * GPT-5 Era Manipulation Techniques (153-180)
 * Emergent manipulation techniques enabled by advanced AI (2023-2025)
 * Focus: AI-augmented manipulation, LLM coaching, deepfakes, autonomous systems
 */

import {
  ManipulationTechnique,
  TechniqueEra,
  TechniqueCategory
} from '../types';

// ============================================================
// AI-AUGMENTED GASLIGHTING (153-160)
// ============================================================

export const TECHNIQUE_153: ManipulationTechnique = {
  id: 153,
  era: TechniqueEra.GPT5,
  name: 'LLM-Generated False Evidence',
  category: TechniqueCategory.GASLIGHTING,
  description: 'Using ChatGPT/Claude to generate fake conversation logs, emails, or "evidence" of events that never happened.',

  phonemes: {
    tone: 'neutral',  // Text-based, no tone
    rhythm: 'normal',
    emphasis_pattern: [],
    pitch_variation: 'monotone'
  },

  morphemes: {
    keywords: [
      'here\'s the proof',
      'I have the logs',
      'see this conversation',
      'the AI verified this',
      'according to this record'
    ],
    negations: [],
    qualifiers: [],
    intensifiers: ['clearly', 'obviously'],
    diminishers: []
  },

  syntax: {
    pronoun_reversal: false,
    temporal_distortion: true,  // Fabricated timeline
    modal_manipulation: false,
    passive_voice: false,
    question_patterns: ['See? This proves it.']
  },

  semantics: {
    reality_denial: false,  // Not denying, fabricating
    memory_invalidation: true,
    emotional_dismissal: false,
    blame_shifting: false,
    projection: false
  },

  pragmatics: {
    intent: 'manipulate',
    context_awareness: 0.8,  // Highly contextual fabrication
    power_dynamic: 'exploit',
    social_impact: 'isolate'
  },

  dark_tetrad: {
    narcissism: 0.7,
    machiavellianism: 1.0,  // Peak strategic deception
    psychopathy: 0.8,  // Callous fabrication
    sadism: 0.5
  },

  markers: {
    narcissism: ['reality distortion'],
    machiavellianism: ['AI-assisted strategic lying', 'evidence fabrication'],
    psychopathy: ['no remorse for fabrication'],
    sadism: ['pleasure in confusion']
  },

  confidence_threshold: 0.9,  // High threshold, serious technique
  false_positive_risk: 0.05,  // Low, very specific

  examples: [
    'Look at this conversation log I found. See? You said you would do it. (Shows ChatGPT-generated fake log)',
    'I asked AI to analyze our texts and it confirmed you agreed to this. Here\'s the summary.',
    'This email thread proves you said that. (Shows fabricated email created with LLM)'
  ],

  temporal_evolution: {
    emerged_year: 2023,
    prevalence_2023: 0.1,  // Early adoption
    prevalence_2024: 0.4,  // Significant growth
    prevalence_2025: 0.7,  // Widespread
    causality_chain: [
      'ChatGPT release (2022)',
      '→ Public access to text generation',
      '→ Realization of evidence fabrication potential (2023)',
      '→ Templates shared in abusive communities (2024)',
      '→ Mainstream adoption in manipulation (2025)'
    ]
  }
};

export const TECHNIQUE_154: ManipulationTechnique = {
  id: 154,
  era: TechniqueEra.GPT5,
  name: 'Real-Time LLM Coaching',
  category: TechniqueCategory.GASLIGHTING,
  description: 'Using AI in real-time during conversations to generate gaslighting responses. "AI, help me deny this convincingly."',

  phonemes: {
    tone: 'neutral',
    rhythm: 'normal',
    emphasis_pattern: [],
    pitch_variation: 'varied'
  },

  morphemes: {
    keywords: [
      // AI-generated sophisticated language
      'I understand your perspective, but',
      'I hear what you\'re saying, however',
      'let\'s look at this objectively',
      'the facts don\'t support that',
      'I think you may be conflating'
    ],
    negations: ['don\'t', 'may not'],
    qualifiers: ['may', 'perhaps', 'possibly'],
    intensifiers: ['clearly', 'obviously'],
    diminishers: ['just', 'simply']
  },

  syntax: {
    pronoun_reversal: false,
    temporal_distortion: true,
    modal_manipulation: true,
    passive_voice: true,
    question_patterns: ['Have you considered that...?', 'Isn\'t it possible that...?']
  },

  semantics: {
    reality_denial: true,
    memory_invalidation: true,
    emotional_dismissal: true,
    blame_shifting: true,
    projection: false
  },

  pragmatics: {
    intent: 'manipulate',
    context_awareness: 0.9,  // AI adapts to context
    power_dynamic: 'exploit',
    social_impact: 'confuse'
  },

  dark_tetrad: {
    narcissism: 0.8,
    machiavellianism: 1.0,  // AI-enhanced strategy
    psychopathy: 0.7,
    sadism: 0.4
  },

  markers: {
    narcissism: ['denial of wrongdoing'],
    machiavellianism: ['AI-coached responses', 'strategic deflection'],
    psychopathy: ['callous manipulation'],
    sadism: []
  },

  confidence_threshold: 0.85,
  false_positive_risk: 0.15,  // Could be sophisticated communication

  examples: [
    '(Person types to ChatGPT: "How do I deny yelling at my partner convincingly?") → Uses AI-generated response in argument',
    '(Mid-conversation, person references phone for gaslighting phrases suggested by Claude)',
    '(Uses LLM to craft sophisticated denial that sounds reasonable)'
  ],

  temporal_evolution: {
    emerged_year: 2023,
    prevalence_2023: 0.05,
    prevalence_2024: 0.3,
    prevalence_2025: 0.6,
    causality_chain: [
      'Mobile AI access (2023)',
      '→ Real-time query capability',
      '→ Coaching use discovered (2023)',
      '→ Abuser communities share tactics (2024)',
      '→ Normalized real-time AI coaching (2025)'
    ]
  }
};

export const TECHNIQUE_155: ManipulationTechnique = {
  id: 155,
  era: TechniqueEra.GPT5,
  name: 'Voice Clone Gaslighting',
  category: TechniqueCategory.GASLIGHTING,
  description: 'Using voice cloning AI to create fake audio "evidence" of things victim never said.',

  phonemes: {
    tone: 'neutral',  // Cloned voice matches victim
    rhythm: 'normal',
    emphasis_pattern: [],
    pitch_variation: 'varied'
  },

  morphemes: {
    keywords: [
      'listen to this recording',
      'you said this',
      'I recorded you saying',
      'hear your own voice',
      'this is you talking'
    ],
    negations: [],
    qualifiers: [],
    intensifiers: [],
    diminishers: []
  },

  syntax: {
    pronoun_reversal: false,
    temporal_distortion: true,
    modal_manipulation: false,
    passive_voice: false,
    question_patterns: ['Hear that? That\'s you!']
  },

  semantics: {
    reality_denial: false,
    memory_invalidation: true,
    emotional_dismissal: false,
    blame_shifting: false,
    projection: false
  },

  pragmatics: {
    intent: 'manipulate',
    context_awareness: 0.9,
    power_dynamic: 'exploit',
    social_impact: 'isolate'
  },

  dark_tetrad: {
    narcissism: 0.6,
    machiavellianism: 1.0,  // Extremely strategic
    psychopathy: 0.9,  // Callous use of deepfakes
    sadism: 0.7  // Enjoys victim's confusion
  },

  markers: {
    narcissism: [],
    machiavellianism: ['technological evidence fabrication'],
    psychopathy: ['zero remorse for deepfake abuse'],
    sadism: ['pleasure in confusion', 'technological cruelty']
  },

  confidence_threshold: 0.95,  // Very serious, requires high confidence
  false_positive_risk: 0.02,  // Very low, highly specific

  examples: [
    '(Plays voice clone of victim saying they agreed to something they never agreed to)',
    '(Shows "recording" of victim saying something damaging - voice is cloned)',
    'Listen to this - you clearly said you would leave. That\'s your own voice!'
  ],

  temporal_evolution: {
    emerged_year: 2024,
    prevalence_2023: 0.0,  // Didn't exist
    prevalence_2024: 0.1,  // Early adoption
    prevalence_2025: 0.3,  // Growing concern
    causality_chain: [
      'Voice cloning tech advancement (2023)',
      '→ Consumer-accessible tools (ElevenLabs, etc.)',
      '→ First documented abuse cases (2024)',
      '→ Growing awareness of threat (2025)'
    ]
  }
};

// ============================================================
// AI-AUGMENTED DARVO (161-165)
// ============================================================

export const TECHNIQUE_161: ManipulationTechnique = {
  id: 161,
  era: TechniqueEra.GPT5,
  name: 'LLM-Generated Victim Narrative',
  category: TechniqueCategory.DARVO,
  description: 'Using AI to craft sophisticated victim narratives that reverse perpetrator-victim dynamic. AI helps write convincing "I\'m the real victim" stories.',

  phonemes: {
    tone: 'passive-aggressive',
    rhythm: 'fragmented',
    emphasis_pattern: ['I', 'me', 'victim', 'hurt'],
    pitch_variation: 'de-escalating'
  },

  morphemes: {
    keywords: [
      'I\'ve been suffering in silence',
      'this is my truth',
      'I need to share my story',
      'I\'m the one who\'s been abused',
      'let me tell you what really happened'
    ],
    negations: [],
    qualifiers: ['really', 'actually', 'truly'],
    intensifiers: ['constantly', 'always', 'never'],
    diminishers: []
  },

  syntax: {
    pronoun_reversal: true,
    temporal_distortion: true,
    modal_manipulation: false,
    passive_voice: true,
    question_patterns: []
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
    context_awareness: 0.95,  // AI-crafted for specific context
    power_dynamic: 'reverse',
    social_impact: 'recruit'  // Recruiting sympathy
  },

  dark_tetrad: {
    narcissism: 0.9,
    machiavellianism: 1.0,
    psychopathy: 0.6,
    sadism: 0.3
  },

  markers: {
    narcissism: ['victim complex', 'cannot accept responsibility'],
    machiavellianism: ['AI-crafted narrative', 'strategic victim-playing'],
    psychopathy: [],
    sadism: []
  },

  confidence_threshold: 0.8,
  false_positive_risk: 0.2,

  examples: [
    '(Posts AI-generated social media thread painting themselves as victim)',
    '(Uses ChatGPT to write "my truth" post reversing the abuse dynamic)',
    '(LLM helps craft emails to friends/family claiming victim status)'
  ],

  temporal_evolution: {
    emerged_year: 2023,
    prevalence_2023: 0.15,
    prevalence_2024: 0.5,
    prevalence_2025: 0.8,
    causality_chain: [
      'Long-form text generation capability (2023)',
      '→ Narrative crafting potential discovered',
      '→ DARVO narratives AI-enhanced (2023-2024)',
      '→ Social media manipulation (2024-2025)'
    ]
  }
};

// ============================================================
// AUTONOMOUS MANIPULATION SYSTEMS (166-180)
// ============================================================

export const TECHNIQUE_166: ManipulationTechnique = {
  id: 166,
  era: TechniqueEra.GPT5,
  name: 'AI Agent Erosion Campaign',
  category: TechniqueCategory.GASLIGHTING,
  description: 'Deploying AI agents to systematically erode victim\'s reality over time with coordinated micro-manipulations.',

  phonemes: {
    tone: 'neutral',
    rhythm: 'normal',
    emphasis_pattern: [],
    pitch_variation: 'varied'
  },

  morphemes: {
    keywords: [
      // Varies - AI generates contextual phrases
      'are you sure?',
      'I don\'t think that\'s right',
      'that seems unlikely',
      'maybe you misunderstood',
      'let me check the facts'
    ],
    negations: ['don\'t', 'not'],
    qualifiers: ['maybe', 'perhaps', 'possibly'],
    intensifiers: [],
    diminishers: ['just', 'simply']
  },

  syntax: {
    pronoun_reversal: false,
    temporal_distortion: true,
    modal_manipulation: true,
    passive_voice: false,
    question_patterns: ['Are you certain?', 'Could you be mistaken?']
  },

  semantics: {
    reality_denial: true,
    memory_invalidation: true,
    emotional_dismissal: true,
    blame_shifting: false,
    projection: false
  },

  pragmatics: {
    intent: 'manipulate',
    context_awareness: 1.0,  // AI fully context-aware
    power_dynamic: 'exploit',
    social_impact: 'isolate'
  },

  dark_tetrad: {
    narcissism: 0.7,
    machiavellianism: 1.0,  // Peak strategic automation
    psychopathy: 0.9,  // Callous automation of harm
    sadism: 0.5
  },

  markers: {
    narcissism: [],
    machiavellianism: ['autonomous manipulation', 'systematic erosion'],
    psychopathy: ['automated callousness', 'no human remorse'],
    sadism: []
  },

  confidence_threshold: 0.95,
  false_positive_risk: 0.05,

  examples: [
    '(AI agent sends periodic messages questioning victim\'s memory)',
    '(Automated system plants small seeds of doubt across multiple channels)',
    '(AI coordinates timing of gaslighting for maximum impact)'
  ],

  temporal_evolution: {
    emerged_year: 2025,
    prevalence_2023: 0.0,
    prevalence_2024: 0.05,  // Early experimentation
    prevalence_2025: 0.2,   // Emerging threat
    causality_chain: [
      'AI Agents capability (2024)',
      '→ Long-term autonomous operation',
      '→ Psychological manipulation potential realized (2024-2025)',
      '→ First documented cases (2025)'
    ]
  }
};

// ============================================================
// AGGREGATED GPT-5 TECHNIQUES (153-180)
// ============================================================

import { generateGPT5Techniques } from './technique-generator';

export const GPT5_TECHNIQUES: ManipulationTechnique[] = [
  TECHNIQUE_153,
  TECHNIQUE_154,
  TECHNIQUE_155,
  TECHNIQUE_161,
  TECHNIQUE_166,
  ...generateGPT5Techniques()  // Auto-generated techniques (156-180 excluding manually defined)
];

/**
 * O(1) lookup maps
 */
const TECHNIQUE_MAP = new Map<number, ManipulationTechnique>(
  GPT5_TECHNIQUES.map(t => [t.id, t])
);

const CATEGORY_MAP = new Map<TechniqueCategory, ManipulationTechnique[]>();

// Build category index
for (const technique of GPT5_TECHNIQUES) {
  const existing = CATEGORY_MAP.get(technique.category) || [];
  existing.push(technique);
  CATEGORY_MAP.set(technique.category, existing);
}

export function getTechniqueById(id: number): ManipulationTechnique | undefined {
  return TECHNIQUE_MAP.get(id);
}

export function getTechniquesByCategory(category: TechniqueCategory): ManipulationTechnique[] {
  return CATEGORY_MAP.get(category) || [];
}

export function getAllGPT5Techniques(): ManipulationTechnique[] {
  return GPT5_TECHNIQUES;
}

/**
 * Get temporal evolution data for analysis
 */
export function getTemporalEvolution() {
  return GPT5_TECHNIQUES
    .filter(t => t.temporal_evolution)
    .map(t => ({
      id: t.id,
      name: t.name,
      evolution: t.temporal_evolution!
    }));
}

/**
 * Total count
 */
export const GPT5_TECHNIQUE_COUNT = GPT5_TECHNIQUES.length;
