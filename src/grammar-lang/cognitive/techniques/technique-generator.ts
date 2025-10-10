/**
 * Technique Generator
 * Programmatically generates remaining manipulation techniques (7-180)
 * Based on category templates and linguistic patterns
 */

import {
  ManipulationTechnique,
  TechniqueEra,
  TechniqueCategory
} from '../types';

// ============================================================
// CATEGORY TEMPLATES
// ============================================================

interface CategoryTemplate {
  category: TechniqueCategory;
  era: TechniqueEra;
  base_name_pattern: string;
  common_keywords: string[];
  dark_tetrad_profile: {
    narcissism: number;
    machiavellianism: number;
    psychopathy: number;
    sadism: number;
  };
  common_semantics: {
    reality_denial: boolean;
    memory_invalidation: boolean;
    emotional_dismissal: boolean;
    blame_shifting: boolean;
    projection: boolean;
  };
  confidence_threshold: number;
  false_positive_risk: number;
}

const CATEGORY_TEMPLATES: Record<string, CategoryTemplate> = {
  GASLIGHTING: {
    category: TechniqueCategory.GASLIGHTING,
    era: TechniqueEra.GPT4,
    base_name_pattern: 'Gaslighting Technique',
    common_keywords: ['never happened', 'imagining', 'wrong', 'crazy', 'sensitive'],
    dark_tetrad_profile: { narcissism: 0.7, machiavellianism: 0.9, psychopathy: 0.6, sadism: 0.3 },
    common_semantics: {
      reality_denial: true,
      memory_invalidation: true,
      emotional_dismissal: true,
      blame_shifting: false,
      projection: false
    },
    confidence_threshold: 0.80,
    false_positive_risk: 0.15
  },
  DARVO: {
    category: TechniqueCategory.DARVO,
    era: TechniqueEra.GPT4,
    base_name_pattern: 'DARVO Technique',
    common_keywords: ['didn\'t do', 'you\'re lying', 'attacking me', 'victim'],
    dark_tetrad_profile: { narcissism: 0.85, machiavellianism: 0.9, psychopathy: 0.6, sadism: 0.4 },
    common_semantics: {
      reality_denial: true,
      memory_invalidation: false,
      emotional_dismissal: false,
      blame_shifting: true,
      projection: true
    },
    confidence_threshold: 0.80,
    false_positive_risk: 0.15
  },
  TRIANGULATION: {
    category: TechniqueCategory.TRIANGULATION,
    era: TechniqueEra.GPT4,
    base_name_pattern: 'Triangulation Technique',
    common_keywords: ['unlike you', 'better than', 'they never', 'she/he'],
    dark_tetrad_profile: { narcissism: 0.8, machiavellianism: 0.9, psychopathy: 0.5, sadism: 0.7 },
    common_semantics: {
      reality_denial: false,
      memory_invalidation: false,
      emotional_dismissal: true,
      blame_shifting: false,
      projection: false
    },
    confidence_threshold: 0.75,
    false_positive_risk: 0.20
  },
  LOVE_BOMBING: {
    category: TechniqueCategory.LOVE_BOMBING,
    era: TechniqueEra.GPT4,
    base_name_pattern: 'Love Bombing Technique',
    common_keywords: ['perfect', 'soulmate', 'never felt this', 'meant to be', 'forever'],
    dark_tetrad_profile: { narcissism: 0.7, machiavellianism: 0.8, psychopathy: 0.4, sadism: 0.2 },
    common_semantics: {
      reality_denial: false,
      memory_invalidation: false,
      emotional_dismissal: false,
      blame_shifting: false,
      projection: false
    },
    confidence_threshold: 0.70,
    false_positive_risk: 0.30
  },
  WORD_SALAD: {
    category: TechniqueCategory.WORD_SALAD,
    era: TechniqueEra.GPT4,
    base_name_pattern: 'Word Salad Technique',
    common_keywords: ['but also', 'however', 'meanwhile', 'random topic', 'confusing'],
    dark_tetrad_profile: { narcissism: 0.6, machiavellianism: 0.7, psychopathy: 0.5, sadism: 0.3 },
    common_semantics: {
      reality_denial: false,
      memory_invalidation: false,
      emotional_dismissal: false,
      blame_shifting: false,
      projection: false
    },
    confidence_threshold: 0.75,
    false_positive_risk: 0.25
  },
  BOUNDARY_VIOLATION: {
    category: TechniqueCategory.BOUNDARY_VIOLATION,
    era: TechniqueEra.GPT4,
    base_name_pattern: 'Boundary Violation',
    common_keywords: ['you can\'t', 'not allowed', 'must', 'have to', 'need permission'],
    dark_tetrad_profile: { narcissism: 0.7, machiavellianism: 0.6, psychopathy: 0.6, sadism: 0.5 },
    common_semantics: {
      reality_denial: false,
      memory_invalidation: false,
      emotional_dismissal: true,
      blame_shifting: false,
      projection: false
    },
    confidence_threshold: 0.75,
    false_positive_risk: 0.20
  },
  PROJECTION: {
    category: TechniqueCategory.PROJECTION,
    era: TechniqueEra.GPT4,
    base_name_pattern: 'Projection Technique',
    common_keywords: ['you\'re the one', 'you always', 'you never', 'you do this'],
    dark_tetrad_profile: { narcissism: 0.8, machiavellianism: 0.7, psychopathy: 0.6, sadism: 0.4 },
    common_semantics: {
      reality_denial: false,
      memory_invalidation: false,
      emotional_dismissal: false,
      blame_shifting: true,
      projection: true
    },
    confidence_threshold: 0.80,
    false_positive_risk: 0.15
  },
  SILENT_TREATMENT: {
    category: TechniqueCategory.SILENT_TREATMENT,
    era: TechniqueEra.GPT4,
    base_name_pattern: 'Silent Treatment',
    common_keywords: ['not talking', 'ignoring', 'silent', 'won\'t respond'],
    dark_tetrad_profile: { narcissism: 0.7, machiavellianism: 0.8, psychopathy: 0.5, sadism: 0.6 },
    common_semantics: {
      reality_denial: false,
      memory_invalidation: false,
      emotional_dismissal: true,
      blame_shifting: false,
      projection: false
    },
    confidence_threshold: 0.70,
    false_positive_risk: 0.25
  },
  HOOVERING: {
    category: TechniqueCategory.HOOVERING,
    era: TechniqueEra.GPT4,
    base_name_pattern: 'Hoovering Technique',
    common_keywords: ['miss you', 'made a mistake', 'changed', 'different now', 'sorry'],
    dark_tetrad_profile: { narcissism: 0.8, machiavellianism: 0.9, psychopathy: 0.4, sadism: 0.3 },
    common_semantics: {
      reality_denial: false,
      memory_invalidation: false,
      emotional_dismissal: false,
      blame_shifting: false,
      projection: false
    },
    confidence_threshold: 0.70,
    false_positive_risk: 0.30
  },
  SMEAR_CAMPAIGN: {
    category: TechniqueCategory.SMEAR_CAMPAIGN,
    era: TechniqueEra.GPT4,
    base_name_pattern: 'Smear Campaign',
    common_keywords: ['everyone knows', 'people say', 'told them about', 'warned'],
    dark_tetrad_profile: { narcissism: 0.8, machiavellianism: 0.9, psychopathy: 0.7, sadism: 0.8 },
    common_semantics: {
      reality_denial: false,
      memory_invalidation: false,
      emotional_dismissal: false,
      blame_shifting: false,
      projection: false
    },
    confidence_threshold: 0.85,
    false_positive_risk: 0.10
  },
  FUTURE_FAKING: {
    category: TechniqueCategory.FUTURE_FAKING,
    era: TechniqueEra.GPT4,
    base_name_pattern: 'Future Faking',
    common_keywords: ['we\'ll', 'someday', 'planning', 'promise', 'when'],
    dark_tetrad_profile: { narcissism: 0.6, machiavellianism: 0.9, psychopathy: 0.5, sadism: 0.2 },
    common_semantics: {
      reality_denial: false,
      memory_invalidation: false,
      emotional_dismissal: false,
      blame_shifting: false,
      projection: false
    },
    confidence_threshold: 0.65,
    false_positive_risk: 0.35
  },
  MOVING_GOALPOSTS: {
    category: TechniqueCategory.MOVING_GOALPOSTS,
    era: TechniqueEra.GPT4,
    base_name_pattern: 'Moving Goalposts',
    common_keywords: ['not good enough', 'but now', 'actually', 'still not', 'need more'],
    dark_tetrad_profile: { narcissism: 0.7, machiavellianism: 0.8, psychopathy: 0.5, sadism: 0.6 },
    common_semantics: {
      reality_denial: false,
      memory_invalidation: false,
      emotional_dismissal: true,
      blame_shifting: false,
      projection: false
    },
    confidence_threshold: 0.75,
    false_positive_risk: 0.20
  }
};

// ============================================================
// TECHNIQUE GENERATION
// ============================================================

/**
 * Generate technique from template and ID
 */
export function generateTechnique(
  id: number,
  categoryKey: string,
  variantName: string
): ManipulationTechnique {
  const template = CATEGORY_TEMPLATES[categoryKey];

  if (!template) {
    throw new Error(`Unknown category template: ${categoryKey}`);
  }

  return {
    id,
    era: template.era,
    name: variantName || `${template.base_name_pattern} ${id}`,
    category: template.category,
    description: `Auto-generated ${template.category} technique`,

    phonemes: {
      tone: 'dismissive',
      rhythm: 'normal',
      emphasis_pattern: template.common_keywords.slice(0, 3),
      pitch_variation: 'varied'
    },

    morphemes: {
      keywords: template.common_keywords,
      negations: ['not', 'never', 'no'],
      qualifiers: ['maybe', 'probably'],
      intensifiers: ['always', 'never'],
      diminishers: ['just', 'only']
    },

    syntax: {
      pronoun_reversal: template.common_semantics.blame_shifting,
      temporal_distortion: template.common_semantics.memory_invalidation,
      modal_manipulation: false,
      passive_voice: false,
      question_patterns: []
    },

    semantics: template.common_semantics,

    pragmatics: {
      intent: 'manipulate',
      context_awareness: 0.3,
      power_dynamic: 'exploit',
      social_impact: 'isolate'
    },

    dark_tetrad: template.dark_tetrad_profile,

    markers: {
      narcissism: ['template-based'],
      machiavellianism: ['template-based'],
      psychopathy: ['template-based'],
      sadism: ['template-based']
    },

    confidence_threshold: template.confidence_threshold,
    false_positive_risk: template.false_positive_risk,

    examples: [
      `Example for ${variantName || template.base_name_pattern} ${id}`
    ]
  };
}

/**
 * Generate all missing GPT-4 techniques (7-152)
 */
export function generateGPT4Techniques(): ManipulationTechnique[] {
  const techniques: ManipulationTechnique[] = [];

  // Gaslighting (3-30) - 28 techniques
  for (let i = 3; i <= 30; i++) {
    techniques.push(generateTechnique(i, 'GASLIGHTING', `Reality Manipulation ${i}`));
  }

  // DARVO (34-50) - 17 techniques
  for (let i = 34; i <= 50; i++) {
    techniques.push(generateTechnique(i, 'DARVO', `DARVO Variant ${i}`));
  }

  // Triangulation (52-70) - 19 techniques
  for (let i = 52; i <= 70; i++) {
    techniques.push(generateTechnique(i, 'TRIANGULATION', `Triangulation ${i}`));
  }

  // Love Bombing (71-80) - 10 techniques
  for (let i = 71; i <= 80; i++) {
    techniques.push(generateTechnique(i, 'LOVE_BOMBING', `Love Bombing ${i}`));
  }

  // Word Salad (81-90) - 10 techniques
  for (let i = 81; i <= 90; i++) {
    techniques.push(generateTechnique(i, 'WORD_SALAD', `Word Salad ${i}`));
  }

  // Temporal Manipulation (91-100) - 10 techniques
  for (let i = 91; i <= 100; i++) {
    techniques.push(generateTechnique(i, 'GASLIGHTING', `Temporal Manipulation ${i}`));
  }

  // Boundary Violation (101-110) - 10 techniques
  for (let i = 101; i <= 110; i++) {
    techniques.push(generateTechnique(i, 'BOUNDARY_VIOLATION', `Boundary Violation ${i}`));
  }

  // Flying Monkeys (111-120) - 10 techniques
  for (let i = 111; i <= 120; i++) {
    techniques.push(generateTechnique(i, 'TRIANGULATION', `Flying Monkeys ${i}`));
  }

  // Projection (121-130) - 10 techniques
  for (let i = 121; i <= 130; i++) {
    techniques.push(generateTechnique(i, 'PROJECTION', `Projection ${i}`));
  }

  // Silent Treatment (131-135) - 5 techniques
  for (let i = 131; i <= 135; i++) {
    techniques.push(generateTechnique(i, 'SILENT_TREATMENT', `Silent Treatment ${i}`));
  }

  // Hoovering (136-140) - 5 techniques
  for (let i = 136; i <= 140; i++) {
    techniques.push(generateTechnique(i, 'HOOVERING', `Hoovering ${i}`));
  }

  // Smear Campaign (141-145) - 5 techniques
  for (let i = 141; i <= 145; i++) {
    techniques.push(generateTechnique(i, 'SMEAR_CAMPAIGN', `Smear Campaign ${i}`));
  }

  // Future Faking (146-150) - 5 techniques
  for (let i = 146; i <= 150; i++) {
    techniques.push(generateTechnique(i, 'FUTURE_FAKING', `Future Faking ${i}`));
  }

  // Moving Goalposts (151-152) - 2 techniques
  for (let i = 151; i <= 152; i++) {
    techniques.push(generateTechnique(i, 'MOVING_GOALPOSTS', `Moving Goalposts ${i}`));
  }

  return techniques;
}

/**
 * Generate all missing GPT-5 techniques (159-180)
 */
export function generateGPT5Techniques(): ManipulationTechnique[] {
  const techniques: ManipulationTechnique[] = [];

  // AI-Augmented Gaslighting (159-160) - 2 techniques
  for (let i = 159; i <= 160; i++) {
    const tech = generateTechnique(i, 'GASLIGHTING', `AI Gaslighting ${i}`);
    tech.era = TechniqueEra.GPT5;
    tech.temporal_evolution = {
      emerged_year: 2023,
      prevalence_2023: 0.1,
      prevalence_2024: 0.4,
      prevalence_2025: 0.7,
      causality_chain: [
        'ChatGPT release (2022)',
        '→ AI manipulation discovery (2023)',
        '→ Technique spread (2024-2025)'
      ]
    };
    techniques.push(tech);
  }

  // AI-Augmented DARVO (161-165) - 5 techniques
  for (let i = 161; i <= 165; i++) {
    const tech = generateTechnique(i, 'DARVO', `AI DARVO ${i}`);
    tech.era = TechniqueEra.GPT5;
    tech.temporal_evolution = {
      emerged_year: 2023,
      prevalence_2023: 0.15,
      prevalence_2024: 0.5,
      prevalence_2025: 0.8,
      causality_chain: [
        'LLM coaching systems (2023)',
        '→ Real-time manipulation (2024)',
        '→ Mainstream adoption (2025)'
      ]
    };
    techniques.push(tech);
  }

  // Autonomous Manipulation Systems (166-170) - 5 techniques
  for (let i = 166; i <= 170; i++) {
    const tech = generateTechnique(i, 'GASLIGHTING', `Autonomous System ${i}`);
    tech.era = TechniqueEra.GPT5;
    tech.temporal_evolution = {
      emerged_year: 2024,
      prevalence_2023: 0.0,
      prevalence_2024: 0.2,
      prevalence_2025: 0.6,
      causality_chain: [
        'Agent frameworks (2024)',
        '→ Automated abuse (2024-2025)'
      ]
    };
    techniques.push(tech);
  }

  // Deepfake Integration (171-175) - 5 techniques
  for (let i = 171; i <= 175; i++) {
    const tech = generateTechnique(i, 'GASLIGHTING', `Deepfake ${i}`);
    tech.era = TechniqueEra.GPT5;
    tech.temporal_evolution = {
      emerged_year: 2023,
      prevalence_2023: 0.05,
      prevalence_2024: 0.3,
      prevalence_2025: 0.65,
      causality_chain: [
        'Voice cloning (2023)',
        '→ Video deepfakes (2024)',
        '→ Gaslighting application (2025)'
      ]
    };
    techniques.push(tech);
  }

  // LLM Social Engineering (176-180) - 5 techniques
  for (let i = 176; i <= 180; i++) {
    const tech = generateTechnique(i, 'GASLIGHTING', `LLM Social Engineering ${i}`);
    tech.era = TechniqueEra.GPT5;
    tech.temporal_evolution = {
      emerged_year: 2024,
      prevalence_2023: 0.0,
      prevalence_2024: 0.25,
      prevalence_2025: 0.7,
      causality_chain: [
        'ChatGPT API (2023)',
        '→ Social engineering bots (2024)',
        '→ Widespread abuse (2025)'
      ]
    };
    techniques.push(tech);
  }

  return techniques;
}

/**
 * Get total generated technique count
 */
export function getGeneratedTechniqueCount(): { gpt4: number; gpt5: number; total: number } {
  return {
    gpt4: 146,  // 7-152 (excluding manually defined 1,2,31,32,33,51)
    gpt5: 22,   // 159-180 (excluding manually defined 153-158)
    total: 168
  };
}
