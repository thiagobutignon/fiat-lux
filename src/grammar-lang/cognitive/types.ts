/**
 * Cognitive OS - Type Definitions
 * Detection engine for manipulation techniques
 */

// ============================================================
// LINGUISTIC STRUCTURE (Chomsky Hierarchy)
// ============================================================

/**
 * PHONEMES - Sound patterns and tone analysis
 */
export interface Phonemes {
  tone: 'neutral' | 'condescending' | 'dismissive' | 'patronizing' | 'aggressive' | 'passive-aggressive';
  rhythm: 'normal' | 'rushed' | 'fragmented' | 'repetitive';
  emphasis_pattern: string[];  // Which words are emphasized
  pitch_variation: 'monotone' | 'varied' | 'escalating' | 'de-escalating';
}

/**
 * MORPHEMES - Minimal units of meaning
 */
export interface Morphemes {
  keywords: string[];           // Key manipulation phrases
  negations: string[];          // Denial patterns
  qualifiers: string[];         // "maybe", "probably", "might"
  intensifiers: string[];       // "always", "never", "absolutely"
  diminishers: string[];        // "just", "only", "merely"
}

/**
 * SYNTAX - Grammatical structure
 */
export interface Syntax {
  pronoun_reversal: boolean;    // "I didn't" → "You did"
  temporal_distortion: boolean; // Time manipulation
  modal_manipulation: boolean;  // Misuse of "could", "should", "would"
  passive_voice: boolean;       // Avoiding responsibility
  question_patterns: string[];  // Leading questions
}

/**
 * SEMANTICS - Meaning analysis
 */
export interface Semantics {
  reality_denial: boolean;      // Denying facts
  memory_invalidation: boolean; // "That never happened"
  emotional_dismissal: boolean; // "You're overreacting"
  blame_shifting: boolean;      // Transferring fault
  projection: boolean;          // Attributing own behavior to others
}

/**
 * PRAGMATICS - Intent and context
 */
export interface Pragmatics {
  intent: 'manipulate' | 'control' | 'confuse' | 'harm' | 'dominate' | 'deceive';
  context_awareness: number;    // 0-1 (how context-dependent)
  power_dynamic: 'exploit' | 'equalize' | 'reverse';
  social_impact: 'isolate' | 'triangulate' | 'recruit' | 'divide';
}

// ============================================================
// DARK TETRAD
// ============================================================

export interface DarkTetradScores {
  narcissism: number;      // 0-1
  machiavellianism: number; // 0-1
  psychopathy: number;     // 0-1
  sadism: number;          // 0-1
}

export interface DarkTetradMarkers {
  narcissism: string[];        // Grandiosity, entitlement markers
  machiavellianism: string[];  // Strategic deception markers
  psychopathy: string[];       // Callousness, lack of remorse
  sadism: string[];            // Pleasure in harm markers
}

// ============================================================
// MANIPULATION TECHNIQUES
// ============================================================

export enum TechniqueEra {
  GPT4 = 'gpt4',    // Techniques 1-152 (classical, well-documented)
  GPT5 = 'gpt5'     // Techniques 153-180 (emergent 2023-2025)
}

export enum TechniqueCategory {
  GASLIGHTING = 'gaslighting',
  TRIANGULATION = 'triangulation',
  LOVE_BOMBING = 'love_bombing',
  DARVO = 'darvo',
  WORD_SALAD = 'word_salad',
  TEMPORAL_MANIPULATION = 'temporal_manipulation',
  BOUNDARY_VIOLATION = 'boundary_violation',
  FLYING_MONKEYS = 'flying_monkeys',
  PROJECTION = 'projection',
  SILENT_TREATMENT = 'silent_treatment',
  HOOVERING = 'hoovering',
  SMEAR_CAMPAIGN = 'smear_campaign',
  FUTURE_FAKING = 'future_faking',
  MOVING_GOALPOSTS = 'moving_goalposts',
  EMOTIONAL_BLACKMAIL = 'emotional_blackmail'
}

export interface ManipulationTechnique {
  id: number;                           // 1-180
  era: TechniqueEra;                    // GPT-4 or GPT-5
  name: string;                         // Human-readable name
  category: TechniqueCategory;          // Primary category
  description: string;                  // Detailed explanation

  // Linguistic structure
  phonemes: Phonemes;
  morphemes: Morphemes;
  syntax: Syntax;
  semantics: Semantics;
  pragmatics: Pragmatics;

  // Dark Tetrad alignment
  dark_tetrad: DarkTetradScores;
  markers: DarkTetradMarkers;

  // Detection
  confidence_threshold: number;         // 0-1 (minimum for detection)
  false_positive_risk: number;          // 0-1 (neurodivergent consideration)

  // Examples
  examples: string[];                   // Real-world examples

  // Temporal (GPT-5 only)
  temporal_evolution?: {
    emerged_year: number;               // When first documented
    prevalence_2023: number;            // 0-1
    prevalence_2024: number;            // 0-1
    prevalence_2025: number;            // 0-1
    causality_chain: string[];          // What enabled this technique
  };
}

// ============================================================
// DETECTION
// ============================================================

export interface DetectionResult {
  technique_id: number;
  technique_name: string;
  confidence: number;                   // 0-1
  matched_markers: {
    phonemes: Partial<Phonemes>;
    morphemes: Partial<Morphemes>;
    syntax: Partial<Syntax>;
    semantics: Partial<Semantics>;
    pragmatics: Partial<Pragmatics>;
  };
  dark_tetrad: DarkTetradScores;
  explanation: string;                  // Glass box: why detected
  sources: string[];                    // Which linguistic elements triggered
  neurodivergent_flag: boolean;        // True if false-positive risk high
  validated: boolean;                   // Constitutional validation
}

export interface AttentionTrace {
  sources: string[];                    // Input text segments
  weights: number[];                    // Attention weights (0-1)
  patterns: string[];                   // Which patterns matched
}

// ============================================================
// CONSTITUTIONAL AI
// ============================================================

export interface ConstitutionalPrinciples {
  privacy: boolean;                     // Never store personal data
  transparency: boolean;                // All detections explainable
  protection: boolean;                  // Prioritize neurodivergent safety
  accuracy: boolean;                    // >95% precision target
  no_diagnosis: boolean;                // Detect patterns, not label people
  context_aware: boolean;               // Cultural sensitivity
  evidence_based: boolean;              // Cite linguistic markers
}

export interface ConstitutionalValidation {
  compliant: boolean;
  violations: string[];
  warnings: string[];
  adjusted_confidence: number;          // After neurodivergent protection
}

// ============================================================
// COGNITIVE GLASS ORGANISM
// ============================================================

export interface CognitiveOrganism {
  format: 'fiat-glass-v1.0';
  type: 'cognitive-defense-organism';

  metadata: {
    name: string;
    version: string;
    specialization: 'manipulation-detection';
    maturity: number;                   // 0-1
    techniques_count: number;           // 180
    created: string;
    generation: number;
  };

  model: {
    architecture: string;
    parameters: number;
    constitutional: boolean;
    focus: 'linguistic-analysis';
  };

  knowledge: {
    techniques: ManipulationTechnique[];
    dark_tetrad_markers: DarkTetradMarkers;
    temporal_tracking: {
      start_year: number;
      end_year: number;
      evolution_log: any[];
    };
    neurodivergent_protection: {
      autism_markers: string[];
      adhd_markers: string[];
      false_positive_threshold: number;
    };
  };

  code: {
    functions: {
      name: string;
      signature: string;
      implementation: string;
      emerged_from: string;
      confidence: number;
    }[];
    emergence_log: any[];
  };

  memory: {
    detected_patterns: any[];
    false_positives: any[];
    evolution_log: any[];
    audit_trail: any[];
  };

  constitutional: ConstitutionalPrinciples;

  evolution: {
    enabled: boolean;
    last_evolution: string;
    generations: number;
    fitness_trajectory: number[];
  };
}

// ============================================================
// PATTERN MATCHING
// ============================================================

export interface PatternMatchConfig {
  technique_ids?: number[];             // Filter specific techniques
  categories?: TechniqueCategory[];     // Filter categories
  min_confidence?: number;              // Minimum confidence (default: 0.8)
  enable_neurodivergent_protection?: boolean; // Default: true
  context?: string;                     // Additional context for analysis
}

export interface PatternMatchResult {
  detections: DetectionResult[];
  total_matches: number;
  highest_confidence: number;
  dark_tetrad_aggregate: DarkTetradScores;
  attention_trace: AttentionTrace;
  constitutional_validation: ConstitutionalValidation;
  processing_time_ms: number;
}
