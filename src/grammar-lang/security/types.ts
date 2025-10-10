/**
 * Security & Behavioral Analysis - Type Definitions
 *
 * Biological-inspired security system for .glass organisms
 * Based on behavioral patterns, not passwords
 */

// ============================================================================
// LINGUISTIC FINGERPRINTING
// ============================================================================

/**
 * Linguistic profile - unique "fingerprint" of user's language patterns
 * Like DNA for communication style
 */
export interface LinguisticProfile {
  user_id: string;
  created_at: number;
  last_updated: number;

  // Lexical patterns (vocabulary)
  vocabulary: {
    distribution: Map<string, number>;      // Word frequency
    unique_words: Set<string>;              // Vocabulary size
    average_word_length: number;
    rare_words_frequency: number;           // How often uses uncommon words
  };

  // Syntactic patterns (grammar)
  syntax: {
    average_sentence_length: number;
    sentence_length_variance: number;
    punctuation_patterns: Map<string, number>;
    grammar_preferences: GrammarPattern[];
    passive_voice_frequency: number;
    question_frequency: number;
  };

  // Semantic patterns (meaning)
  semantics: {
    topic_distribution: Map<string, number>;
    sentiment_baseline: number;             // -1 (negative) to +1 (positive)
    sentiment_variance: number;
    formality_level: number;                // 0 (casual) to 1 (formal)
    hedging_frequency: number;              // "maybe", "I think", etc.
  };

  // Metadata
  samples_analyzed: number;
  confidence: number;                       // 0-1 (more samples = higher)
}

export interface GrammarPattern {
  pattern_type: 'subject_verb_object' | 'verb_subject_object' | 'passive' | 'question' | 'imperative';
  frequency: number;
  examples: string[];
}

// ============================================================================
// TYPING & INTERACTION PATTERNS
// ============================================================================

export interface TypingProfile {
  user_id: string;
  created_at: number;
  last_updated: number;

  // Timing patterns
  timing: {
    keystroke_intervals: number[];          // Time between keystrokes (ms)
    keystroke_interval_avg: number;
    keystroke_interval_variance: number;
    word_pause_duration: number;            // Pause between words
    thinking_pause_duration: number;        // Pause before responding
    sentence_pause_duration: number;        // Pause between sentences
  };

  // Error patterns
  errors: {
    typo_rate: number;                      // Typos per 100 characters
    correction_patterns: string[];          // How they fix mistakes
    backspace_frequency: number;
    delete_frequency: number;
    common_typos: Map<string, string>;      // "teh" -> "the"
  };

  // Input behavior
  input: {
    copy_paste_frequency: number;
    input_burst_detected: boolean;          // Sudden paste of large text
    edit_distance_avg: number;              // How much they edit
    session_length_avg: number;             // Average session duration
  };

  // Device fingerprint
  device: {
    keyboard_layout: string;                // US, BR, etc.
    typical_device_type: 'mobile' | 'desktop' | 'tablet';
    browser_fingerprint?: string;
  };

  samples_analyzed: number;
  confidence: number;
}

// ============================================================================
// EMOTIONAL SIGNATURE
// ============================================================================

export interface EmotionalProfile {
  user_id: string;
  created_at: number;
  last_updated: number;

  // Baseline emotional state (VAD model)
  baseline: {
    valence: number;                        // -1 (negative) to +1 (positive)
    arousal: number;                        // 0 (calm) to 1 (excited)
    dominance: number;                      // 0 (submissive) to 1 (dominant)
  };

  // Normal variance
  variance: {
    valence_variance: number;
    arousal_variance: number;
    dominance_variance: number;
  };

  // Contextual signatures
  contexts: {
    work_mode: EmotionalState;
    casual_mode: EmotionalState;
    stress_mode: EmotionalState;
  };

  // Emotion markers
  markers: {
    joy_markers: string[];                  // "haha", ":)", etc.
    fear_markers: string[];                 // "worried", "scared", etc.
    anger_markers: string[];                // "annoyed", "frustrated", etc.
    sadness_markers: string[];              // "sad", "disappointed", etc.
  };

  samples_analyzed: number;
  confidence: number;
}

export interface EmotionalState {
  valence: number;      // -1 to +1
  arousal: number;      // 0 to 1
  dominance: number;    // 0 to 1
  timestamp: number;
}

// ============================================================================
// TEMPORAL PATTERNS
// ============================================================================

export interface TemporalProfile {
  user_id: string;
  created_at: number;
  last_updated: number;

  // Hourly patterns
  hourly: {
    typical_hours: Set<number>;             // 0-23 (hours of day)
    hour_distribution: Map<number, number>; // How often at each hour
  };

  // Daily patterns
  daily: {
    typical_days: Set<number>;              // 0-6 (Sunday-Saturday)
    day_distribution: Map<number, number>;
  };

  // Session patterns
  sessions: {
    session_duration_avg: number;           // Minutes
    session_duration_variance: number;
    interactions_per_day_avg: number;
    interactions_per_week_avg: number;
  };

  // Offline patterns
  offline: {
    typical_offline_periods: TimePeriod[];
    longest_offline_duration: number;       // Minutes
  };

  // Timezone
  timezone: string;

  samples_analyzed: number;
  confidence: number;
}

export interface TimePeriod {
  start_hour: number;
  end_hour: number;
  confidence: number;
}

// ============================================================================
// COMBINED USER PROFILES
// ============================================================================

export interface UserSecurityProfiles {
  user_id: string;
  linguistic: LinguisticProfile;
  typing: TypingProfile;
  emotional: EmotionalProfile;
  temporal: TemporalProfile;

  // Overall confidence (min of all profiles)
  overall_confidence: number;

  // Last interaction
  last_interaction: number;
}

// ============================================================================
// INTERACTION & ANALYSIS
// ============================================================================

export interface Interaction {
  interaction_id: string;
  user_id: string;
  timestamp: number;

  // Content
  text: string;
  text_length: number;
  word_count: number;

  // Timing (if available)
  typing_data?: {
    keystroke_intervals: number[];
    total_typing_time: number;
    pauses: number[];
    backspaces: number;
    corrections: number;
  };

  // Context
  session_id: string;
  device_type?: 'mobile' | 'desktop' | 'tablet';
  operation_type?: 'query' | 'command' | 'transfer' | 'delete' | 'export';

  // Metadata
  ip_address?: string;
  location?: string;
}

// ============================================================================
// ANOMALY DETECTION
// ============================================================================

export interface AnomalyScore {
  score: number;                            // 0-1 (1 = very anomalous)
  threshold: number;                        // Alert threshold
  alert: boolean;                           // true if score > threshold
  confidence: number;                       // 0-1

  // Breakdown
  components: {
    linguistic?: number;
    typing?: number;
    emotional?: number;
    temporal?: number;
  };

  // Details
  anomalies_detected: string[];
  reason: string;
}

export interface LinguisticAnomaly {
  score: number;
  threshold: number;
  alert: boolean;

  details: {
    vocabulary_deviation: number;
    syntax_deviation: number;
    semantics_deviation: number;
    sentiment_deviation: number;
  };

  specific_anomalies: string[];
}

export interface TypingAnomaly {
  score: number;
  threshold: number;
  alert: boolean;

  details: {
    speed_deviation: number;              // Faster/slower than usual
    error_rate_change: number;            // More/fewer errors
    pause_pattern_change: number;
    input_burst: boolean;                 // Sudden paste
  };

  specific_anomalies: string[];
}

export interface EmotionalAnomaly {
  score: number;
  threshold: number;
  alert: boolean;

  details: {
    valence_deviation: number;
    arousal_deviation: number;
    dominance_deviation: number;
  };

  specific_anomalies: string[];
}

export interface TemporalAnomaly {
  score: number;
  threshold: number;
  alert: boolean;

  details: {
    unusual_hour: boolean;
    unusual_day: boolean;
    unusual_duration: boolean;
    unusual_frequency: boolean;
  };

  specific_anomalies: string[];
}

// ============================================================================
// DURESS & COERCION DETECTION
// ============================================================================

export interface DuressScore {
  score: number;                            // 0-1
  threshold: number;
  alert: boolean;
  confidence: number;

  // Signal breakdown
  signals: {
    linguistic_anomaly: number;
    typing_anomaly: number;
    emotional_anomaly: number;
    temporal_anomaly: number;
    panic_code_detected: boolean;
  };

  // Recommendation
  recommendation: 'allow' | 'challenge' | 'delay' | 'block';
  reason: string;
}

export interface CoercionScore {
  score: number;
  threshold: number;
  alert: boolean;
  confidence: number;

  // Indicators
  indicators: {
    compliance_language: boolean;         // "ok", "sure", "whatever"
    passive_voice_excessive: boolean;
    hedging_excessive: boolean;           // "maybe", "I guess"
    fear_markers: boolean;                // "scared", "afraid"
    submission_markers: boolean;          // "have to", "must"
    rushed_responses: boolean;
    unusual_requests: boolean;
  };

  recommendation: 'allow' | 'challenge' | 'delay' | 'block';
  reason: string;
}

// ============================================================================
// SECURITY CONTEXT
// ============================================================================

export interface SecurityContext {
  user_id: string;
  interaction_id: string;
  timestamp: number;

  // Profiles
  profiles: UserSecurityProfiles;

  // Current state
  duress_score: DuressScore;
  coercion_score: CoercionScore;

  // Operation context
  operation_type?: string;
  is_sensitive_operation: boolean;
  operation_value?: number;                // For transfers, etc.

  // Decision
  decision: 'allow' | 'challenge' | 'delay' | 'block';
  decision_reason: string;
}

// ============================================================================
// STATISTICS & COMPARISON
// ============================================================================

export interface ProfileStatistics {
  mean: number;
  median: number;
  std_deviation: number;
  variance: number;
  min: number;
  max: number;
}

export interface ComparisonResult {
  similarity_score: number;                 // 0-1 (1 = identical)
  difference_score: number;                 // 0-1 (1 = completely different)
  significant_differences: string[];
}

// ============================================================================
// EXPORTS
// ============================================================================

export type ProfileType = 'linguistic' | 'typing' | 'emotional' | 'temporal';

export interface ProfileUpdate {
  profile_type: ProfileType;
  user_id: string;
  interaction: Interaction;
  timestamp: number;
}
