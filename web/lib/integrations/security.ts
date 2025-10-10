/**
 * VERMELHO Integration - Security & Behavioral Analysis
 *
 * This module provides integration with the VERMELHO node (Security/Behavioral).
 * It handles:
 * - Duress detection
 * - Behavioral profiling
 * - Linguistic fingerprinting
 * - Typing pattern analysis
 * - Emotional signature analysis (VAD model)
 * - Temporal pattern analysis
 * - Multi-factor cognitive authentication
 *
 * STATUS: ✅ ACTIVE - Connected to VERMELHO via adapter
 * Architecture: AMARELO → security.ts → vermelho-adapter.ts → VERMELHO Core
 */

// ============================================================================
// Configuration
// ============================================================================

const VERMELHO_ENABLED = true; // ✅ VERMELHO integration active
const VERMELHO_API_URL = process.env.VERMELHO_API_URL || 'http://localhost:3003';

// ============================================================================
// Adapter Import
// ============================================================================

import { getVermelhoAdapter } from './vermelho-adapter';
import type { UserSecurityProfiles, Interaction } from '../../../src/grammar-lang/security/types';

// ============================================================================
// Types
// ============================================================================

export interface DuressAnalysis {
  is_duress: boolean;
  confidence: number;
  indicators: string[];
  severity: 'none' | 'low' | 'medium' | 'high' | 'critical';
  recommended_action: string;
}

export interface BehavioralProfile {
  user_id: string;
  linguistic_signature: {
    vocabulary_size: number;
    avg_sentence_length: number;
    formality_score: number;
    common_phrases: string[];
  };
  typing_patterns: {
    avg_wpm: number;
    keystroke_rhythm: number[];
    error_rate: number;
    pause_patterns: number[];
  };
  emotional_signature: {
    valence: number; // -1 to 1
    arousal: number; // -1 to 1
    dominance: number; // -1 to 1
  };
  temporal_patterns: {
    active_hours: number[];
    session_durations: number[];
    query_frequency: number;
  };
  baseline_established: boolean;
  last_updated: string;
}

export interface EmotionalState {
  valence: number; // Positive (1) to Negative (-1)
  arousal: number; // Excited (1) to Calm (-1)
  dominance: number; // Controlling (1) to Submissive (-1)
  confidence: number;
}

export interface TypingPattern {
  timestamp: number;
  key: string;
  duration: number; // Time key was held
  interval: number; // Time since previous key
}

// ============================================================================
// Duress Detection
// ============================================================================

/**
 * Analyze text for duress indicators
 *
 * @param text - Text to analyze
 * @param userId - User ID for behavioral comparison
 * @param profiles - User security profiles (optional, will fetch if not provided)
 * @returns Promise<DuressAnalysis>
 *
 * INTEGRATION: ✅ Connected to VERMELHO via adapter
 */
export async function analyzeDuress(
  text: string,
  userId: string,
  profiles?: UserSecurityProfiles
): Promise<DuressAnalysis> {
  if (!VERMELHO_ENABLED) {
    console.log('[STUB] analyzeDuress called:', { text, userId });

    return {
      is_duress: false,
      confidence: 0.85,
      indicators: [],
      severity: 'none',
      recommended_action: 'continue',
    };
  }

  try {
    const adapter = getVermelhoAdapter();

    // If profiles not provided, try to fetch from storage
    if (!profiles) {
      profiles = await getBehavioralProfileInternal(userId);
    }

    return await adapter.analyzeDuress(text, userId, profiles);
  } catch (error) {
    console.error('[VERMELHO] analyzeDuress error:', error);

    // Fail-open: return safe result on error
    return {
      is_duress: false,
      confidence: 0,
      indicators: ['Error during analysis'],
      severity: 'none',
      recommended_action: 'continue',
    };
  }
}

/**
 * Internal helper to get behavioral profile with proper type conversion
 */
async function getBehavioralProfileInternal(userId: string): Promise<UserSecurityProfiles> {
  const adapter = getVermelhoAdapter();
  const profile = await adapter.getBehavioralProfile(userId);

  // Convert AMARELO BehavioralProfile → VERMELHO UserSecurityProfiles
  // This is a simplified conversion - in production, you'd fetch the full profile
  return {
    user_id: userId,
    linguistic: {
      user_id: userId,
      created_at: Date.now(),
      last_updated: Date.now(),
      vocabulary: {
        distribution: new Map(),
        unique_words: new Set(profile.linguistic_signature.common_phrases),
        average_word_length: 0,
        rare_words_frequency: 0,
      },
      syntax: {
        average_sentence_length: profile.linguistic_signature.avg_sentence_length,
        sentence_length_variance: 0,
        punctuation_patterns: new Map(),
        grammar_preferences: [],
        passive_voice_frequency: 0,
        question_frequency: 0,
      },
      semantics: {
        topic_distribution: new Map(),
        sentiment_baseline: 0,
        sentiment_variance: 0,
        formality_level: profile.linguistic_signature.formality_score,
        hedging_frequency: 0,
      },
      samples_analyzed: 0,
      confidence: 0.5,
    },
    typing: {
      user_id: userId,
      created_at: Date.now(),
      last_updated: Date.now(),
      timing: {
        keystroke_intervals: profile.typing_patterns.keystroke_rhythm,
        keystroke_interval_avg: 0,
        keystroke_interval_variance: 0,
        word_pause_duration: profile.typing_patterns.pause_patterns[0] || 0,
        thinking_pause_duration: profile.typing_patterns.pause_patterns[1] || 0,
        sentence_pause_duration: profile.typing_patterns.pause_patterns[2] || 0,
      },
      errors: {
        typo_rate: profile.typing_patterns.error_rate,
        correction_patterns: [],
        backspace_frequency: 0,
        delete_frequency: 0,
        common_typos: new Map(),
      },
      input: {
        copy_paste_frequency: 0,
        input_burst_detected: false,
        edit_distance_avg: 0,
        session_length_avg: 0,
      },
      device: {
        keyboard_layout: 'US',
        typical_device_type: 'desktop',
      },
      samples_analyzed: 0,
      confidence: 0.5,
    },
    emotional: {
      user_id: userId,
      created_at: Date.now(),
      last_updated: Date.now(),
      baseline: profile.emotional_signature,
      variance: {
        valence_variance: 0,
        arousal_variance: 0,
        dominance_variance: 0,
      },
      contexts: {
        work_mode: { ...profile.emotional_signature, timestamp: Date.now() },
        casual_mode: { ...profile.emotional_signature, timestamp: Date.now() },
        stress_mode: { ...profile.emotional_signature, timestamp: Date.now() },
      },
      markers: {
        joy_markers: [],
        fear_markers: [],
        anger_markers: [],
        sadness_markers: [],
      },
      samples_analyzed: 0,
      confidence: 0.5,
    },
    temporal: {
      user_id: userId,
      created_at: Date.now(),
      last_updated: Date.now(),
      hourly: {
        typical_hours: new Set(profile.temporal_patterns.active_hours),
        hour_distribution: new Map(),
      },
      daily: {
        typical_days: new Set([1, 2, 3, 4, 5]),
        day_distribution: new Map(),
      },
      sessions: {
        session_duration_avg: profile.temporal_patterns.session_durations[0] || 0,
        session_duration_variance: 0,
        interactions_per_day_avg: profile.temporal_patterns.query_frequency,
        interactions_per_week_avg: profile.temporal_patterns.query_frequency * 7,
      },
      offline: {
        typical_offline_periods: [],
        longest_offline_duration: 0,
      },
      timezone: 'UTC',
      samples_analyzed: 0,
      confidence: 0.5,
    },
    overall_confidence: profile.baseline_established ? 0.8 : 0.3,
    last_interaction: new Date(profile.last_updated).getTime(),
  };
}

/**
 * Analyze query for duress in real-time
 *
 * @param query - Query text
 * @param userId - User ID
 * @param organismId - Organism ID
 * @param profiles - User security profiles (optional)
 * @returns Promise<DuressAnalysis>
 *
 * INTEGRATION: ✅ Connected to VERMELHO via adapter
 */
export async function analyzeQueryDuress(
  query: string,
  userId: string,
  organismId: string,
  profiles?: UserSecurityProfiles
): Promise<DuressAnalysis> {
  if (!VERMELHO_ENABLED) {
    console.log('[STUB] analyzeQueryDuress called:', { query, userId, organismId });

    return {
      is_duress: false,
      confidence: 0.90,
      indicators: [],
      severity: 'none',
      recommended_action: 'continue',
    };
  }

  try {
    const adapter = getVermelhoAdapter();

    if (!profiles) {
      profiles = await getBehavioralProfileInternal(userId);
    }

    return await adapter.analyzeQueryDuress(query, userId, organismId, profiles);
  } catch (error) {
    console.error('[VERMELHO] analyzeQueryDuress error:', error);

    // Fail-open
    return {
      is_duress: false,
      confidence: 0,
      indicators: ['Error during analysis'],
      severity: 'none',
      recommended_action: 'continue',
    };
  }
}

// ============================================================================
// Behavioral Profiling
// ============================================================================

/**
 * Get behavioral profile for a user
 *
 * @param userId - User ID
 * @returns Promise<BehavioralProfile>
 *
 * INTEGRATION: ✅ Connected to VERMELHO via adapter
 */
export async function getBehavioralProfile(userId: string): Promise<BehavioralProfile> {
  if (!VERMELHO_ENABLED) {
    console.log('[STUB] getBehavioralProfile called for user:', userId);

    return {
      user_id: userId,
      linguistic_signature: {
        vocabulary_size: 2500,
        avg_sentence_length: 15.3,
        formality_score: 0.7,
        common_phrases: ['I think', 'Let me know', 'Thanks'],
      },
      typing_patterns: {
        avg_wpm: 65,
        keystroke_rhythm: [0.15, 0.18, 0.12],
        error_rate: 0.03,
        pause_patterns: [0.5, 1.2, 0.8],
      },
      emotional_signature: {
        valence: 0.6,
        arousal: 0.3,
        dominance: 0.5,
      },
      temporal_patterns: {
        active_hours: [9, 10, 11, 14, 15, 16],
        session_durations: [25, 30, 45],
        query_frequency: 12,
      },
      baseline_established: true,
      last_updated: new Date().toISOString(),
    };
  }

  try {
    const adapter = getVermelhoAdapter();
    return await adapter.getBehavioralProfile(userId);
  } catch (error) {
    console.error('[VERMELHO] getBehavioralProfile error:', error);

    // Fail-open with default profile
    return {
      user_id: userId,
      linguistic_signature: {
        vocabulary_size: 0,
        avg_sentence_length: 0,
        formality_score: 0.5,
        common_phrases: [],
      },
      typing_patterns: {
        avg_wpm: 0,
        keystroke_rhythm: [],
        error_rate: 0,
        pause_patterns: [],
      },
      emotional_signature: {
        valence: 0,
        arousal: 0,
        dominance: 0,
      },
      temporal_patterns: {
        active_hours: [],
        session_durations: [],
        query_frequency: 0,
      },
      baseline_established: false,
      last_updated: new Date().toISOString(),
    };
  }
}

/**
 * Update behavioral profile with new data
 *
 * @param userId - User ID
 * @param data - New behavioral data
 * @returns Promise<void>
 *
 * INTEGRATION POINT: Update user's behavioral profile
 * Expected VERMELHO API: securityClient.updateProfile(userId, data)
 */
export async function updateBehavioralProfile(userId: string, data: any): Promise<void> {
  if (!VERMELHO_ENABLED) {
    console.log('[STUB] updateBehavioralProfile called:', { userId, data });
    return;
  }

  try {
    const adapter = getVermelhoAdapter();

    // Convert data to Interaction format
    const interaction: Interaction = {
      timestamp: Date.now(),
      interaction_type: 'query',
      text_content: data.text || '',
      metadata: data,
    };

    await adapter.updateBehavioralProfile(userId, interaction);
  } catch (error) {
    console.error('[VERMELHO] updateBehavioralProfile error:', error);
    // Fail-silent for storage operations
  }
}

// ============================================================================
// Linguistic Fingerprinting
// ============================================================================

/**
 * Analyze linguistic fingerprint
 *
 * @param text - Text to analyze
 * @param userId - User ID for comparison
 * @returns Promise<{ match: boolean; confidence: number; deviations: string[] }>
 *
 * INTEGRATION POINT: Linguistic fingerprint analysis
 * Expected VERMELHO API: securityClient.analyzeLinguisticFingerprint({ text, userId })
 */
export async function analyzeLinguisticFingerprint(
  text: string,
  userId: string
): Promise<{ match: boolean; confidence: number; deviations: string[] }> {
  if (!VERMELHO_ENABLED) {
    console.log('[STUB] analyzeLinguisticFingerprint called:', { text, userId });

    return {
      match: true,
      confidence: 0.92,
      deviations: [],
    };
  }

  try {
    const adapter = getVermelhoAdapter();

    // Get user profiles
    const profiles = await getBehavioralProfileInternal(userId);

    return await adapter.analyzeLinguisticFingerprint(text, userId, profiles);
  } catch (error) {
    console.error('[VERMELHO] analyzeLinguisticFingerprint error:', error);

    // Fail-open with default
    return {
      match: true,
      confidence: 0,
      deviations: ['Error during analysis'],
    };
  }
}

// ============================================================================
// Typing Pattern Analysis
// ============================================================================

/**
 * Analyze typing patterns
 *
 * @param patterns - Array of typing pattern data
 * @param userId - User ID for comparison
 * @returns Promise<{ match: boolean; confidence: number }>
 *
 * INTEGRATION POINT: Typing pattern analysis
 * Expected VERMELHO API: securityClient.analyzeTypingPatterns({ patterns, userId })
 */
export async function analyzeTypingPatterns(
  patterns: TypingPattern[],
  userId: string
): Promise<{ match: boolean; confidence: number }> {
  if (!VERMELHO_ENABLED) {
    console.log('[STUB] analyzeTypingPatterns called:', { patternCount: patterns.length, userId });

    return {
      match: true,
      confidence: 0.88,
    };
  }

  try {
    const adapter = getVermelhoAdapter();

    // Get user profiles
    const profiles = await getBehavioralProfileInternal(userId);

    return await adapter.analyzeTypingPatterns(patterns, userId, profiles);
  } catch (error) {
    console.error('[VERMELHO] analyzeTypingPatterns error:', error);

    // Fail-open with default
    return {
      match: true,
      confidence: 0,
    };
  }
}

// ============================================================================
// Emotional Signature (VAD Model)
// ============================================================================

/**
 * Analyze emotional state from text
 *
 * @param text - Text to analyze
 * @returns Promise<EmotionalState>
 *
 * INTEGRATION POINT: VAD model emotional analysis
 * Expected VERMELHO API: securityClient.analyzeEmotion(text)
 */
export async function analyzeEmotionalState(text: string): Promise<EmotionalState> {
  if (!VERMELHO_ENABLED) {
    console.log('[STUB] analyzeEmotionalState called for text');

    return {
      valence: 0.5,
      arousal: 0.3,
      dominance: 0.4,
      confidence: 0.85,
    };
  }

  try {
    const adapter = getVermelhoAdapter();

    // Get a default user profile (or use system profile)
    // In production, you'd pass a userId to get their specific profile
    const profiles = await getBehavioralProfileInternal('system');

    return await adapter.analyzeEmotionalState(text, profiles);
  } catch (error) {
    console.error('[VERMELHO] analyzeEmotionalState error:', error);

    // Fail-open with neutral emotional state
    return {
      valence: 0,
      arousal: 0,
      dominance: 0,
      confidence: 0,
    };
  }
}

/**
 * Compare emotional state with user baseline
 *
 * @param userId - User ID
 * @param emotionalState - Current emotional state
 * @returns Promise<{ deviation: number; alert: boolean }>
 *
 * INTEGRATION POINT: Compare emotional state with baseline
 * Expected VERMELHO API: securityClient.compareEmotionalState(userId, emotionalState)
 */
export async function compareEmotionalState(
  userId: string,
  emotionalState: EmotionalState
): Promise<{ deviation: number; alert: boolean }> {
  if (!VERMELHO_ENABLED) {
    console.log('[STUB] compareEmotionalState called:', { userId, emotionalState });

    return {
      deviation: 0.15,
      alert: false,
    };
  }

  try {
    const adapter = getVermelhoAdapter();

    // Get user profiles
    const profiles = await getBehavioralProfileInternal(userId);

    return await adapter.compareEmotionalState(userId, emotionalState, profiles);
  } catch (error) {
    console.error('[VERMELHO] compareEmotionalState error:', error);

    // Fail-open with no alert
    return {
      deviation: 0,
      alert: false,
    };
  }
}

// ============================================================================
// Temporal Pattern Analysis
// ============================================================================

/**
 * Analyze temporal patterns
 *
 * @param userId - User ID
 * @param timestamp - Current timestamp
 * @returns Promise<{ anomaly: boolean; confidence: number }>
 *
 * INTEGRATION POINT: Temporal pattern anomaly detection
 * Expected VERMELHO API: securityClient.analyzeTemporalPattern(userId, timestamp)
 */
export async function analyzeTemporalPattern(
  userId: string,
  timestamp: number
): Promise<{ anomaly: boolean; confidence: number }> {
  if (!VERMELHO_ENABLED) {
    console.log('[STUB] analyzeTemporalPattern called:', { userId, timestamp });

    return {
      anomaly: false,
      confidence: 0.90,
    };
  }

  try {
    const adapter = getVermelhoAdapter();

    // Get user profiles
    const profiles = await getBehavioralProfileInternal(userId);

    return await adapter.analyzeTemporalPattern(userId, timestamp, profiles);
  } catch (error) {
    console.error('[VERMELHO] analyzeTemporalPattern error:', error);

    // Fail-open with no anomaly
    return {
      anomaly: false,
      confidence: 0,
    };
  }
}

// ============================================================================
// Multi-Signal Integration
// ============================================================================

/**
 * Comprehensive security analysis combining all signals
 *
 * @param params - Analysis parameters
 * @returns Promise<SecurityAnalysis>
 *
 * INTEGRATION: ✅ Connected to VERMELHO via adapter
 */
export async function comprehensiveSecurityAnalysis(params: {
  text: string;
  userId: string;
  typingPatterns?: TypingPattern[];
  timestamp?: number;
  organismId?: string;
  profiles?: UserSecurityProfiles;
}): Promise<{
  safe: boolean;
  confidence: number;
  alerts: string[];
  recommended_action: string;
}> {
  if (!VERMELHO_ENABLED) {
    console.log('[STUB] comprehensiveSecurityAnalysis called:', params);

    return {
      safe: true,
      confidence: 0.92,
      alerts: [],
      recommended_action: 'continue',
    };
  }

  try {
    const adapter = getVermelhoAdapter();

    let profiles = params.profiles;
    if (!profiles) {
      profiles = await getBehavioralProfileInternal(params.userId);
    }

    return await adapter.comprehensiveSecurityAnalysis({
      text: params.text,
      userId: params.userId,
      profiles,
      timestamp: params.timestamp,
      organismId: params.organismId,
    });
  } catch (error) {
    console.error('[VERMELHO] comprehensiveSecurityAnalysis error:', error);

    // Fail-open
    return {
      safe: true,
      confidence: 0,
      alerts: ['Error during analysis'],
      recommended_action: 'continue',
    };
  }
}

// ============================================================================
// Health & Status
// ============================================================================

/**
 * Check if VERMELHO integration is available
 *
 * @returns boolean
 *
 * INTEGRATION: ✅ Connected to VERMELHO via adapter
 */
export function isVermelhoAvailable(): boolean {
  if (!VERMELHO_ENABLED) {
    return false;
  }

  try {
    const adapter = getVermelhoAdapter();
    return adapter.isAvailable();
  } catch {
    return false;
  }
}

/**
 * Get VERMELHO health status
 *
 * @returns Promise<{ status: string; version: string }>
 *
 * INTEGRATION: ✅ Connected to VERMELHO via adapter
 */
export async function getVermelhoHealth(): Promise<{ status: string; version: string }> {
  if (!VERMELHO_ENABLED) {
    return { status: 'disabled', version: 'stub' };
  }

  try {
    const adapter = getVermelhoAdapter();
    return await adapter.getHealth();
  } catch (error) {
    console.error('[VERMELHO] getVermelhoHealth error:', error);
    return { status: 'error', version: 'unknown' };
  }
}

// ============================================================================
// Export Summary
// ============================================================================

export const SecurityIntegration = {
  // Duress Detection
  analyzeDuress,
  analyzeQueryDuress,

  // Behavioral Profiling
  getBehavioralProfile,
  updateBehavioralProfile,

  // Linguistic Fingerprinting
  analyzeLinguisticFingerprint,

  // Typing Patterns
  analyzeTypingPatterns,

  // Emotional Analysis
  analyzeEmotionalState,
  compareEmotionalState,

  // Temporal Patterns
  analyzeTemporalPattern,

  // Multi-Signal
  comprehensiveSecurityAnalysis,

  // Health
  isVermelhoAvailable,
  getVermelhoHealth,
};
