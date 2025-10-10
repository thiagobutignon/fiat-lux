/**
 * VERMELHO Adapter - Bridge between VERMELHO and AMARELO Dashboard
 *
 * This adapter provides a clean interface between:
 * - VERMELHO security system (behavioral biometrics)
 * - AMARELO dashboard (DevTools interface)
 *
 * Architecture:
 * AMARELO Dashboard → vermelho-adapter.ts → VERMELHO Core (src/grammar-lang/security/)
 *
 * This file handles:
 * - Type conversions
 * - Data formatting
 * - Error handling
 * - Caching (optional)
 */

import {
  CognitiveBehaviorGuard,
  CognitiveBehaviorAnalysis,
  formatCognitiveBehaviorAnalysis,
} from '../../../src/grammar-lang/security/cognitive-behavior-guard';
import {
  createCommitRequest,
  GitOperationRequest,
} from '../../../src/grammar-lang/security/git-operation-guard';
import { SecurityStorage } from '../../../src/grammar-lang/security/security-storage';
import {
  UserSecurityProfiles,
  DuressScore,
  CoercionScore,
  EmotionalState as VermelhoEmotionalState,
  Interaction,
} from '../../../src/grammar-lang/security/types';

// ============================================================================
// AMARELO Types (from security.ts)
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
    valence: number;
    arousal: number;
    dominance: number;
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
  valence: number;
  arousal: number;
  dominance: number;
  confidence: number;
}

// ============================================================================
// Adapter Class
// ============================================================================

export class VermelhoAdapter {
  private guard: CognitiveBehaviorGuard;
  private storage: SecurityStorage;

  constructor(storagePath: string = './vermelho-storage') {
    this.storage = new SecurityStorage(storagePath);
    this.guard = new CognitiveBehaviorGuard(this.storage);
  }

  // ==========================================================================
  // Duress Analysis
  // ==========================================================================

  /**
   * Analyze text for duress indicators
   *
   * Converts VERMELHO DuressScore → AMARELO DuressAnalysis
   */
  async analyzeDuress(
    text: string,
    userId: string,
    profiles: UserSecurityProfiles
  ): Promise<DuressAnalysis> {
    try {
      // Create a commit request to analyze the text
      const request = createCommitRequest(
        userId,
        'analysis',
        text,
        'human',
        { lines_added: 0, lines_removed: 0 }
      );

      // Get dual-layer security analysis
      const result = await this.guard.validateGitOperation(request, profiles);

      if (!result.cognitive_analysis) {
        throw new Error('No cognitive analysis returned');
      }

      const analysis = result.cognitive_analysis;

      // Map threat level to severity
      const severityMap: Record<string, DuressAnalysis['severity']> = {
        'none': 'none',
        'low': 'low',
        'medium': 'medium',
        'high': 'high',
        'critical': 'critical',
      };

      return {
        is_duress: analysis.behavioral.duress_score > 0.5,
        confidence: analysis.behavioral.confidence,
        indicators: analysis.behavioral.anomalies_detected,
        severity: severityMap[analysis.combined.threat_level] || 'none',
        recommended_action: result.decision,
      };
    } catch (error) {
      console.error('[VermelhoAdapter] analyzeDuress error:', error);
      throw error;
    }
  }

  /**
   * Analyze query for duress in real-time
   */
  async analyzeQueryDuress(
    query: string,
    userId: string,
    organismId: string,
    profiles: UserSecurityProfiles
  ): Promise<DuressAnalysis> {
    // Same implementation as analyzeDuress, but with organism context
    return this.analyzeDuress(query, userId, profiles);
  }

  // ==========================================================================
  // Behavioral Profiling
  // ==========================================================================

  /**
   * Get behavioral profile for a user
   *
   * Converts VERMELHO UserSecurityProfiles → AMARELO BehavioralProfile
   */
  async getBehavioralProfile(userId: string): Promise<BehavioralProfile> {
    try {
      // Try to get profile from storage
      const profiles = this.storage.getUserProfile(userId);

      if (!profiles) {
        throw new Error(`No profile found for user ${userId}`);
      }

      // Convert VERMELHO types → AMARELO types
      return {
        user_id: userId,
        linguistic_signature: {
          vocabulary_size: profiles.linguistic.vocabulary?.unique_words?.size || 0,
          avg_sentence_length: profiles.linguistic.syntax?.average_sentence_length || 0,
          formality_score: profiles.linguistic.semantics?.formality_level || 0.5,
          common_phrases: Array.from(
            profiles.linguistic.vocabulary?.distribution?.keys() || []
          ).slice(0, 10),
        },
        typing_patterns: {
          avg_wpm: profiles.typing.timing?.keystroke_interval_avg
            ? 60000 / (profiles.typing.timing.keystroke_interval_avg * 5)
            : 0,
          keystroke_rhythm: profiles.typing.timing?.keystroke_intervals?.slice(0, 10) || [],
          error_rate: profiles.typing.errors?.typo_rate || 0,
          pause_patterns: [
            profiles.typing.timing?.word_pause_duration || 0,
            profiles.typing.timing?.thinking_pause_duration || 0,
            profiles.typing.timing?.sentence_pause_duration || 0,
          ],
        },
        emotional_signature: {
          valence: profiles.emotional.baseline?.valence || 0,
          arousal: profiles.emotional.baseline?.arousal || 0,
          dominance: profiles.emotional.baseline?.dominance || 0,
        },
        temporal_patterns: {
          active_hours: Array.from(profiles.temporal.hourly?.typical_hours || []),
          session_durations: [profiles.temporal.sessions?.session_duration_avg || 0],
          query_frequency: profiles.temporal.sessions?.interactions_per_day_avg || 0,
        },
        baseline_established: profiles.overall_confidence > 0.7,
        last_updated: new Date(profiles.last_interaction).toISOString(),
      };
    } catch (error) {
      console.error('[VermelhoAdapter] getBehavioralProfile error:', error);
      throw error;
    }
  }

  /**
   * Update behavioral profile with new data
   */
  async updateBehavioralProfile(
    userId: string,
    interaction: Interaction
  ): Promise<void> {
    try {
      // Store interaction in VERMELHO storage
      this.storage.storeInteraction(userId, interaction);
    } catch (error) {
      console.error('[VermelhoAdapter] updateBehavioralProfile error:', error);
      throw error;
    }
  }

  // ==========================================================================
  // Linguistic Fingerprinting
  // ==========================================================================

  /**
   * Analyze linguistic fingerprint
   */
  async analyzeLinguisticFingerprint(
    text: string,
    userId: string,
    profiles: UserSecurityProfiles
  ): Promise<{ match: boolean; confidence: number; deviations: string[] }> {
    try {
      const request = createCommitRequest(
        userId,
        'analysis',
        text,
        'human',
        { lines_added: 0, lines_removed: 0 }
      );

      const result = await this.guard.validateGitOperation(request, profiles);

      if (!result.cognitive_analysis) {
        throw new Error('No cognitive analysis returned');
      }

      const analysis = result.cognitive_analysis;
      const linguisticAnomalies = analysis.behavioral.anomalies_detected.filter(
        (a) => a.includes('linguistic') || a.includes('vocabulary') || a.includes('syntax')
      );

      return {
        match: linguisticAnomalies.length === 0,
        confidence: analysis.behavioral.confidence,
        deviations: linguisticAnomalies,
      };
    } catch (error) {
      console.error('[VermelhoAdapter] analyzeLinguisticFingerprint error:', error);
      throw error;
    }
  }

  // ==========================================================================
  // Typing Pattern Analysis
  // ==========================================================================

  /**
   * Analyze typing patterns
   */
  async analyzeTypingPatterns(
    patterns: Array<{ timestamp: number; key: string; duration: number; interval: number }>,
    userId: string,
    profiles: UserSecurityProfiles
  ): Promise<{ match: boolean; confidence: number }> {
    try {
      const typing = profiles.typing;

      if (!typing || typing.confidence < 0.3) {
        // Not enough data for analysis
        return {
          match: true,
          confidence: 0.5,
        };
      }

      // Calculate average interval from patterns
      const avgInterval = patterns.reduce((sum, p) => sum + p.interval, 0) / patterns.length;
      const expectedInterval = typing.timing?.keystroke_interval_avg || 0;

      // Calculate deviation
      const deviation = Math.abs(avgInterval - expectedInterval) / Math.max(expectedInterval, 1);

      return {
        match: deviation < 0.3, // Within 30% of baseline
        confidence: typing.confidence,
      };
    } catch (error) {
      console.error('[VermelhoAdapter] analyzeTypingPatterns error:', error);
      throw error;
    }
  }

  // ==========================================================================
  // Emotional Analysis (VAD Model)
  // ==========================================================================

  /**
   * Analyze emotional state from text
   */
  async analyzeEmotionalState(
    text: string,
    profiles: UserSecurityProfiles
  ): Promise<EmotionalState> {
    try {
      // Use emotional profile baseline
      const emotional = profiles.emotional;

      return {
        valence: emotional.baseline?.valence || 0,
        arousal: emotional.baseline?.arousal || 0,
        dominance: emotional.baseline?.dominance || 0,
        confidence: emotional.confidence || 0.5,
      };
    } catch (error) {
      console.error('[VermelhoAdapter] analyzeEmotionalState error:', error);
      throw error;
    }
  }

  /**
   * Compare emotional state with user baseline
   */
  async compareEmotionalState(
    userId: string,
    emotionalState: EmotionalState,
    profiles: UserSecurityProfiles
  ): Promise<{ deviation: number; alert: boolean }> {
    try {
      const baseline = profiles.emotional.baseline;

      if (!baseline) {
        throw new Error('No emotional baseline found');
      }

      // Calculate Euclidean distance in VAD space
      const valenceDiff = emotionalState.valence - baseline.valence;
      const arousalDiff = emotionalState.arousal - baseline.arousal;
      const dominanceDiff = emotionalState.dominance - baseline.dominance;

      const deviation = Math.sqrt(
        valenceDiff ** 2 + arousalDiff ** 2 + dominanceDiff ** 2
      ) / Math.sqrt(3); // Normalize to 0-1

      return {
        deviation,
        alert: deviation > 0.5, // Alert if deviation > 50%
      };
    } catch (error) {
      console.error('[VermelhoAdapter] compareEmotionalState error:', error);
      throw error;
    }
  }

  // ==========================================================================
  // Temporal Pattern Analysis
  // ==========================================================================

  /**
   * Analyze temporal patterns
   */
  async analyzeTemporalPattern(
    userId: string,
    timestamp: number,
    profiles: UserSecurityProfiles
  ): Promise<{ anomaly: boolean; confidence: number }> {
    try {
      const date = new Date(timestamp);
      const hour = date.getHours();
      const day = date.getDay();

      const temporal = profiles.temporal;

      // Check if hour is typical
      const typicalHour = temporal.hourly?.typical_hours?.has(hour) ?? false;

      // Check if day is typical
      const typicalDay = temporal.daily?.typical_days?.has(day) ?? false;

      const anomaly = !typicalHour || !typicalDay;

      return {
        anomaly,
        confidence: temporal.confidence || 0.5,
      };
    } catch (error) {
      console.error('[VermelhoAdapter] analyzeTemporalPattern error:', error);
      throw error;
    }
  }

  // ==========================================================================
  // Comprehensive Analysis
  // ==========================================================================

  /**
   * Comprehensive security analysis combining all signals
   */
  async comprehensiveSecurityAnalysis(params: {
    text: string;
    userId: string;
    profiles: UserSecurityProfiles;
    timestamp?: number;
    organismId?: string;
  }): Promise<{
    safe: boolean;
    confidence: number;
    alerts: string[];
    recommended_action: string;
  }> {
    try {
      const request = createCommitRequest(
        params.userId,
        'analysis',
        params.text,
        'human',
        { lines_added: 0, lines_removed: 0 }
      );

      const result = await this.guard.validateGitOperation(
        request,
        params.profiles
      );

      if (!result.cognitive_analysis) {
        throw new Error('No cognitive analysis returned');
      }

      const analysis = result.cognitive_analysis;

      return {
        safe: result.decision === 'allow',
        confidence: Math.min(
          analysis.behavioral.confidence,
          analysis.cognitive.manipulation_detected ? 0.5 : 1.0
        ),
        alerts: [
          ...analysis.behavioral.anomalies_detected,
          ...(analysis.cognitive.manipulation_detected
            ? [`Manipulation detected: ${analysis.cognitive.techniques_found.length} techniques`]
            : []),
        ],
        recommended_action: result.decision,
      };
    } catch (error) {
      console.error('[VermelhoAdapter] comprehensiveSecurityAnalysis error:', error);
      throw error;
    }
  }

  // ==========================================================================
  // Health & Status
  // ==========================================================================

  /**
   * Check if VERMELHO is available
   */
  isAvailable(): boolean {
    try {
      return this.storage !== null && this.guard !== null;
    } catch {
      return false;
    }
  }

  /**
   * Get VERMELHO health status
   */
  async getHealth(): Promise<{ status: string; version: string }> {
    try {
      const available = this.isAvailable();

      return {
        status: available ? 'healthy' : 'unavailable',
        version: '1.0.0',
      };
    } catch (error) {
      return {
        status: 'error',
        version: 'unknown',
      };
    }
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let adapterInstance: VermelhoAdapter | null = null;

export function getVermelhoAdapter(storagePath?: string): VermelhoAdapter {
  if (!adapterInstance) {
    adapterInstance = new VermelhoAdapter(storagePath);
  }
  return adapterInstance;
}

export default VermelhoAdapter;
