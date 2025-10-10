/**
 * CINZA Adapter - Bridge between CINZA and AMARELO Dashboard
 *
 * This adapter provides a clean interface between:
 * - CINZA cognitive manipulation detection system
 * - AMARELO dashboard (DevTools interface)
 *
 * Architecture:
 * AMARELO Dashboard → cinza-adapter.ts → CINZA Core (src/grammar-lang/cognitive/)
 *
 * This file handles:
 * - Type conversions
 * - Data formatting
 * - Error handling
 * - Caching (optional)
 */

import {
  detectManipulation,
  PatternMatchResult,
  PatternMatchConfig,
} from '../../../src/grammar-lang/cognitive/detector/pattern-matcher';
import {
  ManipulationTechnique,
  DetectionResult,
  DarkTetradScores,
  ConstitutionalValidation,
} from '../../../src/grammar-lang/cognitive/types';

// ============================================================================
// AMARELO Types (from cognitive.ts)
// ============================================================================

export interface ManipulationAnalysis {
  detected: boolean;
  confidence: number;
  techniques: ManipulationTechnique[];
  severity: 'none' | 'low' | 'medium' | 'high' | 'critical';
  dark_tetrad: {
    narcissism: number;
    machiavellianism: number;
    psychopathy: number;
    sadism: number;
    aggregate: number;
  };
  constitutional_violations: string[];
  recommended_action: string;
}

export interface IntentAnalysis {
  primary_intent: string;
  confidence: number;
  secondary_intents: Array<{ intent: string; confidence: number }>;
  context_awareness: number;
  power_dynamic: number;
}

export interface TechniqueDetail {
  id: string;
  name: string;
  description: string;
  category: string;
  severity: number;
  examples: string[];
  dark_tetrad_association: {
    narcissism: number;
    machiavellianism: number;
    psychopathy: number;
    sadism: number;
  };
}

export interface ManipulationHistory {
  user_id: string;
  total_detections: number;
  time_range: {
    start: number;
    end: number;
  };
  detections: Array<{
    timestamp: number;
    text: string;
    techniques: string[];
    confidence: number;
    severity: string;
  }>;
  patterns: {
    most_common_techniques: Array<{ technique: string; count: number }>;
    severity_distribution: Record<string, number>;
    time_distribution: Record<string, number>;
  };
}

// ============================================================================
// Adapter Class
// ============================================================================

export class CinzaAdapter {
  private detectionCache: Map<string, { result: PatternMatchResult; timestamp: number }>;
  private cacheTTL: number = 5 * 60 * 1000; // 5 minutes

  constructor() {
    this.detectionCache = new Map();
  }

  // ==========================================================================
  // Manipulation Detection
  // ==========================================================================

  /**
   * Detect manipulation in text
   *
   * Converts CINZA PatternMatchResult → AMARELO ManipulationAnalysis
   */
  async detectManipulation(
    text: string,
    config?: PatternMatchConfig
  ): Promise<ManipulationAnalysis> {
    try {
      // Check cache
      const cacheKey = `${text}-${JSON.stringify(config)}`;
      const cached = this.detectionCache.get(cacheKey);

      if (cached && Date.now() - cached.timestamp < this.cacheTTL) {
        return this.convertToManipulationAnalysis(cached.result);
      }

      // Call CINZA detection
      const result = await detectManipulation(text, {
        min_confidence: config?.min_confidence || 0.5,
        enable_neurodivergent_protection: config?.enable_neurodivergent_protection ?? true,
        ...config,
      });

      // Cache result
      this.detectionCache.set(cacheKey, { result, timestamp: Date.now() });

      return this.convertToManipulationAnalysis(result);
    } catch (error) {
      console.error('[CinzaAdapter] detectManipulation error:', error);
      throw error;
    }
  }

  /**
   * Convert CINZA result to AMARELO format
   */
  private convertToManipulationAnalysis(result: PatternMatchResult): ManipulationAnalysis {
    // Calculate severity based on confidence and dark tetrad
    const avgDarkTetrad =
      (result.dark_tetrad_aggregate.narcissism +
        result.dark_tetrad_aggregate.machiavellianism +
        result.dark_tetrad_aggregate.psychopathy +
        result.dark_tetrad_aggregate.sadism) /
      4;

    let severity: ManipulationAnalysis['severity'] = 'none';

    if (result.highest_confidence > 0.8 || avgDarkTetrad > 0.7) {
      severity = 'critical';
    } else if (result.highest_confidence > 0.6 || avgDarkTetrad > 0.5) {
      severity = 'high';
    } else if (result.highest_confidence > 0.4 || avgDarkTetrad > 0.3) {
      severity = 'medium';
    } else if (result.highest_confidence > 0.2) {
      severity = 'low';
    }

    // Determine recommended action
    let recommended_action = 'allow';

    if (severity === 'critical') {
      recommended_action = 'block';
    } else if (severity === 'high') {
      recommended_action = 'challenge';
    } else if (severity === 'medium') {
      recommended_action = 'review';
    }

    return {
      detected: result.total_matches > 0,
      confidence: result.highest_confidence,
      techniques: result.detections.map((d) => d.technique),
      severity,
      dark_tetrad: {
        narcissism: result.dark_tetrad_aggregate.narcissism,
        machiavellianism: result.dark_tetrad_aggregate.machiavellianism,
        psychopathy: result.dark_tetrad_aggregate.psychopathy,
        sadism: result.dark_tetrad_aggregate.sadism,
        aggregate: avgDarkTetrad,
      },
      constitutional_violations: result.constitutional_validation?.violations || [],
      recommended_action,
    };
  }

  // ==========================================================================
  // Intent Analysis
  // ==========================================================================

  /**
   * Analyze intent from text
   *
   * Uses CINZA pragmatics layer
   */
  async analyzeIntent(text: string): Promise<IntentAnalysis> {
    try {
      const result = await detectManipulation(text, {
        min_confidence: 0.3,
      });

      // Extract intent from pragmatics (if available in detections)
      const intents = result.detections
        .filter((d) => d.technique.pragmatics_layer)
        .map((d) => ({
          intent: d.technique.name,
          confidence: d.confidence,
        }))
        .sort((a, b) => b.confidence - a.confidence);

      const primaryIntent = intents[0] || { intent: 'neutral', confidence: 0.5 };
      const secondaryIntents = intents.slice(1, 4);

      return {
        primary_intent: primaryIntent.intent,
        confidence: primaryIntent.confidence,
        secondary_intents: secondaryIntents,
        context_awareness: result.attention_trace?.weights.pragmatics || 0.5,
        power_dynamic: this.calculatePowerDynamic(result),
      };
    } catch (error) {
      console.error('[CinzaAdapter] analyzeIntent error:', error);
      throw error;
    }
  }

  /**
   * Calculate power dynamic from detection result
   */
  private calculatePowerDynamic(result: PatternMatchResult): number {
    // Look for manipulation techniques that indicate power dynamics
    const powerTechniques = result.detections.filter(
      (d) =>
        d.technique.name.toLowerCase().includes('dominance') ||
        d.technique.name.toLowerCase().includes('control') ||
        d.technique.name.toLowerCase().includes('authority')
    );

    if (powerTechniques.length === 0) {
      return 0.5; // Neutral
    }

    return (
      powerTechniques.reduce((sum, t) => sum + t.confidence, 0) /
      powerTechniques.length
    );
  }

  // ==========================================================================
  // Dark Tetrad Analysis
  // ==========================================================================

  /**
   * Analyze Dark Tetrad personality traits
   */
  async analyzeDarkTetrad(
    text: string
  ): Promise<{ scores: DarkTetradScores; interpretation: string }> {
    try {
      const result = await detectManipulation(text, {
        min_confidence: 0.4,
      });

      const scores = result.dark_tetrad_aggregate;
      const avgScore =
        (scores.narcissism + scores.machiavellianism + scores.psychopathy + scores.sadism) / 4;

      let interpretation = 'Normal personality indicators';

      if (avgScore > 0.7) {
        interpretation = 'CRITICAL: Strong Dark Tetrad traits detected';
      } else if (avgScore > 0.5) {
        interpretation = 'WARNING: Elevated Dark Tetrad traits';
      } else if (avgScore > 0.3) {
        interpretation = 'CAUTION: Some Dark Tetrad indicators present';
      }

      return {
        scores,
        interpretation,
      };
    } catch (error) {
      console.error('[CinzaAdapter] analyzeDarkTetrad error:', error);
      throw error;
    }
  }

  // ==========================================================================
  // Constitutional Validation
  // ==========================================================================

  /**
   * Validate text against constitutional principles (Layer 2)
   */
  async validateConstitutional(text: string): Promise<ConstitutionalValidation> {
    try {
      const result = await detectManipulation(text, {
        min_confidence: 0.5,
      });

      return (
        result.constitutional_validation || {
          compliant: true,
          violations: [],
          principles_checked: [],
          layer: 2,
        }
      );
    } catch (error) {
      console.error('[CinzaAdapter] validateConstitutional error:', error);
      throw error;
    }
  }

  // ==========================================================================
  // Technique Catalog
  // ==========================================================================

  /**
   * Get details about a specific manipulation technique
   */
  async getTechniqueDetails(techniqueId: string): Promise<TechniqueDetail | null> {
    try {
      // In a real implementation, this would query the technique database
      // For now, we'll use detection to find it

      // This is a simplified version - in production, you'd have a technique registry
      return null;
    } catch (error) {
      console.error('[CinzaAdapter] getTechniqueDetails error:', error);
      throw error;
    }
  }

  /**
   * List all available manipulation techniques
   */
  async listTechniques(category?: string): Promise<TechniqueDetail[]> {
    try {
      // This would query the full technique catalog
      // For now, return empty array
      return [];
    } catch (error) {
      console.error('[CinzaAdapter] listTechniques error:', error);
      throw error;
    }
  }

  // ==========================================================================
  // History & Analytics (Stub - would integrate with storage)
  // ==========================================================================

  /**
   * Get manipulation detection history for a user
   */
  async getManipulationHistory(
    userId: string,
    timeRange?: { start: number; end: number }
  ): Promise<ManipulationHistory> {
    try {
      // This would query stored detection events
      // For now, return stub
      return {
        user_id: userId,
        total_detections: 0,
        time_range: timeRange || {
          start: Date.now() - 7 * 24 * 60 * 60 * 1000,
          end: Date.now(),
        },
        detections: [],
        patterns: {
          most_common_techniques: [],
          severity_distribution: {},
          time_distribution: {},
        },
      };
    } catch (error) {
      console.error('[CinzaAdapter] getManipulationHistory error:', error);
      throw error;
    }
  }

  // ==========================================================================
  // Health & Status
  // ==========================================================================

  /**
   * Check if CINZA is available
   */
  isAvailable(): boolean {
    try {
      return typeof detectManipulation === 'function';
    } catch {
      return false;
    }
  }

  /**
   * Get CINZA health status
   */
  async getHealth(): Promise<{ status: string; version: string; techniques_loaded: number }> {
    try {
      const available = this.isAvailable();

      return {
        status: available ? 'healthy' : 'unavailable',
        version: '1.0.0',
        techniques_loaded: 180, // 152 GPT-4 era + 28 GPT-5 era
      };
    } catch (error) {
      return {
        status: 'error',
        version: 'unknown',
        techniques_loaded: 0,
      };
    }
  }

  /**
   * Clear detection cache
   */
  clearCache(): void {
    this.detectionCache.clear();
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let adapterInstance: CinzaAdapter | null = null;

export function getCinzaAdapter(): CinzaAdapter {
  if (!adapterInstance) {
    adapterInstance = new CinzaAdapter();
  }
  return adapterInstance;
}

export default CinzaAdapter;
