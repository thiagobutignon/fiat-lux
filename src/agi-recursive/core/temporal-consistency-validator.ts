/**
 * Temporal Consistency Validator
 *
 * Validates that system responses remain consistent over time.
 * Detects:
 * - Concept drift (gradual changes in understanding)
 * - Contradictions (sudden inconsistencies)
 * - Confidence decay (decreasing certainty over time)
 *
 * This completes the "Temporal Consistency Checking" innovation.
 */

import type { Episode } from './episodic-memory';
import type { AgentResponse } from './meta-agent';

// ============================================================================
// Types
// ============================================================================

export interface ConsistencyValidation {
  is_consistent: boolean;
  average_similarity: number;
  inconsistent_episodes: InconsistentEpisode[];
  temporal_drift: TemporalDrift;
  warning: string | null;
  confidence_adjustment: number; // Suggested adjustment to current confidence
}

export interface InconsistentEpisode {
  episode_id: string;
  timestamp: number;
  query: string;
  response: string;
  similarity_score: number;
  reason: string;
}

export interface TemporalDrift {
  concept: string;
  confidence_t0: number; // Initial confidence
  confidence_t1: number; // Current confidence
  drift_magnitude: number; // Absolute change
  drift_rate: number; // Change per day
  is_significant: boolean; // Drift > threshold
  trend: 'increasing' | 'decreasing' | 'stable';
}

export interface ConceptEvolution {
  concept: string;
  first_seen: number;
  last_seen: number;
  appearance_count: number;
  confidence_history: { timestamp: number; confidence: number }[];
  stability_score: number; // 0-1, higher = more stable
}

// ============================================================================
// Temporal Consistency Validator
// ============================================================================

export class TemporalConsistencyValidator {
  private conceptEvolutions: Map<string, ConceptEvolution> = new Map();
  private similarityThreshold: number = 0.7;
  private driftThreshold: number = 0.2; // 20% change is significant

  constructor(similarityThreshold?: number, driftThreshold?: number) {
    if (similarityThreshold) this.similarityThreshold = similarityThreshold;
    if (driftThreshold) this.driftThreshold = driftThreshold;
  }

  /**
   * Validate if current response is consistent with historical responses
   */
  async validateConsistency(
    currentQuery: string,
    currentResponse: AgentResponse,
    historicalEpisodes: Episode[]
  ): Promise<ConsistencyValidation> {
    // 1. Find similar historical queries
    const similar = this.findSimilarQueries(currentQuery, historicalEpisodes, 10);

    if (similar.length === 0) {
      // No history to compare against
      return {
        is_consistent: true,
        average_similarity: 1.0,
        inconsistent_episodes: [],
        temporal_drift: this.calculateDrift([], currentResponse),
        warning: null,
        confidence_adjustment: 0,
      };
    }

    // 2. Compute semantic similarity with historical responses
    const similarities = similar.map((ep) => ({
      episode: ep,
      similarity: this.computeSemanticSimilarity(currentResponse.answer, ep.response),
    }));

    // 3. Identify inconsistencies
    const inconsistent: InconsistentEpisode[] = similarities
      .filter((s) => s.similarity < this.similarityThreshold)
      .map((s) => ({
        episode_id: s.episode.id,
        timestamp: s.episode.timestamp,
        query: s.episode.query,
        response: s.episode.response,
        similarity_score: s.similarity,
        reason: `Response differs significantly (similarity: ${(s.similarity * 100).toFixed(1)}%)`,
      }));

    // 4. Calculate temporal drift
    const drift = this.calculateDrift(similar, currentResponse);

    // 5. Calculate average similarity
    const avg_similarity =
      similarities.reduce((sum, s) => sum + s.similarity, 0) / similarities.length;

    // 6. Generate warning if needed
    let warning: string | null = null;
    if (inconsistent.length > 0) {
      warning = `Response differs from ${inconsistent.length}/${similar.length} historical answers. Confidence may be overestimated.`;
    }

    if (drift.is_significant) {
      warning = (warning || '') + ` Significant drift detected: ${drift.trend} trend in confidence.`;
    }

    // 7. Suggest confidence adjustment
    const confidence_adjustment = this.calculateConfidenceAdjustment(
      avg_similarity,
      drift,
      inconsistent.length,
      similar.length
    );

    return {
      is_consistent: inconsistent.length === 0,
      average_similarity: avg_similarity,
      inconsistent_episodes: inconsistent,
      temporal_drift: drift,
      warning: warning?.trim() || null,
      confidence_adjustment,
    };
  }

  /**
   * Track concept evolution over time
   */
  trackConceptEvolution(concepts: string[], confidence: number, timestamp: number): void {
    for (const concept of concepts) {
      let evolution = this.conceptEvolutions.get(concept);

      if (!evolution) {
        evolution = {
          concept,
          first_seen: timestamp,
          last_seen: timestamp,
          appearance_count: 0,
          confidence_history: [],
          stability_score: 1.0,
        };
        this.conceptEvolutions.set(concept, evolution);
      }

      evolution.last_seen = timestamp;
      evolution.appearance_count++;
      evolution.confidence_history.push({ timestamp, confidence });

      // Recalculate stability (variance of confidence over time)
      const confidences = evolution.confidence_history.map((h) => h.confidence);
      const mean = confidences.reduce((sum, c) => sum + c, 0) / confidences.length;
      const variance =
        confidences.reduce((sum, c) => sum + Math.pow(c - mean, 2), 0) / confidences.length;
      evolution.stability_score = Math.max(0, 1 - variance);
    }
  }

  /**
   * Get concept evolution history
   */
  getConceptEvolution(concept: string): ConceptEvolution | null {
    return this.conceptEvolutions.get(concept) || null;
  }

  /**
   * Get all concepts with significant drift
   */
  getConceptsWithDrift(threshold: number = 0.2): ConceptEvolution[] {
    return Array.from(this.conceptEvolutions.values()).filter((evo) => {
      if (evo.confidence_history.length < 2) return false;

      const first = evo.confidence_history[0].confidence;
      const last = evo.confidence_history[evo.confidence_history.length - 1].confidence;

      return Math.abs(last - first) > threshold;
    });
  }

  /**
   * Detect sudden changes (anomalies) in confidence
   */
  detectAnomalies(concept: string, window_size: number = 5): boolean {
    const evolution = this.conceptEvolutions.get(concept);
    if (!evolution || evolution.confidence_history.length < window_size + 1) {
      return false;
    }

    const recent = evolution.confidence_history.slice(-window_size);
    const latest = evolution.confidence_history[evolution.confidence_history.length - 1];

    const avg_recent = recent.reduce((sum, h) => sum + h.confidence, 0) / recent.length;
    const deviation = Math.abs(latest.confidence - avg_recent);

    // Anomaly if deviation > 30%
    return deviation > 0.3;
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  private findSimilarQueries(
    query: string,
    episodes: Episode[],
    limit: number
  ): Episode[] {
    const query_lower = query.toLowerCase();
    const query_words = new Set(query_lower.split(/\s+/));

    const scored = episodes.map((episode) => {
      const episode_words = new Set(episode.query.toLowerCase().split(/\s+/));
      const intersection = new Set([...query_words].filter((w) => episode_words.has(w)));
      const union = new Set([...query_words, ...episode_words]);
      const similarity = intersection.size / union.size;

      return { episode, similarity };
    });

    return scored
      .filter((s) => s.similarity > 0.3) // At least 30% similarity
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, limit)
      .map((s) => s.episode);
  }

  private computeSemanticSimilarity(response1: string, response2: string): number {
    // Simple word-based similarity (in production, use embeddings)
    const words1 = new Set(response1.toLowerCase().split(/\s+/));
    const words2 = new Set(response2.toLowerCase().split(/\s+/));

    const intersection = new Set([...words1].filter((w) => words2.has(w)));
    const union = new Set([...words1, ...words2]);

    // Jaccard similarity
    const jaccard = intersection.size / union.size;

    // Bonus for similar length (same information density)
    const length_similarity = 1 - Math.abs(words1.size - words2.size) / Math.max(words1.size, words2.size);

    return jaccard * 0.7 + length_similarity * 0.3;
  }

  private calculateDrift(
    historical: Episode[],
    current: AgentResponse
  ): TemporalDrift {
    if (historical.length === 0) {
      return {
        concept: 'overall',
        confidence_t0: current.confidence,
        confidence_t1: current.confidence,
        drift_magnitude: 0,
        drift_rate: 0,
        is_significant: false,
        trend: 'stable',
      };
    }

    const oldest = historical[0];
    const historical_confidence =
      historical.reduce((sum, ep) => sum + ep.confidence, 0) / historical.length;

    const drift_magnitude = Math.abs(current.confidence - historical_confidence);

    // Calculate drift rate (per day)
    const time_span_ms = Date.now() - oldest.timestamp;
    const time_span_days = time_span_ms / (1000 * 60 * 60 * 24);
    const drift_rate = time_span_days > 0 ? drift_magnitude / time_span_days : 0;

    // Determine trend
    let trend: 'increasing' | 'decreasing' | 'stable' = 'stable';
    if (drift_magnitude > this.driftThreshold) {
      trend = current.confidence > historical_confidence ? 'increasing' : 'decreasing';
    }

    return {
      concept: 'overall',
      confidence_t0: historical_confidence,
      confidence_t1: current.confidence,
      drift_magnitude,
      drift_rate,
      is_significant: drift_magnitude > this.driftThreshold,
      trend,
    };
  }

  private calculateConfidenceAdjustment(
    avg_similarity: number,
    drift: TemporalDrift,
    inconsistent_count: number,
    total_count: number
  ): number {
    // Start with no adjustment
    let adjustment = 0;

    // Penalize low similarity
    if (avg_similarity < 0.7) {
      adjustment -= (0.7 - avg_similarity) * 0.5;
    }

    // Penalize significant drift
    if (drift.is_significant) {
      adjustment -= drift.drift_magnitude * 0.3;
    }

    // Penalize high inconsistency rate
    const inconsistency_rate = inconsistent_count / total_count;
    if (inconsistency_rate > 0.3) {
      adjustment -= (inconsistency_rate - 0.3) * 0.4;
    }

    // Clamp to [-0.3, 0] (can only decrease confidence, not increase)
    return Math.max(-0.3, Math.min(0, adjustment));
  }

  /**
   * Export evolution history for analysis
   */
  exportEvolutions(): ConceptEvolution[] {
    return Array.from(this.conceptEvolutions.values());
  }
}
