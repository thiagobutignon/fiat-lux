/**
 * Attention Tracker - Interpretability Layer for AGI System
 *
 * PURPOSE:
 * - Transform black box → glass box
 * - Enable debugging of reasoning chains
 * - Enable regulatory auditing
 * - Generate data for meta-learning
 *
 * WHAT IT TRACKS:
 * For each query, track EXACTLY which concepts from which slices
 * influenced each decision, with weights showing influence strength.
 *
 * USE CASES:
 * 1. Developer: "Why did the system give this answer?"
 * 2. Auditor: "Which knowledge influenced this decision?"
 * 3. Meta-learning: "Which concepts are most influential?"
 * 4. Debugging: "Which slice caused this error?"
 */

/**
 * Single attention trace - records one concept's influence
 */
export interface AttentionTrace {
  concept: string; // e.g., "dependency_inversion", "lazy_evaluation"
  slice: string; // e.g., "finance/budgeting.md", "biology/cells.md"
  weight: number; // 0-1, quanto influenciou (how much it influenced)
  reasoning: string; // WHY this concept was influential
  timestamp: number; // When this trace was recorded
}

/**
 * Attention for a complete query execution
 */
export interface QueryAttention {
  query_id: string; // Unique identifier for this query
  query: string; // The original query text
  timestamp: number; // When query was processed
  traces: AttentionTrace[]; // All attention traces for this query
  total_concepts: number; // Total concepts considered
  top_influencers: AttentionTrace[]; // Top 5 most influential
  decision_path: string[]; // Sequence of decisions made
}

/**
 * Statistics about attention patterns
 */
export interface AttentionStats {
  total_queries: number;
  total_traces: number;
  most_influential_concepts: Array<{
    concept: string;
    count: number;
    average_weight: number;
  }>;
  most_used_slices: Array<{
    slice: string;
    count: number;
    average_weight: number;
  }>;
  average_traces_per_query: number;
  high_confidence_patterns: Array<{
    concepts: string[];
    frequency: number;
  }>;
}

/**
 * AttentionTracker - Core implementation
 *
 * Tracks concept-to-decision influences throughout AGI execution
 */
export class AttentionTracker {
  private queryAttentions: Map<string, QueryAttention> = new Map();
  private currentQueryId: string | null = null;
  private currentTraces: AttentionTrace[] = [];

  /**
   * Start tracking a new query
   */
  startQuery(queryId: string, query: string): void {
    this.currentQueryId = queryId;
    this.currentTraces = [];

    const attention: QueryAttention = {
      query_id: queryId,
      query,
      timestamp: Date.now(),
      traces: [],
      total_concepts: 0,
      top_influencers: [],
      decision_path: [],
    };

    this.queryAttentions.set(queryId, attention);
  }

  /**
   * Add attention trace for current query
   *
   * @param concept - The concept that influenced decision
   * @param slice - The slice containing the concept
   * @param weight - Influence strength (0-1)
   * @param reasoning - WHY this concept influenced decision
   */
  addTrace(
    concept: string,
    slice: string,
    weight: number,
    reasoning: string
  ): void {
    if (!this.currentQueryId) {
      throw new Error('No active query. Call startQuery() first.');
    }

    // Validate weight
    if (weight < 0 || weight > 1) {
      throw new Error(`Invalid weight ${weight}. Must be between 0 and 1.`);
    }

    const trace: AttentionTrace = {
      concept,
      slice,
      weight,
      reasoning,
      timestamp: Date.now(),
    };

    this.currentTraces.push(trace);

    const attention = this.queryAttentions.get(this.currentQueryId);
    if (attention) {
      attention.traces.push(trace);
      attention.total_concepts++;
    }
  }

  /**
   * Add multiple traces at once (batch operation)
   */
  addTraces(traces: Omit<AttentionTrace, 'timestamp'>[]): void {
    for (const trace of traces) {
      this.addTrace(trace.concept, trace.slice, trace.weight, trace.reasoning);
    }
  }

  /**
   * Record a decision point in the reasoning chain
   */
  addDecisionPoint(decision: string): void {
    if (!this.currentQueryId) {
      return;
    }

    const attention = this.queryAttentions.get(this.currentQueryId);
    if (attention) {
      attention.decision_path.push(decision);
    }
  }

  /**
   * Finalize tracking for current query
   *
   * Computes top influencers and prepares for next query
   */
  endQuery(): QueryAttention | null {
    if (!this.currentQueryId) {
      return null;
    }

    const attention = this.queryAttentions.get(this.currentQueryId);
    if (!attention) {
      return null;
    }

    // Sort traces by weight and get top 5
    const sorted = [...attention.traces].sort((a, b) => b.weight - a.weight);
    attention.top_influencers = sorted.slice(0, 5);

    this.currentQueryId = null;
    this.currentTraces = [];

    return attention;
  }

  /**
   * Get attention traces for a specific query
   */
  getQueryAttention(queryId: string): QueryAttention | undefined {
    return this.queryAttentions.get(queryId);
  }

  /**
   * Get all query attentions
   */
  getAllAttentions(): QueryAttention[] {
    return Array.from(this.queryAttentions.values());
  }

  /**
   * Get current traces (for debugging)
   */
  getCurrentTraces(): AttentionTrace[] {
    return [...this.currentTraces];
  }

  /**
   * Analyze attention patterns across all queries
   */
  getStatistics(): AttentionStats {
    const allTraces = Array.from(this.queryAttentions.values()).flatMap(
      (q) => q.traces
    );

    // Count concept occurrences and weights
    const conceptMap = new Map<string, { count: number; totalWeight: number }>();
    const sliceMap = new Map<string, { count: number; totalWeight: number }>();

    for (const trace of allTraces) {
      // Concepts
      const conceptData = conceptMap.get(trace.concept) || {
        count: 0,
        totalWeight: 0,
      };
      conceptData.count++;
      conceptData.totalWeight += trace.weight;
      conceptMap.set(trace.concept, conceptData);

      // Slices
      const sliceData = sliceMap.get(trace.slice) || {
        count: 0,
        totalWeight: 0,
      };
      sliceData.count++;
      sliceData.totalWeight += trace.weight;
      sliceMap.set(trace.slice, sliceData);
    }

    // Sort and get top concepts
    const mostInfluentialConcepts = Array.from(conceptMap.entries())
      .map(([concept, data]) => ({
        concept,
        count: data.count,
        average_weight: data.totalWeight / data.count,
      }))
      .sort((a, b) => b.average_weight - a.average_weight)
      .slice(0, 10);

    // Sort and get top slices
    const mostUsedSlices = Array.from(sliceMap.entries())
      .map(([slice, data]) => ({
        slice,
        count: data.count,
        average_weight: data.totalWeight / data.count,
      }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);

    // Find high-confidence patterns (concepts that frequently appear together)
    const patterns = this.findConceptPatterns();

    return {
      total_queries: this.queryAttentions.size,
      total_traces: allTraces.length,
      most_influential_concepts: mostInfluentialConcepts,
      most_used_slices: mostUsedSlices,
      average_traces_per_query:
        this.queryAttentions.size > 0
          ? allTraces.length / this.queryAttentions.size
          : 0,
      high_confidence_patterns: patterns,
    };
  }

  /**
   * Find concepts that frequently appear together
   */
  private findConceptPatterns(): Array<{ concepts: string[]; frequency: number }> {
    const patternMap = new Map<string, number>();

    for (const attention of this.queryAttentions.values()) {
      // Get unique concepts in this query
      const concepts = [
        ...new Set(attention.traces.map((t) => t.concept)),
      ].sort();

      if (concepts.length < 2) {
        continue;
      }

      // Create pattern key
      const patternKey = concepts.join('|');
      patternMap.set(patternKey, (patternMap.get(patternKey) || 0) + 1);
    }

    // Convert to array and sort by frequency
    return Array.from(patternMap.entries())
      .map(([key, frequency]) => ({
        concepts: key.split('|'),
        frequency,
      }))
      .filter((p) => p.frequency > 1) // Only patterns that appear multiple times
      .sort((a, b) => b.frequency - a.frequency)
      .slice(0, 5);
  }

  /**
   * Export attention data for auditing
   *
   * Returns JSON-serializable object for regulatory compliance
   */
  exportForAudit(): {
    export_timestamp: number;
    total_queries: number;
    queries: Array<{
      query_id: string;
      query: string;
      timestamp: number;
      decision_path: string[];
      traces: Array<{
        concept: string;
        slice: string;
        weight: number;
        reasoning: string;
      }>;
    }>;
  } {
    return {
      export_timestamp: Date.now(),
      total_queries: this.queryAttentions.size,
      queries: Array.from(this.queryAttentions.values()).map((attention) => ({
        query_id: attention.query_id,
        query: attention.query,
        timestamp: attention.timestamp,
        decision_path: attention.decision_path,
        traces: attention.traces.map((t) => ({
          concept: t.concept,
          slice: t.slice,
          weight: t.weight,
          reasoning: t.reasoning,
        })),
      })),
    };
  }

  /**
   * Generate human-readable explanation of a query's reasoning
   */
  explainQuery(queryId: string): string {
    const attention = this.queryAttentions.get(queryId);
    if (!attention) {
      return `No attention data found for query: ${queryId}`;
    }

    const lines: string[] = [];
    lines.push(`═══ REASONING EXPLANATION ═══`);
    lines.push(`Query: "${attention.query}"`);
    lines.push(`Time: ${new Date(attention.timestamp).toISOString()}`);
    lines.push(`Total concepts considered: ${attention.total_concepts}`);
    lines.push('');

    lines.push(`─── DECISION PATH ─────────────`);
    attention.decision_path.forEach((decision, i) => {
      lines.push(`${i + 1}. ${decision}`);
    });
    lines.push('');

    lines.push(`─── TOP 5 INFLUENCES ──────────`);
    attention.top_influencers.forEach((trace, i) => {
      const percentage = (trace.weight * 100).toFixed(1);
      lines.push(`${i + 1}. [${percentage}%] ${trace.concept}`);
      lines.push(`   From: ${trace.slice}`);
      lines.push(`   Why: ${trace.reasoning}`);
      lines.push('');
    });

    lines.push(`─── ALL TRACES (${attention.traces.length}) ────────────`);
    const sortedTraces = [...attention.traces].sort(
      (a, b) => b.weight - a.weight
    );
    sortedTraces.forEach((trace) => {
      const percentage = (trace.weight * 100).toFixed(1);
      lines.push(
        `• [${percentage}%] ${trace.concept} (${trace.slice}): ${trace.reasoning}`
      );
    });

    return lines.join('\n');
  }

  /**
   * Clear all tracking data
   */
  clear(): void {
    this.queryAttentions.clear();
    this.currentQueryId = null;
    this.currentTraces = [];
  }

  /**
   * Get memory usage statistics
   */
  getMemoryStats(): {
    total_queries: number;
    total_traces: number;
    estimated_bytes: number;
  } {
    const allTraces = Array.from(this.queryAttentions.values()).flatMap(
      (q) => q.traces
    );

    // Rough estimate: each trace ~200 bytes (strings + numbers)
    const estimatedBytes =
      this.queryAttentions.size * 500 + // Query metadata
      allTraces.length * 200; // Traces

    return {
      total_queries: this.queryAttentions.size,
      total_traces: allTraces.length,
      estimated_bytes: estimatedBytes,
    };
  }
}

/**
 * Utility: Create a weight based on confidence and relevance
 *
 * Combines agent confidence with slice relevance to compute influence weight
 */
export function computeInfluenceWeight(
  agentConfidence: number, // 0-1
  sliceRelevance: number // 0-1
): number {
  // Geometric mean gives balanced weight
  return Math.sqrt(agentConfidence * sliceRelevance);
}

/**
 * Utility: Extract concepts from agent response
 *
 * Helper to identify which concepts were actually used in reasoning
 */
export function extractInfluentialConcepts(
  agentResponse: string,
  availableConcepts: string[]
): string[] {
  const influential: string[] = [];
  const lowerResponse = agentResponse.toLowerCase();

  for (const concept of availableConcepts) {
    // Check if concept appears in response
    // Could be enhanced with NLP for semantic matching
    if (lowerResponse.includes(concept.toLowerCase())) {
      influential.push(concept);
    }
  }

  return influential;
}
