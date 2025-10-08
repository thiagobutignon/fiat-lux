/**
 * Meta-Agent with Episodic Memory
 *
 * Extends the base MetaAgent with long-term memory capabilities.
 * Learns from past interactions and uses them to inform future responses.
 */

import { MetaAgent, AgentResponse, RecursionTrace, RecursionState } from './meta-agent';
import { EpisodicMemory, Episode, MemoryStats, createMemory } from './episodic-memory';
import { ConstitutionViolation } from './constitution';

// ============================================================================
// Extended Result Interface
// ============================================================================

export interface ProcessResultWithMemory {
  final_answer: string;
  trace: RecursionTrace[];
  emergent_insights: string[];
  reasoning_path: string;
  constitution_violations: ConstitutionViolation[];
  invocations: number;
  max_depth_reached: number;
  memory_used: boolean;
  similar_past_queries: Episode[];
}

// ============================================================================
// Meta-Agent with Memory
// ============================================================================

export class MetaAgentWithMemory extends MetaAgent {
  private memory: EpisodicMemory;
  private useMemory: boolean;

  constructor(
    apiKey: string,
    maxDepth: number = 5,
    maxInvocations: number = 10,
    maxCostUSD: number = 1.0,
    useMemory: boolean = true
  ) {
    super(apiKey, maxDepth, maxInvocations, maxCostUSD);
    this.memory = createMemory();
    this.useMemory = useMemory;
  }

  /**
   * Process query with memory augmentation
   */
  async processWithMemory(query: string): Promise<ProcessResultWithMemory> {
    let similar_past_queries: Episode[] = [];
    let memory_used = false;

    // Step 1: Check if we have relevant past episodes
    if (this.useMemory) {
      similar_past_queries = this.memory.findSimilarQueries(query, 3);

      if (similar_past_queries.length > 0 && similar_past_queries[0]) {
        const most_similar = similar_past_queries[0];

        // If very similar (Jaccard > 0.8), and successful, return cached response
        const query_words = new Set(query.toLowerCase().split(/\s+/));
        const similar_words = new Set(most_similar.query.toLowerCase().split(/\s+/));
        const intersection = new Set([...query_words].filter((w) => similar_words.has(w)));
        const union = new Set([...query_words, ...similar_words]);
        const similarity = intersection.size / union.size;

        if (similarity > 0.8 && most_similar.success && most_similar.confidence > 0.7) {
          memory_used = true;

          return {
            final_answer: `[Using cached response from similar query]\n\n${most_similar.response}`,
            trace: most_similar.execution_trace,
            emergent_insights: most_similar.emergent_insights,
            reasoning_path: 'Memory retrieval (cached response)',
            constitution_violations: [],
            invocations: 0,
            max_depth_reached: 0,
            memory_used: true,
            similar_past_queries,
          };
        }
      }
    }

    // Step 2: Process normally (no cache hit)
    const result = await this.process(query);

    // Step 3: Store in memory
    if (this.useMemory) {
      const all_concepts = this.extractAllConcepts(result.trace);
      const all_domains = this.extractAllDomains(result.trace);
      const agents_used = this.extractAgentsUsed(result.trace);

      // Calculate average confidence
      const confidences = result.trace.map((t) => t.response.confidence);
      const avg_confidence = confidences.length > 0 ? confidences.reduce((a, b) => a + b, 0) / confidences.length : 0.5;

      // Determine success (no violations, confidence > 0.5)
      const success = result.constitution_violations.length === 0 && avg_confidence > 0.5;

      // Calculate total cost
      const total_cost = result.trace.reduce((sum, t) => sum + t.cost_estimate, 0);

      this.memory.addEpisode(
        query,
        result.final_answer,
        all_concepts,
        all_domains,
        agents_used,
        total_cost,
        success,
        avg_confidence,
        result.trace,
        result.emergent_insights
      );
    }

    return {
      ...result,
      invocations: result.trace.length,
      max_depth_reached: Math.max(...result.trace.map((t) => t.depth), 0),
      memory_used,
      similar_past_queries,
    };
  }

  /**
   * Query memory for relevant past experiences
   */
  queryMemory(query: {
    concepts?: string[];
    domains?: string[];
    query_text?: string;
    min_confidence?: number;
    limit?: number;
  }): Episode[] {
    return this.memory.query(query);
  }

  /**
   * Get memory statistics
   */
  getMemoryStats(): MemoryStats {
    return this.memory.getStats();
  }

  /**
   * Consolidate memory (merge similar episodes, discover patterns)
   */
  consolidateMemory() {
    return this.memory.consolidate();
  }

  /**
   * Export memory to JSON (for persistence)
   */
  exportMemory(): string {
    return this.memory.export();
  }

  /**
   * Import memory from JSON
   */
  importMemory(json: string): number {
    return this.memory.import(json);
  }

  /**
   * Clear all memory
   */
  clearMemory(): void {
    this.memory.clear();
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  private extractAllConcepts(traces: RecursionTrace[]): string[] {
    const concepts = new Set<string>();
    traces.forEach((trace) => {
      trace.response.concepts.forEach((c) => concepts.add(c));
    });
    return Array.from(concepts);
  }

  private extractAllDomains(traces: RecursionTrace[]): string[] {
    const domains = new Set<string>();
    traces.forEach((trace) => {
      domains.add(trace.agent_id);
    });
    return Array.from(domains);
  }

  private extractAgentsUsed(traces: RecursionTrace[]): string[] {
    return Array.from(new Set(traces.map((t) => t.agent_id)));
  }
}

/**
 * Create a meta-agent with memory
 */
export function createMetaAgentWithMemory(
  apiKey: string,
  maxDepth: number = 5,
  maxInvocations: number = 10,
  maxCostUSD: number = 1.0,
  useMemory: boolean = true
): MetaAgentWithMemory {
  return new MetaAgentWithMemory(apiKey, maxDepth, maxInvocations, maxCostUSD, useMemory);
}
