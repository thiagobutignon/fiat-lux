/**
 * Episodic Memory System
 *
 * Implements long-term memory for the AGI system, storing past interactions
 * and enabling learning from experience.
 *
 * Key Features:
 * - Stores complete interaction episodes (query, response, context)
 * - Indexes by concepts for fast retrieval
 * - Semantic similarity search
 * - Temporal decay (older memories fade)
 * - Memory consolidation (merge similar episodes)
 *
 * Inspired by human episodic memory:
 * - What happened (query + response)
 * - When it happened (timestamp)
 * - What concepts were involved
 * - What worked (success/failure)
 */

import crypto from 'crypto';
import { AgentResponse, RecursionTrace } from './meta-agent';

// ============================================================================
// Types
// ============================================================================

export interface Episode {
  id: string;
  timestamp: number;
  query: string;
  query_hash: string; // For deduplication
  response: string;
  concepts: string[]; // Concepts involved
  domains: string[]; // Domains consulted
  agents_used: string[];
  cost: number;
  success: boolean; // Whether query was successfully answered
  confidence: number; // Average confidence of responses
  execution_trace: RecursionTrace[];
  emergent_insights: string[]; // Novel concepts discovered
  metadata: {
    depth: number;
    invocations: number;
    violations: number;
  };
}

export interface MemoryQuery {
  concepts?: string[]; // Find episodes with these concepts
  domains?: string[]; // Find episodes from these domains
  query_text?: string; // Semantic search on query
  min_confidence?: number; // Filter by confidence
  limit?: number; // Max results
  since?: number; // Timestamp filter
}

export interface MemoryStats {
  total_episodes: number;
  total_concepts: number;
  total_cost: number;
  average_confidence: number;
  success_rate: number;
  most_common_concepts: Array<{ concept: string; count: number }>;
  most_queried_domains: Array<{ domain: string; count: number }>;
  temporal_coverage: {
    oldest: number;
    newest: number;
    span_hours: number;
  };
}

export interface ConsolidationResult {
  merged_count: number;
  new_insights: string[];
  patterns_discovered: string[];
}

// ============================================================================
// Episodic Memory
// ============================================================================

export class EpisodicMemory {
  private episodes: Map<string, Episode> = new Map();
  private conceptIndex: Map<string, Set<string>> = new Map(); // concept -> episode IDs
  private domainIndex: Map<string, Set<string>> = new Map(); // domain -> episode IDs
  private queryIndex: Map<string, string> = new Map(); // query_hash -> episode ID

  /**
   * Store a new episode in memory
   */
  addEpisode(
    query: string,
    response: string,
    concepts: string[],
    domains: string[],
    agents_used: string[],
    cost: number,
    success: boolean,
    confidence: number,
    execution_trace: RecursionTrace[],
    emergent_insights: string[] = []
  ): Episode {
    const query_hash = this.hashQuery(query);

    // Check if similar episode already exists
    const existing_id = this.queryIndex.get(query_hash);
    if (existing_id) {
      // Update existing episode instead of creating duplicate
      const existing = this.episodes.get(existing_id)!;
      existing.timestamp = Date.now(); // Refresh timestamp
      existing.response = response; // Update with latest response
      existing.confidence = (existing.confidence + confidence) / 2; // Average
      return existing;
    }

    const episode: Episode = {
      id: crypto.randomUUID(),
      timestamp: Date.now(),
      query,
      query_hash,
      response,
      concepts,
      domains,
      agents_used,
      cost,
      success,
      confidence,
      execution_trace,
      emergent_insights,
      metadata: {
        depth: Math.max(...execution_trace.map((t) => t.depth), 0),
        invocations: execution_trace.length,
        violations: 0, // Could be tracked
      },
    };

    // Store episode
    this.episodes.set(episode.id, episode);
    this.queryIndex.set(query_hash, episode.id);

    // Index by concepts
    for (const concept of concepts) {
      if (!this.conceptIndex.has(concept)) {
        this.conceptIndex.set(concept, new Set());
      }
      this.conceptIndex.get(concept)!.add(episode.id);
    }

    // Index by domains
    for (const domain of domains) {
      if (!this.domainIndex.has(domain)) {
        this.domainIndex.set(domain, new Set());
      }
      this.domainIndex.get(domain)!.add(episode.id);
    }

    return episode;
  }

  /**
   * Query memory for relevant past episodes
   */
  query(query: MemoryQuery): Episode[] {
    let candidates = new Set<string>(this.episodes.keys());

    // Filter by concepts
    if (query.concepts && query.concepts.length > 0) {
      const concept_matches = new Set<string>();
      for (const concept of query.concepts) {
        const episode_ids = this.conceptIndex.get(concept);
        if (episode_ids) {
          episode_ids.forEach((id) => concept_matches.add(id));
        }
      }
      candidates = new Set([...candidates].filter((id) => concept_matches.has(id)));
    }

    // Filter by domains
    if (query.domains && query.domains.length > 0) {
      const domain_matches = new Set<string>();
      for (const domain of query.domains) {
        const episode_ids = this.domainIndex.get(domain);
        if (episode_ids) {
          episode_ids.forEach((id) => domain_matches.add(id));
        }
      }
      candidates = new Set([...candidates].filter((id) => domain_matches.has(id)));
    }

    // Filter by confidence
    if (query.min_confidence !== undefined) {
      candidates = new Set(
        [...candidates].filter((id) => {
          const episode = this.episodes.get(id)!;
          return episode.confidence >= query.min_confidence!;
        })
      );
    }

    // Filter by timestamp
    if (query.since !== undefined) {
      candidates = new Set(
        [...candidates].filter((id) => {
          const episode = this.episodes.get(id)!;
          return episode.timestamp >= query.since!;
        })
      );
    }

    // Semantic search on query text (simple substring match for now)
    if (query.query_text) {
      const search_lower = query.query_text.toLowerCase();
      candidates = new Set(
        [...candidates].filter((id) => {
          const episode = this.episodes.get(id)!;
          return episode.query.toLowerCase().includes(search_lower);
        })
      );
    }

    // Convert to episodes and sort by relevance (timestamp * confidence)
    let results = [...candidates]
      .map((id) => this.episodes.get(id)!)
      .sort((a, b) => {
        const score_a = a.timestamp * a.confidence;
        const score_b = b.timestamp * b.confidence;
        return score_b - score_a; // Descending
      });

    // Apply limit
    if (query.limit) {
      results = results.slice(0, query.limit);
    }

    return results;
  }

  /**
   * Get memory statistics
   */
  getStats(): MemoryStats {
    const episodes = Array.from(this.episodes.values());

    if (episodes.length === 0) {
      return {
        total_episodes: 0,
        total_concepts: 0,
        total_cost: 0,
        average_confidence: 0,
        success_rate: 0,
        most_common_concepts: [],
        most_queried_domains: [],
        temporal_coverage: {
          oldest: 0,
          newest: 0,
          span_hours: 0,
        },
      };
    }

    // Concept frequency
    const concept_counts = new Map<string, number>();
    episodes.forEach((ep) => {
      ep.concepts.forEach((concept) => {
        concept_counts.set(concept, (concept_counts.get(concept) || 0) + 1);
      });
    });

    // Domain frequency
    const domain_counts = new Map<string, number>();
    episodes.forEach((ep) => {
      ep.domains.forEach((domain) => {
        domain_counts.set(domain, (domain_counts.get(domain) || 0) + 1);
      });
    });

    // Temporal coverage
    const timestamps = episodes.map((ep) => ep.timestamp);
    const oldest = Math.min(...timestamps);
    const newest = Math.max(...timestamps);
    const span_hours = (newest - oldest) / (1000 * 60 * 60);

    return {
      total_episodes: episodes.length,
      total_concepts: this.conceptIndex.size,
      total_cost: episodes.reduce((sum, ep) => sum + ep.cost, 0),
      average_confidence: episodes.reduce((sum, ep) => sum + ep.confidence, 0) / episodes.length,
      success_rate: episodes.filter((ep) => ep.success).length / episodes.length,
      most_common_concepts: Array.from(concept_counts.entries())
        .map(([concept, count]) => ({ concept, count }))
        .sort((a, b) => b.count - a.count)
        .slice(0, 10),
      most_queried_domains: Array.from(domain_counts.entries())
        .map(([domain, count]) => ({ domain, count }))
        .sort((a, b) => b.count - a.count),
      temporal_coverage: {
        oldest,
        newest,
        span_hours,
      },
    };
  }

  /**
   * Consolidate memory - merge similar episodes and discover patterns
   */
  consolidate(): ConsolidationResult {
    const episodes = Array.from(this.episodes.values());
    let merged_count = 0;
    const new_insights: string[] = [];
    const patterns_discovered: string[] = [];

    // Group by query hash
    const groups = new Map<string, Episode[]>();
    episodes.forEach((ep) => {
      if (!groups.has(ep.query_hash)) {
        groups.set(ep.query_hash, []);
      }
      groups.get(ep.query_hash)!.push(ep);
    });

    // Merge episodes with same query hash
    groups.forEach((group, query_hash) => {
      if (group.length > 1) {
        // Keep most recent, delete others
        group.sort((a, b) => b.timestamp - a.timestamp);
        const keep = group[0];

        for (let i = 1; i < group.length; i++) {
          const episode = group[i];
          this.episodes.delete(episode.id);
          merged_count++;

          // Merge emergent insights
          episode.emergent_insights.forEach((insight) => {
            if (!keep.emergent_insights.includes(insight)) {
              keep.emergent_insights.push(insight);
              new_insights.push(insight);
            }
          });
        }
      }
    });

    // Discover patterns: concepts that frequently appear together
    const concept_pairs = new Map<string, number>();
    episodes.forEach((ep) => {
      for (let i = 0; i < ep.concepts.length; i++) {
        for (let j = i + 1; j < ep.concepts.length; j++) {
          const pair = [ep.concepts[i], ep.concepts[j]].sort().join('::');
          concept_pairs.set(pair, (concept_pairs.get(pair) || 0) + 1);
        }
      }
    });

    // Patterns are pairs that appear in >20% of episodes
    const threshold = episodes.length * 0.2;
    concept_pairs.forEach((count, pair) => {
      if (count >= threshold) {
        patterns_discovered.push(`Pattern: ${pair} (appears in ${count} episodes)`);
      }
    });

    return {
      merged_count,
      new_insights,
      patterns_discovered,
    };
  }

  /**
   * Find similar past queries
   */
  findSimilarQueries(query: string, limit: number = 5): Episode[] {
    const query_lower = query.toLowerCase();
    const query_words = new Set(query_lower.split(/\s+/));

    // Calculate Jaccard similarity for each episode
    const scored_episodes = Array.from(this.episodes.values()).map((episode) => {
      const episode_words = new Set(episode.query.toLowerCase().split(/\s+/));

      // Jaccard similarity
      const intersection = new Set([...query_words].filter((w) => episode_words.has(w)));
      const union = new Set([...query_words, ...episode_words]);
      const similarity = intersection.size / union.size;

      return { episode, similarity };
    });

    // Sort by similarity and return top N
    return scored_episodes
      .filter((s) => s.similarity > 0)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, limit)
      .map((s) => s.episode);
  }

  /**
   * Get episode by ID
   */
  getEpisode(id: string): Episode | undefined {
    return this.episodes.get(id);
  }

  /**
   * Clear all memory (for testing/reset)
   */
  clear(): void {
    this.episodes.clear();
    this.conceptIndex.clear();
    this.domainIndex.clear();
    this.queryIndex.clear();
  }

  /**
   * Export memory to JSON (for persistence)
   */
  export(): string {
    const data = {
      episodes: Array.from(this.episodes.values()),
      metadata: {
        exported_at: Date.now(),
        version: '1.0',
      },
    };
    return JSON.stringify(data, null, 2);
  }

  /**
   * Import memory from JSON
   */
  import(json: string): number {
    const data = JSON.parse(json);
    const episodes = data.episodes as Episode[];

    let imported = 0;
    for (const episode of episodes) {
      // Re-index
      this.episodes.set(episode.id, episode);
      this.queryIndex.set(episode.query_hash, episode.id);

      episode.concepts.forEach((concept) => {
        if (!this.conceptIndex.has(concept)) {
          this.conceptIndex.set(concept, new Set());
        }
        this.conceptIndex.get(concept)!.add(episode.id);
      });

      episode.domains.forEach((domain) => {
        if (!this.domainIndex.has(domain)) {
          this.domainIndex.set(domain, new Set());
        }
        this.domainIndex.get(domain)!.add(episode.id);
      });

      imported++;
    }

    return imported;
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  private hashQuery(query: string): string {
    // Normalize query for hashing
    const normalized = query.toLowerCase().trim().replace(/\s+/g, ' ');
    return crypto.createHash('sha256').update(normalized).digest('hex');
  }
}

/**
 * Create a new episodic memory instance
 */
export function createMemory(): EpisodicMemory {
  return new EpisodicMemory();
}
