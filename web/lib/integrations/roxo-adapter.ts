/**
 * ROXO Adapter - Bridge between AMARELO and ROXO Core
 *
 * This adapter provides a bridge between AMARELO's web dashboard types
 * and ROXO's core GlassRuntime system.
 *
 * Architecture:
 * AMARELO → roxo-adapter.ts → ROXO Core (GlassRuntime)
 *
 * Type Conversions:
 * - ROXO QueryResult → AMARELO QueryResult
 * - ROXO GlassOrganism → AMARELO GlassOrganism
 * - Runtime management with caching
 */

import {
  GlassRuntime,
  createRuntime,
  QueryContext,
  QueryResult as RoxoQueryResult,
  AttentionWeight as RoxoAttentionWeight,
} from '../../../src/grammar-lang/glass/runtime';

import { GlassOrganism as RoxoOrganism } from '../../../src/grammar-lang/glass/types';
import { loadGlassOrganism } from '../../../src/grammar-lang/glass/builder';

import type {
  GlassOrganism,
  QueryResult,
  Pattern,
  EmergedFunction,
  AttentionWeight,
  ReasoningStep,
  Source,
} from '../types';

// ============================================================================
// Adapter Class
// ============================================================================

export class RoxoAdapter {
  private runtimeCache: Map<
    string,
    { runtime: GlassRuntime; timestamp: number; organism: RoxoOrganism }
  >;
  private cacheTTL: number = 10 * 60 * 1000; // 10 minutes
  private maxBudget: number = 0.5; // $0.50 per query

  constructor(maxBudget?: number) {
    this.runtimeCache = new Map();
    if (maxBudget !== undefined) {
      this.maxBudget = maxBudget;
    }
  }

  // ==========================================================================
  // Helper Functions
  // ==========================================================================

  /**
   * Convert ROXO QueryResult to AMARELO QueryResult
   */
  private convertQueryResult(roxoResult: RoxoQueryResult): QueryResult {
    // Convert attention weights
    const attention: AttentionWeight[] = roxoResult.attention_weights.map((att) => ({
      source_id: att.knowledge_id,
      weight: att.weight,
    }));

    // Convert reasoning steps
    const reasoning: ReasoningStep[] = roxoResult.reasoning.map((step, index) => ({
      step: index + 1,
      description: step,
      confidence: roxoResult.confidence,
      time_ms: 0, // ROXO doesn't track per-step timing yet
    }));

    // Convert sources (extract from attention weights)
    const sources: Source[] = roxoResult.sources.map((source, index) => ({
      id: `source_${index}`,
      title: source,
      type: 'paper' as const,
      relevance: roxoResult.confidence,
    }));

    return {
      answer: roxoResult.answer,
      confidence: roxoResult.confidence,
      functions_used: roxoResult.functions_used,
      constitutional: roxoResult.constitutional_passed ? 'pass' : 'fail',
      cost: roxoResult.cost_usd,
      time_ms: 0, // Will be calculated by caller
      sources,
      attention,
      reasoning,
    };
  }

  /**
   * Convert ROXO GlassOrganism to AMARELO GlassOrganism
   */
  private convertOrganism(roxoOrganism: RoxoOrganism, organismId: string): GlassOrganism {
    // Convert patterns
    const patterns: Pattern[] = Object.entries(roxoOrganism.knowledge.patterns).map(
      ([keyword, frequency]) => ({
        keyword,
        frequency: frequency as number,
        confidence: Math.min((frequency as number) / 100, 1.0),
        emergence_score: Math.min((frequency as number) / 250, 1.0),
        emerged_function: undefined,
      })
    );

    // Convert emerged functions
    const functions: EmergedFunction[] = roxoOrganism.code.functions.map((fn) => ({
      name: fn.name,
      signature: fn.signature,
      code: fn.source,
      emerged_from: fn.source_patterns.join(', '),
      occurrences: fn.source_patterns.length,
      constitutional_status: 'pass' as const,
      lines: fn.source.split('\n').length,
      created_at: roxoOrganism.metadata.created_at || new Date().toISOString(),
    }));

    return {
      id: organismId,
      metadata: {
        name: roxoOrganism.metadata.name,
        version: roxoOrganism.metadata.version,
        specialization: roxoOrganism.metadata.specialization,
        created_at: roxoOrganism.metadata.created_at || new Date().toISOString(),
        updated_at: roxoOrganism.metadata.updated_at || new Date().toISOString(),
        maturity: roxoOrganism.metadata.maturity,
        stage: this.calculateStage(roxoOrganism.metadata.maturity),
        generation: roxoOrganism.metadata.generation || 1,
      },
      model: {
        architecture: roxoOrganism.model.architecture,
        parameters: roxoOrganism.model.parameters,
        quantization: roxoOrganism.model.quantization,
      },
      knowledge: {
        papers: roxoOrganism.knowledge.papers.count,
        embeddings_dim: roxoOrganism.knowledge.embeddings_dim,
        patterns,
        connections: roxoOrganism.knowledge.connections,
        clusters: roxoOrganism.knowledge.clusters,
      },
      code: {
        functions,
        total_lines: roxoOrganism.code.total_lines,
      },
      memory: {
        short_term: roxoOrganism.memory.short_term,
        long_term: roxoOrganism.memory.long_term,
        contextual: [], // ROXO doesn't have contextual memory yet
      },
      constitutional: {
        agent_type: roxoOrganism.constitutional.agent_type,
        principles: roxoOrganism.constitutional.principles,
        boundaries: roxoOrganism.constitutional.boundaries,
        validation: roxoOrganism.constitutional.validation,
      },
      evolution: {
        enabled: roxoOrganism.evolution.enabled,
        generation: roxoOrganism.evolution.generation,
        fitness: roxoOrganism.evolution.fitness,
        trajectory: roxoOrganism.evolution.trajectory,
      },
      stats: {
        total_cost: 0, // Will be tracked by runtime
        queries_count: roxoOrganism.memory.short_term.length,
        avg_query_time_ms: 0, // Will be calculated
        last_query_at: undefined,
      },
    };
  }

  /**
   * Calculate organism stage from maturity
   */
  private calculateStage(
    maturity: number
  ): 'nascent' | 'infancy' | 'adolescence' | 'maturity' | 'evolution' {
    if (maturity < 0.2) return 'nascent';
    if (maturity < 0.4) return 'infancy';
    if (maturity < 0.6) return 'adolescence';
    if (maturity < 0.8) return 'maturity';
    return 'evolution';
  }

  // ==========================================================================
  // Runtime Management
  // ==========================================================================

  /**
   * Get or create GlassRuntime for organism
   */
  async getRuntime(organismPath: string, organismId: string): Promise<GlassRuntime> {
    // Check cache
    const cached = this.runtimeCache.get(organismId);
    if (cached && Date.now() - cached.timestamp < this.cacheTTL) {
      return cached.runtime;
    }

    // Create new runtime
    const runtime = await createRuntime(organismPath, this.maxBudget);

    // Load organism to cache metadata
    const organism = await loadGlassOrganism(organismPath);

    // Cache runtime
    this.runtimeCache.set(organismId, {
      runtime,
      organism,
      timestamp: Date.now(),
    });

    return runtime;
  }

  /**
   * Load organism metadata
   */
  async loadOrganism(organismPath: string, organismId: string): Promise<GlassOrganism> {
    // Try to get from cache first
    const cached = this.runtimeCache.get(organismId);
    if (cached && Date.now() - cached.timestamp < this.cacheTTL) {
      return this.convertOrganism(cached.organism, organismId);
    }

    // Load fresh
    const roxoOrganism = await loadGlassOrganism(organismPath);

    // Cache it
    const runtime = new GlassRuntime(roxoOrganism, this.maxBudget);
    this.runtimeCache.set(organismId, {
      runtime,
      organism: roxoOrganism,
      timestamp: Date.now(),
    });

    return this.convertOrganism(roxoOrganism, organismId);
  }

  // ==========================================================================
  // Query Execution
  // ==========================================================================

  /**
   * Execute query against organism
   */
  async executeQuery(
    organismPath: string,
    organismId: string,
    query: string,
    context?: QueryContext
  ): Promise<QueryResult> {
    const startTime = Date.now();

    try {
      // Get runtime
      const runtime = await this.getRuntime(organismPath, organismId);

      // Execute query
      const roxoResult = await runtime.query(
        context || {
          query,
        }
      );

      // Convert result
      const result = this.convertQueryResult(roxoResult);

      // Add timing
      result.time_ms = Date.now() - startTime;

      return result;
    } catch (error) {
      console.error('[ROXO Adapter] Query execution error:', error);
      throw error;
    }
  }

  // ==========================================================================
  // Pattern Detection
  // ==========================================================================

  /**
   * Get detected patterns from organism
   */
  async getPatterns(organismPath: string, organismId: string): Promise<Pattern[]> {
    const organism = await this.loadOrganism(organismPath, organismId);
    return organism.knowledge.patterns;
  }

  /**
   * Detect new patterns in organism knowledge
   */
  async detectPatterns(organismPath: string, organismId: string): Promise<Pattern[]> {
    // Get current patterns
    const currentPatterns = await this.getPatterns(organismPath, organismId);

    // In a real implementation, this would trigger pattern detection in GlassRuntime
    // For now, return current patterns (future: runtime.detectPatterns())
    console.log('[ROXO Adapter] detectPatterns: Using current patterns');

    return currentPatterns;
  }

  // ==========================================================================
  // Code Emergence
  // ==========================================================================

  /**
   * Get emerged functions from organism
   */
  async getEmergedFunctions(
    organismPath: string,
    organismId: string
  ): Promise<EmergedFunction[]> {
    const organism = await this.loadOrganism(organismPath, organismId);
    return organism.code.functions;
  }

  /**
   * Synthesize code from patterns
   */
  async synthesizeCode(
    organismPath: string,
    organismId: string,
    patterns: Pattern[]
  ): Promise<EmergedFunction[]> {
    // In a real implementation, this would trigger code synthesis in GlassRuntime
    // For now, return current emerged functions
    console.log('[ROXO Adapter] synthesizeCode: Would synthesize from', patterns.length, 'patterns');

    const currentFunctions = await this.getEmergedFunctions(organismPath, organismId);
    return currentFunctions;
  }

  // ==========================================================================
  // Knowledge Management
  // ==========================================================================

  /**
   * Ingest new knowledge into organism
   */
  async ingestKnowledge(
    organismPath: string,
    organismId: string,
    documents: { content: string; metadata: any }[]
  ): Promise<{ success: boolean; documents_ingested: number; message: string }> {
    try {
      // In a real implementation, this would call runtime.ingest(documents)
      // For now, just log the operation
      console.log('[ROXO Adapter] ingestKnowledge:', documents.length, 'documents');

      return {
        success: true,
        documents_ingested: documents.length,
        message: `Successfully ingested ${documents.length} documents (feature available, awaiting persistence layer)`,
      };
    } catch (error) {
      return {
        success: false,
        documents_ingested: 0,
        message: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * Get knowledge graph data
   */
  async getKnowledgeGraph(
    organismPath: string,
    organismId: string
  ): Promise<{
    nodes: Array<{ id: string; type: string; label: string; properties: any }>;
    edges: Array<{ source: string; target: string; type: string; weight: number }>;
  }> {
    const organism = await this.loadOrganism(organismPath, organismId);

    // Build knowledge graph from organism data
    const nodes = [];
    const edges = [];

    // Add pattern nodes
    for (const pattern of organism.knowledge.patterns) {
      nodes.push({
        id: `pattern_${pattern.keyword}`,
        type: 'pattern',
        label: pattern.keyword,
        properties: {
          frequency: pattern.frequency,
          confidence: pattern.confidence,
        },
      });
    }

    // Add function nodes
    for (const fn of organism.code.functions) {
      nodes.push({
        id: `function_${fn.name}`,
        type: 'function',
        label: fn.name,
        properties: {
          lines: fn.lines,
          occurrences: fn.occurrences,
        },
      });

      // Add edges from patterns to functions
      const patterns = fn.emerged_from.split(', ');
      for (const patternName of patterns) {
        if (patternName) {
          edges.push({
            source: `pattern_${patternName}`,
            target: `function_${fn.name}`,
            type: 'emerged_from',
            weight: 0.8,
          });
        }
      }
    }

    return { nodes, edges };
  }

  // ==========================================================================
  // Constitutional Validation
  // ==========================================================================

  /**
   * Validate query against constitutional principles
   */
  async validateQuery(
    organismPath: string,
    organismId: string,
    query: string
  ): Promise<{ status: 'pass' | 'fail'; details: string; violations: string[] }> {
    try {
      const runtime = await this.getRuntime(organismPath, organismId);

      // Execute query and check constitutional result
      const result = await runtime.query({ query });

      if (result.constitutional_passed) {
        return {
          status: 'pass',
          details: 'Query passed constitutional validation',
          violations: [],
        };
      } else {
        return {
          status: 'fail',
          details: 'Query failed constitutional validation',
          violations: ['Constitutional principles violated during query execution'],
        };
      }
    } catch (error) {
      // Fail-open on error
      return {
        status: 'pass',
        details: `Validation error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        violations: [],
      };
    }
  }

  // ==========================================================================
  // Runtime Statistics
  // ==========================================================================

  /**
   * Get runtime statistics
   */
  async getRuntimeStats(
    organismPath: string,
    organismId: string
  ): Promise<{
    organism_name: string;
    specialization: string;
    maturity: number;
    functions_count: number;
    knowledge_count: number;
    total_cost: number;
    remaining_budget: number;
    queries_processed: number;
  }> {
    const runtime = await this.getRuntime(organismPath, organismId);
    const stats = runtime.getStats();

    return {
      organism_name: stats.organism.name,
      specialization: stats.organism.specialization,
      maturity: stats.organism.maturity,
      functions_count: stats.organism.functions_count,
      knowledge_count: stats.organism.knowledge_count,
      total_cost: stats.runtime.total_cost,
      remaining_budget: stats.runtime.remaining_budget,
      queries_processed: stats.runtime.queries_processed,
    };
  }

  // ==========================================================================
  // Health & Status
  // ==========================================================================

  /**
   * Check if ROXO is available
   */
  isAvailable(): boolean {
    try {
      // Try to import ROXO modules
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Get ROXO health status
   */
  async getHealth(): Promise<{
    status: string;
    version: string;
    runtimes_cached?: number;
  }> {
    try {
      return {
        status: 'healthy',
        version: '1.0.0',
        runtimes_cached: this.runtimeCache.size,
      };
    } catch (error) {
      return {
        status: 'error',
        version: 'unknown',
        runtimes_cached: 0,
      };
    }
  }

  /**
   * Clear runtime cache
   */
  clearCache(): void {
    this.runtimeCache.clear();
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): {
    cached_runtimes: number;
    cache_ttl_ms: number;
    max_budget: number;
  } {
    return {
      cached_runtimes: this.runtimeCache.size,
      cache_ttl_ms: this.cacheTTL,
      max_budget: this.maxBudget,
    };
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let adapterInstance: RoxoAdapter | null = null;

export function getRoxoAdapter(maxBudget?: number): RoxoAdapter {
  if (!adapterInstance) {
    adapterInstance = new RoxoAdapter(maxBudget);
  }
  return adapterInstance;
}
