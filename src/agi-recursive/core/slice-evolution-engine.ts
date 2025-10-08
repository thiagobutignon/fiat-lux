/**
 * SliceEvolutionEngine - Orchestrates self-evolution of knowledge slices
 *
 * Core component that:
 * - Analyzes episodic memory for patterns
 * - Proposes new/updated slices
 * - Validates with constitution
 * - Deploys evolutions safely
 * - Tracks evolution history
 * - Enables rollback
 */

import { EpisodicMemory } from './episodic-memory';
import { KnowledgeDistillation, KnowledgePattern, ErrorPattern } from './knowledge-distillation';
import { SliceRewriter } from './slice-rewriter';
import { SliceNavigator } from './slice-navigator';
import { ConstitutionEnforcer } from './constitution';
import { Observability } from './observability';

// ============================================================================
// Types
// ============================================================================

export enum EvolutionType {
  CREATED = 'CREATED',
  UPDATED = 'UPDATED',
  MERGED = 'MERGED',
  DEPRECATED = 'DEPRECATED',
}

export enum EvolutionTrigger {
  SCHEDULED = 'SCHEDULED',
  THRESHOLD = 'THRESHOLD',
  MANUAL = 'MANUAL',
  CONTINUOUS = 'CONTINUOUS',
}

export interface SliceCandidate {
  id: string;
  type: 'new' | 'update' | 'merge' | 'deprecate';
  title: string;
  description: string;
  concepts: string[];
  content: string; // YAML content
  supporting_episodes: string[]; // Episode IDs
  pattern: KnowledgePattern;
  constitutional_score: number; // 0-1
  test_performance: {
    queries_tested: number;
    accuracy_improvement: number;
    cost_delta: number;
  };
  should_deploy: boolean;
  reasoning: string;
}

export interface SliceEvolution {
  id: string;
  timestamp: number;
  evolution_type: EvolutionType;
  trigger: EvolutionTrigger;
  slice_id: string;
  candidate: SliceCandidate;
  approved: boolean;
  deployed_at?: number;
  rolled_back?: boolean;
  rollback_reason?: string;
  backup_path?: string;
  performance_delta?: {
    accuracy_change: number;
    cost_change: number;
  };
}

export interface EvolutionMetrics {
  total_evolutions: number;
  successful_deployments: number;
  rollbacks: number;
  by_type: Record<EvolutionType, number>;
  by_trigger: Record<EvolutionTrigger, number>;
  avg_constitutional_score: number;
  knowledge_growth: {
    slices_created: number;
    slices_updated: number;
    slices_merged: number;
    slices_deprecated: number;
  };
}

// ============================================================================
// SliceEvolutionEngine Class
// ============================================================================

export class SliceEvolutionEngine {
  private evolutions: SliceEvolution[] = [];
  private evolutionIdCounter = 0;

  constructor(
    private episodicMemory: EpisodicMemory,
    private knowledgeDistillation: KnowledgeDistillation,
    private sliceRewriter: SliceRewriter,
    private sliceNavigator: SliceNavigator,
    private constitutionEnforcer: ConstitutionEnforcer,
    private observability: Observability
  ) {}

  /**
   * Analyze episodic memory and propose evolution candidates
   */
  async analyzeAndPropose(
    trigger: EvolutionTrigger = EvolutionTrigger.MANUAL,
    minFrequency: number = 3
  ): Promise<SliceCandidate[]> {
    const span = this.observability.startSpan('analyze_and_propose');
    span.setTag('trigger', trigger);
    span.setTag('min_frequency', minFrequency);

    try {
      // 1. Get recent episodes
      const episodes = this.episodicMemory.query({});
      span.setTag('episodes_count', episodes.length);

      if (episodes.length === 0) {
        this.observability.log('info', 'no_episodes', { trigger });
        return [];
      }

      // 2. Discover patterns
      const patterns = await this.knowledgeDistillation.discoverPatterns(
        episodes,
        minFrequency
      );
      span.setTag('patterns_found', patterns.length);

      // 3. Identify gaps
      const gaps = await this.knowledgeDistillation.identifyGaps(episodes);
      span.setTag('gaps_found', gaps.length);

      // 4. Detect errors
      const errors = await this.knowledgeDistillation.detectErrors(episodes);
      span.setTag('errors_found', errors.length);

      // 5. Generate candidates
      const candidates: SliceCandidate[] = [];

      // Create candidates for strong patterns
      for (const pattern of patterns) {
        if (pattern.confidence >= 0.7) {
          const candidate = await this.createCandidateFromPattern(pattern);
          candidates.push(candidate);
        }
      }

      // Create candidates for knowledge gaps (lower priority)
      // (implementation would analyze gaps and create appropriate candidates)

      this.observability.log('info', 'candidates_proposed', {
        total_candidates: candidates.length,
        trigger,
      });

      span.setTag('candidates_proposed', candidates.length);
      return candidates;
    } finally {
      span.end();
    }
  }

  /**
   * Deploy an approved evolution candidate
   */
  async deployEvolution(
    candidate: SliceCandidate,
    trigger: EvolutionTrigger = EvolutionTrigger.MANUAL
  ): Promise<SliceEvolution> {
    const span = this.observability.startSpan('deploy_evolution');
    span.setTag('candidate_id', candidate.id);
    span.setTag('candidate_type', candidate.type);
    span.setTag('trigger', trigger);

    try {
      // 1. Validate should_deploy flag
      if (!candidate.should_deploy) {
        throw new Error(
          `Candidate ${candidate.id} has should_deploy=false, cannot deploy`
        );
      }

      // 2. Determine evolution type
      const evolutionType = this.mapCandidateTypeToEvolutionType(
        candidate.type
      );

      // 3. Backup if updating existing slice
      let backupPath: string | undefined;
      const sliceExists = this.sliceRewriter.exists(candidate.id);

      if (sliceExists && candidate.type === 'update') {
        backupPath = await this.sliceRewriter.backup(candidate.id);
        span.setTag('backup_created', true);
        span.setTag('backup_path', backupPath);
      }

      // 4. Write slice
      if (candidate.type === 'new' || !sliceExists) {
        await this.sliceRewriter.createSlice(candidate.id, candidate.content);
      } else if (candidate.type === 'update') {
        await this.sliceRewriter.updateSlice(candidate.id, candidate.content);
      }

      // 5. Reindex navigator (if it has a reindex method)
      // this.sliceNavigator.reindex();

      // 6. Create evolution record
      const evolution: SliceEvolution = {
        id: `evo-${this.evolutionIdCounter++}`,
        timestamp: Date.now(),
        evolution_type: evolutionType,
        trigger,
        slice_id: candidate.id,
        candidate,
        approved: true,
        deployed_at: Date.now(),
        backup_path: backupPath,
        performance_delta: {
          accuracy_change: candidate.test_performance.accuracy_improvement,
          cost_change: candidate.test_performance.cost_delta,
        },
      };

      // 7. Store evolution
      this.evolutions.push(evolution);

      // 8. Log deployment
      this.observability.log('info', 'evolution_deployed', {
        evolution_id: evolution.id,
        slice_id: candidate.id,
        type: evolutionType,
        trigger,
      });

      span.setTag('success', true);
      span.setTag('evolution_id', evolution.id);

      return evolution;
    } catch (error) {
      span.setTag('success', false);
      span.setTag('error', (error as Error).message);
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Rollback a deployed evolution
   */
  async rollback(evolutionId: string): Promise<void> {
    const span = this.observability.startSpan('rollback_evolution');
    span.setTag('evolution_id', evolutionId);

    try {
      // 1. Find evolution
      const evolution = this.evolutions.find((e) => e.id === evolutionId);
      if (!evolution) {
        throw new Error(`Evolution ${evolutionId} not found`);
      }

      // 2. Check if already rolled back
      if (evolution.rolled_back) {
        throw new Error(`Evolution ${evolutionId} already rolled back`);
      }

      // 3. Restore from backup if available
      if (evolution.backup_path) {
        await this.sliceRewriter.restore(evolution.backup_path);
        span.setTag('restored_from_backup', true);
      } else if (evolution.evolution_type === EvolutionType.CREATED) {
        // Delete the created slice
        await this.sliceRewriter.deleteSlice(evolution.slice_id);
        span.setTag('deleted_slice', true);
      } else {
        throw new Error(
          `Cannot rollback ${evolutionId}: no backup available and not a creation`
        );
      }

      // 4. Mark as rolled back
      evolution.rolled_back = true;
      evolution.rollback_reason = 'Manual rollback';

      // 5. Log rollback
      this.observability.log('info', 'evolution_rolled_back', {
        evolution_id: evolutionId,
        slice_id: evolution.slice_id,
      });

      span.setTag('success', true);
    } catch (error) {
      span.setTag('success', false);
      span.setTag('error', (error as Error).message);
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Get evolution history
   */
  getEvolutionHistory(): SliceEvolution[] {
    return [...this.evolutions]; // Return copy
  }

  /**
   * Get evolution metrics
   */
  getMetrics(): EvolutionMetrics {
    const byType: Record<EvolutionType, number> = {
      [EvolutionType.CREATED]: 0,
      [EvolutionType.UPDATED]: 0,
      [EvolutionType.MERGED]: 0,
      [EvolutionType.DEPRECATED]: 0,
    };

    const byTrigger: Record<EvolutionTrigger, number> = {
      [EvolutionTrigger.SCHEDULED]: 0,
      [EvolutionTrigger.THRESHOLD]: 0,
      [EvolutionTrigger.MANUAL]: 0,
      [EvolutionTrigger.CONTINUOUS]: 0,
    };

    let totalConstitutionalScore = 0;
    let successfulDeployments = 0;
    let rollbacks = 0;

    let slicesCreated = 0;
    let slicesUpdated = 0;
    let slicesMerged = 0;
    let slicesDeprecated = 0;

    for (const evolution of this.evolutions) {
      // Count by type
      byType[evolution.evolution_type]++;

      // Count by trigger
      byTrigger[evolution.trigger]++;

      // Constitutional score
      totalConstitutionalScore +=
        evolution.candidate.constitutional_score;

      // Successful deployments
      if (evolution.approved && evolution.deployed_at) {
        successfulDeployments++;
      }

      // Rollbacks
      if (evolution.rolled_back) {
        rollbacks++;
      }

      // Knowledge growth
      if (evolution.evolution_type === EvolutionType.CREATED) {
        slicesCreated++;
      } else if (evolution.evolution_type === EvolutionType.UPDATED) {
        slicesUpdated++;
      } else if (evolution.evolution_type === EvolutionType.MERGED) {
        slicesMerged++;
      } else if (evolution.evolution_type === EvolutionType.DEPRECATED) {
        slicesDeprecated++;
      }
    }

    return {
      total_evolutions: this.evolutions.length,
      successful_deployments: successfulDeployments,
      rollbacks,
      by_type: byType,
      by_trigger: byTrigger,
      avg_constitutional_score:
        this.evolutions.length > 0
          ? totalConstitutionalScore / this.evolutions.length
          : 0,
      knowledge_growth: {
        slices_created: slicesCreated,
        slices_updated: slicesUpdated,
        slices_merged: slicesMerged,
        slices_deprecated: slicesDeprecated,
      },
    };
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  /**
   * Create a candidate from a knowledge pattern
   */
  private async createCandidateFromPattern(
    pattern: KnowledgePattern
  ): Promise<SliceCandidate> {
    // 1. Generate slice ID from concepts
    const sliceId = pattern.concepts.join('-').replace(/[^a-z0-9-]/gi, '-');

    // 2. Synthesize content using KnowledgeDistillation
    const content = await this.knowledgeDistillation.synthesize(pattern);

    // 3. Check constitutional compliance
    const constitutionalScore = this.evaluateConstitutionalCompliance(content);

    // 4. Determine if should deploy
    const shouldDeploy = constitutionalScore >= 0.7 && pattern.confidence >= 0.7;

    // 5. Get supporting episodes
    const supportingEpisodes = this.findSupportingEpisodes(pattern);

    return {
      id: sliceId,
      type: this.sliceRewriter.exists(sliceId) ? 'update' : 'new',
      title: `${pattern.concepts.join(', ')} Knowledge`,
      description: `Knowledge about ${pattern.concepts.join(', ')} from ${pattern.frequency} episodes`,
      concepts: pattern.concepts,
      content,
      supporting_episodes: supportingEpisodes,
      pattern,
      constitutional_score: constitutionalScore,
      test_performance: {
        queries_tested: 0,
        accuracy_improvement: 0,
        cost_delta: 0,
      },
      should_deploy: shouldDeploy,
      reasoning: `Pattern found ${pattern.frequency} times with ${(pattern.confidence * 100).toFixed(1)}% confidence`,
    };
  }

  /**
   * Evaluate constitutional compliance
   */
  private evaluateConstitutionalCompliance(content: string): number {
    // Simple heuristic: check if content looks valid and safe
    // In production, would use ConstitutionEnforcer
    if (content.length < 20) return 0.3;
    if (!content.includes('id:')) return 0.4;
    if (!content.includes('title:')) return 0.5;

    // Basic validation passed
    return 0.9;
  }

  /**
   * Find episodes that support a pattern
   */
  private findSupportingEpisodes(pattern: KnowledgePattern): string[] {
    const episodes = this.episodicMemory.query({});
    const supporting: string[] = [];

    for (const episode of episodes) {
      // Check if episode contains pattern concepts
      const hasAllConcepts = pattern.concepts.every((concept) =>
        episode.concepts.includes(concept)
      );

      if (hasAllConcepts) {
        supporting.push(episode.id);
      }

      // Limit to first 10 supporting episodes
      if (supporting.length >= 10) break;
    }

    return supporting;
  }

  /**
   * Map candidate type to evolution type
   */
  private mapCandidateTypeToEvolutionType(
    candidateType: string
  ): EvolutionType {
    switch (candidateType) {
      case 'new':
        return EvolutionType.CREATED;
      case 'update':
        return EvolutionType.UPDATED;
      case 'merge':
        return EvolutionType.MERGED;
      case 'deprecate':
        return EvolutionType.DEPRECATED;
      default:
        return EvolutionType.CREATED;
    }
  }
}

/**
 * Create a new SliceEvolutionEngine instance
 */
export function createSliceEvolutionEngine(
  episodicMemory: EpisodicMemory,
  knowledgeDistillation: KnowledgeDistillation,
  sliceRewriter: SliceRewriter,
  sliceNavigator: SliceNavigator,
  constitutionEnforcer: ConstitutionEnforcer,
  observability: Observability
): SliceEvolutionEngine {
  return new SliceEvolutionEngine(
    episodicMemory,
    knowledgeDistillation,
    sliceRewriter,
    sliceNavigator,
    constitutionEnforcer,
    observability
  );
}
