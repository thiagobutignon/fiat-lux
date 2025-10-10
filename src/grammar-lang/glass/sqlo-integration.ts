/**
 * SQLO + Glass Integration
 *
 * Integrates SqloDatabase episodic memory into .glass organisms
 * Enables:
 * - Memory embedded in .glass file
 * - Episodic learning during organism lifecycle
 * - Glass box memory inspection
 * - O(1) memory operations within organism
 *
 * Sprint 2 - DIA 1: Integration Day
 */

import { SqloDatabase, MemoryType, Episode, AttentionTrace } from '../database/sqlo';
import { RbacPolicy, Permission } from '../database/rbac';
import * as fs from 'fs';
import * as path from 'path';

// ============================================================================
// Types
// ============================================================================

/**
 * Glass organism with embedded memory
 */
export interface GlassOrganism {
  metadata: {
    name: string;
    version: string;
    domain: string;
    maturity: number; // 0-100%
    stage: 'nascent' | 'infant' | 'adolescent' | 'mature' | 'evolving' | 'retired';
    created_at: number;
    updated_at: number;
  };

  model: {
    architecture: string;
    parameters: number;
    quantization: string;
  };

  knowledge: {
    papers: number;
    embeddings: number;
    patterns: number;
  };

  code: {
    emerged_functions: string[]; // Functions that emerged from patterns
    confidence: number;
  };

  memory: {
    database_path: string; // Path to embedded .sqlo database
    total_episodes: number;
    short_term_count: number;
    long_term_count: number;
    contextual_count: number;
  };

  constitutional: {
    principles: string[];
    boundaries: string[];
    validators: string[];
  };

  evolution: {
    enabled: boolean;
    generation: number;
    fitness: number;
    parent_id?: string;
  };
}

/**
 * Learning interaction (query + response + outcome)
 */
export interface LearningInteraction {
  query: string;
  response: string;
  confidence: number;
  sources: string[];
  attention_weights: number[];
  outcome: 'success' | 'failure';
  user_id?: string;
}

// ============================================================================
// Glass + SQLO Integration
// ============================================================================

export class GlassMemorySystem {
  private organism: GlassOrganism;
  private database: SqloDatabase;
  private glassPath: string;

  constructor(glassPath: string) {
    this.glassPath = glassPath;
    this.organism = this.loadGlassOrganism(glassPath);
    this.database = this.loadOrCreateMemoryDatabase();
  }

  // ==========================================================================
  // Lifecycle Operations
  // ==========================================================================

  /**
   * Learn from interaction - stores in episodic memory
   * This is how the organism learns over time
   */
  async learn(interaction: LearningInteraction, roleName: string = 'system'): Promise<string> {
    // Create episode from interaction
    const episode: Omit<Episode, 'id'> = {
      query: interaction.query,
      response: interaction.response,
      attention: {
        sources: interaction.sources,
        weights: interaction.attention_weights,
        patterns: this.extractPatterns(interaction)
      },
      outcome: interaction.outcome,
      confidence: interaction.confidence,
      timestamp: Date.now(),
      user_id: interaction.user_id,
      memory_type: this.determineMemoryType(interaction)
    };

    // Store in memory database
    const episodeHash = await this.database.put(episode, roleName);

    // Update organism metadata
    this.updateMemoryStats();
    this.updateMaturity(interaction);
    this.saveGlassOrganism();

    return episodeHash;
  }

  /**
   * Recall similar experiences from memory
   * Uses episodic memory to inform responses
   */
  async recallSimilar(query: string, limit: number = 5): Promise<Episode[]> {
    return await this.database.querySimilar(query, limit);
  }

  /**
   * Get all episodes of a specific memory type
   */
  getMemory(memoryType: MemoryType): Episode[] {
    return this.database.listByType(memoryType);
  }

  /**
   * Get organism's current memory statistics
   */
  getMemoryStats() {
    return {
      ...this.database.getStatistics(),
      maturity: this.organism.metadata.maturity,
      stage: this.organism.metadata.stage
    };
  }

  /**
   * Inspect organism - full glass box view
   */
  inspect(): {
    organism: GlassOrganism;
    memory_stats: any;
    recent_learning: Episode[];
    fitness_trajectory: number[];
  } {
    const recentLearning = this.getMemory(MemoryType.SHORT_TERM).slice(0, 10);

    return {
      organism: this.organism,
      memory_stats: this.getMemoryStats(),
      recent_learning: recentLearning,
      fitness_trajectory: this.calculateFitnessTrajectory()
    };
  }

  // ==========================================================================
  // Maturity & Evolution
  // ==========================================================================

  /**
   * Update organism maturity based on learning
   * Maturity increases as organism learns successfully
   */
  private updateMaturity(interaction: LearningInteraction): void {
    const currentMaturity = this.organism.metadata.maturity;

    // Successful interactions increase maturity
    if (interaction.outcome === 'success') {
      const increment = this.calculateMaturityIncrement(interaction);
      this.organism.metadata.maturity = Math.min(100, currentMaturity + increment);
    }

    // Update stage based on maturity
    this.organism.metadata.stage = this.calculateStage(this.organism.metadata.maturity);
    this.organism.metadata.updated_at = Date.now();
  }

  /**
   * Calculate maturity increment from interaction
   * Higher confidence = more maturity gain
   */
  private calculateMaturityIncrement(interaction: LearningInteraction): number {
    const baseIncrement = 0.1; // Base: 0.1% per successful interaction
    const confidenceBonus = interaction.confidence * 0.2; // Up to 0.2% bonus for high confidence
    return baseIncrement + confidenceBonus;
  }

  /**
   * Calculate organism stage from maturity
   */
  private calculateStage(maturity: number): GlassOrganism['metadata']['stage'] {
    if (maturity === 0) return 'nascent';
    if (maturity < 25) return 'infant';
    if (maturity < 75) return 'adolescent';
    if (maturity < 100) return 'mature';
    return 'evolving';
  }

  /**
   * Calculate fitness trajectory over time
   * Shows how organism is evolving
   */
  private calculateFitnessTrajectory(): number[] {
    const longTerm = this.database.listByType(MemoryType.LONG_TERM);

    // Group by time windows and calculate average confidence
    const windows = 10;
    const windowSize = Math.ceil(longTerm.length / windows);
    const trajectory: number[] = [];

    for (let i = 0; i < windows; i++) {
      const start = i * windowSize;
      const end = Math.min((i + 1) * windowSize, longTerm.length);
      const window = longTerm.slice(start, end);

      if (window.length > 0) {
        const avgConfidence = window.reduce((sum, ep) => sum + ep.confidence, 0) / window.length;
        trajectory.push(avgConfidence);
      }
    }

    return trajectory;
  }

  // ==========================================================================
  // Memory Management
  // ==========================================================================

  /**
   * Determine memory type for new episode
   * Based on confidence and outcome
   */
  private determineMemoryType(interaction: LearningInteraction): MemoryType {
    // High confidence successful interactions go to long-term
    if (interaction.outcome === 'success' && interaction.confidence > 0.8) {
      return MemoryType.LONG_TERM;
    }

    // Failed interactions stay in short-term for learning
    if (interaction.outcome === 'failure') {
      return MemoryType.SHORT_TERM;
    }

    // Everything else is contextual
    return MemoryType.CONTEXTUAL;
  }

  /**
   * Extract patterns from interaction
   * For attention traces
   */
  private extractPatterns(interaction: LearningInteraction): string[] {
    const patterns: string[] = [];

    // Extract domain patterns
    if (interaction.sources.length > 0) {
      patterns.push(`sources:${interaction.sources.length}`);
    }

    // Extract confidence pattern
    if (interaction.confidence > 0.9) {
      patterns.push('high-confidence');
    } else if (interaction.confidence < 0.5) {
      patterns.push('low-confidence');
    }

    // Extract outcome pattern
    patterns.push(`outcome:${interaction.outcome}`);

    return patterns;
  }

  /**
   * Update memory statistics in organism
   */
  private updateMemoryStats(): void {
    const stats = this.database.getStatistics();

    this.organism.memory.total_episodes = stats.total_episodes;
    this.organism.memory.short_term_count = stats.short_term_count;
    this.organism.memory.long_term_count = stats.long_term_count;
    this.organism.memory.contextual_count = stats.contextual_count;
  }

  // ==========================================================================
  // Persistence
  // ==========================================================================

  /**
   * Load .glass organism from file
   */
  private loadGlassOrganism(glassPath: string): GlassOrganism {
    if (!fs.existsSync(glassPath)) {
      throw new Error(`Glass organism not found: ${glassPath}`);
    }

    const content = fs.readFileSync(glassPath, 'utf-8');
    return JSON.parse(content);
  }

  /**
   * Load or create embedded memory database
   */
  private loadOrCreateMemoryDatabase(): SqloDatabase {
    const memoryPath = this.organism.memory.database_path;
    const absolutePath = path.resolve(path.dirname(this.glassPath), memoryPath);

    return new SqloDatabase(absolutePath);
  }

  /**
   * Save .glass organism to file
   */
  private saveGlassOrganism(): void {
    const content = JSON.stringify(this.organism, null, 2);
    fs.writeFileSync(this.glassPath, content);
  }

  /**
   * Export organism with memory for distribution
   */
  async exportGlass(): Promise<{
    glass: GlassOrganism;
    memory_size: number;
    total_size: number;
  }> {
    // Get memory database size
    const memoryPath = this.organism.memory.database_path;
    const absolutePath = path.resolve(path.dirname(this.glassPath), memoryPath);

    let memorySize = 0;
    if (fs.existsSync(absolutePath)) {
      const stats = fs.statSync(absolutePath);
      memorySize = stats.size;
    }

    // Get glass file size
    const glassStats = fs.statSync(this.glassPath);
    const totalSize = glassStats.size + memorySize;

    return {
      glass: this.organism,
      memory_size: memorySize,
      total_size: totalSize
    };
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create new glass organism with embedded memory
 */
export async function createGlassWithMemory(
  name: string,
  domain: string,
  baseDir: string = './organisms'
): Promise<GlassMemorySystem> {
  // Ensure base directory exists
  if (!fs.existsSync(baseDir)) {
    fs.mkdirSync(baseDir, { recursive: true });
  }

  // Create organism directory
  const organismDir = path.join(baseDir, name);
  if (!fs.existsSync(organismDir)) {
    fs.mkdirSync(organismDir, { recursive: true });
  }

  // Create nascent organism
  const organism: GlassOrganism = {
    metadata: {
      name,
      version: '1.0.0',
      domain,
      maturity: 0,
      stage: 'nascent',
      created_at: Date.now(),
      updated_at: Date.now()
    },
    model: {
      architecture: 'transformer',
      parameters: 27_000_000,
      quantization: 'int8'
    },
    knowledge: {
      papers: 0,
      embeddings: 0,
      patterns: 0
    },
    code: {
      emerged_functions: [],
      confidence: 0
    },
    memory: {
      database_path: './memory',
      total_episodes: 0,
      short_term_count: 0,
      long_term_count: 0,
      contextual_count: 0
    },
    constitutional: {
      principles: ['transparency', 'honesty', 'privacy'],
      boundaries: ['no-harm', 'human-oversight'],
      validators: ['constitutional-check']
    },
    evolution: {
      enabled: true,
      generation: 0,
      fitness: 0
    }
  };

  // Save organism
  const glassPath = path.join(organismDir, `${name}.glass`);
  fs.writeFileSync(glassPath, JSON.stringify(organism, null, 2));

  // Create memory system
  return new GlassMemorySystem(glassPath);
}

/**
 * Load existing glass organism
 */
export function loadGlassWithMemory(glassPath: string): GlassMemorySystem {
  return new GlassMemorySystem(glassPath);
}
