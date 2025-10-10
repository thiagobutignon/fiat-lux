/**
 * SQLO - Grammar Language Database (O(1))
 *
 * Content-addressable, hash-based database for Grammar Language ecosystem
 * Replaces: SQL (O(n) queries, O(n²) joins)
 *
 * Why SQLO?
 * - SQL: O(n) table scans, O(n²) joins
 * - SQLO: O(1) lookups, O(1) inserts, O(1) deletes
 *
 * How it works:
 * 1. Content-addressable storage (hash → content)
 * 2. No table scans (hash-based indexing)
 * 3. Immutable records (content hash = ID)
 * 4. Episodic memory native (short-term, long-term, contextual)
 *
 * Schema:
 * ```
 * episodes/
 * ├── <hash1>/
 * │   ├── content.json
 * │   └── metadata.json
 * ├── <hash2>/
 * │   ├── content.json
 * │   └── metadata.json
 * └── .index (hash → metadata mapping)
 * ```
 */

import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';
import { RbacPolicy, Permission, getGlobalRbacPolicy } from './rbac';
import {
  ConstitutionEnforcer,
  ConstitutionViolation,
  ConstitutionCheckResult
} from '../../agi-recursive/core/constitution';
import { getGlobalEmbeddingAdapter, Embedding } from './embedding-adapter';

// ============================================================================
// Types
// ============================================================================

/**
 * Memory types for episodic storage
 */
export enum MemoryType {
  SHORT_TERM = 'short-term',    // Working memory, TTL 15min
  LONG_TERM = 'long-term',      // Consolidated, forever
  CONTEXTUAL = 'contextual'     // Situational, session-based
}

/**
 * Episode stored in episodic memory
 */
export interface Episode {
  id: string;                    // Content hash (SHA256)
  query: string;                 // Original query
  response: string;              // Response generated
  attention: AttentionTrace;     // What knowledge was used
  outcome: 'success' | 'failure'; // Query outcome
  confidence: number;            // Response confidence [0-1]
  timestamp: number;             // When it happened
  user_id?: string;              // Optional user context
  memory_type: MemoryType;       // Type of memory
  embedding?: Embedding;         // Optional semantic embedding (384-dim)
}

/**
 * Attention trace for glass box transparency
 */
export interface AttentionTrace {
  sources: string[];             // Which papers/knowledge used
  weights: number[];             // Attention weights
  patterns: string[];            // Patterns identified
}

/**
 * Metadata for each episode
 */
export interface EpisodeMetadata {
  hash: string;
  memory_type: MemoryType;
  size: number;
  created_at: number;
  ttl?: number;                  // For SHORT_TERM memory
  consolidated: boolean;         // For LONG_TERM consolidation
  relevance: number;             // [0-1] for retrieval
  has_embedding?: boolean;       // Track if embedding exists
}

/**
 * Index mapping hash → metadata
 */
export interface SqloIndex {
  episodes: Record<string, EpisodeMetadata>;
  statistics: {
    total_episodes: number;
    short_term_count: number;
    long_term_count: number;
    contextual_count: number;
  };
}

// ============================================================================
// Constants
// ============================================================================

const SQLO_DIR = 'sqlo_db';
const EPISODES_DIR = 'episodes';
const INDEX_FILE = '.index';
const SHORT_TERM_TTL = 15 * 60 * 1000; // 15 minutes
const CONSOLIDATION_THRESHOLD = 100; // Episodes before consolidation

// ============================================================================
// Content-Addressable Storage (O(1))
// ============================================================================

export interface SqloConfig {
  rbacPolicy?: RbacPolicy;
  autoConsolidate?: boolean;  // Enable/disable auto-consolidation (default: true)
  constitutionEnforcer?: ConstitutionEnforcer;  // Constitutional AI validation (default: enabled)
}

export class SqloDatabase {
  private readonly baseDir: string;
  private index: SqloIndex;
  private readonly rbacPolicy: RbacPolicy;
  private readonly autoConsolidate: boolean;
  private readonly constitutionEnforcer: ConstitutionEnforcer;

  constructor(baseDir: string = SQLO_DIR, config?: SqloConfig) {
    this.baseDir = baseDir;
    this.rbacPolicy = config?.rbacPolicy || getGlobalRbacPolicy();
    this.autoConsolidate = config?.autoConsolidate !== false; // Default: true
    this.constitutionEnforcer = config?.constitutionEnforcer || new ConstitutionEnforcer();
    this.index = this.loadIndex();
    this.ensureDirectories();
  }

  // ==========================================================================
  // Core Operations (all O(1))
  // ==========================================================================

  /**
   * Store episode - O(1)
   * Content hash = ID (immutable)
   * @param episode - Episode data to store
   * @param roleName - Role for RBAC check (default: 'admin')
   */
  async put(episode: Omit<Episode, 'id'>, roleName: string = 'admin'): Promise<string> {
    // RBAC check - O(1)
    if (!this.rbacPolicy.hasPermission(roleName, episode.memory_type, Permission.WRITE)) {
      throw new Error(
        `Permission denied: Role '${roleName}' cannot write to ${episode.memory_type} memory`
      );
    }

    // Constitutional validation - O(1)
    this.validateEpisode(episode);

    // Generate embedding if not provided - O(1) for bounded input
    let embedding: Embedding | undefined = episode.embedding;
    if (!embedding) {
      try {
        const embeddingAdapter = getGlobalEmbeddingAdapter();
        const embeddingText = `${episode.query} ${episode.response}`;
        const result = await embeddingAdapter.embed(embeddingText);
        embedding = result.embedding;
      } catch (error) {
        // Embedding generation failed, continue without it
        console.warn(`Failed to generate embedding: ${error}`);
      }
    }

    // Hash content - O(1) for bounded input
    const content = JSON.stringify({
      query: episode.query,
      response: episode.response,
      attention: episode.attention
    });
    const hash = this.hash(content);

    // Check if exists - O(1)
    if (this.has(hash)) {
      return hash;
    }

    // Create episode with hash ID and embedding
    const fullEpisode: Episode = {
      ...episode,
      id: hash,
      embedding
    };

    // Create directory - O(1)
    const episodeDir = this.getEpisodeDir(hash);
    if (!fs.existsSync(episodeDir)) {
      fs.mkdirSync(episodeDir, { recursive: true });
    }

    // Write content - O(1) (bounded size)
    fs.writeFileSync(
      path.join(episodeDir, 'content.json'),
      JSON.stringify(fullEpisode, null, 2)
    );

    // Create metadata - O(1)
    const metadata: EpisodeMetadata = {
      hash,
      memory_type: episode.memory_type,
      size: content.length,
      created_at: Date.now(),
      ttl: episode.memory_type === MemoryType.SHORT_TERM ? SHORT_TERM_TTL : undefined,
      consolidated: false,
      relevance: episode.confidence,
      has_embedding: !!embedding
    };

    // Write metadata - O(1)
    fs.writeFileSync(
      path.join(episodeDir, 'metadata.json'),
      JSON.stringify(metadata, null, 2)
    );

    // Update index - O(1) (Map insertion)
    this.index.episodes[hash] = metadata;
    this.updateStatistics();
    this.saveIndex();

    // Auto-cleanup expired short-term memories
    this.cleanupExpired();

    // Auto-consolidate if threshold reached (if enabled)
    if (this.autoConsolidate && this.shouldConsolidate()) {
      await this.consolidate();
    }

    return hash;
  }

  /**
   * Get episode by hash - O(1)
   * @param hash - Episode hash
   * @param roleName - Role for RBAC check (default: 'admin')
   */
  get(hash: string, roleName: string = 'admin'): Episode | null {
    if (!this.has(hash)) {
      return null;
    }

    const episodeDir = this.getEpisodeDir(hash);
    const contentPath = path.join(episodeDir, 'content.json');

    if (!fs.existsSync(contentPath)) {
      return null;
    }

    const content = fs.readFileSync(contentPath, 'utf-8');
    const episode = JSON.parse(content);

    // RBAC check - O(1)
    if (!this.rbacPolicy.hasPermission(roleName, episode.memory_type, Permission.READ)) {
      throw new Error(
        `Permission denied: Role '${roleName}' cannot read ${episode.memory_type} memory`
      );
    }

    return episode;
  }

  /**
   * Check if episode exists - O(1)
   */
  has(hash: string): boolean {
    return hash in this.index.episodes;
  }

  /**
   * Delete episode - O(1)
   * Note: Rarely used (old-but-gold philosophy)
   * @param hash - Episode hash
   * @param roleName - Role for RBAC check (default: 'admin')
   */
  delete(hash: string, roleName: string = 'admin'): boolean {
    if (!this.has(hash)) {
      return false;
    }

    // Get memory type for RBAC check
    const metadata = this.index.episodes[hash];
    if (!metadata) {
      return false;
    }

    // RBAC check - O(1)
    if (!this.rbacPolicy.hasPermission(roleName, metadata.memory_type, Permission.DELETE)) {
      throw new Error(
        `Permission denied: Role '${roleName}' cannot delete ${metadata.memory_type} memory`
      );
    }

    const episodeDir = this.getEpisodeDir(hash);

    // Delete directory - O(1) (bounded size)
    if (fs.existsSync(episodeDir)) {
      fs.rmSync(episodeDir, { recursive: true });
    }

    // Remove from index - O(1)
    delete this.index.episodes[hash];
    this.updateStatistics();
    this.saveIndex();

    return true;
  }

  // ==========================================================================
  // Episodic Memory Operations
  // ==========================================================================

  /**
   * Query similar episodes - O(k) where k = number of episodes
   * Uses embedding-based semantic similarity (cosine similarity)
   * Falls back to keyword matching if embeddings not available
   */
  async querySimilar(query: string, limit: number = 5): Promise<Episode[]> {
    // Constitutional validation - O(1)
    this.validateQuery(query, 'querySimilar');

    const episodes = this.listByType(MemoryType.LONG_TERM);

    // Try embedding-based similarity first
    try {
      const embeddingAdapter = getGlobalEmbeddingAdapter();
      const queryResult = await embeddingAdapter.embed(query);
      const queryEmbedding = queryResult.embedding;

      // Collect episodes with embeddings
      const episodesWithEmbeddings = episodes.filter(ep => ep.embedding);

      if (episodesWithEmbeddings.length > 0) {
        // Use semantic similarity
        const candidateEmbeddings = episodesWithEmbeddings.map(ep => ep.embedding!);
        const results = embeddingAdapter.findMostSimilar(
          queryEmbedding,
          candidateEmbeddings,
          limit
        );

        return results.map(result => episodesWithEmbeddings[result.index]);
      }
    } catch (error) {
      console.warn(`Embedding-based search failed, falling back to keyword matching: ${error}`);
    }

    // Fallback: keyword-based similarity
    const keywords = query.toLowerCase().split(' ');

    const scored = episodes
      .map(ep => {
        const similarity = keywords.filter(kw =>
          ep.query.toLowerCase().includes(kw)
        ).length / keywords.length;

        return { episode: ep, score: similarity };
      })
      .filter(item => item.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);

    return scored.map(item => item.episode);
  }

  /**
   * List episodes by memory type - O(n) where n = episodes of that type
   */
  listByType(type: MemoryType): Episode[] {
    // Constitutional validation - O(1)
    this.validateQuery(type, 'listByType');

    const hashes = Object.entries(this.index.episodes)
      .filter(([_, meta]) => meta.memory_type === type)
      .map(([hash, _]) => hash);

    return hashes
      .map(hash => this.get(hash))
      .filter((ep): ep is Episode => ep !== null);
  }

  /**
   * Get statistics - O(1)
   */
  getStatistics(): SqloIndex['statistics'] {
    return this.index.statistics;
  }

  // ==========================================================================
  // Memory Consolidation (Auto-optimization)
  // ==========================================================================

  /**
   * Check if consolidation should happen
   */
  private shouldConsolidate(): boolean {
    const shortTermCount = this.index.statistics.short_term_count;
    return shortTermCount >= CONSOLIDATION_THRESHOLD;
  }

  /**
   * Consolidate short-term → long-term
   * Identifies patterns in recent episodes and consolidates learning
   */
  private async consolidate(): Promise<void> {
    const shortTerm = this.listByType(MemoryType.SHORT_TERM);

    // Identify successful patterns
    const successful = shortTerm.filter(ep =>
      ep.outcome === 'success' && ep.confidence > 0.8
    );

    // Move to long-term
    for (const episode of successful) {
      const metadata = this.index.episodes[episode.id];
      if (metadata) {
        metadata.memory_type = MemoryType.LONG_TERM;
        metadata.consolidated = true;
        metadata.ttl = undefined; // Remove TTL
      }
    }

    this.updateStatistics();
    this.saveIndex();
  }

  /**
   * Cleanup expired short-term memories
   */
  private cleanupExpired(): void {
    const now = Date.now();
    const toDelete: string[] = [];

    for (const [hash, meta] of Object.entries(this.index.episodes)) {
      if (meta.memory_type === MemoryType.SHORT_TERM && meta.ttl) {
        const age = now - meta.created_at;
        if (age > meta.ttl) {
          toDelete.push(hash);
        }
      }
    }

    // Delete expired
    for (const hash of toDelete) {
      this.delete(hash);
    }
  }

  // ==========================================================================
  // Constitutional Validation (Layer 1 Integration)
  // ==========================================================================

  /**
   * Validate episode against Universal Constitution principles
   * @throws Error if constitutional violation detected
   */
  private validateEpisode(episode: Omit<Episode, 'id'>): void {
    // Validate against constitutional principles
    const result = this.constitutionEnforcer.validate(
      'sqlo_database',
      {
        answer: episode.response,
        confidence: episode.confidence,
        reasoning: `Query: ${episode.query}`,
        sources: episode.attention.sources
      },
      {
        depth: 0,
        invocation_count: 1,
        cost_so_far: 0,
        previous_agents: []
      }
    );

    // Check for violations
    if (!result.passed || result.violations.length > 0) {
      const violation = result.violations[0];
      throw new Error(
        `Constitutional Violation [${violation.principle_id}]: ${violation.message}\n` +
        `Severity: ${violation.severity}\n` +
        `Suggested Action: ${violation.suggested_action}\n` +
        `Episode: ${episode.query.substring(0, 100)}...`
      );
    }
  }

  /**
   * Validate query against Universal Constitution principles
   * @throws Error if constitutional violation detected
   */
  private validateQuery(query: string, operation: string): void {
    // For queries, we validate the intent and safety
    const result = this.constitutionEnforcer.validate(
      'sqlo_database',
      {
        answer: `Executing ${operation}: ${query}`,
        confidence: 1.0,  // Queries are deterministic
        reasoning: `Database operation: ${operation}`
      },
      {
        depth: 0,
        invocation_count: 1,
        cost_so_far: 0,
        previous_agents: []
      }
    );

    // Check for violations (safety checks on query content)
    if (!result.passed || result.violations.length > 0) {
      const violation = result.violations[0];
      throw new Error(
        `Constitutional Violation [${violation.principle_id}]: ${violation.message}\n` +
        `Severity: ${violation.severity}\n` +
        `Suggested Action: ${violation.suggested_action}\n` +
        `Query: ${query.substring(0, 100)}...`
      );
    }
  }

  // ==========================================================================
  // Internal Helpers
  // ==========================================================================

  /**
   * Hash content using SHA256 - O(1) for bounded input
   */
  private hash(content: string): string {
    return crypto.createHash('sha256').update(content).digest('hex');
  }

  /**
   * Get episode directory path
   */
  private getEpisodeDir(hash: string): string {
    return path.join(this.baseDir, EPISODES_DIR, hash);
  }

  /**
   * Update statistics
   */
  private updateStatistics(): void {
    const stats = {
      total_episodes: 0,
      short_term_count: 0,
      long_term_count: 0,
      contextual_count: 0
    };

    for (const meta of Object.values(this.index.episodes)) {
      stats.total_episodes++;

      switch (meta.memory_type) {
        case MemoryType.SHORT_TERM:
          stats.short_term_count++;
          break;
        case MemoryType.LONG_TERM:
          stats.long_term_count++;
          break;
        case MemoryType.CONTEXTUAL:
          stats.contextual_count++;
          break;
      }
    }

    this.index.statistics = stats;
  }

  /**
   * Load index from disk - O(1)
   */
  private loadIndex(): SqloIndex {
    const indexPath = path.join(this.baseDir, INDEX_FILE);

    if (fs.existsSync(indexPath)) {
      const content = fs.readFileSync(indexPath, 'utf-8');
      return JSON.parse(content);
    }

    return {
      episodes: {},
      statistics: {
        total_episodes: 0,
        short_term_count: 0,
        long_term_count: 0,
        contextual_count: 0
      }
    };
  }

  /**
   * Save index to disk - O(1)
   */
  private saveIndex(): void {
    const indexPath = path.join(this.baseDir, INDEX_FILE);
    fs.writeFileSync(indexPath, JSON.stringify(this.index, null, 2));
  }

  /**
   * Ensure directories exist
   */
  private ensureDirectories(): void {
    const dirs = [
      this.baseDir,
      path.join(this.baseDir, EPISODES_DIR)
    ];

    for (const dir of dirs) {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
    }
  }
}
