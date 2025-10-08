/**
 * Advanced Caching System for Meta-Agent
 *
 * Implements multiple caching strategies to achieve >95% cache hit rate:
 * 1. Semantic Similarity Caching - Match similar queries
 * 2. Query Normalization - Standardize query format
 * 3. LRU Eviction - Keep most popular queries
 * 4. Query Templates - Abstract variable parameters
 * 5. Pre-warming - Populate with common queries
 */

import crypto from 'crypto';
import { DeduplicationTracker } from './o1-advanced-optimizer.js';

// ============================================================================
// Types
// ============================================================================

export interface CacheEntry<T> {
  key: string;
  value: T;
  timestamp: number;
  hitCount: number;
  lastAccessed: number;
  similarity?: number; // For fuzzy matches
  queryNormalized: string;
  queryOriginal: string;
}

export interface CacheConfig {
  maxSize: number;
  ttlMs: number;
  similarityThreshold: number; // 0-1, default 0.85
  enableSemanticCache: boolean;
  enableTemplateCache: boolean;
  enableLRU: boolean;
}

export interface CacheStats {
  size: number;
  maxSize: number;
  hits: number;
  misses: number;
  hitRate: number;
  semanticHits: number;
  exactHits: number;
  templateHits: number;
  evictions: number;
  avgHitCount: number;
  popularQueries: Array<{ query: string; hits: number }>;
}

// ============================================================================
// Advanced Query Normalizer
// ============================================================================

export class QueryNormalizer {
  /**
   * Normalize query to maximize cache hits
   */
  static normalize(query: string): string {
    let normalized = query.toLowerCase().trim();

    // Remove extra whitespace
    normalized = normalized.replace(/\s+/g, ' ');

    // Remove common punctuation variations
    normalized = normalized.replace(/[?.!,;:]/g, '');

    // Normalize contractions
    normalized = normalized
      .replace(/what's/g, 'what is')
      .replace(/who's/g, 'who is')
      .replace(/it's/g, 'it is')
      .replace(/don't/g, 'do not')
      .replace(/can't/g, 'cannot')
      .replace(/won't/g, 'will not');

    // Remove articles for better matching
    normalized = normalized.replace(/\b(a|an|the)\b/g, '');

    // Remove filler words
    const fillers = ['please', 'could you', 'can you', 'would you'];
    fillers.forEach((filler) => {
      normalized = normalized.replace(new RegExp(`\\b${filler}\\b`, 'g'), '');
    });

    // Normalize spacing again
    normalized = normalized.replace(/\s+/g, ' ').trim();

    return normalized;
  }

  /**
   * Extract query template (replace numbers, names, dates)
   */
  static extractTemplate(query: string): string {
    let template = this.normalize(query);

    // Replace numbers with placeholder
    template = template.replace(/\b\d+(\.\d+)?\b/g, '<NUM>');

    // Replace common named entities (simple heuristic)
    template = template.replace(/\b[A-Z][a-z]+\b/g, '<NAME>');

    // Replace dates
    template = template.replace(
      /\b\d{1,2}\/\d{1,2}\/\d{2,4}\b/g,
      '<DATE>'
    );

    // Replace amounts
    template = template.replace(/\$\d+(\.\d+)?/g, '<AMOUNT>');

    return template;
  }

  /**
   * Calculate Jaccard similarity between two queries
   */
  static calculateSimilarity(query1: string, query2: string): number {
    const words1 = new Set(this.normalize(query1).split(/\s+/));
    const words2 = new Set(this.normalize(query2).split(/\s+/));

    const intersection = new Set([...words1].filter((w) => words2.has(w)));
    const union = new Set([...words1, ...words2]);

    if (union.size === 0) return 0;

    return intersection.size / union.size;
  }

  /**
   * Calculate semantic hash for fuzzy matching
   */
  static semanticHash(query: string): string {
    const normalized = this.normalize(query);
    const words = normalized.split(/\s+/).sort(); // Sort for order-independence

    // Create hash from sorted words
    return crypto
      .createHash('md5')
      .update(words.join('|'))
      .digest('hex')
      .substring(0, 16);
  }
}

// ============================================================================
// LRU Cache with Semantic Matching
// ============================================================================

export class AdvancedCache<T> {
  private cache: Map<string, CacheEntry<T>> = new Map();
  private config: CacheConfig;
  private hits = 0;
  private misses = 0;
  private semanticHits = 0;
  private exactHits = 0;
  private templateHits = 0;
  private evictions = 0;

  // Indexes for fast lookup
  private templateIndex: Map<string, Set<string>> = new Map(); // template -> keys
  private semanticIndex: Map<string, Set<string>> = new Map(); // semantic hash -> keys

  constructor(config?: Partial<CacheConfig>) {
    this.config = {
      maxSize: config?.maxSize ?? 10000,
      ttlMs: config?.ttlMs ?? 3600000, // 1 hour
      similarityThreshold: config?.similarityThreshold ?? 0.85,
      enableSemanticCache: config?.enableSemanticCache ?? true,
      enableTemplateCache: config?.enableTemplateCache ?? true,
      enableLRU: config?.enableLRU ?? true,
    };
  }

  /**
   * Get value from cache with multiple matching strategies
   */
  get(query: string): T | null {
    const normalized = QueryNormalizer.normalize(query);
    const now = Date.now();

    // Strategy 1: Exact normalized match (fastest)
    const exactKey = this.getExactKey(normalized);
    if (exactKey) {
      const entry = this.cache.get(exactKey)!;

      // Check TTL
      if (now - entry.timestamp > this.config.ttlMs) {
        this.remove(exactKey);
        this.misses++;
        return null;
      }

      // Update access time and hit count
      entry.lastAccessed = now;
      entry.hitCount++;
      this.hits++;
      this.exactHits++;

      return entry.value;
    }

    // Strategy 2: Template match (medium speed)
    if (this.config.enableTemplateCache) {
      const templateKey = this.getTemplateKey(query);
      if (templateKey) {
        const entry = this.cache.get(templateKey)!;

        // Check TTL
        if (now - entry.timestamp <= this.config.ttlMs) {
          entry.lastAccessed = now;
          entry.hitCount++;
          this.hits++;
          this.templateHits++;
          return entry.value;
        }
      }
    }

    // Strategy 3: Semantic similarity match (slower, but powerful)
    if (this.config.enableSemanticCache) {
      const semanticMatch = this.getSemanticMatch(query);
      if (semanticMatch) {
        const entry = this.cache.get(semanticMatch.key)!;

        // Check TTL
        if (now - entry.timestamp <= this.config.ttlMs) {
          entry.lastAccessed = now;
          entry.hitCount++;
          entry.similarity = semanticMatch.similarity;
          this.hits++;
          this.semanticHits++;
          return entry.value;
        }
      }
    }

    this.misses++;
    return null;
  }

  /**
   * Set value in cache with eviction if needed
   */
  set(query: string, value: T): void {
    const normalized = QueryNormalizer.normalize(query);
    const key = crypto.createHash('sha256').update(normalized).digest('hex');

    // Check if we need to evict
    if (this.cache.size >= this.config.maxSize && !this.cache.has(key)) {
      this.evictLRU();
    }

    const entry: CacheEntry<T> = {
      key,
      value,
      timestamp: Date.now(),
      hitCount: 0,
      lastAccessed: Date.now(),
      queryNormalized: normalized,
      queryOriginal: query,
    };

    this.cache.set(key, entry);

    // Build indexes
    this.updateIndexes(key, query, normalized);
  }

  /**
   * Pre-warm cache with common queries
   */
  async preWarm(queries: Array<{ query: string; value: T }>): Promise<number> {
    let warmed = 0;

    for (const { query, value } of queries) {
      this.set(query, value);
      warmed++;
    }

    return warmed;
  }

  /**
   * Get cache statistics
   */
  getStats(): CacheStats {
    const totalQueries = this.hits + this.misses;
    const hitRate = totalQueries > 0 ? this.hits / totalQueries : 0;

    // Calculate average hit count
    let totalHitCount = 0;
    for (const entry of this.cache.values()) {
      totalHitCount += entry.hitCount;
    }
    const avgHitCount = this.cache.size > 0 ? totalHitCount / this.cache.size : 0;

    // Get most popular queries
    const popularQueries = Array.from(this.cache.values())
      .sort((a, b) => b.hitCount - a.hitCount)
      .slice(0, 10)
      .map((entry) => ({
        query: entry.queryOriginal,
        hits: entry.hitCount,
      }));

    return {
      size: this.cache.size,
      maxSize: this.config.maxSize,
      hits: this.hits,
      misses: this.misses,
      hitRate,
      semanticHits: this.semanticHits,
      exactHits: this.exactHits,
      templateHits: this.templateHits,
      evictions: this.evictions,
      avgHitCount,
      popularQueries,
    };
  }

  /**
   * Clear cache
   */
  clear(): void {
    this.cache.clear();
    this.templateIndex.clear();
    this.semanticIndex.clear();
    this.hits = 0;
    this.misses = 0;
    this.semanticHits = 0;
    this.exactHits = 0;
    this.templateHits = 0;
    this.evictions = 0;
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<CacheConfig>): void {
    this.config = { ...this.config, ...config };
  }

  // =========================================================================
  // Private Methods
  // =========================================================================

  private getExactKey(normalized: string): string | null {
    const key = crypto.createHash('sha256').update(normalized).digest('hex');
    return this.cache.has(key) ? key : null;
  }

  private getTemplateKey(query: string): string | null {
    const template = QueryNormalizer.extractTemplate(query);
    const candidates = this.templateIndex.get(template);

    if (!candidates || candidates.size === 0) return null;

    // Return most recently accessed
    let bestKey: string | null = null;
    let bestTime = 0;

    for (const key of candidates) {
      const entry = this.cache.get(key);
      if (entry && entry.lastAccessed > bestTime) {
        bestTime = entry.lastAccessed;
        bestKey = key;
      }
    }

    return bestKey;
  }

  private getSemanticMatch(
    query: string
  ): { key: string; similarity: number } | null {
    const semanticHash = QueryNormalizer.semanticHash(query);
    const candidates = this.semanticIndex.get(semanticHash);

    if (!candidates || candidates.size === 0) {
      // Try finding similar queries by checking all entries (expensive, but rare)
      return this.findMostSimilar(query);
    }

    // Check candidates for best match
    let bestMatch: { key: string; similarity: number } | null = null;

    for (const key of candidates) {
      const entry = this.cache.get(key);
      if (!entry) continue;

      const similarity = QueryNormalizer.calculateSimilarity(
        query,
        entry.queryOriginal
      );

      if (
        similarity >= this.config.similarityThreshold &&
        (!bestMatch || similarity > bestMatch.similarity)
      ) {
        bestMatch = { key, similarity };
      }
    }

    return bestMatch;
  }

  private findMostSimilar(
    query: string
  ): { key: string; similarity: number } | null {
    let bestMatch: { key: string; similarity: number } | null = null;

    // Only check a sample for performance (max 100 entries)
    const entries = Array.from(this.cache.entries()).slice(0, 100);

    for (const [key, entry] of entries) {
      const similarity = QueryNormalizer.calculateSimilarity(
        query,
        entry.queryOriginal
      );

      if (
        similarity >= this.config.similarityThreshold &&
        (!bestMatch || similarity > bestMatch.similarity)
      ) {
        bestMatch = { key, similarity };
      }
    }

    return bestMatch;
  }

  private updateIndexes(key: string, original: string, normalized: string): void {
    // Template index
    const template = QueryNormalizer.extractTemplate(original);
    if (!this.templateIndex.has(template)) {
      this.templateIndex.set(template, new Set());
    }
    this.templateIndex.get(template)!.add(key);

    // Semantic index
    const semanticHash = QueryNormalizer.semanticHash(original);
    if (!this.semanticIndex.has(semanticHash)) {
      this.semanticIndex.set(semanticHash, new Set());
    }
    this.semanticIndex.get(semanticHash)!.add(key);
  }

  private evictLRU(): void {
    if (this.cache.size === 0) return;

    // Find least recently used entry
    let lruKey: string | null = null;
    let lruTime = Infinity;

    for (const [key, entry] of this.cache.entries()) {
      if (entry.lastAccessed < lruTime) {
        lruTime = entry.lastAccessed;
        lruKey = key;
      }
    }

    if (lruKey) {
      this.remove(lruKey);
      this.evictions++;
    }
  }

  private remove(key: string): void {
    const entry = this.cache.get(key);
    if (!entry) return;

    // Remove from indexes
    const template = QueryNormalizer.extractTemplate(entry.queryOriginal);
    this.templateIndex.get(template)?.delete(key);

    const semanticHash = QueryNormalizer.semanticHash(entry.queryOriginal);
    this.semanticIndex.get(semanticHash)?.delete(key);

    // Remove from cache
    this.cache.delete(key);
  }
}

// ============================================================================
// Export utilities
// ============================================================================

export { QueryNormalizer };
