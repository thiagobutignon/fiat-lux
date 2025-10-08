/**
 * O(1) Optimizer - Big O(1) Performance Layer
 *
 * PURPOSE:
 * Achieve constant-time performance through:
 * - Aggressive caching
 * - Memoization
 * - Lazy evaluation
 * - Indexed lookups
 * - Early stopping
 *
 * PHILOSOPHY: "O Ócio É Tudo Que Você Precisa"
 * Instead of brute force, use intelligence to avoid work.
 */

// ============================================================================
// Types
// ============================================================================

export interface CacheEntry<T> {
  value: T;
  timestamp: number;
  hits: number;
  cost: number;
}

export interface CacheStats {
  hits: number;
  misses: number;
  hitRate: number;
  savedCost: number;
  savedTime: number;
}

export interface OptimizationConfig {
  // Cache settings
  maxCacheSize: number;
  cacheTTLMs: number;
  enablePredictiveCache: boolean;

  // Lazy evaluation
  enableLazyEval: boolean;
  lazyThreshold: number; // Only compute if confidence > threshold

  // Early stopping
  enableEarlyStopping: boolean;
  confidenceThreshold: number; // Stop if answer confidence > threshold
  maxIterations: number;

  // Indexing
  enableIndexing: boolean;
  rebuildIndexInterval: number;
}

export const DEFAULT_O1_CONFIG: OptimizationConfig = {
  maxCacheSize: 1000,
  cacheTTLMs: 3600000, // 1 hour
  enablePredictiveCache: true,
  enableLazyEval: true,
  lazyThreshold: 0.7,
  enableEarlyStopping: true,
  confidenceThreshold: 0.9,
  maxIterations: 3, // Max 3 agent invocations
  enableIndexing: true,
  rebuildIndexInterval: 600000, // 10 minutes
};

// ============================================================================
// O(1) Cache - Constant Time Lookups
// ============================================================================

export class O1Cache<T> {
  private cache: Map<string, CacheEntry<T>> = new Map();
  private stats = {
    hits: 0,
    misses: 0,
    savedCost: 0,
    savedTime: 0,
  };

  constructor(private config: OptimizationConfig) {}

  /**
   * O(1) lookup - hash map access
   */
  get(key: string): T | null {
    const entry = this.cache.get(key);

    if (!entry) {
      this.stats.misses++;
      return null;
    }

    // Check TTL
    const age = Date.now() - entry.timestamp;
    if (age > this.config.cacheTTLMs) {
      this.cache.delete(key);
      this.stats.misses++;
      return null;
    }

    // Cache hit!
    entry.hits++;
    this.stats.hits++;
    this.stats.savedCost += entry.cost;
    this.stats.savedTime += 1; // Assume 1 LLM call saved

    return entry.value;
  }

  /**
   * O(1) insert
   */
  set(key: string, value: T, cost: number = 0): void {
    // Evict if at capacity (LRU)
    if (this.cache.size >= this.config.maxCacheSize) {
      this.evictLRU();
    }

    this.cache.set(key, {
      value,
      timestamp: Date.now(),
      hits: 0,
      cost,
    });
  }

  /**
   * O(n) eviction (but rare, only when cache full)
   */
  private evictLRU(): void {
    let oldestKey: string | null = null;
    let oldestTime = Infinity;

    for (const [key, entry] of this.cache.entries()) {
      if (entry.timestamp < oldestTime) {
        oldestTime = entry.timestamp;
        oldestKey = key;
      }
    }

    if (oldestKey) {
      this.cache.delete(oldestKey);
    }
  }

  getStats(): CacheStats {
    const total = this.stats.hits + this.stats.misses;
    return {
      hits: this.stats.hits,
      misses: this.stats.misses,
      hitRate: total > 0 ? this.stats.hits / total : 0,
      savedCost: this.stats.savedCost,
      savedTime: this.stats.savedTime,
    };
  }

  clear(): void {
    this.cache.clear();
  }
}

// ============================================================================
// Memoization Decorator
// ============================================================================

/**
 * Memoize function calls for O(1) repeated access
 */
export function memoize<T>(
  fn: (...args: any[]) => T,
  keyFn: (...args: any[]) => string = (...args) => JSON.stringify(args)
): (...args: any[]) => T {
  const cache = new Map<string, T>();

  return (...args: any[]): T => {
    const key = keyFn(...args);

    if (cache.has(key)) {
      return cache.get(key)!;
    }

    const result = fn(...args);
    cache.set(key, result);
    return result;
  };
}

// ============================================================================
// Lazy Evaluator - Only Compute When Needed
// ============================================================================

export class LazyEvaluator<T> {
  private computed: T | null = null;
  private isComputed = false;

  constructor(
    private computeFn: () => T,
    private shouldComputeFn: () => boolean = () => true
  ) {}

  /**
   * O(1) if already computed, otherwise O(computeFn)
   */
  get value(): T {
    if (!this.isComputed && this.shouldComputeFn()) {
      this.computed = this.computeFn();
      this.isComputed = true;
    }

    return this.computed!;
  }

  get isEvaluated(): boolean {
    return this.isComputed;
  }

  reset(): void {
    this.computed = null;
    this.isComputed = false;
  }
}

// ============================================================================
// Indexed Slice Lookup - O(1) Knowledge Access
// ============================================================================

export interface SliceIndex {
  conceptToSlices: Map<string, string[]>; // concept -> [slice1, slice2, ...]
  sliceToPath: Map<string, string>;       // slice -> file path
  domainToSlices: Map<string, string[]>;  // domain -> [slice1, slice2, ...]
}

export class IndexedSliceNavigator {
  private index: SliceIndex = {
    conceptToSlices: new Map(),
    sliceToPath: new Map(),
    domainToSlices: new Map(),
  };

  private lastRebuild = 0;

  constructor(
    private config: OptimizationConfig,
    private slicesRoot: string
  ) {}

  /**
   * O(1) lookup by concept
   */
  findSlicesByConcept(concept: string): string[] {
    this.maybeRebuildIndex();
    return this.index.conceptToSlices.get(concept.toLowerCase()) || [];
  }

  /**
   * O(1) lookup by domain
   */
  findSlicesByDomain(domain: string): string[] {
    this.maybeRebuildIndex();
    return this.index.domainToSlices.get(domain.toLowerCase()) || [];
  }

  /**
   * O(1) get slice path
   */
  getSlicePath(sliceName: string): string | null {
    this.maybeRebuildIndex();
    return this.index.sliceToPath.get(sliceName) || null;
  }

  /**
   * Rebuild index if needed (O(n) but infrequent)
   */
  private maybeRebuildIndex(): void {
    const now = Date.now();
    if (now - this.lastRebuild < this.config.rebuildIndexInterval) {
      return; // Index still fresh
    }

    this.rebuildIndex();
    this.lastRebuild = now;
  }

  /**
   * Build inverted indexes for O(1) lookups
   */
  private rebuildIndex(): void {
    // TODO: Scan slices directory and build indexes
    // This is O(n) but only runs every 10 minutes
    // In production, would use file watcher for incremental updates
  }
}

// ============================================================================
// Early Stopping Coordinator
// ============================================================================

export class EarlyStoppingCoordinator {
  private iteration = 0;

  constructor(private config: OptimizationConfig) {}

  /**
   * Should we stop recursion early?
   */
  shouldStop(context: {
    currentConfidence: number;
    iteration: number;
    hasSufficientInfo: boolean;
  }): boolean {
    if (!this.config.enableEarlyStopping) {
      return false;
    }

    // Stop if high confidence
    if (context.currentConfidence >= this.config.confidenceThreshold) {
      return true;
    }

    // Stop if max iterations
    if (context.iteration >= this.config.maxIterations) {
      return true;
    }

    // Stop if we have sufficient info
    if (context.hasSufficientInfo) {
      return true;
    }

    return false;
  }

  incrementIteration(): number {
    return ++this.iteration;
  }

  reset(): void {
    this.iteration = 0;
  }
}

// ============================================================================
// O(1) Optimizer - Main Coordinator
// ============================================================================

export class O1Optimizer {
  private queryCache: O1Cache<any>;
  private sliceCache: O1Cache<string>;
  private indexedNavigator: IndexedSliceNavigator;
  private earlyStopping: EarlyStoppingCoordinator;

  constructor(
    private config: OptimizationConfig = DEFAULT_O1_CONFIG,
    slicesRoot: string = './slices'
  ) {
    this.queryCache = new O1Cache(config);
    this.sliceCache = new O1Cache(config);
    this.indexedNavigator = new IndexedSliceNavigator(config, slicesRoot);
    this.earlyStopping = new EarlyStoppingCoordinator(config);
  }

  /**
   * Get cached query response (O(1))
   */
  getCachedQuery(query: string): any | null {
    const key = this.hashQuery(query);
    return this.queryCache.get(key);
  }

  /**
   * Cache query response
   */
  cacheQuery(query: string, response: any, cost: number): void {
    const key = this.hashQuery(query);
    this.queryCache.set(key, response, cost);
  }

  /**
   * Get cached slice content (O(1))
   */
  getCachedSlice(sliceName: string): string | null {
    return this.sliceCache.get(sliceName);
  }

  /**
   * Cache slice content
   */
  cacheSlice(sliceName: string, content: string): void {
    this.sliceCache.set(sliceName, content);
  }

  /**
   * Find relevant slices (O(1) via index)
   */
  findRelevantSlices(concepts: string[]): string[] {
    const slices = new Set<string>();

    for (const concept of concepts) {
      const conceptSlices = this.indexedNavigator.findSlicesByConcept(concept);
      conceptSlices.forEach(s => slices.add(s));
    }

    return Array.from(slices);
  }

  /**
   * Should stop recursion?
   */
  shouldStopRecursion(context: {
    currentConfidence: number;
    iteration: number;
    hasSufficientInfo: boolean;
  }): boolean {
    return this.earlyStopping.shouldStop(context);
  }

  /**
   * Get optimization statistics
   */
  getStats() {
    return {
      queryCache: this.queryCache.getStats(),
      sliceCache: this.sliceCache.getStats(),
    };
  }

  /**
   * Hash query for cache key (O(1) for short queries)
   */
  private hashQuery(query: string): string {
    // Simple hash - in production use crypto.createHash
    return query.toLowerCase().trim().slice(0, 100);
  }
}

// ============================================================================
// Usage Example
// ============================================================================

/*
import { O1Optimizer, DEFAULT_O1_CONFIG } from './o1-optimizer';

const optimizer = new O1Optimizer(DEFAULT_O1_CONFIG);

// Check cache first (O(1))
const cached = optimizer.getCachedQuery("What is DDD?");
if (cached) {
  return cached; // Instant response!
}

// Find relevant slices (O(1) via index)
const slices = optimizer.findRelevantSlices(['ddd', 'bounded_context']);

// Check if should stop early
if (optimizer.shouldStopRecursion({
  currentConfidence: 0.92,
  iteration: 1,
  hasSufficientInfo: true
})) {
  return currentAnswer; // Stop early!
}

// Cache result
optimizer.cacheQuery("What is DDD?", answer, cost);

// View stats
console.log(optimizer.getStats());
// {
//   queryCache: { hits: 42, misses: 5, hitRate: 0.89, savedCost: $12.34 },
//   sliceCache: { hits: 156, misses: 12, hitRate: 0.93, savedCost: $0 }
// }
*/
