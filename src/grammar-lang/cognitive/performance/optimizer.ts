/**
 * Performance Optimizer
 * Optimizations to achieve <0.5ms detection time:
 * - Memoization of parsing results
 * - LRU cache for frequent patterns
 * - Lazy loading of techniques
 * - Pre-compiled regex patterns
 * - Profiling and monitoring
 */

// ============================================================
// LRU CACHE
// ============================================================

class LRUCache<K, V> {
  private cache: Map<K, V>;
  private maxSize: number;

  constructor(maxSize: number = 1000) {
    this.cache = new Map();
    this.maxSize = maxSize;
  }

  get(key: K): V | undefined {
    if (!this.cache.has(key)) {
      return undefined;
    }

    // Move to end (most recently used)
    const value = this.cache.get(key)!;
    this.cache.delete(key);
    this.cache.set(key, value);

    return value;
  }

  set(key: K, value: V): void {
    // Delete if exists (to move to end)
    if (this.cache.has(key)) {
      this.cache.delete(key);
    }

    // Add to end
    this.cache.set(key, value);

    // Evict oldest if over capacity
    if (this.cache.size > this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
  }

  clear(): void {
    this.cache.clear();
  }

  size(): number {
    return this.cache.size;
  }

  has(key: K): boolean {
    return this.cache.has(key);
  }
}

// ============================================================
// MEMOIZATION
// ============================================================

/**
 * Memoize function with LRU cache
 */
export function memoize<T extends (...args: any[]) => any>(
  fn: T,
  options: {
    maxCacheSize?: number;
    keyGenerator?: (...args: Parameters<T>) => string;
  } = {}
): T {
  const cache = new LRUCache<string, ReturnType<T>>(
    options.maxCacheSize ?? 1000
  );

  const defaultKeyGenerator = (...args: any[]) => JSON.stringify(args);
  const keyGenerator = options.keyGenerator ?? defaultKeyGenerator;

  return ((...args: Parameters<T>): ReturnType<T> => {
    const key = keyGenerator(...args);

    if (cache.has(key)) {
      return cache.get(key)!;
    }

    const result = fn(...args);
    cache.set(key, result);

    return result;
  }) as T;
}

// ============================================================
// PARSING CACHE
// ============================================================

interface ParsedResult {
  phonemes: any;
  morphemes: any;
  syntax: any;
  semantics: any;
  pragmatics: any;
  timestamp: number;
}

/**
 * Cache for parsed linguistic structures
 * Avoids re-parsing the same text multiple times
 */
export class ParsingCache {
  private cache: LRUCache<string, ParsedResult>;
  private ttl: number; // Time to live in ms

  constructor(maxSize: number = 500, ttl: number = 60000) {
    this.cache = new LRUCache(maxSize);
    this.ttl = ttl;
  }

  get(text: string): ParsedResult | null {
    const result = this.cache.get(text);

    if (!result) {
      return null;
    }

    // Check if expired
    if (Date.now() - result.timestamp > this.ttl) {
      this.cache.delete(text);
      return null;
    }

    return result;
  }

  set(text: string, result: Omit<ParsedResult, 'timestamp'>): void {
    this.cache.set(text, {
      ...result,
      timestamp: Date.now()
    });
  }

  clear(): void {
    this.cache.clear();
  }

  getStats() {
    return {
      size: this.cache.size(),
      ttl: this.ttl
    };
  }
}

// Create global parsing cache
export const globalParsingCache = new ParsingCache();

// ============================================================
// PROFILING
// ============================================================

interface ProfileEntry {
  function_name: string;
  execution_time_ms: number;
  timestamp: number;
  args_hash?: string;
}

/**
 * Performance profiler
 * Track execution times of critical functions
 */
export class Profiler {
  private profiles: ProfileEntry[];
  private enabled: boolean;
  private activeTimers: Map<string, number>;

  constructor(enabled: boolean = false) {
    this.profiles = [];
    this.enabled = enabled;
    this.activeTimers = new Map();
  }

  start(functionName: string, argsHash?: string): void {
    if (!this.enabled) return;

    const key = `${functionName}:${argsHash ?? ''}`;
    this.activeTimers.set(key, Date.now());
  }

  end(functionName: string, argsHash?: string): number {
    if (!this.enabled) return 0;

    const key = `${functionName}:${argsHash ?? ''}`;
    const startTime = this.activeTimers.get(key);

    if (!startTime) {
      console.warn(`Profiler: No start time for ${functionName}`);
      return 0;
    }

    const executionTime = Date.now() - startTime;

    this.profiles.push({
      function_name: functionName,
      execution_time_ms: executionTime,
      timestamp: Date.now(),
      args_hash: argsHash
    });

    this.activeTimers.delete(key);

    return executionTime;
  }

  /**
   * Profile a function call
   */
  profile<T>(functionName: string, fn: () => T, argsHash?: string): T {
    this.start(functionName, argsHash);
    try {
      return fn();
    } finally {
      this.end(functionName, argsHash);
    }
  }

  /**
   * Profile an async function call
   */
  async profileAsync<T>(
    functionName: string,
    fn: () => Promise<T>,
    argsHash?: string
  ): Promise<T> {
    this.start(functionName, argsHash);
    try {
      return await fn();
    } finally {
      this.end(functionName, argsHash);
    }
  }

  getReport() {
    if (this.profiles.length === 0) {
      return {
        total_calls: 0,
        average_time_ms: 0,
        min_time_ms: 0,
        max_time_ms: 0,
        p50_ms: 0,
        p95_ms: 0,
        p99_ms: 0,
        by_function: {}
      };
    }

    const times = this.profiles.map(p => p.execution_time_ms).sort((a, b) => a - b);
    const byFunction: Record<string, {
      count: number;
      avg_ms: number;
      min_ms: number;
      max_ms: number;
    }> = {};

    // Aggregate by function
    for (const entry of this.profiles) {
      if (!byFunction[entry.function_name]) {
        byFunction[entry.function_name] = {
          count: 0,
          avg_ms: 0,
          min_ms: Infinity,
          max_ms: -Infinity
        };
      }

      const stats = byFunction[entry.function_name];
      stats.count++;
      stats.avg_ms = (stats.avg_ms * (stats.count - 1) + entry.execution_time_ms) / stats.count;
      stats.min_ms = Math.min(stats.min_ms, entry.execution_time_ms);
      stats.max_ms = Math.max(stats.max_ms, entry.execution_time_ms);
    }

    return {
      total_calls: this.profiles.length,
      average_time_ms: times.reduce((a, b) => a + b, 0) / times.length,
      min_time_ms: times[0],
      max_time_ms: times[times.length - 1],
      p50_ms: times[Math.floor(times.length * 0.5)],
      p95_ms: times[Math.floor(times.length * 0.95)],
      p99_ms: times[Math.floor(times.length * 0.99)],
      by_function: byFunction
    };
  }

  clear(): void {
    this.profiles = [];
    this.activeTimers.clear();
  }

  enable(): void {
    this.enabled = true;
  }

  disable(): void {
    this.enabled = false;
  }

  isEnabled(): boolean {
    return this.enabled;
  }
}

// Global profiler instance
export const globalProfiler = new Profiler(
  process.env.NODE_ENV === 'development'
);

// ============================================================
// LAZY LOADING
// ============================================================

/**
 * Lazy-load heavy resources
 */
export class LazyLoader<T> {
  private loader: () => T | Promise<T>;
  private value: T | null;
  private loading: Promise<T> | null;

  constructor(loader: () => T | Promise<T>) {
    this.loader = loader;
    this.value = null;
    this.loading = null;
  }

  async get(): Promise<T> {
    if (this.value !== null) {
      return this.value;
    }

    if (this.loading) {
      return this.loading;
    }

    this.loading = Promise.resolve(this.loader()).then(value => {
      this.value = value;
      this.loading = null;
      return value;
    });

    return this.loading;
  }

  isLoaded(): boolean {
    return this.value !== null;
  }

  reset(): void {
    this.value = null;
    this.loading = null;
  }
}

// ============================================================
// REGEX CACHE
// ============================================================

/**
 * Pre-compiled regex cache
 * Avoid recompiling the same regex patterns
 */
export class RegexCache {
  private cache: Map<string, RegExp>;

  constructor() {
    this.cache = new Map();
  }

  get(pattern: string, flags?: string): RegExp {
    const key = `${pattern}:${flags ?? ''}`;

    if (this.cache.has(key)) {
      return this.cache.get(key)!;
    }

    const regex = new RegExp(pattern, flags);
    this.cache.set(key, regex);

    return regex;
  }

  clear(): void {
    this.cache.clear();
  }

  size(): number {
    return this.cache.size;
  }
}

// Global regex cache
export const globalRegexCache = new RegexCache();

// ============================================================
// PERFORMANCE MONITORING
// ============================================================

export interface PerformanceMetrics {
  average_detection_time_ms: number;
  p50_detection_time_ms: number;
  p95_detection_time_ms: number;
  p99_detection_time_ms: number;
  total_detections: number;
  cache_hit_rate: number;
  cache_size: number;
}

/**
 * Monitor and track performance metrics
 */
export class PerformanceMonitor {
  private detectionTimes: number[];
  private cacheHits: number;
  private cacheMisses: number;
  private maxSamples: number;

  constructor(maxSamples: number = 1000) {
    this.detectionTimes = [];
    this.cacheHits = 0;
    this.cacheMisses = 0;
    this.maxSamples = maxSamples;
  }

  recordDetectionTime(timeMs: number): void {
    this.detectionTimes.push(timeMs);

    // Keep only recent samples
    if (this.detectionTimes.length > this.maxSamples) {
      this.detectionTimes.shift();
    }
  }

  recordCacheHit(): void {
    this.cacheHits++;
  }

  recordCacheMiss(): void {
    this.cacheMisses++;
  }

  getMetrics(): PerformanceMetrics {
    if (this.detectionTimes.length === 0) {
      return {
        average_detection_time_ms: 0,
        p50_detection_time_ms: 0,
        p95_detection_time_ms: 0,
        p99_detection_time_ms: 0,
        total_detections: 0,
        cache_hit_rate: 0,
        cache_size: globalParsingCache.getStats().size
      };
    }

    const sorted = [...this.detectionTimes].sort((a, b) => a - b);

    const totalCacheAccesses = this.cacheHits + this.cacheMisses;
    const cacheHitRate = totalCacheAccesses > 0
      ? this.cacheHits / totalCacheAccesses
      : 0;

    return {
      average_detection_time_ms:
        this.detectionTimes.reduce((a, b) => a + b, 0) / this.detectionTimes.length,
      p50_detection_time_ms: sorted[Math.floor(sorted.length * 0.5)],
      p95_detection_time_ms: sorted[Math.floor(sorted.length * 0.95)],
      p99_detection_time_ms: sorted[Math.floor(sorted.length * 0.99)],
      total_detections: this.detectionTimes.length,
      cache_hit_rate: cacheHitRate,
      cache_size: globalParsingCache.getStats().size
    };
  }

  reset(): void {
    this.detectionTimes = [];
    this.cacheHits = 0;
    this.cacheMisses = 0;
  }
}

// Global performance monitor
export const globalPerformanceMonitor = new PerformanceMonitor();

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

/**
 * Hash a string quickly for caching
 */
export function fastHash(str: string): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  return hash.toString(36);
}

/**
 * Debounce a function
 */
export function debounce<T extends (...args: any[]) => any>(
  fn: T,
  delayMs: number
): (...args: Parameters<T>) => void {
  let timer: NodeJS.Timeout | null = null;

  return (...args: Parameters<T>) => {
    if (timer) {
      clearTimeout(timer);
    }

    timer = setTimeout(() => {
      fn(...args);
    }, delayMs);
  };
}

/**
 * Throttle a function
 */
export function throttle<T extends (...args: any[]) => any>(
  fn: T,
  delayMs: number
): (...args: Parameters<T>) => void {
  let lastCall = 0;

  return (...args: Parameters<T>) => {
    const now = Date.now();

    if (now - lastCall >= delayMs) {
      lastCall = now;
      fn(...args);
    }
  };
}
