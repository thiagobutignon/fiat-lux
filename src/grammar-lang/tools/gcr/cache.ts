/**
 * GCR Build Cache
 *
 * O(1) build cache using content-addressable storage.
 * Caches build results by content hash for instant rebuilds.
 */

import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';
import { ContainerImage, ImageLayer } from './types';

// ============================================================================
// Build Cache
// ============================================================================

export class BuildCache {
  private cacheDir: string;

  constructor(cacheDir: string = '.gcr/cache') {
    this.cacheDir = cacheDir;
    this.ensureCacheDir();
  }

  /**
   * Ensure cache directory exists
   */
  private ensureCacheDir(): void {
    if (!fs.existsSync(this.cacheDir)) {
      fs.mkdirSync(this.cacheDir, { recursive: true });
    }
  }

  /**
   * Get cache key from build inputs
   */
  getCacheKey(inputs: BuildCacheInputs): string {
    const hash = crypto.createHash('sha256');

    // Hash all inputs for deterministic cache key
    hash.update(inputs.specHash || '');
    hash.update(inputs.baseImage || '');
    hash.update(JSON.stringify(inputs.buildArgs || {}));
    hash.update(inputs.platform || '');

    if (inputs.layerHashes) {
      for (const layerHash of inputs.layerHashes.sort()) {
        hash.update(layerHash);
      }
    }

    return hash.digest('hex');
  }

  /**
   * Check if build is cached (O(1))
   */
  has(cacheKey: string): boolean {
    return fs.existsSync(this.getCachePath(cacheKey));
  }

  /**
   * Get cached build result (O(1))
   */
  get(cacheKey: string): CachedBuild | null {
    const cachePath = this.getCachePath(cacheKey);

    if (!fs.existsSync(cachePath)) {
      return null;
    }

    try {
      const cached = JSON.parse(fs.readFileSync(cachePath, 'utf-8'));
      return cached as CachedBuild;
    } catch (error) {
      // Invalid cache entry
      return null;
    }
  }

  /**
   * Save build result to cache (O(1))
   */
  set(cacheKey: string, build: CachedBuild): void {
    const cachePath = this.getCachePath(cacheKey);
    fs.writeFileSync(cachePath, JSON.stringify(build, null, 2), 'utf-8');
  }

  /**
   * Invalidate cache entry
   */
  invalidate(cacheKey: string): void {
    const cachePath = this.getCachePath(cacheKey);
    if (fs.existsSync(cachePath)) {
      fs.unlinkSync(cachePath);
    }
  }

  /**
   * Clear entire cache
   */
  clear(): number {
    if (!fs.existsSync(this.cacheDir)) {
      return 0;
    }

    const files = fs.readdirSync(this.cacheDir);
    let count = 0;

    for (const file of files) {
      const filePath = path.join(this.cacheDir, file);
      fs.unlinkSync(filePath);
      count++;
    }

    return count;
  }

  /**
   * Get cache statistics
   */
  getStats(): CacheStats {
    if (!fs.existsSync(this.cacheDir)) {
      return {
        entries: 0,
        totalSize: 0,
        oldestEntry: null,
        newestEntry: null,
      };
    }

    const files = fs.readdirSync(this.cacheDir);
    let totalSize = 0;
    let oldestTime: number | null = null;
    let newestTime: number | null = null;

    for (const file of files) {
      const filePath = path.join(this.cacheDir, file);
      const stat = fs.statSync(filePath);
      totalSize += stat.size;

      const mtime = stat.mtimeMs;
      if (oldestTime === null || mtime < oldestTime) {
        oldestTime = mtime;
      }
      if (newestTime === null || mtime > newestTime) {
        newestTime = mtime;
      }
    }

    return {
      entries: files.length,
      totalSize,
      oldestEntry: oldestTime ? new Date(oldestTime).toISOString() : null,
      newestEntry: newestTime ? new Date(newestTime).toISOString() : null,
    };
  }

  /**
   * Garbage collect old cache entries
   */
  garbageCollect(maxAge: number = 7 * 24 * 60 * 60 * 1000): number {
    if (!fs.existsSync(this.cacheDir)) {
      return 0;
    }

    const now = Date.now();
    const files = fs.readdirSync(this.cacheDir);
    let deletedCount = 0;

    for (const file of files) {
      const filePath = path.join(this.cacheDir, file);
      const stat = fs.statSync(filePath);
      const age = now - stat.mtimeMs;

      if (age > maxAge) {
        fs.unlinkSync(filePath);
        deletedCount++;
      }
    }

    return deletedCount;
  }

  /**
   * Get cache path for key
   */
  private getCachePath(cacheKey: string): string {
    return path.join(this.cacheDir, `${cacheKey}.json`);
  }

  /**
   * List all cached builds
   */
  list(): CachedBuild[] {
    if (!fs.existsSync(this.cacheDir)) {
      return [];
    }

    const files = fs.readdirSync(this.cacheDir);
    const builds: CachedBuild[] = [];

    for (const file of files) {
      if (!file.endsWith('.json')) continue;

      try {
        const cacheKey = file.replace('.json', '');
        const cached = this.get(cacheKey);
        if (cached) {
          builds.push(cached);
        }
      } catch (error) {
        // Skip invalid entries
      }
    }

    return builds;
  }

  /**
   * Get cache hit rate
   */
  getHitRate(): number {
    // This would require tracking hits/misses
    // For now, return 0 (not implemented)
    return 0;
  }
}

// ============================================================================
// Types
// ============================================================================

export interface BuildCacheInputs {
  specHash: string;
  baseImage?: string;
  buildArgs?: Record<string, string>;
  platform?: string;
  layerHashes?: string[];
}

export interface CachedBuild {
  cacheKey: string;
  imageHash: string;
  imageName: string;
  imageVersion: string;
  layers: ImageLayer[];
  buildTime: string;
  buildDuration: number; // milliseconds
  metadata: {
    specHash: string;
    baseImage?: string;
    platform?: string;
  };
}

export interface CacheStats {
  entries: number;
  totalSize: number;
  oldestEntry: string | null;
  newestEntry: string | null;
}

// ============================================================================
// Cache Utilities
// ============================================================================

/**
 * Calculate spec hash
 */
export function hashSpec(specPath: string): string {
  const content = fs.readFileSync(specPath, 'utf-8');
  const hash = crypto.createHash('sha256');
  hash.update(content);
  return hash.digest('hex');
}

/**
 * Check if cache is valid (all layers exist)
 */
export function isCacheValid(cached: CachedBuild, layerBuilder: any): boolean {
  for (const layer of cached.layers) {
    if (!layerBuilder.hasLayer(layer.hash)) {
      return false; // Missing layer
    }
  }
  return true;
}

/**
 * Format cache duration
 */
export function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}min`;
}

// ============================================================================
// Exports
// ============================================================================

// All exports are already declared inline above
