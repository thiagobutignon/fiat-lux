/**
 * Similarity Cache Implementation
 *
 * Improves performance for repeated similarity comparisons by caching results
 */

import { SimilarityAlgorithm } from '../../domain/entities/types'
import { ISimilarityCache, CacheStats } from '../protocols/similarity-cache'

export class SimilarityCache implements ISimilarityCache {
  private cache = new Map<string, number>()
  private hits = 0
  private misses = 0

  private getCacheKey(a: string, b: string, algorithm: SimilarityAlgorithm): string {
    return `${algorithm}:${a}:${b}`
  }

  get(a: string, b: string, algorithm: SimilarityAlgorithm): number | undefined {
    const key = this.getCacheKey(a, b, algorithm)
    const value = this.cache.get(key)
    if (value !== undefined) this.hits++
    else this.misses++
    return value
  }

  set(a: string, b: string, algorithm: SimilarityAlgorithm, value: number): void {
    const key = this.getCacheKey(a, b, algorithm)
    this.cache.set(key, value)
  }

  clear(): void {
    this.cache.clear()
    this.hits = 0
    this.misses = 0
  }

  getHits(): number {
    return this.hits
  }

  getStats(): CacheStats {
    const total = this.hits + this.misses
    return {
      hits: this.hits,
      misses: this.misses,
      size: this.cache.size,
      hitRate: total > 0 ? this.hits / total : 0
    }
  }
}
