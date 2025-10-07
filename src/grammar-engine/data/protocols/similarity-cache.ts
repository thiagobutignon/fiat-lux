/**
 * Similarity Cache Protocol
 *
 * Interface for caching similarity calculations
 */

import { SimilarityAlgorithm } from '../../../grammar-engine/domain/entities/types'

export interface ISimilarityCache {
  get(a: string, b: string, algorithm: SimilarityAlgorithm): number | undefined
  set(a: string, b: string, algorithm: SimilarityAlgorithm, value: number): void
  clear(): void
  getHits(): number
  getStats(): CacheStats
}

export interface CacheStats {
  hits: number
  misses: number
  size: number
  hitRate: number
}
