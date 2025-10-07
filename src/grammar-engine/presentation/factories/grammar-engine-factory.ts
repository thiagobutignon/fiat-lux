/**
 * Grammar Engine Factory
 *
 * Factory function to create configured grammar engine instances
 */

import { GrammarConfig } from '../../domain/entities/types'
import { GrammarEngine } from '../../domain/use-cases/grammar-engine'
import { SimilarityCache } from '../../data/use-cases/similarity-cache-impl'

/**
 * Creates a configured GrammarEngine instance
 *
 * @param config - Grammar configuration
 * @returns Configured GrammarEngine instance
 */
export function makeGrammarEngine(config: GrammarConfig): GrammarEngine {
  const cache = new SimilarityCache()
  return new GrammarEngine(config, cache)
}
