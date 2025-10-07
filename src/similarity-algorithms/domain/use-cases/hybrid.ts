/**
 * Hybrid Similarity Algorithm
 *
 * Combines multiple similarity algorithms using weighted averages
 * for better overall accuracy across different types of typos and variations.
 */

import { levenshteinSimilarity } from './levenshtein'
import { jaroWinklerSimilarity } from './jaro-winkler'

/**
 * Hybrid similarity combining multiple algorithms
 * Uses weighted average of different metrics
 *
 * @param a - First string
 * @param b - Second string
 * @param caseSensitive - Whether to use case-sensitive comparison
 * @returns Similarity score between 0 and 1
 */
export function hybridSimilarity(a: string, b: string, caseSensitive = false): number {
  const s1 = caseSensitive ? a : a.toLowerCase()
  const s2 = caseSensitive ? b : b.toLowerCase()

  const levenshtein = levenshteinSimilarity(s1, s2, true)
  const jaroWinkler = jaroWinklerSimilarity(s1, s2)

  // Weighted average: Levenshtein 60%, Jaro-Winkler 40%
  return levenshtein * 0.6 + jaroWinkler * 0.4
}
