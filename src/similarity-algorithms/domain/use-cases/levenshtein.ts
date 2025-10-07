/**
 * Levenshtein Distance Algorithm
 *
 * Calculates the minimum number of single-character edits (insertions,
 * deletions, or substitutions) needed to transform one string into another.
 *
 * Time complexity: O(m*n) where m, n are string lengths
 */

/**
 * Calculates Levenshtein distance between two strings
 *
 * @param a - First string
 * @param b - Second string
 * @returns Minimum number of single-character edits needed
 */
export function levenshteinDistance(a: string, b: string): number {
  const matrix: number[][] = Array(b.length + 1)
    .fill(null)
    .map(() => Array(a.length + 1).fill(0))

  // Initialize first row and column
  for (let i = 0; i <= b.length; i++) matrix[i][0] = i
  for (let j = 0; j <= a.length; j++) matrix[0][j] = j

  // Calculate distances
  for (let i = 1; i <= b.length; i++) {
    for (let j = 1; j <= a.length; j++) {
      if (b.charAt(i - 1) === a.charAt(j - 1)) {
        matrix[i][j] = matrix[i - 1][j - 1]
      } else {
        matrix[i][j] = Math.min(
          matrix[i - 1][j - 1] + 1, // substitution
          matrix[i][j - 1] + 1,     // insertion
          matrix[i - 1][j] + 1      // deletion
        )
      }
    }
  }

  return matrix[b.length][a.length]
}

/**
 * Calculates similarity score using Levenshtein distance
 * Normalized to 0-1 range
 *
 * @param a - First string
 * @param b - Second string
 * @param caseSensitive - Whether to use case-sensitive comparison
 * @returns Similarity score between 0 and 1
 */
export function levenshteinSimilarity(a: string, b: string, caseSensitive = false): number {
  const s1 = caseSensitive ? a : a.toLowerCase()
  const s2 = caseSensitive ? b : b.toLowerCase()

  const maxLength = Math.max(s1.length, s2.length)
  if (maxLength === 0) return 1

  const distance = levenshteinDistance(s1, s2)
  return (maxLength - distance) / maxLength
}
