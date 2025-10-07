/**
 * Jaro-Winkler Similarity Algorithm
 *
 * Measures similarity between two strings, giving more favorable ratings
 * to strings with common prefixes. Better for short strings and typos
 * at the beginning of words.
 */

/**
 * Calculates Jaro-Winkler similarity between two strings
 *
 * @param s1 - First string
 * @param s2 - Second string
 * @returns Similarity score between 0 and 1
 */
export function jaroWinklerSimilarity(s1: string, s2: string): number {
  if (s1 === s2) return 1.0
  if (s1.length === 0 || s2.length === 0) return 0.0

  const matchWindow = Math.floor(Math.max(s1.length, s2.length) / 2) - 1
  const s1Matches = new Array(s1.length).fill(false)
  const s2Matches = new Array(s2.length).fill(false)

  let matches = 0
  let transpositions = 0

  // Find matches
  for (let i = 0; i < s1.length; i++) {
    const start = Math.max(0, i - matchWindow)
    const end = Math.min(i + matchWindow + 1, s2.length)

    for (let j = start; j < end; j++) {
      if (s2Matches[j] || s1[i] !== s2[j]) continue
      s1Matches[i] = true
      s2Matches[j] = true
      matches++
      break
    }
  }

  if (matches === 0) return 0.0

  // Count transpositions
  let k = 0
  for (let i = 0; i < s1.length; i++) {
    if (!s1Matches[i]) continue
    while (!s2Matches[k]) k++
    if (s1[i] !== s2[k]) transpositions++
    k++
  }

  // Calculate Jaro similarity
  const jaro = (matches / s1.length + matches / s2.length + (matches - transpositions / 2) / matches) / 3

  // Calculate Jaro-Winkler similarity (bonus for common prefix)
  let prefixLength = 0
  for (let i = 0; i < Math.min(s1.length, s2.length, 4); i++) {
    if (s1[i] === s2[i]) prefixLength++
    else break
  }

  return jaro + prefixLength * 0.1 * (1 - jaro)
}
