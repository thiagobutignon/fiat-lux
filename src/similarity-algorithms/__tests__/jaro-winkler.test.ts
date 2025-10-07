/**
 * Jaro-Winkler Similarity Algorithm Tests
 */

import { describe, it, expect } from '../../shared/utils/test-runner'
import { jaroWinklerSimilarity } from '../domain/use-cases/jaro-winkler'

describe('Jaro-Winkler Similarity', () => {
  it('should return 1.0 for identical strings', () => {
    expect.toEqual(jaroWinklerSimilarity('hello', 'hello'), 1.0)
    expect.toEqual(jaroWinklerSimilarity('test', 'test'), 1.0)
  })

  it('should return 0.0 for completely different strings', () => {
    expect.toEqual(jaroWinklerSimilarity('abc', 'xyz'), 0.0)
  })

  it('should return values between 0 and 1', () => {
    const similarity = jaroWinklerSimilarity('martha', 'marhta')
    expect.toBeGreaterThan(similarity, 0.0)
    expect.toBeLessThan(similarity, 1.0)
  })

  it('should handle empty strings', () => {
    expect.toEqual(jaroWinklerSimilarity('', ''), 1.0)
    expect.toEqual(jaroWinklerSimilarity('a', ''), 0.0)
    expect.toEqual(jaroWinklerSimilarity('', 'a'), 0.0)
  })

  it('should favor strings with common prefixes', () => {
    // Jaro-Winkler gives bonus for common prefixes
    const withPrefix = jaroWinklerSimilarity('prefix123', 'prefix456')
    const withoutPrefix = jaroWinklerSimilarity('123prefix', '456prefix')

    expect.toBeGreaterThan(withPrefix, withoutPrefix)
  })

  it('should handle transpositions', () => {
    // "marhta" vs "martha" - one transposition
    const similarity = jaroWinklerSimilarity('martha', 'marhta')
    expect.toBeGreaterThan(similarity, 0.9)
  })

  it('should work well with short strings', () => {
    const similarity = jaroWinklerSimilarity('add', 'ad')
    expect.toBeGreaterThan(similarity, 0.7)
  })

  it('should handle typos at beginning', () => {
    // Jaro-Winkler is particularly good at catching typos at the beginning
    const similarity = jaroWinklerSimilarity('controller', 'controll')
    expect.toBeGreaterThan(similarity, 0.9)
  })
})
