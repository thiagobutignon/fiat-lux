/**
 * Hybrid Similarity Algorithm Tests
 */

import { describe, it, expect } from '../../shared/utils/test-runner'
import { hybridSimilarity } from '../domain/use-cases/hybrid'

describe('Hybrid Similarity', () => {
  it('should return 1.0 for identical strings', () => {
    expect.toEqual(hybridSimilarity('hello', 'hello'), 1.0)
    expect.toEqual(hybridSimilarity('test', 'test'), 1.0)
  })

  it('should return 0.0 for completely different strings', () => {
    const similarity = hybridSimilarity('abc', 'xyz')
    expect.toEqual(similarity, 0.0)
  })

  it('should return values between 0 and 1', () => {
    const similarity = hybridSimilarity('cat', 'bat')
    expect.toBeGreaterThan(similarity, 0.0)
    expect.toBeLessThan(similarity, 1.0)
  })

  it('should be case-insensitive by default', () => {
    const similarity = hybridSimilarity('Hello', 'hello')
    expect.toEqual(similarity, 1.0)
  })

  it('should be case-sensitive when specified', () => {
    const similarity = hybridSimilarity('Hello', 'hello', true)
    expect.toBeLessThan(similarity, 1.0)
  })

  it('should handle common typos well', () => {
    // Test with common typos from our demos
    const typos = [
      { correct: 'add', typo: 'ad' },
      { correct: 'Repository', typo: 'Repo' },
      { correct: 'Controller', typo: 'Control' },
      { correct: 'Account.Params', typo: 'AccountParams' }
    ]

    typos.forEach(({ correct, typo }) => {
      const similarity = hybridSimilarity(correct, typo)
      expect.toBeGreaterThan(similarity, 0.4)
    })
  })

  it('should combine strengths of both algorithms', () => {
    // Hybrid should perform reasonably well on both types
    const prefixTypo = hybridSimilarity('prefix123', 'prefix456')
    const editTypo = hybridSimilarity('kitten', 'sitting')

    expect.toBeGreaterThan(prefixTypo, 0.5)
    expect.toBeGreaterThan(editTypo, 0.4)
  })

  it('should handle empty strings', () => {
    expect.toEqual(hybridSimilarity('', ''), 1.0)
  })
})
