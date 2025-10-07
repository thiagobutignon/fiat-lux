/**
 * Levenshtein Distance Algorithm Tests
 */

import { describe, it, expect } from '../../shared/utils/test-runner'
import { levenshteinDistance, levenshteinSimilarity } from '../domain/use-cases/levenshtein'

describe('Levenshtein Distance', () => {
  it('should return 0 for identical strings', () => {
    expect.toEqual(levenshteinDistance('hello', 'hello'), 0)
    expect.toEqual(levenshteinDistance('test', 'test'), 0)
  })

  it('should return string length for completely different strings', () => {
    expect.toEqual(levenshteinDistance('', 'hello'), 5)
    expect.toEqual(levenshteinDistance('abc', ''), 3)
  })

  it('should calculate single character difference', () => {
    expect.toEqual(levenshteinDistance('cat', 'bat'), 1)
    expect.toEqual(levenshteinDistance('hello', 'hallo'), 1)
  })

  it('should calculate multiple character differences', () => {
    expect.toEqual(levenshteinDistance('kitten', 'sitting'), 3)
    expect.toEqual(levenshteinDistance('saturday', 'sunday'), 3)
  })

  it('should handle empty strings', () => {
    expect.toEqual(levenshteinDistance('', ''), 0)
    expect.toEqual(levenshteinDistance('a', ''), 1)
  })
})

describe('Levenshtein Similarity', () => {
  it('should return 1.0 for identical strings', () => {
    expect.toEqual(levenshteinSimilarity('hello', 'hello'), 1.0)
    expect.toEqual(levenshteinSimilarity('test', 'test'), 1.0)
  })

  it('should return 0.0 for completely different strings of same length', () => {
    const similarity = levenshteinSimilarity('abc', 'xyz')
    expect.toEqual(similarity, 0.0)
  })

  it('should return values between 0 and 1', () => {
    const similarity = levenshteinSimilarity('cat', 'bat')
    expect.toBeGreaterThan(similarity, 0.0)
    expect.toBeLessThan(similarity, 1.0)
  })

  it('should be case-insensitive by default', () => {
    const similarity = levenshteinSimilarity('Hello', 'hello')
    expect.toEqual(similarity, 1.0)
  })

  it('should be case-sensitive when specified', () => {
    const similarity = levenshteinSimilarity('Hello', 'hello', true)
    expect.toBeLessThan(similarity, 1.0)
  })

  it('should handle typos correctly', () => {
    const similarity1 = levenshteinSimilarity('Repository', 'Repo')
    const similarity2 = levenshteinSimilarity('Controller', 'Control')

    expect.toBeGreaterThan(similarity1, 0.3)
    expect.toBeGreaterThan(similarity2, 0.5)
  })
})
