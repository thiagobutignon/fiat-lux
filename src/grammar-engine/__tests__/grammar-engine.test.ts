/**
 * Grammar Engine Tests
 */

import { describe, it, expect, beforeEach } from '../../shared/utils/test-runner'
import { GrammarEngine } from '../domain/use-cases/grammar-engine'
import { GrammarConfig, SimilarityAlgorithm, Severity } from '../domain/entities/types'
import { SimilarityCache } from '../data/use-cases/similarity-cache-impl'

describe('GrammarEngine - Validation', () => {
  let engine: GrammarEngine
  let cache: SimilarityCache

  beforeEach(() => {
    cache = new SimilarityCache()
    const config: GrammarConfig = {
      roles: {
        Action: {
          values: ['create', 'read', 'update', 'delete'],
          required: true
        },
        Resource: {
          values: ['user', 'post', 'comment'],
          required: true
        }
      }
    }
    engine = new GrammarEngine(config, cache)
  })

  it('should validate valid sentences', () => {
    const result = engine.validate({
      Action: 'create',
      Resource: 'user'
    })

    expect.toEqual(result.errors.length, 0)
    expect.toEqual(result.structuralErrors.length, 0)
  })

  it('should detect invalid values', () => {
    const result = engine.validate({
      Action: 'destroy',
      Resource: 'user'
    })

    expect.toEqual(result.errors.length, 1)
    expect.toEqual(result.errors[0].role, 'Action')
    expect.toEqual(result.errors[0].severity, Severity.ERROR)
  })

  it('should detect missing required fields', () => {
    const result = engine.validate({
      Action: 'create'
      // Resource is missing
    })

    expect.toBeGreaterThan(result.errors.length, 0)
    const missingError = result.errors.find(e => e.role === 'Resource')
    expect.toBeDefined(missingError)
  })

  it('should detect unknown roles', () => {
    const result = engine.validate({
      Action: 'create',
      Resource: 'user',
      UnknownRole: 'value'
    })

    const unknownError = result.errors.find(e => e.role === 'UnknownRole')
    expect.toBeDefined(unknownError)
    expect.toEqual(unknownError!.severity, Severity.WARNING)
  })

  it('should provide suggestions for invalid values', () => {
    const result = engine.validate({
      Action: 'creat',
      Resource: 'user'
    })

    expect.toBeDefined(result.errors[0].suggestions)
    expect.toContain(result.errors[0].suggestions!, 'create')
  })
})

describe('GrammarEngine - Repair', () => {
  let engine: GrammarEngine
  let cache: SimilarityCache

  beforeEach(() => {
    cache = new SimilarityCache()
    const config: GrammarConfig = {
      roles: {
        Action: {
          values: ['create', 'read', 'update', 'delete'],
          required: false
        },
        Resource: {
          values: ['user', 'post', 'comment'],
          required: false
        }
      },
      options: {
        similarityThreshold: 0.6
      }
    }
    engine = new GrammarEngine(config, cache)
  })

  it('should repair simple typos', () => {
    const result = engine.repair({
      Action: 'creat',
      Resource: 'user'
    })

    expect.toEqual(result.repaired.Action, 'create')
    expect.toEqual(result.repairs.length, 1)
  })

  it('should not repair if similarity below threshold', () => {
    const result = engine.repair({
      Action: 'xyz',
      Resource: 'user'
    })

    // Should keep original if no good match
    expect.toEqual(result.repaired.Action, 'xyz')
  })

  it('should provide alternatives in repairs', () => {
    const result = engine.repair({
      Action: 'upd',
      Resource: 'user'
    })

    expect.toEqual(result.repaired.Action, 'update')
    expect.toBeDefined(result.repairs[0].alternatives)
    expect.toBeGreaterThan(result.repairs[0].alternatives!.length, 0)
  })

  it('should repair multiple errors', () => {
    const result = engine.repair({
      Action: 'delet',
      Resource: 'usr'
    })

    expect.toEqual(result.repaired.Action, 'delete')
    expect.toEqual(result.repaired.Resource, 'user')
    expect.toEqual(result.repairs.length, 2)
  })

  it('should include confidence scores', () => {
    const result = engine.repair({
      Action: 'create',
      Resource: 'usr'
    })

    expect.toBeGreaterThan(result.repairs[0].confidence, 0)
    expect.toBeLessThan(result.repairs[0].confidence, 1)
  })
})

describe('GrammarEngine - Process', () => {
  let engine: GrammarEngine
  let cache: SimilarityCache

  beforeEach(() => {
    cache = new SimilarityCache()
    const config: GrammarConfig = {
      roles: {
        Action: {
          values: ['create', 'read'],
          required: true
        }
      },
      options: {
        autoRepair: true
      }
    }
    engine = new GrammarEngine(config, cache)
  })

  it('should return valid result for valid input', () => {
    const result = engine.process({
      Action: 'create'
    })

    expect.toBeTruthy(result.isValid)
    expect.toEqual(result.errors.length, 0)
  })

  it('should auto-repair invalid input when enabled', () => {
    const result = engine.process({
      Action: 'creat'
    })

    expect.toBeFalsy(result.isValid)
    expect.toBeDefined(result.repaired)
    expect.toEqual(result.repaired!.Action, 'create')
  })

  it('should include metadata', () => {
    const result = engine.process({
      Action: 'create'
    })

    expect.toBeDefined(result.metadata)
    expect.toBeDefined(result.metadata.processingTimeMs)
    expect.toBeDefined(result.metadata.cacheHits)
    expect.toBeDefined(result.metadata.algorithmsUsed)
  })
})

describe('GrammarEngine - Cache', () => {
  let engine: GrammarEngine
  let cache: SimilarityCache

  beforeEach(() => {
    cache = new SimilarityCache()
    const config: GrammarConfig = {
      roles: {
        Action: {
          values: ['create', 'read', 'update', 'delete'],
          required: false
        }
      },
      options: {
        enableCache: true
      }
    }
    engine = new GrammarEngine(config, cache)
  })

  it('should use cache for repeated calculations', () => {
    // First process
    engine.process({ Action: 'creat' })
    const stats1 = engine.getCacheStats()

    // Second process with same typo
    engine.process({ Action: 'creat' })
    const stats2 = engine.getCacheStats()

    expect.toBeGreaterThan(stats2.hits, stats1.hits)
  })

  it('should clear cache', () => {
    engine.process({ Action: 'creat' })
    engine.clearCache()

    const stats = engine.getCacheStats()
    expect.toEqual(stats.hits, 0)
    expect.toEqual(stats.size, 0)
  })

  it('should track cache statistics', () => {
    engine.process({ Action: 'creat' })
    const stats = engine.getCacheStats()

    expect.toBeDefined(stats.hits)
    expect.toBeDefined(stats.misses)
    expect.toBeDefined(stats.size)
    expect.toBeDefined(stats.hitRate)
  })
})

describe('GrammarEngine - Options', () => {
  it('should respect similarity threshold', () => {
    const cache = new SimilarityCache()
    const config: GrammarConfig = {
      roles: {
        Action: {
          values: ['create'],
          required: false
        }
      },
      options: {
        similarityThreshold: 0.9 // Very high threshold
      }
    }
    const engine = new GrammarEngine(config, cache)

    const result = engine.repair({ Action: 'xyz' })

    // Should not repair because similarity too low
    expect.toEqual(result.repairs.length, 0)
  })

  it('should allow updating options at runtime', () => {
    const cache = new SimilarityCache()
    const config: GrammarConfig = {
      roles: {
        Action: {
          values: ['create'],
          required: false
        }
      },
      options: {
        autoRepair: false
      }
    }
    const engine = new GrammarEngine(config, cache)

    let result = engine.process({ Action: 'creat' })
    expect.toBeUndefined(result.repaired)

    engine.setOptions({ autoRepair: true })
    result = engine.process({ Action: 'creat' })
    expect.toBeDefined(result.repaired)
  })
})
