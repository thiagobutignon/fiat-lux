/**
 * Grammar Engine Use Case
 *
 * Main business logic for grammar validation and auto-repair
 */

import {
  GenericRecord,
  GrammarConfig,
  GrammarOptions,
  SimilarityAlgorithm,
  ValidationError,
  RepairOperation,
  ProcessingResult,
  MatchCandidate,
  Severity
} from '../entities/types'
import { ISimilarityCache } from '../../data/protocols/similarity-cache'
import { levenshteinSimilarity } from '../../../similarity-algorithms/domain/use-cases/levenshtein'
import { jaroWinklerSimilarity } from '../../../similarity-algorithms/domain/use-cases/jaro-winkler'
import { hybridSimilarity } from '../../../similarity-algorithms/domain/use-cases/hybrid'

/**
 * Universal Grammar Engine
 * Generic, configurable grammar validation and auto-repair system
 */
export class GrammarEngine<T extends GenericRecord = GenericRecord> {
  private config: GrammarConfig
  private options: Required<GrammarOptions>
  private cache: ISimilarityCache
  private algorithmMap: Map<SimilarityAlgorithm, (a: string, b: string, caseSensitive: boolean) => number>

  constructor(config: GrammarConfig, cache: ISimilarityCache) {
    this.config = config
    this.cache = cache

    // Set default options
    this.options = {
      similarityThreshold: config.options?.similarityThreshold ?? 0.6,
      similarityAlgorithm: config.options?.similarityAlgorithm ?? SimilarityAlgorithm.HYBRID,
      enableCache: config.options?.enableCache ?? true,
      autoRepair: config.options?.autoRepair ?? true,
      maxSuggestions: config.options?.maxSuggestions ?? 3,
      caseSensitive: config.options?.caseSensitive ?? false
    }

    // Initialize algorithm map
    this.algorithmMap = new Map([
      [SimilarityAlgorithm.LEVENSHTEIN, levenshteinSimilarity],
      [SimilarityAlgorithm.JARO_WINKLER, jaroWinklerSimilarity],
      [SimilarityAlgorithm.HYBRID, hybridSimilarity]
    ])
  }

  /**
   * Calculate similarity between two strings using configured algorithm
   */
  private calculateSimilarity(a: string, b: string, algorithm?: SimilarityAlgorithm): number {
    const algo = algorithm ?? this.options.similarityAlgorithm

    // Check cache first
    if (this.options.enableCache) {
      const cached = this.cache.get(a, b, algo)
      if (cached !== undefined) return cached
    }

    // Calculate similarity
    const similarityFn = this.algorithmMap.get(algo)
    if (!similarityFn) {
      throw new Error(`Unknown similarity algorithm: ${algo}`)
    }

    const similarity = similarityFn(a, b, this.options.caseSensitive)

    // Cache result
    if (this.options.enableCache) {
      this.cache.set(a, b, algo, similarity)
    }

    return similarity
  }

  /**
   * Find best matches for an invalid value
   */
  private findBestMatches(role: string, invalidValue: string): MatchCandidate[] {
    const roleConfig = this.config.roles[role]
    if (!roleConfig) return []

    const candidates: MatchCandidate[] = roleConfig.values.map(value => ({
      value,
      similarity: this.calculateSimilarity(invalidValue, value),
      algorithm: this.options.similarityAlgorithm
    }))

    // Sort by similarity (descending)
    candidates.sort((a, b) => b.similarity - a.similarity)

    return candidates
  }

  /**
   * Validate a single value against role configuration
   */
  private validateValue(role: string, value: any): ValidationError | undefined {
    const roleConfig = this.config.roles[role]
    if (!roleConfig) {
      return {
        role,
        value,
        message: `Unknown role: "${role}"`,
        severity: Severity.ERROR
      }
    }

    // Handle array values
    if (roleConfig.multiple && Array.isArray(value)) {
      return undefined // Validate each item separately
    }

    // Check if value is in allowed list
    if (!roleConfig.values.includes(value)) {
      const suggestions = this.findBestMatches(role, value)
        .slice(0, this.options.maxSuggestions)
        .map(c => c.value)

      return {
        role,
        value,
        message: `Invalid ${role}: "${value}" is not in allowed vocabulary`,
        severity: Severity.ERROR,
        suggestions
      }
    }

    // Custom validator
    if (roleConfig.validator && !roleConfig.validator(value)) {
      return {
        role,
        value,
        message: `Custom validation failed for ${role}: "${value}"`,
        severity: Severity.ERROR
      }
    }

    return undefined
  }

  /**
   * Validate entire sentence structure
   */
  validate(sentence: T): { errors: ValidationError[]; structuralErrors: string[] } {
    const errors: ValidationError[] = []
    const structuralErrors: string[] = []

    // Validate each field
    for (const [role, value] of Object.entries(sentence)) {
      if (value === undefined || value === null) continue

      const roleConfig = this.config.roles[role]
      if (!roleConfig) {
        errors.push({
          role,
          value,
          message: `Unknown role: "${role}"`,
          severity: Severity.WARNING
        })
        continue
      }

      // Handle array values
      if (Array.isArray(value)) {
        if (!roleConfig.multiple) {
          errors.push({
            role,
            value,
            message: `${role} should not be an array`,
            severity: Severity.ERROR
          })
        } else {
          for (const item of value) {
            const error = this.validateValue(role, item)
            if (error) errors.push(error)
          }
        }
      } else {
        const error = this.validateValue(role, value)
        if (error) errors.push(error)
      }
    }

    // Check required fields
    for (const [role, roleConfig] of Object.entries(this.config.roles)) {
      if (roleConfig.required && !(role in sentence)) {
        errors.push({
          role,
          value: undefined,
          message: `Required role "${role}" is missing`,
          severity: Severity.ERROR
        })
      }
    }

    // Validate structural rules
    if (this.config.structuralRules) {
      for (const rule of this.config.structuralRules) {
        if (!rule.validate(sentence)) {
          structuralErrors.push(`${rule.name}: ${rule.message}`)
        }
      }
    }

    return { errors, structuralErrors }
  }

  /**
   * Attempt to repair invalid values
   */
  repair(sentence: T): { repaired: T; repairs: RepairOperation[] } {
    const repaired = { ...sentence } as T
    const repairs: RepairOperation[] = []

    for (const [role, value] of Object.entries(sentence)) {
      if (value === undefined || value === null) continue

      const roleConfig = this.config.roles[role]
      if (!roleConfig) continue

      // Handle array values
      if (Array.isArray(value)) {
        if (roleConfig.multiple) {
          const repairedArray: any[] = []
          for (const item of value) {
            if (!roleConfig.values.includes(item)) {
              const candidates = this.findBestMatches(role, item)
              if (candidates.length > 0 && candidates[0].similarity >= this.options.similarityThreshold) {
                repairedArray.push(candidates[0].value)
                repairs.push({
                  role,
                  original: item,
                  replacement: candidates[0].value,
                  similarity: candidates[0].similarity,
                  algorithm: candidates[0].algorithm,
                  confidence: candidates[0].similarity,
                  message: `Repaired ${role}[]: "${item}" → "${candidates[0].value}" (${(candidates[0].similarity * 100).toFixed(1)}%)`,
                  alternatives: candidates.slice(1, this.options.maxSuggestions).map(c => ({
                    value: c.value,
                    similarity: c.similarity
                  }))
                })
              } else {
                repairedArray.push(item) // Keep original if no good match
              }
            } else {
              repairedArray.push(item)
            }
          }
          (repaired as any)[role] = repairedArray
        }
      } else {
        // Single value
        if (!roleConfig.values.includes(value)) {
          const candidates = this.findBestMatches(role, value)
          if (candidates.length > 0 && candidates[0].similarity >= this.options.similarityThreshold) {
            (repaired as any)[role] = candidates[0].value
            repairs.push({
              role,
              original: value,
              replacement: candidates[0].value,
              similarity: candidates[0].similarity,
              algorithm: candidates[0].algorithm,
              confidence: candidates[0].similarity,
              message: `Repaired ${role}: "${value}" → "${candidates[0].value}" (${(candidates[0].similarity * 100).toFixed(1)}%)`,
              alternatives: candidates.slice(1, this.options.maxSuggestions).map(c => ({
                value: c.value,
                similarity: c.similarity
              }))
            })
          }
        }
      }
    }

    return { repaired, repairs }
  }

  /**
   * Process a sentence: validate and optionally repair
   */
  process(sentence: T): ProcessingResult<T> {
    const startTime = performance.now()

    const { errors, structuralErrors } = this.validate(sentence)
    const isValid = errors.length === 0 && structuralErrors.length === 0

    let repaired: T | undefined
    let repairs: RepairOperation[] | undefined

    if (!isValid && this.options.autoRepair) {
      const repairResult = this.repair(sentence)
      repaired = repairResult.repaired
      repairs = repairResult.repairs
    }

    const endTime = performance.now()

    return {
      original: sentence,
      isValid,
      errors,
      structuralErrors,
      repaired,
      repairs,
      metadata: {
        processingTimeMs: endTime - startTime,
        cacheHits: this.cache.getHits(),
        algorithmsUsed: [this.options.similarityAlgorithm]
      }
    }
  }

  /**
   * Get cache statistics
   */
  getCacheStats() {
    return this.cache.getStats()
  }

  /**
   * Clear cache
   */
  clearCache(): void {
    this.cache.clear()
  }

  /**
   * Update options at runtime
   */
  setOptions(options: Partial<GrammarOptions>): void {
    Object.assign(this.options, options)
  }

  /**
   * Get current configuration
   */
  getConfig(): Readonly<GrammarConfig> {
    return this.config
  }

  /**
   * Get current options
   */
  getOptions(): Readonly<GrammarOptions> {
    return this.options
  }
}
