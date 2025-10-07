/**
 * fiat-lux.ts (Let There Be Light)
 *
 * A Generic Universal Grammar Engine
 *
 * This module provides a flexible, configurable grammar engine that can validate
 * and auto-repair structured data based on customizable grammatical rules.
 * It's generic enough to work with any domain: code architecture, natural language,
 * configuration files, or any structured data that follows grammatical patterns.
 *
 * Core Principles:
 * - Grammar as Data: Rules are declarative and configurable
 * - Multiple Algorithms: Pluggable similarity and repair strategies
 * - Explainability: Every decision is traceable and reportable
 * - Performance: Caching and optimization for large-scale processing
 * - Type Safety: Full TypeScript support with generics
 *
 * @example
 * ```typescript
 * // Define your grammar
 * const codeGrammar = new GrammarEngine({
 *   roles: {
 *     Subject: { values: ["DbAddAccount", "RemoteAddAccount"], required: true },
 *     Verb: { values: ["add", "delete", "update"], required: true },
 *     Object: { values: ["Account.Params", "Survey.Params"], required: false }
 *   }
 * })
 *
 * // Validate and repair
 * const result = codeGrammar.process({
 *   Subject: "DbAddAccount",
 *   Verb: "ad", // typo
 *   Object: "INVALID"
 * })
 * ```
 */

// ============================================================================
// Core Type Definitions
// ============================================================================

/**
 * Generic record type with string keys and values of type T
 */
export type GenericRecord<T = any> = Record<string, T>

/**
 * Configuration for a single role in the grammar
 */
export interface RoleConfig {
  /** Allowed values for this role */
  values: readonly string[]
  /** Whether this role is required in a valid sentence */
  required?: boolean
  /** Whether this role can have multiple values (array) */
  multiple?: boolean
  /** Custom validator function */
  validator?: (value: any) => boolean
  /** Human-readable description */
  description?: string
}

/**
 * Grammar configuration with all roles
 */
export interface GrammarConfig {
  /** Role definitions */
  roles: Record<string, RoleConfig>
  /** Structural rules (e.g., "if Subject exists, Verb is required") */
  structuralRules?: StructuralRule[]
  /** Global configuration */
  options?: GrammarOptions
}

/**
 * Structural validation rule
 */
export interface StructuralRule {
  /** Rule name for reporting */
  name: string
  /** Validation function */
  validate: (sentence: GenericRecord) => boolean
  /** Error message if validation fails */
  message: string
}

/**
 * Grammar engine configuration options
 */
export interface GrammarOptions {
  /** Similarity threshold for auto-repair (0-1) */
  similarityThreshold?: number
  /** Similarity algorithm to use */
  similarityAlgorithm?: SimilarityAlgorithm
  /** Whether to enable caching */
  enableCache?: boolean
  /** Whether to auto-repair invalid values */
  autoRepair?: boolean
  /** Maximum number of repair suggestions to generate */
  maxSuggestions?: number
  /** Case sensitivity for matching */
  caseSensitive?: boolean
}

/**
 * Available similarity algorithms
 */
export enum SimilarityAlgorithm {
  LEVENSHTEIN = "levenshtein",
  JARO_WINKLER = "jaro-winkler",
  HYBRID = "hybrid"
}

/**
 * Severity level for validation errors
 */
export enum Severity {
  ERROR = "error",
  WARNING = "warning",
  INFO = "info"
}

/**
 * Validation error details
 */
export interface ValidationError {
  role: string
  value: any
  message: string
  severity: Severity
  suggestions?: string[]
}

/**
 * Auto-repair operation details
 */
export interface RepairOperation {
  role: string
  original: any
  replacement: any
  similarity: number
  algorithm: SimilarityAlgorithm
  confidence: number
  message: string
  alternatives?: Array<{ value: string; similarity: number }>
}

/**
 * Processing result
 */
export interface ProcessingResult<T = GenericRecord> {
  original: T
  isValid: boolean
  errors: ValidationError[]
  structuralErrors: string[]
  repaired?: T
  repairs?: RepairOperation[]
  metadata: {
    processingTimeMs: number
    cacheHits: number
    algorithmsUsed: SimilarityAlgorithm[]
  }
}

/**
 * Match candidate with similarity score
 */
export interface MatchCandidate {
  value: string
  similarity: number
  algorithm: SimilarityAlgorithm
}

// ============================================================================
// Similarity Algorithms
// ============================================================================

/**
 * Calculates Levenshtein distance between two strings
 * Time complexity: O(m*n) where m, n are string lengths
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
 * Calculates Jaro-Winkler similarity between two strings
 * Better for short strings and typos at the beginning
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

// ============================================================================
// Similarity Cache
// ============================================================================

/**
 * Cache for similarity calculations
 * Improves performance for repeated comparisons
 */
class SimilarityCache {
  private cache = new Map<string, number>()
  private hits = 0
  private misses = 0

  private getCacheKey(a: string, b: string, algorithm: SimilarityAlgorithm): string {
    return `${algorithm}:${a}:${b}`
  }

  get(a: string, b: string, algorithm: SimilarityAlgorithm): number | undefined {
    const key = this.getCacheKey(a, b, algorithm)
    const value = this.cache.get(key)
    if (value !== undefined) this.hits++
    else this.misses++
    return value
  }

  set(a: string, b: string, algorithm: SimilarityAlgorithm, value: number): void {
    const key = this.getCacheKey(a, b, algorithm)
    this.cache.set(key, value)
  }

  clear(): void {
    this.cache.clear()
    this.hits = 0
    this.misses = 0
  }

  getHits(): number {
    return this.hits
  }

  getStats(): { hits: number; misses: number; size: number; hitRate: number } {
    const total = this.hits + this.misses
    return {
      hits: this.hits,
      misses: this.misses,
      size: this.cache.size,
      hitRate: total > 0 ? this.hits / total : 0
    }
  }
}

// ============================================================================
// Main Grammar Engine
// ============================================================================

/**
 * Universal Grammar Engine
 * Generic, configurable grammar validation and auto-repair system
 */
export class GrammarEngine<T extends GenericRecord = GenericRecord> {
  private config: GrammarConfig
  private options: Required<GrammarOptions>
  private cache: SimilarityCache
  private algorithmMap: Map<SimilarityAlgorithm, (a: string, b: string, caseSensitive: boolean) => number>

  constructor(config: GrammarConfig) {
    this.config = config

    // Set default options
    this.options = {
      similarityThreshold: config.options?.similarityThreshold ?? 0.6,
      similarityAlgorithm: config.options?.similarityAlgorithm ?? SimilarityAlgorithm.HYBRID,
      enableCache: config.options?.enableCache ?? true,
      autoRepair: config.options?.autoRepair ?? true,
      maxSuggestions: config.options?.maxSuggestions ?? 3,
      caseSensitive: config.options?.caseSensitive ?? false
    }

    this.cache = new SimilarityCache()

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

// ============================================================================
// Predefined Grammar Configurations
// ============================================================================

/**
 * Example: Clean Architecture grammar for code
 */
export const CLEAN_ARCHITECTURE_GRAMMAR: GrammarConfig = {
  roles: {
    Subject: {
      values: ["DbAddAccount", "RemoteAddAccount", "DbLoadSurvey", "RemoteLoadSurvey", "DbAuthentication", "RemoteAuthentication"],
      required: true,
      description: "The main actor/component performing the action"
    },
    Verb: {
      values: ["add", "delete", "update", "load", "save", "authenticate", "validate"],
      required: true,
      description: "The action being performed"
    },
    Object: {
      values: ["Account.Params", "Survey.Params", "User.Params", "Entity.Data", "Auth.Credentials"],
      required: false,
      description: "The data being acted upon"
    },
    Adverb: {
      values: ["Hasher", "Repository", "ApiAdapter", "Validator", "Encrypter", "TokenGenerator"],
      required: false,
      multiple: true,
      description: "Modifiers that describe how the action is performed"
    },
    Context: {
      values: ["Controller", "MainFactory", "Service", "UseCase", "Presenter"],
      required: false,
      description: "The architectural layer/context"
    }
  },
  structuralRules: [
    {
      name: "VerbObjectAlignment",
      validate: (s) => {
        if (s.Verb === "authenticate" && s.Object && !s.Object.includes("Auth")) {
          return false
        }
        return true
      },
      message: "Verb 'authenticate' requires an Auth-related Object"
    }
  ],
  options: {
    similarityThreshold: 0.65,
    similarityAlgorithm: SimilarityAlgorithm.HYBRID,
    enableCache: true,
    autoRepair: true,
    maxSuggestions: 3,
    caseSensitive: false
  }
}

/**
 * Example: HTTP API grammar
 */
export const HTTP_API_GRAMMAR: GrammarConfig = {
  roles: {
    Method: {
      values: ["GET", "POST", "PUT", "PATCH", "DELETE"],
      required: true,
      description: "HTTP method"
    },
    Resource: {
      values: ["/users", "/posts", "/comments", "/auth", "/profiles"],
      required: true,
      description: "API resource path"
    },
    Status: {
      values: ["200", "201", "400", "401", "403", "404", "500"],
      required: false,
      description: "HTTP status code"
    },
    Handler: {
      values: ["Controller", "Middleware", "Guard", "Interceptor"],
      required: false,
      multiple: true,
      description: "Request handlers"
    }
  },
  options: {
    similarityThreshold: 0.7,
    similarityAlgorithm: SimilarityAlgorithm.LEVENSHTEIN,
    caseSensitive: true
  }
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Format processing result for display
 */
export function formatResult<T extends GenericRecord>(result: ProcessingResult<T>): string {
  const lines: string[] = []

  lines.push("═".repeat(80))
  lines.push("FIAT LUX - Grammar Engine Processing Result")
  lines.push("═".repeat(80))
  lines.push("")
  lines.push("📝 Original Input:")
  lines.push(JSON.stringify(result.original, null, 2))
  lines.push("")

  if (result.isValid) {
    lines.push("✅ Status: VALID")
    lines.push("   All grammatical rules satisfied")
  } else {
    lines.push("❌ Status: INVALID")
    lines.push("")

    if (result.errors.length > 0) {
      lines.push(`   Validation Errors (${result.errors.length}):`)
      result.errors.forEach((error, i) => {
        lines.push(`   ${i + 1}. [${error.severity.toUpperCase()}] ${error.message}`)
        if (error.suggestions && error.suggestions.length > 0) {
          lines.push(`      Suggestions: ${error.suggestions.join(", ")}`)
        }
      })
      lines.push("")
    }

    if (result.structuralErrors.length > 0) {
      lines.push(`   Structural Errors (${result.structuralErrors.length}):`)
      result.structuralErrors.forEach((error, i) => {
        lines.push(`   ${i + 1}. ${error}`)
      })
      lines.push("")
    }

    if (result.repairs && result.repairs.length > 0) {
      lines.push("🔧 Auto-Repair Applied:")
      result.repairs.forEach((repair, i) => {
        lines.push(`   ${i + 1}. ${repair.message}`)
        lines.push(`      Algorithm: ${repair.algorithm}, Confidence: ${(repair.confidence * 100).toFixed(1)}%`)
        if (repair.alternatives && repair.alternatives.length > 0) {
          lines.push(`      Alternatives: ${repair.alternatives.map(a => `${a.value} (${(a.similarity * 100).toFixed(1)}%)`).join(", ")}`)
        }
      })
      lines.push("")
      lines.push("✅ Repaired Output:")
      lines.push(JSON.stringify(result.repaired, null, 2))
      lines.push("")
    }
  }

  lines.push("📊 Metadata:")
  lines.push(`   Processing time: ${result.metadata.processingTimeMs.toFixed(2)}ms`)
  lines.push(`   Cache hits: ${result.metadata.cacheHits}`)
  lines.push(`   Algorithms: ${result.metadata.algorithmsUsed.join(", ")}`)
  lines.push("")
  lines.push("═".repeat(80))

  return lines.join("\n")
}

// ============================================================================
// Demo & Testing
// ============================================================================

/**
 * Run comprehensive demonstration
 */
export function runDemo(): void {
  console.log("\n🌟 FIAT LUX - Universal Grammar Engine\n")
  console.log("A generic, configurable system for validating and repairing structured data\n")

  // Demo 1: Clean Architecture validation
  console.log("═".repeat(80))
  console.log("Demo 1: Clean Architecture - Invalid Object Token")
  console.log("═".repeat(80))

  const engine1 = new GrammarEngine(CLEAN_ARCHITECTURE_GRAMMAR)
  const result1 = engine1.process({
    Subject: "DbAddAccount",
    Verb: "add",
    Object: "BLABAKABA", // noise
    Adverbs: ["Hasher", "Repository"],
    Context: "Controller"
  })
  console.log(formatResult(result1))

  // Demo 2: Multiple errors with different algorithms
  console.log("\n" + "═".repeat(80))
  console.log("Demo 2: Multiple Invalid Tokens with Hybrid Algorithm")
  console.log("═".repeat(80))

  const engine2 = new GrammarEngine(CLEAN_ARCHITECTURE_GRAMMAR)
  const result2 = engine2.process({
    Subject: "RemoteLoadSurvey",
    Verb: "lod", // typo: should be "load"
    Object: "UserParams", // close to "User.Params"
    Adverbs: ["Hash", "Validatr"], // typos
    Context: "Facto" // typo: should be "MainFactory"
  })
  console.log(formatResult(result2))

  // Demo 3: Valid sentence
  console.log("\n" + "═".repeat(80))
  console.log("Demo 3: Valid Sentence - No Repairs Needed")
  console.log("═".repeat(80))

  const engine3 = new GrammarEngine(CLEAN_ARCHITECTURE_GRAMMAR)
  const result3 = engine3.process({
    Subject: "DbLoadSurvey",
    Verb: "load",
    Object: "Survey.Params",
    Adverbs: ["Repository", "Validator"],
    Context: "UseCase"
  })
  console.log(formatResult(result3))

  // Demo 4: HTTP API Grammar
  console.log("\n" + "═".repeat(80))
  console.log("Demo 4: HTTP API Grammar - Different Domain")
  console.log("═".repeat(80))

  const engine4 = new GrammarEngine(HTTP_API_GRAMMAR)
  const result4 = engine4.process({
    Method: "PSOT", // typo: should be "POST"
    Resource: "/user", // typo: should be "/users"
    Status: "201",
    Handler: ["Controller", "Middleware"]
  })
  console.log(formatResult(result4))

  // Demo 5: Algorithm comparison
  console.log("\n" + "═".repeat(80))
  console.log("Demo 5: Algorithm Comparison")
  console.log("═".repeat(80))

  const testPairs = [
    ["add", "ad"],
    ["Repository", "Repo"],
    ["Controller", "Control"],
    ["Account.Params", "AccountParams"],
  ]

  console.log("\nComparing Levenshtein, Jaro-Winkler, and Hybrid algorithms:\n")
  testPairs.forEach(([correct, typo]) => {
    const lev = levenshteinSimilarity(correct, typo, false)
    const jaro = jaroWinklerSimilarity(correct.toLowerCase(), typo.toLowerCase())
    const hybrid = hybridSimilarity(correct, typo, false)

    console.log(`"${typo}" → "${correct}"`)
    console.log(`  Levenshtein:  ${(lev * 100).toFixed(1)}%`)
    console.log(`  Jaro-Winkler: ${(jaro * 100).toFixed(1)}%`)
    console.log(`  Hybrid:       ${(hybrid * 100).toFixed(1)}%`)
    console.log("")
  })

  // Demo 6: Cache performance
  console.log("\n" + "═".repeat(80))
  console.log("Demo 6: Cache Performance")
  console.log("═".repeat(80))

  const engine6 = new GrammarEngine(CLEAN_ARCHITECTURE_GRAMMAR)

  // Process same sentence multiple times
  const testSentence = {
    Subject: "DbAddAcount", // typo
    Verb: "ad", // typo
    Object: "Acount.Params" // typo
  }

  console.log("\nProcessing same sentence 100 times to test caching...\n")
  const iterations = 100
  const startTime = performance.now()

  for (let i = 0; i < iterations; i++) {
    engine6.process(testSentence)
  }

  const endTime = performance.now()
  const stats = engine6.getCacheStats()

  console.log(`Total time: ${(endTime - startTime).toFixed(2)}ms`)
  console.log(`Average per iteration: ${((endTime - startTime) / iterations).toFixed(2)}ms`)
  console.log(`Cache stats:`)
  console.log(`  - Hits: ${stats.hits}`)
  console.log(`  - Misses: ${stats.misses}`)
  console.log(`  - Hit rate: ${(stats.hitRate * 100).toFixed(1)}%`)
  console.log(`  - Cache size: ${stats.size} entries`)

  console.log("\n✨ Demo complete!\n")
}

// ============================================================================
// Run if executed directly
// ============================================================================

if (require.main === module) {
  runDemo()
}
