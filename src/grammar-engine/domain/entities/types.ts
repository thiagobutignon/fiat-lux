/**
 * Grammar Engine Domain Types
 *
 * Core type definitions for the grammar engine domain layer.
 * These types define the structure of grammatical rules, validation results,
 * and repair operations.
 */

// ============================================================================
// Core Types
// ============================================================================

/**
 * Generic record type with string keys and values of type T
 */
export type GenericRecord<T = any> = Record<string, T>

/**
 * Grammatical role in architectural sentences
 */
export type Role = string

// ============================================================================
// Configuration Types
// ============================================================================

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

// ============================================================================
// Enums
// ============================================================================

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

// ============================================================================
// Result Types
// ============================================================================

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
