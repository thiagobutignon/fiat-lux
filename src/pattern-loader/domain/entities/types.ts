/**
 * Pattern Loader Domain Types
 *
 * Complete type definitions for loading and parsing architectural patterns from YAML
 */

// ============================================================================
// Core Pattern Types
// ============================================================================

export interface Pattern {
  id: string
  name: string
  layer: string
  linguisticRole: string
  description: string
  violations?: Violation[]
}

export interface Violation {
  type: string
  message: string
  explanation: string
  severity: 'error' | 'warning' | 'info'
  suggestion?: string
}

// ============================================================================
// Naming Conventions
// ============================================================================

export interface NamingConvention {
  pattern?: string
  patterns?: string[]
  example?: string
  verbStarts?: string[]
}

export interface LayerNamingConventions {
  [element: string]: NamingConvention
}

export interface NamingConventions {
  [layer: string]: LayerNamingConventions
}

// ============================================================================
// Dependency Rules
// ============================================================================

export interface DependencyRule {
  from: string
  to: string[]
}

export interface ForbiddenRule extends DependencyRule {
  reason: string
  severity: string
  type?: string
}

export interface DependencyRules {
  allowed?: DependencyRule[]
  forbidden?: ForbiddenRule[]
}

// ============================================================================
// Main Configuration
// ============================================================================

export interface YAMLPatternConfig {
  version: string
  architecture: string
  grammarType: string
  patterns: Pattern[]
  namingConventions?: NamingConventions
  dependencyRules?: DependencyRules
}

// ============================================================================
// Validation Results
// ============================================================================

export interface NamingValidationResult {
  valid: boolean
  value: string
  layer: string
  element: string
  pattern?: string
  message?: string
}

export interface DependencyValidationResult {
  valid: boolean
  from: string
  to: string
  rule?: ForbiddenRule
  message?: string
}
