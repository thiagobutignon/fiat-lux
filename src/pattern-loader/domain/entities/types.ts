/**
 * Pattern Loader Domain Types
 *
 * Types for loading and parsing architectural patterns from YAML
 */

export interface Pattern {
  id: string
  name: string
  layer: string
  linguisticRole: string
  description: string
}

export interface YAMLPatternConfig {
  version: string
  architecture: string
  grammarType: string
  patterns: Pattern[]
}
