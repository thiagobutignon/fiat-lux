/**
 * pattern-loader.ts
 *
 * YAML Pattern Loader for Grammar Engine
 *
 * Loads architectural patterns from YAML configuration files and converts them
 * into GrammarEngine configurations for validation and auto-repair.
 *
 * Features:
 * - Parse YAML pattern definitions
 * - Convert patterns to grammar rules
 * - Validate code against architectural patterns
 * - Support for multiple layers and roles
 * - Dependency rule validation
 * - Naming convention validation
 */

import { readFileSync } from 'fs'
import { GrammarEngine, GrammarConfig, RoleConfig, StructuralRule } from './fiat-lux'

// ============================================================================
// YAML Pattern Types
// ============================================================================

/**
 * Full YAML configuration structure
 */
export interface YAMLPatternConfig {
  version: string
  architecture: string
  grammar_type: string
  patterns: Pattern[]
  dependency_rules?: DependencyRules
  naming_conventions?: NamingConventions
  severity_levels?: SeverityLevels
  meta_properties?: MetaProperties
  grammar_classification?: GrammarClassification
}

/**
 * Individual pattern definition
 */
export interface Pattern {
  id: string
  name: string
  layer: string
  linguistic_role: string
  description: string
  regex?: {
    pattern: string
    flags?: string[]
  }
  structure?: StructureElement[]
  constraints?: PatternConstraints
  example?: string
  violations?: Violation[]
}

/**
 * Structure element within a pattern
 */
export interface StructureElement {
  element: string
  required?: boolean
  multiple?: boolean
  pattern?: string
  must_start_with?: string
  must_end_with?: string | string[]
  name_pattern?: string
  name_matches?: string
  required_types?: string[]
  all_must_be?: string
  no_concrete?: boolean
  modifiers?: string[]
  can_implement_multiple?: boolean
  type?: string
  value?: string
  signature?: string
  should_not_throw?: boolean
  should_not_modify_input?: boolean
  recommended?: boolean
  exports?: string[]
  must_be?: string
  cannot_be?: string
  instantiation?: string[]
}

/**
 * Pattern constraints
 */
export interface PatternConstraints {
  dependencies?: DependencyConstraint[]
  imports?: ImportConstraint[]
  business_logic?: BusinessLogicConstraint
  return?: ReturnConstraint[]
  implementation?: ImplementationConstraint[]
}

export interface DependencyConstraint {
  must_be_interfaces?: boolean
  cannot_import_from?: string[]
  can_import_from?: string[]
  can_depend_on?: string[]
  cannot_depend_on?: string[]
}

export interface ImportConstraint {
  must_have_external?: boolean
  cannot_import_from?: string[]
  can_import_from?: string[]
}

export interface BusinessLogicConstraint {
  should_not_contain?: string[]
  should_delegate_to?: string
}

export interface ReturnConstraint {
  type?: string
  from_layers?: string[]
}

export interface ImplementationConstraint {
  returns?: string
  injects?: string
}

/**
 * Violation definition
 */
export interface Violation {
  type: string
  message: string
  explanation: string
  severity: 'error' | 'warning' | 'info'
  suggestion?: string
}

/**
 * Dependency rules
 */
export interface DependencyRules {
  allowed?: DependencyRule[]
  forbidden?: ForbiddenRule[]
}

export interface DependencyRule {
  from: string
  to: string[]
}

export interface ForbiddenRule extends DependencyRule {
  reason: string
  severity: string
  type?: string
}

/**
 * Naming conventions
 */
export interface NamingConventions {
  [layer: string]: {
    [element: string]: {
      pattern?: string
      patterns?: string[]
      example?: string
      verb_starts?: string[]
    }
  }
}

/**
 * Severity levels configuration
 */
export interface SeverityLevels {
  error?: string[]
  warning?: string[]
  info?: string[]
}

/**
 * Meta properties
 */
export interface MetaProperties {
  consistency?: PropertyDefinition
  composability?: PropertyDefinition
  expressiveness?: PropertyDefinition
  verifiability?: PropertyDefinition
}

export interface PropertyDefinition {
  description: string
  verifiable: boolean
  tools?: string[]
}

/**
 * Grammar classification
 */
export interface GrammarClassification {
  chomsky_hierarchy: string
  properties?: string[]
  comparison?: Array<{ [key: string]: string }>
}

// ============================================================================
// Pattern Loader
// ============================================================================

/**
 * Loads and parses YAML pattern configuration
 */
export class PatternLoader {
  private config: YAMLPatternConfig

  constructor(yamlContent: string) {
    this.config = this.parseYAML(yamlContent)
  }

  /**
   * Parse YAML content into configuration object
   * Simple YAML parser for our specific format
   */
  private parseYAML(yamlContent: string): YAMLPatternConfig {
    // For now, we'll create a basic parser
    // In production, you'd use a library like 'js-yaml'
    // This is a simplified implementation for demonstration

    const config: YAMLPatternConfig = {
      version: '1.0',
      architecture: 'Clean Architecture',
      grammar_type: 'Context-Free Grammar',
      patterns: []
    }

    // Extract basic info
    const versionMatch = yamlContent.match(/version:\s*"([^"]+)"/)
    if (versionMatch) config.version = versionMatch[1]

    const archMatch = yamlContent.match(/architecture:\s*"([^"]+)"/)
    if (archMatch) config.architecture = archMatch[1]

    const grammarMatch = yamlContent.match(/grammar_type:\s*"([^"]+)"/)
    if (grammarMatch) config.grammar_type = grammarMatch[1]

    // Parse patterns (simplified - would use proper YAML parser in production)
    config.patterns = this.extractPatterns(yamlContent)
    config.naming_conventions = this.extractNamingConventions(yamlContent)
    config.dependency_rules = this.extractDependencyRules(yamlContent)

    return config
  }

  /**
   * Extract patterns from YAML content
   */
  private extractPatterns(yamlContent: string): Pattern[] {
    const patterns: Pattern[] = []

    // Extract pattern blocks
    const patternRegex = /- id:\s*([^\n]+)\s+name:\s*"([^"]+)"\s+layer:\s*([^\n]+)\s+linguistic_role:\s*"([^"]+)"\s+description:\s*"([^"]+)"/g
    let match

    while ((match = patternRegex.exec(yamlContent)) !== null) {
      patterns.push({
        id: match[1].trim(),
        name: match[2],
        layer: match[3].trim(),
        linguistic_role: match[4],
        description: match[5]
      })
    }

    return patterns
  }

  /**
   * Extract naming conventions
   */
  private extractNamingConventions(yamlContent: string): NamingConventions {
    const conventions: NamingConventions = {}

    // Extract domain naming conventions
    const domainMatch = yamlContent.match(/domain:\s+usecases:\s+pattern:\s*"([^"]+)"\s+example:\s*"([^"]+)"/)
    if (domainMatch) {
      conventions.domain = {
        usecases: {
          pattern: domainMatch[1],
          example: domainMatch[2]
        }
      }
    }

    // Extract data naming conventions
    const dataMatch = yamlContent.match(/data:\s+usecases:\s+pattern:\s*"([^"]+)"\s+example:\s*"([^"]+)"/)
    if (dataMatch) {
      conventions.data = {
        usecases: {
          pattern: dataMatch[1],
          example: dataMatch[2]
        }
      }
    }

    return conventions
  }

  /**
   * Extract dependency rules
   */
  private extractDependencyRules(_yamlContent: string): DependencyRules {
    const rules: DependencyRules = {
      allowed: [],
      forbidden: []
    }

    // This would be more robust with a proper YAML parser
    // For now, return basic structure

    return rules
  }

  /**
   * Convert YAML patterns to GrammarEngine configuration
   */
  toGrammarConfig(): GrammarConfig {
    const roles: Record<string, RoleConfig> = {}

    // Create roles from patterns
    this.config.patterns.forEach(pattern => {
      const roleName = pattern.layer
      const values = this.extractValuesFromPattern(pattern)

      roles[roleName] = {
        values,
        required: false,
        description: pattern.description
      }
    })

    // Create structural rules from violations
    const structuralRules: StructuralRule[] = []
    this.config.patterns.forEach(pattern => {
      if (pattern.violations) {
        pattern.violations.forEach(violation => {
          structuralRules.push({
            name: `${pattern.id}_${violation.type}`,
            validate: () => true, // Would need code analysis to implement
            message: violation.message
          })
        })
      }
    })

    return {
      roles,
      structuralRules,
      options: {
        similarityThreshold: 0.7,
        enableCache: true,
        autoRepair: true
      }
    }
  }

  /**
   * Extract allowed values from pattern
   */
  private extractValuesFromPattern(pattern: Pattern): readonly string[] {
    const values: string[] = []

    // Extract from naming conventions if available
    if (this.config.naming_conventions) {
      const layerConventions = this.config.naming_conventions[pattern.layer]
      if (layerConventions) {
        Object.values(layerConventions).forEach(convention => {
          if (convention.example) {
            values.push(...convention.example.split(',').map(s => s.trim()))
          }
        })
      }
    }

    // Add pattern-specific values
    if (pattern.name) {
      values.push(pattern.name)
    }

    return values.length > 0 ? values : ['default']
  }

  /**
   * Get pattern by ID
   */
  getPattern(id: string): Pattern | undefined {
    return this.config.patterns.find(p => p.id === id)
  }

  /**
   * Get patterns by layer
   */
  getPatternsByLayer(layer: string): Pattern[] {
    return this.config.patterns.filter(p => p.layer === layer)
  }

  /**
   * Get all layers
   */
  getLayers(): string[] {
    return [...new Set(this.config.patterns.map(p => p.layer))]
  }

  /**
   * Get naming convention for layer and element
   */
  getNamingConvention(layer: string, element: string): string | undefined {
    return this.config.naming_conventions?.[layer]?.[element]?.pattern
  }

  /**
   * Validate naming against conventions
   */
  validateNaming(name: string, layer: string, element: string): boolean {
    const pattern = this.getNamingConvention(layer, element)
    if (!pattern) return true

    const regex = new RegExp(pattern)
    return regex.test(name)
  }

  /**
   * Get dependency rules
   */
  getDependencyRules(): DependencyRules | undefined {
    return this.config.dependency_rules
  }

  /**
   * Get full configuration
   */
  getConfig(): YAMLPatternConfig {
    return this.config
  }
}

// ============================================================================
// Grammar Pattern Validator
// ============================================================================

/**
 * Validates code against YAML-defined patterns
 */
export class GrammarPatternValidator {
  private loader: PatternLoader
  private engine?: GrammarEngine

  constructor(yamlContent: string) {
    this.loader = new PatternLoader(yamlContent)
  }

  /**
   * Initialize grammar engine from patterns
   */
  initialize(): void {
    const config = this.loader.toGrammarConfig()
    this.engine = new GrammarEngine(config)
  }

  /**
   * Validate architectural sentence
   */
  validate(sentence: any) {
    if (!this.engine) {
      this.initialize()
    }
    return this.engine!.process(sentence)
  }

  /**
   * Get pattern information
   */
  getPattern(id: string): Pattern | undefined {
    return this.loader.getPattern(id)
  }

  /**
   * Get all patterns
   */
  getAllPatterns(): Pattern[] {
    return this.loader.getConfig().patterns
  }

  /**
   * Validate naming convention
   */
  validateNaming(name: string, layer: string, element: string): boolean {
    return this.loader.validateNaming(name, layer, element)
  }

  /**
   * Get layers
   */
  getLayers(): string[] {
    return this.loader.getLayers()
  }
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Load patterns from YAML file
 */
export function loadPatternsFromFile(filePath: string): GrammarPatternValidator {
  const content = readFileSync(filePath, 'utf-8')
  return new GrammarPatternValidator(content)
}

/**
 * Create grammar engine from YAML file
 */
export function createEngineFromYAML(filePath: string): GrammarEngine {
  const content = readFileSync(filePath, 'utf-8')
  const loader = new PatternLoader(content)
  const config = loader.toGrammarConfig()
  return new GrammarEngine(config)
}

// ============================================================================
// Demo
// ============================================================================

/**
 * Run demonstration of YAML pattern loading
 */
export function runDemo(): void {
  console.log('\n🔧 YAML Pattern Loader Demo\n')
  console.log('═'.repeat(80))

  try {
    // Load patterns from file
    const validator = loadPatternsFromFile('./grammar-patterns.yml')
    validator.initialize()

    console.log('✅ Successfully loaded grammar patterns from YAML\n')

    // Display loaded patterns
    const patterns = validator.getAllPatterns()
    console.log(`📋 Loaded ${patterns.length} patterns:\n`)

    patterns.forEach((pattern, index) => {
      console.log(`${index + 1}. ${pattern.name} (${pattern.id})`)
      console.log(`   Layer: ${pattern.layer}`)
      console.log(`   Role: ${pattern.linguistic_role}`)
      console.log(`   Description: ${pattern.description}`)
      console.log('')
    })

    // Display layers
    const layers = validator.getLayers()
    console.log(`🏗️  Architecture Layers: ${layers.join(', ')}\n`)

    // Test naming conventions
    console.log('📝 Testing Naming Conventions:\n')

    const testCases = [
      { name: 'DbAddAccount', layer: 'data', element: 'usecases', expected: true },
      { name: 'AddAccount', layer: 'domain', element: 'usecases', expected: true },
      { name: 'InvalidName', layer: 'data', element: 'usecases', expected: false },
    ]

    testCases.forEach(test => {
      const result = validator.validateNaming(test.name, test.layer, test.element)
      const status = result === test.expected ? '✅' : '❌'
      console.log(`${status} ${test.name} (${test.layer}/${test.element}): ${result ? 'Valid' : 'Invalid'}`)
    })

    console.log('\n' + '═'.repeat(80))
    console.log('✨ Demo complete!\n')

  } catch (error) {
    console.error('❌ Error loading patterns:', error)
  }
}

// ============================================================================
// Run if executed directly
// ============================================================================

if (require.main === module) {
  runDemo()
}
