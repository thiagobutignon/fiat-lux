/**
 * Pattern Loader Use Case
 *
 * Loads and parses YAML pattern configurations with full support for
 * naming conventions, dependency rules, and pattern validation
 */

import {
  YAMLPatternConfig,
  Pattern,
  NamingConventions,
  LayerNamingConventions,
  DependencyRules,
  NamingValidationResult,
  DependencyValidationResult
} from '../entities/types'

export class PatternLoader {
  private config: YAMLPatternConfig

  constructor(yamlContent: string) {
    this.config = this.parseYAML(yamlContent)
  }

  /**
   * Parse YAML content (simplified implementation)
   * In production, use 'js-yaml' library for robust parsing
   */
  private parseYAML(yamlContent: string): YAMLPatternConfig {
    const config: YAMLPatternConfig = {
      version: '1.0',
      architecture: 'Clean Architecture',
      grammarType: 'Context-Free Grammar',
      patterns: []
    }

    // Extract version
    const versionMatch = yamlContent.match(/version:\s*"([^"]+)"/)
    if (versionMatch) config.version = versionMatch[1]

    // Extract architecture
    const archMatch = yamlContent.match(/architecture:\s*"([^"]+)"/)
    if (archMatch) config.architecture = archMatch[1]

    // Extract grammar type
    const grammarMatch = yamlContent.match(/grammar_type:\s*"([^"]+)"/)
    if (grammarMatch) config.grammarType = grammarMatch[1]

    // Extract patterns
    config.patterns = this.extractPatterns(yamlContent)

    // Extract naming conventions
    config.namingConventions = this.extractNamingConventions(yamlContent)

    // Extract dependency rules
    config.dependencyRules = this.extractDependencyRules(yamlContent)

    return config
  }

  /**
   * Extract patterns from YAML
   */
  private extractPatterns(yamlContent: string): Pattern[] {
    const patterns: Pattern[] = []
    const patternRegex = /- id:\s*([^\n]+)\s+name:\s*"([^"]+)"\s+layer:\s*([^\n]+)\s+linguistic_role:\s*"([^"]+)"\s+description:\s*"([^"]+)"/g

    let match
    while ((match = patternRegex.exec(yamlContent)) !== null) {
      patterns.push({
        id: match[1].trim(),
        name: match[2],
        layer: match[3].trim(),
        linguisticRole: match[4],
        description: match[5]
      })
    }

    return patterns
  }

  /**
   * Extract naming conventions from YAML
   */
  private extractNamingConventions(yamlContent: string): NamingConventions {
    const conventions: NamingConventions = {}

    // Helper to extract all elements for a layer
    const extractLayerConventions = (layer: string): LayerNamingConventions => {
      const layerConventions: LayerNamingConventions = {}

      // Find the layer section
      const layerRegex = new RegExp(`${layer}:([\\s\\S]*?)(?=\\n  \\w+:|$)`, 'i')
      const layerMatch = yamlContent.match(layerRegex)

      if (layerMatch) {
        const layerContent = layerMatch[1]

        // Extract all element patterns within this layer
        const elementRegex = /(\w+):\s+pattern:\s*"([^"]+)"\s+example:\s*"([^"]+)"/g
        let match

        while ((match = elementRegex.exec(layerContent)) !== null) {
          layerConventions[match[1]] = {
            pattern: match[2],
            example: match[3]
          }
        }
      }

      return layerConventions
    }

    // Extract conventions for all layers
    const layers = ['domain', 'data', 'infrastructure', 'presentation', 'validation', 'main']

    for (const layer of layers) {
      const layerConventions = extractLayerConventions(layer)
      if (Object.keys(layerConventions).length > 0) {
        conventions[layer] = layerConventions
      }
    }

    return conventions
  }

  /**
   * Extract dependency rules from YAML
   */
  private extractDependencyRules(yamlContent: string): DependencyRules {
    const rules: DependencyRules = {
      allowed: [],
      forbidden: []
    }

    // Extract forbidden rules
    const forbiddenRegex = /- from:\s+(\w+)\s+to:\s+\[([^\]]+)\]\s+reason:\s*"([^"]+)"\s+severity:\s*"([^"]+)"(?:\s+type:\s*"([^"]+)")?/g
    let match

    while ((match = forbiddenRegex.exec(yamlContent)) !== null) {
      rules.forbidden!.push({
        from: match[1],
        to: match[2].split(',').map(s => s.trim()),
        reason: match[3],
        severity: match[4],
        type: match[5]
      })
    }

    // Extract allowed rules
    const allowedRegex = /allowed:\s*\n((?:\s+- from:[\s\S]*?to:\s+\[[^\]]+\]\s*\n)+)/
    const allowedMatch = yamlContent.match(allowedRegex)

    if (allowedMatch) {
      const allowedContent = allowedMatch[1]
      const allowedItemRegex = /- from:\s+(\w+)\s+to:\s+\[([^\]]+)\]/g
      let allowedItemMatch

      while ((allowedItemMatch = allowedItemRegex.exec(allowedContent)) !== null) {
        rules.allowed!.push({
          from: allowedItemMatch[1],
          to: allowedItemMatch[2].split(',').map(s => s.trim())
        })
      }
    }

    return rules
  }

  /**
   * Get all patterns
   */
  getPatterns(): Pattern[] {
    return this.config.patterns
  }

  /**
   * Get pattern by ID
   */
  getPatternById(id: string): Pattern | undefined {
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
    return this.config.namingConventions?.[layer]?.[element]?.pattern
  }

  /**
   * Validate naming against conventions
   */
  validateNaming(value: string, layer: string, element: string): NamingValidationResult {
    const pattern = this.getNamingConvention(layer, element)

    if (!pattern) {
      return {
        valid: true,
        value,
        layer,
        element,
        message: 'No naming convention defined'
      }
    }

    const regex = new RegExp(pattern)
    const valid = regex.test(value)

    return {
      valid,
      value,
      layer,
      element,
      pattern,
      message: valid
        ? 'Naming convention satisfied'
        : `Value "${value}" does not match pattern: ${pattern}`
    }
  }

  /**
   * Validate dependency
   */
  validateDependency(from: string, to: string): DependencyValidationResult {
    const rules = this.config.dependencyRules

    if (!rules || !rules.forbidden) {
      return {
        valid: true,
        from,
        to,
        message: 'No dependency rules defined'
      }
    }

    // Check forbidden rules
    for (const rule of rules.forbidden) {
      if (rule.from === from && rule.to.includes(to)) {
        return {
          valid: false,
          from,
          to,
          rule,
          message: `Forbidden dependency: ${from} → ${to}. ${rule.reason}`
        }
      }
    }

    return {
      valid: true,
      from,
      to,
      message: 'Dependency allowed'
    }
  }

  /**
   * Get examples for a naming convention
   */
  getExamples(layer: string, element: string): string[] {
    const convention = this.config.namingConventions?.[layer]?.[element]
    if (!convention || !convention.example) {
      return []
    }

    return convention.example.split(',').map(s => s.trim())
  }

  /**
   * Get configuration
   */
  getConfig(): YAMLPatternConfig {
    return this.config
  }

  /**
   * Get naming conventions
   */
  getNamingConventions(): NamingConventions | undefined {
    return this.config.namingConventions
  }

  /**
   * Get dependency rules
   */
  getDependencyRules(): DependencyRules | undefined {
    return this.config.dependencyRules
  }

  /**
   * Get summary statistics
   */
  getSummary() {
    return {
      version: this.config.version,
      architecture: this.config.architecture,
      grammarType: this.config.grammarType,
      totalPatterns: this.config.patterns.length,
      layers: this.getLayers(),
      hasNamingConventions: !!this.config.namingConventions,
      hasDependencyRules: !!this.config.dependencyRules
    }
  }
}
