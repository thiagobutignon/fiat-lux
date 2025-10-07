/**
 * Pattern Loader Use Case
 *
 * Loads and parses YAML pattern configurations
 */

import { YAMLPatternConfig, Pattern } from '../entities/types'

export class PatternLoader {
  private config: YAMLPatternConfig

  constructor(yamlContent: string) {
    this.config = this.parseYAML(yamlContent)
  }

  /**
   * Parse YAML content (simplified implementation)
   */
  private parseYAML(yamlContent: string): YAMLPatternConfig {
    // Simplified YAML parsing - in production use 'js-yaml' library
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
   * Get configuration
   */
  getConfig(): YAMLPatternConfig {
    return this.config
  }
}
