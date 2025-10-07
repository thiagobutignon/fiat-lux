#!/usr/bin/env ts-node
/**
 * Grammar Pattern Validator
 * Validates the grammar-patterns.yml file using PatternLoader
 */

import { readFileSync } from 'fs'
import { join } from 'path'

// Import from chomsky
import { PatternLoader } from '/Users/thiagobutignon/dev/chomsky/src/pattern-loader/domain/use-cases/pattern-loader'

async function main() {
  console.log('🔍 Grammar Pattern Validator\n')
  console.log('=' .repeat(80))

  // Load YAML file
  const yamlPath = join(__dirname, 'grammar-patterns.yml')
  console.log(`📄 Loading: ${yamlPath}\n`)

  const yamlContent = readFileSync(yamlPath, 'utf-8')

  // Create pattern loader
  const loader = new PatternLoader(yamlContent)

  // Get summary
  const summary = loader.getSummary()
  console.log('📊 Summary:')
  console.log(`   Version: ${summary.version}`)
  console.log(`   Architecture: ${summary.architecture}`)
  console.log(`   Grammar Type: ${summary.grammarType}`)
  console.log(`   Total Patterns: ${summary.totalPatterns}`)
  console.log(`   Layers: ${summary.layers.join(', ')}`)
  console.log(`   Has Naming Conventions: ${summary.hasNamingConventions}`)
  console.log(`   Has Dependency Rules: ${summary.hasDependencyRules}`)
  console.log()

  // Get all patterns
  const patterns = loader.getPatterns()
  console.log('=' .repeat(80))
  console.log('✅ PATTERNS LOADED SUCCESSFULLY\n')

  for (const pattern of patterns) {
    console.log(`📦 ${pattern.id}: ${pattern.name}`)
    console.log(`   Layer: ${pattern.layer}`)
    console.log(`   Linguistic Role: ${pattern.linguisticRole}`)
    console.log(`   Description: ${pattern.description}`)
    console.log()
  }

  // Validate naming conventions
  console.log('=' .repeat(80))
  console.log('🏷️  NAMING CONVENTIONS\n')

  const namingConventions = loader.getNamingConventions()
  if (namingConventions) {
    for (const [layer, conventions] of Object.entries(namingConventions)) {
      console.log(`Layer: ${layer}`)
      for (const [element, convention] of Object.entries(conventions)) {
        console.log(`  ${element}: ${convention.pattern}`)
        console.log(`    Example: ${convention.example}`)
      }
      console.log()
    }
  }

  // Test naming validation
  console.log('=' .repeat(80))
  console.log('🧪 TESTING NAMING VALIDATION\n')

  const testCases = [
    { layer: 'data', element: 'usecases', value: 'DbAddAccount', expected: true },
    { layer: 'data', element: 'usecases', value: 'AddAccount', expected: false },
    { layer: 'domain', element: 'usecases', value: 'AddAccount', expected: true },
    { layer: 'presentation', element: 'controllers', value: 'SignUpController', expected: true },
    { layer: 'main', element: 'factories', value: 'makeDbAddAccount', expected: true },
  ]

  for (const test of testCases) {
    const result = loader.validateNaming(test.value, test.layer, test.element)
    const status = result.valid === test.expected ? '✅' : '❌'
    console.log(`${status} ${test.layer}/${test.element}: "${test.value}"`)
    console.log(`   Expected: ${test.expected}, Got: ${result.valid}`)
    if (!result.valid) {
      console.log(`   Message: ${result.message}`)
    }
    console.log()
  }

  // Test dependency validation
  console.log('=' .repeat(80))
  console.log('🔗 TESTING DEPENDENCY VALIDATION\n')

  const dependencyTests = [
    { from: 'domain', to: 'data', expected: false },
    { from: 'data', to: 'domain', expected: true },
    { from: 'presentation', to: 'infrastructure', expected: false },
    { from: 'infrastructure', to: 'data', expected: true },
  ]

  for (const test of dependencyTests) {
    const result = loader.validateDependency(test.from, test.to)
    const status = result.valid === test.expected ? '✅' : '❌'
    console.log(`${status} ${test.from} → ${test.to}`)
    console.log(`   Expected: ${test.expected}, Got: ${result.valid}`)
    if (!result.valid) {
      console.log(`   Message: ${result.message}`)
    }
    console.log()
  }

  console.log('=' .repeat(80))
  console.log('✨ Validation Complete!')
}

main().catch(console.error)
