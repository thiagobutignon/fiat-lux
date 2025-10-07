/**
 * Pattern Loader Tests
 */

import { PatternLoader } from '../domain/use-cases/pattern-loader'
import { describe, it, expect, beforeEach } from '../../shared/utils/test-runner'

const sampleYAML = `
version: "1.0"
architecture: "Clean Architecture"
grammar_type: "Context-Free Grammar"

patterns:
  - id: usecase
    name: "Use Case"
    layer: domain
    linguistic_role: "verb phrase"
    description: "Business logic operations"

  - id: entity
    name: "Entity"
    layer: domain
    linguistic_role: "noun"
    description: "Core business objects"

  - id: repository
    name: "Repository"
    layer: data
    linguistic_role: "noun"
    description: "Data access interface"

naming_conventions:
  domain:
    usecases:
      pattern: "^[A-Z][a-z]+[A-Z][a-z]+UseCase$"
      example: "AddAccountUseCase, GetUserUseCase"
    entities:
      pattern: "^[A-Z][a-z]+$"
      example: "User, Account"

  data:
    usecases:
      pattern: "^Db([A-Z][a-z]+)+$"
      example: "DbAddAccount, DbGetUser"

  infrastructure:
    adapters:
      pattern: "^([A-Z][a-z]+)+Adapter$"
      example: "AxiosAdapter, BcryptAdapter"

  presentation:
    controllers:
      pattern: "^([A-Z][a-z]+)+Controller$"
      example: "SignUpController, LoginController"

  validation:
    validators:
      pattern: "^([A-Z][a-z]+)+Validation$"
      example: "EmailValidation, PasswordValidation"

  main:
    factories:
      pattern: "^make([A-Z][a-z]+)+$"
      example: "makeSignUpController, makeLoginController"

dependency_rules:
  allowed:
    - from: presentation
      to: [domain, validation]
    - from: domain
      to: []

  forbidden:
    - from: domain
      to: [data, infrastructure, presentation]
      reason: "Domain layer must not depend on outer layers"
      severity: "error"
      type: "architectural_violation"
`

describe('PatternLoader - Basic Operations', () => {
  let loader: PatternLoader

  beforeEach(() => {
    loader = new PatternLoader(sampleYAML)
  })

  it('should parse YAML and extract version', () => {
    const config = loader.getConfig()
    expect.toEqual(config.version, '1.0')
  })

  it('should extract architecture name', () => {
    const config = loader.getConfig()
    expect.toEqual(config.architecture, 'Clean Architecture')
  })

  it('should extract grammar type', () => {
    const config = loader.getConfig()
    expect.toEqual(config.grammarType, 'Context-Free Grammar')
  })

  it('should extract all patterns', () => {
    const patterns = loader.getPatterns()
    expect.toEqual(patterns.length, 3)
  })

  it('should get pattern by ID', () => {
    const pattern = loader.getPatternById('usecase')
    expect.toEqual(pattern?.name, 'Use Case')
    expect.toEqual(pattern?.layer, 'domain')
  })

  it('should return undefined for non-existent pattern ID', () => {
    const pattern = loader.getPatternById('nonexistent')
    expect.toEqual(pattern, undefined)
  })
})

describe('PatternLoader - Layer Operations', () => {
  let loader: PatternLoader

  beforeEach(() => {
    loader = new PatternLoader(sampleYAML)
  })

  it('should get patterns by layer', () => {
    const domainPatterns = loader.getPatternsByLayer('domain')
    expect.toEqual(domainPatterns.length, 2)
    expect.toEqual(domainPatterns[0].name, 'Use Case')
    expect.toEqual(domainPatterns[1].name, 'Entity')
  })

  it('should get all layers', () => {
    const layers = loader.getLayers()
    expect.toEqual(layers.length, 2)
    expect.toContain(layers, 'domain')
    expect.toContain(layers, 'data')
  })

  it('should return empty array for non-existent layer', () => {
    const patterns = loader.getPatternsByLayer('nonexistent')
    expect.toEqual(patterns.length, 0)
  })
})

describe('PatternLoader - Naming Conventions', () => {
  let loader: PatternLoader

  beforeEach(() => {
    loader = new PatternLoader(sampleYAML)
  })

  it('should get naming convention pattern', () => {
    const pattern = loader.getNamingConvention('domain', 'usecases')
    expect.toEqual(pattern, '^[A-Z][a-z]+[A-Z][a-z]+UseCase$')
  })

  it('should return undefined for non-existent convention', () => {
    const pattern = loader.getNamingConvention('nonexistent', 'nonexistent')
    expect.toEqual(pattern, undefined)
  })

  it('should validate correct naming', () => {
    const result = loader.validateNaming('AddAccountUseCase', 'domain', 'usecases')
    expect.toEqual(result.valid, true)
    expect.toEqual(result.message, 'Naming convention satisfied')
  })

  it('should detect invalid naming', () => {
    const result = loader.validateNaming('invalidName', 'domain', 'usecases')
    expect.toEqual(result.valid, false)
    expect.toBeTruthy((result.message || '').includes('does not match pattern'))
  })

  it('should validate infrastructure adapter naming', () => {
    const result = loader.validateNaming('AxiosAdapter', 'infrastructure', 'adapters')
    expect.toEqual(result.valid, true)
  })

  it('should detect invalid adapter naming', () => {
    const result = loader.validateNaming('axios', 'infrastructure', 'adapters')
    expect.toEqual(result.valid, false)
  })

  it('should validate presentation controller naming', () => {
    const result = loader.validateNaming('SignUpController', 'presentation', 'controllers')
    expect.toEqual(result.valid, true)
  })

  it('should validate data layer naming', () => {
    const result = loader.validateNaming('DbAddAccount', 'data', 'usecases')
    expect.toEqual(result.valid, true)
  })

  it('should validate validation layer naming', () => {
    const result = loader.validateNaming('EmailValidation', 'validation', 'validators')
    expect.toEqual(result.valid, true)
  })

  it('should validate main factory naming', () => {
    const result = loader.validateNaming('makeSignUpController', 'main', 'factories')
    expect.toEqual(result.valid, true)
  })

  it('should allow validation when no convention defined', () => {
    const result = loader.validateNaming('AnyName', 'nonexistent', 'nonexistent')
    expect.toEqual(result.valid, true)
    expect.toEqual(result.message, 'No naming convention defined')
  })
})

describe('PatternLoader - Dependency Validation', () => {
  let loader: PatternLoader

  beforeEach(() => {
    loader = new PatternLoader(sampleYAML)
  })

  it('should allow valid dependencies', () => {
    const result = loader.validateDependency('presentation', 'domain')
    expect.toEqual(result.valid, true)
  })

  it('should detect forbidden dependencies', () => {
    const result = loader.validateDependency('domain', 'data')
    expect.toEqual(result.valid, false)
    expect.toBeTruthy((result.message || '').includes('Forbidden dependency'))
  })

  it('should allow when no rules defined', () => {
    const loaderWithoutRules = new PatternLoader('version: "1.0"')
    const result = loaderWithoutRules.validateDependency('any', 'any')
    expect.toEqual(result.valid, true)
  })
})

describe('PatternLoader - Examples', () => {
  let loader: PatternLoader

  beforeEach(() => {
    loader = new PatternLoader(sampleYAML)
  })

  it('should get examples for convention', () => {
    const examples = loader.getExamples('domain', 'usecases')
    expect.toEqual(examples.length, 2)
    expect.toContain(examples, 'AddAccountUseCase')
    expect.toContain(examples, 'GetUserUseCase')
  })

  it('should return empty array for non-existent convention', () => {
    const examples = loader.getExamples('nonexistent', 'nonexistent')
    expect.toEqual(examples.length, 0)
  })

  it('should get examples for entities', () => {
    const examples = loader.getExamples('domain', 'entities')
    expect.toEqual(examples.length, 2)
    expect.toContain(examples, 'User')
    expect.toContain(examples, 'Account')
  })
})

describe('PatternLoader - Summary', () => {
  let loader: PatternLoader

  beforeEach(() => {
    loader = new PatternLoader(sampleYAML)
  })

  it('should generate summary statistics', () => {
    const summary = loader.getSummary()
    expect.toEqual(summary.version, '1.0')
    expect.toEqual(summary.architecture, 'Clean Architecture')
    expect.toEqual(summary.grammarType, 'Context-Free Grammar')
    expect.toEqual(summary.totalPatterns, 3)
    expect.toEqual(summary.hasNamingConventions, true)
    expect.toEqual(summary.hasDependencyRules, true)
  })

  it('should include all layers in summary', () => {
    const summary = loader.getSummary()
    expect.toEqual(summary.layers.length, 2)
    expect.toContain(summary.layers, 'domain')
    expect.toContain(summary.layers, 'data')
  })
})

describe('PatternLoader - Edge Cases', () => {
  it('should handle empty YAML', () => {
    const loader = new PatternLoader('')
    const patterns = loader.getPatterns()
    expect.toEqual(patterns.length, 0)
  })

  it('should handle YAML with only version', () => {
    const loader = new PatternLoader('version: "2.0"')
    const config = loader.getConfig()
    expect.toEqual(config.version, '2.0')
    expect.toEqual(config.patterns.length, 0)
  })

  it('should handle missing naming conventions', () => {
    const minimalYAML = `
version: "1.0"
patterns:
  - id: test
    name: "Test"
    layer: test
    linguistic_role: "test"
    description: "test"
`
    const loader = new PatternLoader(minimalYAML)
    const conventions = loader.getNamingConventions()
    expect.toEqual(Object.keys(conventions || {}).length, 0)
  })

  it('should handle missing dependency rules', () => {
    const minimalYAML = `
version: "1.0"
patterns:
  - id: test
    name: "Test"
    layer: test
    linguistic_role: "test"
    description: "test"
`
    const loader = new PatternLoader(minimalYAML)
    const rules = loader.getDependencyRules()
    expect.toEqual(rules?.allowed?.length || 0, 0)
    expect.toEqual(rules?.forbidden?.length || 0, 0)
  })
})
