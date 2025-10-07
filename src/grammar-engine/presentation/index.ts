/**
 * Grammar Engine Public API
 *
 * Main entry point for the grammar engine feature
 */

// Types and Entities
export * from '../domain/entities/types'
export * from '../domain/entities/predefined-grammars'

// Use Cases
export { GrammarEngine } from '../domain/use-cases/grammar-engine'

// Factories
export { makeGrammarEngine } from './factories/grammar-engine-factory'

// Utilities
export { formatResult } from './utils/format-result'
