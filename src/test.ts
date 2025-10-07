/**
 * Test Runner Entry Point
 *
 * Runs all unit tests for the project
 */

import { runTests } from './shared/utils/test-runner'

// Import all test files
import './similarity-algorithms/__tests__/levenshtein.test'
import './similarity-algorithms/__tests__/jaro-winkler.test'
import './similarity-algorithms/__tests__/hybrid.test'
import './grammar-engine/__tests__/grammar-engine.test'
import './pattern-loader/__tests__/pattern-loader.test'

// Run tests
runTests().catch(error => {
  console.error('Test runner failed:', error)
  process.exit(1)
})
