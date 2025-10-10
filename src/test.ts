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
import './grammar-lang/database/__tests__/sqlo.test'
import './grammar-lang/database/__tests__/rbac.test'
import './grammar-lang/database/__tests__/consolidation-optimizer.test'
import './grammar-lang/database/__tests__/sqlo-constitutional.test'
import './grammar-lang/database/__tests__/embedding-semantic.test'
import './grammar-lang/glass/__tests__/sqlo-integration.test'

// E2E Tests (requires ANTHROPIC_API_KEY)
import '../tests/e2e-llm-integration.test'

// Run tests
runTests().catch(error => {
  console.error('Test runner failed:', error)
  process.exit(1)
})
