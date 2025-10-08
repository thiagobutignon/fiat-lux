/**
 * Lightweight Test Runner
 *
 * Fast, simple test framework for unit testing without external dependencies.
 * Designed for speed - runs tests synchronously with minimal overhead.
 */

import { AssertionError } from 'assert'

// ============================================================================
// Types
// ============================================================================

export type TestFunction = () => void | Promise<void>

export interface TestCase {
  name: string
  fn: TestFunction
  skip?: boolean
  only?: boolean
}

export interface TestSuite {
  name: string
  tests: TestCase[]
  beforeEach?: () => void | Promise<void>
  afterEach?: () => void | Promise<void>
  beforeAll?: () => void | Promise<void>
  afterAll?: () => void | Promise<void>
}

export interface TestResult {
  suite: string
  test: string
  passed: boolean
  error?: Error
  duration: number
}

export interface TestSummary {
  total: number
  passed: number
  failed: number
  skipped: number
  duration: number
  results: TestResult[]
}

// ============================================================================
// Global State
// ============================================================================

let currentSuite: TestSuite | null = null
const suites: TestSuite[] = []

// ============================================================================
// Test Definition API
// ============================================================================

/**
 * Define a test suite
 */
export function describe(name: string, fn: () => void): void {
  const suite: TestSuite = {
    name,
    tests: []
  }

  currentSuite = suite
  fn()
  currentSuite = null

  suites.push(suite)
}

/**
 * Define a test case
 */
export function it(name: string, fn: TestFunction): void {
  if (!currentSuite) {
    throw new Error('it() must be called inside describe()')
  }

  currentSuite.tests.push({ name, fn })
}

/**
 * Define a test case (alias for it)
 */
export const test = it

/**
 * Skip a test
 */
it.skip = (name: string, fn: TestFunction): void => {
  if (!currentSuite) {
    throw new Error('it.skip() must be called inside describe()')
  }

  currentSuite.tests.push({ name, fn, skip: true })
}

/**
 * Run only this test
 */
it.only = (name: string, fn: TestFunction): void => {
  if (!currentSuite) {
    throw new Error('it.only() must be called inside describe()')
  }

  currentSuite.tests.push({ name, fn, only: true })
}

/**
 * Setup function to run before each test
 */
export function beforeEach(fn: () => void | Promise<void>): void {
  if (!currentSuite) {
    throw new Error('beforeEach() must be called inside describe()')
  }

  currentSuite.beforeEach = fn
}

/**
 * Teardown function to run after each test
 */
export function afterEach(fn: () => void | Promise<void>): void {
  if (!currentSuite) {
    throw new Error('afterEach() must be called inside describe()')
  }

  currentSuite.afterEach = fn
}

/**
 * Setup function to run before all tests in suite
 */
export function beforeAll(fn: () => void | Promise<void>): void {
  if (!currentSuite) {
    throw new Error('beforeAll() must be called inside describe()')
  }

  currentSuite.beforeAll = fn
}

/**
 * Teardown function to run after all tests in suite
 */
export function afterAll(fn: () => void | Promise<void>): void {
  if (!currentSuite) {
    throw new Error('afterAll() must be called inside describe()')
  }

  currentSuite.afterAll = fn
}

// ============================================================================
// Assertion Helpers
// ============================================================================

/**
 * Simple assertion utilities
 */
export const expect = {
  /**
   * Assert that a value is truthy
   */
  toBeTruthy(actual: any): void {
    if (!actual) {
      throw new AssertionError({
        message: `Expected value to be truthy, but got: ${actual}`,
        actual,
        expected: true
      })
    }
  },

  /**
   * Assert that a value is falsy
   */
  toBeFalsy(actual: any): void {
    if (actual) {
      throw new AssertionError({
        message: `Expected value to be falsy, but got: ${actual}`,
        actual,
        expected: false
      })
    }
  },

  /**
   * Assert equality
   */
  toEqual(actual: any, expected: any): void {
    if (actual !== expected) {
      throw new AssertionError({
        message: `Expected ${actual} to equal ${expected}`,
        actual,
        expected
      })
    }
  },

  /**
   * Assert deep equality for objects
   */
  toDeepEqual(actual: any, expected: any): void {
    const actualStr = JSON.stringify(actual)
    const expectedStr = JSON.stringify(expected)

    if (actualStr !== expectedStr) {
      throw new AssertionError({
        message: `Expected objects to be deeply equal`,
        actual: actualStr,
        expected: expectedStr
      })
    }
  },

  /**
   * Assert that value is greater than
   */
  toBeGreaterThan(actual: number, expected: number): void {
    if (actual <= expected) {
      throw new AssertionError({
        message: `Expected ${actual} to be greater than ${expected}`,
        actual,
        expected
      })
    }
  },

  /**
   * Assert that value is less than
   */
  toBeLessThan(actual: number, expected: number): void {
    if (actual >= expected) {
      throw new AssertionError({
        message: `Expected ${actual} to be less than ${expected}`,
        actual,
        expected
      })
    }
  },

  /**
   * Assert that array contains value
   */
  toContain(actual: any[], expected: any): void {
    if (!actual.includes(expected)) {
      throw new AssertionError({
        message: `Expected array to contain ${expected}`,
        actual,
        expected
      })
    }
  },

  /**
   * Assert that function throws
   */
  toThrow(fn: () => any, errorMessage?: string): void {
    try {
      fn()
      throw new AssertionError({
        message: 'Expected function to throw, but it did not'
      })
    } catch (error) {
      if (error instanceof AssertionError && !errorMessage) {
        throw error
      }
      if (errorMessage && !String(error).includes(errorMessage)) {
        throw new AssertionError({
          message: `Expected error message to include "${errorMessage}", but got: ${error}`
        })
      }
    }
  },

  /**
   * Assert that value is undefined
   */
  toBeUndefined(actual: any): void {
    if (actual !== undefined) {
      throw new AssertionError({
        message: `Expected value to be undefined, but got: ${actual}`,
        actual,
        expected: undefined
      })
    }
  },

  /**
   * Assert that value is defined
   */
  toBeDefined(actual: any): void {
    if (actual === undefined) {
      throw new AssertionError({
        message: 'Expected value to be defined, but got undefined',
        actual: undefined,
        expected: 'defined value'
      })
    }
  },

  /**
   * Assert that numeric values are close (within tolerance)
   */
  toBeCloseTo(actual: number, expected: number, precision: number = 2): void {
    const tolerance = Math.pow(10, -precision) / 2
    const diff = Math.abs(actual - expected)

    if (diff > tolerance) {
      throw new AssertionError({
        message: `Expected ${actual} to be close to ${expected} (precision: ${precision}, tolerance: ${tolerance})`,
        actual,
        expected
      })
    }
  }
}

// ============================================================================
// Test Runner
// ============================================================================

/**
 * Run a single test case
 */
async function runTest(suite: TestSuite, testCase: TestCase): Promise<TestResult> {
  const startTime = performance.now()

  try {
    // Run beforeEach
    if (suite.beforeEach) {
      await suite.beforeEach()
    }

    // Run test
    await testCase.fn()

    // Run afterEach
    if (suite.afterEach) {
      await suite.afterEach()
    }

    const duration = performance.now() - startTime

    return {
      suite: suite.name,
      test: testCase.name,
      passed: true,
      duration
    }
  } catch (error) {
    const duration = performance.now() - startTime

    return {
      suite: suite.name,
      test: testCase.name,
      passed: false,
      error: error instanceof Error ? error : new Error(String(error)),
      duration
    }
  }
}

/**
 * Run all tests
 */
export async function runTests(): Promise<TestSummary> {
  const results: TestResult[] = []
  const startTime = performance.now()

  console.log('\n🧪 Running Tests...\n')

  for (const suite of suites) {
    console.log(`📦 ${suite.name}`)

    // Check if any test has .only
    const hasOnly = suite.tests.some(t => t.only)
    const testsToRun = hasOnly ? suite.tests.filter(t => t.only) : suite.tests.filter(t => !t.skip)

    // Run beforeAll
    if (suite.beforeAll) {
      await suite.beforeAll()
    }

    // Run tests
    for (const testCase of testsToRun) {
      const result = await runTest(suite, testCase)
      results.push(result)

      if (result.passed) {
        console.log(`  ✅ ${testCase.name} (${result.duration.toFixed(2)}ms)`)
      } else {
        console.log(`  ❌ ${testCase.name} (${result.duration.toFixed(2)}ms)`)
        if (result.error) {
          console.log(`     ${result.error.message}`)
        }
      }
    }

    // Count skipped tests
    const skippedTests = suite.tests.filter(t => t.skip)
    if (skippedTests.length > 0) {
      skippedTests.forEach(t => {
        console.log(`  ⏭️  ${t.name} (skipped)`)
      })
    }

    // Run afterAll
    if (suite.afterAll) {
      await suite.afterAll()
    }

    console.log('')
  }

  const duration = performance.now() - startTime
  const passed = results.filter(r => r.passed).length
  const failed = results.filter(r => !r.passed).length
  const skipped = suites.reduce((acc, suite) => acc + suite.tests.filter(t => t.skip).length, 0)

  const summary: TestSummary = {
    total: results.length,
    passed,
    failed,
    skipped,
    duration,
    results
  }

  // Print summary
  console.log('═'.repeat(80))
  console.log('Test Summary')
  console.log('═'.repeat(80))
  console.log(`Total:   ${summary.total}`)
  console.log(`✅ Passed: ${summary.passed}`)
  console.log(`❌ Failed: ${summary.failed}`)
  console.log(`⏭️  Skipped: ${summary.skipped}`)
  console.log(`⏱️  Duration: ${summary.duration.toFixed(2)}ms`)
  console.log('═'.repeat(80))

  if (failed > 0) {
    console.log('\n❌ Some tests failed!\n')
    process.exit(1)
  } else {
    console.log('\n✅ All tests passed!\n')
  }

  return summary
}

/**
 * Clear all registered suites (useful for testing the test runner itself)
 */
export function clearSuites(): void {
  suites.length = 0
}
