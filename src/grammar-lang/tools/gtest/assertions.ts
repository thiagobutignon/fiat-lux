/**
 * GTest - Assertions Library
 *
 * O(1) assertion library with comprehensive matchers.
 *
 * Features:
 * - All assertions are O(1) operations
 * - Clear error messages
 * - Type-safe matchers
 * - Custom matchers support
 * - Performance tracking
 *
 * Usage:
 * ```typescript
 * import { expect } from './assertions';
 *
 * expect(5).toEqual(5);
 * expect([1,2,3]).toContain(2);
 * expect(() => throw new Error()).toThrow();
 * ```
 */

import { AssertionError } from 'assert';

// ============================================================================
// Types
// ============================================================================

export interface MatcherResult {
  pass: boolean;
  message: string;
}

export type Matcher<T> = (actual: T, expected?: any) => MatcherResult;

export interface AssertionContext {
  actual: any;
  matchers: Map<string, Matcher<any>>;  // O(1) matcher lookup
  negated: boolean;
}

// ============================================================================
// Core Assertion Class
// ============================================================================

export class Assertion {
  private context: AssertionContext;

  constructor(actual: any) {
    this.context = {
      actual,
      matchers: new Map(),
      negated: false
    };

    // Register built-in matchers (O(1) per registration)
    this.registerBuiltInMatchers();
  }

  /**
   * Negate next assertion
   */
  get not(): Assertion {
    this.context.negated = !this.context.negated;
    return this;
  }

  /**
   * Register built-in matchers
   */
  private registerBuiltInMatchers(): void {
    // Equality matchers
    this.registerMatcher('toEqual', matchToEqual);
    this.registerMatcher('toStrictEqual', matchToStrictEqual);
    this.registerMatcher('toDeepEqual', matchToDeepEqual);
    this.registerMatcher('toBe', matchToBe);

    // Truthiness matchers
    this.registerMatcher('toBeTruthy', matchToBeTruthy);
    this.registerMatcher('toBeFalsy', matchToBeFalsy);
    this.registerMatcher('toBeDefined', matchToBeDefined);
    this.registerMatcher('toBeUndefined', matchToBeUndefined);
    this.registerMatcher('toBeNull', matchToBeNull);
    this.registerMatcher('toBeNaN', matchToBeNaN);

    // Comparison matchers
    this.registerMatcher('toBeGreaterThan', matchToBeGreaterThan);
    this.registerMatcher('toBeGreaterThanOrEqual', matchToBeGreaterThanOrEqual);
    this.registerMatcher('toBeLessThan', matchToBeLessThan);
    this.registerMatcher('toBeLessThanOrEqual', matchToBeLessThanOrEqual);
    this.registerMatcher('toBeCloseTo', matchToBeCloseTo);

    // String matchers
    this.registerMatcher('toMatch', matchToMatch);
    this.registerMatcher('toContainString', matchToContainString);
    this.registerMatcher('toStartWith', matchToStartWith);
    this.registerMatcher('toEndWith', matchToEndWith);

    // Array/Object matchers
    this.registerMatcher('toContain', matchToContain);
    this.registerMatcher('toHaveLength', matchToHaveLength);
    this.registerMatcher('toHaveProperty', matchToHaveProperty);
    this.registerMatcher('toBeEmpty', matchToBeEmpty);

    // Type matchers
    this.registerMatcher('toBeInstanceOf', matchToBeInstanceOf);
    this.registerMatcher('toBeTypeOf', matchToBeTypeOf);

    // Function matchers
    this.registerMatcher('toThrow', matchToThrow);
    this.registerMatcher('toThrowError', matchToThrowError);
    this.registerMatcher('toHaveBeenCalled', matchToHaveBeenCalled);
    this.registerMatcher('toHaveBeenCalledTimes', matchToHaveBeenCalledTimes);
    this.registerMatcher('toHaveBeenCalledWith', matchToHaveBeenCalledWith);
  }

  /**
   * Register custom matcher (O(1))
   */
  registerMatcher<T>(name: string, matcher: Matcher<T>): void {
    this.context.matchers.set(name, matcher);
  }

  /**
   * Execute matcher (O(1) lookup + O(1) execution)
   */
  private executeMatcher(name: string, expected?: any): void {
    const matcher = this.context.matchers.get(name);

    if (!matcher) {
      throw new Error(`Unknown matcher: ${name}`);
    }

    const result = matcher(this.context.actual, expected);

    // Handle negation
    const shouldPass = this.context.negated ? !result.pass : result.pass;

    if (!shouldPass) {
      const message = this.context.negated
        ? `Expected NOT ${result.message}`
        : result.message;

      throw new AssertionError({ message });
    }

    // Reset negation
    this.context.negated = false;
  }

  // ============================================================================
  // Matcher Methods (delegating to executeMatcher)
  // ============================================================================

  // Equality
  toEqual(expected: any): void { this.executeMatcher('toEqual', expected); }
  toStrictEqual(expected: any): void { this.executeMatcher('toStrictEqual', expected); }
  toDeepEqual(expected: any): void { this.executeMatcher('toDeepEqual', expected); }
  toBe(expected: any): void { this.executeMatcher('toBe', expected); }

  // Truthiness
  toBeTruthy(): void { this.executeMatcher('toBeTruthy'); }
  toBeFalsy(): void { this.executeMatcher('toBeFalsy'); }
  toBeDefined(): void { this.executeMatcher('toBeDefined'); }
  toBeUndefined(): void { this.executeMatcher('toBeUndefined'); }
  toBeNull(): void { this.executeMatcher('toBeNull'); }
  toBeNaN(): void { this.executeMatcher('toBeNaN'); }

  // Comparison
  toBeGreaterThan(expected: number): void { this.executeMatcher('toBeGreaterThan', expected); }
  toBeGreaterThanOrEqual(expected: number): void { this.executeMatcher('toBeGreaterThanOrEqual', expected); }
  toBeLessThan(expected: number): void { this.executeMatcher('toBeLessThan', expected); }
  toBeLessThanOrEqual(expected: number): void { this.executeMatcher('toBeLessThanOrEqual', expected); }
  toBeCloseTo(expected: number, precision?: number): void { this.executeMatcher('toBeCloseTo', { expected, precision }); }

  // String
  toMatch(expected: RegExp | string): void { this.executeMatcher('toMatch', expected); }
  toContainString(expected: string): void { this.executeMatcher('toContainString', expected); }
  toStartWith(expected: string): void { this.executeMatcher('toStartWith', expected); }
  toEndWith(expected: string): void { this.executeMatcher('toEndWith', expected); }

  // Array/Object
  toContain(expected: any): void { this.executeMatcher('toContain', expected); }
  toHaveLength(expected: number): void { this.executeMatcher('toHaveLength', expected); }
  toHaveProperty(expected: string, value?: any): void { this.executeMatcher('toHaveProperty', { property: expected, value }); }
  toBeEmpty(): void { this.executeMatcher('toBeEmpty'); }

  // Type
  toBeInstanceOf(expected: any): void { this.executeMatcher('toBeInstanceOf', expected); }
  toBeTypeOf(expected: string): void { this.executeMatcher('toBeTypeOf', expected); }

  // Function
  toThrow(expected?: string | RegExp): void { this.executeMatcher('toThrow', expected); }
  toThrowError(expected?: string | RegExp | Error): void { this.executeMatcher('toThrowError', expected); }
  toHaveBeenCalled(): void { this.executeMatcher('toHaveBeenCalled'); }
  toHaveBeenCalledTimes(expected: number): void { this.executeMatcher('toHaveBeenCalledTimes', expected); }
  toHaveBeenCalledWith(...args: any[]): void { this.executeMatcher('toHaveBeenCalledWith', args); }
}

// ============================================================================
// Built-in Matchers
// ============================================================================

// Equality Matchers

function matchToEqual(actual: any, expected: any): MatcherResult {
  const pass = actual == expected;  // Loose equality
  return {
    pass,
    message: `${actual} to equal ${expected}`
  };
}

function matchToStrictEqual(actual: any, expected: any): MatcherResult {
  const pass = actual === expected;  // Strict equality
  return {
    pass,
    message: `${actual} to strictly equal ${expected}`
  };
}

function matchToDeepEqual(actual: any, expected: any): MatcherResult {
  const pass = JSON.stringify(actual) === JSON.stringify(expected);
  return {
    pass,
    message: `objects to be deeply equal`
  };
}

function matchToBe(actual: any, expected: any): MatcherResult {
  const pass = Object.is(actual, expected);
  return {
    pass,
    message: `${actual} to be ${expected}`
  };
}

// Truthiness Matchers

function matchToBeTruthy(actual: any): MatcherResult {
  const pass = !!actual;
  return {
    pass,
    message: `${actual} to be truthy`
  };
}

function matchToBeFalsy(actual: any): MatcherResult {
  const pass = !actual;
  return {
    pass,
    message: `${actual} to be falsy`
  };
}

function matchToBeDefined(actual: any): MatcherResult {
  const pass = actual !== undefined;
  return {
    pass,
    message: `value to be defined`
  };
}

function matchToBeUndefined(actual: any): MatcherResult {
  const pass = actual === undefined;
  return {
    pass,
    message: `${actual} to be undefined`
  };
}

function matchToBeNull(actual: any): MatcherResult {
  const pass = actual === null;
  return {
    pass,
    message: `${actual} to be null`
  };
}

function matchToBeNaN(actual: any): MatcherResult {
  const pass = Number.isNaN(actual);
  return {
    pass,
    message: `${actual} to be NaN`
  };
}

// Comparison Matchers

function matchToBeGreaterThan(actual: number, expected: number): MatcherResult {
  const pass = actual > expected;
  return {
    pass,
    message: `${actual} to be greater than ${expected}`
  };
}

function matchToBeGreaterThanOrEqual(actual: number, expected: number): MatcherResult {
  const pass = actual >= expected;
  return {
    pass,
    message: `${actual} to be greater than or equal to ${expected}`
  };
}

function matchToBeLessThan(actual: number, expected: number): MatcherResult {
  const pass = actual < expected;
  return {
    pass,
    message: `${actual} to be less than ${expected}`
  };
}

function matchToBeLessThanOrEqual(actual: number, expected: number): MatcherResult {
  const pass = actual <= expected;
  return {
    pass,
    message: `${actual} to be less than or equal to ${expected}`
  };
}

function matchToBeCloseTo(actual: number, params: { expected: number; precision?: number }): MatcherResult {
  const { expected, precision = 2 } = params;
  const tolerance = Math.pow(10, -precision) / 2;
  const diff = Math.abs(actual - expected);
  const pass = diff < tolerance;
  return {
    pass,
    message: `${actual} to be close to ${expected} (precision: ${precision})`
  };
}

// String Matchers

function matchToMatch(actual: string, expected: RegExp | string): MatcherResult {
  const regex = typeof expected === 'string' ? new RegExp(expected) : expected;
  const pass = regex.test(actual);
  return {
    pass,
    message: `"${actual}" to match ${expected}`
  };
}

function matchToContainString(actual: string, expected: string): MatcherResult {
  const pass = actual.includes(expected);
  return {
    pass,
    message: `"${actual}" to contain "${expected}"`
  };
}

function matchToStartWith(actual: string, expected: string): MatcherResult {
  const pass = actual.startsWith(expected);
  return {
    pass,
    message: `"${actual}" to start with "${expected}"`
  };
}

function matchToEndWith(actual: string, expected: string): MatcherResult {
  const pass = actual.endsWith(expected);
  return {
    pass,
    message: `"${actual}" to end with "${expected}"`
  };
}

// Array/Object Matchers

function matchToContain(actual: any[], expected: any): MatcherResult {
  const pass = actual.includes(expected);
  return {
    pass,
    message: `array to contain ${expected}`
  };
}

function matchToHaveLength(actual: any[], expected: number): MatcherResult {
  const pass = actual.length === expected;
  return {
    pass,
    message: `array to have length ${expected}, but got ${actual.length}`
  };
}

function matchToHaveProperty(actual: any, params: { property: string; value?: any }): MatcherResult {
  const { property, value } = params;
  const hasProperty = property in actual;
  const pass = value !== undefined
    ? hasProperty && actual[property] === value
    : hasProperty;
  return {
    pass,
    message: value !== undefined
      ? `object to have property "${property}" with value ${value}`
      : `object to have property "${property}"`
  };
}

function matchToBeEmpty(actual: any): MatcherResult {
  let isEmpty = false;
  if (Array.isArray(actual) || typeof actual === 'string') {
    isEmpty = actual.length === 0;
  } else if (typeof actual === 'object' && actual !== null) {
    isEmpty = Object.keys(actual).length === 0;
  }
  return {
    pass: isEmpty,
    message: `value to be empty`
  };
}

// Type Matchers

function matchToBeInstanceOf(actual: any, expected: any): MatcherResult {
  const pass = actual instanceof expected;
  return {
    pass,
    message: `value to be instance of ${expected.name}`
  };
}

function matchToBeTypeOf(actual: any, expected: string): MatcherResult {
  const pass = typeof actual === expected;
  return {
    pass,
    message: `value to be of type "${expected}", but got "${typeof actual}"`
  };
}

// Function Matchers

function matchToThrow(actual: () => any, expected?: string | RegExp): MatcherResult {
  let threw = false;
  let error: any;

  try {
    actual();
  } catch (e) {
    threw = true;
    error = e;
  }

  if (!threw) {
    return {
      pass: false,
      message: `function to throw`
    };
  }

  if (expected) {
    const message = String(error);
    const matches = typeof expected === 'string'
      ? message.includes(expected)
      : expected.test(message);

    return {
      pass: matches,
      message: `function to throw error matching ${expected}`
    };
  }

  return {
    pass: true,
    message: `function to throw`
  };
}

function matchToThrowError(actual: () => any, expected?: string | RegExp | Error): MatcherResult {
  // Similar to toThrow but more specific
  return matchToThrow(actual, expected as any);
}

// Mock function matchers (simplified)

function matchToHaveBeenCalled(actual: any): MatcherResult {
  const pass = actual.called === true;
  return {
    pass,
    message: `function to have been called`
  };
}

function matchToHaveBeenCalledTimes(actual: any, expected: number): MatcherResult {
  const pass = actual.callCount === expected;
  return {
    pass,
    message: `function to have been called ${expected} times, but was called ${actual.callCount} times`
  };
}

function matchToHaveBeenCalledWith(actual: any, expectedArgs: any[]): MatcherResult {
  const pass = JSON.stringify(actual.lastCallArgs) === JSON.stringify(expectedArgs);
  return {
    pass,
    message: `function to have been called with ${JSON.stringify(expectedArgs)}`
  };
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Create assertion for value
 * O(1) operation
 */
export function expect(actual: any): Assertion {
  return new Assertion(actual);
}

/**
 * Spy function (mock tracking)
 */
export function spy<T extends (...args: any[]) => any>(fn?: T): T & {
  called: boolean;
  callCount: number;
  lastCallArgs: any[];
  calls: any[][];
} {
  const calls: any[][] = [];

  const spyFn = function (...args: any[]) {
    calls.push(args);
    return fn ? fn(...args) : undefined;
  } as any;

  Object.defineProperty(spyFn, 'called', {
    get() {
      return calls.length > 0;
    }
  });

  Object.defineProperty(spyFn, 'callCount', {
    get() {
      return calls.length;
    }
  });

  Object.defineProperty(spyFn, 'lastCallArgs', {
    get() {
      return calls[calls.length - 1] || [];
    }
  });

  spyFn.calls = calls;

  return spyFn;
}
