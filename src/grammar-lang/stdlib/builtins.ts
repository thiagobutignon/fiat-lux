/**
 * Grammar Language Built-in Functions
 *
 * Low-level primitives implemented in TypeScript
 * These are available to all Grammar Language programs
 */

import { Type, Types } from '../core/types';

// ============================================================================
// Built-in Function Registry
// ============================================================================

export interface BuiltinFunction {
  name: string;
  type: Type;
  impl: (...args: any[]) => any;
}

/**
 * All built-in functions available in Grammar Language
 */
export const BUILTINS: BuiltinFunction[] = [
  // Arithmetic
  {
    name: '+',
    type: Types.function([Types.integer(), Types.integer()], Types.integer()),
    impl: (a: number, b: number) => a + b
  },
  {
    name: '-',
    type: Types.function([Types.integer(), Types.integer()], Types.integer()),
    impl: (a: number, b: number) => a - b
  },
  {
    name: '*',
    type: Types.function([Types.integer(), Types.integer()], Types.integer()),
    impl: (a: number, b: number) => a * b
  },
  {
    name: '/',
    type: Types.function([Types.integer(), Types.integer()], Types.integer()),
    impl: (a: number, b: number) => Math.floor(a / b)
  },
  {
    name: '%',
    type: Types.function([Types.integer(), Types.integer()], Types.integer()),
    impl: (a: number, b: number) => a % b
  },

  // Comparison
  {
    name: '=',
    type: Types.function([Types.integer(), Types.integer()], Types.boolean()),
    impl: (a: number, b: number) => a === b
  },
  {
    name: '<',
    type: Types.function([Types.integer(), Types.integer()], Types.boolean()),
    impl: (a: number, b: number) => a < b
  },
  {
    name: '<=',
    type: Types.function([Types.integer(), Types.integer()], Types.boolean()),
    impl: (a: number, b: number) => a <= b
  },
  {
    name: '>',
    type: Types.function([Types.integer(), Types.integer()], Types.boolean()),
    impl: (a: number, b: number) => a > b
  },
  {
    name: '>=',
    type: Types.function([Types.integer(), Types.integer()], Types.boolean()),
    impl: (a: number, b: number) => a >= b
  },

  // String operations
  {
    name: 'concat',
    type: Types.function([Types.string(), Types.string()], Types.string()),
    impl: (a: string, b: string) => a + b
  },
  {
    name: 'builtin-uppercase',
    type: Types.function([Types.string()], Types.string()),
    impl: (s: string) => s.toUpperCase()
  },
  {
    name: 'builtin-lowercase',
    type: Types.function([Types.string()], Types.string()),
    impl: (s: string) => s.toLowerCase()
  },
  {
    name: 'string-length',
    type: Types.function([Types.string()], Types.integer()),
    impl: (s: string) => s.length
  },
  {
    name: 'substring',
    type: Types.function([Types.string(), Types.integer(), Types.integer()], Types.string()),
    impl: (s: string, start: number, end: number) => s.substring(start, end)
  },

  // List operations
  {
    name: 'empty-list',
    type: Types.function([], Types.list(Types.typevar('a'))),
    impl: () => []
  },
  {
    name: 'cons',
    type: Types.function(
      [Types.typevar('a'), Types.list(Types.typevar('a'))],
      Types.list(Types.typevar('a'))
    ),
    impl: (x: any, xs: any[]) => [x, ...xs]
  },
  {
    name: 'head',
    type: Types.function([Types.list(Types.typevar('a'))], Types.typevar('a')),
    impl: (xs: any[]) => {
      if (xs.length === 0) throw new Error('head: empty list');
      return xs[0];
    }
  },
  {
    name: 'tail',
    type: Types.function([Types.list(Types.typevar('a'))], Types.list(Types.typevar('a'))),
    impl: (xs: any[]) => {
      if (xs.length === 0) throw new Error('tail: empty list');
      return xs.slice(1);
    }
  },
  {
    name: 'empty?',
    type: Types.function([Types.list(Types.typevar('a'))], Types.boolean()),
    impl: (xs: any[]) => xs.length === 0
  },

  // IO
  {
    name: 'builtin-print',
    type: Types.function([Types.string()], Types.unit()),
    impl: (s: string) => {
      console.log(s);
      return null;
    }
  },
  {
    name: 'builtin-read-line',
    type: Types.function([], Types.string()),
    impl: () => {
      // In browser/node REPL context
      if (typeof window !== 'undefined') {
        return prompt('') || '';
      }
      // In Node.js
      const readline = require('readline');
      const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
      });
      return new Promise(resolve => {
        rl.question('', (answer: string) => {
          rl.close();
          resolve(answer);
        });
      });
    }
  },

  // Panic (for errors)
  {
    name: 'panic',
    type: Types.function([Types.string()], Types.typevar('a')),
    impl: (msg: string) => {
      throw new Error(`Panic: ${msg}`);
    }
  },

  // Debug
  {
    name: 'debug',
    type: Types.function([Types.typevar('a')], Types.typevar('a')),
    impl: (x: any) => {
      console.log('[DEBUG]', x);
      return x;
    }
  }
];

/**
 * Get built-in function by name
 */
export function getBuiltin(name: string): BuiltinFunction | undefined {
  return BUILTINS.find(b => b.name === name);
}

/**
 * Check if name is a built-in
 */
export function isBuiltin(name: string): boolean {
  return BUILTINS.some(b => b.name === name);
}

/**
 * Get all built-in names
 */
export function getBuiltinNames(): string[] {
  return BUILTINS.map(b => b.name);
}
