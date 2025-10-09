#!/usr/bin/env tsx
/**
 * Grammar Language REPL (Improved)
 *
 * Full-featured REPL with:
 * - Multi-line input
 * - Persistent environment
 * - Pretty printing
 * - Error recovery
 */

import * as readline from 'readline';
import { compile } from '../compiler/compiler';
import { TypeEnv } from '../core/types';
import { BUILTINS } from '../stdlib/builtins';

// ============================================================================
// REPL State
// ============================================================================

interface REPLState {
  env: TypeEnv;
  history: string[];
  definitions: any[];
  lastResult: any;
}

const state: REPLState = {
  env: new TypeEnv(),
  history: [],
  definitions: [],
  lastResult: null
};

// ============================================================================
// REPL Interface
// ============================================================================

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  prompt: 'gl> ',
  completer: (line: string) => {
    // Auto-complete built-in functions
    const builtinNames = BUILTINS.map(b => b.name);
    const hits = builtinNames.filter(name => name.startsWith(line));
    return [hits.length ? hits : builtinNames, line];
  }
});

// ============================================================================
// Pretty Printer
// ============================================================================

function prettyPrint(value: any, depth: number = 0): string {
  if (value === null || value === undefined) {
    return 'unit';
  }

  if (typeof value === 'boolean') {
    return value ? 'true' : 'false';
  }

  if (typeof value === 'number') {
    return String(value);
  }

  if (typeof value === 'string') {
    return `"${value}"`;
  }

  if (Array.isArray(value)) {
    if (value.length === 0) {
      return '[]';
    }
    if (depth > 3) {
      return '[...]';
    }
    const items = value.map(v => prettyPrint(v, depth + 1)).join(', ');
    return `[${items}]`;
  }

  if (typeof value === 'object') {
    const keys = Object.keys(value);
    if (keys.length === 0) {
      return '{}';
    }
    if (depth > 3) {
      return '{...}';
    }
    const fields = keys
      .map(k => `${k}: ${prettyPrint(value[k], depth + 1)}`)
      .join(', ');
    return `{ ${fields} }`;
  }

  if (typeof value === 'function') {
    return '<function>';
  }

  return String(value);
}

// ============================================================================
// Expression Evaluator
// ============================================================================

function evaluateExpression(input: string): any {
  try {
    // Parse S-expression
    const sexpr = parseSimple(input);

    // Check if it's a definition
    if (Array.isArray(sexpr) && (sexpr[0] === 'define' || sexpr[0] === 'type')) {
      // Compile and execute definition
      const allDefs = [...state.definitions, sexpr];
      const result = compile(allDefs);

      if (result.errors.length > 0) {
        throw new Error(result.errors[0].message);
      }

      // Execute the code to register the definition
      eval(result.code);

      // Store definition
      state.definitions.push(sexpr);

      return `Defined ${sexpr[1]}`;
    }

    // It's an expression - wrap in a function and evaluate
    const wrapped = [
      ['define', '__repl_eval', ['unit', '->', 'integer'], sexpr]
    ];

    const allDefs = [...state.definitions, ...wrapped];
    const result = compile(allDefs);

    if (result.errors.length > 0) {
      throw new Error(result.errors[0].message);
    }

    // Execute and get result
    eval(result.code);
    const value = (globalThis as any).__repl_eval();

    return value;

  } catch (e: any) {
    throw new Error(`Evaluation error: ${e.message}`);
  }
}

// ============================================================================
// Simple S-expression Parser
// ============================================================================

function parseSimple(input: string): any {
  input = input.trim();

  // Try to parse as JSON-like structure
  // This is a hack - in production use Grammar Engine
  try {
    // Handle special cases
    if (input === 'true') return true;
    if (input === 'false') return false;
    if (input === 'unit' || input === 'null') return null;

    // Try number
    if (/^-?\d+$/.test(input)) {
      return parseInt(input, 10);
    }

    // Try string
    if (input.startsWith('"') && input.endsWith('"')) {
      return input.slice(1, -1);
    }

    // Try S-expression (simple parser)
    if (input.startsWith('(') && input.endsWith(')')) {
      const inner = input.slice(1, -1);
      const parts = splitSexp(inner);
      return parts.map(parseSimple);
    }

    // Try list literal
    if (input.startsWith('[') && input.endsWith(']')) {
      const inner = input.slice(1, -1);
      if (inner.trim() === '') return [];
      const parts = inner.split(/\s+/);
      return parts.map(parseSimple);
    }

    // Otherwise it's a symbol
    return input;

  } catch (e: any) {
    throw new Error(`Parse error: ${e.message}`);
  }
}

function splitSexp(input: string): string[] {
  const parts: string[] = [];
  let current = '';
  let depth = 0;
  let inString = false;

  for (let i = 0; i < input.length; i++) {
    const char = input[i];

    if (char === '"' && (i === 0 || input[i - 1] !== '\\')) {
      inString = !inString;
      current += char;
    } else if (inString) {
      current += char;
    } else if (char === '(' || char === '[') {
      depth++;
      current += char;
    } else if (char === ')' || char === ']') {
      depth--;
      current += char;
    } else if (char === ' ' && depth === 0) {
      if (current.trim()) {
        parts.push(current.trim());
        current = '';
      }
    } else {
      current += char;
    }
  }

  if (current.trim()) {
    parts.push(current.trim());
  }

  return parts;
}

// ============================================================================
// Commands
// ============================================================================

function handleCommand(cmd: string): void {
  const parts = cmd.split(/\s+/);
  const command = parts[0];

  switch (command) {
    case ':help':
    case ':h':
      console.log(`
Grammar Language REPL

Commands:
  :help, :h          Show this help
  :quit, :q          Exit REPL
  :clear, :c         Clear screen
  :history           Show command history
  :defs              Show all definitions
  :env               Show environment bindings
  :last              Show last result
  :reset             Reset REPL state
  :builtins          Show all built-in functions

Examples:
  (+ 1 2)
  (define double (integer -> integer) (* $1 2))
  (double 5)
  (map double [1 2 3 4 5])
`);
      break;

    case ':quit':
    case ':q':
      console.log('Goodbye!');
      process.exit(0);
      break;

    case ':clear':
    case ':c':
      console.clear();
      break;

    case ':history':
      state.history.forEach((h, i) => console.log(`${i + 1}: ${h}`));
      break;

    case ':defs':
      if (state.definitions.length === 0) {
        console.log('No definitions yet');
      } else {
        state.definitions.forEach(def => {
          console.log(JSON.stringify(def));
        });
      }
      break;

    case ':env':
      console.log('Environment bindings:');
      // TODO: implement env.getAll()
      console.log('(Built-ins + user definitions)');
      break;

    case ':last':
      console.log('Last result:', prettyPrint(state.lastResult));
      break;

    case ':reset':
      state.definitions = [];
      state.history = [];
      state.lastResult = null;
      console.log('REPL state reset');
      break;

    case ':builtins':
      console.log(`Built-in functions (${BUILTINS.length}):`);
      BUILTINS.forEach(b => {
        console.log(`  ${b.name}`);
      });
      break;

    default:
      console.log(`Unknown command: ${command}`);
      console.log('Type :help for available commands');
  }
}

// ============================================================================
// Main REPL Loop
// ============================================================================

console.log('Grammar Language REPL v0.2.0');
console.log('Type :help for commands, :quit to exit\n');

rl.prompt();

rl.on('line', (line: string) => {
  const input = line.trim();

  // Skip empty lines
  if (input === '') {
    rl.prompt();
    return;
  }

  // Handle commands
  if (input.startsWith(':')) {
    handleCommand(input);
    rl.prompt();
    return;
  }

  // Evaluate expression
  try {
    const result = evaluateExpression(input);
    state.lastResult = result;
    state.history.push(input);

    console.log('=>', prettyPrint(result));
  } catch (e: any) {
    console.error('Error:', e.message);
  }

  rl.prompt();
});

rl.on('close', () => {
  console.log('\nGoodbye!');
  process.exit(0);
});

// Handle Ctrl+C
process.on('SIGINT', () => {
  console.log('\n(To exit, type :quit or press Ctrl+D)');
  rl.prompt();
});
