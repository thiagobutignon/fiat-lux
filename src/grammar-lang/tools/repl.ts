#!/usr/bin/env tsx
/**
 * Grammar Language REPL
 *
 * Interactive development environment
 */

import * as readline from 'readline';
import { compile } from '../compiler/compiler';
import { TypeEnv } from '../core/types';

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  prompt: 'gl> '
});

const env = new TypeEnv();
const history: string[] = [];

console.log('Grammar Language REPL v0.1.0');
console.log('Type expressions to evaluate, or :help for commands\n');

rl.prompt();

rl.on('line', (line: string) => {
  const input = line.trim();

  // Commands
  if (input.startsWith(':')) {
    handleCommand(input);
    rl.prompt();
    return;
  }

  if (input === '') {
    rl.prompt();
    return;
  }

  // Try to evaluate expression
  try {
    // Parse S-expression (simple parser for REPL)
    const sexpr = parseREPL(input);

    // Compile to JavaScript
    const result = compile([sexpr]);

    if (result.errors.length > 0) {
      console.error('Error:', result.errors[0].message);
    } else {
      // Execute
      const value = eval(result.code);
      console.log('=>', value);

      history.push(input);
    }
  } catch (e: any) {
    console.error('Error:', e.message);
  }

  rl.prompt();
});

rl.on('close', () => {
  console.log('\nGoodbye!');
  process.exit(0);
});

function handleCommand(cmd: string): void {
  const parts = cmd.split(/\s+/);
  const command = parts[0];

  switch (command) {
    case ':help':
    case ':h':
      console.log(`
Commands:
  :help, :h          Show this help
  :quit, :q          Exit REPL
  :clear, :c         Clear screen
  :history           Show command history
  :type <expr>       Show type of expression
  :load <file>       Load and execute file

Examples:
  (+ 1 2)
  (define double (integer -> integer) (* $1 2))
  (double 5)
`);
      break;

    case ':quit':
    case ':q':
      rl.close();
      break;

    case ':clear':
    case ':c':
      console.clear();
      break;

    case ':history':
      history.forEach((h, i) => console.log(`${i + 1}: ${h}`));
      break;

    case ':type':
      const expr = parts.slice(1).join(' ');
      console.log('Type checking not implemented yet');
      break;

    default:
      console.log(`Unknown command: ${command}`);
      console.log('Type :help for available commands');
  }
}

/**
 * Simple S-expression parser for REPL
 * TODO: Use Grammar Engine for proper parsing
 */
function parseREPL(input: string): any {
  // For now, just use JSON-like parsing
  // In production, use Grammar Engine
  return eval(`(${input})`);
}
