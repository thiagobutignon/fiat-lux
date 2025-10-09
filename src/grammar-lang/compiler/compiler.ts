/**
 * Grammar Language Compiler
 *
 * Main entry point: parse → type check → transpile
 */

import { parseProgram } from './parser';
import { checkProgram } from '../core/type-checker';
import { transpileProgram } from './transpiler';

export interface CompileOptions {
  target?: 'javascript' | 'llvm';
  optimize?: boolean;
  sourceMap?: boolean;
}

export interface CompileResult {
  code: string;
  errors: CompileError[];
  warnings: string[];
}

export interface CompileError {
  message: string;
  location?: any;
}

/**
 * Compile Grammar Language to target language
 *
 * O(n) where n = number of definitions
 * Each step is O(1) per definition
 */
export function compile(source: string | any[], options: CompileOptions = {}): CompileResult {
  const errors: CompileError[] = [];
  const warnings: string[] = [];

  try {
    // Step 1: Parse (already done by Grammar Engine - O(1))
    // For now, assume source is already parsed S-expressions
    const sexprs = Array.isArray(source) ? source : JSON.parse(source);
    const ast = parseProgram(sexprs);

    // Step 2: Type check (O(n) definitions, O(1) per definition)
    try {
      checkProgram(ast);
    } catch (e: any) {
      errors.push({
        message: e.message,
        location: e.loc
      });
      return { code: '', errors, warnings };
    }

    // Step 3: Transpile (O(n) definitions)
    const target = options.target || 'javascript';

    let code: string;
    if (target === 'javascript') {
      code = transpileProgram(ast);
    } else {
      throw new Error(`Target ${target} not implemented yet`);
    }

    return { code, errors, warnings };

  } catch (e: any) {
    errors.push({
      message: `Compilation failed: ${e.message}`
    });
    return { code: '', errors, warnings };
  }
}

/**
 * Compile and run Grammar Language code
 */
export function compileAndRun(source: string | any[], options: CompileOptions = {}): any {
  const result = compile(source, options);

  if (result.errors.length > 0) {
    throw new Error(result.errors.map(e => e.message).join('\n'));
  }

  // Execute the JavaScript code
  return eval(result.code);
}
