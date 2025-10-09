/**
 * Grammar Language Compiler
 *
 * Main entry point: parse → type check → transpile
 */

import { parseProgram } from './parser';
import { checkProgram } from '../core/type-checker';
import { transpileProgram } from './transpiler';
import {
  ModuleRegistry,
  buildDependencyGraph,
  topologicalSort,
  loadPackageManifest,
  resolveDependencies
} from './module-resolver';
import * as path from 'path';

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

/**
 * Compile module with dependencies
 * O(m + n) where m = modules, n = expressions
 */
export function compileModule(
  entrypoint: string,
  options: CompileOptions = {}
): CompileResult {
  const errors: CompileError[] = [];
  const warnings: string[] = [];

  try {
    const rootDir = path.dirname(entrypoint);
    const registry = new ModuleRegistry(rootDir);

    // Load package manifest if exists
    const manifest = loadPackageManifest(rootDir);
    if (manifest) {
      resolveDependencies(manifest, registry);
    }

    // Build dependency graph (BFS)
    const graph = buildDependencyGraph(entrypoint, registry);

    // Topological sort (compilation order)
    const compilationOrder = topologicalSort(graph);

    // Compile each module in order
    const moduleCode: string[] = [];

    for (const modulePath of compilationOrder) {
      const moduleInfo = graph.nodes.get(modulePath)!;

      // Type check module
      try {
        checkProgram(moduleInfo.definitions);
      } catch (e: any) {
        errors.push({
          message: `${moduleInfo.name}: ${e.message}`,
          location: e.loc
        });
        return { code: '', errors, warnings };
      }

      // Transpile module
      const code = transpileProgram(moduleInfo.definitions);

      // Wrap in module namespace
      const wrapped = `
// Module: ${moduleInfo.name}
(function() {
  ${code}

  // Exports
  ${moduleInfo.exports.map(name => `  globalThis.${moduleInfo.name}$${name} = ${name};`).join('\n')}
})();
`;

      moduleCode.push(wrapped);
    }

    // Combine all modules
    const finalCode = moduleCode.join('\n\n');

    return { code: finalCode, errors, warnings };

  } catch (e: any) {
    errors.push({
      message: `Module compilation failed: ${e.message}`
    });
    return { code: '', errors, warnings };
  }
}

/**
 * Compile and bundle multiple modules
 */
export function compileBundle(
  entrypoint: string,
  options: CompileOptions & { output?: string } = {}
): CompileResult {
  const result = compileModule(entrypoint, options);

  if (result.errors.length === 0 && options.output) {
    const fs = require('fs');
    fs.writeFileSync(options.output, result.code);
  }

  return result;
}
