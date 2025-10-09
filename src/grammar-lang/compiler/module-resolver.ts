/**
 * Grammar Language Module Resolver
 *
 * O(1) module resolution and dependency graph building
 */

import * as fs from 'fs';
import * as path from 'path';
import { Definition, ModuleDef, ImportDecl } from '../core/ast';
import { parseProgram } from './parser';

// ============================================================================
// Module Registry
// ============================================================================

export interface ModuleInfo {
  name: string;
  path: string;
  exports: string[];
  imports: ImportDecl[];
  definitions: Definition[];
}

export class ModuleRegistry {
  // Map: module name → module info
  private modules = new Map<string, ModuleInfo>();

  // Map: package name → package root
  private packages = new Map<string, string>();

  constructor(private rootDir: string) {}

  /**
   * Register a module
   */
  register(info: ModuleInfo): void {
    this.modules.set(info.name, info);
  }

  /**
   * Resolve module name to absolute path
   * O(1) lookup
   */
  resolve(name: string, fromFile?: string): string {
    // Relative import: ./module or ../module
    if (name.startsWith('./') || name.startsWith('../')) {
      if (!fromFile) {
        throw new Error(`Relative import ${name} requires fromFile`);
      }
      const dir = path.dirname(fromFile);
      return path.resolve(dir, name + '.gl');
    }

    // Package import: @agi/math
    if (name.startsWith('@')) {
      const [scope, pkg] = name.split('/');
      const pkgRoot = this.packages.get(`${scope}/${pkg}`);
      if (!pkgRoot) {
        throw new Error(`Package not found: ${scope}/${pkg}`);
      }
      return path.join(pkgRoot, 'src/index.gl');
    }

    // Stdlib import: std/list or (std list)
    if (name === 'std' || name.startsWith('std/')) {
      const moduleName = name.replace('std/', '');
      const stdlibPath = path.join(this.rootDir, 'stdlib', `${moduleName}.gl`);
      if (fs.existsSync(stdlibPath)) {
        return stdlibPath;
      }
      throw new Error(`Stdlib module not found: ${name}`);
    }

    // Local module: direct lookup
    const info = this.modules.get(name);
    if (info) {
      return info.path;
    }

    throw new Error(`Module not found: ${name}`);
  }

  /**
   * Get module info
   */
  get(name: string): ModuleInfo | undefined {
    return this.modules.get(name);
  }

  /**
   * Load module from file
   */
  loadModule(filePath: string): ModuleInfo {
    const source = fs.readFileSync(filePath, 'utf-8');

    // Parse as S-expressions (simplified - in production use Grammar Engine)
    const sexprs = this.parseSource(source);
    const definitions = parseProgram(sexprs);

    // Extract module info
    let moduleName = path.basename(filePath, '.gl');
    let exports: string[] = [];
    let imports: ImportDecl[] = [];
    let defs: Definition[] = [];

    for (const def of definitions) {
      if (def.kind === 'module') {
        if (def.name !== '__import__') {
          // Actual module definition
          moduleName = def.name;
          exports = def.exports;
          defs = def.definitions;
          imports.push(...def.imports);
        } else {
          // Import declaration
          imports.push(...def.imports);
        }
      } else {
        // Top-level definition
        defs.push(def);
        if (def.exported) {
          exports.push(def.name);
        }
      }
    }

    return {
      name: moduleName,
      path: filePath,
      exports,
      imports,
      definitions: defs
    };
  }

  /**
   * Simple S-expression parser (for bootstrap)
   * In production, use Grammar Engine
   */
  private parseSource(source: string): any[] {
    // Remove comments
    source = source.replace(/;[^\n]*/g, '');

    // This is a VERY simplified parser - just for testing
    // In production, use Grammar Engine which is O(1)
    try {
      // Try to parse as JSON-like (hack)
      const jsonLike = source
        .replace(/\(/g, '[')
        .replace(/\)/g, ']')
        .replace(/(\w+)/g, '"$1"')
        .replace(/"(\d+)"/g, '$1')
        .replace(/"true"/g, 'true')
        .replace(/"false"/g, 'false');

      return JSON.parse(`[${jsonLike}]`);
    } catch {
      // Fallback: return empty (will be replaced with Grammar Engine)
      return [];
    }
  }
}

// ============================================================================
// Dependency Graph
// ============================================================================

export interface DependencyGraph {
  nodes: Map<string, ModuleInfo>;
  edges: Map<string, Set<string>>;
}

/**
 * Build dependency graph from entry point
 * BFS traversal
 */
export function buildDependencyGraph(
  entrypoint: string,
  registry: ModuleRegistry
): DependencyGraph {
  const graph: DependencyGraph = {
    nodes: new Map(),
    edges: new Map()
  };

  const queue: string[] = [entrypoint];
  const visited = new Set<string>();

  while (queue.length > 0) {
    const currentPath = queue.shift()!;

    if (visited.has(currentPath)) {
      continue;
    }
    visited.add(currentPath);

    // Load module
    const moduleInfo = registry.loadModule(currentPath);
    registry.register(moduleInfo);
    graph.nodes.set(currentPath, moduleInfo);

    // Process imports
    for (const imp of moduleInfo.imports) {
      const depPath = registry.resolve(imp.module, currentPath);

      // Add edge
      if (!graph.edges.has(currentPath)) {
        graph.edges.set(currentPath, new Set());
      }
      graph.edges.get(currentPath)!.add(depPath);

      // Queue dependency
      queue.push(depPath);
    }
  }

  return graph;
}

/**
 * Topological sort for compilation order
 * Detects circular dependencies
 */
export function topologicalSort(graph: DependencyGraph): string[] {
  const sorted: string[] = [];
  const visiting = new Set<string>();
  const visited = new Set<string>();

  function visit(node: string) {
    if (visited.has(node)) {
      return;
    }

    if (visiting.has(node)) {
      throw new Error(`Circular dependency detected: ${node}`);
    }

    visiting.add(node);

    const deps = graph.edges.get(node) || new Set();
    for (const dep of deps) {
      visit(dep);
    }

    visiting.delete(node);
    visited.add(node);
    sorted.push(node);
  }

  for (const node of graph.nodes.keys()) {
    visit(node);
  }

  return sorted;
}

/**
 * Load package manifest (gl.json)
 */
export interface PackageManifest {
  name: string;
  version: string;
  main: string;
  exports?: Record<string, string>;
  dependencies?: Record<string, string>;
  devDependencies?: Record<string, string>;
}

export function loadPackageManifest(dir: string): PackageManifest | null {
  const manifestPath = path.join(dir, 'gl.json');
  if (!fs.existsSync(manifestPath)) {
    return null;
  }

  const content = fs.readFileSync(manifestPath, 'utf-8');
  return JSON.parse(content);
}

/**
 * Resolve all dependencies from manifest
 */
export function resolveDependencies(
  manifest: PackageManifest,
  registry: ModuleRegistry
): void {
  const deps = {
    ...manifest.dependencies,
    ...manifest.devDependencies
  };

  for (const [name, version] of Object.entries(deps)) {
    // In production, fetch from package registry
    // For now, assume local installation in node_modules
    const pkgPath = path.join(process.cwd(), 'node_modules', name);

    if (fs.existsSync(pkgPath)) {
      registry['packages'].set(name, pkgPath);
    }
  }
}
