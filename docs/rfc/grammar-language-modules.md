# RFC: Grammar Language Module System

**Status:** Phase 2 - Q2 2025
**Depends on:** Grammar Language Core (Phase 1)

## Overview

Module system for Grammar Language enabling:
- Code organization across multiple files
- Explicit dependency management
- O(1) module resolution
- AGI-friendly package distribution

## Motivation

AGI will generate thousands of files. Need:
1. **Namespace isolation** - Avoid name collisions
2. **Explicit imports** - No global scope pollution
3. **Fast resolution** - O(1) module lookup
4. **Self-describing** - Manifest declares all dependencies

## Design Principles

1. **Explicit over implicit** - All imports must be declared
2. **Flat is better than nested** - Simple dependency graph
3. **No circular dependencies** - Topological ordering enforced
4. **Single source of truth** - gl.json declares everything

## Module Syntax

### Module Declaration

```scheme
;; File: src/math-utils.gl
(module math-utils
  (export add-vectors scale-vector dot-product))

(define add-vectors ((list integer) (list integer) -> (list integer))
  (map2 + $1 $2))

(define scale-vector (integer (list integer) -> (list integer))
  (map (lambda ((x integer)) (* $1 x)) $2))

(define dot-product ((list integer) (list integer) -> integer)
  (sum (map2 * $1 $2)))

;; Private function (not exported)
(define map2 ((integer integer -> integer) (list integer) (list integer) -> (list integer))
  (if (or (empty? $2) (empty? $3))
    (empty-list)
    (cons
      ($1 (head $2) (head $3))
      (map2 $1 (tail $2) (tail $3)))))
```

### Importing Modules

```scheme
;; File: src/main.gl
(import math-utils (add-vectors scale-vector))
(import (std list) (map filter fold))
(import (std io) (print println))

(define main (unit -> unit)
  (let v1 (list integer) [1 2 3])
  (let v2 (list integer) [4 5 6])
  (let result (list integer) (add-vectors v1 v2))
  (println (to-string result)))
```

### Re-exporting

```scheme
;; File: src/vector-lib.gl
(module vector-lib
  (export
    ;; From math-utils
    add-vectors
    scale-vector
    ;; New exports
    normalize
    magnitude))

(import math-utils (add-vectors scale-vector dot-product))

(define magnitude ((list integer) -> integer)
  (sqrt (dot-product $1 $1)))

(define normalize ((list integer) -> (list float))
  (let mag integer (magnitude $1))
  (map (lambda ((x integer)) (/ x mag)) $1))
```

## Package Format (gl.json)

```json
{
  "name": "@agi/vector-math",
  "version": "1.0.0",
  "description": "Vector math library for AGI",
  "main": "src/index.gl",
  "exports": {
    ".": "./src/index.gl",
    "./utils": "./src/utils.gl"
  },
  "dependencies": {
    "@agi/std": "^2.0.0",
    "@agi/math": "^1.5.0"
  },
  "devDependencies": {
    "@agi/test": "^1.0.0"
  },
  "grammar": {
    "version": "0.2.0",
    "strict": true,
    "typecheck": "strict"
  },
  "scripts": {
    "build": "glc src/index.gl -o dist/",
    "test": "glc test/test.gl && node dist/test.js"
  }
}
```

## Module Resolution Algorithm

### 1. O(1) Lookup

```typescript
interface ModuleRegistry {
  // Map: module name → absolute path
  modules: Map<string, string>;

  // Map: package name → package root
  packages: Map<string, string>;
}

function resolveModule(name: string): string {
  // Check if it's a relative import
  if (name.startsWith('./') || name.startsWith('../')) {
    return path.resolve(currentDir, name);
  }

  // Check if it's a package import
  if (name.startsWith('@')) {
    const [scope, pkg] = name.split('/');
    const pkgRoot = registry.packages.get(`${scope}/${pkg}`);
    return path.join(pkgRoot, 'src/index.gl');
  }

  // Check if it's a stdlib import
  if (name === 'std' || name.startsWith('std/')) {
    return resolveStdlib(name);
  }

  // Otherwise it's a local module
  return registry.modules.get(name) || throwError(`Module not found: ${name}`);
}
```

### 2. Dependency Graph

Build dependency graph upfront:

```typescript
interface DependencyGraph {
  nodes: Map<string, ModuleNode>;
  edges: Map<string, Set<string>>;
}

interface ModuleNode {
  name: string;
  path: string;
  exports: string[];
  imports: Import[];
  ast: Definition[];
}

function buildGraph(entrypoint: string): DependencyGraph {
  const graph: DependencyGraph = { nodes: new Map(), edges: new Map() };
  const queue: string[] = [entrypoint];
  const visited = new Set<string>();

  while (queue.length > 0) {
    const current = queue.shift()!;
    if (visited.has(current)) continue;
    visited.add(current);

    const node = parseModule(current);
    graph.nodes.set(current, node);

    for (const imp of node.imports) {
      const depPath = resolveModule(imp.from);
      graph.edges.set(current, (graph.edges.get(current) || new Set()).add(depPath));
      queue.push(depPath);
    }
  }

  return graph;
}
```

### 3. Topological Sort

Compile modules in dependency order:

```typescript
function topologicalSort(graph: DependencyGraph): string[] {
  const sorted: string[] = [];
  const visiting = new Set<string>();
  const visited = new Set<string>();

  function visit(node: string) {
    if (visited.has(node)) return;
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
```

## Compilation Pipeline

```
1. Read gl.json → Load package manifest
2. Resolve dependencies → Build module registry
3. Parse entrypoint → Get initial imports
4. Build dep graph → BFS from entrypoint
5. Topological sort → Get compilation order
6. Type check → Each module in order
7. Transpile → Generate JavaScript
8. Bundle → Single output file (optional)
```

### Example Compilation

```bash
# Compile single file
glc src/main.gl -o dist/main.js

# Compile with dependencies
glc src/main.gl --bundle -o dist/bundle.js

# Type check only
glc src/main.gl --check

# Watch mode
glc src/main.gl --watch
```

## Package Manager (glpm)

### Commands

```bash
# Initialize new package
glpm init

# Install dependencies
glpm install
glpm install @agi/math
glpm install @agi/math@1.5.0

# Update dependencies
glpm update
glpm update @agi/math

# Remove dependencies
glpm remove @agi/math

# Publish package
glpm publish

# Search packages
glpm search vector

# Info about package
glpm info @agi/math
```

### Registry Structure

```
~/.glpm/
  registry.json       # Package index
  packages/
    @agi/
      math@1.5.0/
        gl.json
        src/
        dist/
      std@2.0.0/
        ...
```

## Standard Library Organization

```
std/
  list.gl          # (import (std list) ...)
  math.gl          # (import (std math) ...)
  string.gl        # (import (std string) ...)
  io.gl            # (import (std io) ...)
  option.gl        # (import (std option) ...)
  result.gl        # (import (std result) ...)
  async.gl         # (import (std async) ...)
  json.gl          # (import (std json) ...)
```

Each stdlib module exports specific functions:

```scheme
;; std/list.gl
(module (std list)
  (export map filter fold reverse append length head tail empty? cons))

;; std/math.gl
(module (std math)
  (export abs max min sqrt pow sin cos tan factorial fibonacci))

;; std/io.gl
(module (std io)
  (export print println read-line read-file write-file))
```

## Type Checking with Modules

### 1. Module-Level Type Environment

```typescript
interface ModuleTypeEnv {
  module: string;
  env: TypeEnv;
  exports: Map<string, Type>;
}

function checkModule(node: ModuleNode, deps: ModuleTypeEnv[]): ModuleTypeEnv {
  const env = new TypeEnv();

  // Add imports from dependencies
  for (const imp of node.imports) {
    const depEnv = deps.find(d => d.module === imp.from);
    if (!depEnv) throw new Error(`Module not found: ${imp.from}`);

    for (const name of imp.names) {
      const type = depEnv.exports.get(name);
      if (!type) throw new Error(`Export not found: ${name} in ${imp.from}`);
      env.bind(name, type);
    }
  }

  // Type check definitions
  for (const def of node.ast) {
    checkDefinition(def, env);
  }

  // Extract exports
  const exports = new Map<string, Type>();
  for (const name of node.exports) {
    const type = env.lookup(name);
    if (!type) throw new Error(`Export not defined: ${name}`);
    exports.set(name, type);
  }

  return { module: node.name, env, exports };
}
```

### 2. Cross-Module Type Checking

Still O(1) per expression! Only module resolution is O(m) where m = number of modules.

```
Total complexity: O(m + n)
  m = number of modules (typically < 1000)
  n = number of expressions (millions)

Each expression: O(1) ✅
Module resolution: O(m) upfront ✅
```

## Security & Sandboxing

### 1. Explicit Permissions

```json
{
  "name": "@agi/file-utils",
  "permissions": {
    "fs": ["read", "write"],
    "net": ["fetch"],
    "process": []
  }
}
```

### 2. Runtime Checks

```typescript
function checkPermission(pkg: string, resource: string, action: string) {
  const perms = registry.get(pkg)?.permissions?.[resource] || [];
  if (!perms.includes(action)) {
    throw new PermissionError(`${pkg} not allowed to ${action} ${resource}`);
  }
}
```

## AGI Use Cases

### 1. Self-Organizing Code

AGI can create new modules dynamically:

```scheme
;; AGI creates optimization module
(module agi-optimizations-v47
  (export optimized-sort optimized-search))

(define optimized-sort ((list integer) -> (list integer))
  ;; AGI discovered faster algorithm
  ...)
```

### 2. Dependency Evolution

AGI can replace dependencies:

```json
{
  "dependencies": {
    "old-lib": "1.0.0"  // AGI replaces with better version
  }
}
```

→

```json
{
  "dependencies": {
    "agi-lib-v2": "1.0.0"  // AGI's improved version
  }
}
```

### 3. Module Versioning

AGI tracks which modules work:

```scheme
;; AGI tries different versions
(import math-utils-v1 (solve))  ;; 70% success rate
(import math-utils-v2 (solve))  ;; 85% success rate
(import math-utils-v3 (solve))  ;; 95% success rate ← use this
```

## Implementation Timeline

### Week 1: Core Module System
- Module parser (import/export syntax)
- Module resolver (O(1) lookup)
- Dependency graph builder

### Week 2: Compilation
- Topological sort
- Cross-module type checking
- Bundle generation

### Week 3: Package Manager
- glpm CLI tool
- Package installation
- Registry integration

### Week 4: Documentation & Tests
- Module examples
- Package creation guide
- Comprehensive tests

## Success Metrics

- ✅ O(1) module resolution
- ✅ Handles 10,000+ modules
- ✅ No circular dependencies
- ✅ Type-safe cross-module calls
- ✅ AGI can create/organize modules
- ✅ <100ms to resolve 1000 modules

## References

- [Grammar Language RFC](./grammar-language.md)
- [Issue #19](https://github.com/thiagobutignon/chomsky/issues/19)
- [TypeScript Module Resolution](https://www.typescriptlang.org/docs/handbook/module-resolution.html) (what NOT to do)

---

**Modules enable AGI to organize knowledge at scale.**
