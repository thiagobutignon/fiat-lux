# Grammar Language Modules

Complete guide to organizing code with the Grammar Language module system.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Module Declaration](#module-declaration)
3. [Importing Modules](#importing-modules)
4. [Package Management](#package-management)
5. [Compilation](#compilation)
6. [Best Practices](#best-practices)

## Quick Start

### Creating a Module

Create `math-utils.gl`:

```scheme
(module math-utils
  (export add-vectors scale-vector))

(define add-vectors ((list integer) (list integer) -> (list integer))
  (map2 + $1 $2))

(define scale-vector (integer (list integer) -> (list integer))
  (map (lambda ((x integer)) (* $1 x)) $2))

;; Private helper (not exported)
(define map2 ((integer integer -> integer) (list integer) (list integer) -> (list integer))
  (if (or (empty? $2) (empty? $3))
    (empty-list)
    (cons
      ($1 (head $2) (head $3))
      (map2 $1 (tail $2) (tail $3)))))
```

### Using the Module

Create `main.gl`:

```scheme
(import math-utils (add-vectors scale-vector))

(define main (unit -> unit)
  (let v1 (list integer) [1 2 3])
  (let v2 (list integer) [4 5 6])
  (let result (list integer) (add-vectors v1 v2))
  (print (to-string result)))  ;; [5 7 9]
```

### Compile and Run

```bash
# Initialize package
glpm init

# Compile with dependencies
glc main.gl --bundle -o dist/bundle.js

# Run
node dist/bundle.js
```

## Module Declaration

### Syntax

```scheme
(module <name>
  (export <name1> <name2> ...)
  <definitions>)
```

### Example: Vector Math Library

```scheme
(module vector-math
  (export magnitude normalize dot-product cross-product))

(define magnitude ((list integer) -> integer)
  (sqrt (dot-product $1 $1)))

(define normalize ((list integer) -> (list float))
  (let mag integer (magnitude $1))
  (map (lambda ((x integer)) (/ x mag)) $1))

(define dot-product ((list integer) (list integer) -> integer)
  (sum (map2 * $1 $2)))

(define cross-product ((list integer) (list integer) -> (list integer))
  (if (and (= (length $1) 3) (= (length $2) 3))
    (let a1 integer (nth $1 0))
    (let a2 integer (nth $1 1))
    (let a3 integer (nth $1 2))
    (let b1 integer (nth $2 0))
    (let b2 integer (nth $2 1))
    (let b3 integer (nth $2 2))
    [(- (* a2 b3) (* a3 b2))
     (- (* a3 b1) (* a1 b3))
     (- (* a1 b2) (* a2 b1))]
    (panic "Cross product requires 3D vectors")))
```

### Module Without Exports

All definitions are private:

```scheme
(module internal-utils
  (define helper (integer -> integer)
    (* $1 2)))
```

## Importing Modules

### Basic Import

```scheme
(import module-name (fn1 fn2 fn3))
```

### Importing from Stdlib

```scheme
(import (std list) (map filter fold))
(import (std math) (sin cos tan))
(import (std io) (print println read-line))
```

### Importing from Packages

```scheme
(import @agi/math (matrix-multiply solve-linear))
(import @agi/ml (train-model predict))
```

### Relative Imports

```scheme
;; From same directory
(import ./utils (helper1 helper2))

;; From parent directory
(import ../common/types (Vector Matrix))

;; From nested directory
(import ./math/linear-algebra (dot-product))
```

### Multiple Imports

```scheme
(import (std list) (map filter fold))
(import (std math) (abs max min))
(import ./utils (helper1 helper2))

(define process ((list integer) -> integer)
  (fold + 0 (map abs (filter (lambda ((x integer)) (> x 0)) $1))))
```

## Package Management

### Initialize Package

```bash
glpm init
```

Creates `gl.json`:

```json
{
  "name": "@agi/my-package",
  "version": "1.0.0",
  "main": "src/index.gl",
  "exports": {
    ".": "./src/index.gl",
    "./utils": "./src/utils.gl"
  },
  "dependencies": {},
  "devDependencies": {}
}
```

### Install Dependencies

```bash
# Install specific package
glpm install @agi/math

# Install with version
glpm install @agi/math@1.5.0

# Install all from gl.json
glpm install
```

### Remove Dependencies

```bash
glpm remove @agi/math
```

### Update Dependencies

```bash
# Update specific
glpm update @agi/math

# Update all
glpm update
```

### Publish Package

```bash
glpm publish
```

## Compilation

### Compile Single File

```bash
glc src/main.gl
```

### Compile with Output

```bash
glc src/main.gl -o dist/main.js
```

### Compile with Dependencies (Bundle)

```bash
glc src/main.gl --bundle -o dist/bundle.js
```

### Type Check Only

```bash
glc src/main.gl --check
```

### Watch Mode

```bash
glc src/main.gl --watch -o dist/main.js
```

### Compile and Run

```bash
glc src/main.gl --run
```

## Best Practices

### 1. One Module Per File

```scheme
;; ✅ Good: math-utils.gl
(module math-utils
  (export add multiply))

;; ❌ Bad: multiple modules in one file
(module math-utils ...)
(module string-utils ...)
```

### 2. Explicit Exports

Only export what's necessary:

```scheme
(module data-processing
  (export process-data))  ;; Public API

;; Private helpers
(define validate-input ...)
(define transform ...)
(define aggregate ...)

;; Public interface
(define process-data (Data -> Result)
  (aggregate (transform (validate-input $1))))
```

### 3. Organize by Feature

```
src/
  math/
    vector.gl
    matrix.gl
    linear-algebra.gl
  data/
    parser.gl
    validator.gl
    transformer.gl
  ui/
    components.gl
    renderer.gl
```

### 4. Use Stdlib

Don't reinvent the wheel:

```scheme
;; ✅ Good: use stdlib
(import (std list) (map filter fold))

;; ❌ Bad: reimplement
(define my-map ...)
(define my-filter ...)
```

### 5. Dependency Injection

Pass dependencies as parameters:

```scheme
(module processor
  (export make-processor))

;; Takes dependencies as parameters
(define make-processor ((Data -> Result) -> (Data -> Output))
  (lambda ((input Data))
    (let result Result ($1 input))
    (format-output result)))
```

### 6. Avoid Circular Dependencies

```scheme
;; ❌ Bad: A imports B, B imports A
;; module-a.gl
(import module-b (helper))

;; module-b.gl
(import module-a (other-helper))

;; ✅ Good: Extract shared code
;; common.gl
(define shared-helper ...)

;; module-a.gl
(import common (shared-helper))

;; module-b.gl
(import common (shared-helper))
```

## Advanced Patterns

### Re-exporting

Create a unified API:

```scheme
;; index.gl - Main entry point
(import ./math/vector (add-vectors scale-vector))
(import ./math/matrix (matrix-multiply transpose))
(import ./data/parser (parse-json parse-csv))

(module my-library
  (export
    ;; Math
    add-vectors
    scale-vector
    matrix-multiply
    transpose
    ;; Data
    parse-json
    parse-csv))
```

### Conditional Imports

Based on feature flags:

```scheme
;; config.gl
(define use-gpu boolean true)

;; main.gl
(import config (use-gpu))

(define compute (Data -> Result)
  (if use-gpu
    (import ./gpu-compute (process))
    (import ./cpu-compute (process)))
  (process $1))
```

### Dynamic Module Loading

AGI can load modules at runtime:

```scheme
(define load-strategy (string -> Strategy)
  (match $1
    ("fast" (import ./strategies/fast (strategy)))
    ("accurate" (import ./strategies/accurate (strategy)))
    ("balanced" (import ./strategies/balanced (strategy)))
    (_ (panic "Unknown strategy"))))
```

## Module Resolution Order

1. **Relative imports** (`./`, `../`)
   - Resolved relative to current file

2. **Stdlib imports** (`std/`, `(std ...)`)
   - Resolved to `stdlib/` directory

3. **Package imports** (`@agi/...`)
   - Resolved to `node_modules/@agi/.../`

4. **Local modules** (no prefix)
   - Resolved from module registry

## Performance

### Module Resolution: O(1)

Each module is looked up once:

```
Build dependency graph: O(m) where m = number of modules
Resolve each module:    O(1) hash lookup
Total:                  O(m)
```

### Type Checking: Still O(1) per expression

Cross-module types are checked in constant time:

```
Check each module:      O(n) where n = expressions in module
Total across modules:   O(m × n)
Per expression:         O(1) ✅
```

### Compilation Order

Topological sort ensures dependencies compile first:

```
A depends on B, C
B depends on D
C depends on D

Compilation order: D → B → C → A
```

## Troubleshooting

### Module Not Found

```
Error: Module not found: math-utils
```

**Solution:**
1. Check the module name is correct
2. Ensure the file exists: `math-utils.gl`
3. Check the import path: `./math-utils` for relative

### Circular Dependency

```
Error: Circular dependency detected: module-a
```

**Solution:**
Extract shared code into a third module:

```scheme
;; Before:
;; A → B → A (circular)

;; After:
;; A → C
;; B → C (no cycle)
```

### Export Not Found

```
Error: Export not found: helper in math-utils
```

**Solution:**
Add to exports:

```scheme
(module math-utils
  (export helper))  ;; ← Add here
```

### Type Mismatch Across Modules

```
Error: Expected integer, got string
```

**Solution:**
Ensure consistent types across modules:

```scheme
;; module-a.gl
(export (process (Data -> integer)))

;; module-b.gl
(import module-a (process))
(process "string")  ;; ❌ Type error
```

## Examples

### Example 1: Math Library

```
math-lib/
  gl.json
  src/
    index.gl       # Main entry
    vector.gl      # Vector operations
    matrix.gl      # Matrix operations
    linear.gl      # Linear algebra
```

**gl.json:**
```json
{
  "name": "@agi/math-lib",
  "version": "1.0.0",
  "main": "src/index.gl",
  "exports": {
    ".": "./src/index.gl",
    "./vector": "./src/vector.gl",
    "./matrix": "./src/matrix.gl"
  }
}
```

### Example 2: Data Pipeline

```scheme
;; pipeline.gl
(import (std list) (map filter fold))
(import ./validators (validate-schema))
(import ./transformers (normalize clean))
(import ./aggregators (group-by summarize))

(module data-pipeline
  (export process-data))

(define process-data ((list Data) -> Report)
  (let validated (list Data) (filter validate-schema $1))
  (let normalized (list Data) (map normalize validated))
  (let cleaned (list Data) (map clean normalized))
  (let grouped (Map String (list Data)) (group-by cleaned))
  (summarize grouped))
```

### Example 3: AGI Module Organization

```scheme
;; agi-core.gl
(module agi-core
  (export learn reason plan execute))

(define learn ((list Example) -> Model)
  ...)

(define reason (Model Query -> Answer)
  ...)

(define plan (Goal Model -> Plan)
  ...)

(define execute (Plan -> Result)
  ...)
```

## Next Steps

- Read the [RFC](../../../docs/rfc/grammar-language-modules.md) for full specification
- Browse [examples](../examples/modules/)
- Check [Standard Library](../stdlib/README.md)

---

**Modules enable AGI to organize knowledge at scale.**
