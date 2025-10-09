# Module System Examples

Examples demonstrating Grammar Language module system.

## Files

- **`math-utils.gl`** - Vector math module
  - Exports: `add-vectors`, `scale-vector`, `dot-product`
  - Private helper: `map2`

- **`main.gl`** - Main module that uses math-utils
  - Imports and uses vector operations
  - Demonstrates module imports

- **`gl.json`** - Package manifest
  - Defines package name, version, exports

## Running

### Using GLC

```bash
# Type check
glc main.gl --check

# Compile with dependencies
glc main.gl --bundle -o dist/bundle.js

# Compile and run
glc main.gl --run
```

### Using GLPM

```bash
# Initialize package
glpm init

# Install dependencies (if any)
glpm install
```

## Expected Output

```
v1 + v2 = [5, 7, 9]
2 * v1 = [2, 4, 6]
v1 · v2 = 32
```

## Module Resolution

1. `main.gl` imports `math-utils`
2. Compiler finds `math-utils.gl` in same directory
3. Builds dependency graph: `math-utils → main`
4. Compiles in order: `math-utils` first, then `main`
5. Type checks cross-module calls: O(1) per expression

## Key Concepts

### Export Declaration

```scheme
(module math-utils
  (export add-vectors scale-vector dot-product))
```

Only exported functions are visible to other modules.

### Import Declaration

```scheme
(import math-utils (add-vectors scale-vector dot-product))
```

Explicitly list what to import - no wildcards.

### Private Functions

```scheme
(define map2 ...)  ;; Not in export list = private
```

Private functions are only accessible within the module.

## Extending

Try adding new modules:

```scheme
;; matrix.gl
(module matrix
  (export matrix-multiply transpose))

;; main.gl
(import matrix (matrix-multiply))
```

## Learn More

- [Module Guide](../../docs/MODULES.md)
- [RFC](../../../../docs/rfc/grammar-language-modules.md)
- [Standard Library](../../stdlib/README.md)
