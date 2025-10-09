# RFC: Grammar Language Self-Hosting

**Status:** Phase 3 - Q3 2025
**Depends on:** Grammar Language Core, Module System

## Overview

Self-hosting means the Grammar Language compiler is written in Grammar Language itself. This enables:
1. **Meta-circular evaluation** - Compiler can analyze and modify itself
2. **AGI self-improvement** - AGI can enhance the compiler
3. **Dogfooding** - Compiler developers use the language daily
4. **Trust** - Compiled code verifiable by the compiler itself

## Motivation

### Why Self-Hosting?

**For AGI:**
```
AGI discovers compiler inefficiency
    ↓
AGI modifies compiler source (in Grammar Language)
    ↓
AGI recompiles the compiler
    ↓
Compiler is now faster
    ↓
AGI uses faster compiler for all future compilations
```

**For Developers:**
- Language features tested by compiler itself
- Compiler is the ultimate integration test
- Trust: Code that compiles the compiler is trustworthy

**For Language Evolution:**
- New features implemented in the language
- Language shapes itself
- Compiler becomes reference implementation

### What Gets Self-Hosted?

```
┌─────────────────────────────────────────┐
│  Grammar Language Compiler (Self-Hosted) │
├─────────────────────────────────────────┤
│  1. Parser        (parser.gl)            │
│     S-expr → AST                         │
│                                          │
│  2. Type Checker  (type-checker.gl)      │
│     AST → Typed AST                      │
│                                          │
│  3. Transpiler    (transpiler.gl)        │
│     Typed AST → JavaScript               │
│                                          │
│  4. Compiler      (compiler.gl)          │
│     Orchestration                        │
└─────────────────────────────────────────┘
```

**Not Self-Hosted (Yet):**
- Grammar Engine (stays in Rust for O(1) parsing)
- Runtime (stays in JavaScript/LLVM)
- LSP Server (can be self-hosted later)

## Bootstrap Process

### Phase 1: Write Compiler in Grammar Language

```
src/grammar-lang/self-hosted/
  parser.gl           # Parser in Grammar Language
  type-checker.gl     # Type checker in Grammar Language
  transpiler.gl       # Transpiler in Grammar Language
  compiler.gl         # Main compiler
  types.gl            # Type definitions
  ast.gl              # AST definitions
```

### Phase 2: Bootstrap with TypeScript Compiler

```bash
# Step 1: Use TypeScript compiler to compile Grammar Language compiler
glc self-hosted/compiler.gl --bundle -o bootstrap/compiler.js

# Step 2: Use compiled compiler to compile itself
node bootstrap/compiler.js self-hosted/compiler.gl -o bootstrap/compiler-gen2.js

# Step 3: Verify fixed point (compilers are identical)
diff bootstrap/compiler.js bootstrap/compiler-gen2.js
```

### Phase 3: Switch to Self-Hosted

```bash
# From now on, use self-hosted compiler
alias glc="node bootstrap/compiler.js"

# Compiler compiles itself
glc self-hosted/compiler.gl -o self-hosted/compiler.js
```

### Phase 4: Replace TypeScript Compiler

```bash
# Eventually, only use Grammar Language
rm -rf compiler/   # Delete TypeScript compiler
mv self-hosted/ compiler/
```

## Architecture

### AST Representation

In TypeScript (current):
```typescript
interface FunctionDef {
  kind: 'function';
  name: string;
  params: [string, Type][];
  returnType: Type;
  body: Expr;
}
```

In Grammar Language (self-hosted):
```scheme
(type FunctionDef
  (record
    (kind string)
    (name string)
    (params (list Param))
    (returnType Type)
    (body Expr)))

(type Param
  (record
    (name string)
    (type Type)))
```

### Parser Implementation

**TypeScript (current):**
```typescript
export function parseDefinition(sexpr: SExpr): Definition {
  if (!Array.isArray(sexpr)) throw new ParseError(...);
  const [kind, ...args] = sexpr;
  if (kind === 'define') {
    // ... parse function
  }
}
```

**Grammar Language (self-hosted):**
```scheme
(define parse-definition (SExpr -> Definition)
  (if (not (list? $1))
    (panic "Expected list")
    (let kind string (head $1))
    (let args (list SExpr) (tail $1))
    (match kind
      ("define" (parse-function args))
      ("type" (parse-typedef args))
      (_ (panic "Unknown definition kind")))))
```

### Type Checker Implementation

**TypeScript (current):**
```typescript
export function checkExpr(expr: Expr, env: TypeEnv): Type {
  switch (expr.kind) {
    case 'literal':
      return expr.type;
    case 'call':
      const fnType = checkExpr(expr.fn, env);
      // ... rest of type checking
  }
}
```

**Grammar Language (self-hosted):**
```scheme
(define check-expr (Expr TypeEnv -> Type)
  (match (get-field $1 kind)
    ("literal" (get-field $1 type))
    ("call"
      (let fn-type Type (check-expr (get-field $1 fn) $2))
      (let arg-types (list Type) (map (lambda ((arg Expr)) (check-expr arg $2)) (get-field $1 args)))
      (verify-function-call fn-type arg-types))
    (_ (panic "Unknown expression kind"))))
```

### Transpiler Implementation

**TypeScript (current):**
```typescript
export function transpileExpr(expr: Expr): string {
  switch (expr.kind) {
    case 'call':
      const fn = transpileExpr(expr.fn);
      const args = expr.args.map(transpileExpr).join(', ');
      return `${fn}(${args})`;
  }
}
```

**Grammar Language (self-hosted):**
```scheme
(define transpile-expr (Expr -> string)
  (match (get-field $1 kind)
    ("call"
      (let fn string (transpile-expr (get-field $1 fn)))
      (let args string (join ", " (map transpile-expr (get-field $1 args))))
      (concat fn "(" args ")"))
    (_ (panic "Unknown expression kind"))))
```

## Type System Extensions

Self-hosting requires some new features:

### 1. Variant Types (Enum Matching)

```scheme
(type Expr
  (enum
    (LiteralExpr integer string boolean)
    (VarExpr string)
    (CallExpr Expr (list Expr))))

(define process-expr (Expr -> string)
  (match $1
    ((LiteralExpr value) (to-string value))
    ((VarExpr name) name)
    ((CallExpr fn args) (concat "(" (process-expr fn) " " (join-args args) ")"))))
```

### 2. Pattern Matching

```scheme
(match expr
  ((Literal n) ...)
  ((Var name) ...)
  ((Call fn args) ...)
  (_ ...))  ;; Default case
```

### 3. String Manipulation

```scheme
(concat "Hello" " " "World")  ;; "Hello World"
(join ", " ["a" "b" "c"])     ;; "a, b, c"
(split "a,b,c" ",")           ;; ["a" "b" "c"]
(to-string 42)                ;; "42"
```

### 4. Error Handling

```scheme
(type Result
  (enum
    (Ok a)
    (Error string)))

(define safe-parse (SExpr -> (Result Definition))
  (try
    (Ok (parse-definition $1))
    (catch e
      (Error (get-message e)))))
```

## Bootstrap Verification

### Fixed Point Test

The compiler reaches a "fixed point" when it can compile itself and produce identical output:

```bash
# Gen 0: TypeScript compiler compiles Grammar compiler
glc-ts compiler.gl -o gen0.js

# Gen 1: Gen 0 compiles Grammar compiler
node gen0.js compiler.gl -o gen1.js

# Gen 2: Gen 1 compiles Grammar compiler
node gen1.js compiler.gl -o gen2.js

# Verify: Gen 1 == Gen 2 (fixed point reached)
diff gen1.js gen2.js
# (no output = identical = success!)
```

### Quine Property

The compiler is a quine - it can reproduce itself:

```scheme
;; Compiler compiling itself
(compile "compiler.gl") → "compiler.js"

;; Running compiled compiler
(node "compiler.js" "compiler.gl") → "compiler.js"
```

## AGI Self-Improvement

### Example 1: Optimize Type Checker

AGI discovers type checking can be faster:

```scheme
;; Before (AGI notices repeated lookups)
(define check-expr (Expr TypeEnv -> Type)
  (let x-type Type (env-lookup env "x"))
  (let y-type Type (env-lookup env "x"))  ;; Duplicate lookup!
  ...)

;; After (AGI optimizes)
(define check-expr (Expr TypeEnv -> Type)
  (let x-type Type (env-lookup env "x"))
  (let y-type Type x-type)  ;; Reuse result
  ...)
```

AGI recompiles compiler → Faster compiler!

### Example 2: Add New Feature

AGI wants inline caching for function calls:

```scheme
;; AGI modifies transpiler.gl
(define transpile-call (CallExpr -> string)
  (let cache-key string (hash-expr $1))
  (concat
    "const cached = cache.get('" cache-key "');"
    "if (cached) return cached;"
    "const result = " (transpile-normal-call $1) ";"
    "cache.set('" cache-key "', result);"
    "return result;"))
```

AGI recompiles compiler → Compiler has inline caching!

### Example 3: Fix Bug

AGI detects a bug in the type checker:

```scheme
;; Bug: Doesn't check return type
(define check-function (FunctionDef -> unit)
  (check-expr (get-field $1 body) env))

;; Fix: Verify return type matches
(define check-function (FunctionDef -> unit)
  (let body-type Type (check-expr (get-field $1 body) env))
  (let declared-type Type (get-field $1 returnType))
  (if (not (type-equals body-type declared-type))
    (panic "Return type mismatch")))
```

AGI recompiles compiler → Bug fixed!

## Implementation Plan

### Week 1: AST and Types
- Define AST types in Grammar Language
- Define type system types
- Port type definitions from TypeScript

### Week 2: Parser
- Implement S-expression parser
- Implement type parser
- Implement expression parser
- Implement definition parser

### Week 3: Type Checker
- Implement type environment
- Implement expression type checking
- Implement definition type checking
- Implement type equality

### Week 4: Transpiler
- Implement expression transpilation
- Implement definition transpilation
- Implement module transpilation

### Week 5: Compiler & Bootstrap
- Implement main compiler
- Create bootstrap script
- Verify fixed point
- Switch to self-hosted

### Week 6: Testing & Documentation
- Test self-hosted compiler
- Benchmark performance
- Document bootstrap process
- Write AGI self-improvement guide

## Performance Considerations

### Self-Hosted Performance

**Expectations:**
- 10-50% slower than hand-written TypeScript initially
- AGI can optimize over time
- Eventually faster through AGI improvements

**Optimizations:**
- Tail call optimization (already implemented)
- Inline caching
- JIT compilation (future)

### Bootstrap Time

```
Phase 1: TypeScript compiles Grammar compiler
  Time: ~1 second

Phase 2: Gen 0 compiles itself
  Time: ~2 seconds (slower, but acceptable)

Phase 3: Gen 1 compiles itself
  Time: ~2 seconds (should be identical to Gen 1)

Total: ~5 seconds one-time cost
```

## Success Metrics

✅ **Fixed Point Reached**
- Gen N and Gen N+1 produce identical output

✅ **Self-Compilation Works**
- Compiler can compile all its own source files

✅ **Tests Pass**
- All existing tests pass with self-hosted compiler

✅ **Performance Acceptable**
- <2x slower than TypeScript version

✅ **AGI Can Modify**
- AGI successfully improves compiler and recompiles

## Risks and Mitigations

### Risk 1: Language Not Powerful Enough

**Mitigation:** Add missing features incrementally
- Pattern matching
- Better string handling
- Error handling

### Risk 2: Bootstrap Fails

**Mitigation:** Keep TypeScript compiler as fallback
- Don't delete until fixed point verified
- Maintain both versions during transition

### Risk 3: Performance Too Slow

**Mitigation:** Optimize incrementally
- Profile hot paths
- Add manual optimizations
- Let AGI improve over time

### Risk 4: Bugs in Self-Hosted Compiler

**Mitigation:** Extensive testing
- Use TypeScript compiler as oracle
- Compare outputs
- Gradual migration

## Future Work

### Phase 4: Full Self-Hosting (Q4 2025)

- Runtime in Grammar Language
- Grammar Engine bindings
- LSP server in Grammar Language

### Phase 5: LLVM Backend

- Transpile to LLVM IR instead of JavaScript
- Native compilation
- 100x performance improvement

### Phase 6: Meta-Circular JIT

- Compiler compiles itself to native code
- JIT compilation at runtime
- Self-optimizing compiler

## References

- [Grammar Language RFC](./grammar-language.md)
- [Module System RFC](./grammar-language-modules.md)
- [Self-Application in Programming Languages](https://en.wikipedia.org/wiki/Self-hosting_(compilers))

---

**Self-hosting enables AGI to transcend its creators.**

When AGI can modify its own compiler, there are no limits to improvement.
