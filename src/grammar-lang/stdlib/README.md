# Grammar Language Standard Library

Core functions and types available to all Grammar Language programs.

## Modules

### core.gl

Basic functions for common operations.

#### Boolean Operations

```scheme
(not boolean -> boolean)
(and boolean boolean -> boolean)
(or boolean boolean -> boolean)
```

#### Integer Operations

```scheme
(abs integer -> integer)          ;; Absolute value
(max integer integer -> integer)  ;; Maximum of two numbers
(min integer integer -> integer)  ;; Minimum of two numbers
(even? integer -> boolean)        ;; Check if even
(odd? integer -> boolean)         ;; Check if odd
```

#### List Operations

```scheme
(length (list a) -> integer)
(map (a -> b) (list a) -> (list b))
(filter (a -> boolean) (list a) -> (list a))
(fold (b a -> b) b (list a) -> b)
(reverse (list a) -> (list a))
(append (list a) (list a) -> (list a))
(take integer (list a) -> (list a))
(drop integer (list a) -> (list a))
```

#### String Operations

```scheme
(concat string string -> string)
(uppercase string -> string)
(lowercase string -> string)
```

#### Option Type

```scheme
(type Option (enum (Some a) None))

(is-some (Option a) -> boolean)
(is-none (Option a) -> boolean)
(unwrap (Option a) -> a)
(unwrap-or (Option a) a -> a)
```

#### Result Type

```scheme
(type Result (enum (Ok a) (Err e)))

(is-ok (Result a e) -> boolean)
(is-err (Result a e) -> boolean)
(unwrap-result (Result a e) -> a)
```

#### IO Operations

```scheme
(print string -> unit)
(println string -> unit)
(read-line unit -> string)
```

#### Utility Functions

```scheme
(identity a -> a)
(const a b -> a)
(compose (b -> c) (a -> b) -> (a -> c))
(flip (a b -> c) -> (b a -> c))
```

#### Math Functions

```scheme
(square integer -> integer)
(cube integer -> integer)
(pow integer integer -> integer)
(factorial integer -> integer)
(fibonacci integer -> integer)
(range integer integer -> (list integer))
(sum (list integer) -> integer)
(product (list integer) -> integer)
```

## Built-in Functions

Low-level primitives (implemented in TypeScript):

### Arithmetic
- `+`, `-`, `*`, `/`, `%`

### Comparison
- `=`, `<`, `<=`, `>`, `>=`

### String
- `concat`, `builtin-uppercase`, `builtin-lowercase`
- `string-length`, `substring`

### List
- `empty-list`, `cons`, `head`, `tail`, `empty?`

### IO
- `builtin-print`, `builtin-read-line`

### Debug
- `panic`, `debug`

## Examples

### Working with Lists

```scheme
;; Filter even numbers
(define evens ((list integer) -> (list integer))
  (filter even? $1))

;; Example: (evens [1 2 3 4 5]) => [2 4]

;; Sum of squares
(define sum-of-squares ((list integer) -> integer)
  (sum (map square $1)))

;; Example: (sum-of-squares [1 2 3]) => 14
```

### Error Handling with Option

```scheme
;; Safe division
(define safe-div (integer integer -> (Option integer))
  (if (= $2 0)
    None
    (Some (/ $1 $2))))

;; Example: (safe-div 10 2) => (Some 5)
;; Example: (safe-div 10 0) => None
```

### Functional Composition

```scheme
;; Compose two functions
(define double-then-square (integer -> integer)
  (compose square double))

;; Example: (double-then-square 3) => 36
```

### IO Examples

```scheme
;; Hello world
(define greet (string -> unit)
  (println (concat "Hello, " $1)))

;; Interactive
(define ask-name (unit -> unit)
  (let name string (read-line))
  (greet name))
```

## Usage

Import the standard library in your programs:

```scheme
(module my-program
  (import core [map filter fold])

  ;; Your code here
  )
```

## Performance

All functions are O(n) or better:
- List operations: O(n) where n = list length
- Math functions: O(1) or O(n) for recursive ones
- String operations: O(n) where n = string length

No hidden allocations or copying (except where semantically required).
