# Grammar Language for VS Code

Official VS Code extension for Grammar Language - the programming language for AGI self-evolution with O(1) type checking.

## Features

### 🎯 Real-time Type Checking

- **O(1) type checking** - Instant feedback as you type
- **Inline diagnostics** - Errors and warnings shown directly in the editor
- **Cross-module type checking** - Validates types across file boundaries

![Type Checking](images/type-checking.gif)

### 🔍 Intelligent Code Navigation

- **Go to Definition** - Jump to function and type definitions (F12)
- **Find References** - Find all usages of a symbol (Shift+F12)
- **Cross-file navigation** - Navigate across modules seamlessly

![Go to Definition](images/goto-definition.gif)

### ✨ Smart Autocomplete

- **Built-in functions** - All 24 built-in primitives
- **User-defined functions** - Your custom functions
- **Type signatures** - See function types inline
- **Import suggestions** - Autocomplete module imports

![Autocomplete](images/autocomplete.gif)

### 📚 Hover Documentation

- **Type information** - See types on hover
- **Function signatures** - View parameter and return types
- **Inline documentation** - Quick reference without leaving the editor

![Hover](images/hover.gif)

### 🎨 Syntax Highlighting

- **S-expression aware** - Properly highlights nested structures
- **Type highlighting** - Distinct colors for types
- **Parameter highlighting** - `$1`, `$2`, etc. clearly visible
- **Comment support** - Semicolon-style comments

## Installation

### From Marketplace

1. Open VS Code
2. Press `Ctrl+P` / `Cmd+P`
3. Type: `ext install agi-recursive.grammar-language-vscode`
4. Press Enter

### From Source

```bash
cd src/grammar-lang/vscode-extension
npm install
npm run compile
```

Then press F5 to launch the extension in debug mode.

## Usage

### Creating a Grammar Language File

1. Create a file with `.gl` extension: `math-utils.gl`
2. Start typing - autocomplete and type checking work automatically

```scheme
;; Define a module
(module math-utils
  (export add multiply))

(define add (integer integer -> integer)
  (+ $1 $2))

(define multiply (integer integer -> integer)
  (* $1 $2))
```

### Importing Modules

```scheme
(import math-utils (add multiply))
(import (std list) (map filter fold))

(define sum ((list integer) -> integer)
  (fold add 0 $1))
```

### Type Checking

The extension automatically type checks your code as you type. Errors are shown inline:

```scheme
(define bad (integer -> integer)
  true)  ;; ❌ Error: expected integer, got boolean
```

### Navigation

- **F12**: Go to definition
- **Shift+F12**: Find all references
- **Alt+F12**: Peek definition
- **Ctrl+Space**: Trigger autocomplete
- **Hover**: Show type information

## Configuration

Configure the extension via VS Code settings:

```json
{
  // Enable/disable type checking
  "grammarLanguage.typeCheck": true,

  // LSP server trace level (for debugging)
  "grammarLanguage.trace.server": "off"
}
```

## Language Features

### Supported

✅ **Diagnostics** - Type errors, parse errors
✅ **Go to Definition** - Functions, types, modules
✅ **Autocomplete** - Functions, types, keywords
✅ **Hover** - Type information, documentation
✅ **Syntax Highlighting** - Full S-expression support
✅ **Bracket Matching** - Parentheses, square brackets
✅ **Auto-closing** - Brackets and quotes

### Coming Soon

🔜 **Rename Refactoring** - Safely rename symbols
🔜 **Format Document** - Auto-format S-expressions
🔜 **Code Actions** - Quick fixes and refactorings
🔜 **Signature Help** - Parameter hints while typing
🔜 **Document Symbols** - Outline view

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `F12` | Go to Definition |
| `Shift+F12` | Find References |
| `Alt+F12` | Peek Definition |
| `Ctrl+Space` | Trigger Autocomplete |
| `Ctrl+.` | Quick Fix |
| `F2` | Rename Symbol (coming soon) |

## Examples

### Example 1: Vector Math Module

```scheme
(module vector-math
  (export add-vectors scale-vector))

(define add-vectors ((list integer) (list integer) -> (list integer))
  (map2 + $1 $2))

(define scale-vector (integer (list integer) -> (list integer))
  (map (lambda ((x integer)) (* $1 x)) $2))
```

### Example 2: Using Stdlib

```scheme
(import (std list) (map filter fold))
(import (std math) (abs max min))

(define process-numbers ((list integer) -> integer)
  (let positive (list integer) (filter (lambda ((x integer)) (> x 0)) $1))
  (let absolute (list integer) (map abs positive))
  (fold max 0 absolute))
```

### Example 3: Type Definitions

```scheme
(type User
  (record
    (id integer)
    (name string)
    (email string)))

(type Result
  (enum
    (Ok integer)
    (Error string)))
```

## Troubleshooting

### Extension Not Working

1. Check Output panel: `View` → `Output` → `Grammar Language Server`
2. Restart the language server: `Ctrl+Shift+P` → `Grammar Language: Restart Server`
3. Check file extension is `.gl`

### Type Checking Issues

1. Ensure `grammarLanguage.typeCheck` is enabled
2. Save the file to trigger type checking
3. Check for parse errors (shown as diagnostics)

### No Autocomplete

1. Trigger manually with `Ctrl+Space`
2. Ensure cursor is in valid position (after `(`)
3. Check extension is activated (status bar shows "Grammar Language")

## Performance

- **Type Checking:** O(1) per expression - instant even on large files
- **Module Resolution:** O(1) lookup - no filesystem overhead
- **Autocomplete:** <10ms response time
- **Go to Definition:** <5ms lookup

**Handles 10,000+ LOC files without lag.**

## Contributing

Found a bug or want to contribute?

- **Issues:** https://github.com/thiagobutignon/chomsky/issues
- **Source:** https://github.com/thiagobutignon/chomsky/tree/main/src/grammar-lang

## Resources

- **Documentation:** [Getting Started](../docs/GETTING_STARTED.md)
- **Module Guide:** [Modules](../docs/MODULES.md)
- **RFC:** [Grammar Language Specification](../../docs/rfc/grammar-language.md)
- **Examples:** [Examples Directory](../examples/)

## License

MIT

---

**Grammar Language scales where TypeScript can't.**

*Built for AGI self-evolution.*
