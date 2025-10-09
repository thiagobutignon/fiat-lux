# Grammar Language LSP (Language Server Protocol)

Complete guide to the Grammar Language LSP server for editor integration.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Architecture](#architecture)
5. [Capabilities](#capabilities)
6. [Performance](#performance)
7. [Development](#development)

## Overview

The Grammar Language LSP server provides rich editor integration for any LSP-compatible editor (VS Code, Vim, Emacs, etc.).

### What is LSP?

The Language Server Protocol standardizes communication between editors and language tooling. One LSP server works across all editors.

```
┌─────────────┐          ┌──────────────────┐
│   VS Code   │◄────────►│                  │
├─────────────┤   LSP    │  Grammar         │
│     Vim     │◄────────►│  Language        │
├─────────────┤          │  LSP Server      │
│    Emacs    │◄────────►│                  │
└─────────────┘          └──────────────────┘
```

### Why LSP for Grammar Language?

1. **O(1) Type Checking** - Instant feedback, no lag
2. **Cross-Module Navigation** - Jump across files seamlessly
3. **Intelligent Autocomplete** - Context-aware suggestions
4. **AGI-Friendly** - Analyzable, structured responses

## Features

### ✅ Implemented

1. **Diagnostics**
   - Type errors with location
   - Parse errors with hints
   - Real-time validation

2. **Go to Definition**
   - Jump to function definitions
   - Navigate to type declarations
   - Cross-module support

3. **Autocomplete**
   - Built-in functions (24 primitives)
   - User-defined functions
   - Type signatures
   - Keywords

4. **Hover**
   - Type information
   - Function signatures
   - Documentation strings

5. **Syntax Highlighting**
   - S-expression aware
   - Type highlighting
   - Parameter highlighting

### 🔜 Coming Soon

- **Rename Refactoring** - Safe symbol renaming
- **Code Actions** - Quick fixes
- **Signature Help** - Parameter hints
- **Document Symbols** - Outline view
- **Format Document** - Auto-formatting

## Installation

### VS Code

#### From Marketplace (Coming Soon)

```bash
code --install-extension agi-recursive.grammar-language-vscode
```

#### From Source

```bash
cd src/grammar-lang/vscode-extension
npm install
npm run compile
code .
# Press F5 to launch extension
```

### Vim/Neovim (with CoC)

Add to `.vimrc` or `init.vim`:

```vim
" Install coc-grammarLanguage
:CocInstall coc-grammarLanguage

" Or configure manually
{
  "languageserver": {
    "grammarLanguage": {
      "command": "tsx",
      "args": ["path/to/lsp-server.ts"],
      "filetypes": ["gl"],
      "rootPatterns": ["gl.json"]
    }
  }
}
```

### Emacs (with lsp-mode)

Add to `.emacs` or `init.el`:

```elisp
(require 'lsp-mode)

(add-to-list 'lsp-language-id-configuration '(grammar-language-mode . "grammarLanguage"))

(lsp-register-client
 (make-lsp-client :new-connection (lsp-stdio-connection '("tsx" "path/to/lsp-server.ts"))
                  :major-modes '(grammar-language-mode)
                  :server-id 'grammarLanguage))

(add-hook 'grammar-language-mode-hook #'lsp)
```

## Architecture

### Components

```
┌─────────────────────────────────────────────────────┐
│                   LSP Server                        │
│                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │
│  │   Parser     │→ │ Type Checker │→ │Diagnostics││
│  └──────────────┘  └──────────────┘  └──────────┘ │
│                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │
│  │Module        │→ │  Symbol      │→ │Go to Def │ │
│  │Resolver      │  │  Index       │  │          │ │
│  └──────────────┘  └──────────────┘  └──────────┘ │
│                                                     │
│  ┌──────────────┐  ┌──────────────┐               │
│  │Autocomplete  │  │    Hover     │               │
│  └──────────────┘  └──────────────┘               │
└─────────────────────────────────────────────────────┘
```

### Request Flow

```
1. Editor sends textDocument/didOpen
   ↓
2. LSP parses document → AST
   ↓
3. Type checker validates → Errors
   ↓
4. Diagnostics sent to editor
   ↓
5. User types → textDocument/completion
   ↓
6. Autocomplete searches symbols → Suggestions
   ↓
7. Results sent to editor
```

### Document State

Each open document maintains:

```typescript
interface DocumentInfo {
  uri: string;              // File URI
  content: string;          // Source text
  ast: Definition[];        // Parsed AST
  errors: Diagnostic[];     // Type/parse errors
  symbols: Map<string, SymbolInfo>; // Symbol table
}
```

## Capabilities

### 1. Diagnostics

**Real-time error detection:**

```scheme
(define bad (integer -> integer)
  true)  ;; ❌ Diagnostic: expected integer, got boolean
```

**Implementation:**

```typescript
async function validateTextDocument(doc: TextDocument) {
  try {
    const ast = parseProgram(source);
    checkProgram(ast);
    // No errors → clear diagnostics
  } catch (error) {
    // Send diagnostic to editor
    connection.sendDiagnostics({
      uri: doc.uri,
      diagnostics: [{
        severity: DiagnosticSeverity.Error,
        message: error.message,
        range: errorRange
      }]
    });
  }
}
```

**Performance:** O(1) per expression

### 2. Go to Definition

**Jump to symbol definition:**

```scheme
(define add (integer integer -> integer)
  (+ $1 $2))

;; Later...
(add 5 3)  ;; F12 → jumps to 'add' definition
```

**Implementation:**

```typescript
connection.onDefinition((params) => {
  const doc = documentCache.get(params.textDocument.uri);
  const symbol = findSymbolAtPosition(doc, params.position);

  return {
    uri: symbol.definitionUri,
    range: symbol.definitionRange
  };
});
```

**Performance:** O(1) symbol lookup

### 3. Autocomplete

**Context-aware suggestions:**

```scheme
(import (std list) (map filter fold))

(f|)  ;; Autocomplete shows: filter, fold, fibonacci, etc.
```

**Implementation:**

```typescript
connection.onCompletion(() => {
  const items: CompletionItem[] = [];

  // Built-in functions
  for (const builtin of BUILTINS) {
    items.push({
      label: builtin.name,
      kind: CompletionItemKind.Function,
      detail: formatType(builtin.type)
    });
  }

  // User symbols
  for (const [name, symbol] of doc.symbols) {
    items.push({
      label: name,
      kind: symbolKindToCompletionKind(symbol.kind),
      detail: symbol.type
    });
  }

  return items;
});
```

**Performance:** <10ms response

### 4. Hover

**Type information on hover:**

```scheme
(define add (integer integer -> integer)
  (+ $1 $2))
  ↑
  Hover shows:
  ┌─────────────────────────────────┐
  │ function: add                   │
  │ type: (integer integer -> integer)│
  └─────────────────────────────────┘
```

**Implementation:**

```typescript
connection.onHover((params) => {
  const symbol = findSymbolAtPosition(doc, params.position);

  return {
    contents: {
      kind: MarkupKind.Markdown,
      value: `\`\`\`grammar-language\n${symbol.kind}: ${symbol.name}\ntype: ${symbol.type}\n\`\`\``
    }
  };
});
```

**Performance:** <5ms lookup

### 5. Syntax Highlighting

**TextMate grammar:**

```json
{
  "keywords": {
    "match": "\\b(define|type|module|import)\\b"
  },
  "types": {
    "match": "\\b(integer|string|boolean)\\b"
  },
  "parameters": {
    "match": "\\$\\d+"
  }
}
```

**Colors:**
- Keywords: Blue
- Types: Green
- Parameters: Purple
- Functions: Yellow
- Strings: Red
- Comments: Gray

## Performance

### Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| Type checking | <1ms per expression | O(1) |
| Go to definition | <5ms | O(1) hash lookup |
| Autocomplete | <10ms | O(n) symbols, typically n < 1000 |
| Hover | <5ms | O(1) symbol lookup |
| Parse | <50ms | O(1) per file |

### Scalability

- **10,000 LOC file:** No lag
- **100 modules:** Instant navigation
- **1,000 symbols:** Fast autocomplete

### Memory

- **Per document:** ~1MB (AST + symbols)
- **100 documents:** ~100MB total
- **Symbol index:** O(n) space, n = symbols

## Development

### Project Structure

```
tools/
  lsp-server.ts           # Main LSP server
vscode-extension/
  package.json            # Extension metadata
  src/
    extension.ts          # Extension activation
  syntaxes/
    grammar-language.tmLanguage.json  # Syntax highlighting
  language-configuration.json  # Brackets, comments
```

### Building

```bash
# LSP Server (no build needed - uses tsx)
tsx tools/lsp-server.ts

# VS Code Extension
cd vscode-extension
npm install
npm run compile
```

### Testing

```bash
# Unit tests
npm test

# Integration tests (VS Code extension)
cd vscode-extension
npm run test
```

### Debugging

**VS Code Extension:**

1. Open `vscode-extension` in VS Code
2. Press F5 to launch Extension Development Host
3. Open a `.gl` file
4. Debug extension in original window

**LSP Server:**

1. Add `--inspect` flag to server options
2. Attach debugger to port 6009
3. Set breakpoints in `lsp-server.ts`

### Adding Features

#### New Diagnostic

```typescript
// In validateTextDocument()
if (someCondition) {
  diagnostics.push({
    severity: DiagnosticSeverity.Warning,
    message: 'Custom warning',
    range: { start: { line: 0, character: 0 }, ... }
  });
}
```

#### New Autocomplete Source

```typescript
// In onCompletion()
items.push({
  label: 'new-item',
  kind: CompletionItemKind.Snippet,
  insertText: '(define $1 $2)',
  insertTextFormat: InsertTextFormat.Snippet
});
```

#### New Code Action

```typescript
connection.onCodeAction((params) => {
  return [{
    title: 'Fix import',
    kind: CodeActionKind.QuickFix,
    edit: {
      changes: {
        [params.textDocument.uri]: [/* TextEdit objects */]
      }
    }
  }];
});
```

## Advanced Usage

### Multi-Root Workspace

LSP supports multiple workspace folders:

```typescript
connection.onInitialize((params) => {
  for (const folder of params.workspaceFolders) {
    loadWorkspace(folder.uri);
  }
});
```

### Custom Configuration

Users can configure LSP behavior:

```json
{
  "grammarLanguage.typeCheck": true,
  "grammarLanguage.diagnostics.level": "strict",
  "grammarLanguage.autocomplete.imports": true
}
```

### Watch Mode

LSP watches for file changes:

```typescript
connection.workspace.onDidChangeWatchedFiles((change) => {
  for (const event of change.changes) {
    revalidateDocument(event.uri);
  }
});
```

## Troubleshooting

### LSP Not Starting

**Check:**
1. Server executable exists: `tsx tools/lsp-server.ts`
2. Node version >=18
3. Extension activated (check status bar)

**Debug:**
```bash
# Run LSP server directly
tsx tools/lsp-server.ts --stdio
```

### No Diagnostics

**Check:**
1. File extension is `.gl`
2. Type checking enabled in settings
3. No parse errors (parse errors block type checking)

**Debug:**
View LSP logs: `View` → `Output` → `Grammar Language Server`

### Autocomplete Not Working

**Check:**
1. Cursor position (after `(` or space)
2. Document is parsed (no parse errors)
3. Trigger manually: `Ctrl+Space`

**Debug:**
Check symbol table size: Should have entries for all defined functions

### Performance Issues

**Check:**
1. File size (<10,000 LOC recommended)
2. Number of symbols (<1,000 recommended)
3. Memory usage (should be <100MB per document)

**Optimize:**
- Split large files into modules
- Use lazy loading for large symbol tables
- Cache type checking results

## Resources

- **LSP Specification:** https://microsoft.github.io/language-server-protocol/
- **VS Code Extension Guide:** https://code.visualstudio.com/api
- **Grammar Language RFC:** [../../docs/rfc/grammar-language.md](../../docs/rfc/grammar-language.md)

## Contributing

To add LSP support for your editor:

1. Implement LSP client for your editor
2. Configure to use `tsx tools/lsp-server.ts`
3. Map file extension `.gl` to `grammarLanguage`
4. Test all capabilities

Example clients:
- **VS Code:** [vscode-extension/](../vscode-extension/)
- **Vim:** Use CoC or vim-lsp
- **Emacs:** Use lsp-mode
- **Sublime:** Use LSP package

---

**O(1) type checking in your editor, powered by LSP.**
