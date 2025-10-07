# Fiat Lux 🌟

**Let There Be Light** - A Universal Grammar Engine for Structured Data

[![GitHub](https://img.shields.io/github/license/thiagobutignon/fiat-lux)](LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg)](https://www.typescriptlang.org/)
[![Claude Code](https://img.shields.io/badge/Built%20with-Claude%20Code-blueviolet.svg)](https://claude.com/claude-code)

## Overview

Fiat Lux is a generic, configurable grammar engine that validates and auto-repairs structured data based on customizable grammatical rules. Built with **Clean Architecture** principles following the `src/[feature]/[use-cases]` pattern.

## Architecture

The project follows Clean Architecture with clear separation of concerns:

```
src/
├── grammar-engine/              # Grammar validation and repair
│   ├── domain/
│   │   ├── entities/           # Types and predefined grammars
│   │   └── use-cases/          # GrammarEngine business logic
│   ├── data/
│   │   ├── protocols/          # Interface definitions
│   │   └── use-cases/          # Implementations (Cache)
│   └── presentation/           # Public API, factories, utilities
│
├── similarity-algorithms/       # String similarity calculations
│   ├── domain/
│   │   └── use-cases/          # Levenshtein, Jaro-Winkler, Hybrid
│   └── presentation/           # Public API
│
└── shared/                      # Shared utilities and types
    ├── types/
    └── utils/
```

## Core Principles

- **Grammar as Data**: Rules are declarative and configurable
- **Multiple Algorithms**: Pluggable similarity and repair strategies
- **Explainability**: Every decision is traceable and reportable
- **Performance**: Caching and optimization for large-scale processing
- **Type Safety**: Full TypeScript support with generics
- **Clean Architecture**: Feature-based organization with clear boundaries

## Features

### 🎯 Generic & Configurable

- Instantiate grammar engines with custom configurations
- Define roles, allowed values, and validation rules
- Support for required/optional fields and arrays
- Custom validators and structural rules

### 🔍 Multiple Similarity Algorithms

- **Levenshtein Distance**: Edit distance for typo detection
- **Jaro-Winkler**: Better for typos at the beginning of strings
- **Hybrid**: Weighted combination (60% Levenshtein + 40% Jaro-Winkler)

### ⚡ Performance Optimization

- Similarity calculation caching with hit rate tracking
- Configurable cache management
- Performance metadata in processing results

### 🔧 Advanced Auto-Repair

- Configurable similarity thresholds
- Multiple repair suggestions with confidence scores
- Alternative suggestions ranked by similarity

## Installation

```bash
npm install fiat-lux
```

## Quick Start

### Basic Usage

```typescript
import { makeGrammarEngine, CLEAN_ARCHITECTURE_GRAMMAR } from 'fiat-lux'

// Create engine with predefined grammar
const engine = makeGrammarEngine(CLEAN_ARCHITECTURE_GRAMMAR)

// Process data
const result = engine.process({
  Subject: "DbAddAccount",
  Verb: "ad", // typo - will be auto-repaired to "add"
  Object: "Account.Params",
  Context: "Controller"
})

console.log(result)
```

### Custom Grammar

```typescript
import { makeGrammarEngine, GrammarConfig, SimilarityAlgorithm } from 'fiat-lux'

const myGrammar: GrammarConfig = {
  roles: {
    Action: {
      values: ["create", "read", "update", "delete"],
      required: true
    },
    Resource: {
      values: ["user", "post", "comment"],
      required: true
    }
  },
  options: {
    similarityThreshold: 0.7,
    similarityAlgorithm: SimilarityAlgorithm.HYBRID
  }
}

const engine = makeGrammarEngine(myGrammar)
```

## API Documentation

### Main Exports

```typescript
import {
  // Grammar Engine
  makeGrammarEngine,
  GrammarEngine,

  // Types
  GrammarConfig,
  ProcessingResult,
  ValidationError,
  RepairOperation,

  // Enums
  SimilarityAlgorithm,
  Severity,

  // Predefined Grammars
  CLEAN_ARCHITECTURE_GRAMMAR,
  HTTP_API_GRAMMAR,

  // Utilities
  formatResult,

  // Similarity Algorithms
  levenshteinSimilarity,
  jaroWinklerSimilarity,
  hybridSimilarity
} from 'fiat-lux'
```

### GrammarEngine Methods

- **`process(sentence: T): ProcessingResult<T>`** - Validate and repair
- **`validate(sentence: T)`** - Validate only
- **`repair(sentence: T)`** - Repair only
- **`getCacheStats()`** - Get cache statistics
- **`clearCache()`** - Clear similarity cache
- **`setOptions(options)`** - Update configuration

## Running Demos

```bash
npm run demo
```

Output includes:
1. Clean Architecture validation with invalid tokens
2. Multiple errors with hybrid algorithm
3. Valid sentence (no repairs needed)
4. HTTP API grammar example
5. Algorithm comparison analysis
6. Cache performance testing

## Project Structure

### Feature Organization

Each feature follows Clean Architecture layers:

- **domain/entities**: Core types and interfaces
- **domain/use-cases**: Business logic (no external dependencies)
- **data/protocols**: Interface definitions for external dependencies
- **data/use-cases**: Implementations
- **presentation**: Public API, factories, and utilities

### Benefits

- **Dependency Rule**: Dependencies point inward
- **Testability**: Easy to mock external dependencies
- **Maintainability**: Clear separation of concerns
- **Scalability**: Easy to add new features
- **Flexibility**: Swap implementations without changing business logic

## Development

```bash
# Install dependencies
npm install

# Run demo
npm run demo

# Run tests (ultra-fast, <5ms)
npm test

# Build
npm run build
```

## Testing

The project includes a **custom lightweight test framework** designed for speed. All 45+ unit tests run in under 5ms!

```bash
$ npm test

🧪 Running Tests...

📦 Levenshtein Distance
  ✅ should return 0 for identical strings (0.11ms)
  ✅ should calculate single character difference (0.01ms)
  ...

📦 GrammarEngine - Validation
  ✅ should validate valid sentences (0.11ms)
  ✅ should detect invalid values (0.10ms)
  ...

════════════════════════════════════════════════════════════════════════════════
Test Summary
════════════════════════════════════════════════════════════════════════════════
Total:   45
✅ Passed: 45
❌ Failed: 0
⏭️  Duration: 2.63ms
════════════════════════════════════════════════════════════════════════════════

✅ All tests passed!
```

### Test Coverage

- **Similarity Algorithms**: 26 tests
  - Levenshtein distance and similarity
  - Jaro-Winkler similarity
  - Hybrid algorithm

- **Grammar Engine**: 19 tests
  - Validation logic
  - Auto-repair functionality
  - Cache performance
  - Configuration options

### Why Custom Test Framework?

Instead of Jest or Mocha (which take 1-2 seconds to start), our custom framework:
- ✅ Runs in **<5ms** (400x faster startup)
- ✅ Zero dependencies
- ✅ Simple API (`describe`, `it`, `expect`)
- ✅ Perfect for TDD workflow

## Documentation

- **[CLAUDE.md](CLAUDE.md)** - AI coding standards
- **[Grammar Analysis Index](GRAMMAR_ANALYSIS_INDEX.md)** - Overview of analyses
- **[Universal Grammar Proof](UNIVERSAL_GRAMMAR_PROOF.md)** - Theoretical foundations

## Contributing

Contributions are welcome! Please read [CLAUDE.md](CLAUDE.md) for coding standards.

## Performance

- **Processing time**: 0.02ms - 0.50ms per validation
- **Cache hit rate**: ~99% after warm-up
- **Average iteration**: <1ms with caching

## License

MIT License - see [LICENSE](LICENSE) file for details

## Credits

Built with [Claude Code](https://claude.com/claude-code) by Anthropic

---

**Fiat Lux** - Let there be light in your structured data! 🌟
