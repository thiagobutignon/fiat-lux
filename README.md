# Fiat Lux 🌟

**Let There Be Light** - A Universal Grammar Engine for Structured Data

[![GitHub](https://img.shields.io/github/license/thiagobutignon/fiat-lux)](LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg)](https://www.typescriptlang.org/)
[![Claude Code](https://img.shields.io/badge/Built%20with-Claude%20Code-blueviolet.svg)](https://claude.com/claude-code)

## Overview

Fiat Lux is a generic, configurable grammar engine that validates and auto-repairs structured data based on customizable grammatical rules. It's designed to work with any domain: code architecture, natural language, configuration files, or any structured data that follows grammatical patterns.

## Core Principles

- **Grammar as Data**: Rules are declarative and configurable
- **Multiple Algorithms**: Pluggable similarity and repair strategies
- **Explainability**: Every decision is traceable and reportable
- **Performance**: Caching and optimization for large-scale processing
- **Type Safety**: Full TypeScript support with generics

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
- Pluggable algorithm system for easy extension

### ⚡ Performance Optimization

- Similarity calculation caching with hit rate tracking
- Configurable cache management
- Performance metadata in processing results
- Optimized O(m×n) algorithms

### 🔧 Advanced Auto-Repair

- Configurable similarity thresholds
- Multiple repair suggestions with confidence scores
- Alternative suggestions ranked by similarity
- Detailed repair reports with algorithm information

### 📊 Rich Type System

- Full TypeScript generics support
- Comprehensive interfaces for all data structures
- Severity levels (ERROR, WARNING, INFO)
- Detailed metadata tracking

## Installation

```bash
npm install fiat-lux
```

Or use directly with TypeScript:

```bash
git clone https://github.com/thiagobutignon/fiat-lux.git
cd fiat-lux
npm install
npx ts-node fiat-lux.ts
```

## Quick Start

### Basic Usage

```typescript
import { GrammarEngine, GrammarConfig } from './fiat-lux'

// Define your grammar
const myGrammar: GrammarConfig = {
  roles: {
    Subject: {
      values: ["DbAddAccount", "RemoteAddAccount"],
      required: true
    },
    Verb: {
      values: ["add", "delete", "update"],
      required: true
    },
    Object: {
      values: ["Account.Params", "Survey.Params"],
      required: false
    }
  }
}

// Create engine
const engine = new GrammarEngine(myGrammar)

// Process data
const result = engine.process({
  Subject: "DbAddAccount",
  Verb: "ad", // typo - will be auto-repaired to "add"
  Object: "AccountParams" // will be suggested "Account.Params"
})

console.log(result)
```

### Using Predefined Grammars

```typescript
import { GrammarEngine, CLEAN_ARCHITECTURE_GRAMMAR } from './fiat-lux'

const engine = new GrammarEngine(CLEAN_ARCHITECTURE_GRAMMAR)

const result = engine.process({
  Subject: "DbAddAccount",
  Verb: "add",
  Object: "Account.Params",
  Adverbs: ["Hasher", "Repository"],
  Context: "Controller"
})
```

## Examples

### Clean Architecture Validation

```typescript
const engine = new GrammarEngine(CLEAN_ARCHITECTURE_GRAMMAR)

const result = engine.process({
  Subject: "RemoteLoadSurvey",
  Verb: "lod", // typo → will suggest "load"
  Object: "UserParams", // typo → will suggest "User.Params"
  Adverbs: ["Hash", "Validatr"], // typos
  Context: "Facto" // typo → will suggest "MainFactory"
})

// Result includes:
// - Original input
// - Validation errors
// - Auto-repaired output
// - Detailed repair operations with confidence scores
// - Alternative suggestions
// - Performance metadata
```

### HTTP API Grammar

```typescript
import { GrammarEngine, HTTP_API_GRAMMAR } from './fiat-lux'

const engine = new GrammarEngine(HTTP_API_GRAMMAR)

const result = engine.process({
  Method: "PSOT", // typo → will suggest "POST"
  Resource: "/user", // typo → will suggest "/users"
  Status: "201",
  Handler: ["Controller", "Middleware"]
})
```

### Custom Grammar with Structural Rules

```typescript
const customGrammar: GrammarConfig = {
  roles: {
    Action: {
      values: ["authenticate", "authorize"],
      required: true
    },
    Target: {
      values: ["User.Credentials", "Auth.Token"],
      required: true
    }
  },
  structuralRules: [
    {
      name: "AuthenticationRequiresCredentials",
      validate: (s) => {
        if (s.Action === "authenticate" && !s.Target?.includes("Credentials")) {
          return false
        }
        return true
      },
      message: "Authentication requires User.Credentials"
    }
  ]
}
```

## API Documentation

### `GrammarEngine<T>`

The main class for grammar validation and repair.

#### Constructor

```typescript
new GrammarEngine<T>(config: GrammarConfig)
```

#### Methods

- **`process(sentence: T): ProcessingResult<T>`** - Validate and repair a sentence
- **`validate(sentence: T): { errors, structuralErrors }`** - Validate only
- **`repair(sentence: T): { repaired, repairs }`** - Repair only
- **`setOptions(options: Partial<GrammarOptions>)`** - Update configuration
- **`getCacheStats()`** - Get cache performance statistics
- **`clearCache()`** - Clear similarity cache

### Configuration Options

```typescript
interface GrammarOptions {
  similarityThreshold?: number      // 0-1, default: 0.6
  similarityAlgorithm?: SimilarityAlgorithm  // default: HYBRID
  enableCache?: boolean             // default: true
  autoRepair?: boolean              // default: true
  maxSuggestions?: number           // default: 3
  caseSensitive?: boolean           // default: false
}
```

### Similarity Algorithms

- `SimilarityAlgorithm.LEVENSHTEIN` - Edit distance
- `SimilarityAlgorithm.JARO_WINKLER` - Good for prefix typos
- `SimilarityAlgorithm.HYBRID` - Weighted combination (recommended)

## Running Demos

```bash
npx ts-node fiat-lux.ts
```

The demo includes:
1. Clean Architecture validation with invalid tokens
2. Multiple errors with hybrid algorithm
3. Valid sentence (no repairs needed)
4. HTTP API grammar example
5. Algorithm comparison analysis
6. Cache performance testing

## Documentation

Comprehensive documentation is available in the repository:

- **[Grammar Analysis Index](GRAMMAR_ANALYSIS_INDEX.md)** - Overview of all analyses
- **[Universal Grammar Proof](UNIVERSAL_GRAMMAR_PROOF.md)** - Theoretical foundations
- **[Clean Architecture Analysis](CLEAN_ARCHITECTURE_GRAMMAR_ANALYSIS.md)** - TypeScript patterns
- **[Quick Reference Guide](GRAMMAR_QUICK_REFERENCE.md)** - Common patterns
- **[Sentence Validation Examples](SENTENCE_VALIDATION_EXAMPLES.md)** - Test cases
- **[CLAUDE.md](CLAUDE.md)** - AI coding standards

## Architecture

```
fiat-lux.ts
├── Core Type Definitions
│   ├── GenericRecord, RoleConfig, GrammarConfig
│   ├── ValidationError, RepairOperation, ProcessingResult
│   └── MatchCandidate, SimilarityAlgorithm, Severity
├── Similarity Algorithms
│   ├── levenshteinDistance()
│   ├── jaroWinklerSimilarity()
│   ├── levenshteinSimilarity()
│   └── hybridSimilarity()
├── SimilarityCache
│   └── Caching with statistics
├── GrammarEngine<T>
│   ├── validate()
│   ├── repair()
│   ├── process()
│   └── Configuration management
├── Predefined Grammars
│   ├── CLEAN_ARCHITECTURE_GRAMMAR
│   └── HTTP_API_GRAMMAR
└── Utilities
    ├── formatResult()
    └── runDemo()
```

## Contributing

Contributions are welcome! Please read [CLAUDE.md](CLAUDE.md) for coding standards.

### Development Setup

```bash
git clone https://github.com/thiagobutignon/fiat-lux.git
cd fiat-lux
npm install
```

### Running Tests

```bash
npx ts-node fiat-lux.ts
```

### Claude Code Integration

This project uses Claude Code for automated code review. Simply mention `@claude` in issues or PRs.

## Use Cases

- **Code Architecture Validation**: Ensure architectural patterns are followed
- **API Schema Validation**: Validate REST API endpoints and methods
- **Configuration Validation**: Check YAML/JSON config files
- **Natural Language Processing**: Grammar checking and correction
- **Data Quality**: Validate structured data formats
- **Linting**: Create custom linting rules for any domain

## Performance

- **Caching**: Similarity calculations are cached for repeated comparisons
- **Optimized Algorithms**: Efficient O(m×n) implementations
- **Batch Processing**: Process multiple sentences efficiently
- **Configurable**: Adjust performance vs accuracy trade-offs

Example cache performance (100 iterations of same sentence):
- Cache hit rate: ~99%
- Average processing time: <1ms per iteration

## License

MIT License - see [LICENSE](LICENSE) file for details

## Credits

Built with [Claude Code](https://claude.com/claude-code) by Anthropic

## Support

- 📖 [Documentation](https://github.com/thiagobutignon/fiat-lux/tree/main/docs)
- 🐛 [Issue Tracker](https://github.com/thiagobutignon/fiat-lux/issues)
- 💬 [Discussions](https://github.com/thiagobutignon/fiat-lux/discussions)

---

**Fiat Lux** - Let there be light in your structured data! 🌟
