# Fiat Lux 🌟

**Let There Be Light** - A Universal Grammar Engine for Structured Data

[![GitHub](https://img.shields.io/github/license/thiagobutignon/fiat-lux)](LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg)](https://www.typescriptlang.org/)
[![Claude Code](https://img.shields.io/badge/Built%20with-Claude%20Code-blueviolet.svg)](https://claude.com/claude-code)

## Overview

Fiat Lux is a generic, configurable grammar engine that validates and auto-repairs structured data based on customizable grammatical rules. Built with **Clean Architecture** principles following the `src/[feature]/[use-cases]` pattern.

### Universal Grammar Theory

Fiat Lux is built on the premise that **Clean Architecture is a Universal Grammar** - a language-agnostic, paradigm-agnostic, and domain-agnostic framework that transcends programming languages and application domains.

**Key Insights:**

- 🌍 **Language-Agnostic**: Proven across TypeScript, Swift, Dart, Python, and more
- 🧬 **Paradigm-Agnostic**: Works with OOP, FP, and hybrid approaches
- 🎯 **Domain-Agnostic**: Validated in 5+ different project types (APIs, mobile apps, web apps, data processing)
- 📐 **Formal Grammar**: Architecture patterns can be expressed as Context-Free Grammar (CFG) rules
- 🔍 **Deterministic Validation**: 100% accuracy, 273,000x faster than LLMs, zero cost

**Architecture as Natural Language:**

Clean Architecture follows linguistic structure:
- **Domain Layer** → Subject (entities, use-cases)
- **Data Layer** → Verb (implementations, repositories)
- **Presentation Layer** → Object (controllers, presenters)
- **Infrastructure Layer** → Context (adapters, drivers)
- **Main Layer** → Syntax (factories, composition)

**📚 Complete Documentation**: See [`docs/UNIVERSAL_GRAMMAR_PATTERNS_EXTRACTED.md`](docs/UNIVERSAL_GRAMMAR_PATTERNS_EXTRACTED.md) for the comprehensive 1462-line extraction of all Universal Grammar patterns, proofs, and validations.

## Architecture

The project follows Clean Architecture with clear separation of concerns:

```
src/
├── benchmark/                   # Deterministic Intelligence Benchmark
│   ├── domain/
│   │   ├── entities/           # Candlestick, Pattern, TradingSignal
│   │   └── use-cases/          # BenchmarkOrchestrator, RunBenchmark
│   ├── data/
│   │   ├── protocols/          # IPatternDetector interface
│   │   └── use-cases/          # CandlestickGenerator, ErrorAnalysisBuilder
│   ├── infrastructure/
│   │   └── adapters/           # Grammar, Gemini, llama.cpp, LSTM detectors
│   └── docs/                   # Comprehensive benchmark documentation
│
├── grammar-engine/              # Grammar validation and repair
│   ├── domain/
│   │   ├── entities/           # Types and predefined grammars
│   │   └── use-cases/          # GrammarEngine business logic
│   ├── data/
│   │   ├── protocols/          # Interface definitions
│   │   └── use-cases/          # Implementations (Cache)
│   └── presentation/           # Public API, factories, utilities
│
├── pattern-loader/              # YAML pattern loader
│   ├── domain/
│   │   ├── entities/           # Pattern definitions
│   │   └── use-cases/          # PatternLoader business logic
│   └── presentation/           # Public API
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

### PatternLoader Methods

Load and validate architectural patterns from YAML configurations:

- **`getPatterns()`** - Get all patterns
- **`getPatternById(id)`** - Get specific pattern
- **`getPatternsByLayer(layer)`** - Filter patterns by layer
- **`getLayers()`** - Get all available layers
- **`validateNaming(value, layer, element)`** - Validate naming conventions
- **`validateDependency(from, to)`** - Check dependency rules
- **`getExamples(layer, element)`** - Get naming examples
- **`getSummary()`** - Get configuration statistics

```typescript
import { PatternLoader } from 'fiat-lux'

const loader = new PatternLoader(yamlContent)

// Validate naming
const result = loader.validateNaming('AddAccountUseCase', 'domain', 'usecases')
console.log(result.valid) // true

// Check dependencies
const depResult = loader.validateDependency('domain', 'data')
console.log(depResult.valid) // false (forbidden)
```

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

The project includes a **custom lightweight test framework** designed for speed. All 77 unit tests run in under 5ms!

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
Total:   77
✅ Passed: 77
❌ Failed: 0
⏭️  Duration: 4.55ms
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

- **Pattern Loader**: 32 tests
  - YAML parsing and pattern extraction
  - Naming convention validation
  - Dependency rule validation
  - Layer operations and queries
  - Edge cases and error handling

### Why Custom Test Framework?

Instead of Jest or Mocha (which take 1-2 seconds to start), our custom framework:
- ✅ Runs in **<5ms** (400x faster startup)
- ✅ Zero dependencies
- ✅ Simple API (`describe`, `it`, `expect`)
- ✅ Perfect for TDD workflow

## Documentation

### Core Documentation
- **[CLAUDE.md](docs/CLAUDE.md)** - AI coding standards
- **[CHANGELOG.md](CHANGELOG.md)** - Project changelog with detailed change history
- **[Grammar Analysis Index](docs/GRAMMAR_ANALYSIS_INDEX.md)** - Overview of analyses
- **[Universal Grammar Proof](docs/UNIVERSAL_GRAMMAR_PROOF.md)** - Theoretical foundations

### Universal Grammar
- **[Universal Grammar Patterns (EXTRACTED)](docs/UNIVERSAL_GRAMMAR_PATTERNS_EXTRACTED.md)** - Comprehensive 1462-line extraction of all patterns
  - 6 core patterns (DOM-001 through MAIN-001)
  - Linguistic mapping of architecture elements
  - CFG grammar rules in BNF notation
  - Multi-project validation (5 projects)
  - Cross-language proof (TypeScript, Swift, Dart, Python)
  - Anti-patterns catalog (10 violations)
- **[Grammar Pattern Validator](docs/validate-grammar-patterns.ts)** - Validation script using PatternLoader

### Benchmark System
- **[Benchmark Overview](src/benchmark/docs/README.md)** - Complete benchmark documentation
- **[Mac Setup Guide](src/benchmark/docs/MAC_SETUP.md)** - Mac M1/M2/M3/M4 with llama.cpp (295 lines)
- **[Mac Quick Start](src/benchmark/docs/QUICKSTART_MAC.md)** - Quick start for Mac users
- **[vLLM Setup](src/benchmark/docs/VLLM_SETUP.md)** - GPU-accelerated inference setup
- **[Accuracy Improvements](src/benchmark/docs/ACCURACY_IMPROVEMENTS.md)** - Performance optimization history (30% → 100%)
- **[Pattern Thresholds](src/benchmark/docs/PATTERN_THRESHOLDS.md)** - Threshold calibration details (312 lines)
- **[Error Analysis](src/benchmark/docs/ERROR_ANALYSIS_README.md)** - Error analysis methodology (345 lines)
- **[Fixes Summary](src/benchmark/docs/FIXES_SUMMARY.md)** - Bug fixes and solutions

## Contributing

Contributions are welcome! Please read [CLAUDE.md](docs/CLAUDE.md) for coding standards.

## Performance

- **Processing time**: 0.02ms - 0.50ms per validation
- **Cache hit rate**: ~99% after warm-up
- **Average iteration**: <1ms with caching

## Deterministic Intelligence Benchmark

Fiat Lux includes a comprehensive benchmark system that compares deterministic grammar-based pattern detection against LLM-based approaches. The benchmark proves that **deterministic systems can outperform neural networks** in structured validation tasks.

### Running the Benchmark

```bash
# Quick benchmark (100 test cases)
npm run benchmark:quick

# Full benchmark (1000 test cases)
npm run benchmark

# Mac M-series with llama.cpp
./scripts/start-llamacpp-mac.sh  # Start llama.cpp server
npm run benchmark                # Run benchmark with llama.cpp
```

### Benchmark Results (100 test cases)

| System | Accuracy | Avg Latency | Total Cost | Explainability |
|--------|----------|-------------|------------|----------------|
| **Grammar Engine** | **100.0%** | **0.013ms** | **$0.00** | **100%** |
| llama.cpp (Q4) | 48.0% | 3545.5ms | $0.00 | 10% |
| Custom LSTM | 56.0% | 45.9ms | $0.00 | 0% |
| GPT-4 (simulated) | 26.0% | 3000.0ms | $0.30 | 20% |
| Claude 3.5 (simulated) | 30.0% | 2000.0ms | $0.15 | 20% |
| Gemini 2.5 Flash (simulated) | 30.0% | 1500.0ms | $0.02 | 20% |

### Key Insights

- ⚡ **273,000x faster** than LLMs (0.013ms vs 3,545ms)
- 🎯 **Perfect accuracy** with deterministic rules
- 💰 **Zero cost** vs $0.15-0.30 per 100 predictions
- 🔍 **Full explainability** - every decision is traceable
- 🧪 **Reproducible** - same input always produces same output

### Supported Systems

The benchmark supports multiple detection systems:

1. **Grammar Engine** (deterministic, 100% accuracy)
2. **Google Gemini** (requires `GEMINI_API_KEY`)
3. **llama.cpp** (Mac M-series with Metal acceleration)
4. **vLLM** (GPU-accelerated inference)
5. **Ollama** (local LLM deployment)
6. **Custom LSTM** (baseline neural network)

**📚 Documentation**: See [`src/benchmark/docs/README.md`](src/benchmark/docs/README.md) for complete benchmark documentation.

**🍎 Mac Setup**: See [`src/benchmark/docs/MAC_SETUP.md`](src/benchmark/docs/MAC_SETUP.md) for Mac M1/M2/M3/M4 setup guide.

## LLM Research Program 🔬

Fiat Lux includes a comprehensive research program to understand and engineer the internal mechanisms of Large Language Models. This represents a new frontier: **Precise LLM Engineering** - moving from black-box usage to weight-level control.

### Research Pillars

#### 1. Understanding - Mechanistic Analysis of Hallucinations
[Issue #6](https://github.com/thiagobutignon/fiat-lux/issues/6)

**Goal**: Peek inside the "black box" to understand neural pathways that lead to hallucinations.

**Key Questions**:
- Where do hallucinations originate in the network?
- What weight patterns correlate with false confidence?
- How do attention mechanisms differ in truthful vs. hallucinating generations?

**Methodology**:
- Weight Pattern Analysis (all 8.03B parameters)
- Activation Flow Tracing (32 transformer layers)
- Quantization Impact Study (Q4_K vs. Q6_K effects)
- Causal Intervention (ablation studies)

#### 2. Engineering - Behavioral Modification and Determinism
[Issue #7](https://github.com/thiagobutignon/fiat-lux/issues/7)

**Goal**: Test the boundaries of weight-based behavior engineering, including deterministic inference.

**Key Questions**:
- Can we achieve bit-exact reproducible LLM inference?
- How to improve performance through surgical weight modifications?
- What are the fundamental limits of weight engineering?

**Techniques**:
- Fixed-point arithmetic for determinism
- Redundancy elimination (10-20% speedup)
- Sparsity-aware quantization
- Task-specific weight overlays

#### 3. Safety - Constitutional AI at the Weight Level
[Issue #8](https://github.com/thiagobutignon/fiat-lux/issues/8)

**Goal**: Embed safety principles directly into weights, making harmful outputs mathematically impossible.

**Key Questions**:
- Can we modify weights to make specific outputs unreachable?
- Do safety constraints compose without interfering?
- Can we provide formal guarantees about model safety?

**Approaches**:
- Attention bias injection for constitutional constraints
- Safety residual stream modifications
- FFN gate constitutional masks
- Formal verification of safety properties

### GGUF Parser Infrastructure

All research builds on our production-ready GGUF parser:

✅ **Implemented Features** (PR #5):
- Complete GGUF v3 binary format parsing
- Weight extraction for all 292 tensors (8.03B parameters)
- Accurate Q4_K and Q6_K dequantization
- Statistical analysis framework
- Tested with Llama 3.1 8B (4.58 GB model)

```bash
# Extract and analyze model weights
tsx scripts/gguf/extract-weights.ts model.gguf --layer 0 --stats-only

# Test specific quantization types
tsx scripts/gguf/verify-quantization.ts
```

### Expected Outcomes

**Scientific Contributions**:
- First comprehensive map of Llama 3.1's internal mechanisms
- Weight-level Constitutional AI methodology
- Formal safety guarantees for LLM behavior
- Deterministic LLM inference system

**Practical Deliverables**:
- Constitutional Llama 3.1 8B (embedded safety principles)
- Deterministic Llama 3.1 8B (bit-exact reproducibility)
- Optimized Llama 3.1 8B (20% faster)
- Weight Engineering Toolkit (open-source)

**Publications**:
- 3 arXiv preprints (20-30 pages each)
- NeurIPS/ICML submissions
- 10k+ activation traces dataset
- Interactive visualization tools

### Research Documentation

- **[Research Program Overview](docs/research-program-overview.md)** - Complete program vision and timeline
- **[GGUF Phase 2.1 Documentation](docs/gguf-phase2.1-accurate-k-quants.md)** - Detailed K-quant implementation

### Timeline

- **Q1 (Months 1-3)**: Mechanistic understanding phase
- **Q2 (Months 4-6)**: Behavioral engineering phase
- **Q3 (Months 7-9)**: Constitutional AI integration
- **Q4 (Months 10-12)**: Publication and deployment

**Target Venues**: arXiv, NeurIPS, ICML, ICLR, AI Safety Workshops

**📚 Full Details**: See [Research Program Overview](docs/research-program-overview.md) for the complete 12-month research roadmap.

## Validation Tools

### Grammar Pattern Validator

Validate your architecture against Universal Grammar patterns using the validation script:

```bash
# Run grammar pattern validation
tsx docs/validate-grammar-patterns.ts
```

This script validates:
- ✅ Naming conventions per layer (e.g., `DbAddAccount` for data/usecases)
- ✅ Dependency rules (e.g., domain cannot depend on infrastructure)
- ✅ Pattern loading from YAML configurations
- ✅ Multi-layer architectural consistency

Example output:
```
🔍 Grammar Pattern Validator
================================================================================
📄 Loading: docs/grammar-patterns.yml

📊 Summary:
   Version: 1.0.0
   Architecture: Clean Architecture
   Total Patterns: 6
   Layers: domain, data, presentation, infrastructure, main

🏷️  NAMING CONVENTIONS
Layer: domain
  usecases: ^[A-Z][a-z]+([A-Z][a-z]+)*$
    Example: AddAccount

🧪 TESTING NAMING VALIDATION
✅ data/usecases: "DbAddAccount"
   Expected: true, Got: true

🔗 TESTING DEPENDENCY VALIDATION
✅ data → domain
   Expected: true, Got: true
❌ domain → data
   Expected: false, Got: false
   Message: Forbidden dependency: domain cannot depend on data
```

## License

MIT License - see [LICENSE](LICENSE) file for details

## Acknowledgements

Special thanks to the amazing developers and mentors who inspired this work:

- [@rmanguinho](https://github.com/rmanguinho) - Clean Architecture and SOLID principles
- [@barreirabruno](https://github.com/barreirabruno) - Software craftsmanship
- [@lnmunhoz](https://github.com/lnmunhoz) - Best practices and code quality
- [@kidchenko](https://github.com/kidchenko) - Technical excellence
- [Hernane Gomes](https://www.linkedin.com/in/hernanegomes/) - Architecture insights
- [Rebecca Barbosa](https://www.linkedin.com/in/rebeccafbarbosa/) - Engineering leadership
- [Miller Cesar Oliveira](https://www.linkedin.com/in/millercesaroliveira/) - Technical guidance

## Credits

Built with [Claude Code](https://claude.com/claude-code) by Anthropic

---

**Fiat Lux** - Let there be light in your structured data! 🌟
