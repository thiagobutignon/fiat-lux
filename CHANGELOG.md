# Changelog

All notable changes to the Fiat Lux project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Anthropic LLM Adapter - Centralized LLM Integration (2025-10-07)

**Core Infrastructure**
- **`src/agi-recursive/llm/anthropic-adapter.ts`**: Complete Anthropic Claude API adapter
  - Centralized LLM integration for all Claude API calls
  - Support for Claude Opus 4 and Sonnet 4.5
  - Automatic cost calculation and tracking
  - Token usage monitoring (input/output)
  - Streaming support for real-time responses
  - Cost estimation without API calls
  - Model comparison utilities

**Model Support**
- **Claude Opus 4**: Latest, most capable model
  - Best for complex reasoning and creative tasks
  - Pricing: $15/1M input tokens, $75/1M output tokens
  - Model ID: `claude-opus-4-20250514`
- **Claude Sonnet 4.5**: Balanced, cost-effective model (default)
  - Fast and efficient for most tasks
  - Pricing: $3/1M input tokens, $15/1M output tokens
  - Model ID: `claude-sonnet-4-5-20250929`

**Cost Tracking**
- Automatic tracking of all API calls
- Real-time cumulative cost monitoring
- Per-request cost calculation
- Total requests counter
- Cost comparison between models
- Estimation tool for planning

**Integration with AGI System**
- **MetaAgent**: Updated to use AnthropicAdapter for all LLM calls
  - Query decomposition now uses adapter
  - Insight composition uses adapter
  - Final synthesis uses adapter
  - Actual cost tracking instead of estimates
- **SpecializedAgent**: Base class updated with adapter
  - All agents now use centralized LLM integration
  - Configurable model selection per agent
  - Automatic cost updates to recursion state
  - Backward compatible with existing agents

**API Features**
- `invoke()`: Standard request/response
- `invokeStream()`: Streaming with async generator
- `estimateCost()`: Pre-call cost estimation
- `compareCosts()`: Compare cost between models
- `getTotalCost()`: Cumulative cost across all calls
- `getTotalRequests()`: Total number of API calls
- `resetStats()`: Reset cost tracking

**Demonstration**
- **`src/agi-recursive/examples/anthropic-adapter-demo.ts`**: Comprehensive adapter demo
  - Model recommendations by task type
  - Cost estimation before API calls
  - Actual invocations with Sonnet 4.5
  - Cost comparison between Opus and Sonnet
  - Streaming response demonstration
  - Cumulative cost tracking
  - Multiple requests cost accumulation
  - Full statistics reporting

**Benefits**
- ✅ Single point of LLM integration (no scattered API calls)
- ✅ Transparent cost tracking (know exactly what you're spending)
- ✅ Easy model switching (change one config, update everywhere)
- ✅ Production-ready (error handling, type safety, streaming)
- ✅ Cost optimization (compare models, estimate before calling)
- ✅ Future-proof (easy to add new models)

#### Slice Navigator - Dynamic Knowledge Discovery (2025-10-07)

**Core Infrastructure**
- **`src/agi-recursive/core/slice-navigator.ts`**: Complete Slice Navigator implementation
  - Dynamic knowledge loading system for AGI agents
  - Load knowledge slices on demand instead of upfront
  - Inverted index for fast concept search (O(1) lookup)
  - Cross-domain navigation through explicit connections
  - In-memory caching for performance optimization

**Knowledge Slices**
- **`src/agi-recursive/slices/`**: Structured knowledge base
  - `financial/budget-homeostasis.slice.yaml`: Budget as homeostatic system
  - `biology/cellular-homeostasis.slice.yaml`: Glucose regulation template
  - `systems/feedback-loops.slice.yaml`: Feedback loop theory
  - Each slice contains: metadata, knowledge, examples, formulas, principles, references

**Slice Structure**
- Metadata: id, domain, concepts, connections, tags
- Content: knowledge (markdown), examples, formulas, principles
- Connections: Explicit links to related slices in other domains
- Discoverable: Agents search by concept and navigate through connections

**Integration**
- MetaAgent initializes SliceNavigator on startup
- All SpecializedAgents have access to navigator
- Agents can search concepts and load relevant knowledge dynamically
- Enables compositional understanding across domains

**Performance**
- Inverted index: concept → slice_ids (fast lookup)
- Domain index: domain → slice_ids (filtering)
- Cache: Recently used slices stay in memory
- BFS pathfinding: Find shortest connection between slices

**Demonstration**
- **`src/agi-recursive/examples/slice-navigation-demo.ts`**: Complete navigation demo
  - 6 test cases showing search, loading, navigation, caching
  - Agent knowledge discovery simulation
  - Cross-domain connection finding
  - Performance benchmarks (2.6x speedup from cache)

**Benefits**
- ✅ Scalable: Add knowledge without modifying code
- ✅ Discoverable: Agents find relevant slices through search
- ✅ Composable: Slices connect across domains
- ✅ Performant: Caching and indexing for fast access
- ✅ Structured: Consistent YAML format across all slices

#### Anti-Corruption Layer for AGI System (2025-10-07)

**Core ACL Infrastructure**
- **`src/agi-recursive/core/anti-corruption-layer.ts`**: Complete Anti-Corruption Layer implementation
  - Validates communication between agents to prevent domain corruption
  - 5 critical validation checks: domain boundaries, constitutional compliance, loop detection, content safety, budget limits
  - Prevents hallucination cascades, infinite recursion, cost explosions, prompt injection
  - Full audit trail with invocation history and context hashing

**Domain Translation System**
- **DomainTranslator**: Semantic translation between agent domains
  - Valid concept mappings (homeostasis → budget_equilibrium, feedback_loop → spending_monitoring)
  - Forbidden translation detection (DNA → financial blocked)
  - Prevents semantic corruption when concepts cross domain boundaries
  - Available translations API for discovery

**Safety Mechanisms**
- Domain boundary enforcement: Prevents agents from speaking outside expertise
- Loop detection: Identifies cycles (A→B→C→A) and consecutive same-agent invocations
- Content safety filtering: Blocks dangerous patterns (SQL injection, rm -rf) with context awareness
- Budget enforcement: Hard limits on depth (≤5), invocations (≤10), cost (≤$1.00)
- Constitutional violation errors with severity levels (warning, error, fatal)

**Integration with MetaAgent**
- ACL validation added before Constitution checks in recursive processing
- Fatal violations stop agent processing immediately
- Warnings logged but allow continuation
- Full integration with existing ConstitutionEnforcer

**Demonstration**
- **`src/agi-recursive/examples/acl-protection-demo.ts`**: Comprehensive ACL demonstration
  - 7 test cases showing all protection mechanisms
  - Domain boundary violations
  - Valid and forbidden cross-domain translations
  - Loop detection
  - Content safety filtering
  - Budget tracking
  - Invocation history audit trail
  - Educational output explaining each protection mechanism

#### Universal Grammar Documentation (2025-10-07)
- **`docs/UNIVERSAL_GRAMMAR_PATTERNS_EXTRACTED.md`**: Comprehensive 1462-line extraction of all Universal Grammar patterns from Fiat Lux research
  - 6 core patterns (DOM-001 through MAIN-001) with deep structure specifications
  - Linguistic mapping of architecture elements as natural language components
  - Context-Free Grammar (CFG) rules in BNF notation
  - Multi-project validation across 5 different projects
  - Cross-language proof (TypeScript, Swift, Dart, Python)
  - Complete Fiat Lux Grammar Engine API documentation
  - 10 anti-patterns catalog with detection strategies
  - Comprehensive naming conventions for all architectural layers
- **`docs/validate-grammar-patterns.ts`**: Validation script using PatternLoader
  - Tests naming conventions (e.g., "DbAddAccount" for data/usecases)
  - Validates dependency rules (e.g., domain → infrastructure forbidden)
  - Demonstrates grammar loading and validation API
  - Comprehensive test suite for all validation scenarios

#### Deterministic Intelligence Benchmark System (2025-10-07)

**Core Benchmark Infrastructure**
- Full benchmark system restructured following Clean Architecture principles
- Feature-based organization: `src/benchmark/` with domain/, data/, infrastructure/ layers
- All files renamed to kebab-case following project conventions
- 8 comprehensive documentation files organized in `src/benchmark/docs/`

**Benchmark Components**
- **`src/benchmark/domain/entities/`**: Core domain entities
  - `benchmark-result.ts`: Result aggregation and comparison
  - `candlestick.ts`: Trading candlestick with pattern recognition
  - `candlestick-sequence.ts`: Sequence processing
  - `error-analysis.ts`: Confusion matrix and error metrics (precision, recall, F1)
  - `pattern.ts`: Pattern detection entities
  - `trading-signal.ts`: Signal classification (BULLISH, BEARISH, NEUTRAL)

- **`src/benchmark/domain/use-cases/`**: Business logic
  - `benchmark-orchestrator.ts`: Main benchmark coordination and execution
  - `run-benchmark.ts`: Individual benchmark runner with error analysis

- **`src/benchmark/data/`**: Data layer
  - `protocols/pattern-detector.ts`: Interface for pattern detection systems
  - `use-cases/candlestick-generator.ts`: Test case generation (498 lines)
  - `use-cases/error-analysis-builder.ts`: Error analysis aggregation

- **`src/benchmark/infrastructure/adapters/`**: External integrations
  - `grammar-pattern-detector.ts`: Deterministic grammar engine (100% accuracy)
  - `gemini-pattern-detector.ts`: Google Gemini API integration
  - `llamacpp-detector.ts`: llama.cpp for Mac M-series (Metal acceleration)
  - `vllm-pattern-detector.ts`: vLLM for high-performance GPU inference
  - `local-llama-detector.ts`: Ollama local LLM integration
  - `lstm-pattern-detector.ts`: Custom LSTM baseline
  - `llm-pattern-detector.ts`: Base class for simulated LLMs

**Scripts and Utilities**
- **`scripts/benchmark/run-benchmark.ts`**: CLI entry point
- **`scripts/benchmark/debug-accuracy.ts`**: Detailed failure analysis
- **`scripts/benchmark/export-error-analysis.ts`**: JSON export for visualization
- **`scripts/start-llamacpp-mac.sh`**: Automated Mac M-series setup script
  - Detects Mac specs (cores, model)
  - Optimizes thread count (80% of available cores)
  - Configures Metal acceleration (-ngl 999)
  - Supports custom model and port selection

**Documentation**
- **`src/benchmark/docs/README.md`**: Complete benchmark overview
- **`src/benchmark/docs/MAC_SETUP.md`**: Mac M1/M2/M3/M4 setup guide (295 lines)
  - llama.cpp installation via Homebrew
  - Model quantization options (Q4_K_M, Q8_0, FP16)
  - Performance tuning for Apple Silicon
  - Troubleshooting common issues
- **`src/benchmark/docs/QUICKSTART_MAC.md`**: Quick start guide for Mac users
- **`src/benchmark/docs/VLLM_SETUP.md`**: vLLM GPU acceleration setup
- **`src/benchmark/docs/ACCURACY_IMPROVEMENTS.md`**: Performance optimization history
  - Documents improvement from 30% → 87% → 97% → 100% accuracy
  - Pattern threshold tuning methodology
  - Detection algorithm enhancements
- **`src/benchmark/docs/FIXES_SUMMARY.md`**: Bug fixes and solutions
- **`src/benchmark/docs/PATTERN_THRESHOLDS.md`**: Threshold calibration details (312 lines)
- **`src/benchmark/docs/ERROR_ANALYSIS_README.md`**: Error analysis methodology (345 lines)

### Performance Results (100 test cases)

| System | Accuracy | Avg Latency | Total Cost | Explainability |
|--------|----------|-------------|------------|----------------|
| **Grammar Engine** | 100.0% | 0.013ms | $0.00 | 100% |
| llama.cpp (Q4) | 48.0% | 3545.5ms | $0.00 | 10% |
| Custom LSTM | 56.0% | 45.9ms | $0.00 | 0% |
| GPT-4 (simulated) | 26.0% | 3000.0ms | $0.30 | 20% |
| Claude 3.5 (simulated) | 30.0% | 2000.0ms | $0.15 | 20% |
| Gemini 2.5 Flash (simulated) | 30.0% | 1500.0ms | $0.02 | 20% |

**Key Findings:**
- Grammar Engine achieved perfect accuracy (100%) with deterministic rules
- 273,000x faster than LLMs (0.013ms vs 3.5s)
- Zero cost vs $0.15-0.30 per 100 predictions
- Full explainability vs black-box neural networks

### Changed

#### Project Restructuring (2025-10-07)
- Moved all benchmark code from `landing/src/` to `src/benchmark/`
- Renamed all TypeScript files from PascalCase to kebab-case
- Updated all import paths to reflect new structure
- Organized documentation into `src/benchmark/docs/`
- Migrated scripts to `scripts/benchmark/`

**File Migration Summary:**
- 18 TypeScript source files
- 8 documentation files
- 3 executable scripts
- Total: 29 files reorganized

### Technical Details

**Architecture Patterns:**
- Clean Architecture with Dependency Inversion
- Strategy Pattern for pluggable detection algorithms
- Factory Pattern for detector instantiation
- Builder Pattern for error analysis aggregation

**Performance Optimizations:**
- Caching for pattern detection results
- Early exit for deterministic rules
- Vectorized similarity calculations
- Memory-efficient sequence processing

**Error Analysis Features:**
- Confusion matrix generation
- Per-pattern accuracy tracking
- False positive/negative analysis
- Statistical significance testing

## [0.2.0] - 2025-10-06

### Added
- Next.js landing page for Fiat Lux showcase
- Interactive demos and documentation
- Visual grammar examples

## [0.1.0] - 2025-10-05

### Added
- **Pattern Loader**: YAML-based architectural pattern loading
- **Naming Convention Validation**: Per-layer naming rules
- **Dependency Rule Validation**: Inter-layer dependency checking
- **Clean Architecture Patterns**: Predefined patterns for domain, data, presentation, infrastructure, and main layers
- **PatternLoader API**: Complete API for pattern management and validation
- Comprehensive test suite (32 tests) for pattern loading

### Changed
- Restructured project following Clean Architecture principles
- Feature-based organization: `src/[feature]/[domain|data|presentation]`
- Renamed all layers to match conventions (domain, data, presentation, infrastructure, main)

## [0.0.1] - Initial Release

### Added
- **Grammar Engine**: Generic, configurable validation and auto-repair
- **Similarity Algorithms**: Levenshtein, Jaro-Winkler, Hybrid
- **Performance Optimization**: Similarity caching with hit rate tracking
- **Auto-Repair**: Multiple suggestions with confidence scores
- **Predefined Grammars**: Clean Architecture and HTTP API grammars
- **Type Safety**: Full TypeScript support with generics
- **Custom Test Framework**: Ultra-fast testing (<5ms for 77 tests)
- Complete documentation and examples

---

**Note**: This changelog follows the principles of [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
