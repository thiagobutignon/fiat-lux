# 🔧 Benchmark Module Refactoring Plan

## Current Structure (Horizontal Layers) ❌

```
src/benchmark/
├── domain/
│   ├── entities/         ← All entities together
│   └── use-cases/        ← All use-cases together
├── data/
│   ├── protocols/
│   └── use-cases/
└── infrastructure/
    └── adapters/         ← All adapters together
```

**Problem**: Cannot follow Grammar Engine pattern. AI cannot generate code with 100% accuracy using horizontal layers.

## New Structure (Vertical Slices) ✅

```
src/benchmark/
├── _shared/                           ← Shared domain entities
│   └── domain/
│       └── entities/
│           ├── candlestick.ts
│           ├── candlestick-sequence.ts
│           ├── trading-signal.ts
│           └── pattern.ts
│
├── run-benchmark/                     ← Use-Case 1: Execute benchmark
│   ├── domain/
│   │   ├── entities/
│   │   │   └── benchmark-result.ts
│   │   └── use-cases/
│   │       └── run-benchmark.ts
│   ├── data/
│   │   └── protocols/
│   │       └── pattern-detector.ts
│   └── main/
│       └── index.ts
│
├── orchestrate-benchmark/             ← Use-Case 2: Coordinate suite
│   ├── domain/
│   │   └── use-cases/
│   │       └── benchmark-orchestrator.ts
│   └── main/
│       └── index.ts
│
├── generate-candlestick/              ← Use-Case 3: Generate test data
│   ├── domain/
│   │   └── use-cases/
│   │       └── candlestick-generator.ts
│   └── main/
│       └── index.ts
│
├── analyze-errors/                    ← Use-Case 4: Analyze detection errors
│   ├── domain/
│   │   ├── entities/
│   │   │   └── error-analysis.ts
│   │   └── use-cases/
│   │       └── error-analysis-builder.ts
│   └── main/
│       └── index.ts
│
└── detect-pattern/                    ← Use-Case 5: Pattern detection
    ├── data/
    │   └── protocols/
    │       └── pattern-detector.ts    (copy of protocol)
    ├── infrastructure/
    │   └── adapters/
    │       ├── grammar-pattern-detector.ts    (100% accuracy)
    │       ├── lstm-pattern-detector.ts       (75% accuracy)
    │       ├── llm-pattern-detector.ts        (82-89% accuracy)
    │       ├── gemini-pattern-detector.ts     (Real API)
    │       ├── llamacpp-detector.ts
    │       ├── local-llama-detector.ts
    │       └── vllm-pattern-detector.ts
    └── main/
        └── index.ts
```

## Migration Strategy

### Phase 1: Create _shared/ ✅ (Next)
- Move common entities to `_shared/domain/entities/`
- Keep originals for backward compatibility
- Update imports to use shared entities

### Phase 2: Migrate run-benchmark/
- Create use-case folder
- Move `benchmark-result.ts` entity
- Move `run-benchmark.ts` use-case
- Update imports

### Phase 3: Migrate generate-candlestick/
- Create use-case folder
- Move `candlestick-generator.ts`
- Export TestCase type

### Phase 4: Migrate detect-pattern/
- Create use-case folder
- Move all detector adapters
- Copy protocol interface

### Phase 5: Migrate analyze-errors/
- Create use-case folder
- Move `error-analysis.ts` entity
- Move `error-analysis-builder.ts` use-case

### Phase 6: Migrate orchestrate-benchmark/
- Create use-case folder
- Move `benchmark-orchestrator.ts`
- Update to use new import paths

### Phase 7: Cleanup
- Delete old structure
- Update all imports across project
- Run build validation
- Run Grammar Engine validation

## Dependency Graph

### Shared Entities (Used by ALL)
```
_shared/domain/entities/
├── candlestick.ts               (no deps)
├── pattern.ts                   (no deps)
├── candlestick-sequence.ts      (depends: candlestick)
└── trading-signal.ts            (depends: pattern)
```

### Use-Case Dependencies
```
run-benchmark/
└── Uses: _shared entities, pattern-detector protocol

generate-candlestick/
└── Uses: _shared entities

detect-pattern/
└── Uses: _shared entities, implements protocol

analyze-errors/
└── Uses: _shared entities, error-analysis entity

orchestrate-benchmark/
└── Uses: ALL other use-cases
```

## Import Path Changes

### Before (Horizontal)
```typescript
import { Candlestick } from '../../domain/entities/candlestick';
import { IPatternDetector } from '../../data/protocols/pattern-detector';
```

### After (Vertical)
```typescript
import { Candlestick } from '../_shared/domain/entities/candlestick';
import { IPatternDetector } from '../detect-pattern/data/protocols/pattern-detector';
```

## Benefits

✅ **Grammar-aligned**: Each use-case follows Subject-Verb-Object-Context pattern
✅ **Self-contained**: Each use-case has all layers it needs
✅ **Discoverable**: Navigate by feature, not by layer
✅ **AI-friendly**: Grammar Engine can generate code with 100% accuracy
✅ **Scalable**: Add new use-cases without touching others
✅ **Testable**: Each use-case can be tested independently

## Validation

After migration, validate with Grammar Engine:
```bash
npm run grammar:validate src/benchmark/
```

Expected result: **100% accuracy** (same as pattern detection)
