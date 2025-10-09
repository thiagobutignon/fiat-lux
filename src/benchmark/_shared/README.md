# 📦 Shared Domain Entities

This directory contains **domain entities** that are shared across **all** benchmark use-cases.

## What Goes Here?

**Only core domain entities that are:**
1. Used by multiple use-cases
2. Represent fundamental domain concepts
3. Have no business logic specific to one use-case

## Current Shared Entities

### `candlestick.ts`
Core domain entity representing a single candlestick (OHLC data).

**Used by:**
- `generate-candlestick/` - Creates test candlesticks
- `detect-pattern/` - Analyzes candlestick patterns
- `analyze-errors/` - Reports errors with candlestick data

### `candlestick-sequence.ts`
Collection of candlesticks forming a time series.

**Used by:**
- All pattern detectors
- Benchmark orchestrator
- Test data generation

### `trading-signal.ts`
Represents a trading decision (BUY/SELL/HOLD) with confidence and reasoning.

**Used by:**
- All pattern detectors (output)
- Benchmark runner (comparison)
- Error analyzer (expected vs predicted)

### `pattern.ts`
Defines candlestick pattern types and characteristics.

**Used by:**
- Pattern detectors (output)
- Test generator (creates specific patterns)
- Error analyzer (pattern-specific accuracy)

## What Does NOT Go Here?

❌ **Use-case specific entities**: Go in the use-case's own `domain/entities/`
- `benchmark-result.ts` → Belongs in `run-benchmark/domain/entities/`
- `error-analysis.ts` → Belongs in `analyze-errors/domain/entities/`

❌ **Use-cases**: Go in `[use-case]/domain/use-cases/`

❌ **Adapters/Infrastructure**: Go in `[use-case]/infrastructure/adapters/`

❌ **Protocols/Interfaces**: Go in `[use-case]/data/protocols/`

## Import Examples

### From a use-case
```typescript
// ✅ Correct: Import from _shared
import { Candlestick } from '../_shared/domain/entities/candlestick';
import { TradingSignal } from '../_shared/domain/entities/trading-signal';

// ❌ Wrong: Don't import from old structure
import { Candlestick } from '../../domain/entities/candlestick';
```

### From another module
```typescript
// From src/grammar-engine/
import { Candlestick } from '../benchmark/_shared/domain/entities/candlestick';
```

## Design Principle

**Shared Kernel Pattern**: These entities form a "shared kernel" of the benchmark domain. They are stable and change infrequently. Each use-case can depend on the shared kernel, but use-cases should NOT depend on each other.

```
Use-Case 1  ──┐
              ├──→  _shared/  (Shared Kernel)
Use-Case 2  ──┤
              │
Use-Case 3  ──┘

❌ Use-Case 1 → Use-Case 2  (Don't do this!)
```
