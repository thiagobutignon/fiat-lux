/**
 * Shared Domain Entities for Benchmark Module
 *
 * These entities form the shared kernel of the benchmark domain.
 * All use-cases can import from here.
 */

// Core domain entities
export { Candlestick } from './domain/entities/candlestick';
export { CandlestickSequence } from './domain/entities/candlestick-sequence';
export { TradingSignal, SignalType } from './domain/entities/trading-signal';
export { Pattern, PatternType, SignalStrength } from './domain/entities/pattern';
