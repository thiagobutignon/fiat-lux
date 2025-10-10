# Grammar Language - Tooling Suite

Production-ready O(1) performance tools for Grammar Language development, benchmarking, and debugging.

## 🚀 Features

### GBench - Benchmarking Tool
High-performance benchmarking suite with O(1) operations and statistical analysis.

- ⚡ **O(1) Hash-Based Registration**: Constant-time benchmark add/remove/lookup
- 📊 **Statistical Analysis**: Mean, median, p50, p95, p99, stddev, variance
- 🧠 **Memory Tracking**: Heap usage, peak memory, GC events
- 🔍 **GVCS-Inspired Comparison**: Fitness-based selection and regression detection
- 📝 **Multi-Format Reporting**: Console (colorized), JSON, CSV, Markdown, HTML
- 🚨 **Automated Regression Detection**: Configurable thresholds with history tracking

### GDebug - Debugging Tool
Complete O(1) debugging toolkit with advanced inspection capabilities.

- 🎯 **O(1) Breakpoint Management**: Hash-indexed breakpoints with conditional support
- 🔬 **Deep Variable Inspection**: Properties, prototypes, constructors, type info
- 👁️ **Watch Expressions**: Auto-updating expressions with change detection
- 🚶 **Step Execution**: Step-over, step-into, step-out, continue modes
- 🎬 **Replay Debugging**: Record and replay execution with full state
- 📚 **Stack Trace Management**: Async call chains, source maps, frame filtering
- 🌐 **Multi-Scope Inspection**: Local, closure, module, global scopes
- 🗺️ **Source Map Support**: Resolve compiled code back to source

---

## 📦 Installation

```bash
cd src/grammar-lang/tools
```

All tools are self-contained with no external dependencies beyond TypeScript.

---

## 🔧 GBench Usage

### Quick Start

```typescript
import { suite, report } from './gbench';

// Create benchmark suite
const bench = suite('my-benchmarks');

// Add benchmarks
bench.add('hash-map-lookup', () => {
  map.get(key);
});

bench.add('array-lookup', () => {
  array.find(x => x.key === key);
});

// Run all benchmarks
await bench.runAll({ iterations: 10000 });

// Display results
const result = bench.getResult('hash-map-lookup');
report(result);
```

### Advanced Features

```typescript
import {
  suite,
  compare,
  reportComparison,
  createRegressionDetector
} from './gbench';

// Performance comparison
const baseline = bench.getResult('v1.0.0');
const candidate = bench.getResult('v1.0.1');

const comparison = compare('Baseline', baseline, 'Optimized', candidate);
reportComparison(comparison);

// Automated regression detection
const detector = createRegressionDetector(10); // 10% threshold

detector.addResult('myFunc', baseline);
detector.addResult('myFunc', candidate);

if (detector.detectRegression('myFunc')) {
  console.log(detector.getReport('myFunc'));
}
```

### Output Formats

```typescript
import { BenchmarkExporter } from './gbench';

const results = await bench.runAll();

// Export to different formats
BenchmarkExporter.export(
  Array.from(results.values()),
  'json',
  './benchmarks.json'
);

BenchmarkExporter.export(
  Array.from(results.values()),
  'csv',
  './benchmarks.csv'
);

BenchmarkExporter.export(
  Array.from(results.values()),
  'markdown',
  './BENCHMARKS.md'
);
```

### Demo

```bash
npx ts-node src/grammar-lang/tools/gbench/demo.ts
```

**Output:**
```
✅ Hash Map vs Array: 153x faster (4.9M ops/sec vs 32K ops/sec)
✅ Version comparison: v1.0.1 shows 54% improvement
✅ Memory tracking: 1.44 MB vs 13.80 MB peak
```

---

## 🐛 GDebug Usage

### Quick Start

```typescript
import { createDebugger } from './gdebug';

// Create debugger instance
const dbg = createDebugger();

// Add breakpoint
const bp = dbg.breakpoints.add('app.ts', 42);

// Check if should break
const hit = dbg.breakpoints.shouldBreak('app.ts', 42);
if (hit) {
  dbg.breakpoints.recordHit(hit.id, locals, stackDepth);
}

// Inspect variables
dbg.inspector.setLocals({ x: 10, y: 20, name: 'test' });
const variable = dbg.inspector.getVariable('x');
console.log(variable); // { name: 'x', value: 10, type: 'number', ... }
```

### Conditional Breakpoints

```typescript
// Function-based condition
dbg.breakpoints.add('app.ts', 42, {
  condition: (locals) => locals.x > 10,
  log_message: 'x exceeded threshold'
});

// Max hits
dbg.breakpoints.add('loop.ts', 15, {
  max_hits: 5 // Break only first 5 times
});
```

### Variable Inspection

```typescript
// Deep object inspection
const result = dbg.inspector.inspect('config');
console.log(result.properties);      // Object properties
console.log(result.constructor_name); // Constructor
console.log(result.prototype);        // Prototype chain

// Watch expressions
const watchId = dbg.inspector.addWatch('x * 2', () => {
  const x = dbg.inspector.getVariable('x');
  return x ? x.value * 2 : null;
});

// Update and check changes
dbg.inspector.updateWatch(watchId, evaluator);
const watch = dbg.inspector.getWatch(watchId);
console.log(watch.change_count); // Number of changes
```

### Step Execution

```typescript
// Step into function
dbg.stepper.stepInto('app.ts', 42);

// Step over line
dbg.stepper.stepOver('app.ts', 43);

// Step out of function
dbg.stepper.stepOut('app.ts', 44, currentDepth);

// Continue until breakpoint
dbg.stepper.continue('app.ts', 45);

// Check if should stop
const result = dbg.stepper.shouldStop('app.ts', 46, depth);
if (result.stopped) {
  console.log(`Stopped: ${result.reason}`);
}
```

### Replay Debugging

```typescript
// Start recording
dbg.recorder.startRecording();

// Execute code... each step is recorded
dbg.recorder.record('app.ts', 10, 0);
dbg.recorder.record('utils.ts', 20, 1);

// Stop recording
dbg.recorder.stopRecording();

// Replay execution
dbg.recorder.startReplay();

let step = dbg.recorder.nextReplayStep();
while (step) {
  console.log(`${step.file}:${step.line} (depth ${step.depth})`);
  step = dbg.recorder.nextReplayStep();
}
```

### Stack Traces

```typescript
// Push frame
dbg.stackTrace.push('main', 'app.ts', 1, 0);
dbg.stackTrace.push('processData', 'utils.ts', 10, 0, { data: [...] });
dbg.stackTrace.push('transform', 'utils.ts', 20, 0, {}, true); // async

// Get call stack
const callStack = dbg.stackTrace.getCallStack();
callStack.frames.forEach(frame => {
  console.log(`${frame.function_name} at ${frame.file}:${frame.line}`);
});

// Format for display
console.log(dbg.stackTrace.formatStackTrace());

// Record error
try {
  throw new Error('Division by zero');
} catch (error) {
  dbg.stackTrace.recordError(error as Error);
}
```

### Source Maps

```typescript
// Register source map
dbg.sourceMap.registerSourceMap('bundle.js', sourceMapObject);

// Resolve location
const location = dbg.sourceMap.resolve('bundle.js', 100, 5);
console.log(`Original: ${location.original_file}:${location.original_line}`);
```

### Scope Inspection

```typescript
// Set multiple scopes
dbg.scopeInspector.setScope('local', { x: 10 });
dbg.scopeInspector.setScope('closure', { capturedVar: 'hello' });
dbg.scopeInspector.setScope('global', { VERSION: '1.0.0' });

// Search across scopes (local -> closure -> module -> global)
const variable = dbg.scopeInspector.getVariable('capturedVar');
console.log(variable.scope); // 'closure'

// Get scope summary
const summary = dbg.scopeInspector.getSummary();
console.log(summary); // { local: 1, closure: 1, global: 1, module: 0 }
```

### Demo

```bash
npx ts-node src/grammar-lang/tools/gdebug/demo.ts
```

**Output:**
```
✅ Breakpoint Management: 3 breakpoints (1 conditional, 1 hit)
✅ Variable Inspection: 4 variables inspected with types
✅ Step Execution: 4 steps recorded + replay
✅ Stack Trace: 3 frames with async support
✅ Source Map: Resolved bundle.js → app.ts
✅ Scope Inspector: 5 variables in 4 scopes
```

---

## 📊 Performance Characteristics

### GBench
- **Benchmark Registration**: O(1) via hash map
- **Execution**: O(n) where n = iterations
- **Statistical Calculations**: O(n log n) for percentiles
- **Memory Tracking**: O(1) via Node.js API

### GDebug
- **Breakpoint Operations**: O(1) via hash indexing
- **Variable Lookup**: O(1) via Map
- **Stack Push/Pop**: O(1) array operations
- **Step Recording**: O(1) append, O(n) replay where n = steps

---

## 🧪 Testing

Unit tests are provided for both GBench and GDebug:

```
gbench/__tests__/
  ├── suite.test.ts      - Suite management tests
  └── compare.test.ts    - Comparison & regression tests

gdebug/__tests__/
  ├── breakpoints.test.ts - Breakpoint system tests
  └── inspector.test.ts   - Variable inspection tests
```

To run tests (requires Jest/Vitest):
```bash
npm test
```

---

## 📁 Project Structure

```
tools/
├── gbench/
│   ├── suite.ts          - Benchmark suite management (366 lines)
│   ├── metrics.ts        - Performance metrics (368 lines)
│   ├── compare.ts        - Comparison & regression (363 lines)
│   ├── report.ts         - Multi-format reporting (463 lines)
│   ├── index.ts          - Public API (59 lines)
│   ├── demo.ts           - Demo (193 lines)
│   └── __tests__/        - Unit tests
│
├── gdebug/
│   ├── breakpoints.ts    - Breakpoint management (366 lines)
│   ├── inspector.ts      - Variable inspection (371 lines)
│   ├── stepper.ts        - Step execution (392 lines)
│   ├── trace.ts          - Stack traces (445 lines)
│   ├── index.ts          - Public API (139 lines)
│   ├── demo.ts           - Demo (497 lines)
│   └── __tests__/        - Unit tests
│
└── README.md             - This file
```

**Total:** 12 implementation files, ~3,610 lines of production code

---

## 🎯 Design Philosophy

Both tools follow these principles:

1. **O(1) Operations**: Hash-based data structures for constant-time lookups
2. **Zero Dependencies**: Self-contained TypeScript implementations
3. **Type Safety**: Comprehensive TypeScript types throughout
4. **GVCS-Inspired**: Concepts from Genetic Version Control System
5. **Production-Ready**: Error handling, edge cases, memory limits

---

## 🚀 Real-World Results

### GBench Validation
```
Hash Map vs Array Lookup (10,000 items):
  HashMap:  0.000ms avg  (4,992,928 ops/sec)
  Array:    0.031ms avg  (32,592 ops/sec)

  Improvement: 153x faster (99.35% reduction)
```

### GDebug Validation
```
Breakpoint System:
  - 3 breakpoints created (O(1) each)
  - Conditional breakpoint: x > 10 ✓
  - Hit tracking: 1 hit recorded with timestamp

Step Execution:
  - 4 steps recorded in 0ms
  - Replay: 4/4 steps successful

Stack Traces:
  - 3 frames tracked (depth: 0 → 3 → 0)
  - Async chain created
  - Error trace captured
```

---

## 📝 License

Part of the Chomsky/Grammar Language project.

---

## 🤝 Contributing

These tools are production-ready and validated. Future enhancements:

- [ ] Integration with Grammar Language compiler
- [ ] CLI tools for standalone usage
- [ ] VS Code extension for visual debugging
- [ ] Performance regression CI/CD integration
- [ ] Flame graph generation for benchmarks

---

## 📚 API Reference

### GBench

**Suite Management**
- `suite(name: string): GBenchSuite` - Create/get benchmark suite
- `benchSuite.add(name, fn): this` - Add benchmark (O(1))
- `benchSuite.remove(name): boolean` - Remove benchmark (O(1))
- `benchSuite.runBenchmark(name, config): Promise<BenchmarkResult>` - Run single benchmark
- `benchSuite.runAll(config): Promise<Map<string, BenchmarkResult>>` - Run all benchmarks

**Comparison**
- `compare(name1, result1, name2, result2): ComparisonResult` - Compare two results
- `createRegressionDetector(threshold): RegressionDetector` - Create regression detector

**Reporting**
- `report(result: BenchmarkResult): void` - Print result to console
- `reportComparison(comparison: ComparisonResult): void` - Print comparison
- `BenchmarkExporter.export(results, format, path): void` - Export results

### GDebug

**Breakpoints**
- `createBreakpointManager(): BreakpointManager` - Create manager
- `manager.add(file, line, options): Breakpoint` - Add breakpoint (O(1))
- `manager.shouldBreak(file, line, locals): Breakpoint | null` - Check breakpoint (O(1))
- `manager.recordHit(id, locals, depth): BreakpointHit` - Record hit (O(1))

**Inspection**
- `createInspector(): VariableInspector` - Create inspector
- `inspector.getVariable(name): Variable | null` - Get variable (O(1))
- `inspector.inspect(name): InspectionResult | null` - Deep inspect (O(k) for k properties)
- `inspector.addWatch(expr, evaluator): string` - Add watch (O(1))

**Stepping**
- `createStepController(): StepController` - Create controller
- `stepper.stepOver/stepInto/stepOut/continue(...)` - Step operations
- `stepper.shouldStop(file, line, depth): StepResult` - Check if should stop (O(1))

**Stack Traces**
- `createStackTraceManager(maxDepth): StackTraceManager` - Create manager
- `stackTrace.push(name, file, line, ...): StackFrame` - Push frame (O(1))
- `stackTrace.pop(): StackFrame | null` - Pop frame (O(1))
- `stackTrace.formatStackTrace(): string` - Format for display

---

**Built with ❤️ for Grammar Language**
