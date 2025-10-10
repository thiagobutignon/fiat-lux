/**
 * GDebug Demo - Debugging Toolkit
 *
 * Demonstrates O(1) debugging capabilities for Grammar Language.
 */

import {
  createDebugger,
  BreakpointManager,
  VariableInspector,
  StepController,
  StackTraceManager,
  features
} from './index';

// ============================================================================
// Demo Program (Target for Debugging)
// ============================================================================

class Calculator {
  private history: number[] = [];

  add(a: number, b: number): number {
    const result = a + b;
    this.history.push(result);
    return result;
  }

  multiply(a: number, b: number): number {
    const result = a * b;
    this.history.push(result);
    return result;
  }

  fibonacci(n: number): number {
    if (n <= 1) return n;
    return this.fibonacci(n - 1) + this.fibonacci(n - 2);
  }

  async asyncCompute(a: number, b: number): Promise<number> {
    await this.delay(10);
    const sum = this.add(a, b);
    await this.delay(10);
    const product = this.multiply(a, b);
    return sum + product;
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// ============================================================================
// Main Demo
// ============================================================================

async function main() {
  console.log('🐛 GDebug Demo - O(1) Debugging Toolkit\n');
  console.log('═'.repeat(70));

  // Create debugger instance
  const dbg = createDebugger({ maxStackDepth: 50 });

  console.log('\n✨ GDebug Features:');
  features().forEach((feature, i) => {
    console.log(`  ${i + 1}. ${feature}`);
  });

  // =========================================================================
  // Demo 1: Breakpoint Management
  // =========================================================================

  console.log('\n📊 Demo 1: Breakpoint Management');
  console.log('─'.repeat(70));

  const bp = dbg.breakpoints;

  // Add simple breakpoint
  const bp1 = bp.add('calculator.ts', 10);
  console.log(`✅ Added breakpoint: ${bp1.id}`);

  // Add conditional breakpoint
  const bp2 = bp.add('calculator.ts', 15, {
    condition: (locals: Record<string, any>) => locals.a > 10,
    log_message: 'a is greater than 10'
  });
  console.log(`✅ Added conditional breakpoint: ${bp2.id}`);

  // Add breakpoint with max hits
  const bp3 = bp.add('calculator.ts', 20, { max_hits: 3 });
  console.log(`✅ Added breakpoint with max hits: ${bp3.id}`);

  // Test breakpoint hit
  const shouldBreak1 = bp.shouldBreak('calculator.ts', 10);
  console.log(`\n🔍 Should break at calculator.ts:10? ${shouldBreak1 ? '✓ YES' : '✗ NO'}`);

  if (shouldBreak1) {
    const hit = bp.recordHit(shouldBreak1.id, { a: 5, b: 3 }, 1);
    console.log(`   Hit recorded at ${new Date(hit.timestamp).toISOString()}`);
  }

  // Test conditional breakpoint
  const shouldBreak2 = bp.shouldBreak('calculator.ts', 15, { a: 5, b: 3 });
  console.log(`🔍 Should break at calculator.ts:15 (a=5)? ${shouldBreak2 ? '✓ YES' : '✗ NO'}`);

  const shouldBreak3 = bp.shouldBreak('calculator.ts', 15, { a: 15, b: 3 });
  console.log(`🔍 Should break at calculator.ts:15 (a=15)? ${shouldBreak3 ? '✓ YES' : '✗ NO'}`);

  // Display breakpoint stats
  const stats = bp.getStats();
  console.log(`\n📈 Breakpoint Stats:`);
  console.log(`   Total: ${stats.total}`);
  console.log(`   Enabled: ${stats.enabled}`);
  console.log(`   With conditions: ${stats.with_conditions}`);
  console.log(`   Total hits: ${stats.total_hits}`);

  // =========================================================================
  // Demo 2: Variable Inspection
  // =========================================================================

  console.log('\n📊 Demo 2: Variable Inspection');
  console.log('─'.repeat(70));

  const inspector = dbg.inspector;

  // Set local variables
  inspector.setLocals({
    x: 42,
    name: 'Chomsky',
    config: { debug: true, verbose: false },
    items: [1, 2, 3, 4, 5]
  });

  // Inspect primitive
  const xVar = inspector.getVariable('x');
  if (xVar) {
    console.log(`\n🔍 Variable 'x':`);
    console.log(`   Value: ${xVar.value}`);
    console.log(`   Type: ${xVar.type}`);
    console.log(`   Size: ${xVar.size_bytes} bytes`);
  }

  // Deep inspect object
  const configInspection = inspector.inspect('config');
  if (configInspection) {
    console.log(`\n🔍 Deep inspection of 'config':`);
    console.log(`   Type: ${configInspection.variable.type}`);
    console.log(`   Properties:`, configInspection.properties);
    console.log(`   Constructor: ${configInspection.constructor_name}`);
  }

  // Add watch expression
  const watchId1 = inspector.addWatch('x * 2', () => {
    const x = inspector.getVariable('x');
    return x ? x.value * 2 : null;
  });
  console.log(`\n👁️  Added watch expression: 'x * 2'`);

  const watch1 = inspector.getWatch(watchId1);
  if (watch1) {
    console.log(`   Value: ${watch1.value}`);
    console.log(`   Type: ${watch1.type}`);
    console.log(`   Changes: ${watch1.change_count}`);
  }

  // Track value changes
  inspector.trackValue('x', 42);
  inspector.trackValue('x', 43);
  inspector.trackValue('x', 44);

  const history = inspector.getValueHistory('x');
  console.log(`\n📜 Value history for 'x': ${history.length} entries`);

  // Get all variables
  const allVars = inspector.getAllVariables();
  console.log(`\n📋 All variables (${allVars.length}):`);
  allVars.forEach((v: any) => {
    console.log(`   ${v.name}: ${v.type} = ${JSON.stringify(v.value)}`);
  });

  // =========================================================================
  // Demo 3: Step Execution
  // =========================================================================

  console.log('\n📊 Demo 3: Step Execution');
  console.log('─'.repeat(70));

  const stepper = dbg.stepper;
  const recorder = dbg.recorder;

  // Start recording
  recorder.startRecording();

  // Simulate execution with stepping
  console.log('\n🎬 Simulating program execution...\n');

  // Step 1: main() entry
  stepper.stepInto('main.ts', 1);
  recorder.record('main.ts', 1, 0);
  console.log('  1. main.ts:1 (depth 0) - main() entry');

  const step1 = stepper.shouldStop('main.ts', 1, 0);
  if (step1.stopped) {
    console.log(`     ⏸️  STOPPED: ${step1.reason}`);
  }

  // Step 2: function call
  stepper.stepInto('main.ts', 5);
  recorder.record('calculator.ts', 10, 1, { a: 5, b: 3 });
  console.log('  2. calculator.ts:10 (depth 1) - add(5, 3)');

  const step2 = stepper.shouldStop('calculator.ts', 10, 1);
  if (step2.stopped) {
    console.log(`     ⏸️  STOPPED: ${step2.reason}`);
  }

  // Step 3: step over
  stepper.stepOver('calculator.ts', 10);
  recorder.record('calculator.ts', 11, 1);
  console.log('  3. calculator.ts:11 (depth 1) - next line');

  const step3 = stepper.shouldStop('calculator.ts', 11, 1);
  if (step3.stopped) {
    console.log(`     ⏸️  STOPPED: ${step3.reason}`);
  }

  // Step 4: step out
  stepper.stepOut('calculator.ts', 12, 1);
  recorder.record('main.ts', 6, 0);
  console.log('  4. main.ts:6 (depth 0) - return to main');

  const step4 = stepper.shouldStop('main.ts', 6, 0);
  if (step4.stopped) {
    console.log(`     ⏸️  STOPPED: ${step4.reason}`);
  }

  // Stop recording and show stats
  recorder.stopRecording();

  const stats2 = recorder.getStats();
  console.log(`\n📈 Execution Stats:`);
  console.log(`   Total steps: ${stats2.total_steps}`);
  console.log(`   Duration: ${stats2.duration_ms}ms`);
  console.log(`   Recording: ${stats2.recording ? 'ON' : 'OFF'}`);

  // Replay demonstration
  console.log(`\n⏮️  Replay Mode:`);
  recorder.startReplay();

  let step = recorder.nextReplayStep();
  let stepNum = 1;
  while (step) {
    console.log(`   Step ${stepNum}: ${step.file}:${step.line} (depth ${step.depth})`);
    step = recorder.nextReplayStep();
    stepNum++;
  }

  // =========================================================================
  // Demo 4: Stack Trace Management
  // =========================================================================

  console.log('\n📊 Demo 4: Stack Trace Management');
  console.log('─'.repeat(70));

  const stackTrace = dbg.stackTrace;

  // Simulate nested function calls
  console.log('\n🎬 Simulating nested calls...\n');

  const frame1 = stackTrace.push('main', 'main.ts', 1, 0);
  console.log(`  1. Pushed: ${frame1.function_name} at ${frame1.file}:${frame1.line}`);

  const frame2 = stackTrace.push('processData', 'utils.ts', 10, 0, { data: [1, 2, 3] });
  console.log(`  2. Pushed: ${frame2.function_name} at ${frame2.file}:${frame2.line}`);

  const frame3 = stackTrace.push('transform', 'utils.ts', 20, 0, {}, true); // async
  console.log(`  3. Pushed: ${frame3.function_name} at ${frame3.file}:${frame3.line} [async]`);

  // Get current frame
  const current = stackTrace.current();
  if (current) {
    console.log(`\n📍 Current frame: ${current.function_name} (depth ${current.depth})`);
  }

  // Get call stack
  const callStack = stackTrace.getCallStack();
  console.log(`\n📚 Call Stack (${callStack.current_depth}/${callStack.max_depth}):`);
  callStack.frames.forEach((frame: any, i: number) => {
    const indent = '  '.repeat(i);
    const asyncTag = frame.is_async ? ' [async]' : '';
    console.log(`${indent}${i + 1}. ${frame.function_name} (${frame.file}:${frame.line})${asyncTag}`);
  });

  // Format stack trace
  console.log('\n📋 Formatted Stack Trace:');
  console.log(stackTrace.formatStackTrace());

  // Create async chain
  const chainId = stackTrace.createAsyncChain();
  console.log(`\n⛓️  Created async chain: ${chainId}`);

  // Simulate error
  const error = new Error('Division by zero');
  stackTrace.recordError(error);

  const lastError = stackTrace.getLastError();
  if (lastError) {
    console.log(`\n❌ Last Error:`);
    console.log(`   Message: ${lastError.error.message}`);
    console.log(`   Stack depth: ${lastError.stack.length}`);
    console.log(`   Timestamp: ${new Date(lastError.timestamp).toISOString()}`);
  }

  // Pop frames
  console.log(`\n⬇️  Popping frames...`);
  let poppedFrame = stackTrace.pop();
  while (poppedFrame) {
    console.log(`   Popped: ${poppedFrame.function_name}`);
    poppedFrame = stackTrace.pop();
  }

  console.log(`\n📊 Final depth: ${stackTrace.getDepth()}`);

  // =========================================================================
  // Demo 5: Source Map Resolution
  // =========================================================================

  console.log('\n📊 Demo 5: Source Map Support');
  console.log('─'.repeat(70));

  const sourceMap = dbg.sourceMap;
  const frameFilter = dbg.frameFilter;

  // Register mock source map
  sourceMap.registerSourceMap('calculator.js', {
    version: 3,
    sources: ['calculator.ts'],
    mappings: 'AAAA,CAAC'
  });
  console.log(`\n🗺️  Registered source map: calculator.js -> calculator.ts`);

  // Resolve location
  const location = sourceMap.resolve('calculator.js', 10, 5);
  if (location) {
    console.log(`\n📍 Resolved location:`);
    console.log(`   Generated: ${location.generated_file}:${location.generated_line}:${location.generated_column}`);
    console.log(`   Original:  ${location.original_file}:${location.original_line}:${location.original_column}`);
  }

  // Frame filtering
  frameFilter.addFilter('node_modules');
  frameFilter.addFilter('internal');
  console.log(`\n🔍 Added frame filters: node_modules, internal`);

  const testFrames = [
    { id: '1', function_name: 'myFunc', file: 'app.ts', line: 10, column: 0, depth: 0, timestamp: Date.now() },
    { id: '2', function_name: 'require', file: 'node_modules/module.js', line: 50, column: 0, depth: 1, timestamp: Date.now() },
    { id: '3', function_name: 'internal', file: 'internal/process.js', line: 100, column: 0, depth: 2, timestamp: Date.now() }
  ];

  const filtered = frameFilter.filter(testFrames);
  console.log(`\n📋 Filtered frames (${filtered.length}/${testFrames.length}):`);
  filtered.forEach((f: any) => {
    console.log(`   ${f.function_name} (${f.file}:${f.line})`);
  });

  // =========================================================================
  // Demo 6: Scope Inspector
  // =========================================================================

  console.log('\n📊 Demo 6: Scope Inspector');
  console.log('─'.repeat(70));

  const scopeInspector = dbg.scopeInspector;

  // Set different scopes
  scopeInspector.setScope('local', { x: 10, y: 20 });
  scopeInspector.setScope('closure', { capturedVar: 'hello' });
  scopeInspector.setScope('global', { globalConfig: { version: '1.0.0' } });
  scopeInspector.setScope('module', { MODULE_NAME: 'Calculator' });

  console.log(`\n🔍 Scope Summary:`);
  const summary = scopeInspector.getSummary();
  Object.entries(summary).forEach(([scope, count]) => {
    console.log(`   ${scope}: ${count} variable(s)`);
  });

  // Search for variable across scopes
  const foundVar = scopeInspector.getVariable('x');
  if (foundVar) {
    console.log(`\n📍 Found 'x' in scope: ${foundVar.scope}`);
    console.log(`   Value: ${foundVar.value}`);
    console.log(`   Type: ${foundVar.type}`);
    console.log(`   Writable: ${foundVar.writable}`);
  }

  const closureVar = scopeInspector.getVariable('capturedVar');
  if (closureVar) {
    console.log(`\n📍 Found 'capturedVar' in scope: ${closureVar.scope}`);
    console.log(`   Value: ${closureVar.value}`);
  }

  // =========================================================================
  // Summary
  // =========================================================================

  console.log('\n═'.repeat(70));
  console.log('✅ GDebug Demo Complete!');
  console.log('\nKey Features Demonstrated:');
  console.log('  ✅ O(1) breakpoint management (add/remove/check)');
  console.log('  ✅ Conditional breakpoints with hit tracking');
  console.log('  ✅ Deep variable inspection with type info');
  console.log('  ✅ Watch expressions with change detection');
  console.log('  ✅ Step execution (over/into/out/continue)');
  console.log('  ✅ Replay debugging');
  console.log('  ✅ Stack trace management with async support');
  console.log('  ✅ Error trace recording');
  console.log('  ✅ Source map resolution');
  console.log('  ✅ Frame filtering');
  console.log('  ✅ Multi-scope variable inspection');
  console.log('\n🚀 Ready for production debugging!');
  console.log('═'.repeat(70) + '\n');
}

// Run demo
if (require.main === module) {
  main().catch(console.error);
}

export { main };
