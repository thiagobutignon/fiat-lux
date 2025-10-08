/**
 * Test Runner for Observability
 * Simple, fast test execution
 */

import { Observability, LogLevel } from '../core/observability';

let passed = 0;
let failed = 0;

function test(name: string, fn: () => void) {
  try {
    fn();
    console.log(`✅ ${name}`);
    passed++;
  } catch (error: any) {
    console.log(`❌ ${name}`);
    console.log(`   ${error.message}`);
    failed++;
  }
}

function assert(condition: boolean, message: string) {
  if (!condition) {
    throw new Error(message);
  }
}

console.log('🧪 Testing Observability Layer\n');

// Test 1: Basic logging
test('should log structured events', () => {
  const obs = new Observability(LogLevel.INFO);
  obs.log(LogLevel.INFO, 'test_event', { foo: 'bar' });

  const logs = obs.getLogs();
  assert(logs.length === 1, 'Should have 1 log entry');
  assert(logs[0].level === LogLevel.INFO, 'Log level should be INFO');
  assert(logs[0].event === 'test_event', 'Event name should match');
  assert(logs[0].data.foo === 'bar', 'Data should match');
});

// Test 2: Log level filtering
test('should respect log level filtering', () => {
  const obs = new Observability(LogLevel.WARN);

  obs.log(LogLevel.DEBUG, 'debug_event', {});
  obs.log(LogLevel.INFO, 'info_event', {});
  obs.log(LogLevel.WARN, 'warn_event', {});
  obs.log(LogLevel.ERROR, 'error_event', {});

  const logs = obs.getLogs();
  assert(logs.length === 2, 'Should have 2 logs (WARN and ERROR)');
  assert(logs[0].level === LogLevel.WARN, 'First should be WARN');
  assert(logs[1].level === LogLevel.ERROR, 'Second should be ERROR');
});

// Test 3: Metrics
test('should record metrics', () => {
  const obs = new Observability();

  obs.metric('request_count', 10);
  obs.metric('response_time_ms', 250.5);

  const metrics = obs.getMetrics();
  assert(metrics.length === 2, 'Should have 2 metrics');
  assert(metrics[0].name === 'request_count', 'First metric name should match');
  assert(metrics[0].value === 10, 'First metric value should match');
  assert(metrics[1].value === 250.5, 'Second metric value should match');
});

// Test 4: Metrics with tags
test('should support metric tags', () => {
  const obs = new Observability();

  obs.metric('http_requests', 1, { method: 'GET', status: '200' });

  const metrics = obs.getMetrics();
  assert(metrics[0].tags !== undefined, 'Tags should exist');
  assert(metrics[0].tags!.method === 'GET', 'Method tag should match');
  assert(metrics[0].tags!.status === '200', 'Status tag should match');
});

// Test 5: Spans
test('should create and track spans', () => {
  const obs = new Observability();

  const span = obs.startSpan('test_operation');
  assert(span.name === 'test_operation', 'Span name should match');

  span.setTag('foo', 'bar');
  span.end();

  const spans = obs.getSpans();
  assert(spans.length === 1, 'Should have 1 span');
  assert(spans[0].tags.foo === 'bar', 'Tag should be set');
  assert(spans[0].duration_ms !== undefined, 'Duration should be set');
  assert(spans[0].duration_ms! >= 0, 'Duration should be non-negative');
});

// Test 6: Nested spans
test('should support nested spans', () => {
  const obs = new Observability();

  const parent = obs.startSpan('parent');
  const child1 = obs.startSpan('child1', parent);
  const child2 = obs.startSpan('child2', parent);

  child1.end();
  child2.end();
  parent.end();

  const spans = obs.getSpans();
  assert(spans.length === 3, 'Should have 3 spans');
  assert(spans[1].parent_id === parent.id, 'Child1 should have parent');
  assert(spans[2].parent_id === parent.id, 'Child2 should have parent');
});

// Test 7: Error logging
test('should log errors with stack traces', () => {
  const obs = new Observability();

  const error = new Error('Test error');
  obs.error(error, { context: 'test' });

  const logs = obs.getLogs();
  assert(logs.length === 1, 'Should have 1 log');
  assert(logs[0].level === LogLevel.ERROR, 'Should be ERROR level');
  assert(logs[0].data.message === 'Test error', 'Error message should match');
  assert(logs[0].data.stack !== undefined, 'Stack trace should exist');
});

// Test 8: Clear
test('should clear all data', () => {
  const obs = new Observability();

  obs.log(LogLevel.INFO, 'test', {});
  obs.metric('test', 1);
  const span = obs.startSpan('test');
  span.end();

  assert(obs.getLogs().length === 1, 'Should have logs');
  assert(obs.getMetrics().length === 1, 'Should have metrics');
  assert(obs.getSpans().length === 1, 'Should have spans');

  obs.clear();

  assert(obs.getLogs().length === 0, 'Logs should be cleared');
  assert(obs.getMetrics().length === 0, 'Metrics should be cleared');
  assert(obs.getSpans().length === 0, 'Spans should be cleared');
});

// Test 9: Statistics
test('should return statistics', () => {
  const obs = new Observability(LogLevel.DEBUG); // Fresh instance

  obs.log(LogLevel.INFO, 'event1', {});
  obs.log(LogLevel.ERROR, 'event2', {});
  obs.metric('metric1', 10);
  const span = obs.startSpan('span1');
  span.end();

  const stats = obs.getStats();
  assert(stats.total_logs === 2, `Should count 2 logs, got ${stats.total_logs}`);
  assert(stats.total_metrics === 1, 'Should count metrics');
  assert(stats.total_spans === 1, 'Should count spans');
  assert(stats.log_levels[LogLevel.INFO] === 1, `Should count 1 INFO log, got ${stats.log_levels[LogLevel.INFO]}`);
  assert(stats.log_levels[LogLevel.ERROR] === 1, 'Should count ERROR logs');
});

// Test 10: Metric statistics
test('should calculate metric statistics', () => {
  const obs = new Observability();

  obs.metric('response_time', 100);
  obs.metric('response_time', 200);
  obs.metric('response_time', 300);

  const stats = obs.getMetricStats('response_time');
  assert(stats !== null, 'Stats should exist');
  assert(stats!.count === 3, 'Should count 3 metrics');
  assert(stats!.sum === 600, 'Sum should be 600');
  assert(stats!.avg === 200, 'Average should be 200');
  assert(stats!.min === 100, 'Min should be 100');
  assert(stats!.max === 300, 'Max should be 300');
});

// Summary
console.log('\n' + '='.repeat(70));
console.log(`Total: ${passed + failed}`);
console.log(`✅ Passed: ${passed}`);
console.log(`❌ Failed: ${failed}`);
console.log('='.repeat(70));

if (failed > 0) {
  process.exit(1);
}
