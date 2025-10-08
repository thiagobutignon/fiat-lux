/**
 * Observability Tests (TDD: Red Phase)
 *
 * Tests for structured logging, metrics, and tracing system.
 */

import { Observability, LogLevel, LogEntry, Metric, Span } from '../core/observability';

// ============================================================================
// Test Helpers
// ============================================================================

function mockConsole() {
  const originalLog = console.log;
  const originalError = console.error;
  const logs: string[] = [];

  console.log = (...args: any[]) => {
    logs.push(args.map(String).join(' '));
  };

  console.error = (...args: any[]) => {
    logs.push(args.map(String).join(' '));
  };

  return {
    logs,
    restore: () => {
      console.log = originalLog;
      console.error = originalError;
    },
  };
}

// ============================================================================
// Tests
// ============================================================================

describe('Observability', () => {
  describe('log', () => {
    it('should log structured events', () => {
      const obs = new Observability(LogLevel.INFO);
      const mock = mockConsole();

      obs.log(LogLevel.INFO, 'test_event', { foo: 'bar' });

      const logs = obs.getLogs();
      expect(logs.length).toBe(1);
      expect(logs[0].level).toBe(LogLevel.INFO);
      expect(logs[0].event).toBe('test_event');
      expect(logs[0].data.foo).toBe('bar');
      expect(logs[0].timestamp).toBeGreaterThan(0);

      mock.restore();
    });

    it('should respect log level filtering', () => {
      const obs = new Observability(LogLevel.WARN);
      const mock = mockConsole();

      obs.log(LogLevel.DEBUG, 'debug_event', {});
      obs.log(LogLevel.INFO, 'info_event', {});
      obs.log(LogLevel.WARN, 'warn_event', {});
      obs.log(LogLevel.ERROR, 'error_event', {});

      const logs = obs.getLogs();
      expect(logs.length).toBe(2); // Only WARN and ERROR
      expect(logs[0].level).toBe(LogLevel.WARN);
      expect(logs[1].level).toBe(LogLevel.ERROR);

      mock.restore();
    });

    it('should format log entries as JSON', () => {
      const obs = new Observability(LogLevel.INFO);
      const mock = mockConsole();

      obs.log(LogLevel.INFO, 'test', { value: 123 });

      expect(mock.logs.length).toBeGreaterThan(0);
      const parsed = JSON.parse(mock.logs[0]);
      expect(parsed.event).toBe('test');
      expect(parsed.data.value).toBe(123);

      mock.restore();
    });
  });

  describe('metric', () => {
    it('should record metrics with name and value', () => {
      const obs = new Observability();

      obs.metric('request_count', 10);
      obs.metric('response_time_ms', 250.5);

      const metrics = obs.getMetrics();
      expect(metrics.length).toBe(2);
      expect(metrics[0].name).toBe('request_count');
      expect(metrics[0].value).toBe(10);
      expect(metrics[1].name).toBe('response_time_ms');
      expect(metrics[1].value).toBe(250.5);
    });

    it('should support tags for metrics', () => {
      const obs = new Observability();

      obs.metric('http_requests', 1, { method: 'GET', status: '200' });

      const metrics = obs.getMetrics();
      expect(metrics[0].tags).toEqual({ method: 'GET', status: '200' });
    });

    it('should timestamp metrics', () => {
      const obs = new Observability();

      const before = Date.now();
      obs.metric('test_metric', 42);
      const after = Date.now();

      const metrics = obs.getMetrics();
      expect(metrics[0].timestamp).toBeGreaterThanOrEqual(before);
      expect(metrics[0].timestamp).toBeLessThanOrEqual(after);
    });
  });

  describe('startSpan', () => {
    it('should create a span with name', () => {
      const obs = new Observability();

      const span = obs.startSpan('test_operation');

      expect(span).toBeDefined();
      expect(span.name).toBe('test_operation');
    });

    it('should track span duration', async () => {
      const obs = new Observability();

      const span = obs.startSpan('async_operation');
      await new Promise((resolve) => setTimeout(resolve, 50));
      span.end();

      const spans = obs.getSpans();
      expect(spans.length).toBe(1);
      expect(spans[0].duration_ms).toBeGreaterThanOrEqual(50);
      expect(spans[0].duration_ms).toBeLessThan(100);
    });

    it('should support span tags', () => {
      const obs = new Observability();

      const span = obs.startSpan('database_query');
      span.setTag('query', 'SELECT * FROM users');
      span.setTag('rows', 42);
      span.end();

      const spans = obs.getSpans();
      expect(spans[0].tags.query).toBe('SELECT * FROM users');
      expect(spans[0].tags.rows).toBe(42);
    });

    it('should handle nested spans', () => {
      const obs = new Observability();

      const parent = obs.startSpan('parent_operation');
      const child1 = obs.startSpan('child_1', parent);
      const child2 = obs.startSpan('child_2', parent);

      child1.end();
      child2.end();
      parent.end();

      const spans = obs.getSpans();
      expect(spans.length).toBe(3);
      expect(spans[1].parent_id).toBe(parent.id);
      expect(spans[2].parent_id).toBe(parent.id);
    });
  });

  describe('error', () => {
    it('should log errors with stack traces', () => {
      const obs = new Observability();
      const mock = mockConsole();

      const error = new Error('Test error');
      obs.error(error, { context: 'test' });

      const logs = obs.getLogs();
      expect(logs.length).toBe(1);
      expect(logs[0].level).toBe(LogLevel.ERROR);
      expect(logs[0].event).toBe('error');
      expect(logs[0].data.message).toBe('Test error');
      expect(logs[0].data.stack).toBeDefined();
      expect(logs[0].data.context.context).toBe('test');

      mock.restore();
    });

    it('should handle errors without stack traces', () => {
      const obs = new Observability();
      const mock = mockConsole();

      const error = { message: 'Custom error' } as Error;
      obs.error(error, {});

      const logs = obs.getLogs();
      expect(logs[0].data.message).toBe('Custom error');
      expect(logs[0].data.stack).toBeUndefined();

      mock.restore();
    });
  });

  describe('clear', () => {
    it('should clear all logs, metrics, and spans', () => {
      const obs = new Observability();

      obs.log(LogLevel.INFO, 'test', {});
      obs.metric('test', 1);
      const span = obs.startSpan('test');
      span.end();

      expect(obs.getLogs().length).toBe(1);
      expect(obs.getMetrics().length).toBe(1);
      expect(obs.getSpans().length).toBe(1);

      obs.clear();

      expect(obs.getLogs().length).toBe(0);
      expect(obs.getMetrics().length).toBe(0);
      expect(obs.getSpans().length).toBe(0);
    });
  });

  describe('getStats', () => {
    it('should return statistics about collected data', () => {
      const obs = new Observability();

      obs.log(LogLevel.INFO, 'event1', {});
      obs.log(LogLevel.ERROR, 'event2', {});
      obs.metric('metric1', 10);
      const span = obs.startSpan('span1');
      span.end();

      const stats = obs.getStats();
      expect(stats.total_logs).toBe(2);
      expect(stats.total_metrics).toBe(1);
      expect(stats.total_spans).toBe(1);
      expect(stats.log_levels.INFO).toBe(1);
      expect(stats.log_levels.ERROR).toBe(1);
    });
  });
});

// ============================================================================
// Test Utilities (minimal test framework)
// ============================================================================

function describe(name: string, fn: () => void) {
  console.log(`\n📦 ${name}`);
  fn();
}

function it(name: string, fn: () => void | Promise<void>) {
  try {
    const result = fn();
    if (result instanceof Promise) {
      result
        .then(() => console.log(`  ✅ ${name}`))
        .catch((error) => {
          console.error(`  ❌ ${name}`);
          console.error(`     ${error.message}`);
        });
    } else {
      console.log(`  ✅ ${name}`);
    }
  } catch (error: any) {
    console.error(`  ❌ ${name}`);
    console.error(`     ${error.message}`);
  }
}

const expect = (value: any) => ({
  toBe: (expected: any) => {
    if (value !== expected) {
      throw new Error(`Expected ${value} to be ${expected}`);
    }
  },
  toEqual: (expected: any) => {
    if (JSON.stringify(value) !== JSON.stringify(expected)) {
      throw new Error(
        `Expected ${JSON.stringify(value)} to equal ${JSON.stringify(expected)}`
      );
    }
  },
  toBeDefined: () => {
    if (value === undefined) {
      throw new Error('Expected value to be defined');
    }
  },
  toBeUndefined: () => {
    if (value !== undefined) {
      throw new Error('Expected value to be undefined');
    }
  },
  toBeGreaterThan: (expected: number) => {
    if (value <= expected) {
      throw new Error(`Expected ${value} to be greater than ${expected}`);
    }
  },
  toBeGreaterThanOrEqual: (expected: number) => {
    if (value < expected) {
      throw new Error(`Expected ${value} to be greater than or equal to ${expected}`);
    }
  },
  toBeLessThan: (expected: number) => {
    if (value >= expected) {
      throw new Error(`Expected ${value} to be less than ${expected}`);
    }
  },
  toBeLessThanOrEqual: (expected: number) => {
    if (value > expected) {
      throw new Error(`Expected ${value} to be less than or equal to ${expected}`);
    }
  },
});
