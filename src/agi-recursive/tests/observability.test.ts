/**
 * @file observability.test.ts
 * Tests for Observability Layer
 *
 * Key capabilities tested:
 * - Structured logging with log levels
 * - Metrics collection and statistics
 * - Distributed tracing (spans)
 * - Export capabilities (JSON, CSV)
 * - Filtering by level, event, and metric name
 * - Error recording
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import {
  Observability,
  LogLevel,
  LogEntry,
  Metric,
  SpanData,
  createObservability,
} from '../core/observability';

describe('Observability', () => {
  let obs: Observability;
  let consoleSpy: any;

  beforeEach(() => {
    obs = new Observability(LogLevel.DEBUG);
    // Spy on console to suppress output during tests
    consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    consoleSpy.mockRestore();
  });

  describe('Logging', () => {
    it('should log events with DEBUG level', () => {
      obs.log(LogLevel.DEBUG, 'test_event', { key: 'value' });

      const logs = obs.getLogs();
      expect(logs.length).toBe(1);
      expect(logs[0].level).toBe(LogLevel.DEBUG);
      expect(logs[0].event).toBe('test_event');
      expect(logs[0].data).toEqual({ key: 'value' });
    });

    it('should log events with INFO level', () => {
      obs.log(LogLevel.INFO, 'info_event', { info: 'data' });

      const logs = obs.getLogs();
      expect(logs.length).toBe(1);
      expect(logs[0].level).toBe(LogLevel.INFO);
    });

    it('should log events with WARN level', () => {
      obs.log(LogLevel.WARN, 'warn_event', { warning: 'data' });

      const logs = obs.getLogs();
      expect(logs.length).toBe(1);
      expect(logs[0].level).toBe(LogLevel.WARN);
    });

    it('should log events with ERROR level', () => {
      obs.log(LogLevel.ERROR, 'error_event', { error: 'data' });

      const logs = obs.getLogs();
      expect(logs.length).toBe(1);
      expect(logs[0].level).toBe(LogLevel.ERROR);
    });

    it('should include timestamp in log entries', () => {
      const before = Date.now();
      obs.log(LogLevel.INFO, 'test', {});
      const after = Date.now();

      const logs = obs.getLogs();
      expect(logs[0].timestamp).toBeGreaterThanOrEqual(before);
      expect(logs[0].timestamp).toBeLessThanOrEqual(after);
    });

    it('should respect log level threshold', () => {
      const obsWarn = new Observability(LogLevel.WARN);

      obsWarn.log(LogLevel.DEBUG, 'debug', {});
      obsWarn.log(LogLevel.INFO, 'info', {});
      obsWarn.log(LogLevel.WARN, 'warn', {});
      obsWarn.log(LogLevel.ERROR, 'error', {});

      const logs = obsWarn.getLogs();
      expect(logs.length).toBe(2); // Only WARN and ERROR
      expect(logs[0].level).toBe(LogLevel.WARN);
      expect(logs[1].level).toBe(LogLevel.ERROR);
    });

    it('should output to console.log for non-error levels', () => {
      obs.log(LogLevel.INFO, 'test', { data: 'value' });

      expect(console.log).toHaveBeenCalled();
    });

    it('should output to console.error for ERROR level', () => {
      obs.log(LogLevel.ERROR, 'error_event', { error: 'details' });

      expect(console.error).toHaveBeenCalled();
    });
  });

  describe('Metrics', () => {
    it('should track metrics', () => {
      obs.metric('request_count', 42);

      const metrics = obs.getMetrics();
      expect(metrics.length).toBe(1);
      expect(metrics[0].name).toBe('request_count');
      expect(metrics[0].value).toBe(42);
    });

    it('should track metrics with tags', () => {
      obs.metric('response_time', 123, { endpoint: '/api/test', status: 200 });

      const metrics = obs.getMetrics();
      expect(metrics[0].tags).toEqual({ endpoint: '/api/test', status: 200 });
    });

    it('should include timestamp in metrics', () => {
      const before = Date.now();
      obs.metric('test_metric', 100);
      const after = Date.now();

      const metrics = obs.getMetrics();
      expect(metrics[0].timestamp).toBeGreaterThanOrEqual(before);
      expect(metrics[0].timestamp).toBeLessThanOrEqual(after);
    });

    it('should track multiple metrics', () => {
      obs.metric('metric1', 10);
      obs.metric('metric2', 20);
      obs.metric('metric1', 15);

      const metrics = obs.getMetrics();
      expect(metrics.length).toBe(3);
    });
  });

  describe('Tracing (Spans)', () => {
    it('should start and end span', () => {
      const span = obs.startSpan('test_operation');
      span.end();

      const spans = obs.getSpans();
      expect(spans.length).toBe(1);
      expect(spans[0].name).toBe('test_operation');
      expect(spans[0].end_time).toBeDefined();
      expect(spans[0].duration_ms).toBeDefined();
    });

    it('should calculate span duration', () => {
      const span = obs.startSpan('timed_operation');

      // Wait a bit
      const startTime = Date.now();
      while (Date.now() - startTime < 10) {
        // Busy wait for ~10ms
      }

      span.end();

      const spans = obs.getSpans();
      expect(spans[0].duration_ms).toBeGreaterThanOrEqual(10);
    });

    it('should set tags on span', () => {
      const span = obs.startSpan('operation');
      span.setTag('user_id', 123);
      span.setTag('success', true);
      span.end();

      const spans = obs.getSpans();
      expect(spans[0].tags.user_id).toBe(123);
      expect(spans[0].tags.success).toBe(true);
    });

    it('should support nested spans with parent', () => {
      const parent = obs.startSpan('parent_operation');
      const child = obs.startSpan('child_operation', parent);

      child.end();
      parent.end();

      const spans = obs.getSpans();
      expect(spans.length).toBe(2);
      expect(spans[1].parent_id).toBe(parent.id);
    });

    it('should generate unique span IDs', () => {
      const span1 = obs.startSpan('op1');
      const span2 = obs.startSpan('op2');

      expect(span1.id).not.toBe(span2.id);
    });

    it('should track start time for span', () => {
      const before = Date.now();
      const span = obs.startSpan('test');
      const after = Date.now();

      const spans = obs.getSpans();
      expect(spans[0].start_time).toBeGreaterThanOrEqual(before);
      expect(spans[0].start_time).toBeLessThanOrEqual(after);
    });
  });

  describe('Error Recording', () => {
    it('should record error with context', () => {
      const error = new Error('Test error');
      obs.error(error, { user_id: 123, action: 'test' });

      const logs = obs.getLogs();
      expect(logs.length).toBe(1);
      expect(logs[0].level).toBe(LogLevel.ERROR);
      expect(logs[0].event).toBe('error');
      expect(logs[0].data.message).toBe('Test error');
      expect(logs[0].data.stack).toBeDefined();
      expect(logs[0].data.context).toEqual({ user_id: 123, action: 'test' });
    });

    it('should include stack trace in error', () => {
      const error = new Error('Test error');
      obs.error(error, {});

      const logs = obs.getLogs();
      expect(logs[0].data.stack).toContain('Error: Test error');
    });
  });

  describe('Data Retrieval', () => {
    it('should return copy of logs', () => {
      obs.log(LogLevel.INFO, 'test', {});

      const logs1 = obs.getLogs();
      const logs2 = obs.getLogs();

      expect(logs1).not.toBe(logs2); // Different array instances
      expect(logs1).toEqual(logs2); // Same content
    });

    it('should return copy of metrics', () => {
      obs.metric('test', 10);

      const metrics1 = obs.getMetrics();
      const metrics2 = obs.getMetrics();

      expect(metrics1).not.toBe(metrics2);
      expect(metrics1).toEqual(metrics2);
    });

    it('should return copy of spans', () => {
      const span = obs.startSpan('test');
      span.end();

      const spans1 = obs.getSpans();
      const spans2 = obs.getSpans();

      expect(spans1).not.toBe(spans2);
      expect(spans1).toEqual(spans2);
    });
  });

  describe('Clear Data', () => {
    it('should clear all logs', () => {
      obs.log(LogLevel.INFO, 'test', {});
      obs.clear();

      expect(obs.getLogs().length).toBe(0);
    });

    it('should clear all metrics', () => {
      obs.metric('test', 10);
      obs.clear();

      expect(obs.getMetrics().length).toBe(0);
    });

    it('should clear all spans', () => {
      const span = obs.startSpan('test');
      span.end();
      obs.clear();

      expect(obs.getSpans().length).toBe(0);
    });
  });

  describe('Statistics', () => {
    beforeEach(() => {
      obs.log(LogLevel.DEBUG, 'debug1', {});
      obs.log(LogLevel.INFO, 'info1', {});
      obs.log(LogLevel.INFO, 'info2', {});
      obs.log(LogLevel.WARN, 'warn1', {});
      obs.log(LogLevel.ERROR, 'error1', {});

      obs.metric('request_count', 10);
      obs.metric('response_time', 100);
      obs.metric('request_count', 20);

      const span1 = obs.startSpan('operation1');
      const span2 = obs.startSpan('operation2');
      span1.end();
      span2.end();
    });

    it('should count total logs', () => {
      const stats = obs.getStats();
      expect(stats.total_logs).toBe(5);
    });

    it('should count logs by level', () => {
      const stats = obs.getStats();
      expect(stats.log_levels[LogLevel.DEBUG]).toBe(1);
      expect(stats.log_levels[LogLevel.INFO]).toBe(2);
      expect(stats.log_levels[LogLevel.WARN]).toBe(1);
      expect(stats.log_levels[LogLevel.ERROR]).toBe(1);
    });

    it('should count total metrics', () => {
      const stats = obs.getStats();
      expect(stats.total_metrics).toBe(3);
    });

    it('should list unique metric names', () => {
      const stats = obs.getStats();
      expect(stats.metric_names).toContain('request_count');
      expect(stats.metric_names).toContain('response_time');
      expect(stats.metric_names.length).toBe(2);
    });

    it('should count total spans', () => {
      const stats = obs.getStats();
      expect(stats.total_spans).toBe(2);
    });

    it('should list unique span names', () => {
      const stats = obs.getStats();
      expect(stats.span_names).toContain('operation1');
      expect(stats.span_names).toContain('operation2');
    });
  });

  describe('Export Capabilities', () => {
    beforeEach(() => {
      obs.log(LogLevel.INFO, 'test_event', { key: 'value' });
      obs.metric('test_metric', 42, { tag: 'value' });
      const span = obs.startSpan('test_span');
      span.setTag('test', true);
      span.end();
    });

    it('should export logs as JSON', () => {
      const json = obs.exportLogsJSON();
      const parsed = JSON.parse(json);

      expect(Array.isArray(parsed)).toBe(true);
      expect(parsed.length).toBe(1);
      expect(parsed[0].event).toBe('test_event');
    });

    it('should export metrics as JSON', () => {
      const json = obs.exportMetricsJSON();
      const parsed = JSON.parse(json);

      expect(Array.isArray(parsed)).toBe(true);
      expect(parsed.length).toBe(1);
      expect(parsed[0].name).toBe('test_metric');
    });

    it('should export spans as JSON', () => {
      const json = obs.exportSpansJSON();
      const parsed = JSON.parse(json);

      expect(Array.isArray(parsed)).toBe(true);
      expect(parsed.length).toBe(1);
      expect(parsed[0].name).toBe('test_span');
    });

    it('should export metrics as CSV', () => {
      const csv = obs.exportMetricsCSV();

      expect(csv).toContain('timestamp,name,value,tags');
      expect(csv).toContain('test_metric');
      expect(csv).toContain('42');
    });

    it('should format CSV with proper escaping', () => {
      obs.metric('metric_with_tags', 100, { key: 'value' });

      const csv = obs.exportMetricsCSV();
      const lines = csv.split('\n');

      expect(lines.length).toBe(3); // Header + 2 data rows
      expect(lines[0]).toBe('timestamp,name,value,tags');
    });
  });

  describe('Filtering', () => {
    beforeEach(() => {
      obs.log(LogLevel.DEBUG, 'event1', {});
      obs.log(LogLevel.INFO, 'event1', {});
      obs.log(LogLevel.WARN, 'event2', {});
      obs.log(LogLevel.ERROR, 'event3', {});

      obs.metric('metric1', 10);
      obs.metric('metric2', 20);
      obs.metric('metric1', 30);
    });

    it('should filter logs by level', () => {
      const infoLogs = obs.getLogsByLevel(LogLevel.INFO);

      expect(infoLogs.length).toBe(1);
      expect(infoLogs[0].level).toBe(LogLevel.INFO);
    });

    it('should filter logs by event', () => {
      const event1Logs = obs.getLogsByEvent('event1');

      expect(event1Logs.length).toBe(2);
      expect(event1Logs[0].event).toBe('event1');
      expect(event1Logs[1].event).toBe('event1');
    });

    it('should filter metrics by name', () => {
      const metric1 = obs.getMetricsByName('metric1');

      expect(metric1.length).toBe(2);
      expect(metric1[0].value).toBe(10);
      expect(metric1[1].value).toBe(30);
    });

    it('should return empty array for non-existent filters', () => {
      expect(obs.getLogsByLevel(LogLevel.DEBUG).length).toBe(1);
      expect(obs.getLogsByEvent('nonexistent').length).toBe(0);
      expect(obs.getMetricsByName('nonexistent').length).toBe(0);
    });
  });

  describe('Metric Statistics', () => {
    beforeEach(() => {
      obs.metric('response_time', 100);
      obs.metric('response_time', 200);
      obs.metric('response_time', 150);
    });

    it('should calculate metric statistics', () => {
      const stats = obs.getMetricStats('response_time');

      expect(stats).toBeDefined();
      expect(stats!.count).toBe(3);
      expect(stats!.sum).toBe(450);
      expect(stats!.avg).toBe(150);
      expect(stats!.min).toBe(100);
      expect(stats!.max).toBe(200);
    });

    it('should return null for non-existent metric', () => {
      const stats = obs.getMetricStats('nonexistent');

      expect(stats).toBeNull();
    });

    it('should handle single metric', () => {
      obs.metric('single_metric', 42);

      const stats = obs.getMetricStats('single_metric');

      expect(stats!.count).toBe(1);
      expect(stats!.sum).toBe(42);
      expect(stats!.avg).toBe(42);
      expect(stats!.min).toBe(42);
      expect(stats!.max).toBe(42);
    });
  });

  describe('Log Level Management', () => {
    it('should set log level', () => {
      obs.setLogLevel(LogLevel.WARN);

      expect(obs.getLogLevel()).toBe(LogLevel.WARN);
    });

    it('should respect updated log level', () => {
      obs.setLogLevel(LogLevel.ERROR);

      obs.log(LogLevel.DEBUG, 'debug', {});
      obs.log(LogLevel.INFO, 'info', {});
      obs.log(LogLevel.WARN, 'warn', {});
      obs.log(LogLevel.ERROR, 'error', {});

      const logs = obs.getLogs();
      expect(logs.length).toBe(1); // Only ERROR
      expect(logs[0].level).toBe(LogLevel.ERROR);
    });

    it('should get current log level', () => {
      const obsDebug = new Observability(LogLevel.DEBUG);
      expect(obsDebug.getLogLevel()).toBe(LogLevel.DEBUG);

      const obsInfo = new Observability(LogLevel.INFO);
      expect(obsInfo.getLogLevel()).toBe(LogLevel.INFO);
    });
  });

  describe('Factory Function', () => {
    it('should create Observability instance with default log level', () => {
      const instance = createObservability();

      expect(instance).toBeInstanceOf(Observability);
      expect(instance.getLogLevel()).toBe(LogLevel.INFO);
    });

    it('should create Observability instance with custom log level', () => {
      const instance = createObservability(LogLevel.DEBUG);

      expect(instance).toBeInstanceOf(Observability);
      expect(instance.getLogLevel()).toBe(LogLevel.DEBUG);
    });
  });
});
