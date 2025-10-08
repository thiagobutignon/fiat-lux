/**
 * Observability Layer
 *
 * Provides structured logging, metrics, and distributed tracing
 * for the AGI self-evolution system.
 *
 * Features:
 * - Structured logging with log levels
 * - Metrics collection (counters, gauges, histograms)
 * - Distributed tracing (OpenTelemetry-compatible)
 * - In-memory storage for analysis
 * - Export capabilities (JSON, CSV)
 */

// ============================================================================
// Types
// ============================================================================

export enum LogLevel {
  DEBUG = 'debug',
  INFO = 'info',
  WARN = 'warn',
  ERROR = 'error',
}

export interface LogEntry {
  timestamp: number;
  level: LogLevel;
  event: string;
  data: any;
}

export interface Metric {
  timestamp: number;
  name: string;
  value: number;
  tags?: Record<string, string | number>;
}

export interface SpanData {
  id: string;
  name: string;
  start_time: number;
  end_time?: number;
  duration_ms?: number;
  tags: Record<string, any>;
  parent_id?: string;
}

export interface Span {
  id: string;
  name: string;
  end(): void;
  setTag(key: string, value: any): void;
}

export interface ObservabilityStats {
  total_logs: number;
  total_metrics: number;
  total_spans: number;
  log_levels: Record<LogLevel, number>;
  metric_names: string[];
  span_names: string[];
}

// ============================================================================
// Log Level Ordering
// ============================================================================

const LOG_LEVEL_ORDER: Record<LogLevel, number> = {
  [LogLevel.DEBUG]: 0,
  [LogLevel.INFO]: 1,
  [LogLevel.WARN]: 2,
  [LogLevel.ERROR]: 3,
};

// ============================================================================
// Observability Class
// ============================================================================

export class Observability {
  private logs: LogEntry[] = [];
  private metrics: Metric[] = [];
  private spans: SpanData[] = [];
  private logLevel: LogLevel;

  constructor(logLevel: LogLevel = LogLevel.INFO) {
    this.logLevel = logLevel;
  }

  /**
   * Log structured event
   */
  log(level: LogLevel, event: string, data: any): void {
    // Check if this log level should be recorded
    if (LOG_LEVEL_ORDER[level] < LOG_LEVEL_ORDER[this.logLevel]) {
      return; // Skip logs below threshold
    }

    const entry: LogEntry = {
      timestamp: Date.now(),
      level,
      event,
      data,
    };

    this.logs.push(entry);

    // Output to console
    const formatted = JSON.stringify(entry);

    if (level === LogLevel.ERROR) {
      console.error(formatted);
    } else {
      console.log(formatted);
    }
  }

  /**
   * Track metric
   */
  metric(name: string, value: number, tags?: Record<string, string | number>): void {
    const metric: Metric = {
      timestamp: Date.now(),
      name,
      value,
      tags,
    };

    this.metrics.push(metric);
  }

  /**
   * Start trace span
   */
  startSpan(name: string, parent?: Span): Span {
    const spanData: SpanData = {
      id: this.generateId(),
      name,
      start_time: Date.now(),
      tags: {},
      parent_id: parent?.id,
    };

    this.spans.push(spanData);

    // Return span interface
    const span: Span = {
      id: spanData.id,
      name: spanData.name,

      end: () => {
        spanData.end_time = Date.now();
        spanData.duration_ms = spanData.end_time - spanData.start_time;
      },

      setTag: (key: string, value: any) => {
        spanData.tags[key] = value;
      },
    };

    return span;
  }

  /**
   * Record error
   */
  error(error: Error, context: any): void {
    this.log(LogLevel.ERROR, 'error', {
      message: error.message,
      stack: error.stack,
      context,
    });
  }

  /**
   * Get all logs
   */
  getLogs(): LogEntry[] {
    return [...this.logs];
  }

  /**
   * Get all metrics
   */
  getMetrics(): Metric[] {
    return [...this.metrics];
  }

  /**
   * Get all spans
   */
  getSpans(): SpanData[] {
    return [...this.spans];
  }

  /**
   * Clear all data
   */
  clear(): void {
    this.logs = [];
    this.metrics = [];
    this.spans = [];
  }

  /**
   * Get statistics
   */
  getStats(): ObservabilityStats {
    // Count logs by level
    const log_levels: Record<LogLevel, number> = {
      [LogLevel.DEBUG]: 0,
      [LogLevel.INFO]: 0,
      [LogLevel.WARN]: 0,
      [LogLevel.ERROR]: 0,
    };

    for (const log of this.logs) {
      log_levels[log.level]++;
    }

    // Unique metric names
    const metric_names = [...new Set(this.metrics.map((m) => m.name))];

    // Unique span names
    const span_names = [...new Set(this.spans.map((s) => s.name))];

    return {
      total_logs: this.logs.length,
      total_metrics: this.metrics.length,
      total_spans: this.spans.length,
      log_levels,
      metric_names,
      span_names,
    };
  }

  /**
   * Export logs as JSON
   */
  exportLogsJSON(): string {
    return JSON.stringify(this.logs, null, 2);
  }

  /**
   * Export metrics as JSON
   */
  exportMetricsJSON(): string {
    return JSON.stringify(this.metrics, null, 2);
  }

  /**
   * Export spans as JSON (OpenTelemetry format)
   */
  exportSpansJSON(): string {
    return JSON.stringify(this.spans, null, 2);
  }

  /**
   * Export metrics as CSV
   */
  exportMetricsCSV(): string {
    const lines: string[] = [];

    // Header
    lines.push('timestamp,name,value,tags');

    // Data rows
    for (const metric of this.metrics) {
      const tags = metric.tags ? JSON.stringify(metric.tags) : '';
      lines.push(
        `${metric.timestamp},${metric.name},${metric.value},"${tags}"`
      );
    }

    return lines.join('\n');
  }

  /**
   * Get logs filtered by level
   */
  getLogsByLevel(level: LogLevel): LogEntry[] {
    return this.logs.filter((log) => log.level === level);
  }

  /**
   * Get logs filtered by event name
   */
  getLogsByEvent(event: string): LogEntry[] {
    return this.logs.filter((log) => log.event === event);
  }

  /**
   * Get metrics filtered by name
   */
  getMetricsByName(name: string): Metric[] {
    return this.metrics.filter((metric) => metric.name === name);
  }

  /**
   * Get metric statistics (sum, avg, min, max)
   */
  getMetricStats(
    name: string
  ): { count: number; sum: number; avg: number; min: number; max: number } | null {
    const filtered = this.getMetricsByName(name);

    if (filtered.length === 0) {
      return null;
    }

    const values = filtered.map((m) => m.value);
    const sum = values.reduce((a, b) => a + b, 0);
    const avg = sum / values.length;
    const min = Math.min(...values);
    const max = Math.max(...values);

    return {
      count: filtered.length,
      sum,
      avg,
      min,
      max,
    };
  }

  /**
   * Generate unique ID for spans
   */
  private generateId(): string {
    return `span_${Date.now()}_${Math.random().toString(36).substring(7)}`;
  }

  /**
   * Set log level
   */
  setLogLevel(level: LogLevel): void {
    this.logLevel = level;
  }

  /**
   * Get current log level
   */
  getLogLevel(): LogLevel {
    return this.logLevel;
  }
}

/**
 * Create a new Observability instance
 */
export function createObservability(logLevel: LogLevel = LogLevel.INFO): Observability {
  return new Observability(logLevel);
}
