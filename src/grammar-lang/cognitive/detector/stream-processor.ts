/**
 * Stream Processor - Real-time Manipulation Detection
 * Process text streams incrementally with debouncing and event emission
 * Enables real-time detection as text is typed or received
 */

import { EventEmitter } from 'events';
import { detectManipulation } from './pattern-matcher';
import {
  PatternMatchConfig,
  PatternMatchResult,
  DetectionResult
} from '../types';

// ============================================================
// TYPES
// ============================================================

export interface StreamProcessorConfig extends PatternMatchConfig {
  debounce_ms?: number;           // Debounce delay (default: 300ms)
  min_text_length?: number;       // Minimum text length to analyze (default: 10)
  context_window?: number;        // How many previous characters to keep (default: 500)
  enable_incremental?: boolean;   // Enable incremental updates (default: true)
  emit_partial_results?: boolean; // Emit results for incomplete text (default: false)
}

export interface StreamAnalysisEvent {
  timestamp: number;
  text: string;
  text_length: number;
  is_incremental: boolean;
  result: PatternMatchResult;
}

export interface StreamStats {
  total_chunks_processed: number;
  total_characters_processed: number;
  total_detections: number;
  average_processing_time_ms: number;
  session_start_time: number;
  session_duration_ms: number;
}

// ============================================================
// STREAM PROCESSOR CLASS
// ============================================================

export class StreamProcessor extends EventEmitter {
  private config: StreamProcessorConfig;
  private buffer: string;
  private contextBuffer: string;
  private debounceTimer: NodeJS.Timeout | null;
  private isProcessing: boolean;
  private stats: StreamStats;
  private lastResult: PatternMatchResult | null;

  constructor(config: StreamProcessorConfig = {}) {
    super();

    this.config = {
      debounce_ms: config.debounce_ms ?? 300,
      min_text_length: config.min_text_length ?? 10,
      context_window: config.context_window ?? 500,
      enable_incremental: config.enable_incremental ?? true,
      emit_partial_results: config.emit_partial_results ?? false,
      ...config
    };

    this.buffer = '';
    this.contextBuffer = '';
    this.debounceTimer = null;
    this.isProcessing = false;
    this.lastResult = null;

    this.stats = {
      total_chunks_processed: 0,
      total_characters_processed: 0,
      total_detections: 0,
      average_processing_time_ms: 0,
      session_start_time: Date.now(),
      session_duration_ms: 0
    };
  }

  // ============================================================
  // PUBLIC API
  // ============================================================

  /**
   * Add text to the stream
   * Text will be buffered and processed with debouncing
   */
  public push(text: string): void {
    this.buffer += text;
    this.stats.total_characters_processed += text.length;

    // Emit raw text event
    this.emit('text', {
      text,
      buffer_length: this.buffer.length,
      timestamp: Date.now()
    });

    // Debounce processing
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }

    this.debounceTimer = setTimeout(() => {
      this.processBuffer();
    }, this.config.debounce_ms);
  }

  /**
   * Force immediate processing of buffered text
   */
  public async flush(): Promise<PatternMatchResult | null> {
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = null;
    }

    return this.processBuffer();
  }

  /**
   * Clear buffer and reset state
   */
  public clear(): void {
    this.buffer = '';
    this.contextBuffer = '';
    this.lastResult = null;

    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = null;
    }

    this.emit('cleared', {
      timestamp: Date.now()
    });
  }

  /**
   * Get current buffer content
   */
  public getBuffer(): string {
    return this.buffer;
  }

  /**
   * Get processing statistics
   */
  public getStats(): StreamStats {
    return {
      ...this.stats,
      session_duration_ms: Date.now() - this.stats.session_start_time
    };
  }

  /**
   * Get last analysis result
   */
  public getLastResult(): PatternMatchResult | null {
    return this.lastResult;
  }

  /**
   * Stop stream processor and clean up
   */
  public stop(): void {
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = null;
    }

    this.removeAllListeners();
    this.emit('stopped', {
      stats: this.getStats(),
      timestamp: Date.now()
    });
  }

  // ============================================================
  // PRIVATE METHODS
  // ============================================================

  /**
   * Process buffered text
   */
  private async processBuffer(): Promise<PatternMatchResult | null> {
    // Check if processing already in progress
    if (this.isProcessing) {
      return null;
    }

    // Check minimum length
    if (this.buffer.length < this.config.min_text_length!) {
      return null;
    }

    this.isProcessing = true;

    try {
      const startTime = Date.now();

      // Combine context + current buffer
      const fullText = this.config.enable_incremental
        ? this.contextBuffer + this.buffer
        : this.buffer;

      // Detect manipulation
      const result = await detectManipulation(fullText, this.config);

      const processingTime = Date.now() - startTime;

      // Update stats
      this.stats.total_chunks_processed++;
      this.stats.total_detections += result.total_matches;
      this.stats.average_processing_time_ms =
        (this.stats.average_processing_time_ms * (this.stats.total_chunks_processed - 1) +
          processingTime) /
        this.stats.total_chunks_processed;

      // Store result
      this.lastResult = result;

      // Emit analysis event
      const event: StreamAnalysisEvent = {
        timestamp: Date.now(),
        text: fullText,
        text_length: fullText.length,
        is_incremental: this.config.enable_incremental!,
        result
      };

      this.emit('analysis', event);

      // Emit detection events for each technique detected
      if (result.total_matches > 0) {
        this.emit('detection', {
          timestamp: Date.now(),
          detections: result.detections,
          highest_confidence: result.highest_confidence,
          dark_tetrad: result.dark_tetrad_aggregate
        });

        // Emit high-confidence alerts
        const highConfidenceDetections = result.detections.filter(
          d => d.confidence >= 0.9
        );

        if (highConfidenceDetections.length > 0) {
          this.emit('alert', {
            timestamp: Date.now(),
            severity: 'high',
            detections: highConfidenceDetections,
            message: `High-confidence manipulation detected: ${highConfidenceDetections.map(d => d.technique_name).join(', ')}`
          });
        }
      }

      // Update context window
      if (this.config.enable_incremental && this.config.context_window! > 0) {
        const combinedText = this.contextBuffer + this.buffer;
        if (combinedText.length > this.config.context_window!) {
          this.contextBuffer = combinedText.slice(-this.config.context_window!);
        } else {
          this.contextBuffer = combinedText;
        }
      }

      // Clear buffer after processing
      this.buffer = '';

      return result;

    } catch (error) {
      this.emit('error', {
        timestamp: Date.now(),
        error: error instanceof Error ? error.message : String(error),
        buffer_length: this.buffer.length
      });

      return null;

    } finally {
      this.isProcessing = false;
    }
  }
}

// ============================================================
// CONVENIENCE FUNCTIONS
// ============================================================

/**
 * Create a stream processor with default config
 */
export function createStreamProcessor(config?: StreamProcessorConfig): StreamProcessor {
  return new StreamProcessor(config);
}

/**
 * Process a stream of text chunks
 * Returns an async iterator that yields analysis results
 */
export async function* processTextStream(
  chunks: AsyncIterable<string> | Iterable<string>,
  config?: StreamProcessorConfig
): AsyncGenerator<StreamAnalysisEvent> {
  const processor = createStreamProcessor(config);

  // Set up event listener for analysis
  const analysisQueue: StreamAnalysisEvent[] = [];
  let resolveNext: ((value: StreamAnalysisEvent) => void) | null = null;

  processor.on('analysis', (event: StreamAnalysisEvent) => {
    if (resolveNext) {
      resolveNext(event);
      resolveNext = null;
    } else {
      analysisQueue.push(event);
    }
  });

  // Process chunks
  const processChunks = async () => {
    for await (const chunk of chunks) {
      processor.push(chunk);
    }
    await processor.flush();
    processor.stop();
  };

  processChunks();

  // Yield results as they come
  while (true) {
    if (analysisQueue.length > 0) {
      yield analysisQueue.shift()!;
    } else {
      // Wait for next analysis
      const event = await new Promise<StreamAnalysisEvent | null>((resolve) => {
        resolveNext = resolve;
        // Set timeout to check if processor stopped
        setTimeout(() => {
          if (resolveNext === resolve) {
            resolveNext = null;
            resolve(null);
          }
        }, 1000);
      });

      if (event === null) {
        break; // Processor stopped
      }

      yield event;
    }
  }
}

/**
 * Monitor live text input (e.g., from stdin, websocket, etc.)
 */
export function monitorLiveInput(
  onDetection: (event: StreamAnalysisEvent) => void,
  config?: StreamProcessorConfig
): StreamProcessor {
  const processor = createStreamProcessor(config);

  processor.on('analysis', (event: StreamAnalysisEvent) => {
    onDetection(event);
  });

  processor.on('alert', (alert: any) => {
    console.warn(`🚨 ALERT: ${alert.message}`);
  });

  processor.on('error', (error: any) => {
    console.error(`❌ ERROR: ${error.error}`);
  });

  return processor;
}
