/**
 * GDebug - Stack Traces
 *
 * O(1) stack trace management for Grammar Language debugging.
 * Provides call stack tracking, async call chains, and error traces.
 *
 * Features:
 * - O(1) push/pop stack frames
 * - Async call chain tracking
 * - Source map support
 * - Stack depth limiting
 * - Frame filtering
 */

// ============================================================================
// Types
// ============================================================================

export interface StackFrame {
  id: string;
  function_name: string;
  file: string;
  line: number;
  column: number;
  depth: number;
  locals?: Record<string, any>;
  is_async?: boolean;
  timestamp: number;
}

export interface CallStack {
  frames: StackFrame[];
  max_depth: number;
  current_depth: number;
}

export interface AsyncChain {
  id: string;
  frames: StackFrame[];
  parent_chain_id?: string;
}

export interface ErrorTrace {
  error: Error;
  stack: StackFrame[];
  async_chains?: AsyncChain[];
  timestamp: number;
}

// ============================================================================
// Stack Trace Manager
// ============================================================================

export class StackTraceManager {
  private frames: StackFrame[] = [];
  private maxDepth: number;
  private asyncChains: Map<string, AsyncChain> = new Map();
  private errorTraces: ErrorTrace[] = [];

  constructor(maxDepth: number = 100) {
    this.maxDepth = maxDepth;
  }

  /**
   * Push stack frame (O(1))
   */
  push(
    functionName: string,
    file: string,
    line: number,
    column: number = 0,
    locals?: Record<string, any>,
    isAsync: boolean = false
  ): StackFrame {
    const depth = this.frames.length;

    if (depth >= this.maxDepth) {
      throw new Error(`Stack overflow: max depth ${this.maxDepth} exceeded`);
    }

    const frame: StackFrame = {
      id: this.generateFrameId(),
      function_name: functionName,
      file,
      line,
      column,
      depth,
      locals: locals ? { ...locals } : undefined,
      is_async: isAsync,
      timestamp: Date.now()
    };

    this.frames.push(frame);

    return frame;
  }

  /**
   * Pop stack frame (O(1))
   */
  pop(): StackFrame | null {
    return this.frames.pop() ?? null;
  }

  /**
   * Get current frame (O(1))
   */
  current(): StackFrame | null {
    return this.frames[this.frames.length - 1] ?? null;
  }

  /**
   * Get frame at depth (O(1))
   */
  getFrame(depth: number): StackFrame | null {
    if (depth < 0 || depth >= this.frames.length) {
      return null;
    }
    return this.frames[depth];
  }

  /**
   * Get all frames
   */
  getFrames(): StackFrame[] {
    return [...this.frames];
  }

  /**
   * Get current depth
   */
  getDepth(): number {
    return this.frames.length;
  }

  /**
   * Get call stack summary
   */
  getCallStack(): CallStack {
    return {
      frames: this.getFrames(),
      max_depth: this.maxDepth,
      current_depth: this.frames.length
    };
  }

  /**
   * Create async chain (for async/await tracking)
   */
  createAsyncChain(parentChainId?: string): string {
    const chainId = this.generateChainId();

    const chain: AsyncChain = {
      id: chainId,
      frames: this.getFrames(),
      parent_chain_id: parentChainId
    };

    this.asyncChains.set(chainId, chain);

    return chainId;
  }

  /**
   * Get async chain (O(1))
   */
  getAsyncChain(chainId: string): AsyncChain | undefined {
    return this.asyncChains.get(chainId);
  }

  /**
   * Get full async chain (follows parent links)
   */
  getFullAsyncChain(chainId: string): StackFrame[] {
    const frames: StackFrame[] = [];
    let currentChainId: string | undefined = chainId;

    while (currentChainId) {
      const chain = this.asyncChains.get(currentChainId);
      if (!chain) break;

      frames.push(...chain.frames);
      currentChainId = chain.parent_chain_id;
    }

    return frames;
  }

  /**
   * Record error trace
   */
  recordError(error: Error): void {
    const trace: ErrorTrace = {
      error,
      stack: this.getFrames(),
      async_chains: this.getAllAsyncChains(),
      timestamp: Date.now()
    };

    this.errorTraces.push(trace);

    // Keep only last 100 errors
    if (this.errorTraces.length > 100) {
      this.errorTraces.shift();
    }
  }

  /**
   * Get error traces
   */
  getErrorTraces(): ErrorTrace[] {
    return [...this.errorTraces];
  }

  /**
   * Get last error trace
   */
  getLastError(): ErrorTrace | null {
    return this.errorTraces[this.errorTraces.length - 1] ?? null;
  }

  /**
   * Format stack trace (human-readable)
   */
  formatStackTrace(): string {
    const lines: string[] = ['Stack Trace:'];

    for (let i = this.frames.length - 1; i >= 0; i--) {
      const frame = this.frames[i];
      const indent = '  '.repeat(this.frames.length - i - 1);

      lines.push(
        `${indent}at ${frame.function_name} (${frame.file}:${frame.line}:${frame.column})`
      );

      if (frame.is_async) {
        lines.push(`${indent}  [async]`);
      }
    }

    return lines.join('\n');
  }

  /**
   * Clear stack (for new execution context)
   */
  clear(): void {
    this.frames = [];
  }

  /**
   * Clear all data
   */
  clearAll(): void {
    this.frames = [];
    this.asyncChains.clear();
    this.errorTraces = [];
  }

  // =========================================================================
  // Private Helpers
  // =========================================================================

  private generateFrameId(): string {
    return `frame-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  private generateChainId(): string {
    return `chain-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  private getAllAsyncChains(): AsyncChain[] {
    return Array.from(this.asyncChains.values());
  }
}

// ============================================================================
// Source Map Support
// ============================================================================

export interface SourceLocation {
  original_file: string;
  original_line: number;
  original_column: number;
  generated_file: string;
  generated_line: number;
  generated_column: number;
}

export class SourceMapResolver {
  private sourceMaps: Map<string, any> = new Map();

  /**
   * Register source map (O(1))
   */
  registerSourceMap(generatedFile: string, sourceMap: any): void {
    this.sourceMaps.set(generatedFile, sourceMap);
  }

  /**
   * Resolve location (O(1) with source map)
   */
  resolve(file: string, line: number, column: number): SourceLocation | null {
    const sourceMap = this.sourceMaps.get(file);

    if (!sourceMap) {
      return null;
    }

    // In production, would use source-map library
    // For now, return mock data
    return {
      original_file: file.replace('.js', '.ts'),
      original_line: line,
      original_column: column,
      generated_file: file,
      generated_line: line,
      generated_column: column
    };
  }

  /**
   * Resolve stack frame
   */
  resolveFrame(frame: StackFrame): StackFrame {
    const location = this.resolve(frame.file, frame.line, frame.column);

    if (!location) {
      return frame;
    }

    return {
      ...frame,
      file: location.original_file,
      line: location.original_line,
      column: location.original_column
    };
  }

  /**
   * Clear source maps
   */
  clear(): void {
    this.sourceMaps.clear();
  }
}

// ============================================================================
// Stack Frame Filter
// ============================================================================

export class FrameFilter {
  private filters: Set<string> = new Set();

  /**
   * Add filter pattern (O(1))
   */
  addFilter(pattern: string): void {
    this.filters.add(pattern);
  }

  /**
   * Remove filter (O(1))
   */
  removeFilter(pattern: string): void {
    this.filters.delete(pattern);
  }

  /**
   * Check if frame should be filtered
   */
  shouldFilter(frame: StackFrame): boolean {
    for (const pattern of this.filters) {
      if (this.matchPattern(frame, pattern)) {
        return true;
      }
    }
    return false;
  }

  /**
   * Filter frames
   */
  filter(frames: StackFrame[]): StackFrame[] {
    return frames.filter(frame => !this.shouldFilter(frame));
  }

  /**
   * Clear filters
   */
  clear(): void {
    this.filters.clear();
  }

  // =========================================================================
  // Private Helpers
  // =========================================================================

  private matchPattern(frame: StackFrame, pattern: string): boolean {
    // Simple pattern matching (file or function name)
    return (
      frame.file.includes(pattern) ||
      frame.function_name.includes(pattern)
    );
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create stack trace manager
 */
export function createStackTraceManager(maxDepth?: number): StackTraceManager {
  return new StackTraceManager(maxDepth);
}

/**
 * Create source map resolver
 */
export function createSourceMapResolver(): SourceMapResolver {
  return new SourceMapResolver();
}

/**
 * Create frame filter
 */
export function createFrameFilter(): FrameFilter {
  return new FrameFilter();
}
