/**
 * GDebug - Step Execution
 *
 * O(1) step-by-step execution control for Grammar Language debugging.
 * Provides step-over, step-into, step-out, and continue operations.
 *
 * Features:
 * - O(1) step state management
 * - Step over (next line in current scope)
 * - Step into (enter function calls)
 * - Step out (exit current function)
 * - Continue (run until next breakpoint)
 * - Step count tracking
 */

// ============================================================================
// Types
// ============================================================================

export type StepMode =
  | 'step-over'   // Execute next line in current function
  | 'step-into'   // Step into function calls
  | 'step-out'    // Step out of current function
  | 'continue';   // Continue until next breakpoint

export interface StepState {
  mode: StepMode;
  current_file: string;
  current_line: number;
  current_depth: number;
  target_depth?: number; // For step-out
  step_count: number;
  paused: boolean;
}

export interface StepResult {
  stopped: boolean;
  file: string;
  line: number;
  depth: number;
  reason: StepStopReason;
}

export type StepStopReason =
  | 'step-complete'
  | 'breakpoint-hit'
  | 'function-entry'
  | 'function-exit'
  | 'error';

// ============================================================================
// Step Controller
// ============================================================================

export class StepController {
  private state: StepState;
  private stepHistory: Array<{ file: string; line: number; depth: number }> = [];

  constructor() {
    this.state = {
      mode: 'continue',
      current_file: '',
      current_line: 0,
      current_depth: 0,
      step_count: 0,
      paused: false
    };
  }

  /**
   * Step over (next line in current scope)
   */
  stepOver(file: string, line: number): void {
    this.state.mode = 'step-over';
    this.state.current_file = file;
    this.state.current_line = line;
    this.state.paused = false;
    this.state.step_count++;
  }

  /**
   * Step into (enter function calls)
   */
  stepInto(file: string, line: number): void {
    this.state.mode = 'step-into';
    this.state.current_file = file;
    this.state.current_line = line;
    this.state.paused = false;
    this.state.step_count++;
  }

  /**
   * Step out (exit current function)
   */
  stepOut(file: string, line: number, currentDepth: number): void {
    this.state.mode = 'step-out';
    this.state.current_file = file;
    this.state.current_line = line;
    this.state.current_depth = currentDepth;
    this.state.target_depth = currentDepth - 1; // Exit to parent
    this.state.paused = false;
    this.state.step_count++;
  }

  /**
   * Continue (run until next breakpoint)
   */
  continue(file: string, line: number): void {
    this.state.mode = 'continue';
    this.state.current_file = file;
    this.state.current_line = line;
    this.state.paused = false;
    this.state.step_count = 0; // Reset counter
  }

  /**
   * Check if should stop at location (O(1))
   */
  shouldStop(file: string, line: number, depth: number): StepResult {
    // Already paused
    if (this.state.paused) {
      return {
        stopped: false,
        file,
        line,
        depth,
        reason: 'step-complete'
      };
    }

    // Record step history
    this.recordStep(file, line, depth);

    // Check based on mode
    switch (this.state.mode) {
      case 'step-over':
        return this.checkStepOver(file, line, depth);

      case 'step-into':
        return this.checkStepInto(file, line, depth);

      case 'step-out':
        return this.checkStepOut(file, line, depth);

      case 'continue':
        return this.checkContinue(file, line, depth);

      default:
        return {
          stopped: false,
          file,
          line,
          depth,
          reason: 'step-complete'
        };
    }
  }

  /**
   * Pause execution
   */
  pause(): void {
    this.state.paused = true;
  }

  /**
   * Resume execution
   */
  resume(): void {
    this.state.paused = false;
  }

  /**
   * Reset stepper
   */
  reset(): void {
    this.state = {
      mode: 'continue',
      current_file: '',
      current_line: 0,
      current_depth: 0,
      step_count: 0,
      paused: false
    };
    this.stepHistory = [];
  }

  /**
   * Get current state
   */
  getState(): StepState {
    return { ...this.state };
  }

  /**
   * Get step history
   */
  getHistory(): Array<{ file: string; line: number; depth: number }> {
    return [...this.stepHistory];
  }

  /**
   * Get step count
   */
  getStepCount(): number {
    return this.state.step_count;
  }

  // =========================================================================
  // Private Helpers
  // =========================================================================

  private checkStepOver(file: string, line: number, depth: number): StepResult {
    // Stop if:
    // 1. Different line in same file and same depth
    // 2. OR returned to shallower depth (function returned)

    const sameLine = line === this.state.current_line;
    const sameFile = file === this.state.current_file;
    const sameDepth = depth === this.state.current_depth;
    const shallowerDepth = depth < this.state.current_depth;

    if (!sameLine && sameFile && (sameDepth || shallowerDepth)) {
      this.state.paused = true;
      return {
        stopped: true,
        file,
        line,
        depth,
        reason: 'step-complete'
      };
    }

    return {
      stopped: false,
      file,
      line,
      depth,
      reason: 'step-complete'
    };
  }

  private checkStepInto(file: string, line: number, depth: number): StepResult {
    // Stop at any different line (including deeper calls)
    const differentLine = line !== this.state.current_line || file !== this.state.current_file;

    if (differentLine) {
      this.state.paused = true;

      const reason = depth > this.state.current_depth
        ? 'function-entry'
        : 'step-complete';

      return {
        stopped: true,
        file,
        line,
        depth,
        reason
      };
    }

    return {
      stopped: false,
      file,
      line,
      depth,
      reason: 'step-complete'
    };
  }

  private checkStepOut(file: string, line: number, depth: number): StepResult {
    // Stop when we reach target depth (parent scope)
    if (this.state.target_depth !== undefined && depth <= this.state.target_depth) {
      this.state.paused = true;
      return {
        stopped: true,
        file,
        line,
        depth,
        reason: 'function-exit'
      };
    }

    return {
      stopped: false,
      file,
      line,
      depth,
      reason: 'step-complete'
    };
  }

  private checkContinue(file: string, line: number, depth: number): StepResult {
    // Continue mode doesn't stop (handled by breakpoints)
    return {
      stopped: false,
      file,
      line,
      depth,
      reason: 'step-complete'
    };
  }

  private recordStep(file: string, line: number, depth: number): void {
    this.stepHistory.push({ file, line, depth });

    // Keep only last 1000 steps
    if (this.stepHistory.length > 1000) {
      this.stepHistory.shift();
    }
  }
}

// ============================================================================
// Step Recorder (for replay debugging)
// ============================================================================

export interface StepRecord {
  file: string;
  line: number;
  depth: number;
  timestamp: number;
  locals?: Record<string, any>;
}

export class StepRecorder {
  private records: StepRecord[] = [];
  private recording: boolean = false;
  private replayIndex: number = 0;

  /**
   * Start recording
   */
  startRecording(): void {
    this.recording = true;
    this.records = [];
  }

  /**
   * Stop recording
   */
  stopRecording(): void {
    this.recording = false;
  }

  /**
   * Record step (O(1))
   */
  record(file: string, line: number, depth: number, locals?: Record<string, any>): void {
    if (!this.recording) return;

    this.records.push({
      file,
      line,
      depth,
      timestamp: Date.now(),
      locals: locals ? { ...locals } : undefined
    });
  }

  /**
   * Get all records
   */
  getRecords(): StepRecord[] {
    return [...this.records];
  }

  /**
   * Start replay
   */
  startReplay(): void {
    this.replayIndex = 0;
  }

  /**
   * Get next replay step (O(1))
   */
  nextReplayStep(): StepRecord | null {
    if (this.replayIndex >= this.records.length) {
      return null;
    }

    return this.records[this.replayIndex++];
  }

  /**
   * Get previous replay step (O(1))
   */
  previousReplayStep(): StepRecord | null {
    if (this.replayIndex <= 0) {
      return null;
    }

    this.replayIndex--;
    return this.records[this.replayIndex];
  }

  /**
   * Jump to step (O(1))
   */
  jumpToStep(index: number): StepRecord | null {
    if (index < 0 || index >= this.records.length) {
      return null;
    }

    this.replayIndex = index;
    return this.records[index];
  }

  /**
   * Clear records
   */
  clear(): void {
    this.records = [];
    this.replayIndex = 0;
    this.recording = false;
  }

  /**
   * Get stats
   */
  getStats(): {
    total_steps: number;
    recording: boolean;
    replay_index: number;
    duration_ms: number;
  } {
    const duration = this.records.length > 0
      ? this.records[this.records.length - 1].timestamp - this.records[0].timestamp
      : 0;

    return {
      total_steps: this.records.length,
      recording: this.recording,
      replay_index: this.replayIndex,
      duration_ms: duration
    };
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create step controller
 */
export function createStepController(): StepController {
  return new StepController();
}

/**
 * Create step recorder
 */
export function createStepRecorder(): StepRecorder {
  return new StepRecorder();
}
