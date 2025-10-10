/**
 * GDebug - Grammar Language Debugger
 *
 * O(1) debugging toolkit for Grammar Language.
 * Inspired by GVCS fitness tracking and genetic algorithms.
 *
 * Features:
 * - O(1) breakpoint management
 * - Deep variable inspection
 * - Step-by-step execution
 * - Stack trace tracking
 * - Async debugging support
 */

// ============================================================================
// Breakpoints
// ============================================================================

export {
  // Types
  Breakpoint,
  BreakpointHit,
  BreakpointCondition,

  // Classes
  BreakpointManager,
  BreakpointGroup,

  // Factory Functions
  createBreakpointManager,
  createGroup
} from './breakpoints';

import { createBreakpointManager as _createBreakpointManager } from './breakpoints';

// ============================================================================
// Variable Inspector
// ============================================================================

export {
  // Types
  Variable,
  VariableScope,
  WatchExpression,
  InspectionResult,

  // Classes
  VariableInspector,
  ScopeInspector,

  // Factory Functions
  createInspector,
  createScopeInspector
} from './inspector';

import { createInspector as _createInspector, createScopeInspector as _createScopeInspector } from './inspector';

// ============================================================================
// Step Execution
// ============================================================================

export {
  // Types
  StepMode,
  StepState,
  StepResult,
  StepStopReason,
  StepRecord,

  // Classes
  StepController,
  StepRecorder,

  // Factory Functions
  createStepController,
  createStepRecorder
} from './stepper';

import { createStepController as _createStepController, createStepRecorder as _createStepRecorder } from './stepper';

// ============================================================================
// Stack Traces
// ============================================================================

export {
  // Types
  StackFrame,
  CallStack,
  AsyncChain,
  ErrorTrace,
  SourceLocation,

  // Classes
  StackTraceManager,
  SourceMapResolver,
  FrameFilter,

  // Factory Functions
  createStackTraceManager,
  createSourceMapResolver,
  createFrameFilter
} from './trace';

import {
  createStackTraceManager as _createStackTraceManager,
  createSourceMapResolver as _createSourceMapResolver,
  createFrameFilter as _createFrameFilter
} from './trace';

// ============================================================================
// Convenience Factory
// ============================================================================

/**
 * Create a complete debugger instance with all components
 */
export function createDebugger(config: {
  maxStackDepth?: number;
} = {}) {
  return {
    breakpoints: _createBreakpointManager(),
    inspector: _createInspector(),
    scopeInspector: _createScopeInspector(),
    stepper: _createStepController(),
    recorder: _createStepRecorder(),
    stackTrace: _createStackTraceManager(config.maxStackDepth),
    sourceMap: _createSourceMapResolver(),
    frameFilter: _createFrameFilter()
  };
}

/**
 * Get GDebug version info
 */
export function version(): string {
  return '1.0.0';
}

/**
 * Get GDebug features summary
 */
export function features(): string[] {
  return [
    'O(1) breakpoint management',
    'Deep variable inspection',
    'Step-by-step execution (over/into/out)',
    'Stack trace tracking',
    'Async call chain tracking',
    'Source map support',
    'Watch expressions',
    'Replay debugging',
    'Error trace recording',
    'Conditional breakpoints',
    'Hit count tracking'
  ];
}
