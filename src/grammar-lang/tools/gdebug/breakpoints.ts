/**
 * GDebug - Breakpoints System
 *
 * O(1) breakpoint management for Grammar Language debugging.
 * Hash-based storage for constant-time breakpoint operations.
 *
 * Features:
 * - O(1) breakpoint add/remove/check
 * - Conditional breakpoints
 * - Hit count tracking
 * - Breakpoint groups
 * - Enable/disable without removal
 */

// ============================================================================
// Types
// ============================================================================

export interface Breakpoint {
  id: string;
  file: string;
  line: number;
  column?: number;
  condition?: string;
  enabled: boolean;
  hit_count: number;
  max_hits?: number;
  log_message?: string;
  created_at: number;
}

export interface BreakpointHit {
  breakpoint_id: string;
  timestamp: number;
  file: string;
  line: number;
  locals?: Record<string, any>;
  stack_depth: number;
}

export type BreakpointCondition = (locals: Record<string, any>) => boolean;

// ============================================================================
// Breakpoint Manager
// ============================================================================

export class BreakpointManager {
  // O(1) lookups
  private breakpoints: Map<string, Breakpoint> = new Map(); // by ID
  private locationIndex: Map<string, Set<string>> = new Map(); // by "file:line" -> Set<IDs>
  private hitHistory: Map<string, BreakpointHit[]> = new Map(); // by ID -> hits

  // Condition evaluators (stored separately)
  private conditions: Map<string, BreakpointCondition> = new Map();

  /**
   * Add breakpoint (O(1))
   */
  add(
    file: string,
    line: number,
    options: {
      column?: number;
      condition?: string | BreakpointCondition;
      max_hits?: number;
      log_message?: string;
    } = {}
  ): Breakpoint {
    const id = this.generateId(file, line, options.column);

    if (this.breakpoints.has(id)) {
      throw new Error(`Breakpoint already exists at ${file}:${line}`);
    }

    const breakpoint: Breakpoint = {
      id,
      file,
      line,
      column: options.column,
      condition: typeof options.condition === 'string' ? options.condition : undefined,
      enabled: true,
      hit_count: 0,
      max_hits: options.max_hits,
      log_message: options.log_message,
      created_at: Date.now()
    };

    // Store breakpoint (O(1))
    this.breakpoints.set(id, breakpoint);

    // Index by location (O(1))
    const locationKey = this.locationKey(file, line);
    if (!this.locationIndex.has(locationKey)) {
      this.locationIndex.set(locationKey, new Set());
    }
    this.locationIndex.get(locationKey)!.add(id);

    // Store condition evaluator if function
    if (typeof options.condition === 'function') {
      this.conditions.set(id, options.condition);
    }

    return breakpoint;
  }

  /**
   * Remove breakpoint (O(1))
   */
  remove(id: string): boolean {
    const breakpoint = this.breakpoints.get(id);
    if (!breakpoint) {
      return false;
    }

    // Remove from breakpoints map
    this.breakpoints.delete(id);

    // Remove from location index
    const locationKey = this.locationKey(breakpoint.file, breakpoint.line);
    const locationSet = this.locationIndex.get(locationKey);
    if (locationSet) {
      locationSet.delete(id);
      if (locationSet.size === 0) {
        this.locationIndex.delete(locationKey);
      }
    }

    // Remove condition
    this.conditions.delete(id);

    // Remove hit history
    this.hitHistory.delete(id);

    return true;
  }

  /**
   * Check if should break at location (O(1))
   */
  shouldBreak(
    file: string,
    line: number,
    locals?: Record<string, any>
  ): Breakpoint | null {
    const locationKey = this.locationKey(file, line);
    const breakpointIds = this.locationIndex.get(locationKey);

    if (!breakpointIds || breakpointIds.size === 0) {
      return null; // No breakpoints at this location
    }

    // Check each breakpoint at this location
    for (const id of breakpointIds) {
      const breakpoint = this.breakpoints.get(id)!;

      // Skip if disabled
      if (!breakpoint.enabled) {
        continue;
      }

      // Check max hits
      if (breakpoint.max_hits !== undefined && breakpoint.hit_count >= breakpoint.max_hits) {
        continue;
      }

      // Check condition
      if (!this.checkCondition(breakpoint, locals)) {
        continue;
      }

      // Breakpoint hit!
      return breakpoint;
    }

    return null;
  }

  /**
   * Record breakpoint hit (O(1))
   */
  recordHit(
    breakpointId: string,
    locals?: Record<string, any>,
    stackDepth: number = 0
  ): BreakpointHit {
    const breakpoint = this.breakpoints.get(breakpointId);
    if (!breakpoint) {
      throw new Error(`Breakpoint ${breakpointId} not found`);
    }

    // Increment hit count
    breakpoint.hit_count++;

    // Record hit
    const hit: BreakpointHit = {
      breakpoint_id: breakpointId,
      timestamp: Date.now(),
      file: breakpoint.file,
      line: breakpoint.line,
      locals,
      stack_depth: stackDepth
    };

    if (!this.hitHistory.has(breakpointId)) {
      this.hitHistory.set(breakpointId, []);
    }
    this.hitHistory.get(breakpointId)!.push(hit);

    return hit;
  }

  /**
   * Enable breakpoint (O(1))
   */
  enable(id: string): boolean {
    const breakpoint = this.breakpoints.get(id);
    if (!breakpoint) {
      return false;
    }
    breakpoint.enabled = true;
    return true;
  }

  /**
   * Disable breakpoint (O(1))
   */
  disable(id: string): boolean {
    const breakpoint = this.breakpoints.get(id);
    if (!breakpoint) {
      return false;
    }
    breakpoint.enabled = false;
    return true;
  }

  /**
   * Get breakpoint (O(1))
   */
  get(id: string): Breakpoint | undefined {
    return this.breakpoints.get(id);
  }

  /**
   * Get all breakpoints at location (O(1))
   */
  getAtLocation(file: string, line: number): Breakpoint[] {
    const locationKey = this.locationKey(file, line);
    const breakpointIds = this.locationIndex.get(locationKey);

    if (!breakpointIds) {
      return [];
    }

    return Array.from(breakpointIds).map(id => this.breakpoints.get(id)!);
  }

  /**
   * Get all breakpoints
   */
  getAll(): Breakpoint[] {
    return Array.from(this.breakpoints.values());
  }

  /**
   * Get hit history (O(1))
   */
  getHitHistory(id: string): BreakpointHit[] {
    return this.hitHistory.get(id) ?? [];
  }

  /**
   * Clear all breakpoints
   */
  clearAll(): void {
    this.breakpoints.clear();
    this.locationIndex.clear();
    this.hitHistory.clear();
    this.conditions.clear();
  }

  /**
   * Clear breakpoints by file
   */
  clearFile(file: string): void {
    for (const [id, breakpoint] of this.breakpoints) {
      if (breakpoint.file === file) {
        this.remove(id);
      }
    }
  }

  /**
   * Get statistics
   */
  getStats(): {
    total: number;
    enabled: number;
    disabled: number;
    with_conditions: number;
    total_hits: number;
  } {
    let enabled = 0;
    let disabled = 0;
    let withConditions = 0;
    let totalHits = 0;

    for (const breakpoint of this.breakpoints.values()) {
      if (breakpoint.enabled) enabled++;
      else disabled++;

      if (breakpoint.condition || this.conditions.has(breakpoint.id)) {
        withConditions++;
      }

      totalHits += breakpoint.hit_count;
    }

    return {
      total: this.breakpoints.size,
      enabled,
      disabled,
      with_conditions: withConditions,
      total_hits: totalHits
    };
  }

  // =========================================================================
  // Private Helpers
  // =========================================================================

  private generateId(file: string, line: number, column?: number): string {
    return column !== undefined
      ? `${file}:${line}:${column}`
      : `${file}:${line}`;
  }

  private locationKey(file: string, line: number): string {
    return `${file}:${line}`;
  }

  private checkCondition(breakpoint: Breakpoint, locals?: Record<string, any>): boolean {
    // No condition = always true
    if (!breakpoint.condition && !this.conditions.has(breakpoint.id)) {
      return true;
    }

    // Function condition
    const conditionFn = this.conditions.get(breakpoint.id);
    if (conditionFn) {
      try {
        return conditionFn(locals ?? {});
      } catch (error) {
        console.warn(`Breakpoint condition error: ${error}`);
        return false;
      }
    }

    // String condition (would need eval - unsafe, so skip)
    // In production, would use a safe expression evaluator
    return true;
  }
}

// ============================================================================
// Breakpoint Groups (for organizing breakpoints)
// ============================================================================

export class BreakpointGroup {
  private name: string;
  private breakpointIds: Set<string> = new Set();

  constructor(name: string) {
    this.name = name;
  }

  /**
   * Add breakpoint to group (O(1))
   */
  add(breakpointId: string): void {
    this.breakpointIds.add(breakpointId);
  }

  /**
   * Remove breakpoint from group (O(1))
   */
  remove(breakpointId: string): void {
    this.breakpointIds.delete(breakpointId);
  }

  /**
   * Check if breakpoint is in group (O(1))
   */
  has(breakpointId: string): boolean {
    return this.breakpointIds.has(breakpointId);
  }

  /**
   * Get all breakpoint IDs
   */
  getAll(): string[] {
    return Array.from(this.breakpointIds);
  }

  /**
   * Clear group
   */
  clear(): void {
    this.breakpointIds.clear();
  }

  getName(): string {
    return this.name;
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create breakpoint manager
 */
export function createBreakpointManager(): BreakpointManager {
  return new BreakpointManager();
}

/**
 * Create breakpoint group
 */
export function createGroup(name: string): BreakpointGroup {
  return new BreakpointGroup(name);
}
