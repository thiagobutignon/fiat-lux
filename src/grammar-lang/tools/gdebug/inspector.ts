/**
 * GDebug - Variable Inspector
 *
 * O(1) variable inspection and evaluation for Grammar Language debugging.
 * Provides deep object inspection, watch expressions, and value tracking.
 *
 * Features:
 * - O(1) variable lookup by name
 * - Deep object inspection
 * - Watch expressions
 * - Value change tracking
 * - Type information
 * - Memory size estimation
 */

// ============================================================================
// Types
// ============================================================================

export interface Variable {
  name: string;
  value: any;
  type: string;
  scope: VariableScope;
  writable: boolean;
  size_bytes?: number;
  reference_count?: number;
}

export type VariableScope = 'local' | 'closure' | 'global' | 'module';

export interface WatchExpression {
  id: string;
  expression: string;
  value: any;
  type: string;
  last_changed: number;
  change_count: number;
}

export interface InspectionResult {
  variable: Variable;
  properties?: Record<string, any>;
  prototype?: any;
  constructor_name?: string;
}

// ============================================================================
// Variable Inspector
// ============================================================================

export class VariableInspector {
  // O(1) storage
  private locals: Map<string, any> = new Map();
  private watches: Map<string, WatchExpression> = new Map();
  private valueHistory: Map<string, any[]> = new Map();

  /**
   * Set local variables (O(n) where n = number of vars, but O(1) per var)
   */
  setLocals(locals: Record<string, any>): void {
    this.locals.clear();
    for (const [name, value] of Object.entries(locals)) {
      this.locals.set(name, value);
    }
  }

  /**
   * Get variable (O(1))
   */
  getVariable(name: string): Variable | null {
    if (!this.locals.has(name)) {
      return null;
    }

    const value = this.locals.get(name);

    return {
      name,
      value,
      type: this.getType(value),
      scope: 'local', // For now, all are local
      writable: true,
      size_bytes: this.estimateSize(value)
    };
  }

  /**
   * Inspect variable deeply (O(1) for access + O(k) for k properties)
   */
  inspect(name: string): InspectionResult | null {
    const variable = this.getVariable(name);
    if (!variable) {
      return null;
    }

    const result: InspectionResult = { variable };

    // Inspect object properties
    if (typeof variable.value === 'object' && variable.value !== null) {
      result.properties = {};

      // Get own properties
      for (const key of Object.keys(variable.value)) {
        result.properties[key] = variable.value[key];
      }

      // Get prototype
      const proto = Object.getPrototypeOf(variable.value);
      if (proto !== null && proto !== Object.prototype) {
        result.prototype = proto.constructor?.name ?? 'Object';
      }

      // Get constructor name
      result.constructor_name = variable.value.constructor?.name;
    }

    return result;
  }

  /**
   * Get all variables
   */
  getAllVariables(): Variable[] {
    const variables: Variable[] = [];

    for (const [name] of this.locals) {
      const variable = this.getVariable(name);
      if (variable) {
        variables.push(variable);
      }
    }

    return variables;
  }

  /**
   * Add watch expression (O(1))
   */
  addWatch(expression: string, evaluator: () => any): string {
    const id = `watch-${this.watches.size}`;

    const value = this.safeEvaluate(evaluator);

    const watch: WatchExpression = {
      id,
      expression,
      value,
      type: this.getType(value),
      last_changed: Date.now(),
      change_count: 0
    };

    this.watches.set(id, watch);

    return id;
  }

  /**
   * Update watch expression (O(1))
   */
  updateWatch(id: string, evaluator: () => any): boolean {
    const watch = this.watches.get(id);
    if (!watch) {
      return false;
    }

    const newValue = this.safeEvaluate(evaluator);

    // Check if value changed
    if (!this.isEqual(watch.value, newValue)) {
      watch.value = newValue;
      watch.type = this.getType(newValue);
      watch.last_changed = Date.now();
      watch.change_count++;
    }

    return true;
  }

  /**
   * Remove watch (O(1))
   */
  removeWatch(id: string): boolean {
    return this.watches.delete(id);
  }

  /**
   * Get watch (O(1))
   */
  getWatch(id: string): WatchExpression | undefined {
    return this.watches.get(id);
  }

  /**
   * Get all watches
   */
  getAllWatches(): WatchExpression[] {
    return Array.from(this.watches.values());
  }

  /**
   * Track value changes
   */
  trackValue(name: string, value: any): void {
    if (!this.valueHistory.has(name)) {
      this.valueHistory.set(name, []);
    }

    this.valueHistory.get(name)!.push({
      value,
      timestamp: Date.now()
    });

    // Keep only last 100 values (prevent memory leak)
    const history = this.valueHistory.get(name)!;
    if (history.length > 100) {
      history.shift();
    }
  }

  /**
   * Get value history
   */
  getValueHistory(name: string): any[] {
    return this.valueHistory.get(name) ?? [];
  }

  /**
   * Evaluate expression safely
   */
  evaluate(expression: string): { success: boolean; value?: any; error?: string } {
    try {
      // In production, would use a safe expression evaluator
      // For now, just return the variable value if it exists
      const value = this.locals.get(expression);
      if (value !== undefined) {
        return { success: true, value };
      }

      return { success: false, error: `Variable '${expression}' not found` };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  /**
   * Clear all data
   */
  clear(): void {
    this.locals.clear();
    this.watches.clear();
    this.valueHistory.clear();
  }

  // =========================================================================
  // Private Helpers
  // =========================================================================

  private getType(value: any): string {
    if (value === null) return 'null';
    if (value === undefined) return 'undefined';
    if (Array.isArray(value)) return 'array';

    const type = typeof value;

    if (type === 'object') {
      return value.constructor?.name ?? 'object';
    }

    return type;
  }

  private estimateSize(value: any): number {
    if (value === null || value === undefined) return 0;

    const type = typeof value;

    switch (type) {
      case 'boolean':
        return 4;
      case 'number':
        return 8;
      case 'string':
        return value.length * 2; // UTF-16
      case 'object':
        return this.estimateObjectSize(value);
      default:
        return 8; // Default
    }
  }

  private estimateObjectSize(obj: any): number {
    let size = 0;

    if (Array.isArray(obj)) {
      for (const item of obj) {
        size += this.estimateSize(item);
      }
    } else {
      for (const [key, value] of Object.entries(obj)) {
        size += key.length * 2; // Key
        size += this.estimateSize(value); // Value
      }
    }

    return size;
  }

  private isEqual(a: any, b: any): boolean {
    if (a === b) return true;
    if (a === null || b === null) return false;
    if (typeof a !== typeof b) return false;

    if (typeof a === 'object') {
      return JSON.stringify(a) === JSON.stringify(b);
    }

    return false;
  }

  private safeEvaluate(evaluator: () => any): any {
    try {
      return evaluator();
    } catch (error) {
      return { error: (error as Error).message };
    }
  }
}

// ============================================================================
// Scope Inspector (for closure/global scopes)
// ============================================================================

export class ScopeInspector {
  private scopes: Map<VariableScope, Map<string, any>> = new Map();

  constructor() {
    this.scopes.set('local', new Map());
    this.scopes.set('closure', new Map());
    this.scopes.set('global', new Map());
    this.scopes.set('module', new Map());
  }

  /**
   * Set variables in scope (O(1) per variable)
   */
  setScope(scope: VariableScope, variables: Record<string, any>): void {
    const scopeMap = this.scopes.get(scope)!;
    scopeMap.clear();

    for (const [name, value] of Object.entries(variables)) {
      scopeMap.set(name, value);
    }
  }

  /**
   * Get variable from any scope (O(1) per scope)
   */
  getVariable(name: string): Variable | null {
    // Search in order: local -> closure -> module -> global
    const searchOrder: VariableScope[] = ['local', 'closure', 'module', 'global'];

    for (const scope of searchOrder) {
      const scopeMap = this.scopes.get(scope)!;
      if (scopeMap.has(name)) {
        const value = scopeMap.get(name);
        return {
          name,
          value,
          type: typeof value,
          scope,
          writable: scope !== 'global' // globals are readonly
        };
      }
    }

    return null;
  }

  /**
   * Get all variables in scope
   */
  getScopeVariables(scope: VariableScope): Variable[] {
    const scopeMap = this.scopes.get(scope)!;
    const variables: Variable[] = [];

    for (const [name, value] of scopeMap) {
      variables.push({
        name,
        value,
        type: typeof value,
        scope,
        writable: scope !== 'global'
      });
    }

    return variables;
  }

  /**
   * Get all scopes summary
   */
  getSummary(): Record<VariableScope, number> {
    return {
      local: this.scopes.get('local')!.size,
      closure: this.scopes.get('closure')!.size,
      global: this.scopes.get('global')!.size,
      module: this.scopes.get('module')!.size
    };
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create variable inspector
 */
export function createInspector(): VariableInspector {
  return new VariableInspector();
}

/**
 * Create scope inspector
 */
export function createScopeInspector(): ScopeInspector {
  return new ScopeInspector();
}
