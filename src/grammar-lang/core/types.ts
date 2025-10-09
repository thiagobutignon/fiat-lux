/**
 * Grammar Language Type System
 *
 * O(1) type checking - no inference, explicit types only
 * Types are grammar rules
 */

// ============================================================================
// Core Type Definitions
// ============================================================================

export type Type =
  | PrimitiveType
  | CompoundType;

export type PrimitiveType =
  | { kind: 'integer' }
  | { kind: 'string' }
  | { kind: 'boolean' }
  | { kind: 'unit' };

export type CompoundType =
  | ListType
  | RecordType
  | EnumType
  | FunctionType
  | TypeVariable;

export interface ListType {
  kind: 'list';
  element: Type;
}

export interface RecordType {
  kind: 'record';
  fields: Map<string, Type>;
}

export interface EnumType {
  kind: 'enum';
  variants: Map<string, Type | null>;
}

export interface FunctionType {
  kind: 'function';
  params: Type[];
  return: Type;
}

export interface TypeVariable {
  kind: 'typevar';
  name: string;
}

// ============================================================================
// Type Environment
// ============================================================================

/**
 * Type environment for O(1) lookup
 */
export class TypeEnv {
  private bindings: Map<string, Type>;
  private parent?: TypeEnv;

  constructor(parent?: TypeEnv) {
    this.bindings = new Map();
    this.parent = parent;
  }

  /**
   * O(1) - bind a variable to a type
   */
  bind(name: string, type: Type): void {
    this.bindings.set(name, type);
  }

  /**
   * O(1) - lookup a variable's type
   */
  lookup(name: string): Type | undefined {
    const local = this.bindings.get(name);
    if (local) return local;
    return this.parent?.lookup(name);
  }

  /**
   * O(1) - create child environment
   */
  extend(): TypeEnv {
    return new TypeEnv(this);
  }
}

// ============================================================================
// Type Equality (O(1) - structural equality)
// ============================================================================

/**
 * O(1) type equality check
 * No inference - types must match exactly
 */
export function typeEquals(t1: Type, t2: Type): boolean {
  // Primitive types
  if (t1.kind === 'integer' && t2.kind === 'integer') return true;
  if (t1.kind === 'string' && t2.kind === 'string') return true;
  if (t1.kind === 'boolean' && t2.kind === 'boolean') return true;
  if (t1.kind === 'unit' && t2.kind === 'unit') return true;

  // List type
  if (t1.kind === 'list' && t2.kind === 'list') {
    return typeEquals(t1.element, t2.element);
  }

  // Function type
  if (t1.kind === 'function' && t2.kind === 'function') {
    if (t1.params.length !== t2.params.length) return false;
    for (let i = 0; i < t1.params.length; i++) {
      if (!typeEquals(t1.params[i], t2.params[i])) return false;
    }
    return typeEquals(t1.return, t2.return);
  }

  // Record type
  if (t1.kind === 'record' && t2.kind === 'record') {
    if (t1.fields.size !== t2.fields.size) return false;
    for (const [name, type] of t1.fields) {
      const t2Type = t2.fields.get(name);
      if (!t2Type || !typeEquals(type, t2Type)) return false;
    }
    return true;
  }

  // Enum type
  if (t1.kind === 'enum' && t2.kind === 'enum') {
    if (t1.variants.size !== t2.variants.size) return false;
    for (const [name, type] of t1.variants) {
      const t2Type = t2.variants.get(name);
      if (type === null && t2Type !== null) return false;
      if (type !== null && (t2Type === null || !typeEquals(type, t2Type))) {
        return false;
      }
    }
    return true;
  }

  // Type variable
  if (t1.kind === 'typevar' && t2.kind === 'typevar') {
    return t1.name === t2.name;
  }

  return false;
}

// ============================================================================
// Type Constructors (Helper functions)
// ============================================================================

export const Types = {
  integer: (): Type => ({ kind: 'integer' }),
  string: (): Type => ({ kind: 'string' }),
  boolean: (): Type => ({ kind: 'boolean' }),
  unit: (): Type => ({ kind: 'unit' }),

  list: (element: Type): Type => ({ kind: 'list', element }),

  record: (fields: [string, Type][]): Type => ({
    kind: 'record',
    fields: new Map(fields)
  }),

  enum: (variants: [string, Type | null][]): Type => ({
    kind: 'enum',
    variants: new Map(variants)
  }),

  function: (params: Type[], ret: Type): Type => ({
    kind: 'function',
    params,
    return: ret
  }),

  typevar: (name: string): Type => ({ kind: 'typevar', name })
};

// ============================================================================
// Type Formatting
// ============================================================================

export function formatType(t: Type): string {
  switch (t.kind) {
    case 'integer': return 'integer';
    case 'string': return 'string';
    case 'boolean': return 'boolean';
    case 'unit': return 'unit';

    case 'list':
      return `(list ${formatType(t.element)})`;

    case 'function': {
      const params = t.params.map(formatType).join(' ');
      return `(${params} -> ${formatType(t.return)})`;
    }

    case 'record': {
      const fields = Array.from(t.fields.entries())
        .map(([name, type]) => `(${name} ${formatType(type)})`)
        .join(' ');
      return `(record ${fields})`;
    }

    case 'enum': {
      const variants = Array.from(t.variants.entries())
        .map(([name, type]) =>
          type ? `(${name} ${formatType(type)})` : name
        )
        .join(' ');
      return `(enum ${variants})`;
    }

    case 'typevar':
      return t.name;
  }
}
