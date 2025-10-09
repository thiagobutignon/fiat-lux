/**
 * Grammar Language Type Checker
 *
 * O(1) type checking per expression
 * No inference - all types must be explicit
 */

import { Expr, Definition, FunctionDef, TypeDef, ModuleDef } from './ast';
import { Type, TypeEnv, typeEquals, formatType } from './types';

// ============================================================================
// Type Checker Errors
// ============================================================================

export class TypeCheckError extends Error {
  constructor(message: string, public loc?: any) {
    super(message);
    this.name = 'TypeCheckError';
  }
}

// ============================================================================
// Expression Type Checker (O(1) per expression)
// ============================================================================

/**
 * Type check an expression in O(1)
 * Returns the type of the expression or throws TypeCheckError
 */
export function checkExpr(expr: Expr, env: TypeEnv): Type {
  switch (expr.kind) {
    case 'literal':
      // O(1) - literal has explicit type
      return expr.type;

    case 'var':
      // O(1) - lookup in environment
      const varType = env.lookup(expr.name);
      if (!varType) {
        throw new TypeCheckError(
          `Undefined variable: ${expr.name}`,
          expr.loc
        );
      }
      return varType;

    case 'let': {
      // O(1) - check value matches declared type
      const valueType = checkExpr(expr.value, env);
      if (!typeEquals(valueType, expr.type)) {
        throw new TypeCheckError(
          `Type mismatch in let binding: expected ${formatType(expr.type)}, got ${formatType(valueType)}`,
          expr.loc
        );
      }

      // If there's a body, check it in extended environment
      if (expr.body) {
        const bodyEnv = env.extend();
        bodyEnv.bind(expr.name, expr.type);
        return checkExpr(expr.body, bodyEnv);
      }

      return expr.type;
    }

    case 'if': {
      // O(1) - check condition is boolean
      const condType = checkExpr(expr.condition, env);
      if (condType.kind !== 'boolean') {
        throw new TypeCheckError(
          `If condition must be boolean, got ${formatType(condType)}`,
          expr.loc
        );
      }

      // O(1) - check branches have same type
      const thenType = checkExpr(expr.then, env);
      const elseType = checkExpr(expr.else, env);

      if (!typeEquals(thenType, elseType)) {
        throw new TypeCheckError(
          `If branches must have same type: then=${formatType(thenType)}, else=${formatType(elseType)}`,
          expr.loc
        );
      }

      return thenType;
    }

    case 'call': {
      // O(1) - check function type
      const fnType = checkExpr(expr.fn, env);
      if (fnType.kind !== 'function') {
        throw new TypeCheckError(
          `Cannot call non-function: ${formatType(fnType)}`,
          expr.loc
        );
      }

      // O(1) - check argument count
      if (expr.args.length !== fnType.params.length) {
        throw new TypeCheckError(
          `Wrong number of arguments: expected ${fnType.params.length}, got ${expr.args.length}`,
          expr.loc
        );
      }

      // O(k) where k = number of args - check each arg type
      for (let i = 0; i < expr.args.length; i++) {
        const argType = checkExpr(expr.args[i], env);
        const paramType = fnType.params[i];

        if (!typeEquals(argType, paramType)) {
          throw new TypeCheckError(
            `Argument ${i + 1} type mismatch: expected ${formatType(paramType)}, got ${formatType(argType)}`,
            expr.loc
          );
        }
      }

      return fnType.return;
    }

    case 'lambda': {
      // O(1) - create function type
      const paramTypes = expr.params.map(([_, type]) => type);

      // Check body in extended environment
      const bodyEnv = env.extend();
      for (const [name, type] of expr.params) {
        bodyEnv.bind(name, type);
      }

      const bodyType = checkExpr(expr.body, bodyEnv);

      return {
        kind: 'function',
        params: paramTypes,
        return: bodyType
      };
    }

    case 'list': {
      // O(n) where n = list length - check all elements have same type
      if (expr.elements.length === 0) {
        throw new TypeCheckError(
          'Cannot infer type of empty list - use explicit type annotation',
          expr.loc
        );
      }

      const firstType = checkExpr(expr.elements[0], env);

      for (let i = 1; i < expr.elements.length; i++) {
        const elemType = checkExpr(expr.elements[i], env);
        if (!typeEquals(elemType, firstType)) {
          throw new TypeCheckError(
            `List elements must have same type: expected ${formatType(firstType)}, got ${formatType(elemType)} at index ${i}`,
            expr.loc
          );
        }
      }

      return { kind: 'list', element: firstType };
    }

    case 'record': {
      // O(k) where k = number of fields
      const fieldTypes = new Map<string, Type>();

      for (const [name, expr] of expr.fields) {
        fieldTypes.set(name, checkExpr(expr, env));
      }

      return { kind: 'record', fields: fieldTypes };
    }
  }
}

// ============================================================================
// Definition Type Checker
// ============================================================================

export function checkDefinition(def: Definition, env: TypeEnv): void {
  switch (def.kind) {
    case 'function': {
      // Forward declare function (for recursion)
      const fnType: Type = {
        kind: 'function',
        params: def.params.map(([_, t]) => t),
        return: def.returnType
      };
      env.bind(def.name, fnType);

      // O(1) - create function environment with parameters
      const fnEnv = env.extend();
      for (const [name, type] of def.params) {
        fnEnv.bind(name, type);
      }

      // Check body type matches declared return type
      const bodyType = checkExpr(def.body, fnEnv);
      if (!typeEquals(bodyType, def.returnType)) {
        throw new TypeCheckError(
          `Function ${def.name} body type mismatch: expected ${formatType(def.returnType)}, got ${formatType(bodyType)}`,
          def.loc
        );
      }

      break;
    }

    case 'typedef':
      // O(1) - just register the type alias
      env.bind(def.name, def.type);
      break;

    case 'module': {
      // O(n) where n = number of definitions
      const moduleEnv = env.extend();

      // Process imports
      for (const imp of def.imports) {
        // TODO: resolve module imports
      }

      // Check all definitions
      for (const definition of def.definitions) {
        checkDefinition(definition, moduleEnv);
      }
      break;
    }
  }
}

// ============================================================================
// Program Type Checker
// ============================================================================

export function checkProgram(definitions: Definition[]): TypeEnv {
  const env = new TypeEnv();

  // Add built-in types and functions
  addBuiltins(env);

  // Type check all definitions in order
  for (const def of definitions) {
    checkDefinition(def, env);
  }

  return env;
}

function addBuiltins(env: TypeEnv): void {
  // Primitive operators
  env.bind('+', { kind: 'function', params: [{ kind: 'integer' }, { kind: 'integer' }], return: { kind: 'integer' } });
  env.bind('-', { kind: 'function', params: [{ kind: 'integer' }, { kind: 'integer' }], return: { kind: 'integer' } });
  env.bind('*', { kind: 'function', params: [{ kind: 'integer' }, { kind: 'integer' }], return: { kind: 'integer' } });
  env.bind('/', { kind: 'function', params: [{ kind: 'integer' }, { kind: 'integer' }], return: { kind: 'integer' } });

  env.bind('=', { kind: 'function', params: [{ kind: 'integer' }, { kind: 'integer' }], return: { kind: 'boolean' } });
  env.bind('<', { kind: 'function', params: [{ kind: 'integer' }, { kind: 'integer' }], return: { kind: 'boolean' } });
  env.bind('<=', { kind: 'function', params: [{ kind: 'integer' }, { kind: 'integer' }], return: { kind: 'boolean' } });
  env.bind('>', { kind: 'function', params: [{ kind: 'integer' }, { kind: 'integer' }], return: { kind: 'boolean' } });
  env.bind('>=', { kind: 'function', params: [{ kind: 'integer' }, { kind: 'integer' }], return: { kind: 'boolean' } });

  // Boolean operators
  env.bind('and', { kind: 'function', params: [{ kind: 'boolean' }, { kind: 'boolean' }], return: { kind: 'boolean' } });
  env.bind('or', { kind: 'function', params: [{ kind: 'boolean' }, { kind: 'boolean' }], return: { kind: 'boolean' } });
  env.bind('not', { kind: 'function', params: [{ kind: 'boolean' }], return: { kind: 'boolean' } });
}
