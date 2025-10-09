/**
 * Grammar Language Parser
 *
 * Parses S-expressions into Grammar Language AST
 * Uses Grammar Engine for O(1) parsing
 */

import { Expr, Definition, FunctionDef, TypeDef, ModuleDef } from '../core/ast';
import { Type, Types } from '../core/types';

// ============================================================================
// S-Expression Types (from Grammar Engine)
// ============================================================================

type SExpr = string | number | boolean | null | SExpr[];

// ============================================================================
// Parser Errors
// ============================================================================

export class ParseError extends Error {
  constructor(message: string, public sexpr?: SExpr) {
    super(message);
    this.name = 'ParseError';
  }
}

// ============================================================================
// Type Parser
// ============================================================================

export function parseType(sexpr: SExpr): Type {
  if (typeof sexpr === 'string') {
    // Primitive type or type variable
    switch (sexpr) {
      case 'integer': return Types.integer();
      case 'string': return Types.string();
      case 'boolean': return Types.boolean();
      case 'unit': return Types.unit();
      default: return Types.typevar(sexpr);
    }
  }

  if (!Array.isArray(sexpr)) {
    throw new ParseError('Expected type expression', sexpr);
  }

  // Check for function type first: (T1 T2 ... -> R) or [T1 T2 -> R]
  const arrowIdx = sexpr.findIndex(x => x === '->');
  if (arrowIdx !== -1) {
    const params = sexpr.slice(0, arrowIdx).map(parseType);
    const returnType = parseType(sexpr[arrowIdx + 1]);
    return Types.function(params, returnType);
  }

  const [head, ...rest] = sexpr;

  if (head === 'list') {
    // (list T)
    if (rest.length !== 1) {
      throw new ParseError('List type requires exactly 1 argument', sexpr);
    }
    return Types.list(parseType(rest[0]));
  }

  if (head === 'record') {
    // (record (field1 T1) (field2 T2) ...)
    const fields: [string, Type][] = [];
    for (const field of rest) {
      if (!Array.isArray(field) || field.length !== 2) {
        throw new ParseError('Record field must be (name type)', field);
      }
      const [name, type] = field;
      if (typeof name !== 'string') {
        throw new ParseError('Field name must be string', name);
      }
      fields.push([name, parseType(type)]);
    }
    return Types.record(fields);
  }

  if (head === 'enum') {
    // (enum (Variant1 T1) Variant2 ...)
    const variants: [string, Type | null][] = [];
    for (const variant of rest) {
      if (typeof variant === 'string') {
        variants.push([variant, null]);
      } else if (Array.isArray(variant) && variant.length === 2) {
        const [name, type] = variant;
        if (typeof name !== 'string') {
          throw new ParseError('Variant name must be string', name);
        }
        variants.push([name, parseType(type)]);
      } else {
        throw new ParseError('Invalid enum variant', variant);
      }
    }
    return Types.enum(variants);
  }

  throw new ParseError('Unknown type expression', sexpr);
}

// ============================================================================
// Expression Parser
// ============================================================================

export function parseExpr(sexpr: SExpr): Expr {
  // Literals
  if (typeof sexpr === 'number') {
    return {
      kind: 'literal',
      value: sexpr,
      type: Types.integer()
    };
  }

  if (typeof sexpr === 'string') {
    // String literal starts with "
    if (sexpr.startsWith('"')) {
      return {
        kind: 'literal',
        value: sexpr.slice(1, -1), // Remove quotes
        type: Types.string()
      };
    }

    // Boolean literals
    if (sexpr === 'true' || sexpr === 'false') {
      return {
        kind: 'literal',
        value: sexpr === 'true',
        type: Types.boolean()
      };
    }

    // Variable
    return {
      kind: 'var',
      name: sexpr
    };
  }

  if (sexpr === null) {
    return {
      kind: 'literal',
      value: null,
      type: Types.unit()
    };
  }

  if (!Array.isArray(sexpr)) {
    throw new ParseError('Expected expression', sexpr);
  }

  const [head, ...args] = sexpr;

  // (let name type value)
  if (head === 'let') {
    if (args.length !== 3) {
      throw new ParseError('Let requires 3 arguments: name, type, value', sexpr);
    }
    const [name, typeExpr, valueExpr] = args;
    if (typeof name !== 'string') {
      throw new ParseError('Let name must be string', name);
    }
    return {
      kind: 'let',
      name,
      type: parseType(typeExpr),
      value: parseExpr(valueExpr)
    };
  }

  // (if cond then else)
  if (head === 'if') {
    if (args.length !== 3) {
      throw new ParseError('If requires 3 arguments', sexpr);
    }
    return {
      kind: 'if',
      condition: parseExpr(args[0]),
      then: parseExpr(args[1]),
      else: parseExpr(args[2])
    };
  }

  // (lambda (params) body)
  if (head === 'lambda') {
    if (args.length !== 2) {
      throw new ParseError('Lambda requires 2 arguments', sexpr);
    }
    const [paramsExpr, bodyExpr] = args;
    if (!Array.isArray(paramsExpr)) {
      throw new ParseError('Lambda params must be list', paramsExpr);
    }

    const params: [string, Type][] = [];
    for (const param of paramsExpr) {
      if (!Array.isArray(param) || param.length !== 2) {
        throw new ParseError('Lambda param must be (name type)', param);
      }
      const [name, typeExpr] = param;
      if (typeof name !== 'string') {
        throw new ParseError('Param name must be string', name);
      }
      params.push([name, parseType(typeExpr)]);
    }

    return {
      kind: 'lambda',
      params,
      body: parseExpr(bodyExpr)
    };
  }

  // Function call: (fn arg1 arg2 ...)
  return {
    kind: 'call',
    fn: parseExpr(head),
    args: args.map(parseExpr)
  };
}

// ============================================================================
// Definition Parser
// ============================================================================

export function parseDefinition(sexpr: SExpr): Definition {
  if (!Array.isArray(sexpr)) {
    throw new ParseError('Expected definition', sexpr);
  }

  const [kind, ...args] = sexpr;

  // (define name [(params) -> return] body)
  // OR (define name (params -> return) body)
  if (kind === 'define') {
    if (args.length !== 3) {
      throw new ParseError('Define requires 3 arguments', sexpr);
    }
    const [nameExpr, typeExpr, bodyExpr] = args;

    if (typeof nameExpr !== 'string') {
      throw new ParseError('Function name must be string', nameExpr);
    }

    // Parse function type signature
    // Can be either [T1 T2 -> R] or (T1 T2 -> R)
    let funcType: Type;
    if (Array.isArray(typeExpr) && typeExpr[0] !== '->') {
      // It's a type expression like (integer -> integer)
      funcType = parseType(typeExpr);
    } else if (Array.isArray(typeExpr)) {
      // It's [T1 T2 -> R] format
      funcType = parseType(typeExpr);
    } else {
      throw new ParseError('Expected function type', typeExpr);
    }

    if (funcType.kind !== 'function') {
      throw new ParseError('Function definition requires function type', typeExpr);
    }

    // Generate param names ($1, $2, ...)
    const params: [string, Type][] = funcType.params.map((t, i) => [`$${i + 1}`, t]);

    return {
      kind: 'function',
      name: nameExpr,
      params,
      returnType: funcType.return,
      body: parseExpr(bodyExpr),
      exported: false
    };
  }

  // (type name T)
  if (kind === 'type') {
    if (args.length !== 2) {
      throw new ParseError('Type definition requires 2 arguments', sexpr);
    }
    const [nameExpr, typeExpr] = args;

    if (typeof nameExpr !== 'string') {
      throw new ParseError('Type name must be string', nameExpr);
    }

    return {
      kind: 'typedef',
      name: nameExpr,
      type: parseType(typeExpr),
      exported: false
    };
  }

  // (module name (export ...) definitions...)
  // OR (module name definitions...)
  if (kind === 'module') {
    if (args.length < 1) {
      throw new ParseError('Module requires name', sexpr);
    }
    const [nameExpr, ...rest] = args;

    if (typeof nameExpr !== 'string') {
      throw new ParseError('Module name must be string', nameExpr);
    }

    let exports: string[] = [];
    let defsStart = 0;

    // Check if second arg is export declaration: (export name1 name2 ...)
    if (rest.length > 0 && Array.isArray(rest[0]) && rest[0][0] === 'export') {
      const exportExpr = rest[0];
      exports = exportExpr.slice(1).map(e => {
        if (typeof e !== 'string') {
          throw new ParseError('Export name must be string', e);
        }
        return e;
      });
      defsStart = 1;
    }

    const definitions = rest.slice(defsStart).map(parseDefinition);

    // Mark exported definitions
    for (const def of definitions) {
      if (def.kind !== 'module' && exports.includes(def.name)) {
        def.exported = true;
      }
    }

    return {
      kind: 'module',
      name: nameExpr,
      imports: [],
      exports,
      definitions
    };
  }

  // (import module (names...))
  // OR (import (std module) (names...))
  if (kind === 'import') {
    if (args.length !== 2) {
      throw new ParseError('Import requires 2 arguments: module and names', sexpr);
    }
    const [moduleExpr, namesExpr] = args;

    let moduleName: string;
    if (typeof moduleExpr === 'string') {
      moduleName = moduleExpr;
    } else if (Array.isArray(moduleExpr)) {
      // (std module) format
      moduleName = moduleExpr.join('/');
    } else {
      throw new ParseError('Module name must be string or list', moduleExpr);
    }

    if (!Array.isArray(namesExpr)) {
      throw new ParseError('Import names must be list', namesExpr);
    }

    const names = namesExpr.map(n => {
      if (typeof n !== 'string') {
        throw new ParseError('Import name must be string', n);
      }
      return n;
    });

    // Import is not a full definition, store as placeholder
    // Will be handled specially in module resolution
    return {
      kind: 'module',
      name: '__import__',
      imports: [{ module: moduleName, names }],
      exports: [],
      definitions: []
    };
  }

  throw new ParseError('Unknown definition kind', sexpr);
}

// ============================================================================
// Program Parser
// ============================================================================

export function parseProgram(sexprs: SExpr[]): Definition[] {
  return sexprs.map(parseDefinition);
}
