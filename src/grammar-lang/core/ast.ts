/**
 * Grammar Language AST
 *
 * Abstract Syntax Tree for Grammar Language
 * S-expression based, directly from Grammar Engine
 */

import { Type } from './types';

// ============================================================================
// Expression AST
// ============================================================================

export type Expr =
  | LiteralExpr
  | VarExpr
  | LetExpr
  | IfExpr
  | FunctionCallExpr
  | LambdaExpr
  | ListExpr
  | RecordExpr;

export interface LiteralExpr {
  kind: 'literal';
  value: number | string | boolean | null;
  type: Type;
  loc?: SourceLocation;
}

export interface VarExpr {
  kind: 'var';
  name: string;
  loc?: SourceLocation;
}

export interface LetExpr {
  kind: 'let';
  name: string;
  type: Type;
  value: Expr;
  body?: Expr; // Optional body (let can be standalone binding)
  loc?: SourceLocation;
}

export interface IfExpr {
  kind: 'if';
  condition: Expr;
  then: Expr;
  else: Expr;
  loc?: SourceLocation;
}

export interface FunctionCallExpr {
  kind: 'call';
  fn: Expr;
  args: Expr[];
  loc?: SourceLocation;
}

export interface LambdaExpr {
  kind: 'lambda';
  params: [string, Type][];
  body: Expr;
  loc?: SourceLocation;
}

export interface ListExpr {
  kind: 'list';
  elements: Expr[];
  loc?: SourceLocation;
}

export interface RecordExpr {
  kind: 'record';
  fields: Map<string, Expr>;
  loc?: SourceLocation;
}

// ============================================================================
// Definition AST
// ============================================================================

export type Definition =
  | FunctionDef
  | TypeDef
  | ModuleDef;

export interface FunctionDef {
  kind: 'function';
  name: string;
  params: [string, Type][];
  returnType: Type;
  body: Expr;
  exported: boolean;
  loc?: SourceLocation;
}

export interface TypeDef {
  kind: 'typedef';
  name: string;
  type: Type;
  exported: boolean;
  loc?: SourceLocation;
}

export interface ModuleDef {
  kind: 'module';
  name: string;
  imports: ImportDecl[];
  exports: string[];
  definitions: Definition[];
  loc?: SourceLocation;
}

export interface ImportDecl {
  module: string;
  names: string[];
  loc?: SourceLocation;
}

// ============================================================================
// Source Location
// ============================================================================

export interface SourceLocation {
  file: string;
  line: number;
  column: number;
}

// ============================================================================
// AST Utilities
// ============================================================================

export function isExpr(node: any): node is Expr {
  return node && typeof node.kind === 'string' &&
    ['literal', 'var', 'let', 'if', 'call', 'lambda', 'list', 'record'].includes(node.kind);
}

export function isDef(node: any): node is Definition {
  return node && typeof node.kind === 'string' &&
    ['function', 'typedef', 'module'].includes(node.kind);
}
