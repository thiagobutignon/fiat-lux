#!/usr/bin/env tsx
/**
 * Grammar Language LSP Server
 *
 * Provides editor integration:
 * - Diagnostics (type errors, parse errors)
 * - Go to definition
 * - Autocomplete
 * - Hover documentation
 * - Rename refactoring
 */

import {
  createConnection,
  TextDocuments,
  Diagnostic,
  DiagnosticSeverity,
  ProposedFeatures,
  InitializeParams,
  DidChangeConfigurationNotification,
  CompletionItem,
  CompletionItemKind,
  TextDocumentPositionParams,
  TextDocumentSyncKind,
  InitializeResult,
  Location,
  Range,
  Position,
  Hover,
  MarkupKind
} from 'vscode-languageserver/node';

import { TextDocument } from 'vscode-languageserver-textdocument';
import { parseProgram } from '../compiler/parser';
import { checkProgram } from '../core/type-checker';
import { ModuleRegistry } from '../compiler/module-resolver';
import { BUILTINS } from '../stdlib/builtins';
import * as path from 'path';

// ============================================================================
// LSP Server Connection
// ============================================================================

const connection = createConnection(ProposedFeatures.all);
const documents = new TextDocuments(TextDocument);

let hasConfigurationCapability = false;
let hasWorkspaceFolderCapability = false;
let hasDiagnosticRelatedInformationCapability = false;

// ============================================================================
// Document State
// ============================================================================

interface DocumentInfo {
  uri: string;
  content: string;
  ast: any[];
  errors: Diagnostic[];
  symbols: Map<string, SymbolInfo>;
}

interface SymbolInfo {
  name: string;
  type: string;
  kind: 'function' | 'type' | 'variable';
  location: Range;
  documentation?: string;
}

const documentCache = new Map<string, DocumentInfo>();
const moduleRegistry = new ModuleRegistry(process.cwd());

// ============================================================================
// Initialization
// ============================================================================

connection.onInitialize((params: InitializeParams) => {
  const capabilities = params.capabilities;

  hasConfigurationCapability = !!(
    capabilities.workspace && !!capabilities.workspace.configuration
  );
  hasWorkspaceFolderCapability = !!(
    capabilities.workspace && !!capabilities.workspace.workspaceFolders
  );
  hasDiagnosticRelatedInformationCapability = !!(
    capabilities.textDocument &&
    capabilities.textDocument.publishDiagnostics &&
    capabilities.textDocument.publishDiagnostics.relatedInformation
  );

  const result: InitializeResult = {
    capabilities: {
      textDocumentSync: TextDocumentSyncKind.Incremental,
      completionProvider: {
        resolveProvider: true,
        triggerCharacters: ['(', ' ', '$']
      },
      hoverProvider: true,
      definitionProvider: true,
      referencesProvider: true,
      renameProvider: true
    }
  };

  if (hasWorkspaceFolderCapability) {
    result.capabilities.workspace = {
      workspaceFolders: {
        supported: true
      }
    };
  }

  return result;
});

connection.onInitialized(() => {
  if (hasConfigurationCapability) {
    connection.client.register(DidChangeConfigurationNotification.type, undefined);
  }

  if (hasWorkspaceFolderCapability) {
    connection.workspace.onDidChangeWorkspaceFolders(_event => {
      connection.console.log('Workspace folder change event received.');
    });
  }

  connection.console.log('Grammar Language LSP Server initialized');
});

// ============================================================================
// Diagnostics
// ============================================================================

async function validateTextDocument(textDocument: TextDocument): Promise<void> {
  const text = textDocument.getText();
  const diagnostics: Diagnostic[] = [];
  const symbols = new Map<string, SymbolInfo>();

  try {
    // Parse S-expressions (simplified - in production use Grammar Engine)
    const sexprs = parseSource(text);
    const ast = parseProgram(sexprs);

    // Type check
    try {
      checkProgram(ast);

      // Extract symbols for autocomplete/hover
      for (const def of ast) {
        if (def.kind === 'function') {
          symbols.set(def.name, {
            name: def.name,
            type: formatType(def.returnType),
            kind: 'function',
            location: def.loc || { start: { line: 0, character: 0 }, end: { line: 0, character: 0 } },
            documentation: `Function: ${def.name}`
          });
        } else if (def.kind === 'typedef') {
          symbols.set(def.name, {
            name: def.name,
            type: formatType(def.type),
            kind: 'type',
            location: def.loc || { start: { line: 0, character: 0 }, end: { line: 0, character: 0 } },
            documentation: `Type: ${def.name}`
          });
        }
      }

      // Cache document info
      documentCache.set(textDocument.uri, {
        uri: textDocument.uri,
        content: text,
        ast,
        errors: [],
        symbols
      });

    } catch (typeError: any) {
      // Type error
      diagnostics.push({
        severity: DiagnosticSeverity.Error,
        range: {
          start: { line: 0, character: 0 },
          end: { line: 0, character: text.length }
        },
        message: typeError.message,
        source: 'Grammar Language Type Checker'
      });
    }

  } catch (parseError: any) {
    // Parse error
    diagnostics.push({
      severity: DiagnosticSeverity.Error,
      range: {
        start: { line: 0, character: 0 },
        end: { line: 0, character: text.length }
      },
      message: parseError.message,
      source: 'Grammar Language Parser'
    });
  }

  // Send diagnostics
  connection.sendDiagnostics({ uri: textDocument.uri, diagnostics });
}

documents.onDidChangeContent(change => {
  validateTextDocument(change.document);
});

// ============================================================================
// Autocomplete
// ============================================================================

connection.onCompletion(
  (_textDocumentPosition: TextDocumentPositionParams): CompletionItem[] => {
    const items: CompletionItem[] = [];

    // Built-in functions
    for (const builtin of BUILTINS) {
      items.push({
        label: builtin.name,
        kind: CompletionItemKind.Function,
        detail: formatType(builtin.type),
        documentation: `Built-in function: ${builtin.name}`
      });
    }

    // Document symbols
    const doc = documentCache.get(_textDocumentPosition.textDocument.uri);
    if (doc) {
      for (const [name, symbol] of doc.symbols) {
        items.push({
          label: name,
          kind: symbol.kind === 'function' ? CompletionItemKind.Function :
                symbol.kind === 'type' ? CompletionItemKind.Class :
                CompletionItemKind.Variable,
          detail: symbol.type,
          documentation: symbol.documentation
        });
      }
    }

    // Keywords
    const keywords = ['define', 'type', 'module', 'import', 'export', 'if', 'let', 'lambda'];
    for (const kw of keywords) {
      items.push({
        label: kw,
        kind: CompletionItemKind.Keyword
      });
    }

    return items;
  }
);

connection.onCompletionResolve(
  (item: CompletionItem): CompletionItem => {
    // Additional details can be added here
    return item;
  }
);

// ============================================================================
// Hover
// ============================================================================

connection.onHover(
  (_params: TextDocumentPositionParams): Hover | null => {
    const doc = documentCache.get(_params.textDocument.uri);
    if (!doc) return null;

    // Find symbol at position (simplified - needs proper position mapping)
    const line = _params.position.line;
    const char = _params.position.character;

    for (const [name, symbol] of doc.symbols) {
      if (symbol.location.start.line === line) {
        return {
          contents: {
            kind: MarkupKind.Markdown,
            value: [
              '```grammar-language',
              `${symbol.kind}: ${name}`,
              `type: ${symbol.type}`,
              '```',
              '',
              symbol.documentation || ''
            ].join('\n')
          }
        };
      }
    }

    return null;
  }
);

// ============================================================================
// Go to Definition
// ============================================================================

connection.onDefinition(
  (_params: TextDocumentPositionParams): Location | null => {
    const doc = documentCache.get(_params.textDocument.uri);
    if (!doc) return null;

    // Find symbol at cursor (simplified)
    for (const [name, symbol] of doc.symbols) {
      // In production, proper position matching needed
      return {
        uri: _params.textDocument.uri,
        range: symbol.location
      };
    }

    return null;
  }
);

// ============================================================================
// Rename
// ============================================================================

connection.onRenameRequest(
  (_params) => {
    // TODO: Implement rename refactoring
    return null;
  }
);

// ============================================================================
// Utilities
// ============================================================================

function parseSource(source: string): any[] {
  // Remove comments
  source = source.replace(/;[^\n]*/g, '');

  // Simplified parser (in production use Grammar Engine)
  try {
    const jsonLike = source
      .replace(/\(/g, '[')
      .replace(/\)/g, ']')
      .replace(/(\w+)/g, '"$1"')
      .replace(/"(\d+)"/g, '$1')
      .replace(/"true"/g, 'true')
      .replace(/"false"/g, 'false')
      .replace(/"->"/g, '"->"');

    const wrapped = `[${jsonLike}]`;
    return JSON.parse(wrapped);
  } catch {
    return [];
  }
}

function formatType(type: any): string {
  if (!type) return 'unknown';

  switch (type.kind) {
    case 'integer': return 'integer';
    case 'string': return 'string';
    case 'boolean': return 'boolean';
    case 'unit': return 'unit';
    case 'function':
      const params = type.params.map((p: any) => formatType(p)).join(' ');
      const ret = formatType(type.return);
      return `(${params} -> ${ret})`;
    case 'list':
      return `(list ${formatType(type.element)})`;
    case 'typevar':
      return type.name;
    default:
      return 'unknown';
  }
}

// ============================================================================
// Document Management
// ============================================================================

documents.onDidOpen(e => {
  connection.console.log(`Document opened: ${e.document.uri}`);
  validateTextDocument(e.document);
});

documents.onDidClose(e => {
  connection.console.log(`Document closed: ${e.document.uri}`);
  documentCache.delete(e.document.uri);
});

// ============================================================================
// Start Server
// ============================================================================

documents.listen(connection);
connection.listen();
