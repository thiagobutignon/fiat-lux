/**
 * Feature Slice Protocol AST Extensions
 *
 * Additional AST nodes for Feature Slice Protocol directives
 * Extends the base Grammar Language AST
 */

import { Type } from './types';
import { Expr, Definition, SourceLocation } from './ast';

// ============================================================================
// Feature Slice Directives
// ============================================================================

/**
 * Agent Configuration (@agent)
 * Defines the LLM agent's system prompt and configuration
 */
export interface AgentConfigDef {
  kind: 'agent-config';
  name: string;
  domain: string;
  expertise: string[];
  constitutional: string[];
  knowledge: string;
  constraints: string[];
  prompt: {
    role: string;
    tone: string;
    knowledgeSources: string;
    constitutionalPrinciples: string;
    attentionTracking: boolean;
  };
  loc?: SourceLocation;
}

/**
 * Layer Definition (@layer)
 * Organizes code into Clean Architecture layers
 */
export interface LayerDef {
  kind: 'layer';
  layerType: 'domain' | 'data' | 'infrastructure' | 'validation' | 'presentation';
  definitions: Definition[];
  loc?: SourceLocation;
}

/**
 * Observability Configuration (@observable)
 * Defines metrics and traces
 */
export interface ObservabilityDef {
  kind: 'observability';
  metrics: MetricDef[];
  traces: TraceDef[];
  loc?: SourceLocation;
}

export interface MetricDef {
  name: string;
  type: 'counter' | 'gauge' | 'histogram';
  labels?: string[];
  buckets?: number[];
  description: string;
}

export interface TraceDef {
  name: string;
  enabled: boolean;
  export: string;
  description?: string;
}

/**
 * Network Configuration (@network)
 * Defines API routes and inter-agent communication
 */
export interface NetworkDef {
  kind: 'network';
  apiConfig: {
    protocol: string;
    port: number;
    cors: boolean;
  };
  routes: RouteDef[];
  exposedFunctions: ExposedFunctionDef[];
  loc?: SourceLocation;
}

export interface RouteDef {
  method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
  path: string;
  request?: Type;
  response: Type;
  query?: Type;
  auth?: string;
  rateLimit?: string;
  constitutional?: boolean;
  handler: string; // Function name
}

export interface ExposedFunctionDef {
  functionName: string;
  exposedAs: string;
}

/**
 * Storage Configuration (@storage)
 * Defines data persistence strategy
 */
export interface StorageDef {
  kind: 'storage';
  relational?: {
    type: string;
    url: string;
    migrations?: string;
    connectionPool?: number;
  };
  cache?: {
    type: string;
    url: string;
    ttlDefault?: number;
  };
  files?: {
    text: string;
    images: string;
    video: string;
    documents: string;
  };
  embeddings?: {
    type: string;
    dimensions: number;
    distance: string;
  };
  loc?: SourceLocation;
}

/**
 * Multi-tenant Configuration (@multitenant)
 * Defines tenant isolation and compliance
 */
export interface MultitenantDef {
  kind: 'multitenant';
  isolation: 'database' | 'schema' | 'row_level';
  auth: {
    type: string;
    issuer: string;
    audience: string;
  };
  config: {
    llmModel: string;
    knowledgeBase: string;
    rateLimits: string;
  };
  compliance: {
    dataResidency: string;
    auditLogging: string;
  };
  loc?: SourceLocation;
}

/**
 * UI Configuration (@ui)
 * Defines UI components
 */
export interface UIDef {
  kind: 'ui';
  components: ComponentDef[];
  loc?: SourceLocation;
}

export interface ComponentDef {
  name: string;
  state?: Type;
  render: Expr; // S-expression for JSX-like syntax
  methods: MethodDef[];
}

export interface MethodDef {
  name: string;
  params: [string, Type][];
  body: Expr;
}

/**
 * Main Entry Point (@main)
 * Defines application startup
 */
export interface MainDef {
  kind: 'main';
  startFunction: string; // Function name for startup
  errorHandler?: string; // Function name for error handling
  shutdownHandler?: string; // Function name for cleanup
  loc?: SourceLocation;
}

/**
 * Complete Feature Slice
 * Represents entire feature slice with all directives
 */
export interface FeatureSliceDef {
  kind: 'feature-slice';
  name: string;
  version: string;
  agent?: AgentConfigDef;
  layers: LayerDef[];
  observability?: ObservabilityDef;
  network?: NetworkDef;
  storage?: StorageDef;
  multitenant?: MultitenantDef;
  ui?: UIDef;
  main?: MainDef;
  loc?: SourceLocation;
}

// ============================================================================
// Type Guards
// ============================================================================

export function isAgentConfig(node: any): node is AgentConfigDef {
  return node && node.kind === 'agent-config';
}

export function isLayer(node: any): node is LayerDef {
  return node && node.kind === 'layer';
}

export function isObservability(node: any): node is ObservabilityDef {
  return node && node.kind === 'observability';
}

export function isNetwork(node: any): node is NetworkDef {
  return node && node.kind === 'network';
}

export function isStorage(node: any): node is StorageDef {
  return node && node.kind === 'storage';
}

export function isMultitenant(node: any): node is MultitenantDef {
  return node && node.kind === 'multitenant';
}

export function isUI(node: any): node is UIDef {
  return node && node.kind === 'ui';
}

export function isMain(node: any): node is MainDef {
  return node && node.kind === 'main';
}

export function isFeatureSlice(node: any): node is FeatureSliceDef {
  return node && node.kind === 'feature-slice';
}

// ============================================================================
// Extended Definition Type (includes Feature Slice directives)
// ============================================================================

export type ExtendedDefinition =
  | Definition
  | AgentConfigDef
  | LayerDef
  | ObservabilityDef
  | NetworkDef
  | StorageDef
  | MultitenantDef
  | UIDef
  | MainDef
  | FeatureSliceDef;

export function isExtendedDef(node: any): node is ExtendedDefinition {
  return node && typeof node.kind === 'string' &&
    [
      'function', 'typedef', 'module',
      'agent-config', 'layer', 'observability', 'network',
      'storage', 'multitenant', 'ui', 'main', 'feature-slice'
    ].includes(node.kind);
}
