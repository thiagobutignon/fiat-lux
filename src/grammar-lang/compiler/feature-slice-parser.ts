/**
 * Feature Slice Protocol Parser
 *
 * Parses Feature Slice directives (@agent, @layer, etc.)
 */

import { parseType } from './parser';
import { ParseError } from './parser';
import {
  AgentConfigDef,
  LayerDef,
  ObservabilityDef,
  NetworkDef,
  StorageDef,
  MultitenantDef,
  UIDef,
  MainDef,
  FeatureSliceDef,
  MetricDef,
  TraceDef,
  RouteDef,
  ExposedFunctionDef,
  ComponentDef,
  MethodDef,
  ExtendedDefinition
} from '../core/feature-slice-ast';
import { parseDefinition, parseExpr } from './parser';

type SExpr = string | number | boolean | null | SExpr[];

// ============================================================================
// Agent Config Parser
// ============================================================================

export function parseAgentConfig(sexpr: SExpr): AgentConfigDef {
  if (!Array.isArray(sexpr)) {
    throw new ParseError('Agent config must be a record', sexpr);
  }

  const [head, ...args] = sexpr;

  if (head !== 'agent-config') {
    throw new ParseError('Expected agent-config', sexpr);
  }

  // Parse as record with fields
  const config: any = {
    kind: 'agent-config',
    name: '',
    domain: '',
    expertise: [],
    constitutional: [],
    knowledge: '',
    constraints: [],
    prompt: {
      role: '',
      tone: '',
      knowledgeSources: '',
      constitutionalPrinciples: '',
      attentionTracking: false
    }
  };

  for (const arg of args) {
    if (!Array.isArray(arg) || arg.length !== 2) {
      throw new ParseError('Agent config field must be (key value)', arg);
    }
    const [key, value] = arg;

    if (key === 'name' && typeof value === 'string') {
      config.name = value;
    } else if (key === 'domain' && typeof value === 'string') {
      config.domain = value;
    } else if (key === 'expertise' && Array.isArray(value)) {
      config.expertise = value.filter(v => typeof v === 'string');
    } else if (key === 'constitutional' && Array.isArray(value)) {
      config.constitutional = value.filter(v => typeof v === 'string');
    } else if (key === 'knowledge' && typeof value === 'string') {
      config.knowledge = value;
    } else if (key === 'constraints' && Array.isArray(value)) {
      config.constraints = value.filter(v => typeof v === 'string');
    } else if (key === 'prompt' && Array.isArray(value)) {
      // Parse prompt record
      for (const field of value) {
        if (Array.isArray(field) && field.length === 2) {
          const [pk, pv] = field;
          if (pk === 'role') config.prompt.role = pv;
          else if (pk === 'tone') config.prompt.tone = pv;
          else if (pk === 'knowledge-sources') config.prompt.knowledgeSources = pv;
          else if (pk === 'constitutional-principles') config.prompt.constitutionalPrinciples = pv;
          else if (pk === 'attention-tracking') config.prompt.attentionTracking = pv === true;
        }
      }
    }
  }

  return config as AgentConfigDef;
}

// ============================================================================
// Layer Parser
// ============================================================================

export function parseLayer(sexpr: SExpr): LayerDef {
  if (!Array.isArray(sexpr)) {
    throw new ParseError('Layer must be a list', sexpr);
  }

  const [head, layerTypeExpr, ...defExprs] = sexpr;

  if (head !== 'layer') {
    throw new ParseError('Expected layer', sexpr);
  }

  if (typeof layerTypeExpr !== 'string') {
    throw new ParseError('Layer type must be string', layerTypeExpr);
  }

  const validLayers = ['domain', 'data', 'infrastructure', 'validation', 'presentation'];
  if (!validLayers.includes(layerTypeExpr)) {
    throw new ParseError(`Invalid layer type: ${layerTypeExpr}`, layerTypeExpr);
  }

  const definitions = defExprs.map(parseDefinition);

  return {
    kind: 'layer',
    layerType: layerTypeExpr as any,
    definitions
  };
}

// ============================================================================
// Observability Parser
// ============================================================================

export function parseObservability(sexpr: SExpr): ObservabilityDef {
  if (!Array.isArray(sexpr)) {
    throw new ParseError('Observability must be a record', sexpr);
  }

  const [head, ...args] = sexpr;

  if (head !== 'observable') {
    throw new ParseError('Expected observable', sexpr);
  }

  const metrics: MetricDef[] = [];
  const traces: TraceDef[] = [];

  for (const arg of args) {
    if (!Array.isArray(arg) || arg.length < 2) continue;

    const [key, value] = arg;

    if (key === 'metrics' && Array.isArray(value)) {
      // Parse metrics record
      for (const metricExpr of value) {
        if (Array.isArray(metricExpr)) {
          const [name, metricDef] = metricExpr;
          if (typeof name === 'string' && Array.isArray(metricDef)) {
            const [metricType, ...metricArgs] = metricDef;

            const metric: MetricDef = {
              name,
              type: metricType as any,
              description: ''
            };

            for (const marg of metricArgs) {
              if (Array.isArray(marg) && marg.length === 2) {
                const [mk, mv] = marg;
                if (mk === 'labels' && Array.isArray(mv)) {
                  metric.labels = mv.filter(v => typeof v === 'string');
                } else if (mk === 'buckets' && Array.isArray(mv)) {
                  metric.buckets = mv.filter(v => typeof v === 'number');
                } else if (mk === 'description' && typeof mv === 'string') {
                  metric.description = mv;
                }
              }
            }

            metrics.push(metric);
          }
        }
      }
    } else if (key === 'traces' && Array.isArray(value)) {
      // Parse traces record
      for (const traceExpr of value) {
        if (Array.isArray(traceExpr) && traceExpr.length >= 2) {
          const [name, traceDef] = traceExpr;
          if (typeof name === 'string' && Array.isArray(traceDef)) {
            const trace: TraceDef = {
              name,
              enabled: false,
              export: ''
            };

            for (const targ of traceDef) {
              if (Array.isArray(targ) && targ.length === 2) {
                const [tk, tv] = targ;
                if (tk === 'enabled') trace.enabled = tv === true;
                else if (tk === 'export' && typeof tv === 'string') trace.export = tv;
                else if (tk === 'description' && typeof tv === 'string') trace.description = tv;
              }
            }

            traces.push(trace);
          }
        }
      }
    }
  }

  return {
    kind: 'observability',
    metrics,
    traces
  };
}

// ============================================================================
// Network Parser
// ============================================================================

export function parseNetwork(sexpr: SExpr): NetworkDef {
  if (!Array.isArray(sexpr)) {
    throw new ParseError('Network must be a record', sexpr);
  }

  const [head, ...args] = sexpr;

  if (head !== 'network') {
    throw new ParseError('Expected network', sexpr);
  }

  const apiConfig = {
    protocol: 'http',
    port: 8080,
    cors: false
  };
  const routes: RouteDef[] = [];
  const exposedFunctions: ExposedFunctionDef[] = [];

  for (const arg of args) {
    if (!Array.isArray(arg) || arg.length < 2) continue;

    const [key, value] = arg;

    if (key === 'api-config' && Array.isArray(value)) {
      for (const field of value) {
        if (Array.isArray(field) && field.length === 2) {
          const [k, v] = field;
          if (k === 'protocol' && typeof v === 'string') apiConfig.protocol = v;
          else if (k === 'port' && typeof v === 'number') apiConfig.port = v;
          else if (k === 'cors') apiConfig.cors = v === true;
        }
      }
    } else if (key === 'routes' && Array.isArray(value)) {
      for (const routeExpr of value) {
        if (Array.isArray(routeExpr)) {
          const route: any = {
            method: 'GET',
            path: '',
            response: null,
            handler: ''
          };

          for (const field of routeExpr) {
            if (Array.isArray(field) && field.length === 2) {
              const [k, v] = field;
              if (k === 'method' && typeof v === 'string') route.method = v;
              else if (k === 'path' && typeof v === 'string') route.path = v;
              else if (k === 'request') route.request = parseType(v);
              else if (k === 'response') route.response = parseType(v);
              else if (k === 'query') route.query = parseType(v);
              else if (k === 'auth' && typeof v === 'string') route.auth = v;
              else if (k === 'rate-limit' && typeof v === 'string') route.rateLimit = v;
              else if (k === 'constitutional') route.constitutional = v === true;
              else if (k === 'handler' && typeof v === 'string') route.handler = v;
            }
          }

          routes.push(route);
        }
      }
    } else if (key === 'exposed-functions' && Array.isArray(value)) {
      for (const expExpr of value) {
        if (Array.isArray(expExpr) && expExpr.length === 2) {
          const [fnName, exposedAs] = expExpr;
          if (typeof fnName === 'string' && typeof exposedAs === 'string') {
            exposedFunctions.push({ functionName: fnName, exposedAs });
          }
        }
      }
    }
  }

  return {
    kind: 'network',
    apiConfig,
    routes,
    exposedFunctions
  };
}

// ============================================================================
// Storage Parser
// ============================================================================

export function parseStorage(sexpr: SExpr): StorageDef {
  if (!Array.isArray(sexpr)) {
    throw new ParseError('Storage must be a record', sexpr);
  }

  const [head, ...args] = sexpr;

  if (head !== 'storage') {
    throw new ParseError('Expected storage', sexpr);
  }

  const storage: StorageDef = {
    kind: 'storage'
  };

  for (const arg of args) {
    if (!Array.isArray(arg) || arg.length < 2) continue;

    const [key, value] = arg;

    if (key === 'relational' && Array.isArray(value)) {
      storage.relational = {} as any;
      for (const field of value) {
        if (Array.isArray(field) && field.length === 2) {
          const [k, v] = field;
          if (k === 'type') storage.relational!.type = v as string;
          else if (k === 'url') storage.relational!.url = v as string;
          else if (k === 'migrations') storage.relational!.migrations = v as string;
          else if (k === 'connection-pool') storage.relational!.connectionPool = v as number;
        }
      }
    } else if (key === 'cache' && Array.isArray(value)) {
      storage.cache = {} as any;
      for (const field of value) {
        if (Array.isArray(field) && field.length === 2) {
          const [k, v] = field;
          if (k === 'type') storage.cache!.type = v as string;
          else if (k === 'url') storage.cache!.url = v as string;
          else if (k === 'ttl-default') storage.cache!.ttlDefault = v as number;
        }
      }
    } else if (key === 'files' && Array.isArray(value)) {
      storage.files = {} as any;
      for (const field of value) {
        if (Array.isArray(field) && field.length === 2) {
          const [k, v] = field;
          if (k === 'text') storage.files!.text = v as string;
          else if (k === 'images') storage.files!.images = v as string;
          else if (k === 'video') storage.files!.video = v as string;
          else if (k === 'documents') storage.files!.documents = v as string;
        }
      }
    } else if (key === 'embeddings' && Array.isArray(value)) {
      storage.embeddings = {} as any;
      for (const field of value) {
        if (Array.isArray(field) && field.length === 2) {
          const [k, v] = field;
          if (k === 'type') storage.embeddings!.type = v as string;
          else if (k === 'dimensions') storage.embeddings!.dimensions = v as number;
          else if (k === 'distance') storage.embeddings!.distance = v as string;
        }
      }
    }
  }

  return storage;
}

// ============================================================================
// Main/UI/Multitenant Parsers (simplified)
// ============================================================================

export function parseMain(sexpr: SExpr): MainDef {
  if (!Array.isArray(sexpr)) {
    throw new ParseError('Main must be a record', sexpr);
  }

  const [head, startFn] = sexpr;

  if (head !== 'main') {
    throw new ParseError('Expected main', sexpr);
  }

  return {
    kind: 'main',
    startFunction: typeof startFn === 'string' ? startFn : 'start'
  };
}

// ============================================================================
// Feature Slice Parser (Main)
// ============================================================================

export function parseFeatureSlice(sexprs: SExpr[]): FeatureSliceDef {
  const featureSlice: FeatureSliceDef = {
    kind: 'feature-slice',
    name: 'unnamed',
    version: '1.0.0',
    layers: []
  };

  for (const sexpr of sexprs) {
    if (!Array.isArray(sexpr)) continue;

    const [head] = sexpr;

    if (head === 'agent-config') {
      featureSlice.agent = parseAgentConfig(sexpr);
    } else if (head === 'layer') {
      featureSlice.layers.push(parseLayer(sexpr));
    } else if (head === 'observable') {
      featureSlice.observability = parseObservability(sexpr);
    } else if (head === 'network') {
      featureSlice.network = parseNetwork(sexpr);
    } else if (head === 'storage') {
      featureSlice.storage = parseStorage(sexpr);
    } else if (head === 'main') {
      featureSlice.main = parseMain(sexpr);
    }
    // Can add more directives here
  }

  // Extract name from agent config if available
  if (featureSlice.agent) {
    featureSlice.name = featureSlice.agent.name.toLowerCase().replace(/\s+/g, '-');
  }

  return featureSlice;
}

// ============================================================================
// Extended Definition Parser (includes Feature Slice directives)
// ============================================================================

export function parseExtendedDefinition(sexpr: SExpr): ExtendedDefinition {
  if (!Array.isArray(sexpr)) {
    throw new ParseError('Expected definition or directive', sexpr);
  }

  const [head] = sexpr;

  // Check if it's a Feature Slice directive
  if (head === 'agent-config') return parseAgentConfig(sexpr);
  if (head === 'layer') return parseLayer(sexpr);
  if (head === 'observable') return parseObservability(sexpr);
  if (head === 'network') return parseNetwork(sexpr);
  if (head === 'storage') return parseStorage(sexpr);
  if (head === 'main') return parseMain(sexpr);

  // Otherwise, parse as regular definition
  return parseDefinition(sexpr);
}
