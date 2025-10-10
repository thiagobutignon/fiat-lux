// Types for .glass organisms
export interface GlassOrganism {
  id: string;
  metadata: {
    name: string;
    version: string;
    specialization: string;
    created_at: string;
    updated_at: string;
    maturity: number; // 0.0 - 1.0
    stage: 'nascent' | 'infancy' | 'adolescence' | 'maturity' | 'evolution';
    generation: number;
  };
  model: {
    architecture: string;
    parameters: number;
    quantization: string;
  };
  knowledge: {
    papers: number;
    embeddings_dim: number;
    patterns: Pattern[];
    connections: number;
    clusters: number;
  };
  code: {
    functions: EmergedFunction[];
    total_lines: number;
  };
  memory: {
    short_term: any[];
    long_term: any[];
    contextual: any[];
  };
  constitutional: {
    agent_type: string;
    principles: string[];
    boundaries: Record<string, boolean>;
    validation: string;
  };
  evolution: {
    enabled: boolean;
    generation: number;
    fitness: number;
    trajectory: FitnessPoint[];
  };
  stats: {
    total_cost: number;
    queries_count: number;
    avg_query_time_ms: number;
    last_query_at?: string;
  };
}

export interface Pattern {
  keyword: string;
  frequency: number;
  confidence: number;
  emergence_score: number;
  emerged_function?: string;
}

export interface EmergedFunction {
  name: string;
  signature: string;
  code: string;
  emerged_from: string;
  occurrences: number;
  constitutional_status: 'pass' | 'fail';
  lines: number;
  created_at: string;
}

export interface FitnessPoint {
  generation: number;
  fitness: number;
  timestamp: string;
}

export interface QueryResult {
  answer: string;
  confidence: number;
  functions_used: string[];
  constitutional: 'pass' | 'fail';
  cost: number;
  time_ms: number;
  sources: Source[];
  attention: AttentionWeight[];
  reasoning: ReasoningStep[];
}

export interface Source {
  id: string;
  title: string;
  type: 'paper' | 'document' | 'trial';
  relevance: number;
}

export interface AttentionWeight {
  source_id: string;
  weight: number; // 0.0 - 1.0
}

export interface ReasoningStep {
  step: number;
  description: string;
  confidence: number;
  time_ms: number;
}

// System stats
export interface SystemStats {
  total_organisms: number;
  total_queries: number;
  total_cost: number;
  budget_limit: number;
  health: 'healthy' | 'warning' | 'critical';
  uptime: number;
}

// Debug types
export interface ConstitutionalLog {
  id: string;
  timestamp: string;
  organism_id: string;
  principle: string;
  status: 'pass' | 'fail' | 'warning';
  details: string;
  context?: any;
  query_id?: string;
}

export interface LLMCall {
  id: string;
  timestamp: string;
  organism_id: string;
  task_type: string; // 'intent-analysis', 'code-synthesis', 'query-execution'
  model: string; // 'claude-opus-4', 'claude-sonnet-4.5'
  tokens_in: number;
  tokens_out: number;
  cost: number;
  latency_ms: number;
  prompt: string;
  response: string;
  constitutional_status: 'pass' | 'fail';
  query_id?: string;
}

export interface PerformanceMetrics {
  query_processing_ms: number;
  pattern_detection_ms: number;
  knowledge_access_ms: number;
  llm_latency_ms: number;
  total_ms: number;
}

export interface EvolutionData {
  organism_id: string;
  current_generation: number;
  current_fitness: number;
  maturity: number;
  versions: VersionInfo[];
  canary_status?: {
    current_version: string;
    canary_version: string;
    current_traffic: number;
    canary_traffic: number;
    status: 'monitoring' | 'promoting' | 'rolling_back';
  };
}

export interface VersionInfo {
  version: string;
  generation: number;
  fitness: number;
  traffic_percent: number;
  deployed_at: string;
  status: 'active' | 'canary' | 'old-but-gold' | 'deprecated';
}
