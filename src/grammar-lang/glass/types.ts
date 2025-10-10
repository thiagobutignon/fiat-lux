/**
 * .glass - Digital Organism Types
 *
 * .glass is not a file - it's a LIVING DIGITAL ORGANISM
 * - Self-contained
 * - Code emerges from knowledge
 * - 100% glass box (inspectable)
 * - Constitutional AI embedded
 */

import { ConstitutionEnforcer } from '../../agi-recursive/core/constitution';

/**
 * Lifecycle stages of a .glass organism
 */
export enum GlassLifecycleStage {
  NASCENT = 'nascent',       // 0% - just created
  INFANCY = 'infancy',       // 0-25% - absorbing knowledge
  ADOLESCENCE = 'adolescence', // 25-75% - patterns emerging
  MATURITY = 'maturity',     // 75-100% - fully specialized
  EVOLUTION = 'evolution',   // continuous improvement
  RETIREMENT = 'retirement'  // graceful shutdown
}

/**
 * Maturity level (0.0 to 1.0)
 */
export type GlassMaturity = number; // 0.0 = nascent, 1.0 = mature

/**
 * Generation number (for cloning/reproduction)
 */
export type GlassGeneration = number;

/**
 * Metadata - Cell Identity
 */
export interface GlassMetadata {
  format: 'fiat-glass-v1.0';
  type: 'digital-organism';
  name: string;
  version: string;
  created: string; // ISO timestamp
  specialization: string; // domain (e.g., "oncology", "finance")
  maturity: GlassMaturity; // 0.0 to 1.0
  stage: GlassLifecycleStage;
  generation: GlassGeneration;
  parent: string | null; // parent .glass hash (if cloned)
}

/**
 * Base Model - DNA
 * 27M parameters transformer
 */
export interface GlassModel {
  architecture: string; // e.g., "transformer-27M"
  parameters: number; // e.g., 27_000_000
  weights: Uint8Array | null; // binary weights (null if not loaded)
  quantization: 'int8' | 'float16' | 'float32';
  constitutional_embedding: boolean; // principles embedded in weights
}

/**
 * Knowledge - RNA (mutable)
 */
export interface GlassKnowledge {
  papers: {
    count: number;
    sources: string[]; // e.g., ["pubmed:10000", "arxiv:2000"]
    embeddings: Float32Array | null; // vector database
    indexed: boolean;
  };
  patterns: {
    // Auto-identified patterns
    [patternType: string]: number; // count
  };
  connections: {
    // Knowledge graph
    nodes: number;
    edges: number;
    clusters: number;
  };
}

/**
 * Emerged Function - Protein
 * Code that EMERGED from patterns (NOT programmed!)
 */
export interface GlassFunction {
  name: string;
  signature: string; // e.g., "(CancerType, Drug, Stage) -> Efficacy"
  source_patterns: string[]; // patterns that triggered emergence
  confidence: number; // 0.0 to 1.0
  accuracy: number; // 0.0 to 1.0
  constitutional: boolean; // passes constitutional checks
  implementation: string; // .gl code (readable!)
  emerged_at: string; // ISO timestamp
  trigger: string; // what triggered emergence
  validated: boolean;
}

/**
 * Code - Emerged Functions (Proteins)
 */
export interface GlassCode {
  functions: GlassFunction[];
  emergence_log: {
    [functionName: string]: {
      emerged_at: string;
      trigger: string;
      pattern_count: number;
      validated: boolean;
    };
  };
}

/**
 * Memory - Episodic (learning)
 */
export interface GlassMemory {
  short_term: any[]; // recent interactions
  long_term: any[]; // consolidated memories
  contextual: any[]; // context-specific
}

/**
 * Constitutional Boundaries - Membrane
 * Uses ConstitutionEnforcer from /src/agi-recursive/core/constitution.ts
 */
export interface GlassConstitutional {
  agent_type: string; // 'universal' | 'biology' | 'financial' - determines which constitution to use
  principles: string[]; // principle IDs being enforced
  boundaries: {
    [rule: string]: boolean; // domain-specific boundaries
  };
  validation: 'native'; // native layer for validation
}

/**
 * Evolution - Metabolism
 */
export interface GlassEvolution {
  enabled: boolean;
  last_evolution: string | null; // ISO timestamp
  generations: number;
  fitness_trajectory: number[]; // [0.72, 0.81, 0.87, ...]
}

/**
 * Complete .glass Structure
 * This is a DIGITAL ORGANISM, not just a file
 */
export interface GlassOrganism {
  // METADATA (Cell Identity)
  metadata: GlassMetadata;

  // DNA (Base Model - 27M params)
  model: GlassModel;

  // RNA (Knowledge - Mutable)
  knowledge: GlassKnowledge;

  // PROTEINS (Emerged Functions)
  code: GlassCode;

  // MEMORY (Episodic)
  memory: GlassMemory;

  // MEMBRANE (Constitutional Boundaries)
  constitutional: GlassConstitutional;

  // METABOLISM (Self-Evolution)
  evolution: GlassEvolution;
}

/**
 * Builder Options
 */
export interface GlassBuilderOptions {
  name: string;
  specialization?: string;
  baseModel?: string; // path to base model weights
  constitutional?: string[]; // principles to embed
}

/**
 * Glass File Format
 * Hybrid: JSON metadata + Binary weights
 */
export interface GlassFile {
  header: {
    magic: 'GLASS'; // magic bytes
    version: 1;
    metadata_size: number; // bytes
    model_size: number; // bytes
    knowledge_size: number; // bytes
  };
  metadata: GlassMetadata;
  model_weights: Uint8Array | null;
  knowledge_embeddings: Float32Array | null;
  organism: GlassOrganism; // full structure
}
