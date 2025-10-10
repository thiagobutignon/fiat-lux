/**
 * Glass Builder - Creates .glass Digital Organisms
 *
 * This builder creates LIVING ORGANISMS, not just files.
 * - Starts at 0% maturity (nascent)
 * - Grows organically through knowledge ingestion
 * - Code EMERGES from patterns (not programmed)
 * - 100% glass box (inspectable at every stage)
 */

import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';
import {
  GlassOrganism,
  GlassMetadata,
  GlassModel,
  GlassKnowledge,
  GlassCode,
  GlassMemory,
  GlassConstitutional,
  GlassEvolution,
  GlassBuilderOptions,
  GlassLifecycleStage,
  GlassFile
} from './types';

/**
 * Glass Builder
 * Exported for use in CLI and other modules
 */
export class GlassBuilder {
  private organism: GlassOrganism;

  constructor(options: GlassBuilderOptions) {
    this.organism = this.createNascentOrganism(options);
  }

  /**
   * Create a nascent (0% maturity) organism
   */
  private createNascentOrganism(options: GlassBuilderOptions): GlassOrganism {
    const now = new Date().toISOString();

    // METADATA - Cell Identity
    const metadata: GlassMetadata = {
      format: 'fiat-glass-v1.0',
      type: 'digital-organism',
      name: options.name,
      version: '1.0.0',
      created: now,
      specialization: options.specialization || 'general',
      maturity: 0.0, // NASCENT - 0%
      stage: GlassLifecycleStage.NASCENT,
      generation: 1,
      parent: null
    };

    // DNA - Base Model (27M params)
    const model: GlassModel = {
      architecture: 'transformer-27M',
      parameters: 27_000_000,
      weights: null, // Will be loaded from base model
      quantization: 'int8',
      constitutional_embedding: true
    };

    // RNA - Knowledge (empty at birth)
    const knowledge: GlassKnowledge = {
      papers: {
        count: 0,
        sources: [],
        embeddings: null,
        indexed: false
      },
      patterns: {},
      connections: {
        nodes: 0,
        edges: 0,
        clusters: 0
      }
    };

    // PROTEINS - Code (empty, will emerge)
    const code: GlassCode = {
      functions: [],
      emergence_log: {}
    };

    // MEMORY - Episodic (empty)
    const memory: GlassMemory = {
      short_term: [],
      long_term: [],
      contextual: []
    };

    // MEMBRANE - Constitutional Boundaries
    // Determine agent_type based on specialization
    let agent_type = 'universal';
    if (options.specialization && (options.specialization.includes('bio') || options.specialization.includes('onco') || options.specialization.includes('medical'))) {
      agent_type = 'biology';
    } else if (options.specialization && (options.specialization.includes('fin') || options.specialization.includes('econ'))) {
      agent_type = 'financial';
    }

    const constitutional: GlassConstitutional = {
      agent_type, // 'universal', 'biology', or 'financial'
      principles: options.constitutional || [
        'transparency',
        'honesty',
        'privacy',
        'safety'
      ],
      boundaries: {
        cannot_harm: true,
        must_cite_sources: true,
        cannot_diagnose: true, // medical domain
        confidence_threshold_required: true
      },
      validation: 'native'
    };

    // METABOLISM - Evolution
    const evolution: GlassEvolution = {
      enabled: true,
      last_evolution: null,
      generations: 0,
      fitness_trajectory: [0.0] // starts at 0
    };

    return {
      metadata,
      model,
      knowledge,
      code,
      memory,
      constitutional,
      evolution
    };
  }

  /**
   * Get the organism
   */
  public getOrganism(): GlassOrganism {
    return this.organism;
  }

  /**
   * Save organism to .glass file
   */
  public async save(outputPath: string): Promise<void> {
    const glassFile: GlassFile = {
      header: {
        magic: 'GLASS',
        version: 1,
        metadata_size: 0, // will calculate
        model_size: 0,
        knowledge_size: 0
      },
      metadata: this.organism.metadata,
      model_weights: this.organism.model.weights,
      knowledge_embeddings: this.organism.knowledge.papers.embeddings,
      organism: this.organism
    };

    // For now, save as JSON (later will be binary)
    const json = JSON.stringify(this.organism, null, 2);
    glassFile.header.metadata_size = Buffer.from(json).length;

    // Write to file
    fs.writeFileSync(outputPath, json, 'utf-8');

    console.log(`✅ Created ${this.organism.metadata.name}.glass`);
    console.log(`   Size: ${this.getSize(json)} (nascent)`);
    console.log(`   Maturity: ${(this.organism.metadata.maturity * 100).toFixed(0)}%`);
    console.log(`   Status: ${this.organism.metadata.stage}`);
  }

  /**
   * Load organism from .glass file
   */
  public static async load(inputPath: string): Promise<GlassOrganism> {
    const json = fs.readFileSync(inputPath, 'utf-8');
    const organism = JSON.parse(json) as GlassOrganism;
    return organism;
  }

  /**
   * Get size in human-readable format
   */
  private getSize(json: string): string {
    const bytes = Buffer.from(json).length;
    if (bytes < 1024) return `${bytes}B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
  }

  /**
   * Get organism info (for status command)
   */
  public getInfo(): {
    name: string;
    maturity: number;
    stage: GlassLifecycleStage;
    functions_emerged: number;
    patterns_detected: number;
    knowledge_count: number;
  } {
    return {
      name: this.organism.metadata.name,
      maturity: this.organism.metadata.maturity,
      stage: this.organism.metadata.stage,
      functions_emerged: this.organism.code.functions.length,
      patterns_detected: Object.keys(this.organism.knowledge.patterns).length,
      knowledge_count: this.organism.knowledge.papers.count
    };
  }

  /**
   * Calculate hash of organism (for content-addressing)
   */
  public getHash(): string {
    const json = JSON.stringify(this.organism);
    return crypto.createHash('sha256').update(json).digest('hex').substring(0, 16);
  }
}

/**
 * Factory function - create nascent organism
 */
export function createGlassOrganism(options: GlassBuilderOptions): GlassBuilder {
  return new GlassBuilder(options);
}

/**
 * Factory function - load existing organism
 */
export async function loadGlassOrganism(path: string): Promise<GlassOrganism> {
  return GlassBuilder.load(path);
}
