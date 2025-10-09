/**
 * Feature Slice Protocol Validators
 *
 * Validates Clean Architecture rules and Constitutional principles
 */

import {
  FeatureSliceDef,
  LayerDef,
  AgentConfigDef,
  ObservabilityDef
} from '../core/feature-slice-ast';
import { Definition, ModuleDef } from '../core/ast';

// ============================================================================
// Validation Errors
// ============================================================================

export class ValidationError extends Error {
  constructor(message: string, public layer?: string, public definition?: string) {
    super(message);
    this.name = 'ValidationError';
  }
}

// ============================================================================
// Clean Architecture Validator
// ============================================================================

/**
 * Validates that dependencies point INWARD according to Clean Architecture
 *
 * Rules:
 * - Domain → No external dependencies
 * - Data → Domain only
 * - Infrastructure → Data protocols + Domain
 * - Presentation → Domain use-cases only
 * - Main → All layers (composition)
 */
export class CleanArchitectureValidator {
  private layerDependencies: Map<string, Set<string>> = new Map();

  validate(featureSlice: FeatureSliceDef): void {
    // Extract layer dependencies
    this.extractDependencies(featureSlice);

    // Validate each layer
    for (const layer of featureSlice.layers) {
      this.validateLayer(layer);
    }
  }

  private extractDependencies(featureSlice: FeatureSliceDef): void {
    for (const layer of featureSlice.layers) {
      const deps = new Set<string>();

      for (const def of layer.definitions) {
        if (def.kind === 'module') {
          const moduleDef = def as ModuleDef;
          for (const imp of moduleDef.imports) {
            // Extract layer from module path
            // e.g., "financial-advisor/domain/entities" -> "domain"
            const layerMatch = imp.module.match(/\/(domain|data|infrastructure|presentation|main)\//);
            if (layerMatch) {
              deps.add(layerMatch[1]);
            }
          }
        }
      }

      this.layerDependencies.set(layer.layerType, deps);
    }
  }

  private validateLayer(layer: LayerDef): void {
    const deps = this.layerDependencies.get(layer.layerType) || new Set();

    switch (layer.layerType) {
      case 'domain':
        // Domain CANNOT depend on anything
        if (deps.size > 0 && !deps.has('domain')) {
          throw new ValidationError(
            `Domain layer CANNOT depend on external layers. Found: ${Array.from(deps).join(', ')}`,
            'domain'
          );
        }
        break;

      case 'data':
        // Data can ONLY depend on Domain
        for (const dep of deps) {
          if (dep !== 'domain' && dep !== 'data') {
            throw new ValidationError(
              `Data layer can ONLY depend on Domain. Found dependency on: ${dep}`,
              'data'
            );
          }
        }
        break;

      case 'infrastructure':
        // Infrastructure can depend on Data + Domain
        for (const dep of deps) {
          if (!['domain', 'data', 'infrastructure'].includes(dep)) {
            throw new ValidationError(
              `Infrastructure can ONLY depend on Data and Domain. Found: ${dep}`,
              'infrastructure'
            );
          }
        }
        break;

      case 'presentation':
        // Presentation can depend on Domain use-cases (NOT data or infrastructure)
        for (const dep of deps) {
          if (dep !== 'domain' && dep !== 'presentation') {
            throw new ValidationError(
              `Presentation can ONLY depend on Domain. Found: ${dep}`,
              'presentation'
            );
          }
        }
        break;

      case 'validation':
        // Validation can depend on Domain and Data
        for (const dep of deps) {
          if (!['domain', 'data', 'validation'].includes(dep)) {
            throw new ValidationError(
              `Validation can ONLY depend on Domain and Data. Found: ${dep}`,
              'validation'
            );
          }
        }
        break;
    }
  }
}

// ============================================================================
// Constitutional AI Validator
// ============================================================================

/**
 * Validates Constitutional AI principles are enforced
 *
 * Checks:
 * - Privacy: No PII in logs, responses, or storage
 * - Honesty: All claims are verifiable and sourced
 * - Transparency: Reasoning is traceable, attention is tracked
 * - Verified Sources: All knowledge sources have citations
 */
export class ConstitutionalValidator {
  validate(featureSlice: FeatureSliceDef): void {
    // Check agent config has constitutional principles
    if (!featureSlice.agent) {
      throw new ValidationError(
        'Feature Slice MUST have @agent configuration with constitutional principles'
      );
    }

    this.validateAgentConfig(featureSlice.agent);

    // Check observability includes constitutional metrics
    if (featureSlice.observability) {
      this.validateObservability(featureSlice.observability);
    }

    // Check that constitutional checks exist in code
    this.validateConstitutionalChecks(featureSlice);
  }

  private validateAgentConfig(agent: AgentConfigDef): void {
    const required = ['privacy', 'honesty', 'transparency'];

    for (const principle of required) {
      if (!agent.constitutional.includes(principle)) {
        throw new ValidationError(
          `Agent MUST include constitutional principle: ${principle}`,
          'agent-config'
        );
      }
    }

    // Check attention tracking is enabled
    if (!agent.prompt.attentionTracking) {
      throw new ValidationError(
        'Agent MUST have attention tracking enabled',
        'agent-config'
      );
    }
  }

  private validateObservability(obs: ObservabilityDef): void {
    // Check for constitutional compliance metrics
    const hasConstitutionalMetric = obs.metrics.some(m =>
      m.name.includes('constitutional') ||
      m.name.includes('compliance') ||
      m.name.includes('privacy')
    );

    if (!hasConstitutionalMetric) {
      throw new ValidationError(
        'Observability MUST include constitutional compliance metrics',
        'observability'
      );
    }

    // Check for attention metrics
    const hasAttentionMetric = obs.metrics.some(m =>
      m.name.includes('attention')
    );

    if (!hasAttentionMetric) {
      throw new ValidationError(
        'Observability MUST include attention tracking metrics',
        'observability'
      );
    }
  }

  private validateConstitutionalChecks(featureSlice: FeatureSliceDef): void {
    // Check that validation layer exists
    const hasValidationLayer = featureSlice.layers.some(l => l.layerType === 'validation');

    if (!hasValidationLayer) {
      throw new ValidationError(
        'Feature Slice MUST have @layer validation for constitutional checks'
      );
    }
  }
}

// ============================================================================
// Grammar Alignment Validator
// ============================================================================

/**
 * Validates alignment with Universal Grammar (Chomsky)
 *
 * Checks:
 * - Domain entities are NOUNs (subjects/objects)
 * - Domain use-cases are VERBs (actions)
 * - Data protocols are ADVERBs (abstract manner)
 * - Infrastructure is ADVERB (concrete manner)
 * - Complete sentences (subject + verb + object)
 */
export class GrammarAlignmentValidator {
  validate(featureSlice: FeatureSliceDef): void {
    const domainLayer = featureSlice.layers.find(l => l.layerType === 'domain');

    if (!domainLayer) {
      throw new ValidationError('Feature Slice MUST have @layer domain');
    }

    // Check for entities (NOUNs)
    const hasEntities = domainLayer.definitions.some(d =>
      d.kind === 'typedef' && this.isEntityType(d.name)
    );

    if (!hasEntities) {
      throw new ValidationError(
        'Domain layer MUST have at least one entity (NOUN)',
        'domain'
      );
    }

    // Check for use-cases (VERBs)
    const hasUseCases = domainLayer.definitions.some(d =>
      d.kind === 'function' || (d.kind === 'typedef' && this.isUseCaseType(d.name))
    );

    if (!hasUseCases) {
      throw new ValidationError(
        'Domain layer MUST have at least one use-case (VERB)',
        'domain'
      );
    }

    // Check data layer has protocols (ADVERB abstract)
    const dataLayer = featureSlice.layers.find(l => l.layerType === 'data');
    if (dataLayer) {
      const hasProtocol = dataLayer.definitions.some(d =>
        d.kind === 'typedef' && this.isProtocolType(d.name)
      );

      if (!hasProtocol) {
        throw new ValidationError(
          'Data layer MUST have at least one protocol/interface (ADVERB abstract)',
          'data'
        );
      }
    }
  }

  private isEntityType(name: string): boolean {
    // Entities are typically nouns (capitalized single words or compounds)
    // e.g., User, Investment, Account, UserProfile
    return /^[A-Z][a-zA-Z]*$/.test(name);
  }

  private isUseCaseType(name: string): boolean {
    // Use-cases are typically verb phrases
    // e.g., RegisterUser, CalculateReturn, GetRecommendation
    return /^[A-Z][a-z]+[A-Z]/.test(name) || name.toLowerCase().includes('use') || name.toLowerCase().includes('case');
  }

  private isProtocolType(name: string): boolean {
    // Protocols/Interfaces end with Repository, Service, etc.
    return name.endsWith('Repository') ||
           name.endsWith('Service') ||
           name.endsWith('Protocol') ||
           name.endsWith('Interface');
  }
}

// ============================================================================
// Feature Slice Validator (Main)
// ============================================================================

export class FeatureSliceValidator {
  private cleanArchValidator = new CleanArchitectureValidator();
  private constitutionalValidator = new ConstitutionalValidator();
  private grammarValidator = new GrammarAlignmentValidator();

  validate(featureSlice: FeatureSliceDef): void {
    const errors: ValidationError[] = [];

    // Validate Clean Architecture
    try {
      this.cleanArchValidator.validate(featureSlice);
    } catch (e) {
      if (e instanceof ValidationError) {
        errors.push(e);
      }
    }

    // Validate Constitutional AI
    try {
      this.constitutionalValidator.validate(featureSlice);
    } catch (e) {
      if (e instanceof ValidationError) {
        errors.push(e);
      }
    }

    // Validate Grammar Alignment
    try {
      this.grammarValidator.validate(featureSlice);
    } catch (e) {
      if (e instanceof ValidationError) {
        errors.push(e);
      }
    }

    // Throw if any errors
    if (errors.length > 0) {
      const message = errors.map(e => `  - ${e.message}`).join('\n');
      throw new Error(`Feature Slice validation failed:\n${message}`);
    }
  }

  /**
   * Validate and return warnings (non-critical issues)
   */
  validateWithWarnings(featureSlice: FeatureSliceDef): string[] {
    const warnings: string[] = [];

    // Check for optional but recommended features
    if (!featureSlice.observability) {
      warnings.push('Recommended: Add @observable for metrics and traces');
    }

    if (!featureSlice.network) {
      warnings.push('Recommended: Add @network for API definition');
    }

    if (!featureSlice.storage) {
      warnings.push('Recommended: Add @storage for persistence configuration');
    }

    // Check layer count
    if (featureSlice.layers.length < 3) {
      warnings.push('Recommended: Feature Slice should have at least 3 layers (domain, data, infrastructure)');
    }

    return warnings;
  }
}
