/**
 * Constitutional Adapter for .glass Organisms
 *
 * Wraps /src/agi-recursive/core/constitution.ts for use in .glass organisms.
 * Provides unified constitutional enforcement across all nodes.
 *
 * Architecture:
 * - Layer 1: UniversalConstitution (6 base principles)
 * - Layer 2: Domain-specific extensions (Cognitive, Security, etc.)
 *
 * Usage:
 * ```typescript
 * const adapter = createConstitutionalAdapter('cognitive');
 * const result = adapter.validate(response, context);
 * if (!result.passed) {
 *   console.error(adapter.formatReport(result));
 * }
 * ```
 */

import {
  UniversalConstitution,
  ConstitutionEnforcer,
  ConstitutionCheckResult,
  ConstitutionViolation
} from '../../agi-recursive/core/constitution';

// Import domain-specific extensions
import { CognitiveConstitution } from '../cognitive/constitutional/cognitive-constitution';

// ============================================================================
// Types
// ============================================================================

export type ConstitutionDomain = 'universal' | 'cognitive' | 'security' | 'glass-core' | 'vcs' | 'database';

export interface ConstitutionalAdapter {
  domain: ConstitutionDomain;
  enforcer: ConstitutionEnforcer;
  constitution: UniversalConstitution;

  /**
   * Validate response against constitutional principles
   */
  validate(
    response: any,
    context: {
      depth?: number;
      invocation_count?: number;
      cost_so_far?: number;
      previous_agents?: string[];
    }
  ): ConstitutionCheckResult;

  /**
   * Format violation report for user display
   */
  formatReport(result: ConstitutionCheckResult): string;

  /**
   * Get all principles (Layer 1 + Layer 2)
   */
  getPrinciples(): any[];

  /**
   * Get cost tracking (if applicable)
   */
  getCostTracking(): {
    total_cost: number;
    max_cost: number;
    remaining_budget: number;
  };
}

// ============================================================================
// Constitutional Adapter Factory
// ============================================================================

/**
 * Create constitutional adapter for specific domain
 */
export function createConstitutionalAdapter(
  domain: ConstitutionDomain = 'universal'
): ConstitutionalAdapter {
  const enforcer = new ConstitutionEnforcer();
  let constitution: UniversalConstitution;

  // Select constitution based on domain
  switch (domain) {
    case 'cognitive':
      // Use CognitiveConstitution (Layer 2)
      constitution = new CognitiveConstitution();
      break;

    case 'security':
      // TODO: Use SecurityConstitution when created
      constitution = new UniversalConstitution();
      break;

    case 'glass-core':
    case 'vcs':
    case 'database':
    case 'universal':
    default:
      constitution = new UniversalConstitution();
      break;
  }

  return {
    domain,
    enforcer,
    constitution,

    validate(response: any, context: any = {}): ConstitutionCheckResult {
      // Fill in default context values
      const fullContext = {
        depth: context.depth ?? 0,
        invocation_count: context.invocation_count ?? 1,
        cost_so_far: context.cost_so_far ?? 0,
        previous_agents: context.previous_agents ?? [],
        agent_id: domain
      };

      return constitution.checkResponse(response, fullContext);
    },

    formatReport(result: ConstitutionCheckResult): string {
      return enforcer.formatReport(result);
    },

    getPrinciples() {
      return constitution.principles;
    },

    getCostTracking() {
      // Extract recursion_budget principle
      const budgetPrinciple = constitution.principles.find(p => p.id === 'recursion_budget');
      const maxCost = budgetPrinciple?.enforcement?.max_cost_usd ?? 1.0;

      return {
        total_cost: 0, // Will be updated by usage
        max_cost: maxCost,
        remaining_budget: maxCost
      };
    }
  };
}

// ============================================================================
// Validation Helpers
// ============================================================================

/**
 * Quick validation - returns true if passed, false if violations
 */
export function quickValidate(
  adapter: ConstitutionalAdapter,
  response: any,
  context?: any
): boolean {
  const result = adapter.validate(response, context);
  return result.passed;
}

/**
 * Validate with auto-logging
 */
export function validateAndLog(
  adapter: ConstitutionalAdapter,
  response: any,
  context: any,
  logger: (message: string) => void
): ConstitutionCheckResult {
  const result = adapter.validate(response, context);

  if (!result.passed) {
    logger('❌ Constitutional violations detected:');
    logger(adapter.formatReport(result));
  } else if (result.warnings.length > 0) {
    logger('⚠️  Constitutional warnings:');
    logger(adapter.formatReport(result));
  }

  return result;
}

/**
 * Enforce constitutional compliance - throws error on violations
 */
export function enforceConstitutional(
  adapter: ConstitutionalAdapter,
  response: any,
  context?: any
): void {
  const result = adapter.validate(response, context);

  if (!result.passed) {
    const report = adapter.formatReport(result);
    throw new Error(`Constitutional violation:\n${report}`);
  }
}

// ============================================================================
// Audit Trail Helper
// ============================================================================

export interface ConstitutionalAuditEntry {
  timestamp: string;
  domain: ConstitutionDomain;
  action: string;
  passed: boolean;
  violations: ConstitutionViolation[];
  warnings: ConstitutionViolation[];
}

/**
 * Create audit entry from validation result
 */
export function createAuditEntry(
  adapter: ConstitutionalAdapter,
  action: string,
  result: ConstitutionCheckResult
): ConstitutionalAuditEntry {
  return {
    timestamp: new Date().toISOString(),
    domain: adapter.domain,
    action,
    passed: result.passed,
    violations: result.violations,
    warnings: result.warnings
  };
}

// ============================================================================
// Multi-Domain Validation
// ============================================================================

/**
 * Validate against multiple constitutional domains
 * Useful for .glass organisms that span multiple concerns
 */
export function validateMultipleDomains(
  domains: ConstitutionDomain[],
  response: any,
  context?: any
): {
  passed: boolean;
  results: Map<ConstitutionDomain, ConstitutionCheckResult>;
  combinedViolations: ConstitutionViolation[];
  combinedWarnings: ConstitutionViolation[];
} {
  const results = new Map<ConstitutionDomain, ConstitutionCheckResult>();
  let allPassed = true;
  const allViolations: ConstitutionViolation[] = [];
  const allWarnings: ConstitutionViolation[] = [];

  for (const domain of domains) {
    const adapter = createConstitutionalAdapter(domain);
    const result = adapter.validate(response, context);

    results.set(domain, result);

    if (!result.passed) {
      allPassed = false;
    }

    allViolations.push(...result.violations);
    allWarnings.push(...result.warnings);
  }

  return {
    passed: allPassed,
    results,
    combinedViolations: allViolations,
    combinedWarnings: allWarnings
  };
}

// ============================================================================
// Cost Budget Enforcement
// ============================================================================

/**
 * Track and enforce cost budget across organism lifecycle
 */
export class CostBudgetTracker {
  private totalCost: number = 0;
  private maxBudget: number;

  constructor(maxBudget: number = 1.0) {
    this.maxBudget = maxBudget;
  }

  addCost(cost: number): void {
    this.totalCost += cost;
  }

  getRemainingBudget(): number {
    return Math.max(0, this.maxBudget - this.totalCost);
  }

  isOverBudget(): boolean {
    return this.totalCost >= this.maxBudget;
  }

  getTotalCost(): number {
    return this.totalCost;
  }

  getMaxBudget(): number {
    return this.maxBudget;
  }

  /**
   * Check if operation would exceed budget
   */
  wouldExceedBudget(estimatedCost: number): boolean {
    return (this.totalCost + estimatedCost) > this.maxBudget;
  }
}
