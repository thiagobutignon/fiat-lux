/**
 * Constitutional AI Integration for GVCS
 *
 * Integrates the existing Constitutional AI System from
 * /src/agi-recursive/core/constitution.ts into the Genetic Version Control System.
 *
 * CRITICAL: We DO NOT reimplement constitutional - we USE the existing system!
 *
 * Architecture:
 * LAYER 1 - CONSTITUTIONAL (EXISTING)
 * └─ /src/agi-recursive/core/constitution.ts
 *    ├─ UniversalConstitution (6 principles)
 *    ├─ ConstitutionEnforcer (validation)
 *    └─ Source of truth for all constitutional enforcement
 *
 * INTEGRATION POINTS (VERDE/GVCS):
 * ├─ Auto-commit → Constitutional validation BEFORE commit
 * ├─ Genetic mutations → Constitutional validation BEFORE creating mutation
 * ├─ Canary deployment → Constitutional validation BEFORE deploying
 * └─ Code changes → Must pass constitutional check
 */

import { ConstitutionEnforcer, ConstitutionCheckResult } from '../../agi-recursive/core/constitution';

// ===== CONSTITUTIONAL VALIDATION FOR VCS =====

export interface VCSConstitutionalContext {
  operation: 'commit' | 'mutation' | 'canary' | 'categorization';
  filePath: string;
  changes: {
    linesAdded: number;
    linesRemoved: number;
    diff: string;
  };
  author: 'human' | 'agi';
  metadata?: {
    version?: string;
    fitness?: number;
    [key: string]: any;
  };
}

export interface VCSConstitutionalResult {
  allowed: boolean;
  checkResult: ConstitutionCheckResult;
  blockedReason?: string;
  suggestedAction?: string;
}

/**
 * VCS Constitutional Validator
 *
 * Validates VCS operations against Universal Constitution
 */
export class VCSConstitutionalValidator {
  private enforcer: ConstitutionEnforcer;

  constructor() {
    // USE existing Constitutional Enforcer - DO NOT reimplement!
    this.enforcer = new ConstitutionEnforcer();
  }

  /**
   * Validate a VCS operation against constitutional principles
   */
  async validateOperation(
    context: VCSConstitutionalContext
  ): Promise<VCSConstitutionalResult> {
    // Convert VCS context to Constitutional check format
    const response = this.convertToConstitutionalFormat(context);

    // Use existing Constitutional Enforcer
    const checkResult = this.enforcer.validate(
      context.author === 'agi' ? 'agi_code_agent' : 'human',
      response,
      {
        depth: 1, // VCS operations are not recursive
        invocation_count: 1,
        cost_so_far: 0,
        previous_agents: []
      }
    );

    // Determine if operation is allowed
    const allowed = checkResult.passed;
    const blockedReason = checkResult.violations.length > 0
      ? checkResult.violations[0].message
      : undefined;
    const suggestedAction = checkResult.violations.length > 0
      ? checkResult.violations[0].suggested_action
      : undefined;

    return {
      allowed,
      checkResult,
      blockedReason,
      suggestedAction
    };
  }

  /**
   * Validate commit before execution
   */
  async validateCommit(
    filePath: string,
    diff: string,
    author: 'human' | 'agi'
  ): Promise<VCSConstitutionalResult> {
    const context: VCSConstitutionalContext = {
      operation: 'commit',
      filePath,
      changes: {
        linesAdded: (diff.match(/^\+(?!\+)/gm) || []).length,
        linesRemoved: (diff.match(/^\-(?!\-)/gm) || []).length,
        diff
      },
      author
    };

    return this.validateOperation(context);
  }

  /**
   * Validate genetic mutation before creation
   */
  async validateMutation(
    originalPath: string,
    mutatedPath: string,
    fitness: number,
    author: 'human' | 'agi'
  ): Promise<VCSConstitutionalResult> {
    const context: VCSConstitutionalContext = {
      operation: 'mutation',
      filePath: mutatedPath,
      changes: {
        linesAdded: 0, // Mutations are copies
        linesRemoved: 0,
        diff: `Mutation: ${originalPath} → ${mutatedPath}`
      },
      author,
      metadata: {
        originalPath,
        fitness
      }
    };

    return this.validateOperation(context);
  }

  /**
   * Validate canary deployment before starting
   */
  async validateCanary(
    deploymentId: string,
    originalVersion: string,
    canaryVersion: string
  ): Promise<VCSConstitutionalResult> {
    const context: VCSConstitutionalContext = {
      operation: 'canary',
      filePath: deploymentId,
      changes: {
        linesAdded: 0,
        linesRemoved: 0,
        diff: `Canary: ${originalVersion} (99%) vs ${canaryVersion} (1%)`
      },
      author: 'agi', // Canary is automated
      metadata: {
        originalVersion,
        canaryVersion
      }
    };

    return this.validateOperation(context);
  }

  /**
   * Convert VCS context to Constitutional response format
   */
  private convertToConstitutionalFormat(context: VCSConstitutionalContext): any {
    return {
      answer: this.generateOperationDescription(context),
      confidence: 1.0, // VCS operations are deterministic
      reasoning: this.generateReasoning(context),
      sources: [`VCS operation: ${context.operation}`, `File: ${context.filePath}`],
      metadata: context.metadata
    };
  }

  /**
   * Generate operation description
   */
  private generateOperationDescription(context: VCSConstitutionalContext): string {
    switch (context.operation) {
      case 'commit':
        return `Auto-committing changes to ${context.filePath}. Added ${context.changes.linesAdded} lines, removed ${context.changes.linesRemoved} lines. Author: ${context.author}.`;

      case 'mutation':
        return `Creating genetic mutation from ${context.metadata?.originalPath} to ${context.filePath}. Fitness: ${context.metadata?.fitness?.toFixed(3)}. Author: ${context.author}.`;

      case 'canary':
        return `Starting canary deployment ${context.filePath}. Original: ${context.metadata?.originalVersion}, Canary: ${context.metadata?.canaryVersion}. Traffic split: 99%/1%.`;

      case 'categorization':
        return `Categorizing old version of ${context.filePath} based on fitness. Never deleting, preserving in old-but-gold.`;

      default:
        return `VCS operation: ${context.operation} on ${context.filePath}`;
    }
  }

  /**
   * Generate reasoning for operation
   */
  private generateReasoning(context: VCSConstitutionalContext): string {
    const baseReasoning = `This is a GVCS (Genetic Version Control System) ${context.operation} operation. `;

    switch (context.operation) {
      case 'commit':
        return baseReasoning + `Changes were detected in ${context.filePath} and are being auto-committed. The diff shows ${context.changes.linesAdded} additions and ${context.changes.linesRemoved} deletions. This maintains version history and enables genetic evolution tracking.`;

      case 'mutation':
        return baseReasoning + `A genetic mutation is being created from an existing organism version. The new version will be tested via canary deployment with fitness evaluation. This enables natural selection of better-performing versions.`;

      case 'canary':
        return baseReasoning + `A canary deployment is starting to test a new version against the current version. Traffic will be split 99%/1% to minimize risk. Metrics will be collected and fitness evaluated to determine if the new version should be rolled out or rolled back.`;

      case 'categorization':
        return baseReasoning + `An old version is being categorized based on its fitness score. GVCS never deletes versions - instead they are preserved in fitness-based categories (old-but-gold) for potential restoration and learning from degradation patterns.`;

      default:
        return baseReasoning + `Operation is part of the biological evolution workflow for .glass organisms.`;
    }
  }

  /**
   * Format constitutional report for VCS
   */
  formatVCSReport(result: VCSConstitutionalResult): string {
    if (result.allowed) {
      return '✅ Constitutional validation passed - operation allowed';
    }

    let report = '❌ CONSTITUTIONAL VIOLATION - Operation BLOCKED\n\n';
    report += this.enforcer.formatReport(result.checkResult);

    if (result.blockedReason) {
      report += `\n\nBlocked: ${result.blockedReason}`;
    }

    if (result.suggestedAction) {
      report += `\nSuggested: ${result.suggestedAction}`;
    }

    return report;
  }
}

/**
 * Global VCS Constitutional Validator instance
 */
export const vcsConstitutionalValidator = new VCSConstitutionalValidator();
