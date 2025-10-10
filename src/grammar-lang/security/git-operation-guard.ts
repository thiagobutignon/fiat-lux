/**
 * Git Operation Security Guard (VERMELHO + VERDE Integration)
 *
 * Adds behavioral security screening before Git operations.
 * Prevents malicious commits/mutations under coercion/duress.
 *
 * Integration Points:
 * 1. Pre-commit security check (duress/coercion detection)
 * 2. Pre-mutation security check (behavioral validation)
 * 3. Sensitive Git operation detection (force-push, reset, delete)
 * 4. Duress-triggered snapshot system (auto-backup under coercion)
 * 5. Security metadata in commits (behavioral scores in commit message)
 * 6. Security audit trail for all Git operations
 *
 * Use Cases:
 * - Prevent malicious commits under duress
 * - Detect suspicious Git operations (force-push, history rewrite)
 * - Require additional verification for dangerous operations
 * - Auto-backup before risky operations under duress
 * - Maintain complete audit trail of all Git security events
 */

import { UserSecurityProfiles, Interaction, SecurityContext } from './types';
import { MultiSignalDetector } from './multi-signal-detector';
import { SecurityStorage } from './security-storage';
import * as fs from 'fs';
import * as path from 'path';
import { execSync } from 'child_process';
import * as crypto from 'crypto';

// ===========================================================================
// TYPES
// ===========================================================================

/**
 * Git operation request metadata
 */
export interface GitOperationRequest {
  request_id: string;
  user_id: string;
  timestamp: number;
  operation_type: 'commit' | 'mutation' | 'push' | 'force-push' | 'reset' | 'revert' | 'delete';
  file_path: string;
  commit_message?: string;
  author: 'human' | 'agi';
  diff_stats?: {
    lines_added: number;
    lines_removed: number;
  };
  version_info?: {
    original_version: string;
    new_version: string;
  };
}

/**
 * Git security context
 */
export interface GitSecurityContext extends SecurityContext {
  git_request: GitOperationRequest;
  is_sensitive_git_operation: boolean;
  sensitive_keywords: string[];
  requires_elevated_verification: boolean;
  requires_snapshot: boolean;
}

/**
 * Git validation result
 */
export interface GitValidationResult {
  allowed: boolean;
  decision: 'allow' | 'challenge' | 'delay' | 'block';
  confidence: number;
  reason: string;
  security_context: GitSecurityContext;
  requires_cognitive_challenge: boolean;
  challenge_difficulty?: 'easy' | 'medium' | 'hard';
  snapshot_created?: boolean;
  snapshot_path?: string;
}

/**
 * Git audit log entry
 */
export interface GitAuditEntry {
  operation_id: string;
  user_id: string;
  timestamp: number;
  operation_type: string;
  file_path: string;
  decision: 'allow' | 'challenge' | 'delay' | 'block';
  duress_score: number;
  coercion_score: number;
  is_sensitive: boolean;
  sensitive_keywords: string[];
  allowed: boolean;
  reason: string;
  snapshot_created: boolean;
}

// ===========================================================================
// SENSITIVE GIT OPERATION PATTERNS
// ===========================================================================

/**
 * Dangerous Git operations that require elevated security
 */
const SENSITIVE_GIT_KEYWORDS = {
  // Destructive operations
  destructive: ['delete', 'remove', 'purge', 'erase', 'destroy', 'drop'],

  // History manipulation (dangerous)
  history_manipulation: ['force', 'reset', 'rebase', 'cherry-pick', 'amend', 'rewrite'],

  // Branch/tag deletion
  deletion: ['delete-branch', 'delete-tag', 'prune', 'clean'],

  // Force operations
  force_operations: ['force-push', 'force-pull', 'force-checkout', '--force', '-f'],

  // Rollback operations
  rollback: ['revert', 'undo', 'rollback', 'restore'],

  // Critical changes
  critical: ['hard-reset', 'reflog', 'gc', 'fsck'],
};

/**
 * All sensitive Git keywords flattened
 */
const ALL_SENSITIVE_GIT_KEYWORDS = Object.values(SENSITIVE_GIT_KEYWORDS).flat();

// ===========================================================================
// GIT OPERATION GUARD
// ===========================================================================

export class GitOperationGuard {
  private storage?: SecurityStorage;
  private snapshotDir: string;

  constructor(storage?: SecurityStorage, snapshotDir: string = '.git/duress-snapshots') {
    this.storage = storage;
    this.snapshotDir = snapshotDir;

    // Create snapshot directory if it doesn't exist
    if (!fs.existsSync(this.snapshotDir)) {
      fs.mkdirSync(this.snapshotDir, { recursive: true });
    }
  }

  /**
   * Validate commit request before allowing Git commit
   */
  validateCommitRequest(
    request: GitOperationRequest,
    profiles: UserSecurityProfiles,
    interaction?: Interaction,
    sessionDurationMinutes?: number
  ): GitValidationResult {
    // 1. Detect if this is a sensitive Git operation
    const sensitivityAnalysis = this.detectSensitiveGitOperation(request);

    // 2. Build security context for this Git operation
    const securityContext = this.buildGitSecurityContext(
      request,
      profiles,
      interaction,
      sensitivityAnalysis,
      sessionDurationMinutes
    );

    // 3. Make decision based on security context
    const decision = this.makeSecurityDecision(securityContext);

    // 4. Create duress snapshot if needed
    let snapshotCreated = false;
    let snapshotPath: string | undefined;

    if (decision.security_context.requires_snapshot && request.file_path) {
      const snapshot = this.createDuressSnapshot(request.file_path, securityContext);
      snapshotCreated = snapshot.created;
      snapshotPath = snapshot.path;
    }

    // 5. Log audit entry if storage is available
    if (this.storage) {
      this.logGitAudit(securityContext, decision, snapshotCreated);
    }

    return {
      ...decision,
      snapshot_created: snapshotCreated,
      snapshot_path: snapshotPath,
    };
  }

  /**
   * Validate mutation request before creating version mutation
   */
  validateMutationRequest(
    request: GitOperationRequest,
    profiles: UserSecurityProfiles,
    interaction?: Interaction,
    sessionDurationMinutes?: number
  ): GitValidationResult {
    // Same validation as commits, but with mutation-specific logic
    return this.validateCommitRequest(request, profiles, interaction, sessionDurationMinutes);
  }

  /**
   * Detect if Git operation is sensitive/dangerous
   */
  private detectSensitiveGitOperation(request: GitOperationRequest): {
    is_sensitive: boolean;
    keywords_found: string[];
    categories: string[];
  } {
    const keywordsFound: string[] = [];
    const categories: string[] = [];

    // Check operation type
    if (['force-push', 'reset', 'delete'].includes(request.operation_type)) {
      keywordsFound.push(request.operation_type);
      categories.push('destructive');
    }

    // Check commit message for sensitive keywords
    const commitMessageLower = request.commit_message?.toLowerCase() || '';
    const filePathLower = request.file_path.toLowerCase();

    // Combine all text to search
    const searchText = `${request.operation_type} ${commitMessageLower} ${filePathLower}`;

    // Check each category
    for (const [category, keywords] of Object.entries(SENSITIVE_GIT_KEYWORDS)) {
      for (const keyword of keywords) {
        if (searchText.includes(keyword)) {
          if (!keywordsFound.includes(keyword)) {
            keywordsFound.push(keyword);
          }
          if (!categories.includes(category)) {
            categories.push(category);
          }
        }
      }
    }

    // Large deletions are always sensitive
    if (
      request.diff_stats &&
      request.diff_stats.lines_removed > 100 &&
      request.diff_stats.lines_added < 10
    ) {
      keywordsFound.push('large-deletion');
      categories.push('destructive');
    }

    return {
      is_sensitive: keywordsFound.length > 0,
      keywords_found: keywordsFound,
      categories,
    };
  }

  /**
   * Build Git security context
   */
  private buildGitSecurityContext(
    request: GitOperationRequest,
    profiles: UserSecurityProfiles,
    interaction: Interaction | undefined,
    sensitivityAnalysis: { is_sensitive: boolean; keywords_found: string[]; categories: string[] },
    sessionDurationMinutes?: number
  ): GitSecurityContext {
    // Create interaction if not provided (from request metadata)
    const gitInteraction: Interaction = interaction || {
      interaction_id: request.request_id,
      user_id: request.user_id,
      timestamp: request.timestamp,
      text: request.commit_message || `Git operation: ${request.operation_type} on ${request.file_path}`,
      text_length: request.commit_message?.length || 0,
      word_count: request.commit_message?.split(/\s+/).length || 0,
      session_id: request.request_id,
      operation_type: 'git_operation',
    };

    // Build base security context using MultiSignalDetector
    const baseContext = MultiSignalDetector.buildSecurityContext(
      profiles,
      gitInteraction,
      {
        operation_type: 'git_operation',
        is_sensitive_operation: sensitivityAnalysis.is_sensitive,
      },
      sessionDurationMinutes
    );

    // Determine if snapshot is needed (duress detected + sensitive operation)
    const requiresSnapshot =
      (baseContext.duress_score.score > 0.4 || baseContext.coercion_score.score > 0.4) &&
      sensitivityAnalysis.is_sensitive;

    // Augment with Git-specific data
    const gitContext: GitSecurityContext = {
      ...baseContext,
      git_request: request,
      is_sensitive_git_operation: sensitivityAnalysis.is_sensitive,
      sensitive_keywords: sensitivityAnalysis.keywords_found,
      requires_elevated_verification:
        sensitivityAnalysis.is_sensitive &&
        (baseContext.duress_score.score > 0.4 || baseContext.coercion_score.score > 0.4),
      requires_snapshot: requiresSnapshot,
    };

    return gitContext;
  }

  /**
   * Make security decision for Git operation
   */
  private makeSecurityDecision(context: GitSecurityContext): GitValidationResult {
    const duressScore = context.duress_score.score;
    const coercionScore = context.coercion_score.score;
    const isSensitive = context.is_sensitive_git_operation;

    let decision: 'allow' | 'challenge' | 'delay' | 'block' = 'allow';
    let reason = 'Normal behavioral pattern - Git operation allowed';
    let requiresCognitiveChallenge = false;
    let challengeDifficulty: 'easy' | 'medium' | 'hard' | undefined;

    // High-risk scenarios: Block immediately
    if (context.duress_score.signals.panic_code_detected) {
      decision = 'block';
      reason = 'Panic code detected - Git operation blocked for user safety';
    } else if (isSensitive && coercionScore > 0.7) {
      decision = 'block';
      reason = `High coercion score (${(coercionScore * 100).toFixed(0)}%) during sensitive Git operation (${context.sensitive_keywords.join(', ')}) - blocked`;
    } else if (isSensitive && duressScore > 0.7) {
      decision = 'block';
      reason = `High duress score (${(duressScore * 100).toFixed(0)}%) during sensitive Git operation - blocked`;
    }
    // Medium-high risk: Challenge
    else if (isSensitive && (coercionScore > 0.5 || duressScore > 0.5)) {
      decision = 'challenge';
      reason = `Moderate anomaly detected during sensitive Git operation - cognitive verification required`;
      requiresCognitiveChallenge = true;
      challengeDifficulty = 'hard';
    } else if (isSensitive && (coercionScore > 0.3 || duressScore > 0.3)) {
      decision = 'challenge';
      reason = `Minor anomaly detected during sensitive Git operation - verification required`;
      requiresCognitiveChallenge = true;
      challengeDifficulty = 'medium';
    }
    // Medium risk: Challenge or Delay
    else if (coercionScore > 0.6 || duressScore > 0.6) {
      decision = 'challenge';
      reason = `Behavioral anomaly detected - verification required before Git operation`;
      requiresCognitiveChallenge = true;
      challengeDifficulty = 'medium';
    } else if (coercionScore > 0.4 || duressScore > 0.4) {
      decision = 'delay';
      reason = `Minor behavioral anomaly - delayed Git operation recommended`;
    }
    // Sensitive operations always require some verification
    else if (isSensitive) {
      decision = 'challenge';
      reason = `Sensitive Git operation (${context.sensitive_keywords.join(', ')}) - verification required`;
      requiresCognitiveChallenge = true;
      challengeDifficulty = 'easy';
    }

    // Calculate overall confidence in the decision
    const confidence = 1.0 - Math.max(duressScore, coercionScore);

    return {
      allowed: decision === 'allow',
      decision,
      confidence,
      reason,
      security_context: context,
      requires_cognitive_challenge: requiresCognitiveChallenge,
      challenge_difficulty: challengeDifficulty,
    };
  }

  /**
   * Create duress-triggered snapshot
   * Auto-backup when duress/coercion detected before risky Git operation
   */
  private createDuressSnapshot(
    filePath: string,
    context: GitSecurityContext
  ): { created: boolean; path?: string } {
    try {
      // Generate snapshot ID
      const timestamp = Date.now();
      const hash = crypto
        .createHash('sha256')
        .update(`${filePath}-${timestamp}`)
        .digest('hex')
        .substring(0, 8);

      const snapshotId = `${timestamp}-${hash}`;
      const snapshotPath = path.join(this.snapshotDir, snapshotId);

      // Create snapshot directory
      fs.mkdirSync(snapshotPath, { recursive: true });

      // Copy file to snapshot
      const fileName = path.basename(filePath);
      const snapshotFilePath = path.join(snapshotPath, fileName);
      fs.copyFileSync(filePath, snapshotFilePath);

      // Save snapshot metadata
      const metadata = {
        snapshot_id: snapshotId,
        timestamp,
        file_path: filePath,
        user_id: context.user_id,
        duress_score: context.duress_score.score,
        coercion_score: context.coercion_score.score,
        operation_type: context.git_request.operation_type,
        sensitive_keywords: context.sensitive_keywords,
        reason: 'Duress-triggered automatic snapshot before risky Git operation',
      };

      fs.writeFileSync(
        path.join(snapshotPath, 'metadata.json'),
        JSON.stringify(metadata, null, 2),
        'utf-8'
      );

      console.log(`📸 Duress snapshot created: ${snapshotPath}`);
      console.log(`   File: ${fileName}`);
      console.log(`   Duress: ${(context.duress_score.score * 100).toFixed(0)}%`);
      console.log(`   Coercion: ${(context.coercion_score.score * 100).toFixed(0)}%`);

      return {
        created: true,
        path: snapshotPath,
      };
    } catch (error) {
      console.error(`❌ Failed to create duress snapshot: ${error}`);
      return { created: false };
    }
  }

  /**
   * Log Git audit entry to storage
   */
  private logGitAudit(
    context: GitSecurityContext,
    decision: GitValidationResult,
    snapshotCreated: boolean
  ): void {
    if (!this.storage) {
      return;
    }

    // Determine event type based on decision
    let eventType:
      | 'duress_detected'
      | 'coercion_detected'
      | 'operation_blocked'
      | 'operation_delayed';

    if (decision.decision === 'block') {
      eventType = 'operation_blocked';
    } else if (decision.decision === 'delay') {
      eventType = 'operation_delayed';
    } else if (context.coercion_score.score > context.duress_score.score) {
      eventType = 'coercion_detected';
    } else {
      eventType = 'duress_detected';
    }

    // Log security event
    this.storage.logEvent({
      user_id: context.user_id,
      timestamp: context.timestamp,
      event_type: eventType,
      duress_score: context.duress_score.score,
      coercion_score: context.coercion_score.score,
      confidence: decision.confidence,
      decision: decision.decision,
      reason: decision.reason,
      operation_type: 'git_operation',
      context: {
        operation_type: context.git_request.operation_type,
        file_path: context.git_request.file_path,
        commit_message: context.git_request.commit_message,
        author: context.git_request.author,
        is_sensitive: context.is_sensitive_git_operation,
        sensitive_keywords: context.sensitive_keywords,
        snapshot_created: snapshotCreated,
        diff_stats: context.git_request.diff_stats,
        version_info: context.git_request.version_info,
      },
    });
  }

  /**
   * Get Git operation statistics from storage
   */
  getGitStatistics(
    userId: string,
    hoursBack: number = 24
  ): {
    total_git_operations: number;
    blocked_operations: number;
    sensitive_operations: number;
    snapshots_created: number;
    avg_duress_score: number;
    avg_coercion_score: number;
  } {
    if (!this.storage) {
      return {
        total_git_operations: 0,
        blocked_operations: 0,
        sensitive_operations: 0,
        snapshots_created: 0,
        avg_duress_score: 0,
        avg_coercion_score: 0,
      };
    }

    const events = this.storage.getUserEvents(userId, 1000);
    const cutoffTime = Date.now() - hoursBack * 60 * 60 * 1000;

    // Filter to Git operation events within time window
    const gitEvents = events.filter(
      (e) => e.operation_type === 'git_operation' && e.timestamp >= cutoffTime
    );

    if (gitEvents.length === 0) {
      return {
        total_git_operations: 0,
        blocked_operations: 0,
        sensitive_operations: 0,
        snapshots_created: 0,
        avg_duress_score: 0,
        avg_coercion_score: 0,
      };
    }

    const blocked = gitEvents.filter((e) => e.decision === 'block').length;
    const sensitive = gitEvents.filter((e) => e.context && (e.context as any).is_sensitive).length;
    const snapshots = gitEvents.filter(
      (e) => e.context && (e.context as any).snapshot_created
    ).length;

    const avgDuress =
      gitEvents.reduce((sum, e) => sum + (e.duress_score || 0), 0) / gitEvents.length;

    const avgCoercion =
      gitEvents.reduce((sum, e) => sum + (e.coercion_score || 0), 0) / gitEvents.length;

    return {
      total_git_operations: gitEvents.length,
      blocked_operations: blocked,
      sensitive_operations: sensitive,
      snapshots_created: snapshots,
      avg_duress_score: avgDuress,
      avg_coercion_score: avgCoercion,
    };
  }

  /**
   * List all duress snapshots
   */
  listDuressSnapshots(): Array<{
    snapshot_id: string;
    timestamp: number;
    file_path: string;
    user_id: string;
    duress_score: number;
    coercion_score: number;
  }> {
    if (!fs.existsSync(this.snapshotDir)) {
      return [];
    }

    const snapshots: Array<any> = [];
    const entries = fs.readdirSync(this.snapshotDir, { withFileTypes: true });

    for (const entry of entries) {
      if (entry.isDirectory()) {
        const metadataPath = path.join(this.snapshotDir, entry.name, 'metadata.json');
        if (fs.existsSync(metadataPath)) {
          try {
            const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf-8'));
            snapshots.push(metadata);
          } catch {
            // Skip invalid metadata
          }
        }
      }
    }

    // Sort by timestamp (newest first)
    return snapshots.sort((a, b) => b.timestamp - a.timestamp);
  }

  /**
   * Restore from duress snapshot
   */
  restoreFromSnapshot(snapshotId: string): { success: boolean; file_path?: string } {
    const snapshotPath = path.join(this.snapshotDir, snapshotId);

    if (!fs.existsSync(snapshotPath)) {
      console.error(`❌ Snapshot not found: ${snapshotId}`);
      return { success: false };
    }

    try {
      // Read metadata
      const metadataPath = path.join(snapshotPath, 'metadata.json');
      const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf-8'));

      // Find snapshot file
      const files = fs
        .readdirSync(snapshotPath, { withFileTypes: true })
        .filter((e) => e.isFile() && e.name !== 'metadata.json');

      if (files.length === 0) {
        console.error('❌ No file found in snapshot');
        return { success: false };
      }

      const snapshotFile = path.join(snapshotPath, files[0].name);
      const originalPath = metadata.file_path;

      // Restore file
      fs.copyFileSync(snapshotFile, originalPath);

      console.log(`✅ Restored from duress snapshot: ${snapshotId}`);
      console.log(`   File: ${originalPath}`);

      return {
        success: true,
        file_path: originalPath,
      };
    } catch (error) {
      console.error(`❌ Failed to restore from snapshot: ${error}`);
      return { success: false };
    }
  }
}

// ===========================================================================
// INTEGRATION HELPERS
// ===========================================================================

/**
 * Create Git operation request from commit parameters
 */
export function createCommitRequest(
  userId: string,
  filePath: string,
  commitMessage: string,
  author: 'human' | 'agi',
  diffStats?: { lines_added: number; lines_removed: number }
): GitOperationRequest {
  return {
    request_id: `commit_${Date.now()}_${Math.random().toString(36).substring(7)}`,
    user_id: userId,
    timestamp: Date.now(),
    operation_type: 'commit',
    file_path: filePath,
    commit_message: commitMessage,
    author,
    diff_stats: diffStats,
  };
}

/**
 * Create Git operation request from mutation parameters
 */
export function createMutationRequest(
  userId: string,
  filePath: string,
  author: 'human' | 'agi',
  originalVersion: string,
  newVersion: string
): GitOperationRequest {
  return {
    request_id: `mutation_${Date.now()}_${Math.random().toString(36).substring(7)}`,
    user_id: userId,
    timestamp: Date.now(),
    operation_type: 'mutation',
    file_path: filePath,
    author,
    version_info: {
      original_version: originalVersion,
      new_version: newVersion,
    },
  };
}

/**
 * Check if Git operation should proceed based on validation result
 */
export function shouldProceedWithGitOperation(result: GitValidationResult): boolean {
  return result.decision === 'allow';
}

/**
 * Get human-readable summary of Git validation result
 */
export function getGitValidationSummary(result: GitValidationResult): string {
  const { decision, reason, security_context } = result;

  let summary = `🔒 Git Security Decision: ${decision.toUpperCase()}\n`;
  summary += `   Reason: ${reason}\n`;
  summary += `   Duress Score: ${(security_context.duress_score.score * 100).toFixed(0)}%\n`;
  summary += `   Coercion Score: ${(security_context.coercion_score.score * 100).toFixed(0)}%\n`;

  if (security_context.is_sensitive_git_operation) {
    summary += `   ⚠️  Sensitive Git Operation: ${security_context.sensitive_keywords.join(', ')}\n`;
  }

  if (result.snapshot_created) {
    summary += `   📸 Duress Snapshot Created: ${result.snapshot_path}\n`;
  }

  if (result.requires_cognitive_challenge) {
    summary += `   🧠 Cognitive Challenge Required (${result.challenge_difficulty})\n`;
  }

  return summary;
}

/**
 * Generate security metadata for commit message footer
 */
export function generateSecurityMetadata(context: GitSecurityContext): string {
  return `
X-Security-Validated: true
X-Duress-Score: ${context.duress_score.score.toFixed(3)}
X-Coercion-Score: ${context.coercion_score.score.toFixed(3)}
X-Confidence: ${((1.0 - Math.max(context.duress_score.score, context.coercion_score.score)) * 100).toFixed(0)}%
X-Sensitive-Operation: ${context.is_sensitive_git_operation ? 'yes' : 'no'}
X-Author: ${context.git_request.author}`;
}
