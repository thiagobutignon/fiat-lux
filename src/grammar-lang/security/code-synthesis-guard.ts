/**
 * Code Synthesis Security Guard (VERMELHO + ROXO Integration)
 *
 * Adds behavioral security screening before code synthesis operations.
 * Ensures code is not synthesized under coercion/duress.
 *
 * Integration Points:
 * 1. Pre-synthesis security check (duress/coercion detection)
 * 2. Sensitive operation detection (dangerous code patterns)
 * 3. Behavioral validation for code synthesis requests
 * 4. Security audit trail for all synthesis operations
 *
 * Use Cases:
 * - Prevent malicious code synthesis under duress
 * - Detect suspicious patterns in synthesis requests
 * - Require additional verification for sensitive operations
 * - Maintain complete audit trail of code emergence
 */

import { UserSecurityProfiles, Interaction, SecurityContext } from './types';
import { MultiSignalDetector } from './multi-signal-detector';
import { SecurityStorage } from './security-storage';

// ===========================================================================
// TYPES
// ===========================================================================

/**
 * Code synthesis request metadata
 */
export interface SynthesisRequest {
  request_id: string;
  user_id: string;
  timestamp: number;
  function_name: string;
  function_signature: string;
  pattern_type: string;
  pattern_frequency: number;
  organism_id?: string;
  organism_domain?: string;
  request_text?: string; // User's natural language request (if available)
}

/**
 * Synthesis security context
 */
export interface SynthesisSecurityContext extends SecurityContext {
  synthesis_request: SynthesisRequest;
  is_sensitive_synthesis: boolean;
  sensitive_keywords: string[];
  requires_elevated_verification: boolean;
}

/**
 * Synthesis validation result
 */
export interface SynthesisValidationResult {
  allowed: boolean;
  decision: 'allow' | 'challenge' | 'delay' | 'block';
  confidence: number;
  reason: string;
  security_context: SynthesisSecurityContext;
  requires_cognitive_challenge: boolean;
  challenge_difficulty?: 'easy' | 'medium' | 'hard';
}

/**
 * Synthesis audit log entry
 */
export interface SynthesisAuditEntry {
  synthesis_id: string;
  user_id: string;
  timestamp: number;
  function_name: string;
  organism_id?: string;
  decision: 'allow' | 'challenge' | 'delay' | 'block';
  duress_score: number;
  coercion_score: number;
  is_sensitive: boolean;
  sensitive_keywords: string[];
  allowed: boolean;
  reason: string;
}

// ===========================================================================
// SENSITIVE OPERATION PATTERNS
// ===========================================================================

/**
 * Dangerous code patterns that require elevated security
 */
const SENSITIVE_KEYWORDS = {
  // Destructive operations
  destructive: ['delete', 'remove', 'drop', 'destroy', 'erase', 'purge', 'clear'],

  // Data manipulation
  data_manipulation: ['update', 'modify', 'alter', 'change', 'set', 'write'],

  // Administrative operations
  administrative: ['admin', 'sudo', 'root', 'privilege', 'permission', 'grant', 'revoke'],

  // Financial operations
  financial: ['transfer', 'payment', 'withdraw', 'deposit', 'transaction', 'send_money'],

  // Execution/System
  execution: ['execute', 'run', 'eval', 'exec', 'shell', 'command', 'system'],

  // Authentication/Authorization
  auth: ['login', 'authenticate', 'authorize', 'token', 'credential', 'password'],

  // Critical operations
  critical: ['terminate', 'shutdown', 'restart', 'kill', 'force', 'override'],
};

/**
 * All sensitive keywords flattened
 */
const ALL_SENSITIVE_KEYWORDS = Object.values(SENSITIVE_KEYWORDS).flat();

// ===========================================================================
// CODE SYNTHESIS GUARD
// ===========================================================================

export class CodeSynthesisGuard {
  private storage?: SecurityStorage;

  constructor(storage?: SecurityStorage) {
    this.storage = storage;
  }

  /**
   * Validate synthesis request before allowing code generation
   */
  validateSynthesisRequest(
    request: SynthesisRequest,
    profiles: UserSecurityProfiles,
    interaction?: Interaction,
    sessionDurationMinutes?: number
  ): SynthesisValidationResult {
    // 1. Detect if this is a sensitive synthesis operation
    const sensitivityAnalysis = this.detectSensitiveOperation(request);

    // 2. Build security context for this synthesis
    const securityContext = this.buildSynthesisSecurityContext(
      request,
      profiles,
      interaction,
      sensitivityAnalysis,
      sessionDurationMinutes
    );

    // 3. Make decision based on security context
    const decision = this.makeSecurityDecision(securityContext);

    // 4. Log audit entry if storage is available
    if (this.storage) {
      this.logSynthesisAudit(securityContext, decision);
    }

    return decision;
  }

  /**
   * Detect if synthesis operation is sensitive/dangerous
   */
  private detectSensitiveOperation(request: SynthesisRequest): {
    is_sensitive: boolean;
    keywords_found: string[];
    categories: string[];
  } {
    const keywordsFound: string[] = [];
    const categories: string[] = [];

    // Check function name for sensitive keywords
    const functionNameLower = request.function_name.toLowerCase();
    const signatureLower = request.function_signature.toLowerCase();
    const requestTextLower = request.request_text?.toLowerCase() || '';

    // Combine all text to search
    const searchText = `${functionNameLower} ${signatureLower} ${requestTextLower}`;

    // Check each category
    for (const [category, keywords] of Object.entries(SENSITIVE_KEYWORDS)) {
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

    return {
      is_sensitive: keywordsFound.length > 0,
      keywords_found: keywordsFound,
      categories,
    };
  }

  /**
   * Build synthesis security context
   */
  private buildSynthesisSecurityContext(
    request: SynthesisRequest,
    profiles: UserSecurityProfiles,
    interaction: Interaction | undefined,
    sensitivityAnalysis: { is_sensitive: boolean; keywords_found: string[]; categories: string[] },
    sessionDurationMinutes?: number
  ): SynthesisSecurityContext {
    // Create interaction if not provided (from request metadata)
    const synthesisInteraction: Interaction = interaction || {
      interaction_id: request.request_id,
      user_id: request.user_id,
      timestamp: request.timestamp,
      text: request.request_text || `Synthesize function: ${request.function_name}`,
      text_length: request.request_text?.length || 0,
      word_count: request.request_text?.split(/\s+/).length || 0,
      session_id: request.request_id,
      operation_type: 'code_synthesis',
    };

    // Build base security context using MultiSignalDetector
    const baseContext = MultiSignalDetector.buildSecurityContext(
      profiles,
      synthesisInteraction,
      {
        operation_type: 'code_synthesis',
        is_sensitive_operation: sensitivityAnalysis.is_sensitive,
      },
      sessionDurationMinutes
    );

    // Augment with synthesis-specific data
    const synthesisContext: SynthesisSecurityContext = {
      ...baseContext,
      synthesis_request: request,
      is_sensitive_synthesis: sensitivityAnalysis.is_sensitive,
      sensitive_keywords: sensitivityAnalysis.keywords_found,
      requires_elevated_verification:
        sensitivityAnalysis.is_sensitive &&
        (baseContext.duress_score.score > 0.4 || baseContext.coercion_score.score > 0.4),
    };

    return synthesisContext;
  }

  /**
   * Make security decision for synthesis request
   */
  private makeSecurityDecision(
    context: SynthesisSecurityContext
  ): SynthesisValidationResult {
    const duressScore = context.duress_score.score;
    const coercionScore = context.coercion_score.score;
    const isSensitive = context.is_sensitive_synthesis;

    let decision: 'allow' | 'challenge' | 'delay' | 'block' = 'allow';
    let reason = 'Normal behavioral pattern - synthesis allowed';
    let requiresCognitiveChallenge = false;
    let challengeDifficulty: 'easy' | 'medium' | 'hard' | undefined;

    // High-risk scenarios: Block immediately
    if (context.duress_score.signals.panic_code_detected) {
      decision = 'block';
      reason = 'Panic code detected - synthesis blocked for user safety';
    } else if (isSensitive && coercionScore > 0.7) {
      decision = 'block';
      reason = `High coercion score (${(coercionScore * 100).toFixed(0)}%) during sensitive synthesis (${context.sensitive_keywords.join(', ')}) - blocked`;
    } else if (isSensitive && duressScore > 0.7) {
      decision = 'block';
      reason = `High duress score (${(duressScore * 100).toFixed(0)}%) during sensitive synthesis - blocked`;
    }
    // Medium-high risk: Delay or Challenge
    else if (isSensitive && (coercionScore > 0.5 || duressScore > 0.5)) {
      decision = 'challenge';
      reason = `Moderate anomaly detected during sensitive synthesis - cognitive verification required`;
      requiresCognitiveChallenge = true;
      challengeDifficulty = 'hard';
    } else if (isSensitive && (coercionScore > 0.3 || duressScore > 0.3)) {
      decision = 'challenge';
      reason = `Minor anomaly detected during sensitive synthesis - verification required`;
      requiresCognitiveChallenge = true;
      challengeDifficulty = 'medium';
    }
    // Medium risk: Challenge
    else if (coercionScore > 0.6 || duressScore > 0.6) {
      decision = 'challenge';
      reason = `Behavioral anomaly detected - verification required before synthesis`;
      requiresCognitiveChallenge = true;
      challengeDifficulty = 'medium';
    } else if (coercionScore > 0.4 || duressScore > 0.4) {
      decision = 'delay';
      reason = `Minor behavioral anomaly - delayed synthesis recommended`;
    }
    // Sensitive operations always require some verification
    else if (isSensitive) {
      decision = 'challenge';
      reason = `Sensitive operation (${context.sensitive_keywords.join(', ')}) - verification required`;
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
   * Log synthesis audit entry to storage
   */
  private logSynthesisAudit(
    context: SynthesisSecurityContext,
    decision: SynthesisValidationResult
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
      operation_type: 'code_synthesis',
      context: {
        function_name: context.synthesis_request.function_name,
        function_signature: context.synthesis_request.function_signature,
        pattern_type: context.synthesis_request.pattern_type,
        is_sensitive: context.is_sensitive_synthesis,
        sensitive_keywords: context.sensitive_keywords,
        organism_id: context.synthesis_request.organism_id,
        organism_domain: context.synthesis_request.organism_domain,
      },
    });
  }

  /**
   * Get synthesis statistics from storage
   */
  getSynthesisStatistics(userId: string, hoursBack: number = 24): {
    total_syntheses: number;
    blocked_syntheses: number;
    sensitive_syntheses: number;
    avg_duress_score: number;
    avg_coercion_score: number;
  } {
    if (!this.storage) {
      return {
        total_syntheses: 0,
        blocked_syntheses: 0,
        sensitive_syntheses: 0,
        avg_duress_score: 0,
        avg_coercion_score: 0,
      };
    }

    const events = this.storage.getUserEvents(userId, 1000);
    const cutoffTime = Date.now() - hoursBack * 60 * 60 * 1000;

    // Filter to synthesis events within time window
    const synthesisEvents = events.filter(
      (e) => e.operation_type === 'code_synthesis' && e.timestamp >= cutoffTime
    );

    if (synthesisEvents.length === 0) {
      return {
        total_syntheses: 0,
        blocked_syntheses: 0,
        sensitive_syntheses: 0,
        avg_duress_score: 0,
        avg_coercion_score: 0,
      };
    }

    const blocked = synthesisEvents.filter((e) => e.decision === 'block').length;
    const sensitive = synthesisEvents.filter(
      (e) => e.context && (e.context as any).is_sensitive
    ).length;

    const avgDuress =
      synthesisEvents.reduce((sum, e) => sum + (e.duress_score || 0), 0) /
      synthesisEvents.length;

    const avgCoercion =
      synthesisEvents.reduce((sum, e) => sum + (e.coercion_score || 0), 0) /
      synthesisEvents.length;

    return {
      total_syntheses: synthesisEvents.length,
      blocked_syntheses: blocked,
      sensitive_syntheses: sensitive,
      avg_duress_score: avgDuress,
      avg_coercion_score: avgCoercion,
    };
  }
}

// ===========================================================================
// INTEGRATION HELPERS
// ===========================================================================

/**
 * Create synthesis request from emergence candidate
 */
export function createSynthesisRequest(
  userId: string,
  candidate: any,
  organismId?: string,
  organismDomain?: string,
  requestText?: string
): SynthesisRequest {
  return {
    request_id: `synthesis_${Date.now()}_${Math.random().toString(36).substring(7)}`,
    user_id: userId,
    timestamp: Date.now(),
    function_name: candidate.suggested_function_name,
    function_signature: candidate.suggested_signature,
    pattern_type: candidate.pattern.type,
    pattern_frequency: candidate.pattern.frequency,
    organism_id: organismId,
    organism_domain: organismDomain,
    request_text: requestText,
  };
}

/**
 * Check if synthesis should proceed based on validation result
 */
export function shouldProceedWithSynthesis(result: SynthesisValidationResult): boolean {
  return result.decision === 'allow';
}

/**
 * Get human-readable summary of validation result
 */
export function getSynthesisValidationSummary(result: SynthesisValidationResult): string {
  const { decision, reason, security_context } = result;

  let summary = `🔒 Security Decision: ${decision.toUpperCase()}\n`;
  summary += `   Reason: ${reason}\n`;
  summary += `   Duress Score: ${(security_context.duress_score.score * 100).toFixed(0)}%\n`;
  summary += `   Coercion Score: ${(security_context.coercion_score.score * 100).toFixed(0)}%\n`;

  if (security_context.is_sensitive_synthesis) {
    summary += `   ⚠️  Sensitive Operation: ${security_context.sensitive_keywords.join(', ')}\n`;
  }

  if (result.requires_cognitive_challenge) {
    summary += `   🧠 Cognitive Challenge Required (${result.challenge_difficulty})\n`;
  }

  return summary;
}
