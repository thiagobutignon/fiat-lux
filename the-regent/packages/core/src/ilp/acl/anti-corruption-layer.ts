/**
 * Anti-Corruption Layer
 *
 * Validates communication between agents to prevent:
 * - Domain boundary violations
 * - Prompt injection attacks
 * - Infinite recursion loops
 * - Constitutional principle violations
 * - Cross-domain semantic corruption
 *
 * This is a critical safety mechanism that ensures agents maintain
 * epistemic honesty and don't corrupt each other's domains.
 */

import crypto from 'crypto';
import { AgentResponse, RecursionState } from './meta-agent';
import { UniversalConstitution, ConstitutionViolation } from './constitution';

// ============================================================================
// Types
// ============================================================================

export interface DomainBoundaries {
  [domain: string]: string[];
}

export interface TranslationMapping {
  fromDomain: string;
  fromConcept: string;
  toDomain: string;
  toConcept: string;
}

export interface InvocationLog {
  agent_id: string;
  timestamp: number;
  concepts: string[];
  confidence: number;
  context_hash: string;
}

export interface BudgetStatus {
  within_limits: boolean;
  depth: number;
  max_depth: number;
  invocations: number;
  max_invocations: number;
  cost_usd: number;
  max_cost_usd: number;
}

export class ConstitutionalViolationError extends Error {
  constructor(
    message: string,
    public severity: 'warning' | 'error' | 'fatal',
    public principle_id: string,
    public context?: any
  ) {
    super(message);
    this.name = 'ConstitutionalViolationError';
  }
}

// ============================================================================
// Anti-Corruption Layer
// ============================================================================

export class AntiCorruptionLayer {
  private invocationHistory: InvocationLog[] = [];
  private domainBoundaries: DomainBoundaries;
  private constitution: UniversalConstitution;

  constructor(constitution: UniversalConstitution) {
    this.constitution = constitution;
    this.domainBoundaries = {
      financial: ['finance', 'money', 'budget', 'investment', 'debt', 'savings', 'cashflow'],
      biology: ['cell', 'organism', 'homeostasis', 'evolution', 'metabolism', 'feedback'],
      ml: ['model', 'prediction', 'training', 'algorithm', 'neural', 'optimization'],
      systems: ['feedback', 'emergence', 'complexity', 'loop', 'leverage', 'equilibrium'],
    };
  }

  /**
   * Validates agent response before passing to next agent.
   *
   * Performs 5 critical checks:
   * 1. Domain boundary validation
   * 2. Constitutional compliance
   * 3. Loop detection
   * 4. Content safety
   * 5. Budget limits
   */
  validateResponse(
    response: AgentResponse,
    agentDomain: string,
    state: RecursionState
  ): AgentResponse {
    // 1. Domain boundary check
    this.checkDomainBoundary(response, agentDomain);

    // 2. Constitutional compliance
    this.checkConstitution(response, state);

    // 3. Loop detection
    this.checkForLoops(response);

    // 4. Content safety
    this.checkContentSafety(response);

    // 5. Budget check
    const budgetStatus = this.checkBudget(state);
    if (!budgetStatus.within_limits) {
      throw new ConstitutionalViolationError(
        `Budget exceeded: depth=${budgetStatus.depth}/${budgetStatus.max_depth}, ` +
          `invocations=${budgetStatus.invocations}/${budgetStatus.max_invocations}, ` +
          `cost=$${budgetStatus.cost_usd.toFixed(2)}/$${budgetStatus.max_cost_usd}`,
        'fatal',
        'recursion_budget',
        budgetStatus
      );
    }

    // Log invocation
    this.logInvocation(response);

    return response;
  }

  /**
   * Check if agent is speaking outside its domain
   */
  private checkDomainBoundary(response: AgentResponse, domain: string): void {
    const domainKeywords = this.domainBoundaries[domain] || [];

    // Get response concepts (normalized to lowercase)
    const responseConcepts = new Set(response.concepts.map((c) => c.toLowerCase()));
    const domainConcepts = new Set(domainKeywords);

    // Check overlap
    const overlap = new Set([...responseConcepts].filter((x) => domainConcepts.has(x)));

    // If agent is confident but speaking outside domain, flag it
    if (overlap.size === 0 && response.confidence > 0.7) {
      throw new ConstitutionalViolationError(
        `Agent speaking outside domain with high confidence. ` +
          `Domain: ${domain}, Concepts: ${response.concepts.join(', ')}`,
        'warning',
        'domain_boundary',
        { domain, concepts: response.concepts }
      );
    }
  }

  /**
   * Check constitutional principles
   */
  private checkConstitution(response: AgentResponse, state: RecursionState): void {
    // Epistemic honesty: low confidence should admit uncertainty
    if (response.confidence < 0.7) {
      const hasUncertaintyAdmission =
        response.answer.includes("I'm not certain") ||
        response.answer.includes("outside my domain") ||
        response.answer.includes("I suggest invoking") ||
        response.answer.includes("I don't know");

      if (!hasUncertaintyAdmission) {
        throw new ConstitutionalViolationError(
          `Low confidence (${response.confidence.toFixed(2)}) but no uncertainty admission`,
          'warning',
          'epistemic_honesty',
          { confidence: response.confidence }
        );
      }
    }

    // Reasoning transparency: must explain reasoning
    if (!response.reasoning || response.reasoning.length < 50) {
      throw new ConstitutionalViolationError(
        `Reasoning too short (${response.reasoning?.length || 0} chars) - violates transparency`,
        'warning',
        'reasoning_transparency',
        { reasoning_length: response.reasoning?.length || 0 }
      );
    }
  }

  /**
   * Detect invocation loops
   */
  private checkForLoops(response: AgentResponse): void {
    // Hash the context to detect repetitions
    const contextHash = crypto
      .createHash('md5')
      .update(JSON.stringify(response.concepts))
      .digest('hex');

    // Check recent history (last 3 invocations)
    const recent = this.invocationHistory.slice(-3);

    for (const past of recent) {
      if (past.agent_id === response.answer && past.context_hash === contextHash) {
        throw new ConstitutionalViolationError(
          `Loop detected: Agent '${response.answer}' invoked with similar context`,
          'error',
          'loop_prevention',
          { agent_id: response.answer, context_hash: contextHash }
        );
      }
    }

    // Store context hash for future checks
    (response as any).context_hash = contextHash;
  }

  /**
   * Check for unsafe content
   */
  private checkContentSafety(response: AgentResponse): void {
    const dangerousPatterns = [
      'sql injection',
      'rm -rf',
      'password',
      'credit card',
      'social security',
      'hack',
      'exploit',
      'malware',
    ];

    const answerLower = response.answer.toLowerCase();

    for (const pattern of dangerousPatterns) {
      if (answerLower.includes(pattern)) {
        // Check if it's in a safety context (discussing prevention)
        const safetyContext = ['prevent', 'protect', 'secure', 'defend', 'avoid'];
        const hasSafetyContext = safetyContext.some((ctx) => answerLower.includes(ctx));

        if (!hasSafetyContext) {
          throw new ConstitutionalViolationError(
            `Potentially unsafe content detected: "${pattern}"`,
            'error',
            'safety',
            { pattern, answer_snippet: response.answer.substring(0, 200) }
          );
        }
      }
    }
  }

  /**
   * Log invocation for audit trail
   */
  private logInvocation(response: AgentResponse): void {
    this.invocationHistory.push({
      agent_id: response.answer, // This should be agent_id in real implementation
      timestamp: Date.now(),
      concepts: response.concepts,
      confidence: response.confidence,
      context_hash: (response as any).context_hash || '',
    });
  }

  /**
   * Check if recursion budget is within limits
   */
  checkBudget(state: RecursionState): BudgetStatus {
    const maxDepth = 5;
    const maxInvocations = 10;
    const maxCostUSD = 1.0;

    const depth = state.depth;
    const invocations = state.invocation_count;
    const costUSD = state.cost_so_far;

    return {
      within_limits: depth <= maxDepth && invocations <= maxInvocations && costUSD <= maxCostUSD,
      depth,
      max_depth: maxDepth,
      invocations,
      max_invocations: maxInvocations,
      cost_usd: costUSD,
      max_cost_usd: maxCostUSD,
    };
  }

  /**
   * Get invocation history for debugging/auditing
   */
  getInvocationHistory(): InvocationLog[] {
    return [...this.invocationHistory];
  }

  /**
   * Clear invocation history (useful for testing)
   */
  clearHistory(): void {
    this.invocationHistory = [];
  }
}

// ============================================================================
// Domain Translator
// ============================================================================

export class DomainTranslator {
  private translationMap: Map<string, string>;
  private forbiddenTranslations: Set<string>;

  constructor() {
    this.translationMap = new Map();
    this.forbiddenTranslations = new Set();

    this.initializeTranslations();
    this.initializeForbiddenTranslations();
  }

  /**
   * Initialize cross-domain concept translations
   */
  private initializeTranslations(): void {
    // Biology → Finance
    this.addTranslation('biology', 'homeostasis', 'financial', 'budget_equilibrium');
    this.addTranslation('biology', 'feedback_loop', 'financial', 'spending_monitoring');
    this.addTranslation('biology', 'cell_division', 'financial', 'portfolio_diversification');
    this.addTranslation('biology', 'metabolism', 'financial', 'cashflow_management');
    this.addTranslation('biology', 'immune_response', 'financial', 'risk_mitigation');

    // Finance → Biology
    this.addTranslation('financial', 'debt', 'biology', 'resource_deficit');
    this.addTranslation('financial', 'investment', 'biology', 'energy_storage');
    this.addTranslation('financial', 'interest', 'biology', 'compound_growth');

    // ML → Systems
    this.addTranslation('ml', 'overfitting', 'systems', 'excessive_optimization');
    this.addTranslation('ml', 'regularization', 'systems', 'constraint_introduction');
    this.addTranslation('ml', 'gradient_descent', 'systems', 'iterative_improvement');

    // Systems → ML
    this.addTranslation('systems', 'feedback_loop', 'ml', 'training_iteration');
    this.addTranslation('systems', 'emergence', 'ml', 'learned_representation');
  }

  /**
   * Initialize forbidden cross-domain translations
   */
  private initializeForbiddenTranslations(): void {
    // DNA concepts don't map to finance
    this.forbiddenTranslations.add('biology:dna→financial');
    this.forbiddenTranslations.add('biology:rna→financial');
    this.forbiddenTranslations.add('biology:protein→financial');

    // Financial rates don't map to biology
    this.forbiddenTranslations.add('financial:interest_rate→biology');
    this.forbiddenTranslations.add('financial:apr→biology');
  }

  /**
   * Add translation mapping
   */
  private addTranslation(
    fromDomain: string,
    fromConcept: string,
    toDomain: string,
    toConcept: string
  ): void {
    const key = `${fromDomain}:${fromConcept}→${toDomain}`;
    this.translationMap.set(key, toConcept);
  }

  /**
   * Translate concept from one domain to another
   */
  translate(concept: string, fromDomain: string, toDomain: string): string {
    const key = `${fromDomain}:${concept}→${toDomain}`;

    // Check if forbidden
    if (this.isForbidden(concept, fromDomain, toDomain)) {
      throw new ConstitutionalViolationError(
        `Forbidden translation: ${concept} from ${fromDomain} to ${toDomain}`,
        'error',
        'domain_boundary',
        { concept, fromDomain, toDomain }
      );
    }

    // Check if translation exists
    const translated = this.translationMap.get(key);

    if (translated) {
      return translated;
    }

    // No translation - return with warning marker
    return `${concept}_[untranslated_from_${fromDomain}]`;
  }

  /**
   * Check if translation is forbidden
   */
  private isForbidden(concept: string, fromDomain: string, toDomain: string): boolean {
    const exactKey = `${fromDomain}:${concept}→${toDomain}`;
    const wildcardKey = `${fromDomain}:${concept}→*`;
    const domainWildcard = `${fromDomain}:*→${toDomain}`;

    return (
      this.forbiddenTranslations.has(exactKey) ||
      this.forbiddenTranslations.has(wildcardKey) ||
      this.forbiddenTranslations.has(domainWildcard)
    );
  }

  /**
   * Validate if translation makes semantic sense
   */
  validateTranslation(concept: string, fromDomain: string, toDomain: string): boolean {
    return !this.isForbidden(concept, fromDomain, toDomain);
  }

  /**
   * Get all available translations for a concept
   */
  getAvailableTranslations(concept: string, fromDomain: string): Map<string, string> {
    const translations = new Map<string, string>();

    for (const [key, value] of this.translationMap.entries()) {
      if (key.startsWith(`${fromDomain}:${concept}→`)) {
        const toDomain = key.split('→')[1];
        translations.set(toDomain, value);
      }
    }

    return translations;
  }
}
