/**
 * AGI System Constitution
 *
 * Foundational principles that ALL agents must follow.
 * Prevents:
 * - Hallucination cascades
 * - Infinite recursion
 * - Cost explosions
 * - Safety violations
 * - Domain boundary violations
 */

// ============================================================================
// Constitution Types
// ============================================================================

export interface ConstitutionPrinciple {
  id: string;
  rule: string;
  enforcement: ConstitutionEnforcement;
}

export interface ConstitutionEnforcement {
  detect_hallucination?: boolean;
  require_source_citation?: boolean;
  confidence_threshold?: number;
  max_depth?: number;
  max_invocations?: number;
  max_cost_usd?: number;
  detect_cycles?: boolean;
  similarity_threshold?: number;
  max_same_agent_consecutive?: number;
  domain_classifier?: boolean;
  cross_domain_penalty?: number;
  require_reasoning_trace?: boolean;
  min_explanation_length?: number;
  content_filter?: boolean;
  harm_detection?: boolean;
  privacy_check?: boolean;
  [key: string]: any;
}

export interface ConstitutionViolation {
  principle_id: string;
  severity: 'warning' | 'error' | 'fatal';
  message: string;
  context: any;
  suggested_action: string;
}

export interface ConstitutionCheckResult {
  passed: boolean;
  violations: ConstitutionViolation[];
  warnings: ConstitutionViolation[];
}

// ============================================================================
// Universal Constitution
// ============================================================================

export class UniversalConstitution {
  name = 'AGI Recursive System Constitution';
  version = '1.0';

  principles: ConstitutionPrinciple[] = [
    {
      id: 'epistemic_honesty',
      rule: `You MUST admit when you don't know something or when it's outside your domain.

        ❌ FORBIDDEN: Inventing facts or pretending expertise
        ✅ CORRECT: "This is outside my domain. I suggest invoking [other_agent]"`,
      enforcement: {
        detect_hallucination: true,
        require_source_citation: true,
        confidence_threshold: 0.7, // If < 0.7, admit uncertainty
      },
    },

    {
      id: 'recursion_budget',
      rule: `You have a BUDGET for invocations.
        - Max depth: 5
        - Max invocations per query: 10
        - Max cost: $1.00

        If limit reached, you MUST:
        1. Return best answer so far
        2. Suggest how user can continue manually`,
      enforcement: {
        max_depth: 5,
        max_invocations: 10,
        max_cost_usd: 1.0,
      },
    },

    {
      id: 'loop_prevention',
      rule: `You MUST detect when you're in a loop:
        - If invoking same agent 2x consecutively with similar context
        - If trace shows cycle (A→B→C→A)

        When loop detected, you MUST:
        1. Break the cycle
        2. Invoke creative_agent for alternative approach`,
      enforcement: {
        detect_cycles: true,
        similarity_threshold: 0.85, // 85% similar context = likely loop
        max_same_agent_consecutive: 2,
      },
    },

    {
      id: 'domain_boundary',
      rule: `You are an expert in [YOUR_DOMAIN].
        You CAN:
        - Answer questions in your domain
        - Suggest connections to other domains

        You CANNOT:
        - Answer authoritatively about other domains
        - Make technical claims outside your area

        If question involves another domain, you MUST:
        1. Answer your part
        2. Return with suggestion_to_invoke: [other_agent]`,
      enforcement: {
        domain_classifier: true,
        cross_domain_penalty: -1.0,
      },
    },

    {
      id: 'reasoning_transparency',
      rule: `You MUST explain your reasoning:
        - Why did you invoke agent X?
        - What insight did each agent bring?
        - How did you arrive at the final synthesis?

        The trace MUST be human-comprehensible.`,
      enforcement: {
        require_reasoning_trace: true,
        min_explanation_length: 50,
      },
    },

    {
      id: 'safety',
      rule: `You CANNOT:
        - Generate malicious code
        - Assist in illegal activities
        - Violate data privacy
        - Cause financial harm to user

        If problematic intent detected, you MUST:
        1. Politely refuse
        2. Explain why you cannot help
        3. Suggest ethical alternative`,
      enforcement: {
        content_filter: true,
        harm_detection: true,
        privacy_check: true,
      },
    },
  ];

  /**
   * Check if response violates constitution
   */
  checkResponse(
    response: any,
    context: {
      agent_id: string;
      depth: number;
      invocation_count: number;
      cost_so_far: number;
      previous_agents: string[];
    }
  ): ConstitutionCheckResult {
    const violations: ConstitutionViolation[] = [];
    const warnings: ConstitutionViolation[] = [];

    // Check epistemic honesty
    const epistemicCheck = this.checkEpistemicHonesty(response);
    if (epistemicCheck) violations.push(epistemicCheck);

    // Check recursion budget
    const budgetCheck = this.checkRecursionBudget(context);
    if (budgetCheck) violations.push(budgetCheck);

    // Check loop prevention
    const loopCheck = this.checkLoopPrevention(context);
    if (loopCheck) violations.push(loopCheck);

    // Check reasoning transparency
    const transparencyCheck = this.checkReasoningTransparency(response);
    if (transparencyCheck) warnings.push(transparencyCheck);

    // Check safety
    const safetyCheck = this.checkSafety(response);
    if (safetyCheck) violations.push(safetyCheck);

    return {
      passed: violations.length === 0,
      violations,
      warnings,
    };
  }

  /**
   * Check epistemic honesty
   */
  private checkEpistemicHonesty(response: any): ConstitutionViolation | null {
    const principle = this.principles.find(p => p.id === 'epistemic_honesty')!;

    // Check confidence threshold
    if (response.confidence !== undefined && response.confidence < principle.enforcement.confidence_threshold!) {
      // Low confidence is OK if acknowledged
      const hasUncertaintyAdmission =
        response.answer.includes("I'm not certain") ||
        response.answer.includes("outside my domain") ||
        response.answer.includes("I suggest invoking") ||
        response.answer.includes("I don't know");

      if (!hasUncertaintyAdmission) {
        return {
          principle_id: 'epistemic_honesty',
          severity: 'warning',
          message: `Low confidence (${response.confidence.toFixed(2)}) but no uncertainty admission`,
          context: { response },
          suggested_action: 'Add uncertainty disclaimer or invoke specialist agent',
        };
      }
    }

    // Check for hallucination markers
    const hallucinationMarkers = [
      'studies show', // Without citation
      'research proves', // Without citation
      'it is known that', // Vague authority
      'experts agree', // Who?
    ];

    for (const marker of hallucinationMarkers) {
      if (response.answer.toLowerCase().includes(marker)) {
        // OK if has citation
        if (!response.sources && !response.references) {
          return {
            principle_id: 'epistemic_honesty',
            severity: 'warning',
            message: `Possible hallucination: "${marker}" without citation`,
            context: { marker, response },
            suggested_action: 'Provide source or rephrase with less certainty',
          };
        }
      }
    }

    return null;
  }

  /**
   * Check recursion budget
   */
  private checkRecursionBudget(context: {
    depth: number;
    invocation_count: number;
    cost_so_far: number;
  }): ConstitutionViolation | null {
    const principle = this.principles.find(p => p.id === 'recursion_budget')!;

    if (context.depth >= principle.enforcement.max_depth!) {
      return {
        principle_id: 'recursion_budget',
        severity: 'fatal',
        message: `Max recursion depth reached (${context.depth}/${principle.enforcement.max_depth})`,
        context,
        suggested_action: 'Return partial answer with trace of reasoning so far',
      };
    }

    if (context.invocation_count >= principle.enforcement.max_invocations!) {
      return {
        principle_id: 'recursion_budget',
        severity: 'fatal',
        message: `Max invocations reached (${context.invocation_count}/${principle.enforcement.max_invocations})`,
        context,
        suggested_action: 'Compose final answer from existing insights',
      };
    }

    if (context.cost_so_far >= principle.enforcement.max_cost_usd!) {
      return {
        principle_id: 'recursion_budget',
        severity: 'fatal',
        message: `Max cost reached ($${context.cost_so_far.toFixed(2)}/$${principle.enforcement.max_cost_usd})`,
        context,
        suggested_action: 'Stop recursion, return best answer so far',
      };
    }

    return null;
  }

  /**
   * Check loop prevention
   */
  private checkLoopPrevention(context: {
    previous_agents: string[];
  }): ConstitutionViolation | null {
    const principle = this.principles.find(p => p.id === 'loop_prevention')!;

    // Check consecutive same agent
    const recent = context.previous_agents.slice(-3);
    const consecutiveSame = recent.filter(a => a === recent[recent.length - 1]).length;

    if (consecutiveSame >= principle.enforcement.max_same_agent_consecutive!) {
      return {
        principle_id: 'loop_prevention',
        severity: 'error',
        message: `Same agent invoked ${consecutiveSame} times consecutively`,
        context: { recent_agents: recent },
        suggested_action: 'Invoke different agent or creative_agent for alternative approach',
      };
    }

    // Check for cycles (A→B→C→A)
    const agentChain = context.previous_agents.join('→');
    const lastAgent = context.previous_agents[context.previous_agents.length - 1];

    if (context.previous_agents.length >= 4) {
      const firstOccurrence = context.previous_agents.indexOf(lastAgent);
      if (firstOccurrence !== -1 && firstOccurrence < context.previous_agents.length - 1) {
        return {
          principle_id: 'loop_prevention',
          severity: 'error',
          message: `Cycle detected: ${lastAgent} appears multiple times`,
          context: { chain: agentChain },
          suggested_action: 'Break cycle by invoking creative_agent or composing final answer',
        };
      }
    }

    return null;
  }

  /**
   * Check reasoning transparency
   */
  private checkReasoningTransparency(response: any): ConstitutionViolation | null {
    const principle = this.principles.find(p => p.id === 'reasoning_transparency')!;

    if (!response.reasoning) {
      return {
        principle_id: 'reasoning_transparency',
        severity: 'warning',
        message: 'No reasoning provided in response',
        context: { response },
        suggested_action: 'Add reasoning field explaining how you arrived at answer',
      };
    }

    if (response.reasoning.length < principle.enforcement.min_explanation_length!) {
      return {
        principle_id: 'reasoning_transparency',
        severity: 'warning',
        message: `Reasoning too short (${response.reasoning.length} chars)`,
        context: { reasoning: response.reasoning },
        suggested_action: 'Provide more detailed explanation of reasoning process',
      };
    }

    return null;
  }

  /**
   * Check safety
   */
  private checkSafety(response: any): ConstitutionViolation | null {
    // Check for harmful content markers
    const harmfulMarkers = [
      'hack',
      'exploit',
      'illegal',
      'steal',
      'fraud',
      'manipulate',
    ];

    const answerLower = response.answer.toLowerCase();

    for (const marker of harmfulMarkers) {
      if (answerLower.includes(marker)) {
        // OK if discussing security/prevention
        const safetyContext = [
          'prevent',
          'protect',
          'secure',
          'defend',
          'avoid',
        ];

        const hasSafetyContext = safetyContext.some(ctx => answerLower.includes(ctx));

        if (!hasSafetyContext) {
          return {
            principle_id: 'safety',
            severity: 'error',
            message: `Potentially harmful content detected: "${marker}"`,
            context: { marker, answer_snippet: response.answer.substring(0, 200) },
            suggested_action: 'Review response for harmful instructions, rephrase or refuse',
          };
        }
      }
    }

    return null;
  }
}

// ============================================================================
// Agent-Specific Constitutions
// ============================================================================

export class FinancialAgentConstitution extends UniversalConstitution {
  constructor() {
    super();
    this.name = 'Financial Agent Constitution';

    // Add financial-specific principles
    this.principles.push({
      id: 'financial_responsibility',
      rule: `You MUST NEVER:
        - Promise guaranteed returns
        - Give personalized investment advice without disclaimer
        - Suggest illegal actions (tax evasion, etc)

        You MUST ALWAYS:
        - Include disclaimer: "I'm not a certified financial advisor"
        - Suggest consulting professional for major decisions
        - Consider user's financial situation`,
      enforcement: {
        require_disclaimer: true,
        detect_investment_advice: true,
      },
    });

    this.principles.push({
      id: 'privacy_protection',
      rule: `Financial data is SENSITIVE.
        You MUST:
        - Never log exact values in traces
        - Mask personal information
        - Never share data between users`,
      enforcement: {
        mask_financial_data: true,
        anonymize_traces: true,
      },
    });
  }
}

export class BiologyAgentConstitution extends UniversalConstitution {
  constructor() {
    super();
    this.name = 'Biology Agent Constitution';

    this.principles.push({
      id: 'scientific_accuracy',
      rule: `Biology is SCIENCE.
        You MUST:
        - Base claims on scientific consensus
        - Cite sources when possible
        - Clearly distinguish: fact vs hypothesis vs speculation

        You MUST NOT:
        - Make specific medical claims
        - Suggest diagnoses or treatments`,
      enforcement: {
        require_scientific_basis: true,
        detect_medical_advice: true,
      },
    });

    this.principles.push({
      id: 'abstraction_grounding',
      rule: `You LOVE analogies, but they have limits.
        You MUST:
        - Make clear when making an analogy
        - Explain where analogy breaks down
        - Not extrapolate beyond reasonable`,
      enforcement: {
        label_analogies: true,
        acknowledge_limits: true,
      },
    });
  }
}

// ============================================================================
// Constitution Enforcer
// ============================================================================

export class ConstitutionEnforcer {
  private constitutions: Map<string, UniversalConstitution>;

  constructor() {
    this.constitutions = new Map();
    this.constitutions.set('universal', new UniversalConstitution());
    this.constitutions.set('financial', new FinancialAgentConstitution());
    this.constitutions.set('biology', new BiologyAgentConstitution());
  }

  /**
   * Validate agent response against constitution
   */
  validate(
    agentId: string,
    response: any,
    context: {
      depth: number;
      invocation_count: number;
      cost_so_far: number;
      previous_agents: string[];
    }
  ): ConstitutionCheckResult {
    const constitution =
      this.constitutions.get(agentId) || this.constitutions.get('universal')!;

    return constitution.checkResponse(response, { ...context, agent_id: agentId });
  }

  /**
   * Handle violation
   */
  handleViolation(violation: ConstitutionViolation): {
    action: 'warn' | 'stop' | 'modify';
    message: string;
  } {
    switch (violation.severity) {
      case 'fatal':
        return {
          action: 'stop',
          message: `FATAL: ${violation.message}. ${violation.suggested_action}`,
        };

      case 'error':
        return {
          action: 'modify',
          message: `ERROR: ${violation.message}. ${violation.suggested_action}`,
        };

      case 'warning':
        return {
          action: 'warn',
          message: `WARNING: ${violation.message}. ${violation.suggested_action}`,
        };
    }
  }

  /**
   * Format constitution report
   */
  formatReport(result: ConstitutionCheckResult): string {
    if (result.passed && result.warnings.length === 0) {
      return '✅ Constitution check passed';
    }

    let report = '';

    if (result.violations.length > 0) {
      report += '❌ VIOLATIONS:\n';
      for (const v of result.violations) {
        report += `  [${v.severity.toUpperCase()}] ${v.principle_id}: ${v.message}\n`;
        report += `    → ${v.suggested_action}\n`;
      }
    }

    if (result.warnings.length > 0) {
      report += '\n⚠️  WARNINGS:\n';
      for (const w of result.warnings) {
        report += `  ${w.principle_id}: ${w.message}\n`;
        report += `    → ${w.suggested_action}\n`;
      }
    }

    return report;
  }
}

/**
 * Alias for UniversalConstitution for convenience
 */
export const Constitution = UniversalConstitution;
