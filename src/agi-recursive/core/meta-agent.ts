/**
 * AGI Meta-Agent
 *
 * Orchestrates recursive composition of specialized agents.
 * Intelligence emerges from composition, not model size.
 */

import Anthropic from '@anthropic-ai/sdk';
import {
  ConstitutionEnforcer,
  ConstitutionCheckResult,
  ConstitutionViolation,
  UniversalConstitution,
} from './constitution';
import {
  AntiCorruptionLayer,
  DomainTranslator,
  ConstitutionalViolationError,
} from './anti-corruption-layer';

// ============================================================================
// Types
// ============================================================================

export interface AgentResponse {
  answer: string;
  concepts: string[];
  suggestions_to_invoke?: string[];
  confidence: number;
  reasoning: string;
  sources?: string[];
  references?: string[];
}

export interface RecursionTrace {
  depth: number;
  agent_id: string;
  query: string;
  response: AgentResponse;
  timestamp: number;
  cost_estimate: number;
}

export interface RecursionState {
  depth: number;
  invocation_count: number;
  cost_so_far: number;
  previous_agents: string[];
  traces: RecursionTrace[];
  insights: Map<string, AgentResponse>;
}

export interface QueryDecomposition {
  domains: string[];
  reasoning: string;
  primary_domain?: string;
}

export interface CompositionResult {
  synthesis: string;
  should_recurse: boolean;
  missing_perspectives?: string[];
  confidence: number;
}

// ============================================================================
// Base Agent Class
// ============================================================================

export abstract class SpecializedAgent {
  protected client: Anthropic;
  protected systemPrompt: string;
  protected defaultTemperature: number;

  constructor(apiKey: string, systemPrompt: string, temperature: number = 0.5) {
    this.client = new Anthropic({ apiKey });
    this.systemPrompt = systemPrompt;
    this.defaultTemperature = temperature;
  }

  abstract getDomain(): string;

  async process(query: string, context: RecursionState): Promise<AgentResponse> {
    const contextPrompt = this.buildContextPrompt(context);

    const message = await this.client.messages.create({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 2000,
      temperature: this.defaultTemperature,
      system: this.systemPrompt,
      messages: [
        {
          role: 'user',
          content: `${contextPrompt}

QUERY: ${query}

You MUST respond with valid JSON:
{
  "answer": "your detailed answer",
  "concepts": ["concept1", "concept2"],
  "suggestions_to_invoke": ["agent_id"],
  "confidence": 0.85,
  "reasoning": "explain your thought process"
}`,
        },
      ],
    });

    const responseText = message.content[0].type === 'text' ? message.content[0].text : '';

    try {
      const parsed = JSON.parse(responseText);
      return {
        answer: parsed.answer || '',
        concepts: parsed.concepts || [],
        suggestions_to_invoke: parsed.suggestions_to_invoke,
        confidence: parsed.confidence || 0.5,
        reasoning: parsed.reasoning || '',
      };
    } catch (e) {
      // Fallback if not valid JSON
      return {
        answer: responseText,
        concepts: [],
        confidence: 0.3,
        reasoning: 'Failed to parse JSON response',
      };
    }
  }

  private buildContextPrompt(context: RecursionState): string {
    if (context.traces.length === 0) return '';

    const previousInsights = Array.from(context.insights.entries())
      .map(([agent, response]) => `[${agent}]: ${response.answer}`)
      .join('\n');

    return `PREVIOUS INSIGHTS:
${previousInsights}

RECURSION DEPTH: ${context.depth}
PREVIOUS AGENTS: ${context.previous_agents.join(' → ')}`;
  }
}

// ============================================================================
// Meta-Agent (Orchestrator)
// ============================================================================

export class MetaAgent {
  private client: Anthropic;
  private agents: Map<string, SpecializedAgent>;
  private maxDepth: number;
  private maxInvocations: number;
  private maxCostUSD: number;
  private constitutionEnforcer: ConstitutionEnforcer;
  private antiCorruptionLayer: AntiCorruptionLayer;
  private domainTranslator: DomainTranslator;

  constructor(
    apiKey: string,
    maxDepth: number = 5,
    maxInvocations: number = 10,
    maxCostUSD: number = 1.0
  ) {
    this.client = new Anthropic({ apiKey });
    this.agents = new Map();
    this.maxDepth = maxDepth;
    this.maxInvocations = maxInvocations;
    this.maxCostUSD = maxCostUSD;
    this.constitutionEnforcer = new ConstitutionEnforcer();
    this.antiCorruptionLayer = new AntiCorruptionLayer(new UniversalConstitution());
    this.domainTranslator = new DomainTranslator();
  }

  registerAgent(id: string, agent: SpecializedAgent): void {
    this.agents.set(id, agent);
  }

  /**
   * Main recursive processing loop
   */
  async process(query: string): Promise<{
    final_answer: string;
    trace: RecursionTrace[];
    emergent_insights: string[];
    reasoning_path: string;
    constitution_violations: ConstitutionViolation[];
  }> {
    const state: RecursionState = {
      depth: 0,
      invocation_count: 0,
      cost_so_far: 0,
      previous_agents: [],
      traces: [],
      insights: new Map(),
    };

    const allViolations: ConstitutionViolation[] = [];

    await this.recursiveProcess(query, state, allViolations);

    const synthesis = await this.synthesizeFinal(state);

    return {
      final_answer: synthesis.answer,
      trace: state.traces,
      emergent_insights: this.extractEmergentInsights(state),
      reasoning_path: this.formatReasoningPath(state),
      constitution_violations: allViolations,
    };
  }

  /**
   * Recursive processing with Constitution enforcement
   */
  private async recursiveProcess(
    query: string,
    state: RecursionState,
    violations: ConstitutionViolation[]
  ): Promise<void> {
    // Check budget constraints
    const budgetCheck = this.constitutionEnforcer.validate(
      'meta',
      { answer: '', confidence: 1.0, reasoning: 'budget check' },
      {
        depth: state.depth,
        invocation_count: state.invocation_count,
        cost_so_far: state.cost_so_far,
        previous_agents: state.previous_agents,
      }
    );

    if (!budgetCheck.passed) {
      violations.push(...budgetCheck.violations);
      return; // Stop recursion
    }

    // Step 1: Decompose query
    const decomposition = await this.decomposeQuery(query, state);

    // Step 2: Invoke specialist agents
    for (const domain of decomposition.domains) {
      const agent = this.agents.get(domain);
      if (!agent) continue;

      state.invocation_count++;
      state.depth++;
      state.previous_agents.push(domain);

      const response = await agent.process(query, state);

      // Validate against Anti-Corruption Layer FIRST
      try {
        this.antiCorruptionLayer.validateResponse(response, agent.getDomain(), state);
      } catch (error) {
        if (error instanceof ConstitutionalViolationError) {
          violations.push({
            principle_id: error.principle_id,
            severity: error.severity,
            message: error.message,
            context: error.context,
            suggested_action: 'Review agent response for domain boundary violations',
          });

          // Fatal ACL violations stop processing
          if (error.severity === 'fatal') {
            state.depth--;
            continue;
          }
        } else {
          throw error;
        }
      }

      // Validate against Constitution
      const constitutionCheck = this.constitutionEnforcer.validate(domain, response, {
        depth: state.depth,
        invocation_count: state.invocation_count,
        cost_so_far: state.cost_so_far,
        previous_agents: state.previous_agents,
      });

      if (!constitutionCheck.passed) {
        violations.push(...constitutionCheck.violations);

        // Handle fatal violations
        const hasFatal = constitutionCheck.violations.some((v) => v.severity === 'fatal');
        if (hasFatal) {
          state.depth--;
          continue; // Skip this agent
        }
      }

      // Add warnings to violations list
      violations.push(...constitutionCheck.warnings);

      const trace: RecursionTrace = {
        depth: state.depth,
        agent_id: domain,
        query,
        response,
        timestamp: Date.now(),
        cost_estimate: this.estimateCost(response),
      };

      state.traces.push(trace);
      state.insights.set(domain, response);
      state.cost_so_far += trace.cost_estimate;

      state.depth--;
    }

    // Step 3: Check if composition suggests recursion
    const composition = await this.composeInsights(state);

    if (composition.should_recurse && state.depth < this.maxDepth) {
      await this.recursiveProcess(composition.synthesis, state, violations);
    }
  }

  /**
   * Decompose query into relevant domains
   */
  private async decomposeQuery(
    query: string,
    state: RecursionState
  ): Promise<QueryDecomposition> {
    const availableDomains = Array.from(this.agents.keys());

    const message = await this.client.messages.create({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1000,
      temperature: 0.3,
      system: `You are a meta-reasoning system. Your job is to decompose queries into relevant domains.

Available domains: ${availableDomains.join(', ')}

Respond with JSON:
{
  "domains": ["domain1", "domain2"],
  "reasoning": "why these domains",
  "primary_domain": "most relevant domain"
}`,
      messages: [
        {
          role: 'user',
          content: query,
        },
      ],
    });

    const responseText = message.content[0].type === 'text' ? message.content[0].text : '';

    try {
      const parsed = JSON.parse(responseText);
      return {
        domains: parsed.domains || availableDomains,
        reasoning: parsed.reasoning || '',
        primary_domain: parsed.primary_domain,
      };
    } catch (e) {
      // Default to all domains
      return {
        domains: availableDomains,
        reasoning: 'Failed to decompose, using all domains',
      };
    }
  }

  /**
   * Compose insights from multiple agents
   */
  private async composeInsights(state: RecursionState): Promise<CompositionResult> {
    if (state.insights.size === 0) {
      return {
        synthesis: '',
        should_recurse: false,
        confidence: 0,
      };
    }

    const insightsText = Array.from(state.insights.entries())
      .map(([agent, response]) => `[${agent}]:\n${response.answer}\nConfidence: ${response.confidence}\nConcepts: ${response.concepts.join(', ')}`)
      .join('\n\n');

    const message = await this.client.messages.create({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 2000,
      temperature: 0.7,
      system: `You are a synthesis engine. Your job is to compose insights from multiple specialists.

Look for:
- Emergent patterns (what no single agent saw)
- Contradictions (need resolution)
- Missing perspectives (suggest recursion)

Respond with JSON:
{
  "synthesis": "composed insight",
  "should_recurse": false,
  "missing_perspectives": ["domain"],
  "confidence": 0.9
}`,
      messages: [
        {
          role: 'user',
          content: `INSIGHTS FROM SPECIALISTS:\n\n${insightsText}\n\nCompose these insights. Should we recurse to explore further?`,
        },
      ],
    });

    const responseText = message.content[0].type === 'text' ? message.content[0].text : '';

    try {
      const parsed = JSON.parse(responseText);
      return {
        synthesis: parsed.synthesis || '',
        should_recurse: parsed.should_recurse || false,
        missing_perspectives: parsed.missing_perspectives,
        confidence: parsed.confidence || 0.5,
      };
    } catch (e) {
      return {
        synthesis: responseText,
        should_recurse: false,
        confidence: 0.3,
      };
    }
  }

  /**
   * Final synthesis of all insights
   */
  private async synthesizeFinal(state: RecursionState): Promise<AgentResponse> {
    const traceText = state.traces
      .map((t) => `[Depth ${t.depth}] ${t.agent_id}: ${t.response.answer}`)
      .join('\n\n');

    const message = await this.client.messages.create({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 3000,
      temperature: 0.5,
      system: `You are the final synthesis agent. Compose all insights into a coherent answer.

IMPORTANT:
- Highlight emergent insights (what no single agent could see)
- Show how different perspectives combined
- Provide actionable recommendations`,
      messages: [
        {
          role: 'user',
          content: `FULL REASONING TRACE:\n\n${traceText}\n\nProvide final synthesis as JSON:
{
  "answer": "comprehensive answer",
  "concepts": ["emergent concepts"],
  "confidence": 0.95,
  "reasoning": "how insights were composed"
}`,
        },
      ],
    });

    const responseText = message.content[0].type === 'text' ? message.content[0].text : '';

    try {
      const parsed = JSON.parse(responseText);
      return {
        answer: parsed.answer || responseText,
        concepts: parsed.concepts || [],
        confidence: parsed.confidence || 0.5,
        reasoning: parsed.reasoning || '',
      };
    } catch (e) {
      return {
        answer: responseText,
        concepts: [],
        confidence: 0.3,
        reasoning: 'Failed to parse final synthesis',
      };
    }
  }

  /**
   * Extract emergent insights (concepts no single agent mentioned)
   */
  private extractEmergentInsights(state: RecursionState): string[] {
    const allConcepts = new Set<string>();
    const agentConcepts = new Map<string, Set<string>>();

    for (const [agentId, response] of state.insights.entries()) {
      const concepts = new Set(response.concepts);
      agentConcepts.set(agentId, concepts);

      for (const concept of concepts) {
        allConcepts.add(concept);
      }
    }

    // Find concepts mentioned by synthesis but not by individual agents
    const emergent: string[] = [];

    // This is a simplified heuristic - in practice, you'd do semantic analysis
    for (const trace of state.traces) {
      for (const concept of trace.response.concepts) {
        const mentionedByOthers = Array.from(agentConcepts.values()).filter((set) =>
          set.has(concept)
        ).length;

        if (mentionedByOthers === 0) {
          emergent.push(concept);
        }
      }
    }

    return Array.from(new Set(emergent));
  }

  /**
   * Format reasoning path for human comprehension
   */
  private formatReasoningPath(state: RecursionState): string {
    let path = 'REASONING PATH:\n\n';

    for (const trace of state.traces) {
      const indent = '  '.repeat(trace.depth);
      path += `${indent}[${trace.agent_id}] (depth: ${trace.depth}, confidence: ${trace.response.confidence})\n`;
      path += `${indent}→ ${trace.response.reasoning}\n`;
      path += `${indent}→ Concepts: ${trace.response.concepts.join(', ')}\n\n`;
    }

    return path;
  }

  /**
   * Estimate cost of API call (rough heuristic)
   */
  private estimateCost(response: AgentResponse): number {
    // Rough estimate: $0.01 per 1000 tokens
    // Average response ~500 tokens
    return 0.005;
  }
}
