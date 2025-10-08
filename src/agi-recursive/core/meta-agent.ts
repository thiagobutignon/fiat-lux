/**
 * AGI Meta-Agent
 *
 * Orchestrates recursive composition of specialized agents.
 * Intelligence emerges from composition, not model size.
 */

import path from 'path';
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
import { SliceNavigator } from './slice-navigator';
import { AnthropicAdapter, LLMResponse, ClaudeModel } from '../llm/anthropic-adapter';
import {
  AttentionTracker,
  QueryAttention,
  computeInfluenceWeight,
  extractInfluentialConcepts,
} from './attention-tracker';

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
// Utility Functions
// ============================================================================

/**
 * Extract JSON from markdown code blocks if present
 * Handles multiple formats:
 * - ```json\n{...}\n```
 * - ```\n{...}\n```
 * - Plain JSON
 */
function extractJSON(text: string): string {
  // Try to match markdown code blocks
  const codeBlockRegex = /```(?:json)?\s*\n?([\s\S]*?)```/;
  const match = text.match(codeBlockRegex);

  if (match && match[1]) {
    return match[1].trim();
  }

  // Fallback: try to extract JSON by finding first { and last }
  const firstBrace = text.indexOf('{');
  const lastBrace = text.lastIndexOf('}');

  if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
    return text.substring(firstBrace, lastBrace + 1).trim();
  }

  // Last resort: return trimmed text
  return text.trim();
}

// ============================================================================
// Base Agent Class
// ============================================================================

export abstract class SpecializedAgent {
  protected llm: AnthropicAdapter;
  protected systemPrompt: string;
  protected defaultTemperature: number;
  protected defaultModel: ClaudeModel;
  protected sliceNavigator?: SliceNavigator;

  constructor(
    apiKey: string,
    systemPrompt: string,
    temperature: number = 0.5,
    model: ClaudeModel = 'claude-sonnet-4-5'
  ) {
    this.llm = new AnthropicAdapter(apiKey);
    this.systemPrompt = systemPrompt;
    this.defaultTemperature = temperature;
    this.defaultModel = model;
  }

  /**
   * Set slice navigator for dynamic knowledge loading
   */
  setSliceNavigator(navigator: SliceNavigator): void {
    this.sliceNavigator = navigator;
  }

  abstract getDomain(): string;

  async process(query: string, context: RecursionState): Promise<AgentResponse> {
    const contextPrompt = this.buildContextPrompt(context);

    const fullPrompt = `${contextPrompt}

QUERY: ${query}

You MUST respond with valid JSON:
{
  "answer": "your detailed answer",
  "concepts": ["concept1", "concept2"],
  "suggestions_to_invoke": ["agent_id"],
  "confidence": 0.85,
  "reasoning": "explain your thought process"
}`;

    const llmResponse = await this.llm.invoke(this.systemPrompt, fullPrompt, {
      model: this.defaultModel,
      max_tokens: 2000,
      temperature: this.defaultTemperature,
    });

    // Update context with actual cost
    context.cost_so_far += llmResponse.usage.cost_usd;

    try {
      // Extract JSON from potential markdown code blocks
      const cleanedText = extractJSON(llmResponse.text);
      const parsed = JSON.parse(cleanedText);
      return {
        answer: parsed.answer || '',
        concepts: parsed.concepts || [],
        suggestions_to_invoke: parsed.suggestions_to_invoke,
        confidence: parsed.confidence || 0.5,
        reasoning: parsed.reasoning || '',
      };
    } catch (e) {
      // Fallback if not valid JSON
      const errorMessage = e instanceof Error ? e.message : 'Unknown error';
      console.warn(`❌ Failed to parse JSON from ${this.getDomain()}:`, errorMessage);
      console.warn('Raw response (first 200 chars):', llmResponse.text.substring(0, 200));

      return {
        answer: llmResponse.text,
        concepts: [],
        confidence: 0.3,
        reasoning: `Failed to parse JSON response: ${errorMessage}`,
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
  private llm: AnthropicAdapter;
  private agents: Map<string, SpecializedAgent>;
  private maxDepth: number;
  private maxInvocations: number;
  private maxCostUSD: number;
  private constitutionEnforcer: ConstitutionEnforcer;
  private antiCorruptionLayer: AntiCorruptionLayer;
  private domainTranslator: DomainTranslator;
  private sliceNavigator: SliceNavigator;
  private attentionTracker: AttentionTracker;

  constructor(
    apiKey: string,
    maxDepth: number = 5,
    maxInvocations: number = 10,
    maxCostUSD: number = 1.0
  ) {
    this.llm = new AnthropicAdapter(apiKey);
    this.agents = new Map();
    this.maxDepth = maxDepth;
    this.maxInvocations = maxInvocations;
    this.maxCostUSD = maxCostUSD;
    this.constitutionEnforcer = new ConstitutionEnforcer();
    this.antiCorruptionLayer = new AntiCorruptionLayer(new UniversalConstitution());
    this.domainTranslator = new DomainTranslator();
    this.attentionTracker = new AttentionTracker();

    // Initialize slice navigator
    const slicesDir = path.join(__dirname, '..', 'slices');
    this.sliceNavigator = new SliceNavigator(slicesDir);
  }

  /**
   * Initialize the meta-agent (must be called before processing)
   */
  async initialize(): Promise<void> {
    await this.sliceNavigator.initialize();
  }

  registerAgent(id: string, agent: SpecializedAgent): void {
    this.agents.set(id, agent);
    // Give agent access to slice navigator
    agent.setSliceNavigator(this.sliceNavigator);
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
    attention: QueryAttention | null;
  }> {
    const state: RecursionState = {
      depth: 0,
      invocation_count: 0,
      cost_so_far: 0,
      previous_agents: [],
      traces: [],
      insights: new Map(),
    };

    // Start attention tracking
    const queryId = `query_${Date.now()}_${Math.random().toString(36).substring(7)}`;
    this.attentionTracker.startQuery(queryId, query);

    const allViolations: ConstitutionViolation[] = [];

    await this.recursiveProcess(query, state, allViolations);

    const synthesis = await this.synthesizeFinal(state);

    // End attention tracking
    const attention = this.attentionTracker.endQuery();

    return {
      final_answer: synthesis.answer,
      trace: state.traces,
      emergent_insights: this.extractEmergentInsights(state),
      reasoning_path: this.formatReasoningPath(state),
      constitution_violations: allViolations,
      attention,
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

    // Track query decomposition decision
    this.attentionTracker.addDecisionPoint(
      `Query decomposed into domains: ${decomposition.domains.join(', ')}`
    );

    // Track domain selection as attention traces
    for (const domain of decomposition.domains) {
      this.attentionTracker.addTrace(
        'domain_selection',
        `meta-agent/decomposition`,
        0.8, // High weight as this is a critical decision
        `Selected ${domain} based on: ${decomposition.reasoning}`
      );
    }

    // Step 2: Invoke specialist agents
    for (const domain of decomposition.domains) {
      const agent = this.agents.get(domain);
      if (!agent) continue;

      state.invocation_count++;
      state.depth++;
      state.previous_agents.push(domain);

      // Track cost before agent call
      const costBefore = state.cost_so_far;

      const response = await agent.process(query, state);

      // Calculate actual cost (agent.process updates state.cost_so_far)
      const actualCost = state.cost_so_far - costBefore;

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
        cost_estimate: actualCost,
      };

      state.traces.push(trace);
      state.insights.set(domain, response);

      // Track attention: which concepts from this agent influenced the decision
      this.attentionTracker.addDecisionPoint(
        `Agent ${domain} invoked with confidence ${response.confidence}`
      );

      for (const concept of response.concepts) {
        // Weight based on agent confidence and concept relevance
        const weight = response.confidence;
        this.attentionTracker.addTrace(
          concept,
          `agent/${domain}`,
          weight,
          `${domain} contributed concept "${concept}": ${response.reasoning.substring(0, 100)}...`
        );
      }

      // Track references if available
      if (response.references) {
        for (const reference of response.references) {
          this.attentionTracker.addTrace(
            'knowledge_reference',
            reference,
            response.confidence * 0.7, // Slightly lower weight for references
            `${domain} referenced: ${reference}`
          );
        }
      }

      state.depth--;
    }

    // Step 3: Check if composition suggests recursion
    const composition = await this.composeInsights(state);

    // Track composition decision
    this.attentionTracker.addDecisionPoint(
      `Composition confidence: ${composition.confidence}, should_recurse: ${composition.should_recurse}`
    );

    if (composition.should_recurse) {
      this.attentionTracker.addTrace(
        'composition_recursion',
        'meta-agent/composition',
        composition.confidence,
        `Composition suggests recursion due to: ${composition.missing_perspectives?.join(', ') || 'further exploration needed'}`
      );
    }

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

    const systemPrompt = `You are a meta-reasoning system. Your job is to decompose queries into relevant domains.

Available domains: ${availableDomains.join(', ')}

Respond with JSON:
{
  "domains": ["domain1", "domain2"],
  "reasoning": "why these domains",
  "primary_domain": "most relevant domain"
}`;

    const llmResponse = await this.llm.invoke(systemPrompt, query, {
      model: 'claude-sonnet-4-5',
      max_tokens: 1000,
      temperature: 0.3,
    });

    // Update cost tracking
    state.cost_so_far += llmResponse.usage.cost_usd;

    try {
      // Extract JSON from potential markdown code blocks
      const cleanedText = extractJSON(llmResponse.text);
      const parsed = JSON.parse(cleanedText);
      return {
        domains: parsed.domains || availableDomains,
        reasoning: parsed.reasoning || '',
        primary_domain: parsed.primary_domain,
      };
    } catch (e) {
      // Fallback: use first 2 domains instead of all to limit cost
      const errorMessage = e instanceof Error ? e.message : 'Unknown error';
      console.warn('❌ Query decomposition failed:', errorMessage);
      console.warn('Raw response (first 200 chars):', llmResponse.text.substring(0, 200));

      const fallbackDomains = availableDomains.slice(0, 2);
      return {
        domains: fallbackDomains,
        reasoning: `Failed to decompose: ${errorMessage}. Using fallback strategy (first ${fallbackDomains.length} domains).`,
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

    const systemPrompt = `You are a synthesis engine. Your job is to compose insights from multiple specialists.

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
}`;

    const llmResponse = await this.llm.invoke(
      systemPrompt,
      `INSIGHTS FROM SPECIALISTS:\n\n${insightsText}\n\nCompose these insights. Should we recurse to explore further?`,
      {
        model: 'claude-sonnet-4-5',
        max_tokens: 2000,
        temperature: 0.7,
      }
    );

    // Update cost tracking
    state.cost_so_far += llmResponse.usage.cost_usd;

    try {
      // Extract JSON from potential markdown code blocks
      const cleanedText = extractJSON(llmResponse.text);
      const parsed = JSON.parse(cleanedText);
      return {
        synthesis: parsed.synthesis || '',
        should_recurse: parsed.should_recurse || false,
        missing_perspectives: parsed.missing_perspectives,
        confidence: parsed.confidence || 0.5,
      };
    } catch (e) {
      const errorMessage = e instanceof Error ? e.message : 'Unknown error';
      console.warn('❌ Failed to parse composition response:', errorMessage);
      console.warn('Raw response (first 200 chars):', llmResponse.text.substring(0, 200));

      return {
        synthesis: llmResponse.text,
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

    const systemPrompt = `You are the final synthesis agent. Compose all insights into a coherent answer.

IMPORTANT:
- Highlight emergent insights (what no single agent could see)
- Show how different perspectives combined
- Provide actionable recommendations`;

    const llmResponse = await this.llm.invoke(
      systemPrompt,
      `FULL REASONING TRACE:\n\n${traceText}\n\nProvide final synthesis as JSON:
{
  "answer": "comprehensive answer",
  "concepts": ["emergent concepts"],
  "confidence": 0.95,
  "reasoning": "how insights were composed"
}`,
      {
        model: 'claude-sonnet-4-5',
        max_tokens: 3000,
        temperature: 0.5,
      }
    );

    // Update cost tracking
    state.cost_so_far += llmResponse.usage.cost_usd;

    try {
      // Extract JSON from potential markdown code blocks
      const cleanedText = extractJSON(llmResponse.text);
      const parsed = JSON.parse(cleanedText);
      return {
        answer: parsed.answer || llmResponse.text,
        concepts: parsed.concepts || [],
        confidence: parsed.confidence || 0.5,
        reasoning: parsed.reasoning || '',
      };
    } catch (e) {
      const errorMessage = e instanceof Error ? e.message : 'Unknown error';
      console.warn('❌ Failed to parse final synthesis:', errorMessage);
      console.warn('Raw response (first 200 chars):', llmResponse.text.substring(0, 200));

      return {
        answer: llmResponse.text,
        concepts: [],
        confidence: 0.3,
        reasoning: `Failed to parse final synthesis: ${errorMessage}`,
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
   * Get total cost spent across all LLM calls
   */
  getTotalCost(): number {
    return this.llm.getTotalCost();
  }

  /**
   * Get total number of LLM requests made
   */
  getTotalRequests(): number {
    return this.llm.getTotalRequests();
  }

  /**
   * Get the attention tracker for interpretability analysis
   */
  getAttentionTracker(): AttentionTracker {
    return this.attentionTracker;
  }

  /**
   * Export attention data for regulatory auditing
   */
  exportAttentionForAudit() {
    return this.attentionTracker.exportForAudit();
  }

  /**
   * Get human-readable explanation of a query's reasoning
   */
  explainQuery(queryId: string): string {
    return this.attentionTracker.explainQuery(queryId);
  }

  /**
   * Get attention statistics across all queries
   */
  getAttentionStats() {
    return this.attentionTracker.getStatistics();
  }
}
