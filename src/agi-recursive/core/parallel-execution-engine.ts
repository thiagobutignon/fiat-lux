/**
 * Parallel Execution Engine
 *
 * Enables true "Quantum-like Superposition" by executing multiple
 * agent paths simultaneously (in parallel), then "collapsing" to final decision.
 *
 * This transforms the system from sequential to parallel composition.
 */

import type { SpecializedAgent, AgentResponse, RecursionState } from './meta-agent';

// ============================================================================
// Types
// ============================================================================

export interface ParallelExecution {
  query: string;
  agents: string[];
  responses: Map<string, AgentResponse>;
  execution_times_ms: Map<string, number>;
  success_count: number;
  failure_count: number;
  total_time_ms: number;
  speedup_factor: number; // vs sequential
}

export interface CollapsedResult {
  final_answer: string;
  contributing_agents: string[];
  confidence: number;
  reasoning_synthesis: string;
  parallel_execution: ParallelExecution;
}

export interface SuperpositionState {
  query: string;
  active_paths: number;
  collapsed: boolean;
  collapse_trigger: 'threshold' | 'timeout' | 'complete';
  entropy: number; // Measure of uncertainty across paths
}

// ============================================================================
// Parallel Execution Engine
// ============================================================================

export class ParallelExecutionEngine {
  private maxParallelAgents: number = 5;
  private collapseThreshold: number = 0.8; // If one path has >80% confidence, collapse early
  private timeout_ms: number = 10000; // 10 seconds max

  constructor(
    maxParallelAgents?: number,
    collapseThreshold?: number,
    timeout_ms?: number
  ) {
    if (maxParallelAgents) this.maxParallelAgents = maxParallelAgents;
    if (collapseThreshold) this.collapseThreshold = collapseThreshold;
    if (timeout_ms) this.timeout_ms = timeout_ms;
  }

  /**
   * Execute multiple agents in parallel (TRUE SUPERPOSITION)
   */
  async executeParallel(
    query: string,
    agents: Map<string, SpecializedAgent>,
    domains: string[],
    state: RecursionState
  ): Promise<ParallelExecution> {
    const start = Date.now();

    // Limit parallelism
    const selected_domains = domains.slice(0, this.maxParallelAgents);

    // Create execution promises
    const executions = selected_domains.map(async (domain) => {
      const agent = agents.get(domain);
      if (!agent) return null;

      const agent_start = Date.now();
      try {
        const response = await agent.process(query, state);
        const agent_time = Date.now() - agent_start;

        return {
          domain,
          response,
          execution_time_ms: agent_time,
          success: true,
        };
      } catch (error) {
        return {
          domain,
          response: null,
          execution_time_ms: Date.now() - agent_start,
          success: false,
          error,
        };
      }
    });

    // Execute ALL agents simultaneously
    const results = await Promise.all(executions);

    // Aggregate results
    const responses = new Map<string, AgentResponse>();
    const execution_times = new Map<string, number>();
    let success_count = 0;
    let failure_count = 0;

    for (const result of results) {
      if (!result) continue;

      execution_times.set(result.domain, result.execution_time_ms);

      if (result.success && result.response) {
        responses.set(result.domain, result.response);
        success_count++;
      } else {
        failure_count++;
      }
    }

    const total_time = Date.now() - start;

    // Calculate speedup vs sequential
    const sequential_time = Array.from(execution_times.values()).reduce((sum, t) => sum + t, 0);
    const speedup_factor = sequential_time / total_time;

    return {
      query,
      agents: selected_domains,
      responses,
      execution_times_ms: execution_times,
      success_count,
      failure_count,
      total_time_ms: total_time,
      speedup_factor,
    };
  }

  /**
   * Execute with early collapse if one path is highly confident
   */
  async executeWithEarlyCollapse(
    query: string,
    agents: Map<string, SpecializedAgent>,
    domains: string[],
    state: RecursionState
  ): Promise<ParallelExecution> {
    const selected_domains = domains.slice(0, this.maxParallelAgents);
    const start = Date.now();

    const responses = new Map<string, AgentResponse>();
    const execution_times = new Map<string, number>();
    let success_count = 0;
    let failure_count = 0;

    // Create abort controller for early termination
    const abortController = new AbortController();

    // Execute agents with collapse detection
    const executions = selected_domains.map(async (domain) => {
      const agent = agents.get(domain);
      if (!agent) return null;

      const agent_start = Date.now();

      try {
        const response = await agent.process(query, state);
        const agent_time = Date.now() - agent_start;

        execution_times.set(domain, agent_time);
        responses.set(domain, response);
        success_count++;

        // Check for early collapse
        if (response.confidence >= this.collapseThreshold) {
          console.log(
            `[ParallelExecution] Early collapse triggered by ${domain} (confidence: ${response.confidence})`
          );
          abortController.abort(); // Signal other executions to stop
        }

        return { domain, response, success: true, execution_time_ms: agent_time };
      } catch (error) {
        failure_count++;
        return { domain, response: null, success: false, execution_time_ms: Date.now() - agent_start };
      }
    });

    // Wait for all executions (some may abort early)
    await Promise.allSettled(executions);

    const total_time = Date.now() - start;
    const sequential_time = Array.from(execution_times.values()).reduce((sum, t) => sum + t, 0);
    const speedup_factor = sequential_time / total_time;

    return {
      query,
      agents: selected_domains,
      responses,
      execution_times_ms: execution_times,
      success_count,
      failure_count,
      total_time_ms: total_time,
      speedup_factor,
    };
  }

  /**
   * Collapse multiple agent responses into single coherent answer
   */
  async collapse(execution: ParallelExecution): Promise<CollapsedResult> {
    const responses = Array.from(execution.responses.entries());

    if (responses.length === 0) {
      throw new Error('Cannot collapse: no successful agent responses');
    }

    // If only one response, return it directly
    if (responses.length === 1) {
      const [agent_id, response] = responses[0];
      return {
        final_answer: response.answer,
        contributing_agents: [agent_id],
        confidence: response.confidence,
        reasoning_synthesis: response.reasoning,
        parallel_execution: execution,
      };
    }

    // Multiple responses: synthesize
    const sorted_by_confidence = responses.sort((a, b) => b[1].confidence - a[1].confidence);

    // Weight responses by confidence
    const total_confidence = sorted_by_confidence.reduce((sum, [, r]) => sum + r.confidence, 0);
    const avg_confidence = total_confidence / sorted_by_confidence.length;

    // Synthesize reasoning
    const reasoning_parts = sorted_by_confidence.map(
      ([agent_id, response]) =>
        `[${agent_id}, confidence: ${response.confidence}]: ${response.reasoning}`
    );

    // Take highest confidence answer as base, enrich with insights from others
    const [primary_agent, primary_response] = sorted_by_confidence[0];
    const contributing_agents = sorted_by_confidence.map(([id]) => id);

    return {
      final_answer: primary_response.answer,
      contributing_agents,
      confidence: avg_confidence,
      reasoning_synthesis: reasoning_parts.join('\n\n'),
      parallel_execution: execution,
    };
  }

  /**
   * Calculate entropy across multiple paths (measure of uncertainty)
   */
  calculateSuperpositionEntropy(responses: Map<string, AgentResponse>): number {
    if (responses.size === 0) return 1.0; // Maximum uncertainty

    // Collect all unique concepts across responses
    const concept_frequencies = new Map<string, number>();
    let total_concepts = 0;

    for (const response of responses.values()) {
      for (const concept of response.concepts) {
        concept_frequencies.set(concept, (concept_frequencies.get(concept) || 0) + 1);
        total_concepts++;
      }
    }

    // Calculate Shannon entropy
    let entropy = 0;
    for (const freq of concept_frequencies.values()) {
      const prob = freq / total_concepts;
      entropy -= prob * Math.log2(prob);
    }

    // Normalize to 0-1
    const max_entropy = Math.log2(concept_frequencies.size);
    return max_entropy > 0 ? entropy / max_entropy : 0;
  }

  /**
   * Get metrics about parallel execution efficiency
   */
  getEfficiencyMetrics(execution: ParallelExecution): {
    speedup_factor: number;
    parallel_efficiency: number; // 0-1, 1 = perfect scaling
    load_balance: number; // 0-1, 1 = perfectly balanced
    cost_reduction: number; // % reduction vs sequential
  } {
    const execution_times = Array.from(execution.execution_times_ms.values());
    const max_time = Math.max(...execution_times);
    const total_time = execution_times.reduce((sum, t) => sum + t, 0);

    // Parallel efficiency: actual speedup / ideal speedup
    const ideal_speedup = execution_times.length;
    const actual_speedup = execution.speedup_factor;
    const parallel_efficiency = actual_speedup / ideal_speedup;

    // Load balance: 1 - (variance / mean)
    const mean_time = total_time / execution_times.length;
    const variance =
      execution_times.reduce((sum, t) => sum + Math.pow(t - mean_time, 2), 0) /
      execution_times.length;
    const load_balance = Math.max(0, 1 - Math.sqrt(variance) / mean_time);

    // Cost reduction: assuming cost proportional to time
    const cost_reduction = ((total_time - max_time) / total_time) * 100;

    return {
      speedup_factor: execution.speedup_factor,
      parallel_efficiency,
      load_balance,
      cost_reduction,
    };
  }
}
