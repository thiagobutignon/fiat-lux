/**
 * Cognitive Load Balancer
 *
 * Distributes complexity across agents automatically based on:
 * - Current agent load
 * - Task complexity estimation
 * - Historical response times
 * - Available resources
 *
 * This is a BREAKTHROUGH innovation - few AGI systems implement
 * automatic cognitive load distribution.
 */

// ============================================================================
// Types
// ============================================================================

export interface CognitiveLoad {
  agent_id: string;
  current_tasks: number;
  average_response_time_ms: number;
  context_size_tokens: number;
  complexity_score: number; // 0-1, higher = more loaded
  available_capacity: number; // 0-1, higher = more available
  last_updated: number;
}

export interface TaskComplexity {
  query: string;
  estimated_tokens: number;
  estimated_time_ms: number;
  required_knowledge_depth: number; // 0-1
  interdomain_dependencies: number;
  complexity_score: number; // 0-1
}

export interface AgentAssignment {
  agent_id: string;
  subtask: string;
  estimated_load: number;
  priority: number; // 0-1, higher = more urgent
  rationale: string;
}

export interface LoadBalancingMetrics {
  total_agents: number;
  average_load: number;
  max_load: number;
  min_load: number;
  load_variance: number;
  balance_score: number; // 0-1, higher = better balanced
  rebalancing_count: number;
}

// ============================================================================
// Cognitive Load Balancer
// ============================================================================

export class CognitiveLoadBalancer {
  private agentLoads: Map<string, CognitiveLoad> = new Map();
  private taskHistory: TaskComplexity[] = [];
  private assignmentHistory: AgentAssignment[] = [];
  private metrics: LoadBalancingMetrics;

  constructor() {
    this.metrics = {
      total_agents: 0,
      average_load: 0,
      max_load: 0,
      min_load: 1,
      load_variance: 0,
      balance_score: 1,
      rebalancing_count: 0,
    };
  }

  /**
   * Register an agent and initialize its load tracking
   */
  registerAgent(agent_id: string): void {
    this.agentLoads.set(agent_id, {
      agent_id,
      current_tasks: 0,
      average_response_time_ms: 0,
      context_size_tokens: 0,
      complexity_score: 0,
      available_capacity: 1.0,
      last_updated: Date.now(),
    });

    this.metrics.total_agents++;
    this.updateMetrics();
  }

  /**
   * Estimate task complexity using multiple heuristics
   */
  async estimateComplexity(query: string, domains: string[]): Promise<TaskComplexity> {
    // Heuristic 1: Token estimation (simple word count * 1.3)
    const estimated_tokens = Math.ceil(query.split(/\s+/).length * 1.3);

    // Heuristic 2: Time estimation based on historical data
    const similar_tasks = this.taskHistory
      .filter((t) => this.calculateSimilarity(t.query, query) > 0.7)
      .slice(-5);

    const estimated_time_ms =
      similar_tasks.length > 0
        ? similar_tasks.reduce((sum, t) => sum + t.estimated_time_ms, 0) / similar_tasks.length
        : 2000; // Default 2 seconds

    // Heuristic 3: Knowledge depth (number of technical terms / total words)
    const technical_terms = this.countTechnicalTerms(query);
    const total_words = query.split(/\s+/).length;
    const required_knowledge_depth = Math.min(technical_terms / total_words, 1);

    // Heuristic 4: Interdomain dependencies
    const interdomain_dependencies = domains.length > 1 ? domains.length - 1 : 0;

    // Aggregate complexity score
    const complexity_score = Math.min(
      (estimated_tokens / 1000) * 0.3 +
        (estimated_time_ms / 10000) * 0.3 +
        required_knowledge_depth * 0.2 +
        (interdomain_dependencies / 5) * 0.2,
      1
    );

    const complexity: TaskComplexity = {
      query,
      estimated_tokens,
      estimated_time_ms,
      required_knowledge_depth,
      interdomain_dependencies,
      complexity_score,
    };

    this.taskHistory.push(complexity);
    return complexity;
  }

  /**
   * Distribute tasks across agents based on current load
   */
  async distribute(
    query: string,
    available_agents: string[],
    domains: string[]
  ): Promise<AgentAssignment[]> {
    // 1. Estimate task complexity
    const complexity = await this.estimateComplexity(query, domains);

    // 2. Get current loads
    const loads = available_agents.map((agent_id) => this.agentLoads.get(agent_id)!);

    // 3. Sort by available capacity (least loaded first)
    const sorted = loads.sort((a, b) => b.available_capacity - a.available_capacity);

    // 4. Assign tasks
    const assignments: AgentAssignment[] = [];

    for (let i = 0; i < domains.length; i++) {
      const domain = domains[i];
      const agent = sorted[i % sorted.length]; // Round-robin with capacity weighting

      // Calculate load impact
      const load_impact = complexity.complexity_score * 0.2; // Each task increases load by complexity * 20%
      const estimated_load = agent.complexity_score + load_impact;

      assignments.push({
        agent_id: agent.agent_id,
        subtask: `Process query in ${domain} domain`,
        estimated_load,
        priority: 1 - i / domains.length, // First domains have higher priority
        rationale: `Assigned to ${agent.agent_id} (capacity: ${(agent.available_capacity * 100).toFixed(1)}%, load: ${(agent.complexity_score * 100).toFixed(1)}%)`,
      });

      // Update load prediction
      agent.complexity_score = Math.min(estimated_load, 1);
      agent.available_capacity = 1 - agent.complexity_score;
    }

    this.assignmentHistory.push(...assignments);
    this.updateMetrics();

    return assignments;
  }

  /**
   * Update agent load after task completion
   */
  updateAgentLoad(
    agent_id: string,
    actual_time_ms: number,
    actual_tokens: number,
    task_completed: boolean
  ): void {
    const load = this.agentLoads.get(agent_id);
    if (!load) return;

    // Update current tasks
    load.current_tasks = Math.max(0, load.current_tasks - 1);

    // Update average response time (exponential moving average)
    if (load.average_response_time_ms === 0) {
      load.average_response_time_ms = actual_time_ms;
    } else {
      load.average_response_time_ms = load.average_response_time_ms * 0.7 + actual_time_ms * 0.3;
    }

    // Update context size
    load.context_size_tokens = actual_tokens;

    // Recalculate complexity score
    const time_factor = Math.min(load.average_response_time_ms / 10000, 1); // Normalize to 10s
    const context_factor = Math.min(load.context_size_tokens / 10000, 1); // Normalize to 10k tokens
    const task_factor = Math.min(load.current_tasks / 5, 1); // Normalize to 5 concurrent tasks

    load.complexity_score = time_factor * 0.4 + context_factor * 0.3 + task_factor * 0.3;
    load.available_capacity = 1 - load.complexity_score;
    load.last_updated = Date.now();

    this.updateMetrics();
  }

  /**
   * Check if rebalancing is needed
   */
  shouldRebalance(): boolean {
    // Rebalance if variance is high (agents have very different loads)
    return this.metrics.load_variance > 0.3;
  }

  /**
   * Get current load balancing metrics
   */
  getMetrics(): LoadBalancingMetrics {
    return { ...this.metrics };
  }

  /**
   * Get current agent loads
   */
  getAgentLoads(): Map<string, CognitiveLoad> {
    return new Map(this.agentLoads);
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  private calculateSimilarity(query1: string, query2: string): number {
    const words1 = new Set(query1.toLowerCase().split(/\s+/));
    const words2 = new Set(query2.toLowerCase().split(/\s+/));

    const intersection = new Set([...words1].filter((w) => words2.has(w)));
    const union = new Set([...words1, ...words2]);

    return intersection.size / union.size;
  }

  private countTechnicalTerms(query: string): number {
    const technical_patterns = [
      /\b(algorithm|architecture|system|model|agent|framework|protocol)\b/gi,
      /\b(budget|financial|economic|investment|portfolio)\b/gi,
      /\b(biology|cell|homeostasis|organism|feedback)\b/gi,
      /\b(optimization|efficiency|performance|scalability)\b/gi,
    ];

    let count = 0;
    for (const pattern of technical_patterns) {
      const matches = query.match(pattern);
      if (matches) count += matches.length;
    }

    return count;
  }

  private updateMetrics(): void {
    const loads = Array.from(this.agentLoads.values());

    if (loads.length === 0) {
      return;
    }

    // Calculate average load
    const total_load = loads.reduce((sum, l) => sum + l.complexity_score, 0);
    this.metrics.average_load = total_load / loads.length;

    // Calculate max/min
    this.metrics.max_load = Math.max(...loads.map((l) => l.complexity_score));
    this.metrics.min_load = Math.min(...loads.map((l) => l.complexity_score));

    // Calculate variance
    const variance_sum = loads.reduce(
      (sum, l) => sum + Math.pow(l.complexity_score - this.metrics.average_load, 2),
      0
    );
    this.metrics.load_variance = variance_sum / loads.length;

    // Calculate balance score (1 - variance, so higher = better balanced)
    this.metrics.balance_score = Math.max(0, 1 - this.metrics.load_variance);
  }

  /**
   * Export load history for analysis
   */
  exportHistory(): {
    agent_loads: CognitiveLoad[];
    task_history: TaskComplexity[];
    assignment_history: AgentAssignment[];
    metrics: LoadBalancingMetrics;
  } {
    return {
      agent_loads: Array.from(this.agentLoads.values()),
      task_history: [...this.taskHistory],
      assignment_history: [...this.assignmentHistory],
      metrics: { ...this.metrics },
    };
  }
}
