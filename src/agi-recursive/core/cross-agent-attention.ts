/**
 * Cross-Agent Attention Mechanism
 *
 * Implements multi-head attention between specialized agents, allowing
 * agents to selectively attend to outputs from other agents during processing.
 *
 * Instead of linear agent composition (A → B → C → Meta), this enables
 * parallel collaborative processing where agents influence each other:
 *
 *     Finance ←→ Biology ←→ Systems
 *         ↘      ↓      ↙
 *           MetaAgent
 *
 * Key Features:
 * - Multi-head attention with configurable heads
 * - Learned attention weights per agent pair
 * - Query-Key-Value attention mechanism
 * - Attention weight visualization
 * - Cross-domain concept blending
 *
 * Based on "Attention Is All You Need" (Vaswani et al., 2017) but adapted
 * for agent collaboration rather than token sequences.
 */

export interface AgentOutput {
  agent_id: string;
  domain: string;
  answer: string;
  concepts: string[];
  confidence: number;
  reasoning: string;
  embedding?: number[]; // Optional vector representation
}

export interface AttentionHead {
  head_id: number;
  query_transform: number[][]; // Q projection matrix
  key_transform: number[][]; // K projection matrix
  value_transform: number[][]; // V projection matrix
}

export interface AttentionWeights {
  from_agent: string;
  to_agent: string;
  weight: number;
  head_contributions: number[]; // Weight per head
}

export interface AttendedOutput {
  agent_id: string;
  original_output: AgentOutput;
  attended_concepts: string[]; // Concepts influenced by other agents
  attention_weights: AttentionWeights[];
  blended_confidence: number;
  cross_domain_insights: string[];
}

export interface CrossAgentAttentionConfig {
  num_heads: number; // Number of attention heads
  embedding_dim: number; // Dimension of agent output embeddings
  head_dim: number; // Dimension per head
  dropout: number; // Dropout rate for regularization
  temperature: number; // Softmax temperature for attention
  enable_self_attention: boolean; // Whether agents can attend to themselves
  learn_weights: boolean; // Enable weight learning from history
}

/**
 * Multi-Head Cross-Agent Attention
 *
 * Allows agents to attend to outputs from other agents, creating
 * collaborative processing instead of sequential composition.
 */
export class CrossAgentAttention {
  private config: CrossAgentAttentionConfig;
  private attention_heads: AttentionHead[];
  private attention_history: Map<string, AttentionWeights[]>;
  private weight_statistics: Map<string, { sum: number; count: number }>;

  constructor(config: Partial<CrossAgentAttentionConfig> = {}) {
    this.config = {
      num_heads: config.num_heads ?? 4,
      embedding_dim: config.embedding_dim ?? 512,
      head_dim: config.head_dim ?? 64,
      dropout: config.dropout ?? 0.1,
      temperature: config.temperature ?? 1.0,
      enable_self_attention: config.enable_self_attention ?? false,
      learn_weights: config.learn_weights ?? true,
    };

    this.attention_heads = [];
    this.attention_history = new Map();
    this.weight_statistics = new Map();

    this.initializeHeads();
  }

  /**
   * Initialize attention heads with random projection matrices
   */
  private initializeHeads(): void {
    for (let i = 0; i < this.config.num_heads; i++) {
      this.attention_heads.push({
        head_id: i,
        query_transform: this.randomMatrix(this.config.head_dim, this.config.embedding_dim),
        key_transform: this.randomMatrix(this.config.head_dim, this.config.embedding_dim),
        value_transform: this.randomMatrix(this.config.head_dim, this.config.embedding_dim),
      });
    }
  }

  /**
   * Generate random projection matrix (Xavier initialization)
   */
  private randomMatrix(rows: number, cols: number): number[][] {
    const matrix: number[][] = [];
    const scale = Math.sqrt(2.0 / (rows + cols));

    for (let i = 0; i < rows; i++) {
      const row: number[] = [];
      for (let j = 0; j < cols; j++) {
        row.push((Math.random() * 2 - 1) * scale);
      }
      matrix.push(row);
    }

    return matrix;
  }

  /**
   * Apply multi-head attention to agent outputs
   *
   * Each agent attends to all other agents' outputs, weighted by
   * learned attention scores.
   */
  applyAttention(agent_outputs: AgentOutput[]): AttendedOutput[] {
    if (agent_outputs.length === 0) {
      return [];
    }

    // Create embeddings for each agent output
    const embeddings = agent_outputs.map((output) => this.createEmbedding(output));

    // Calculate attention for each agent
    const attended_outputs: AttendedOutput[] = [];

    for (let i = 0; i < agent_outputs.length; i++) {
      const query_agent = agent_outputs[i];
      const query_embedding = embeddings[i];

      // Calculate attention weights to all other agents
      const attention_weights: AttentionWeights[] = [];

      for (let j = 0; j < agent_outputs.length; j++) {
        // Skip self-attention if disabled
        if (!this.config.enable_self_attention && i === j) {
          continue;
        }

        const key_agent = agent_outputs[j];
        const key_embedding = embeddings[j];

        const weight = this.calculateAttentionWeight(query_embedding, key_embedding, query_agent, key_agent);

        attention_weights.push({
          from_agent: query_agent.agent_id,
          to_agent: key_agent.agent_id,
          weight: weight.total_weight,
          head_contributions: weight.head_weights,
        });
      }

      // Normalize attention weights
      const normalized_weights = this.normalizeWeights(attention_weights);

      // Blend outputs based on attention
      const attended = this.blendOutputs(query_agent, agent_outputs, normalized_weights);

      attended_outputs.push(attended);

      // Store attention history for learning
      if (this.config.learn_weights) {
        this.attention_history.set(query_agent.agent_id, normalized_weights);
        this.updateWeightStatistics(normalized_weights);
      }
    }

    return attended_outputs;
  }

  /**
   * Create embedding vector from agent output
   *
   * Uses concepts, confidence, and domain as features.
   * In production, could use pre-trained embeddings.
   */
  private createEmbedding(output: AgentOutput): number[] {
    // Use provided embedding if available
    if (output.embedding && output.embedding.length === this.config.embedding_dim) {
      return output.embedding;
    }

    // Simple embedding: hash concepts and encode domain
    const embedding = new Array(this.config.embedding_dim).fill(0);

    // Encode concepts
    output.concepts.forEach((concept, idx) => {
      const hash = this.hashString(concept);
      embedding[hash % this.config.embedding_dim] += 1.0;
    });

    // Encode domain
    const domain_hash = this.hashString(output.domain);
    embedding[domain_hash % this.config.embedding_dim] += 2.0;

    // Encode confidence
    embedding[0] = output.confidence;

    // Normalize
    const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    return embedding.map((val) => val / (norm || 1));
  }

  /**
   * Simple string hash function
   */
  private hashString(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = (hash << 5) - hash + str.charCodeAt(i);
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  }

  /**
   * Calculate attention weight using multi-head mechanism
   */
  private calculateAttentionWeight(
    query_embedding: number[],
    key_embedding: number[],
    query_agent: AgentOutput,
    key_agent: AgentOutput
  ): { total_weight: number; head_weights: number[] } {
    const head_weights: number[] = [];

    for (const head of this.attention_heads) {
      // Project query and key
      const query_proj = this.matmul(head.query_transform, query_embedding);
      const key_proj = this.matmul(head.key_transform, key_embedding);

      // Calculate scaled dot-product attention
      const dot_product = this.dotProduct(query_proj, key_proj);
      const scale = Math.sqrt(this.config.head_dim);
      const scaled_score = dot_product / scale;

      head_weights.push(scaled_score);
    }

    // Apply temperature scaling
    const temp_scaled = head_weights.map((w) => w / this.config.temperature);

    // Softmax per head (but we'll aggregate later)
    const head_contributions = temp_scaled.map((w) => Math.exp(w));

    // If learning enabled, bias towards historically useful connections
    let total_weight = head_contributions.reduce((sum, w) => sum + w, 0) / this.config.num_heads;

    if (this.config.learn_weights) {
      const stat_key = `${query_agent.agent_id}→${key_agent.agent_id}`;
      const stats = this.weight_statistics.get(stat_key);
      if (stats && stats.count > 0) {
        const historical_avg = stats.sum / stats.count;
        total_weight = 0.7 * total_weight + 0.3 * historical_avg; // Blend current and historical
      }
    }

    return { total_weight, head_weights: head_contributions };
  }

  /**
   * Matrix-vector multiplication
   */
  private matmul(matrix: number[][], vector: number[]): number[] {
    return matrix.map((row) => this.dotProduct(row, vector));
  }

  /**
   * Dot product of two vectors
   */
  private dotProduct(a: number[], b: number[]): number {
    return a.reduce((sum, val, idx) => sum + val * b[idx], 0);
  }

  /**
   * Normalize attention weights using softmax
   */
  private normalizeWeights(weights: AttentionWeights[]): AttentionWeights[] {
    if (weights.length === 0) return [];

    const raw_weights = weights.map((w) => w.weight);
    const max_weight = Math.max(...raw_weights);

    // Softmax
    const exp_weights = raw_weights.map((w) => Math.exp(w - max_weight));
    const sum_exp = exp_weights.reduce((sum, w) => sum + w, 0);

    return weights.map((w, idx) => ({
      ...w,
      weight: exp_weights[idx] / sum_exp,
    }));
  }

  /**
   * Blend agent outputs based on attention weights
   */
  private blendOutputs(
    query_agent: AgentOutput,
    all_agents: AgentOutput[],
    attention_weights: AttentionWeights[]
  ): AttendedOutput {
    // Start with original concepts
    const attended_concepts = new Set<string>(query_agent.concepts);

    // Blend concepts from other agents based on attention
    for (const weight of attention_weights) {
      if (weight.from_agent === query_agent.agent_id && weight.weight > 0.1) {
        const source_agent = all_agents.find((a) => a.agent_id === weight.to_agent);
        if (source_agent) {
          // Add highly weighted concepts
          source_agent.concepts.forEach((concept) => {
            if (weight.weight > 0.2) {
              attended_concepts.add(concept);
            }
          });
        }
      }
    }

    // Calculate blended confidence
    let blended_confidence = query_agent.confidence;
    for (const weight of attention_weights) {
      const source_agent = all_agents.find((a) => a.agent_id === weight.to_agent);
      if (source_agent) {
        blended_confidence += weight.weight * source_agent.confidence * 0.3;
      }
    }
    blended_confidence = Math.min(1.0, blended_confidence);

    // Generate cross-domain insights
    const cross_domain_insights = this.generateInsights(query_agent, all_agents, attention_weights);

    return {
      agent_id: query_agent.agent_id,
      original_output: query_agent,
      attended_concepts: Array.from(attended_concepts),
      attention_weights,
      blended_confidence,
      cross_domain_insights,
    };
  }

  /**
   * Generate cross-domain insights from attention patterns
   */
  private generateInsights(
    query_agent: AgentOutput,
    all_agents: AgentOutput[],
    attention_weights: AttentionWeights[]
  ): string[] {
    const insights: string[] = [];

    // Find high-attention connections
    const high_attention = attention_weights.filter((w) => w.weight > 0.3);

    for (const weight of high_attention) {
      const source_agent = all_agents.find((a) => a.agent_id === weight.to_agent);
      if (source_agent && source_agent.domain !== query_agent.domain) {
        insights.push(
          `${query_agent.domain} ←→ ${source_agent.domain}: Cross-domain attention (${(weight.weight * 100).toFixed(1)}%)`
        );

        // Find overlapping concepts
        const overlap = query_agent.concepts.filter((c) => source_agent.concepts.includes(c));
        if (overlap.length > 0) {
          insights.push(`Shared concepts: ${overlap.join(', ')}`);
        }
      }
    }

    return insights;
  }

  /**
   * Update weight statistics for learning
   */
  private updateWeightStatistics(weights: AttentionWeights[]): void {
    for (const weight of weights) {
      const key = `${weight.from_agent}→${weight.to_agent}`;
      const stats = this.weight_statistics.get(key) || { sum: 0, count: 0 };
      stats.sum += weight.weight;
      stats.count++;
      this.weight_statistics.set(key, stats);
    }
  }

  /**
   * Get attention history for a specific agent
   */
  getAttentionHistory(agent_id: string): AttentionWeights[] {
    return this.attention_history.get(agent_id) || [];
  }

  /**
   * Get learned weight statistics
   */
  getWeightStatistics(): Map<string, { avg: number; count: number }> {
    const stats = new Map<string, { avg: number; count: number }>();

    for (const [key, stat] of this.weight_statistics.entries()) {
      stats.set(key, {
        avg: stat.sum / stat.count,
        count: stat.count,
      });
    }

    return stats;
  }

  /**
   * Visualize attention weights as matrix
   */
  visualizeAttention(attended_outputs: AttendedOutput[]): string {
    const agents = attended_outputs.map((a) => a.agent_id);
    const matrix: number[][] = [];

    // Build attention matrix
    for (const output of attended_outputs) {
      const row = new Array(agents.length).fill(0);
      for (const weight of output.attention_weights) {
        const target_idx = agents.indexOf(weight.to_agent);
        if (target_idx >= 0) {
          row[target_idx] = weight.weight;
        }
      }
      matrix.push(row);
    }

    // Format as ASCII
    let viz = '\nAttention Matrix:\n';
    viz += '     ' + agents.map((a) => a.substring(0, 5).padEnd(5)).join(' ') + '\n';

    for (let i = 0; i < matrix.length; i++) {
      viz += agents[i].substring(0, 5).padEnd(5);
      for (let j = 0; j < matrix[i].length; j++) {
        const val = matrix[i][j];
        const bar = '█'.repeat(Math.round(val * 5));
        viz += bar.padEnd(5) + ' ';
      }
      viz += '\n';
    }

    return viz;
  }

  /**
   * Export attention configuration
   */
  exportConfig(): CrossAgentAttentionConfig {
    return { ...this.config };
  }

  /**
   * Export learned weights for persistence
   */
  exportWeights(): string {
    const data = {
      config: this.config,
      statistics: Array.from(this.weight_statistics.entries()).map(([key, stat]) => ({
        connection: key,
        avg_weight: stat.sum / stat.count,
        count: stat.count,
      })),
      attention_heads: this.attention_heads.map((head) => ({
        head_id: head.head_id,
        // Note: In production, would serialize full matrices
        // For now, just store metadata
      })),
    };

    return JSON.stringify(data, null, 2);
  }

  /**
   * Clear attention history and statistics
   */
  clear(): void {
    this.attention_history.clear();
    this.weight_statistics.clear();
  }
}

/**
 * Create cross-agent attention with default configuration
 */
export function createCrossAgentAttention(
  config?: Partial<CrossAgentAttentionConfig>
): CrossAgentAttention {
  return new CrossAgentAttention(config);
}

/**
 * Utility: Extract attention matrix from attended outputs
 */
export function extractAttentionMatrix(attended_outputs: AttendedOutput[]): number[][] {
  const agents = attended_outputs.map((a) => a.agent_id);
  const matrix: number[][] = [];

  for (const output of attended_outputs) {
    const row = new Array(agents.length).fill(0);
    for (const weight of output.attention_weights) {
      const target_idx = agents.indexOf(weight.to_agent);
      if (target_idx >= 0) {
        row[target_idx] = weight.weight;
      }
    }
    matrix.push(row);
  }

  return matrix;
}
