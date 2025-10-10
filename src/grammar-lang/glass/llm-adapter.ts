/**
 * LLM Adapter for .glass Organisms
 *
 * Wraps /src/agi-recursive/llm/anthropic-adapter.ts for use in .glass organisms.
 * Provides:
 * - Automatic constitutional validation
 * - Cost tracking per organism
 * - Model selection (Opus vs Sonnet)
 * - Streaming support
 * - Task-specific prompting
 *
 * Usage:
 * ```typescript
 * const llm = createGlassLLM('cognitive');
 * const result = await llm.invoke('Analyze this text...', { task: 'reasoning' });
 * console.log(result.text);
 * console.log(llm.getTotalCost());
 * ```
 */

import {
  AnthropicAdapter,
  createAdapter,
  getRecommendedModel,
  ClaudeModel,
  LLMResponse,
  LLMConfig
} from '../../agi-recursive/llm/anthropic-adapter';

import {
  ConstitutionalAdapter,
  createConstitutionalAdapter,
  ConstitutionDomain,
  CostBudgetTracker
} from './constitutional-adapter';

// ============================================================================
// Types
// ============================================================================

export type GlassTask =
  | 'code-synthesis'     // Generate .gl code (ROXO)
  | 'pattern-detection'  // Detect semantic patterns (ROXO)
  | 'intent-analysis'    // Analyze intent/pragmatics (CINZA)
  | 'semantic-analysis'  // Deep semantic understanding (CINZA)
  | 'sentiment-analysis' // Emotional state analysis (VERMELHO)
  | 'reasoning'          // General reasoning (default)
  | 'creative'           // Creative tasks
  | 'fast';              // Fast/cheap tasks

export interface GlassLLMConfig extends Partial<LLMConfig> {
  task?: GlassTask;
  enable_constitutional?: boolean;
  max_cost_usd?: number;
}

export interface GlassLLMResponse extends LLMResponse {
  constitutional_check?: any;
  cost_budget_remaining?: number;
}

// ============================================================================
// Glass LLM Adapter
// ============================================================================

export class GlassLLM {
  private llm: AnthropicAdapter;
  private constitutional: ConstitutionalAdapter;
  private costTracker: CostBudgetTracker;
  private domain: ConstitutionDomain;

  constructor(
    domain: ConstitutionDomain = 'universal',
    maxBudget: number = 1.0,
    apiKey?: string
  ) {
    this.llm = createAdapter(apiKey);
    this.constitutional = createConstitutionalAdapter(domain);
    this.costTracker = new CostBudgetTracker(maxBudget);
    this.domain = domain;
  }

  /**
   * Invoke LLM with constitutional validation
   */
  async invoke(
    query: string,
    config: GlassLLMConfig = {}
  ): Promise<GlassLLMResponse> {
    // Select model based on task
    const model = this.selectModel(config.task, config.model);

    // Build system prompt
    const systemPrompt = this.buildSystemPrompt(config.task);

    // Estimate cost
    const estimate = this.llm.estimateCost(systemPrompt, query, model);

    // Check budget
    if (this.costTracker.wouldExceedBudget(estimate.estimated_cost)) {
      throw new Error(
        `Operation would exceed budget. Remaining: $${this.costTracker.getRemainingBudget().toFixed(4)}, Required: $${estimate.estimated_cost.toFixed(4)}`
      );
    }

    // Invoke LLM
    const response = await this.llm.invoke(systemPrompt, query, {
      model,
      max_tokens: config.max_tokens,
      temperature: config.temperature ?? this.getTaskTemperature(config.task),
      top_p: config.top_p,
      top_k: config.top_k
    });

    // Track cost
    this.costTracker.addCost(response.usage.cost_usd);

    // Constitutional validation (if enabled)
    let constitutionalCheck = undefined;
    if (config.enable_constitutional !== false) {
      constitutionalCheck = this.constitutional.validate(
        {
          answer: response.text,
          reasoning: `LLM ${config.task || 'reasoning'} task`,
          confidence: this.estimateConfidence(response)
        },
        {
          depth: 0,
          invocation_count: 1,
          cost_so_far: this.costTracker.getTotalCost(),
          previous_agents: []
        }
      );

      // Log if violations
      if (!constitutionalCheck.passed) {
        console.warn('⚠️  LLM response has constitutional violations:');
        console.warn(this.constitutional.formatReport(constitutionalCheck));
      }
    }

    return {
      ...response,
      constitutional_check: constitutionalCheck,
      cost_budget_remaining: this.costTracker.getRemainingBudget()
    };
  }

  /**
   * Invoke with streaming
   */
  async *invokeStream(
    query: string,
    config: GlassLLMConfig = {}
  ): AsyncGenerator<string, GlassLLMResponse, undefined> {
    const model = this.selectModel(config.task, config.model);
    const systemPrompt = this.buildSystemPrompt(config.task);

    // Check budget (estimate)
    const estimate = this.llm.estimateCost(systemPrompt, query, model);
    if (this.costTracker.wouldExceedBudget(estimate.estimated_cost)) {
      throw new Error(`Operation would exceed budget`);
    }

    const stream = this.llm.invokeStream(systemPrompt, query, {
      model,
      max_tokens: config.max_tokens,
      temperature: config.temperature ?? this.getTaskTemperature(config.task)
    });

    let fullText = '';

    for await (const chunk of stream) {
      fullText += chunk;
      yield chunk;
    }

    // TODO: Get final usage from stream
    // For now, estimate usage
    const estimatedUsage = {
      input_tokens: Math.ceil(query.length / 4),
      output_tokens: Math.ceil(fullText.length / 4),
      cost_usd: estimate.estimated_cost
    };

    // Track cost
    this.costTracker.addCost(estimatedUsage.cost_usd);

    // Constitutional validation
    let constitutionalCheck = undefined;
    if (config.enable_constitutional !== false) {
      constitutionalCheck = this.constitutional.validate(
        { answer: fullText },
        { depth: 0, invocation_count: 1, cost_so_far: this.costTracker.getTotalCost(), previous_agents: [] }
      );
    }

    return {
      text: fullText,
      usage: estimatedUsage,
      model: model,
      stop_reason: null,
      constitutional_check: constitutionalCheck,
      cost_budget_remaining: this.costTracker.getRemainingBudget()
    };
  }

  /**
   * Select model based on task
   */
  private selectModel(task?: GlassTask, explicitModel?: ClaudeModel): ClaudeModel {
    if (explicitModel) {
      return explicitModel;
    }

    switch (task) {
      case 'code-synthesis':
        return 'claude-opus-4'; // Code generation needs best reasoning
      case 'pattern-detection':
        return 'claude-sonnet-4-5'; // Pattern detection is fast
      case 'intent-analysis':
        return 'claude-opus-4'; // Intent needs deep understanding
      case 'semantic-analysis':
        return 'claude-opus-4'; // Semantics needs best model
      case 'sentiment-analysis':
        return 'claude-sonnet-4-5'; // Sentiment is straightforward
      case 'reasoning':
        return 'claude-opus-4';
      case 'creative':
        return 'claude-opus-4';
      case 'fast':
        return 'claude-sonnet-4-5';
      default:
        return 'claude-sonnet-4-5'; // Default to cost-effective
    }
  }

  /**
   * Build system prompt based on task
   */
  private buildSystemPrompt(task?: GlassTask): string {
    const basePrompt = `You are an expert AI assistant integrated into the Chomsky .glass organism system.`;

    const taskPrompts: Record<GlassTask, string> = {
      'code-synthesis': `${basePrompt}

Your task: Synthesize .gl (Grammar Language) code from patterns.

Requirements:
- Generate valid .gl syntax
- Follow functional programming principles
- Include type safety
- Optimize for O(1) performance where possible
- Include comments explaining logic

Format your response as:
\`\`\`gl
// .gl code here
\`\`\``,

      'pattern-detection': `${basePrompt}

Your task: Detect semantic patterns in code/text.

Requirements:
- Identify recurring patterns
- Explain pattern significance
- Suggest pattern name/category
- Estimate confidence (0-1)

Format your response as JSON:
{
  "patterns": [
    {"name": "...", "confidence": 0.9, "explanation": "..."}
  ]
}`,

      'intent-analysis': `${basePrompt}

Your task: Analyze communicative intent using pragmatics.

Requirements:
- Detect primary intent (manipulate, control, confuse, etc.)
- Identify secondary intents
- Explain reasoning chain
- Consider context (relationship, power dynamics)
- Provide confidence score

Format your response as JSON:
{
  "primary_intent": "...",
  "secondary_intents": ["..."],
  "confidence": 0.85,
  "reasoning": ["step 1", "step 2"]
}`,

      'semantic-analysis': `${basePrompt}

Your task: Deep semantic analysis of text.

Requirements:
- Analyze meaning beyond surface structure
- Identify semantic relations
- Detect implicit meanings
- Explain ambiguities
- Provide linguistic evidence

Format your response as JSON:
{
  "meaning": "...",
  "relations": ["..."],
  "implicit_meanings": ["..."],
  "evidence": ["..."]
}`,

      'sentiment-analysis': `${basePrompt}

Your task: Analyze emotional state and sentiment.

Requirements:
- Detect primary emotion (anger, fear, joy, sadness, etc.)
- Identify emotional intensity (0-1)
- Detect emotional transitions
- Explain reasoning

Format your response as JSON:
{
  "primary_emotion": "...",
  "intensity": 0.7,
  "secondary_emotions": ["..."],
  "reasoning": "..."
}`,

      'reasoning': `${basePrompt}

Your task: General reasoning and problem-solving.

Requirements:
- Think step-by-step
- Show your reasoning
- Provide clear explanations
- Cite sources when possible`,

      'creative': `${basePrompt}

Your task: Creative generation.

Requirements:
- Be creative and original
- Maintain coherence
- Follow any constraints given`,

      'fast': `${basePrompt}

Your task: Quick, efficient responses.

Requirements:
- Be concise
- Focus on key points
- Maintain accuracy`
    };

    return taskPrompts[task as GlassTask] || taskPrompts['reasoning'];
  }

  /**
   * Get recommended temperature for task
   */
  private getTaskTemperature(task?: GlassTask): number {
    switch (task) {
      case 'code-synthesis':
        return 0.3; // Low temperature for precise code
      case 'pattern-detection':
        return 0.5; // Balanced
      case 'intent-analysis':
        return 0.4; // Low-medium for accuracy
      case 'semantic-analysis':
        return 0.4; // Low-medium for accuracy
      case 'sentiment-analysis':
        return 0.5; // Balanced
      case 'reasoning':
        return 0.5;
      case 'creative':
        return 0.8; // Higher for creativity
      case 'fast':
        return 0.5;
      default:
        return 0.5;
    }
  }

  /**
   * Estimate confidence from LLM response
   */
  private estimateConfidence(response: LLMResponse): number {
    // Heuristic: longer responses with definitive language = higher confidence
    const text = response.text.toLowerCase();

    let confidence = 0.7; // Base confidence

    // Increase for hedging language
    if (text.includes('uncertain') || text.includes('not sure') || text.includes('maybe')) {
      confidence -= 0.2;
    }

    // Increase for definitive language
    if (text.includes('clearly') || text.includes('definitely') || text.includes('certainly')) {
      confidence += 0.1;
    }

    return Math.max(0, Math.min(1, confidence));
  }

  /**
   * Get total cost for this organism
   */
  getTotalCost(): number {
    return this.costTracker.getTotalCost();
  }

  /**
   * Get remaining budget
   */
  getRemainingBudget(): number {
    return this.costTracker.getRemainingBudget();
  }

  /**
   * Check if over budget
   */
  isOverBudget(): boolean {
    return this.costTracker.isOverBudget();
  }

  /**
   * Get cost tracking stats
   */
  getCostStats() {
    return {
      total_cost: this.costTracker.getTotalCost(),
      max_budget: this.costTracker.getMaxBudget(),
      remaining_budget: this.costTracker.getRemainingBudget(),
      over_budget: this.costTracker.isOverBudget()
    };
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create GlassLLM instance for specific domain
 */
export function createGlassLLM(
  domain: ConstitutionDomain = 'universal',
  maxBudget: number = 1.0,
  apiKey?: string
): GlassLLM {
  return new GlassLLM(domain, maxBudget, apiKey);
}

/**
 * Create LLM instance with auto-detection from organism type
 */
export function createLLMForOrganism(
  organismType: string,
  maxBudget: number = 1.0
): GlassLLM {
  // Map organism types to constitutional domains
  const domainMap: Record<string, ConstitutionDomain> = {
    'cognitive-defense-organism': 'cognitive',
    'security-organism': 'security',
    'glass-core-organism': 'glass-core',
    'vcs-organism': 'vcs',
    'database-organism': 'database'
  };

  const domain = domainMap[organismType] || 'universal';
  return createGlassLLM(domain, maxBudget);
}
