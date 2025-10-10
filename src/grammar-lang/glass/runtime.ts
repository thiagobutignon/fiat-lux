/**
 * Glass Runtime - Execute .glass Digital Organisms
 *
 * 🚀 THIS IS WHERE THE MAGIC HAPPENS 🚀
 *
 * Runtime capabilities:
 * 1. Load .glass organism
 * 2. Execute emerged functions
 * 3. Query knowledge base
 * 4. Track attention (which knowledge was used)
 * 5. Format results with confidence + sources
 * 6. Constitutional compliance validation
 *
 * Flow:
 * User Query → Parse Intent → Select Functions → Execute → Track Attention → Format Results
 */

import { GlassOrganism, GlassFunction } from './types';
import { loadGlassOrganism } from './builder';
import { createGlassLLM, GlassLLM } from './llm-adapter';
import { ConstitutionalAdapter, createConstitutionalAdapter } from './constitutional-adapter';

// ============================================================================
// Types
// ============================================================================

export interface QueryContext {
  query: string;
  user_id?: string;
  session_id?: string;
  context_history?: QueryResult[];
}

export interface AttentionWeight {
  knowledge_id: string;
  weight: number; // 0.0 to 1.0
  reason: string;
}

export interface QueryResult {
  query: string;
  answer: string;
  confidence: number; // 0.0 to 1.0
  functions_used: string[];
  attention_weights: AttentionWeight[];
  sources: string[];
  reasoning: string[];
  timestamp: string;
  cost_usd: number;
  constitutional_passed: boolean;
}

export interface ExecutionContext {
  organism: GlassOrganism;
  query: string;
  selected_functions: GlassFunction[];
  attention: Map<string, number>;
  reasoning: string[];
}

// ============================================================================
// Glass Runtime Engine
// ============================================================================

export class GlassRuntime {
  private organism: GlassOrganism;
  private llm: GlassLLM;
  private constitutional: ConstitutionalAdapter;
  private attentionMap: Map<string, number>;
  private totalCost: number;

  constructor(organism: GlassOrganism, maxBudget: number = 0.5) {
    this.organism = organism;
    this.attentionMap = new Map();
    this.totalCost = 0;

    // Create LLM with organism's domain
    const domain = this.mapSpecializationToDomain(organism.metadata.specialization);
    this.llm = createGlassLLM(domain, maxBudget);

    // Create constitutional adapter
    this.constitutional = createConstitutionalAdapter(domain);
  }

  /**
   * Execute query against organism
   */
  async query(context: QueryContext): Promise<QueryResult> {
    console.log(`\n🔍 Processing query: "${context.query}"`);
    console.log(`   Organism: ${this.organism.metadata.name} (${this.organism.metadata.specialization})`);
    console.log(`   Functions available: ${this.organism.code.functions.length}`);
    console.log('');

    const startTime = Date.now();
    const reasoning: string[] = [];

    try {
      // 1. Analyze query intent
      console.log('   🧠 Analyzing query intent...');
      const intent = await this.analyzeQueryIntent(context.query);
      reasoning.push(`Detected intent: ${intent.primary_intent}`);
      console.log(`      Intent: ${intent.primary_intent}`);

      // 2. Select relevant functions
      console.log('   🎯 Selecting relevant functions...');
      const selectedFunctions = await this.selectFunctions(context.query, intent);
      reasoning.push(`Selected ${selectedFunctions.length} function(s): ${selectedFunctions.map(f => f.name).join(', ')}`);
      console.log(`      Selected: ${selectedFunctions.map(f => f.name).join(', ')}`);

      // 3. Execute functions (simulated - would compile .gl code in production)
      console.log('   ⚙️  Executing functions...');
      const executionResults = await this.executeFunctions(selectedFunctions, context.query);
      reasoning.push(`Executed functions, retrieved knowledge from ${executionResults.knowledge_accessed.length} sources`);
      console.log(`      Knowledge accessed: ${executionResults.knowledge_accessed.length} sources`);

      // 4. Track attention
      console.log('   👁️  Tracking attention weights...');
      this.trackAttention(executionResults.knowledge_accessed);

      // 5. Synthesize answer
      console.log('   💬 Synthesizing answer...');
      const answer = await this.synthesizeAnswer(context.query, executionResults, selectedFunctions);
      reasoning.push(`Synthesized final answer with ${answer.confidence.toFixed(0)}% confidence`);
      console.log(`      Confidence: ${(answer.confidence * 100).toFixed(0)}%`);

      // 6. Constitutional validation
      console.log('   ⚖️  Validating constitutional compliance...');
      const constitutionalCheck = this.constitutional.validate(
        {
          answer: answer.text,
          confidence: answer.confidence,
          sources: answer.sources,
          reasoning: reasoning.join('; ')
        },
        {
          depth: 0,
          invocation_count: 1,
          cost_so_far: this.totalCost,
          previous_agents: []
        }
      );

      if (!constitutionalCheck.passed) {
        console.warn('   ⚠️  Constitutional violations detected:');
        console.warn(this.constitutional.formatReport(constitutionalCheck));
      } else {
        console.log('      ✅ Constitutional compliance verified');
      }

      // 7. Format result
      const duration = Date.now() - startTime;
      console.log(`   ✅ Query completed in ${duration}ms`);
      console.log('');

      const result: QueryResult = {
        query: context.query,
        answer: answer.text,
        confidence: answer.confidence,
        functions_used: selectedFunctions.map(f => f.name),
        attention_weights: this.getAttentionWeights(),
        sources: answer.sources,
        reasoning,
        timestamp: new Date().toISOString(),
        cost_usd: this.llm.getTotalCost(),
        constitutional_passed: constitutionalCheck.passed
      };

      // Update organism memory (episodic learning)
      this.updateMemory(result);

      return result;

    } catch (error: any) {
      console.error('   ❌ Query execution failed:', error.message);
      throw error;
    }
  }

  /**
   * Analyze query intent using LLM
   */
  private async analyzeQueryIntent(query: string): Promise<{
    primary_intent: string;
    secondary_intents: string[];
    confidence: number;
  }> {
    const response = await this.llm.invoke(
      `Analyze the intent of this query in the context of ${this.organism.metadata.specialization}:

Query: "${query}"

Available functions: ${this.organism.code.functions.map(f => f.name).join(', ')}

Return JSON:
\`\`\`json
{
  "primary_intent": "what user wants to know",
  "secondary_intents": ["additional aspects"],
  "confidence": 0.9
}
\`\`\``,
      {
        task: 'intent-analysis',
        max_tokens: 500,
        enable_constitutional: true
      }
    );

    try {
      const jsonMatch = response.text.match(/```(?:json)?\n([\s\S]*?)\n```/);
      if (jsonMatch) {
        return JSON.parse(jsonMatch[1]);
      }
    } catch (e) {
      // Fallback
    }

    return {
      primary_intent: 'information_request',
      secondary_intents: [],
      confidence: 0.5
    };
  }

  /**
   * Select relevant functions for query
   */
  private async selectFunctions(
    query: string,
    intent: any
  ): Promise<GlassFunction[]> {
    const response = await this.llm.invoke(
      `Select the most relevant functions to answer this query:

Query: "${query}"
Intent: ${intent.primary_intent}

Available functions:
${this.organism.code.functions.map(f => `- ${f.name}: ${f.signature}`).join('\n')}

Return JSON array of function names:
\`\`\`json
{
  "functions": ["function_name_1", "function_name_2"],
  "reasoning": "why these functions"
}
\`\`\``,
      {
        task: 'reasoning',
        max_tokens: 500,
        enable_constitutional: true
      }
    );

    try {
      const jsonMatch = response.text.match(/```(?:json)?\n([\s\S]*?)\n```/);
      if (jsonMatch) {
        const data = JSON.parse(jsonMatch[1]);
        const functionNames = data.functions || [];

        // Find selected functions in organism
        return this.organism.code.functions.filter(f =>
          functionNames.includes(f.name)
        );
      }
    } catch (e) {
      // Fallback: return all functions
    }

    return this.organism.code.functions;
  }

  /**
   * Execute functions (simulated - would compile .gl code in production)
   */
  private async executeFunctions(
    functions: GlassFunction[],
    query: string
  ): Promise<{
    knowledge_accessed: string[];
    results: any[];
  }> {
    // In production, this would:
    // 1. Compile .gl code to executable
    // 2. Execute in sandboxed environment
    // 3. Track which knowledge was accessed
    // 4. Return results

    // For now, simulate by querying knowledge base
    const knowledgeAccessed: string[] = [];
    const results: any[] = [];

    for (const fn of functions) {
      // Simulate function execution by querying knowledge related to patterns
      const patterns = fn.source_patterns;

      for (const pattern of patterns) {
        // Simulate knowledge retrieval
        const knowledgeIds = this.getKnowledgeForPattern(pattern);
        knowledgeAccessed.push(...knowledgeIds);

        results.push({
          function: fn.name,
          pattern,
          knowledge_count: knowledgeIds.length,
          confidence: fn.confidence
        });
      }
    }

    return {
      knowledge_accessed: [...new Set(knowledgeAccessed)], // Unique
      results
    };
  }

  /**
   * Get knowledge IDs related to pattern
   */
  private getKnowledgeForPattern(pattern: string): string[] {
    // Simulate knowledge retrieval
    // In production, would query actual knowledge base
    const count = this.organism.knowledge.patterns[pattern] || 0;
    const percentage = Math.min(count / 250, 1.0); // Assume 250 max

    // Generate simulated knowledge IDs
    const knowledgeCount = Math.floor(percentage * 10); // 0-10 knowledge sources
    return Array.from({ length: knowledgeCount }, (_, i) =>
      `${pattern}_knowledge_${i + 1}`
    );
  }

  /**
   * Track attention weights for knowledge accessed
   */
  private trackAttention(knowledgeIds: string[]): void {
    // Calculate attention weights based on frequency
    const total = knowledgeIds.length;

    for (const id of knowledgeIds) {
      const current = this.attentionMap.get(id) || 0;
      this.attentionMap.set(id, current + 1 / total);
    }
  }

  /**
   * Get attention weights sorted by importance
   */
  private getAttentionWeights(): AttentionWeight[] {
    const weights = Array.from(this.attentionMap.entries())
      .map(([id, weight]) => ({
        knowledge_id: id,
        weight,
        reason: `Used in query processing`
      }))
      .sort((a, b) => b.weight - a.weight);

    return weights.slice(0, 10); // Top 10
  }

  /**
   * Synthesize final answer using LLM
   */
  private async synthesizeAnswer(
    query: string,
    executionResults: any,
    functions: GlassFunction[]
  ): Promise<{
    text: string;
    confidence: number;
    sources: string[];
  }> {
    const response = await this.llm.invoke(
      `Synthesize an answer to this query based on the function execution results:

Query: "${query}"

Functions executed:
${functions.map(f => `- ${f.name} (${f.signature})`).join('\n')}

Execution results:
${JSON.stringify(executionResults.results, null, 2)}

Knowledge accessed: ${executionResults.knowledge_accessed.length} sources

Requirements:
1. Answer the user's query directly
2. Cite specific knowledge sources
3. Include confidence level
4. Explain reasoning if confidence < 80%
5. Be concise and clear

Return JSON:
\`\`\`json
{
  "answer": "direct answer to query",
  "confidence": 0.85,
  "sources": ["source1", "source2"],
  "reasoning": "brief explanation"
}
\`\`\``,
      {
        task: 'reasoning',
        max_tokens: 1000,
        enable_constitutional: true
      }
    );

    try {
      const jsonMatch = response.text.match(/```(?:json)?\n([\s\S]*?)\n```/);
      if (jsonMatch) {
        const data = JSON.parse(jsonMatch[1]);
        return {
          text: data.answer || 'Unable to generate answer',
          confidence: data.confidence || 0.5,
          sources: data.sources || []
        };
      }
    } catch (e) {
      // Fallback
    }

    return {
      text: response.text,
      confidence: 0.5,
      sources: []
    };
  }

  /**
   * Update organism memory with query result
   */
  private updateMemory(result: QueryResult): void {
    // Add to short-term memory
    this.organism.memory.short_term.push({
      type: 'query',
      query: result.query,
      answer: result.answer,
      confidence: result.confidence,
      timestamp: result.timestamp
    });

    // Keep only last 100 in short-term
    if (this.organism.memory.short_term.length > 100) {
      // Move oldest to long-term
      const old = this.organism.memory.short_term.shift();
      this.organism.memory.long_term.push(old);
    }
  }

  /**
   * Map organism specialization to constitutional domain
   */
  private mapSpecializationToDomain(specialization: string): any {
    if (specialization.includes('bio') || specialization.includes('onco') || specialization.includes('medical')) {
      return 'biology';
    }
    if (specialization.includes('fin') || specialization.includes('econ')) {
      return 'financial';
    }
    return 'universal';
  }

  /**
   * Get runtime statistics
   */
  getStats() {
    return {
      organism: {
        name: this.organism.metadata.name,
        specialization: this.organism.metadata.specialization,
        maturity: this.organism.metadata.maturity,
        functions_count: this.organism.code.functions.length,
        knowledge_count: this.organism.knowledge.papers.count
      },
      runtime: {
        total_cost: this.llm.getTotalCost(),
        remaining_budget: this.llm.getRemainingBudget(),
        queries_processed: this.organism.memory.short_term.length,
        attention_tracked: this.attentionMap.size
      }
    };
  }

  /**
   * Format result for display
   */
  static formatResult(result: QueryResult): string {
    let output = '';

    output += `\n${'='.repeat(80)}\n`;
    output += `QUERY: ${result.query}\n`;
    output += `${'='.repeat(80)}\n\n`;

    output += `📝 ANSWER:\n`;
    output += `${result.answer}\n\n`;

    output += `📊 METADATA:\n`;
    output += `├── Confidence: ${(result.confidence * 100).toFixed(0)}%\n`;
    output += `├── Functions used: ${result.functions_used.join(', ')}\n`;
    output += `├── Constitutional: ${result.constitutional_passed ? '✅ PASS' : '❌ FAIL'}\n`;
    output += `├── Cost: $${result.cost_usd.toFixed(4)}\n`;
    output += `└── Timestamp: ${result.timestamp}\n\n`;

    if (result.sources.length > 0) {
      output += `📚 SOURCES:\n`;
      result.sources.forEach((source, i) => {
        output += `${i + 1}. ${source}\n`;
      });
      output += '\n';
    }

    if (result.attention_weights.length > 0) {
      output += `👁️  ATTENTION (Top ${Math.min(5, result.attention_weights.length)}):\n`;
      result.attention_weights.slice(0, 5).forEach(att => {
        output += `├── ${att.knowledge_id}: ${(att.weight * 100).toFixed(1)}%\n`;
      });
      output += '\n';
    }

    if (result.reasoning.length > 0) {
      output += `🧠 REASONING:\n`;
      result.reasoning.forEach((step, i) => {
        output += `${i + 1}. ${step}\n`;
      });
      output += '\n';
    }

    output += `${'='.repeat(80)}\n`;

    return output;
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create runtime from .glass file
 */
export async function createRuntime(
  glassPath: string,
  maxBudget: number = 0.5
): Promise<GlassRuntime> {
  const organism = await loadGlassOrganism(glassPath);
  return new GlassRuntime(organism, maxBudget);
}

/**
 * Quick query - create runtime, execute query, return result
 */
export async function quickQuery(
  glassPath: string,
  query: string,
  maxBudget: number = 0.5
): Promise<QueryResult> {
  const runtime = await createRuntime(glassPath, maxBudget);
  return runtime.query({ query });
}
