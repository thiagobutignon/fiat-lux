/**
 * LLM-Powered Code Synthesis for ROXO (Code Emergence)
 *
 * Replaces hardcoded templates with real LLM code generation.
 * Uses GlassLLM with 'code-synthesis' task.
 */

import { createGlassLLM, GlassLLM } from './llm-adapter';
import { EmergenceCandidate } from './patterns';

// ============================================================================
// LLM Code Synthesis
// ============================================================================

export class LLMCodeSynthesizer {
  private llm: GlassLLM;

  constructor(maxBudget: number = 0.5) {
    // Create LLM for glass-core domain with budget
    this.llm = createGlassLLM('glass-core', maxBudget);
  }

  /**
   * Synthesize .gl code from emergence candidate
   */
  async synthesize(candidate: EmergenceCandidate, organism: any): Promise<string> {
    const prompt = this.buildSynthesisPrompt(candidate, organism);

    const response = await this.llm.invoke(prompt, {
      task: 'code-synthesis',
      max_tokens: 2000,
      enable_constitutional: true
    });

    // Extract .gl code from response
    const glCode = this.extractCode(response.text);

    return glCode;
  }

  /**
   * Build prompt for code synthesis
   */
  private buildSynthesisPrompt(candidate: EmergenceCandidate, organism: any): string {
    const { suggested_function_name, suggested_signature, pattern, supporting_patterns } = candidate;

    return `Synthesize .gl (Grammar Language) function from the following emergence pattern:

**Function Name**: ${suggested_function_name}
**Signature**: ${suggested_signature}
**Domain**: ${organism.metadata.specialization}
**Pattern Type**: ${pattern.type}
**Pattern Frequency**: ${pattern.frequency} occurrences
**Confidence**: ${(pattern.confidence * 100).toFixed(0)}%

**Supporting Patterns**:
${supporting_patterns.map((p: string) => `- ${p}`).join('\n')}

**Pattern Description**:
${pattern.keywords.join(', ')} - Pattern detected from ${pattern.frequency} occurrences in knowledge base

**Requirements**:
1. Generate valid .gl syntax (similar to functional languages)
2. Query knowledge base using: \`query_knowledge_base(pattern: "...", filters: [...])\`
3. Use pattern matching where appropriate: \`match value: | case1 -> result1 | case2 -> result2\`
4. Include confidence calculation
5. Return structured output matching ${suggested_signature}
6. Add comments explaining key logic
7. Follow O(1) performance where possible
8. Include constitutional checks (confidence thresholds, source citations)

**Example .gl structure**:
\`\`\`gl
function ${suggested_function_name}(${this.extractParams(suggested_signature)}) -> ${this.extractReturnType(suggested_signature)}:
  # Query knowledge base
  results = query_knowledge_base(
    pattern: "${pattern.type}",
    filters: [/* relevant filters */]
  )

  # Process results
  processed = /* logic here */

  # Calculate confidence
  confidence = calculate_confidence(results.size)

  # Return with metadata
  return OutputType(
    value: processed,
    confidence: confidence,
    sources: results.citations
  )
\`\`\`

Generate ONLY the .gl function code (no explanations outside code block).`;
  }

  /**
   * Extract parameters from signature
   */
  private extractParams(signature: string): string {
    const match = signature.match(/\(([^)]*)\)/);
    return match ? match[1] : '';
  }

  /**
   * Extract return type from signature
   */
  private extractReturnType(signature: string): string {
    const match = signature.match(/->\s*(.+)/);
    return match ? match[1].trim() : 'Output';
  }

  /**
   * Extract .gl code from LLM response
   */
  private extractCode(response: string): string {
    // Look for code blocks
    const codeBlockMatch = response.match(/```(?:gl)?\n([\s\S]*?)\n```/);

    if (codeBlockMatch) {
      return codeBlockMatch[1].trim();
    }

    // If no code block, try to extract function definition
    const functionMatch = response.match(/function\s+\w+[\s\S]*?(?=\n\n|$)/);

    if (functionMatch) {
      return functionMatch[0].trim();
    }

    // Fallback: return response as-is
    return response.trim();
  }

  /**
   * Get synthesis cost
   */
  getTotalCost(): number {
    return this.llm.getTotalCost();
  }

  /**
   * Get remaining budget
   */
  getRemainingBudget(): number {
    return this.llm.getRemainingBudget();
  }

  /**
   * Get cost stats
   */
  getCostStats() {
    return this.llm.getCostStats();
  }
}

// ============================================================================
// Factory
// ============================================================================

/**
 * Create LLM code synthesizer
 */
export function createLLMCodeSynthesizer(maxBudget: number = 0.5): LLMCodeSynthesizer {
  return new LLMCodeSynthesizer(maxBudget);
}
