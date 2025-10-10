/**
 * LLM-Powered Semantic Pattern Detection
 *
 * Replaces keyword-based correlation with LLM semantic analysis.
 * Uses GlassLLM with 'pattern-detection' task.
 */

import { createGlassLLM, GlassLLM } from './llm-adapter';
import { EnhancedPattern, PatternCorrelation } from './patterns';

// ============================================================================
// LLM Pattern Detector
// ============================================================================

export class LLMPatternDetector {
  private llm: GlassLLM;

  constructor(maxBudget: number = 0.3) {
    // Create LLM for glass-core domain
    this.llm = createGlassLLM('glass-core', maxBudget);
  }

  /**
   * Detect semantic correlations between patterns using LLM
   */
  async detectSemanticCorrelations(
    patterns: EnhancedPattern[],
    threshold: number = 0.6
  ): Promise<PatternCorrelation[]> {
    const correlations: PatternCorrelation[] = [];

    // Build pattern descriptions for LLM
    const patternDescriptions = patterns.map(p => ({
      type: p.type,
      keywords: p.keywords.join(', '),
      frequency: p.frequency,
      confidence: p.confidence
    }));

    // Invoke LLM for semantic pattern analysis
    const prompt = this.buildCorrelationPrompt(patternDescriptions);

    const response = await this.llm.invoke(prompt, {
      task: 'pattern-detection',
      max_tokens: 2000,
      enable_constitutional: true
    });

    // Parse LLM response
    const detected = this.parseCorrelations(response.text, threshold);

    return detected;
  }

  /**
   * Build prompt for correlation detection
   */
  private buildCorrelationPrompt(patterns: any[]): string {
    return `Analyze semantic correlations between these knowledge patterns:

${patterns.map((p, i) => `${i + 1}. **${p.type}** (${p.keywords}) - Frequency: ${p.frequency}, Confidence: ${p.confidence.toFixed(2)}`).join('\n')}

For each pair of patterns that are semantically related:
1. Calculate correlation strength (0.0 to 1.0) based on:
   - Semantic similarity of concepts
   - Potential for co-occurrence in knowledge base
   - Shared domain concepts
   - Logical dependencies

2. Estimate co-occurrence count (how often they might appear together)

Return ONLY a JSON array of correlations above 0.5 strength:
\`\`\`json
{
  "correlations": [
    {
      "pattern_a": "pattern_type_1",
      "pattern_b": "pattern_type_2",
      "strength": 0.85,
      "co_occurrence": 120,
      "reasoning": "Brief explanation why these patterns correlate"
    }
  ]
}
\`\`\`

Focus on finding meaningful semantic relationships, not just keyword overlap.`;
  }

  /**
   * Parse correlations from LLM response
   */
  private parseCorrelations(response: string, threshold: number): PatternCorrelation[] {
    try {
      // Extract JSON from response
      const jsonMatch = response.match(/```(?:json)?\n([\s\S]*?)\n```/);

      if (!jsonMatch) {
        console.warn('⚠️  LLM response does not contain JSON block');
        return [];
      }

      const data = JSON.parse(jsonMatch[1]);

      if (!data.correlations || !Array.isArray(data.correlations)) {
        console.warn('⚠️  LLM response missing correlations array');
        return [];
      }

      // Filter by threshold and map to PatternCorrelation
      return data.correlations
        .filter((c: any) => c.strength >= threshold)
        .map((c: any) => ({
          pattern_a: c.pattern_a,
          pattern_b: c.pattern_b,
          strength: c.strength,
          co_occurrence: c.co_occurrence
        }));

    } catch (error) {
      console.error('❌ Error parsing LLM correlations:', error);
      return [];
    }
  }

  /**
   * Analyze pattern cluster potential using LLM
   */
  async analyzeClusterPotential(
    patterns: EnhancedPattern[],
    correlations: PatternCorrelation[]
  ): Promise<{
    cluster_name: string;
    strength: number;
    potential_functions: string[];
  }> {
    const prompt = `Analyze this pattern cluster for function emergence potential:

**Patterns**:
${patterns.map(p => `- ${p.type} (${p.keywords.join(', ')}): ${p.frequency} occurrences, ${(p.confidence * 100).toFixed(0)}% confidence`).join('\n')}

**Correlations**:
${correlations.map(c => `- ${c.pattern_a} ↔ ${c.pattern_b}: ${(c.strength * 100).toFixed(0)}% correlation`).join('\n')}

Analyze:
1. What unified concept do these patterns represent?
2. What's the cluster strength (0.0-1.0)?
3. What functions could emerge from this cluster?

Return JSON:
\`\`\`json
{
  "cluster_name": "descriptive_name",
  "strength": 0.85,
  "potential_functions": ["function_name_1", "function_name_2"],
  "reasoning": "Brief explanation"
}
\`\`\``;

    const response = await this.llm.invoke(prompt, {
      task: 'pattern-detection',
      max_tokens: 1000,
      enable_constitutional: true
    });

    // Parse response
    try {
      const jsonMatch = response.text.match(/```(?:json)?\n([\s\S]*?)\n```/);
      if (jsonMatch) {
        const data = JSON.parse(jsonMatch[1]);
        return {
          cluster_name: data.cluster_name || 'unnamed_cluster',
          strength: data.strength || 0.5,
          potential_functions: data.potential_functions || []
        };
      }
    } catch (error) {
      console.error('❌ Error parsing cluster analysis:', error);
    }

    // Fallback
    return {
      cluster_name: 'unnamed_cluster',
      strength: 0.5,
      potential_functions: []
    };
  }

  /**
   * Get LLM cost stats
   */
  getCostStats() {
    return this.llm.getCostStats();
  }

  /**
   * Get total cost
   */
  getTotalCost(): number {
    return this.llm.getTotalCost();
  }
}

// ============================================================================
// Factory
// ============================================================================

/**
 * Create LLM pattern detector
 */
export function createLLMPatternDetector(maxBudget: number = 0.3): LLMPatternDetector {
  return new LLMPatternDetector(maxBudget);
}
