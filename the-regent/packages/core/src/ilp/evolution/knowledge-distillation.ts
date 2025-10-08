/**
 * KnowledgeDistillation - Extract patterns from episodic memory
 *
 * Analyzes episodic memory to:
 * - Discover recurring concept patterns
 * - Identify knowledge gaps
 * - Detect systematic errors
 * - Synthesize new knowledge
 */

import { EpisodicMemory, Episode } from './episodic-memory';
import { AnthropicAdapter } from '../llm/anthropic-adapter';
import { Observability } from './observability';

// ============================================================================
// Types
// ============================================================================

export interface KnowledgePattern {
  concepts: string[];
  frequency: number;
  domains: string[];
  confidence: number;
  representative_queries: string[];
  emergent_insight: string;
}

export interface ErrorPattern {
  concept: string;
  frequency: number;
  typical_error: string;
  suggested_fix: string;
}

export interface KnowledgeGap {
  concept: string;
  evidence: string[]; // Episode IDs showing the gap
}

// ============================================================================
// KnowledgeDistillation Class
// ============================================================================

export class KnowledgeDistillation {
  constructor(
    private episodicMemory: EpisodicMemory,
    private llm: AnthropicAdapter,
    private observability: Observability
  ) {}

  /**
   * Discover recurring patterns in episodes
   */
  async discoverPatterns(
    episodes: Episode[],
    minFrequency: number = 3
  ): Promise<KnowledgePattern[]> {
    const span = this.observability.startSpan('discover_patterns');
    span.setTag('episodes_count', episodes.length);
    span.setTag('min_frequency', minFrequency);

    try {
      // 1. Extract concept co-occurrences
      const conceptSets = episodes
        .filter((ep) => ep.success)
        .map((ep) => ({
          concepts: ep.concepts.sort(),
          domains: ep.domains,
          query: ep.query,
        }));

      // 2. Group by concept combination
      const patternMap = new Map<string, {
        concepts: string[];
        domains: Set<string>;
        queries: string[];
        count: number;
        totalConfidence: number;
      }>();

      for (const set of conceptSets) {
        if (set.concepts.length === 0) continue;

        const key = set.concepts.join('|');
        const existing = patternMap.get(key);

        if (existing) {
          existing.count++;
          existing.queries.push(set.query);
          set.domains.forEach((d) => existing.domains.add(d));
        } else {
          patternMap.set(key, {
            concepts: set.concepts,
            domains: new Set(set.domains),
            queries: [set.query],
            count: 1,
            totalConfidence: 0,
          });
        }
      }

      // 3. Filter by minimum frequency and convert to patterns
      const patterns: KnowledgePattern[] = [];

      for (const [key, data] of patternMap.entries()) {
        if (data.count >= minFrequency) {
          // Calculate confidence based on frequency and episode success rate
          const confidence = Math.min(data.count / (minFrequency * 2), 1.0);

          patterns.push({
            concepts: data.concepts,
            frequency: data.count,
            domains: Array.from(data.domains),
            confidence,
            representative_queries: data.queries.slice(0, 3),
            emergent_insight: `Pattern of ${data.concepts.join(', ')} appears ${data.count} times`,
          });
        }
      }

      // 4. Sort by frequency (most common first)
      patterns.sort((a, b) => b.frequency - a.frequency);

      this.observability.log('info', 'patterns_discovered', {
        total_patterns: patterns.length,
        min_frequency: minFrequency,
      });

      span.setTag('patterns_found', patterns.length);
      return patterns;
    } finally {
      span.end();
    }
  }

  /**
   * Identify knowledge gaps from low-confidence episodes
   */
  async identifyGaps(episodes: Episode[]): Promise<KnowledgeGap[]> {
    const span = this.observability.startSpan('identify_gaps');

    try {
      const gaps = new Map<string, string[]>();

      // Find failed or low-confidence episodes
      const problematic = episodes.filter(
        (ep) => !ep.success || ep.confidence < 0.5
      );

      for (const ep of problematic) {
        // Extract potential concepts from query
        const queryWords = ep.query
          .toLowerCase()
          .split(/\s+/)
          .filter((w) => w.length > 4);

        for (const word of queryWords) {
          const existing = gaps.get(word);
          if (existing) {
            existing.push(ep.id);
          } else {
            gaps.set(word, [ep.id]);
          }
        }

        // Also add any concepts that were attempted but failed
        for (const concept of ep.concepts) {
          if (!ep.success) {
            const existing = gaps.get(concept);
            if (existing) {
              existing.push(ep.id);
            } else {
              gaps.set(concept, [ep.id]);
            }
          }
        }
      }

      const result: KnowledgeGap[] = Array.from(gaps.entries()).map(
        ([concept, evidence]) => ({
          concept,
          evidence,
        })
      );

      this.observability.log('info', 'gaps_identified', {
        total_gaps: result.length,
      });

      span.setTag('gaps_found', result.length);
      return result;
    } finally {
      span.end();
    }
  }

  /**
   * Detect systematic errors in episodes
   */
  async detectErrors(episodes: Episode[]): Promise<ErrorPattern[]> {
    const span = this.observability.startSpan('detect_errors');

    try {
      const errorMap = new Map<string, {
        count: number;
        examples: string[];
      }>();

      // Find failed episodes
      const failed = episodes.filter((ep) => !ep.success);

      for (const ep of failed) {
        // Extract error concepts from query
        const queryWords = ep.query
          .toLowerCase()
          .split(/\s+/)
          .filter((w) => w.length > 4);

        for (const word of queryWords) {
          const existing = errorMap.get(word);
          if (existing) {
            existing.count++;
            existing.examples.push(ep.query);
          } else {
            errorMap.set(word, {
              count: 1,
              examples: [ep.query],
            });
          }
        }
      }

      // Convert to error patterns
      const errors: ErrorPattern[] = [];

      for (const [concept, data] of errorMap.entries()) {
        if (data.count >= 2) {
          errors.push({
            concept,
            frequency: data.count,
            typical_error: `Failed to answer questions about ${concept}`,
            suggested_fix: `Add knowledge slice about ${concept} to ${this.inferDomain(concept)} domain`,
          });
        }
      }

      // Sort by frequency
      errors.sort((a, b) => b.frequency - a.frequency);

      this.observability.log('info', 'errors_detected', {
        total_errors: errors.length,
      });

      span.setTag('errors_found', errors.length);
      return errors;
    } finally {
      span.end();
    }
  }

  /**
   * Synthesize new knowledge from pattern
   */
  async synthesize(pattern: KnowledgePattern): Promise<string> {
    const span = this.observability.startSpan('synthesize_knowledge');
    span.setTag('pattern_concepts', pattern.concepts.join(','));

    try {
      const systemPrompt = `You are a knowledge synthesizer. Your job is to create a new knowledge slice (in YAML format) based on recurring patterns from user queries.

The slice should:
1. Include all concepts from the pattern
2. Provide clear, accurate information
3. Follow YAML format exactly
4. Be suitable for ${pattern.domains.join(', ')} domain(s)`;

      const userPrompt = `Create a knowledge slice for this pattern:

Concepts: ${pattern.concepts.join(', ')}
Frequency: ${pattern.frequency} occurrences
Domains: ${pattern.domains.join(', ')}
Example queries:
${pattern.representative_queries.map((q, i) => `${i + 1}. ${q}`).join('\n')}

Generate YAML in this format:
\`\`\`yaml
id: <slug-from-concepts>
title: <descriptive-title>
description: <what-this-slice-covers>
concepts:
  - <concept1>
  - <concept2>
domains:
  - <domain1>
content: |
  <actual-knowledge-content>

  Use multiple paragraphs to explain thoroughly.
\`\`\``;

      const response = await this.llm.invoke(systemPrompt, userPrompt, {
        model: 'claude-sonnet-4-5',
        max_tokens: 2000,
        temperature: 0.3,
      });

      // Extract YAML from markdown code block if present
      let yaml = response.text;
      const codeBlockMatch = yaml.match(/```(?:yaml)?\s*\n([\s\S]*?)\n```/);
      if (codeBlockMatch) {
        yaml = codeBlockMatch[1];
      }

      this.observability.log('info', 'knowledge_synthesized', {
        concepts: pattern.concepts,
        cost: response.usage.cost_usd,
      });

      span.setTag('cost', response.usage.cost_usd);
      return yaml;
    } finally {
      span.end();
    }
  }

  /**
   * Infer domain from concept name
   */
  private inferDomain(concept: string): string {
    const lowerConcept = concept.toLowerCase();

    // Simple heuristics
    if (
      lowerConcept.includes('interest') ||
      lowerConcept.includes('budget') ||
      lowerConcept.includes('invest') ||
      lowerConcept.includes('money')
    ) {
      return 'financial';
    }

    if (
      lowerConcept.includes('cell') ||
      lowerConcept.includes('bio') ||
      lowerConcept.includes('organ') ||
      lowerConcept.includes('dna')
    ) {
      return 'biology';
    }

    if (
      lowerConcept.includes('system') ||
      lowerConcept.includes('feedback') ||
      lowerConcept.includes('loop') ||
      lowerConcept.includes('network')
    ) {
      return 'systems';
    }

    return 'general';
  }

  /**
   * Get statistics about patterns
   */
  getPatternStats(patterns: KnowledgePattern[]): {
    total: number;
    avg_frequency: number;
    avg_confidence: number;
    most_common_domains: string[];
  } {
    if (patterns.length === 0) {
      return {
        total: 0,
        avg_frequency: 0,
        avg_confidence: 0,
        most_common_domains: [],
      };
    }

    const totalFrequency = patterns.reduce((sum, p) => sum + p.frequency, 0);
    const totalConfidence = patterns.reduce((sum, p) => sum + p.confidence, 0);

    const domainCounts = new Map<string, number>();
    for (const pattern of patterns) {
      for (const domain of pattern.domains) {
        domainCounts.set(domain, (domainCounts.get(domain) || 0) + 1);
      }
    }

    const mostCommon = Array.from(domainCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([domain]) => domain);

    return {
      total: patterns.length,
      avg_frequency: totalFrequency / patterns.length,
      avg_confidence: totalConfidence / patterns.length,
      most_common_domains: mostCommon,
    };
  }
}

/**
 * Create a new KnowledgeDistillation instance
 */
export function createKnowledgeDistillation(
  memory: EpisodicMemory,
  llm: AnthropicAdapter,
  observability: Observability
): KnowledgeDistillation {
  return new KnowledgeDistillation(memory, llm, observability);
}
