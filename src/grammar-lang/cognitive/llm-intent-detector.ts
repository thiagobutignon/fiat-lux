/**
 * LLM-Powered Intent Detection for CINZA (Cognitive Defense)
 *
 * Replaces rule-based intent detection with LLM semantic analysis.
 * Uses GlassLLM with 'intent-analysis' task.
 */

import { createGlassLLM, GlassLLM } from '../glass/llm-adapter';
import { Pragmatics, Morphemes, Syntax, Semantics } from './types';

// ============================================================================
// LLM Intent Detector
// ============================================================================

export class LLMIntentDetector {
  private llm: GlassLLM;

  constructor(maxBudget: number = 0.2) {
    // Create LLM for cognitive domain
    this.llm = createGlassLLM('cognitive', maxBudget);
  }

  /**
   * Detect intent using LLM semantic analysis
   */
  async detectIntent(
    morphemes: Morphemes,
    syntax: Syntax,
    semantics: Semantics,
    originalText: string
  ): Promise<Pragmatics['intent']> {
    const prompt = this.buildIntentPrompt(morphemes, syntax, semantics, originalText);

    const response = await this.llm.invoke(prompt, {
      task: 'intent-analysis',
      max_tokens: 500,
      enable_constitutional: true
    });

    // Parse intent from response
    return this.parseIntent(response.text);
  }

  /**
   * Build prompt for intent analysis
   */
  private buildIntentPrompt(
    morphemes: Morphemes,
    syntax: Syntax,
    semantics: Semantics,
    originalText: string
  ): string {
    return `Analyze the communicative intent of this text using pragmatics:

**Original Text**: "${originalText}"

**Linguistic Analysis**:

**Morphemes**:
- Keywords: ${morphemes.keywords.join(', ') || 'none'}
- Negations: ${morphemes.negations.join(', ') || 'none'}
- Intensifiers: ${morphemes.intensifiers.join(', ') || 'none'}
- Qualifiers: ${morphemes.qualifiers.join(', ') || 'none'}

**Syntax**:
- Temporal distortion: ${syntax.temporal_distortion}
- Pronoun reversal: ${syntax.pronoun_reversal}
- Modal manipulation: ${syntax.modal_manipulation}
- Question patterns: ${syntax.question_patterns.length}

**Semantics**:
- Reality denial: ${semantics.reality_denial}
- Memory invalidation: ${semantics.memory_invalidation}
- Emotional dismissal: ${semantics.emotional_dismissal}
- Blame shifting: ${semantics.blame_shifting}
- Projection: ${semantics.projection}

**Intent Categories**:
- **manipulate**: Attempt to alter perception of reality, gaslight, or distort truth
- **control**: Exert power/dominance, enforce compliance
- **deceive**: Mislead or create false impressions
- **confuse**: Create uncertainty, mental fog, doubt
- **dominate**: Assert superiority, dismiss feelings/thoughts
- **harm**: Intent to cause psychological/emotional damage

**Task**: Determine the primary communicative intent based on:
1. Linguistic patterns (morphemes, syntax, semantics)
2. Power dynamics implied
3. Emotional impact on recipient
4. Context of manipulation tactics

Return ONLY JSON:
\`\`\`json
{
  "intent": "manipulate|control|deceive|confuse|dominate|harm",
  "confidence": 0.85,
  "reasoning": ["reason 1", "reason 2"],
  "secondary_intents": ["intent2", "intent3"]
}
\`\`\``;
  }

  /**
   * Parse intent from LLM response
   */
  private parseIntent(response: string): Pragmatics['intent'] {
    try {
      // Extract JSON from response
      const jsonMatch = response.match(/```(?:json)?\n([\s\S]*?)\n```/);

      if (jsonMatch) {
        const data = JSON.parse(jsonMatch[1]);

        // Validate intent
        const validIntents: Pragmatics['intent'][] = [
          'manipulate', 'control', 'deceive', 'confuse', 'dominate', 'harm'
        ];

        if (validIntents.includes(data.intent)) {
          return data.intent;
        }
      }
    } catch (error) {
      console.warn('⚠️  Error parsing LLM intent:', error);
    }

    // Fallback to manipulate
    return 'manipulate';
  }

  /**
   * Analyze full pragmatics using LLM
   */
  async analyzePragmatics(
    morphemes: Morphemes,
    syntax: Syntax,
    semantics: Semantics,
    originalText: string
  ): Promise<Pragmatics> {
    const prompt = `Analyze pragmatics (communicative intent, power dynamics, social impact) of this text:

**Text**: "${originalText}"

**Linguistic Features**:
- Morphemes: ${JSON.stringify(morphemes, null, 2)}
- Syntax: ${JSON.stringify(syntax, null, 2)}
- Semantics: ${JSON.stringify(semantics, null, 2)}

**Analysis Required**:
1. **Intent**: Primary communicative intent (manipulate|control|deceive|confuse|dominate|harm)
2. **Context Awareness**: How context-dependent is this? (0.0 = works anywhere, 1.0 = highly specific)
3. **Power Dynamic**: How is power being manipulated? (exploit|reverse)
   - exploit: Using existing power imbalance
   - reverse: DARVO - reversing victim/perpetrator roles
4. **Social Impact**: Effect on social relationships (isolate|triangulate|recruit|divide)
   - isolate: Separating victim from support
   - triangulate: Using third party to manipulate
   - recruit: Gaining sympathy/allies
   - divide: Creating conflict between others

Return JSON:
\`\`\`json
{
  "intent": "manipulate",
  "context_awareness": 0.7,
  "power_dynamic": "exploit",
  "social_impact": "isolate",
  "confidence": 0.85,
  "reasoning": "Brief explanation"
}
\`\`\``;

    const response = await this.llm.invoke(prompt, {
      task: 'intent-analysis',
      max_tokens: 800,
      enable_constitutional: true
    });

    // Parse full pragmatics
    return this.parsePragmatics(response.text);
  }

  /**
   * Parse full pragmatics from LLM response
   */
  private parsePragmatics(response: string): Pragmatics {
    try {
      const jsonMatch = response.match(/```(?:json)?\n([\s\S]*?)\n```/);

      if (jsonMatch) {
        const data = JSON.parse(jsonMatch[1]);

        return {
          intent: data.intent || 'manipulate',
          context_awareness: data.context_awareness || 0.5,
          power_dynamic: data.power_dynamic || 'exploit',
          social_impact: data.social_impact || 'isolate'
        };
      }
    } catch (error) {
      console.warn('⚠️  Error parsing LLM pragmatics:', error);
    }

    // Fallback
    return {
      intent: 'manipulate',
      context_awareness: 0.5,
      power_dynamic: 'exploit',
      social_impact: 'isolate'
    };
  }

  /**
   * Get cost stats
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
 * Create LLM intent detector
 */
export function createLLMIntentDetector(maxBudget: number = 0.2): LLMIntentDetector {
  return new LLMIntentDetector(maxBudget);
}
