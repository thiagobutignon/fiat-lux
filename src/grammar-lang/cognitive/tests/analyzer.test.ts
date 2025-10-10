/**
 * Analyzer Tests
 * Tests Sprint 2 components: intent detection, temporal tracking, cultural filters
 */

import { describe, it, expect } from '@jest/globals';
import { parseMorphemes } from '../parser/morphemes';
import { parseSyntax } from '../parser/syntax';
import { parseSemantics } from '../parser/semantics';
import { parsePragmatics } from '../parser/pragmatics';

import {
  detectEnhancedIntent,
  analyzeWithContext
} from '../analyzer/intent-detector';

import {
  extractTemporalData,
  analyzeTemporalPatterns,
  predictFuturePrevalence,
  generateEvolutionGraphs
} from '../analyzer/temporal-tracker';

import {
  applyCulturalFilters,
  inferCulturalContext,
  getRecommendedAdjustment
} from '../analyzer/cultural-filters';

describe('Enhanced Intent Detection', () => {
  it('should detect intent with context', () => {
    const morphemes = parseMorphemes("You're crazy!");
    const syntax = parseSyntax("You're crazy!");
    const semantics = parseSemantics("You're crazy!");
    const pragmatics = parsePragmatics(morphemes, syntax, semantics);

    const context = {
      relationship_type: 'intimate' as const,
      power_dynamic: 'equal' as const,
      history_of_manipulation: false,
      previous_detections: [],
      conversation_length: 1
    };

    const intent = detectEnhancedIntent(morphemes, syntax, semantics, pragmatics, context);

    expect(intent.primary_intent).toBeDefined();
    expect(intent.confidence).toBeGreaterThan(0);
    expect(intent.context_adjusted_confidence).toBeGreaterThan(0);
    expect(intent.reasoning).toBeInstanceOf(Array);
  });

  it('should adjust confidence based on history of manipulation', () => {
    const morphemes = parseMorphemes("That didn't happen");
    const syntax = parseSyntax("That didn't happen");
    const semantics = parseSemantics("That didn't happen");
    const pragmatics = parsePragmatics(morphemes, syntax, semantics);

    const contextWithHistory = {
      relationship_type: 'intimate' as const,
      power_dynamic: 'superior' as const,
      history_of_manipulation: true,
      previous_detections: [],
      conversation_length: 5
    };

    const contextWithoutHistory = {
      relationship_type: 'intimate' as const,
      power_dynamic: 'superior' as const,
      history_of_manipulation: false,
      previous_detections: [],
      conversation_length: 5
    };

    const intentWith = detectEnhancedIntent(morphemes, syntax, semantics, pragmatics, contextWithHistory);
    const intentWithout = detectEnhancedIntent(morphemes, syntax, semantics, pragmatics, contextWithoutHistory);

    expect(intentWith.context_adjusted_confidence).toBeGreaterThan(intentWithout.context_adjusted_confidence);
  });

  it('should provide reasoning chain', () => {
    const morphemes = parseMorphemes("You're wrong");
    const syntax = parseSyntax("You're wrong");
    const semantics = parseSemantics("You're wrong");
    const pragmatics = parsePragmatics(morphemes, syntax, semantics);

    const context = {
      relationship_type: 'professional' as const,
      power_dynamic: 'equal' as const,
      history_of_manipulation: false,
      previous_detections: [],
      conversation_length: 1
    };

    const intent = detectEnhancedIntent(morphemes, syntax, semantics, pragmatics, context);

    expect(intent.reasoning).toBeDefined();
    expect(intent.reasoning.length).toBeGreaterThan(0);
  });

  it('should detect secondary intents', () => {
    const morphemes = parseMorphemes("That never happened. You're lying.");
    const syntax = parseSyntax("That never happened. You're lying.");
    const semantics = parseSemantics("That never happened. You're lying.");
    const pragmatics = parsePragmatics(morphemes, syntax, semantics);

    const context = {
      relationship_type: 'intimate' as const,
      power_dynamic: 'superior' as const,
      history_of_manipulation: false,
      previous_detections: [],
      conversation_length: 1
    };

    const intent = detectEnhancedIntent(morphemes, syntax, semantics, pragmatics, context);

    expect(intent.secondary_intents).toBeDefined();
    expect(Array.isArray(intent.secondary_intents)).toBe(true);
  });
});

describe('Contextual Analysis', () => {
  it('should analyze with full context', () => {
    const morphemes = parseMorphemes("You're imagining things");
    const syntax = parseSyntax("You're imagining things");
    const semantics = parseSemantics("You're imagining things");
    const pragmatics = parsePragmatics(morphemes, syntax, semantics);

    const context = {
      relationship_type: 'intimate' as const,
      power_dynamic: 'superior' as const,
      history_of_manipulation: true,
      previous_detections: [],
      conversation_length: 10
    };

    const analysis = analyzeWithContext(morphemes, syntax, semantics, pragmatics, context);

    expect(analysis.intent).toBeDefined();
    expect(analysis.relationship_risk_score).toBeGreaterThanOrEqual(0);
    expect(analysis.relationship_risk_score).toBeLessThanOrEqual(1);
    expect(analysis.intervention_urgency).toBeDefined();
    expect(['low', 'medium', 'high', 'critical']).toContain(analysis.intervention_urgency);
  });

  it('should determine intervention urgency', () => {
    const morphemes = parseMorphemes("You're crazy! That never happened!");
    const syntax = parseSyntax("You're crazy! That never happened!");
    const semantics = parseSemantics("You're crazy! That never happened!");
    const pragmatics = parsePragmatics(morphemes, syntax, semantics);

    const highRiskContext = {
      relationship_type: 'intimate' as const,
      power_dynamic: 'superior' as const,
      history_of_manipulation: true,
      previous_detections: Array(5).fill({ confidence: 0.9 }),
      conversation_length: 10
    };

    const analysis = analyzeWithContext(morphemes, syntax, semantics, pragmatics, highRiskContext);

    expect(['high', 'critical']).toContain(analysis.intervention_urgency);
  });
});

describe('Temporal Tracking', () => {
  it('should extract temporal data', () => {
    const temporalData = extractTemporalData();

    expect(temporalData).toBeInstanceOf(Array);
    expect(temporalData.length).toBeGreaterThan(0);

    temporalData.forEach(data => {
      expect(data.technique_id).toBeGreaterThanOrEqual(153);
      expect(data.technique_id).toBeLessThanOrEqual(180);
      expect(data.emerged_year).toBeGreaterThanOrEqual(2023);
      expect(data.emerged_year).toBeLessThanOrEqual(2025);
    });
  });

  it('should analyze temporal patterns', () => {
    const analysis = analyzeTemporalPatterns(2025);

    expect(analysis.current_year).toBe(2025);
    expect(analysis.total_techniques_active).toBeGreaterThan(0);
    expect(analysis.gpt4_era_active).toBe(152);
    expect(analysis.gpt5_era_active).toBe(28);
    expect(analysis.emerging_techniques).toBeInstanceOf(Array);
    expect(analysis.causality_chains).toBeInstanceOf(Array);
  });

  it('should predict future prevalence', () => {
    const prediction = predictFuturePrevalence(153, 2026);

    if (prediction) {
      expect(prediction.prevalence).toBeGreaterThanOrEqual(0);
      expect(prediction.prevalence).toBeLessThanOrEqual(1);
      expect(prediction.confidence).toBeGreaterThanOrEqual(0);
      expect(prediction.confidence).toBeLessThanOrEqual(1);
    }
  });

  it('should generate evolution graphs', () => {
    const graphs = generateEvolutionGraphs();

    expect(graphs).toBeInstanceOf(Array);
    expect(graphs.length).toBeGreaterThan(0);

    graphs.forEach(graph => {
      expect(graph.technique_id).toBeDefined();
      expect(graph.technique_name).toBeDefined();
      expect(graph.datapoints).toBeInstanceOf(Array);
      expect(graph.datapoints.length).toBe(3); // 2023, 2024, 2025
    });
  });

  it('should track causality chains', () => {
    const analysis = analyzeTemporalPatterns(2025);

    expect(analysis.causality_chains.length).toBeGreaterThan(0);

    analysis.causality_chains.forEach(chain => {
      expect(chain.root_cause).toBeDefined();
      expect(chain.year).toBeGreaterThanOrEqual(2022);
      expect(chain.downstream_effects).toBeInstanceOf(Array);
    });
  });
});

describe('Cultural Filters', () => {
  it('should apply cultural adjustments', () => {
    const morphemes = parseMorphemes("That is wrong");
    const syntax = parseSyntax("That is wrong");
    const semantics = parseSemantics("That is wrong");

    const culturalContext = {
      culture: 'DE',
      language: 'en',
      communication_style: 'low-context' as const,
      translation_involved: false
    };

    const adjustment = applyCulturalFilters(
      morphemes,
      syntax,
      semantics,
      culturalContext,
      0.8
    );

    expect(adjustment.confidence_multiplier).toBeGreaterThanOrEqual(0.5);
    expect(adjustment.confidence_multiplier).toBeLessThanOrEqual(1.5);
    expect(adjustment.threshold_adjustment).toBeGreaterThanOrEqual(-0.2);
    expect(adjustment.threshold_adjustment).toBeLessThanOrEqual(0.2);
    expect(adjustment.warnings).toBeInstanceOf(Array);
    expect(adjustment.cultural_factors).toBeInstanceOf(Array);
  });

  it('should detect high-context culture patterns', () => {
    const morphemes = parseMorphemes("Maybe that is difficult to arrange");
    const syntax = parseSyntax("Maybe that is difficult to arrange");
    const semantics = parseSemantics("Maybe that is difficult to arrange");

    const japaneseContext = {
      culture: 'JP',
      language: 'en',
      communication_style: 'high-context' as const,
      translation_involved: false
    };

    const adjustment = applyCulturalFilters(
      morphemes,
      syntax,
      semantics,
      japaneseContext,
      0.8
    );

    // High-context cultures should have threshold adjustment
    expect(adjustment.threshold_adjustment).toBeGreaterThan(0);
    expect(adjustment.cultural_factors.length).toBeGreaterThan(0);
  });

  it('should detect translation artifacts', () => {
    const morphemes = parseMorphemes("Please do the needful");
    const syntax = parseSyntax("Please do the needful");
    const semantics = parseSemantics("Please do the needful");

    const translatedContext = {
      culture: 'IN',
      language: 'en',
      communication_style: 'mixed' as const,
      translation_involved: true
    };

    const adjustment = applyCulturalFilters(
      morphemes,
      syntax,
      semantics,
      translatedContext,
      0.8
    );

    expect(adjustment.confidence_multiplier).toBeLessThan(1.0);
    expect(adjustment.warnings.length).toBeGreaterThan(0);
  });

  it('should infer cultural context', () => {
    const morphemes = parseMorphemes("Bless your heart, that's interesting");
    const syntax = parseSyntax("Bless your heart, that's interesting");

    const inferredContext = inferCulturalContext(morphemes, syntax, 'en');

    expect(inferredContext.culture).toBeDefined();
    expect(inferredContext.language).toBe('en');
    expect(inferredContext.communication_style).toBeDefined();
    expect(inferredContext.translation_involved).toBeDefined();
  });

  it('should provide cultural recommendations', () => {
    const recommendation = getRecommendedAdjustment('BR');

    expect(recommendation.confidence_multiplier).toBeDefined();
    expect(recommendation.threshold_increase).toBeDefined();
    expect(recommendation.reasoning).toBeDefined();
    expect(recommendation.reasoning.length).toBeGreaterThan(0);
  });

  it('should handle unknown cultures gracefully', () => {
    const recommendation = getRecommendedAdjustment('UNKNOWN');

    expect(recommendation.confidence_multiplier).toBeDefined();
    expect(recommendation.threshold_increase).toBeDefined();
  });
});

describe('Integration Tests', () => {
  it('should complete full analysis pipeline', () => {
    const text = "That never happened";

    // Parse
    const morphemes = parseMorphemes(text);
    const syntax = parseSyntax(text);
    const semantics = parseSemantics(text);
    const pragmatics = parsePragmatics(morphemes, syntax, semantics);

    // Intent
    const context = {
      relationship_type: 'intimate' as const,
      power_dynamic: 'superior' as const,
      history_of_manipulation: true,
      previous_detections: [],
      conversation_length: 5
    };

    const analysis = analyzeWithContext(morphemes, syntax, semantics, pragmatics, context);

    // Cultural
    const culturalContext = {
      culture: 'US',
      language: 'en',
      communication_style: 'low-context' as const,
      translation_involved: false
    };

    const culturalAdjustment = applyCulturalFilters(
      morphemes,
      syntax,
      semantics,
      culturalContext,
      analysis.intent.confidence
    );

    expect(analysis).toBeDefined();
    expect(culturalAdjustment).toBeDefined();
    expect(analysis.intent.primary_intent).toBeDefined();
    expect(culturalAdjustment.confidence_multiplier).toBeDefined();
  });
});
