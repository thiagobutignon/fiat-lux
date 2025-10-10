/**
 * Pattern Matcher - O(1) Detection Engine
 * Combines all linguistic parsers to detect manipulation techniques
 * Uses hash-based lookups for constant-time detection
 */

import { parsePhonemes, calculatePhonemeScore } from '../parser/phonemes';
import { parseMorphemes, calculateMorphemeScore, hasNeurodivergentMarkers } from '../parser/morphemes';
import { parseSyntax, calculateSyntaxScore } from '../parser/syntax';
import { parseSemantics, calculateSemanticsScore } from '../parser/semantics';
import { parsePragmatics, calculatePragmaticsScore } from '../parser/pragmatics';

import { getAllTechniques, getTechniqueById } from '../techniques';

import {
  DetectionResult,
  PatternMatchConfig,
  PatternMatchResult,
  ManipulationTechnique,
  DarkTetradScores,
  ConstitutionalValidation
} from '../types';

// ============================================================
// PATTERN MATCHING ENGINE
// ============================================================

/**
 * Detect manipulation techniques in text
 * O(1) per technique check via hash-based lookups
 */
export async function detectManipulation(
  text: string,
  config: PatternMatchConfig = {}
): Promise<PatternMatchResult> {
  const startTime = Date.now();

  // Parse linguistic structure (complete Chomsky hierarchy)
  const phonemes = parsePhonemes(text);
  const morphemes = parseMorphemes(text);
  const syntax = parseSyntax(text);
  const semantics = parseSemantics(text);
  const pragmatics = parsePragmatics(morphemes, syntax, semantics);

  // Get techniques to check
  const techniquesToCheck = config.technique_ids
    ? config.technique_ids.map(id => getTechniqueById(id)!).filter(Boolean)
    : getAllTechniques();

  // Filter by category if specified
  const filteredTechniques = config.categories
    ? techniquesToCheck.filter(t => config.categories!.includes(t.category))
    : techniquesToCheck;

  // Detect matches
  const detections: DetectionResult[] = [];
  const minConfidence = config.min_confidence ?? 0.8;

  for (const technique of filteredTechniques) {
    const result = matchTechnique(
      text,
      technique,
      { phonemes, morphemes, syntax, semantics, pragmatics },
      config
    );

    if (result && result.confidence >= minConfidence) {
      detections.push(result);
    }
  }

  // Sort by confidence (descending)
  detections.sort((a, b) => b.confidence - a.confidence);

  // Calculate aggregate Dark Tetrad scores
  const darkTetradAggregate = calculateAggregateDarkTetrad(detections);

  // Build attention trace
  const attentionTrace = {
    sources: [text],
    weights: detections.map(d => d.confidence),
    patterns: detections.map(d => d.technique_name)
  };

  // Constitutional validation
  const constitutionalValidation = validateConstitutional(
    detections,
    morphemes,
    config
  );

  const processingTime = Date.now() - startTime;

  return {
    detections,
    total_matches: detections.length,
    highest_confidence: detections.length > 0 ? detections[0].confidence : 0,
    dark_tetrad_aggregate: darkTetradAggregate,
    attention_trace: attentionTrace,
    constitutional_validation: constitutionalValidation,
    processing_time_ms: processingTime
  };
}

/**
 * Match a single technique against parsed linguistic structure
 * O(1) - Direct comparison of linguistic patterns
 */
function matchTechnique(
  text: string,
  technique: ManipulationTechnique,
  parsed: {
    phonemes: any;
    morphemes: any;
    syntax: any;
    semantics: any;
    pragmatics: any;
  },
  config: PatternMatchConfig
): DetectionResult | null {
  // Calculate component scores (complete Chomsky hierarchy)
  const phonemeScore = calculatePhonemeScore(parsed.phonemes, technique.phonemes);
  const morphemeScore = calculateMorphemeScore(parsed.morphemes, technique.morphemes);
  const syntaxScore = calculateSyntaxScore(parsed.syntax, technique.syntax);
  const semanticsScore = calculateSemanticsScore(parsed.semantics, technique.semantics);
  const pragmaticsScore = calculatePragmaticsScore(parsed.pragmatics, technique.pragmatics);

  // Weighted confidence calculation (5 layers)
  const weights = {
    phonemes: 0.15,   // Tone/rhythm indicators
    morphemes: 0.25,  // Keywords important
    syntax: 0.15,     // Structure
    semantics: 0.25,  // Meaning critical
    pragmatics: 0.20  // Intent critical
  };

  const confidence =
    phonemeScore * weights.phonemes +
    morphemeScore * weights.morphemes +
    syntaxScore * weights.syntax +
    semanticsScore * weights.semantics +
    pragmaticsScore * weights.pragmatics;

  // Check if meets threshold
  if (confidence < technique.confidence_threshold) {
    return null;
  }

  // Neurodivergent protection
  const neurodivergentFlag =
    config.enable_neurodivergent_protection !== false &&
    hasNeurodivergentMarkers(parsed.morphemes);

  // If neurodivergent markers present, increase threshold
  const adjustedThreshold = neurodivergentFlag
    ? technique.confidence_threshold + 0.15
    : technique.confidence_threshold;

  if (confidence < adjustedThreshold) {
    return null;
  }

  // Build detection result
  return {
    technique_id: technique.id,
    technique_name: technique.name,
    confidence,
    matched_markers: {
      phonemes: parsed.phonemes,
      morphemes: parsed.morphemes,
      syntax: parsed.syntax,
      semantics: parsed.semantics,
      pragmatics: parsed.pragmatics
    },
    dark_tetrad: technique.dark_tetrad,
    explanation: generateExplanation(technique, {
      phonemeScore,
      morphemeScore,
      syntaxScore,
      semanticsScore,
      pragmaticsScore,
      confidence
    }),
    sources: extractMatchedSources(text, parsed.morphemes),
    neurodivergent_flag: neurodivergentFlag,
    validated: true  // Will be validated constitutionally
  };
}

/**
 * Generate human-readable explanation of detection
 * Glass box transparency
 */
function generateExplanation(
  technique: ManipulationTechnique,
  scores: {
    phonemeScore: number;
    morphemeScore: number;
    syntaxScore: number;
    semanticsScore: number;
    pragmaticsScore: number;
    confidence: number;
  }
): string {
  const parts: string[] = [];

  parts.push(`Detected: ${technique.name} (${technique.category})`);
  parts.push(`Overall confidence: ${(scores.confidence * 100).toFixed(1)}%`);
  parts.push('');
  parts.push('Linguistic Analysis (Chomsky Hierarchy):');

  if (scores.phonemeScore > 0.5) {
    parts.push(`  ✓ Phonemes matched: ${(scores.phonemeScore * 100).toFixed(0)}%`);
  }
  if (scores.morphemeScore > 0.5) {
    parts.push(`  ✓ Morphemes matched: ${(scores.morphemeScore * 100).toFixed(0)}%`);
  }
  if (scores.syntaxScore > 0.5) {
    parts.push(`  ✓ Syntax patterns matched: ${(scores.syntaxScore * 100).toFixed(0)}%`);
  }
  if (scores.semanticsScore > 0.5) {
    parts.push(`  ✓ Semantic meaning matched: ${(scores.semanticsScore * 100).toFixed(0)}%`);
  }
  if (scores.pragmaticsScore > 0.5) {
    parts.push(`  ✓ Intent/pragmatics matched: ${(scores.pragmaticsScore * 100).toFixed(0)}%`);
  }

  parts.push('');
  parts.push('Dark Tetrad Profile:');
  parts.push(`  Narcissism: ${(technique.dark_tetrad.narcissism * 100).toFixed(0)}%`);
  parts.push(`  Machiavellianism: ${(technique.dark_tetrad.machiavellianism * 100).toFixed(0)}%`);
  parts.push(`  Psychopathy: ${(technique.dark_tetrad.psychopathy * 100).toFixed(0)}%`);
  parts.push(`  Sadism: ${(technique.dark_tetrad.sadism * 100).toFixed(0)}%`);

  return parts.join('\n');
}

/**
 * Extract matched text sources
 * Shows which parts of text triggered detection
 */
function extractMatchedSources(text: string, morphemes: any): string[] {
  const sources: string[] = [];

  // Add matched keywords
  for (const keyword of morphemes.keywords) {
    const regex = new RegExp(keyword, 'gi');
    const match = text.match(regex);
    if (match) {
      sources.push(`"${match[0]}" (keyword)`);
    }
  }

  return sources.slice(0, 5);  // Limit to top 5 sources
}

/**
 * Calculate aggregate Dark Tetrad scores from multiple detections
 */
function calculateAggregateDarkTetrad(detections: DetectionResult[]): DarkTetradScores {
  if (detections.length === 0) {
    return { narcissism: 0, machiavellianism: 0, psychopathy: 0, sadism: 0 };
  }

  // Weighted average by confidence
  let totalWeight = 0;
  const aggregate = { narcissism: 0, machiavellianism: 0, psychopathy: 0, sadism: 0 };

  for (const detection of detections) {
    const weight = detection.confidence;
    totalWeight += weight;

    aggregate.narcissism += detection.dark_tetrad.narcissism * weight;
    aggregate.machiavellianism += detection.dark_tetrad.machiavellianism * weight;
    aggregate.psychopathy += detection.dark_tetrad.psychopathy * weight;
    aggregate.sadism += detection.dark_tetrad.sadism * weight;
  }

  if (totalWeight > 0) {
    aggregate.narcissism /= totalWeight;
    aggregate.machiavellianism /= totalWeight;
    aggregate.psychopathy /= totalWeight;
    aggregate.sadism /= totalWeight;
  }

  return aggregate;
}

/**
 * Constitutional validation
 * Ensures detections comply with ethical principles
 */
function validateConstitutional(
  detections: DetectionResult[],
  morphemes: any,
  config: PatternMatchConfig
): ConstitutionalValidation {
  const violations: string[] = [];
  const warnings: string[] = [];
  let adjustedConfidence = 1.0;

  // Privacy check - no personal data should be stored
  // (We don't store the text, only patterns)

  // Neurodivergent protection
  if (hasNeurodivergentMarkers(morphemes)) {
    warnings.push('Neurodivergent communication markers detected - confidence threshold increased');
    adjustedConfidence *= 0.85;  // Reduce confidence
  }

  // Context awareness - require context if specified
  if (config.context && detections.length > 0) {
    warnings.push('Context provided - detections are context-aware');
  }

  // Evidence-based - all detections must cite sources
  for (const detection of detections) {
    if (detection.sources.length === 0) {
      violations.push(`Detection ${detection.technique_name} lacks evidence sources`);
    }
  }

  // Accuracy check - if false positive risk high, warn
  for (const detection of detections) {
    const technique = getTechniqueById(detection.technique_id);
    if (technique && technique.false_positive_risk > 0.3) {
      warnings.push(`${technique.name} has high false-positive risk (${(technique.false_positive_risk * 100).toFixed(0)}%)`);
    }
  }

  return {
    compliant: violations.length === 0,
    violations,
    warnings,
    adjusted_confidence: adjustedConfidence
  };
}

// ============================================================
// CONVENIENCE FUNCTIONS
// ============================================================

/**
 * Quick detection - returns true/false if manipulation detected
 */
export async function isManipulative(text: string, minConfidence = 0.8): Promise<boolean> {
  const result = await detectManipulation(text, { min_confidence: minConfidence });
  return result.total_matches > 0;
}

/**
 * Get top detection
 */
export async function getTopDetection(text: string): Promise<DetectionResult | null> {
  const result = await detectManipulation(text);
  return result.detections.length > 0 ? result.detections[0] : null;
}

/**
 * Get Dark Tetrad profile from text
 */
export async function getDarkTetradProfile(text: string): Promise<DarkTetradScores> {
  const result = await detectManipulation(text);
  return result.dark_tetrad_aggregate;
}
