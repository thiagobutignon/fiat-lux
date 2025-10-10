/**
 * Temporal Causality Tracker
 * Sprint 2: Analysis Layer
 *
 * Tracks evolution of manipulation techniques from 2023 → 2025
 * Focus on GPT-5 era techniques (153-180) with temporal metadata
 *
 * Features:
 * - Technique prevalence over time
 * - Causality chain analysis (ChatGPT → public access → abuse patterns)
 * - Emergence prediction (when new techniques will appear)
 * - Evolution graphs (visualization data)
 */

import { ManipulationTechnique, TechniqueEra } from '../types';
import { getAllTechniques, getTechniquesByEra } from '../techniques';

// ============================================================
// TYPES
// ============================================================

export interface TemporalData {
  technique_id: number;
  technique_name: string;
  emerged_year: number;
  prevalence_2023: number;
  prevalence_2024: number;
  prevalence_2025: number;
  growth_rate: number;  // Annual growth percentage
  causality_chain: string[];
  peak_year?: number;
  decline_predicted?: boolean;
}

export interface TemporalAnalysis {
  current_year: number;
  total_techniques_active: number;
  gpt4_era_active: number;
  gpt5_era_active: number;
  emerging_techniques: TemporalData[];  // Appeared in last year
  declining_techniques: TemporalData[];  // Decreasing prevalence
  peak_techniques: TemporalData[];  // At maximum prevalence
  causality_chains: CausalityChain[];
}

export interface CausalityChain {
  root_cause: string;
  year: number;
  downstream_effects: {
    year: number;
    effect: string;
    techniques_enabled: number[];
  }[];
}

export interface EvolutionGraph {
  technique_id: number;
  technique_name: string;
  datapoints: {
    year: number;
    prevalence: number;
    confidence: number;
  }[];
}

// ============================================================
// TEMPORAL TRACKING
// ============================================================

/**
 * Extract temporal data from all techniques
 */
export function extractTemporalData(): TemporalData[] {
  const techniques = getAllTechniques();
  const temporalData: TemporalData[] = [];

  for (const technique of techniques) {
    // Only GPT-5 era techniques have temporal evolution data
    if (technique.era === TechniqueEra.GPT5 && technique.temporal_evolution) {
      const te = technique.temporal_evolution;

      // Calculate growth rate (2023 → 2025)
      const growthRate = te.prevalence_2023 > 0
        ? ((te.prevalence_2025 - te.prevalence_2023) / te.prevalence_2023) * 100
        : 0;

      // Determine peak year
      const prevalences = [
        { year: 2023, value: te.prevalence_2023 },
        { year: 2024, value: te.prevalence_2024 },
        { year: 2025, value: te.prevalence_2025 }
      ];
      const peak = prevalences.reduce((max, p) => p.value > max.value ? p : max, prevalences[0]);

      // Predict decline (if 2025 < 2024)
      const declinePredicted = te.prevalence_2025 < te.prevalence_2024;

      temporalData.push({
        technique_id: technique.id,
        technique_name: technique.name,
        emerged_year: te.emerged_year,
        prevalence_2023: te.prevalence_2023,
        prevalence_2024: te.prevalence_2024,
        prevalence_2025: te.prevalence_2025,
        growth_rate: growthRate,
        causality_chain: te.causality_chain,
        peak_year: peak.year,
        decline_predicted: declinePredicted
      });
    }
  }

  return temporalData;
}

/**
 * Analyze temporal patterns across all techniques
 */
export function analyzeTemporalPatterns(currentYear: number = 2025): TemporalAnalysis {
  const temporalData = extractTemporalData();
  const allTechniques = getAllTechniques();

  // Count active techniques by era
  const gpt4Active = allTechniques.filter(t => t.era === TechniqueEra.GPT4).length;
  const gpt5Active = allTechniques.filter(t => t.era === TechniqueEra.GPT5).length;

  // Emerging techniques (appeared in last year)
  const emerging = temporalData.filter(t => t.emerged_year >= currentYear - 1);

  // Declining techniques
  const declining = temporalData.filter(t => t.decline_predicted);

  // Peak techniques (at maximum prevalence this year)
  const peak = temporalData.filter(t => t.peak_year === currentYear);

  // Build causality chains
  const causalityChains = buildCausalityChains(temporalData);

  return {
    current_year: currentYear,
    total_techniques_active: allTechniques.length,
    gpt4_era_active: gpt4Active,
    gpt5_era_active: gpt5Active,
    emerging_techniques: emerging,
    declining_techniques: declining,
    peak_techniques: peak,
    causality_chains: causalityChains
  };
}

/**
 * Build causality chains from temporal data
 * Maps root causes (ChatGPT release, etc.) to downstream effects
 */
function buildCausalityChains(temporalData: TemporalData[]): CausalityChain[] {
  const chains: Map<string, CausalityChain> = new Map();

  for (const data of temporalData) {
    // Extract root cause (first element in chain)
    if (data.causality_chain.length === 0) continue;

    const rootCause = data.causality_chain[0];
    const rootYear = data.emerged_year;

    if (!chains.has(rootCause)) {
      chains.set(rootCause, {
        root_cause: rootCause,
        year: rootYear,
        downstream_effects: []
      });
    }

    const chain = chains.get(rootCause)!;

    // Add downstream effects (rest of chain)
    for (let i = 1; i < data.causality_chain.length; i++) {
      const effect = data.causality_chain[i];
      const effectYear = rootYear + i;  // Approximate: each step = 1 year

      // Find existing effect or create new one
      let existingEffect = chain.downstream_effects.find(
        e => e.effect === effect && e.year === effectYear
      );

      if (!existingEffect) {
        existingEffect = {
          year: effectYear,
          effect,
          techniques_enabled: []
        };
        chain.downstream_effects.push(existingEffect);
      }

      existingEffect.techniques_enabled.push(data.technique_id);
    }
  }

  return Array.from(chains.values());
}

/**
 * Generate evolution graphs for visualization
 * Returns datapoints for charting technique prevalence over time
 */
export function generateEvolutionGraphs(): EvolutionGraph[] {
  const temporalData = extractTemporalData();
  const graphs: EvolutionGraph[] = [];

  for (const data of temporalData) {
    graphs.push({
      technique_id: data.technique_id,
      technique_name: data.technique_name,
      datapoints: [
        { year: 2023, prevalence: data.prevalence_2023, confidence: 0.9 },
        { year: 2024, prevalence: data.prevalence_2024, confidence: 0.95 },
        { year: 2025, prevalence: data.prevalence_2025, confidence: 0.85 }  // Lower confidence = prediction
      ]
    });
  }

  return graphs;
}

/**
 * Predict future prevalence (2026+)
 * Uses linear extrapolation from 2023-2025 trend
 */
export function predictFuturePrevalence(
  techniqueId: number,
  targetYear: number
): { prevalence: number; confidence: number } | null {
  const temporalData = extractTemporalData();
  const data = temporalData.find(t => t.technique_id === techniqueId);

  if (!data) {
    return null;
  }

  // Calculate linear trend from existing data
  const years = [2023, 2024, 2025];
  const prevalences = [data.prevalence_2023, data.prevalence_2024, data.prevalence_2025];

  // Simple linear regression: y = mx + b
  const n = years.length;
  const sumX = years.reduce((a, b) => a + b, 0);
  const sumY = prevalences.reduce((a, b) => a + b, 0);
  const sumXY = years.reduce((sum, year, i) => sum + year * prevalences[i], 0);
  const sumX2 = years.reduce((sum, year) => sum + year * year, 0);

  const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
  const intercept = (sumY - slope * sumX) / n;

  // Predict prevalence
  const predictedPrevalence = slope * targetYear + intercept;

  // Confidence decreases with distance from known data
  const yearDistance = targetYear - 2025;
  const confidence = Math.max(0.3, 0.85 - (yearDistance * 0.1));

  return {
    prevalence: Math.max(0, Math.min(1, predictedPrevalence)),
    confidence
  };
}

/**
 * Get techniques by prevalence threshold in a given year
 */
export function getTechniquesByPrevalence(
  year: number,
  minPrevalence: number
): TemporalData[] {
  const temporalData = extractTemporalData();

  return temporalData.filter(data => {
    let prevalence = 0;

    if (year === 2023) {
      prevalence = data.prevalence_2023;
    } else if (year === 2024) {
      prevalence = data.prevalence_2024;
    } else if (year === 2025) {
      prevalence = data.prevalence_2025;
    } else {
      // Predict for future years
      const prediction = predictFuturePrevalence(data.technique_id, year);
      prevalence = prediction ? prediction.prevalence : 0;
    }

    return prevalence >= minPrevalence;
  });
}

/**
 * Analyze causality strength between events
 * Returns correlation score (0-1) between root cause and technique emergence
 */
export function analyzeCausalityStrength(rootCause: string): {
  root_cause: string;
  total_techniques_enabled: number;
  average_emergence_lag_years: number;
  strength_score: number;
} {
  const temporalData = extractTemporalData();

  // Find all techniques with this root cause
  const affectedTechniques = temporalData.filter(
    t => t.causality_chain.length > 0 && t.causality_chain[0] === rootCause
  );

  if (affectedTechniques.length === 0) {
    return {
      root_cause: rootCause,
      total_techniques_enabled: 0,
      average_emergence_lag_years: 0,
      strength_score: 0
    };
  }

  // Calculate average emergence lag
  // Assuming root causes from 2022 (ChatGPT) or 2023
  const rootYear = rootCause.toLowerCase().includes('chatgpt') ? 2022 : 2023;
  const emergenceLags = affectedTechniques.map(t => t.emerged_year - rootYear);
  const avgLag = emergenceLags.reduce((a, b) => a + b, 0) / emergenceLags.length;

  // Strength score based on:
  // - Number of techniques enabled (more = stronger)
  // - Average prevalence growth (faster = stronger)
  // - Lag time (shorter = stronger)
  const avgGrowthRate = affectedTechniques.reduce((sum, t) => sum + t.growth_rate, 0) / affectedTechniques.length;

  const strengthScore = Math.min(1.0,
    (affectedTechniques.length / 10) * 0.4 +  // Quantity factor
    (avgGrowthRate / 500) * 0.3 +              // Growth factor
    (1 / (avgLag + 1)) * 0.3                    // Immediacy factor
  );

  return {
    root_cause: rootCause,
    total_techniques_enabled: affectedTechniques.length,
    average_emergence_lag_years: avgLag,
    strength_score: strengthScore
  };
}

/**
 * Generate temporal summary report
 */
export function generateTemporalReport(year: number = 2025): string {
  const analysis = analyzeTemporalPatterns(year);
  const lines: string[] = [];

  lines.push(`🕐 TEMPORAL ANALYSIS REPORT - ${year}`);
  lines.push('');
  lines.push(`Total Techniques Active: ${analysis.total_techniques_active}`);
  lines.push(`  - GPT-4 Era (Classical): ${analysis.gpt4_era_active}`);
  lines.push(`  - GPT-5 Era (Emergent): ${analysis.gpt5_era_active}`);
  lines.push('');

  if (analysis.emerging_techniques.length > 0) {
    lines.push(`📈 Emerging Techniques (${analysis.emerging_techniques.length}):`);
    for (const tech of analysis.emerging_techniques.slice(0, 5)) {
      lines.push(`  - ${tech.technique_name} (${tech.emerged_year})`);
      lines.push(`    Growth: ${tech.growth_rate.toFixed(0)}%`);
    }
    lines.push('');
  }

  if (analysis.peak_techniques.length > 0) {
    lines.push(`⚡ Peak Techniques (${analysis.peak_techniques.length}):`);
    for (const tech of analysis.peak_techniques.slice(0, 5)) {
      const prevalence = year === 2023 ? tech.prevalence_2023 :
                         year === 2024 ? tech.prevalence_2024 :
                         tech.prevalence_2025;
      lines.push(`  - ${tech.technique_name}: ${(prevalence * 100).toFixed(0)}% prevalence`);
    }
    lines.push('');
  }

  if (analysis.declining_techniques.length > 0) {
    lines.push(`📉 Declining Techniques (${analysis.declining_techniques.length}):`);
    for (const tech of analysis.declining_techniques.slice(0, 5)) {
      lines.push(`  - ${tech.technique_name}`);
    }
    lines.push('');
  }

  lines.push(`🔗 Causality Chains (${analysis.causality_chains.length}):`);
  for (const chain of analysis.causality_chains.slice(0, 3)) {
    lines.push(`  ${chain.root_cause} (${chain.year})`);
    const totalEnabled = chain.downstream_effects.reduce(
      (sum, e) => sum + e.techniques_enabled.length,
      0
    );
    lines.push(`    → Enabled ${totalEnabled} techniques`);
    lines.push(`    → ${chain.downstream_effects.length} downstream effects`);
  }

  return lines.join('\n');
}

/**
 * Get techniques that share a causality path
 * Useful for understanding technique clusters
 */
export function getTechniquesBySharedCausality(causalityKeyword: string): TemporalData[] {
  const temporalData = extractTemporalData();

  return temporalData.filter(t =>
    t.causality_chain.some(step => step.toLowerCase().includes(causalityKeyword.toLowerCase()))
  );
}

/**
 * Calculate temporal similarity between two techniques
 * Returns 0-1 score based on shared causality and emergence timing
 */
export function calculateTemporalSimilarity(techniqueId1: number, techniqueId2: number): number {
  const temporalData = extractTemporalData();
  const tech1 = temporalData.find(t => t.technique_id === techniqueId1);
  const tech2 = temporalData.find(t => t.technique_id === techniqueId2);

  if (!tech1 || !tech2) {
    return 0;
  }

  // Similarity factors:
  // 1. Emerged in same year
  const sameYear = tech1.emerged_year === tech2.emerged_year ? 0.3 : 0;

  // 2. Shared causality steps
  const sharedSteps = tech1.causality_chain.filter(step =>
    tech2.causality_chain.includes(step)
  ).length;
  const maxSteps = Math.max(tech1.causality_chain.length, tech2.causality_chain.length);
  const causalitySimilarity = maxSteps > 0 ? (sharedSteps / maxSteps) * 0.4 : 0;

  // 3. Similar growth rate
  const growthDifference = Math.abs(tech1.growth_rate - tech2.growth_rate);
  const growthSimilarity = Math.max(0, (100 - growthDifference) / 100) * 0.3;

  return sameYear + causalitySimilarity + growthSimilarity;
}
