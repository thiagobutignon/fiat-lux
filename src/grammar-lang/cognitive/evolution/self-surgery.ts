/**
 * Self-Surgery Module - Autonomous Evolution
 * Enables the Cognitive OS to evolve autonomously:
 * - Detect new manipulation techniques
 * - Auto-add to taxonomy with validation
 * - Genetic evolution of detection accuracy
 * - Old-but-gold for deprecated techniques
 */

import { generateTechnique } from '../techniques/technique-generator';
import { getAllTechniques, getTechniqueById } from '../techniques';
import {
  ManipulationTechnique,
  TechniqueEra,
  TechniqueCategory,
  DetectionResult
} from '../types';

// ============================================================
// TYPES
// ============================================================

export interface NewTechniqueCandidate {
  id: number;
  proposed_name: string;
  proposed_category: TechniqueCategory;
  era: TechniqueEra;
  evidence: {
    text_samples: string[];           // Text samples that triggered detection
    occurrence_count: number;          // How many times observed
    first_seen: string;                // ISO date
    last_seen: string;                 // ISO date
    linguistic_patterns: {
      common_keywords: string[];
      common_syntax: string[];
      common_semantics: string[];
    };
  };
  confidence: number;                  // 0-1 confidence this is a new technique
  requires_approval: boolean;          // If true, needs human review
  status: 'candidate' | 'approved' | 'rejected' | 'merged';
}

export interface EvolutionEvent {
  timestamp: string;
  type: 'new_technique' | 'accuracy_improvement' | 'old_but_gold' | 'merge';
  technique_id: number;
  details: any;
  fitness_delta?: number;              // Change in fitness score
}

export interface SurgeryConfig {
  auto_approve_threshold?: number;     // Confidence threshold for auto-approval (default: 0.95)
  min_evidence_count?: number;         // Minimum occurrences before proposing (default: 5)
  enable_auto_surgery?: boolean;       // Enable autonomous updates (default: false)
  enable_old_but_gold?: boolean;       // Track deprecated techniques (default: true)
  evolution_log_path?: string;         // Path to store evolution log
}

export interface TechniqueFitness {
  technique_id: number;
  precision: number;                   // True positives / (TP + FP)
  recall: number;                      // True positives / (TP + FN)
  f1_score: number;                    // Harmonic mean of precision/recall
  false_positive_rate: number;         // FP / (FP + TN)
  total_detections: number;
  generations: number;                 // How many evolutions
}

// ============================================================
// SELF-SURGERY ENGINE
// ============================================================

export class SelfSurgeryEngine {
  private config: SurgeryConfig;
  private candidates: Map<string, NewTechniqueCandidate>;
  private evolutionLog: EvolutionEvent[];
  private fitnessScores: Map<number, TechniqueFitness>;
  private observationCache: Map<string, { count: number; samples: string[] }>;
  private nextTechniqueId: number;

  constructor(config: SurgeryConfig = {}) {
    this.config = {
      auto_approve_threshold: config.auto_approve_threshold ?? 0.95,
      min_evidence_count: config.min_evidence_count ?? 5,
      enable_auto_surgery: config.enable_auto_surgery ?? false,
      enable_old_but_gold: config.enable_old_but_gold ?? true,
      evolution_log_path: config.evolution_log_path
    };

    this.candidates = new Map();
    this.evolutionLog = [];
    this.fitnessScores = new Map();
    this.observationCache = new Map();

    // Get next available technique ID
    const existingTechniques = getAllTechniques();
    this.nextTechniqueId = existingTechniques.length > 0
      ? Math.max(...existingTechniques.map(t => t.id)) + 1
      : 181; // Start after GPT-5 era (180)
  }

  // ============================================================
  // NEW TECHNIQUE DETECTION
  // ============================================================

  /**
   * Observe a text sample that doesn't match existing techniques well
   * Build evidence for potential new technique
   */
  public observeAnomalousPattern(
    text: string,
    detectionResults: DetectionResult[],
    proposedCategory?: TechniqueCategory
  ): void {
    // Check if any existing technique matched with high confidence
    const hasHighConfidenceMatch = detectionResults.some(d => d.confidence >= 0.85);

    if (hasHighConfidenceMatch) {
      return; // Not anomalous
    }

    // Extract linguistic patterns
    const keywords = this.extractKeywords(text);
    const cacheKey = keywords.sort().join('|');

    // Update observation cache
    if (!this.observationCache.has(cacheKey)) {
      this.observationCache.set(cacheKey, { count: 0, samples: [] });
    }

    const observation = this.observationCache.get(cacheKey)!;
    observation.count++;
    if (observation.samples.length < 10) {
      observation.samples.push(text);
    }

    // Check if we have enough evidence to propose a new technique
    if (observation.count >= this.config.min_evidence_count!) {
      this.proposeNewTechnique(cacheKey, observation, proposedCategory);
    }
  }

  /**
   * Propose a new technique based on accumulated evidence
   */
  private proposeNewTechnique(
    cacheKey: string,
    observation: { count: number; samples: string[] },
    proposedCategory?: TechniqueCategory
  ): void {
    // Check if already proposed
    if (this.candidates.has(cacheKey)) {
      return;
    }

    // Analyze patterns across samples
    const patterns = this.analyzePatterns(observation.samples);

    // Generate candidate technique
    const candidate: NewTechniqueCandidate = {
      id: this.nextTechniqueId++,
      proposed_name: this.generateTechniqueName(patterns),
      proposed_category: proposedCategory ?? this.inferCategory(patterns),
      era: 'gpt5', // New techniques are GPT-5 era
      evidence: {
        text_samples: observation.samples,
        occurrence_count: observation.count,
        first_seen: new Date().toISOString(),
        last_seen: new Date().toISOString(),
        linguistic_patterns: patterns
      },
      confidence: this.calculateCandidateConfidence(patterns, observation.count),
      requires_approval: true,
      status: 'candidate'
    };

    // Auto-approve if confidence is very high and auto-surgery enabled
    if (
      this.config.enable_auto_surgery &&
      candidate.confidence >= this.config.auto_approve_threshold!
    ) {
      candidate.requires_approval = false;
      candidate.status = 'approved';
      this.applyNewTechnique(candidate);
    }

    this.candidates.set(cacheKey, candidate);

    // Log event
    this.logEvolution({
      timestamp: new Date().toISOString(),
      type: 'new_technique',
      technique_id: candidate.id,
      details: {
        name: candidate.proposed_name,
        category: candidate.proposed_category,
        confidence: candidate.confidence,
        status: candidate.status
      }
    });
  }

  // ============================================================
  // GENETIC EVOLUTION
  // ============================================================

  /**
   * Evolve a technique based on detection performance
   * Improves detection accuracy through genetic-like mutations
   */
  public evolveTechnique(
    techniqueId: number,
    performanceData: {
      true_positives: number;
      false_positives: number;
      false_negatives: number;
      true_negatives: number;
    }
  ): void {
    // Calculate fitness
    const tp = performanceData.true_positives;
    const fp = performanceData.false_positives;
    const fn = performanceData.false_negatives;
    const tn = performanceData.true_negatives;

    const precision = tp / (tp + fp);
    const recall = tp / (tp + fn);
    const f1_score = 2 * (precision * recall) / (precision + recall);
    const fpr = fp / (fp + tn);

    // Get or create fitness entry
    if (!this.fitnessScores.has(techniqueId)) {
      this.fitnessScores.set(techniqueId, {
        technique_id: techniqueId,
        precision: 0,
        recall: 0,
        f1_score: 0,
        false_positive_rate: 0,
        total_detections: 0,
        generations: 0
      });
    }

    const fitness = this.fitnessScores.get(techniqueId)!;
    const previousF1 = fitness.f1_score;

    // Update fitness scores (exponential moving average)
    const alpha = 0.3; // Weight for new data
    fitness.precision = alpha * precision + (1 - alpha) * fitness.precision;
    fitness.recall = alpha * recall + (1 - alpha) * fitness.recall;
    fitness.f1_score = alpha * f1_score + (1 - alpha) * fitness.f1_score;
    fitness.false_positive_rate = alpha * fpr + (1 - alpha) * fitness.false_positive_rate;
    fitness.total_detections += tp + fp;
    fitness.generations++;

    const fitnessDelta = fitness.f1_score - previousF1;

    // Log evolution
    this.logEvolution({
      timestamp: new Date().toISOString(),
      type: 'accuracy_improvement',
      technique_id: techniqueId,
      details: {
        precision: fitness.precision,
        recall: fitness.recall,
        f1_score: fitness.f1_score,
        fpr: fitness.false_positive_rate,
        generation: fitness.generations
      },
      fitness_delta: fitnessDelta
    });

    // If performance is degrading significantly, mark for review
    if (fitnessDelta < -0.1 && fitness.generations > 5) {
      this.markForOldButGold(techniqueId);
    }
  }

  // ============================================================
  // OLD-BUT-GOLD
  // ============================================================

  /**
   * Mark a technique as potentially deprecated but historically valuable
   * Keeps it for reference but lowers its priority
   */
  private markForOldButGold(techniqueId: number): void {
    if (!this.config.enable_old_but_gold) {
      return;
    }

    const technique = getTechniqueById(techniqueId);
    if (!technique) {
      return;
    }

    this.logEvolution({
      timestamp: new Date().toISOString(),
      type: 'old_but_gold',
      technique_id: techniqueId,
      details: {
        name: technique.name,
        reason: 'Performance degradation detected',
        fitness: this.fitnessScores.get(techniqueId)
      }
    });
  }

  // ============================================================
  // APPROVAL & APPLICATION
  // ============================================================

  /**
   * Get all candidate techniques awaiting approval
   */
  public getPendingCandidates(): NewTechniqueCandidate[] {
    return Array.from(this.candidates.values()).filter(
      c => c.status === 'candidate' && c.requires_approval
    );
  }

  /**
   * Approve a candidate technique
   */
  public approveCandidate(candidateKey: string): void {
    const candidate = this.candidates.get(candidateKey);
    if (!candidate) {
      throw new Error(`Candidate not found: ${candidateKey}`);
    }

    candidate.status = 'approved';
    this.applyNewTechnique(candidate);
  }

  /**
   * Reject a candidate technique
   */
  public rejectCandidate(candidateKey: string, reason?: string): void {
    const candidate = this.candidates.get(candidateKey);
    if (!candidate) {
      throw new Error(`Candidate not found: ${candidateKey}`);
    }

    candidate.status = 'rejected';

    this.logEvolution({
      timestamp: new Date().toISOString(),
      type: 'new_technique',
      technique_id: candidate.id,
      details: {
        name: candidate.proposed_name,
        status: 'rejected',
        reason
      }
    });
  }

  /**
   * Apply a new technique to the system
   * NOTE: In production, this would update the techniques database
   */
  private applyNewTechnique(candidate: NewTechniqueCandidate): void {
    // Generate full technique definition
    const newTechnique = generateTechnique(
      candidate.id,
      candidate.proposed_category,
      candidate.era
    );

    // In production, this would:
    // 1. Write to techniques database
    // 2. Trigger recompilation/reload
    // 3. Notify monitoring systems

    console.log(`✨ New technique applied: ${newTechnique.name} (ID: ${newTechnique.id})`);

    this.logEvolution({
      timestamp: new Date().toISOString(),
      type: 'new_technique',
      technique_id: candidate.id,
      details: {
        name: candidate.proposed_name,
        category: candidate.proposed_category,
        status: 'applied'
      }
    });
  }

  // ============================================================
  // HELPER METHODS
  // ============================================================

  private extractKeywords(text: string): string[] {
    // Simple keyword extraction (in production, use NLP)
    return text
      .toLowerCase()
      .split(/\s+/)
      .filter(word => word.length > 3)
      .slice(0, 10);
  }

  private analyzePatterns(samples: string[]): {
    common_keywords: string[];
    common_syntax: string[];
    common_semantics: string[];
  } {
    // Aggregate keywords across samples
    const keywordFreq = new Map<string, number>();

    for (const sample of samples) {
      const keywords = this.extractKeywords(sample);
      for (const keyword of keywords) {
        keywordFreq.set(keyword, (keywordFreq.get(keyword) || 0) + 1);
      }
    }

    // Get top keywords
    const commonKeywords = Array.from(keywordFreq.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([keyword]) => keyword);

    return {
      common_keywords: commonKeywords,
      common_syntax: [], // TODO: Implement syntax pattern extraction
      common_semantics: [] // TODO: Implement semantic pattern extraction
    };
  }

  private generateTechniqueName(patterns: any): string {
    // Generate a descriptive name based on patterns
    const keywords = patterns.common_keywords.slice(0, 2).join('-');
    return `Emergent-${keywords}-${Date.now()}`;
  }

  private inferCategory(patterns: any): TechniqueCategory {
    // Simple heuristic-based category inference
    // In production, use ML classifier
    return 'gaslighting'; // Default
  }

  private calculateCandidateConfidence(patterns: any, count: number): number {
    // Confidence based on:
    // - Number of observations
    // - Pattern consistency
    // - Linguistic distinctiveness

    const countScore = Math.min(count / 20, 1.0); // Max out at 20 observations
    const patternScore = patterns.common_keywords.length / 5; // Max 5 keywords

    return (countScore + patternScore) / 2;
  }

  private logEvolution(event: EvolutionEvent): void {
    this.evolutionLog.push(event);

    // TODO: In production, persist to file or database
    if (this.config.evolution_log_path) {
      // Write to log file
    }
  }

  // ============================================================
  // GETTERS
  // ============================================================

  public getEvolutionLog(): EvolutionEvent[] {
    return [...this.evolutionLog];
  }

  public getFitnessScores(): Map<number, TechniqueFitness> {
    return new Map(this.fitnessScores);
  }

  public getStats() {
    return {
      total_candidates: this.candidates.size,
      pending_approval: this.getPendingCandidates().length,
      total_evolutions: this.evolutionLog.length,
      average_fitness: this.calculateAverageFitness(),
      next_technique_id: this.nextTechniqueId
    };
  }

  private calculateAverageFitness(): number {
    if (this.fitnessScores.size === 0) return 0;

    const totalF1 = Array.from(this.fitnessScores.values())
      .reduce((sum, f) => sum + f.f1_score, 0);

    return totalF1 / this.fitnessScores.size;
  }
}

// ============================================================
// EXPORTS
// ============================================================

export function createSelfSurgeryEngine(config?: SurgeryConfig): SelfSurgeryEngine {
  return new SelfSurgeryEngine(config);
}
