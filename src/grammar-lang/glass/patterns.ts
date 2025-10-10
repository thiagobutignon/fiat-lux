/**
 * Pattern Detection Engine
 *
 * Refines pattern detection to prepare for CODE EMERGENCE
 *
 * Key concepts:
 * 1. Pattern Threshold - When does a pattern trigger function emergence?
 * 2. Pattern Clustering - Group similar patterns
 * 3. Pattern Correlation - Relationships between patterns
 * 4. Emergence Readiness - Is pattern ready to synthesize code?
 *
 * Threshold for emergence: 100+ occurrences with 80%+ confidence
 */

import { GlassOrganism, GlassFunction } from './types';
import { createLLMPatternDetector, LLMPatternDetector } from './llm-pattern-detection';

/**
 * Enhanced Pattern (beyond simple count)
 */
export interface EnhancedPattern {
  type: string; // e.g., "drug_efficacy"
  frequency: number; // how many times seen
  confidence: number; // 0.0 to 1.0
  documents: string[]; // document IDs
  first_seen: string; // ISO timestamp
  last_seen: string;

  // Enhanced fields
  keywords: string[]; // keywords that trigger this pattern
  correlations: {
    pattern: string; // correlated pattern
    strength: number; // 0.0 to 1.0
  }[];
  cluster: string | null; // cluster ID
  emergence_ready: boolean; // ready to synthesize function?
  emergence_score: number; // 0.0 to 1.0 (combines frequency + confidence)
}

/**
 * Pattern Cluster
 * Groups related patterns together
 */
export interface PatternCluster {
  id: string;
  name: string; // e.g., "treatment_efficacy_cluster"
  patterns: string[]; // pattern types in this cluster
  strength: number; // 0.0 to 1.0
  potential_functions: string[]; // function names that could emerge
}

/**
 * Pattern Correlation
 * Relationship between two patterns
 */
export interface PatternCorrelation {
  pattern_a: string;
  pattern_b: string;
  strength: number; // 0.0 to 1.0
  co_occurrence: number; // how many times they appear together
}

/**
 * Emergence Candidate
 * Pattern that is ready to become a function
 */
export interface EmergenceCandidate {
  pattern: EnhancedPattern;
  suggested_function_name: string;
  suggested_signature: string;
  confidence: number;
  supporting_patterns: string[]; // other patterns that support this
}

/**
 * Pattern Detection Engine
 */
export class PatternDetectionEngine {
  private organism: GlassOrganism;
  private enhancedPatterns: Map<string, EnhancedPattern>;
  private clusters: PatternCluster[];
  private correlations: PatternCorrelation[];
  private llmDetector?: LLMPatternDetector;

  // Thresholds
  private readonly EMERGENCE_FREQUENCY_THRESHOLD = 100;
  private readonly EMERGENCE_CONFIDENCE_THRESHOLD = 0.8;
  private readonly EMERGENCE_SCORE_THRESHOLD = 0.75;
  private readonly CORRELATION_THRESHOLD = 0.6;

  constructor(organism: GlassOrganism, useLLM: boolean = false, maxBudget: number = 0.3) {
    this.organism = organism;
    this.enhancedPatterns = new Map();
    this.clusters = [];
    this.correlations = [];

    // Optionally use LLM for semantic pattern detection
    if (useLLM) {
      this.llmDetector = createLLMPatternDetector(maxBudget);
    }
  }

  /**
   * Analyze patterns in organism
   */
  public analyze(): {
    enhanced_patterns: EnhancedPattern[];
    clusters: PatternCluster[];
    correlations: PatternCorrelation[];
    emergence_candidates: EmergenceCandidate[];
  } {
    // 1. Enhance basic patterns
    this.enhancePatterns();

    // 2. Detect correlations (keyword-based)
    this.detectCorrelations();

    // 3. Cluster patterns
    this.clusterPatterns();

    // 4. Identify emergence candidates
    const emergenceCandidates = this.identifyEmergenceCandidates();

    return {
      enhanced_patterns: Array.from(this.enhancedPatterns.values()),
      clusters: this.clusters,
      correlations: this.correlations,
      emergence_candidates: emergenceCandidates
    };
  }

  /**
   * Analyze patterns using LLM semantic detection (async)
   */
  public async analyzeWithLLM(): Promise<{
    enhanced_patterns: EnhancedPattern[];
    clusters: PatternCluster[];
    correlations: PatternCorrelation[];
    emergence_candidates: EmergenceCandidate[];
  }> {
    if (!this.llmDetector) {
      console.warn('⚠️  LLM detector not enabled, falling back to keyword-based analysis');
      return this.analyze();
    }

    console.log('   🤖 Using LLM for semantic pattern detection...');

    // 1. Enhance basic patterns
    this.enhancePatterns();

    // 2. Detect correlations using LLM semantic analysis
    await this.detectCorrelationsWithLLM();

    // 3. Cluster patterns (using LLM-detected correlations)
    this.clusterPatterns();

    // 4. Identify emergence candidates
    const emergenceCandidates = this.identifyEmergenceCandidates();

    const totalCost = this.llmDetector.getTotalCost();
    console.log(`   💰 Pattern detection cost: $${totalCost.toFixed(4)}`);

    return {
      enhanced_patterns: Array.from(this.enhancedPatterns.values()),
      clusters: this.clusters,
      correlations: this.correlations,
      emergence_candidates: emergenceCandidates
    };
  }

  /**
   * Detect correlations using LLM semantic analysis
   */
  private async detectCorrelationsWithLLM(): Promise<void> {
    const patterns = Array.from(this.enhancedPatterns.values());

    // Use LLM to detect semantic correlations
    const llmCorrelations = await this.llmDetector!.detectSemanticCorrelations(
      patterns,
      this.CORRELATION_THRESHOLD
    );

    this.correlations = llmCorrelations;

    // Update pattern correlations
    for (const corr of llmCorrelations) {
      const patternA = this.enhancedPatterns.get(corr.pattern_a);
      const patternB = this.enhancedPatterns.get(corr.pattern_b);

      if (patternA && patternB) {
        patternA.correlations.push({
          pattern: corr.pattern_b,
          strength: corr.strength
        });
        patternB.correlations.push({
          pattern: corr.pattern_a,
          strength: corr.strength
        });
      }
    }

    console.log(`   ✅ LLM detected ${llmCorrelations.length} semantic correlations`);
  }

  /**
   * Enhance basic patterns with additional metadata
   */
  private enhancePatterns(): void {
    const basicPatterns = this.organism.knowledge.patterns;
    const now = new Date().toISOString();

    for (const [type, frequency] of Object.entries(basicPatterns)) {
      // Calculate confidence (simplified - based on frequency)
      const confidence = Math.min(frequency / 200, 1.0);

      // Calculate emergence score (combines frequency and confidence)
      const freqScore = Math.min(frequency / this.EMERGENCE_FREQUENCY_THRESHOLD, 1.0);
      const emergenceScore = (freqScore * 0.6) + (confidence * 0.4);

      // Determine emergence readiness
      const emergenceReady =
        frequency >= this.EMERGENCE_FREQUENCY_THRESHOLD &&
        confidence >= this.EMERGENCE_CONFIDENCE_THRESHOLD;

      // Extract keywords from pattern type
      const keywords = type.replace('_pattern', '').split('_');

      this.enhancedPatterns.set(type, {
        type,
        frequency,
        confidence,
        documents: [], // would be populated from actual documents
        first_seen: now,
        last_seen: now,
        keywords,
        correlations: [],
        cluster: null,
        emergence_ready: emergenceReady,
        emergence_score: emergenceScore
      });
    }
  }

  /**
   * Detect correlations between patterns
   */
  private detectCorrelations(): void {
    const patterns = Array.from(this.enhancedPatterns.values());

    for (let i = 0; i < patterns.length; i++) {
      for (let j = i + 1; j < patterns.length; j++) {
        const patternA = patterns[i];
        const patternB = patterns[j];

        // Calculate correlation strength (simplified - based on keyword overlap)
        const keywordsA = new Set(patternA.keywords);
        const keywordsB = new Set(patternB.keywords);
        const intersection = new Set([...keywordsA].filter(k => keywordsB.has(k)));
        const union = new Set([...keywordsA, ...keywordsB]);

        const strength = intersection.size / union.size;

        if (strength >= this.CORRELATION_THRESHOLD) {
          // Estimate co-occurrence (simplified)
          const coOccurrence = Math.min(patternA.frequency, patternB.frequency) * strength;

          const correlation: PatternCorrelation = {
            pattern_a: patternA.type,
            pattern_b: patternB.type,
            strength,
            co_occurrence: Math.floor(coOccurrence)
          };

          this.correlations.push(correlation);

          // Update pattern correlations
          patternA.correlations.push({
            pattern: patternB.type,
            strength
          });
          patternB.correlations.push({
            pattern: patternA.type,
            strength
          });
        }
      }
    }
  }

  /**
   * Cluster patterns
   */
  private clusterPatterns(): void {
    const patterns = Array.from(this.enhancedPatterns.values());
    const clustered = new Set<string>();

    let clusterIndex = 0;

    for (const pattern of patterns) {
      if (clustered.has(pattern.type)) continue;

      // Create new cluster
      const clusterPatterns = [pattern.type];
      clustered.add(pattern.type);

      // Find correlated patterns
      for (const corr of pattern.correlations) {
        if (!clustered.has(corr.pattern) && corr.strength >= this.CORRELATION_THRESHOLD) {
          clusterPatterns.push(corr.pattern);
          clustered.add(corr.pattern);
        }
      }

      if (clusterPatterns.length > 0) {
        const clusterId = `cluster_${clusterIndex++}`;

        // Generate cluster name from keywords
        const allKeywords = clusterPatterns
          .flatMap(p => this.enhancedPatterns.get(p)?.keywords || []);
        const uniqueKeywords = Array.from(new Set(allKeywords));
        const clusterName = uniqueKeywords.slice(0, 3).join('_') + '_cluster';

        // Calculate cluster strength (average of pattern emergence scores)
        const avgStrength = clusterPatterns
          .map(p => this.enhancedPatterns.get(p)?.emergence_score || 0)
          .reduce((a, b) => a + b, 0) / clusterPatterns.length;

        // Generate potential function names
        const potentialFunctions = this.generateFunctionNames(clusterPatterns);

        const cluster: PatternCluster = {
          id: clusterId,
          name: clusterName,
          patterns: clusterPatterns,
          strength: avgStrength,
          potential_functions: potentialFunctions
        };

        this.clusters.push(cluster);

        // Update patterns with cluster ID
        for (const patternType of clusterPatterns) {
          const p = this.enhancedPatterns.get(patternType);
          if (p) p.cluster = clusterId;
        }
      }
    }
  }

  /**
   * Generate function names from pattern cluster
   */
  private generateFunctionNames(patternTypes: string[]): string[] {
    const functionNames: string[] = [];

    for (const patternType of patternTypes) {
      const pattern = this.enhancedPatterns.get(patternType);
      if (!pattern || !pattern.emergence_ready) continue;

      // Generate function name from pattern keywords
      // e.g., "efficacy_pattern" → "analyze_efficacy"
      const keywords = pattern.keywords;

      // Common function prefixes based on domain
      const prefixes = ['analyze', 'calculate', 'predict', 'evaluate', 'assess'];

      for (const prefix of prefixes) {
        if (keywords.some(k => ['efficacy', 'outcome', 'response'].includes(k))) {
          functionNames.push(`${prefix}_${keywords[0]}`);
          break;
        }
      }
    }

    return Array.from(new Set(functionNames)); // unique
  }

  /**
   * Identify emergence candidates
   * Patterns that are ready to become functions
   */
  private identifyEmergenceCandidates(): EmergenceCandidate[] {
    const candidates: EmergenceCandidate[] = [];

    for (const pattern of this.enhancedPatterns.values()) {
      if (!pattern.emergence_ready) continue;

      // Generate function signature
      const functionName = this.generateFunctionName(pattern);
      const signature = this.generateSignature(pattern, functionName);

      // Find supporting patterns (correlated patterns that are also ready)
      const supportingPatterns = pattern.correlations
        .filter(c => {
          const p = this.enhancedPatterns.get(c.pattern);
          return p && p.emergence_ready;
        })
        .map(c => c.pattern);

      // Calculate candidate confidence (pattern confidence + support bonus)
      const supportBonus = Math.min(supportingPatterns.length * 0.1, 0.2);
      const confidence = Math.min(pattern.confidence + supportBonus, 1.0);

      candidates.push({
        pattern,
        suggested_function_name: functionName,
        suggested_signature: signature,
        confidence,
        supporting_patterns: supportingPatterns
      });
    }

    // Sort by confidence (highest first)
    candidates.sort((a, b) => b.confidence - a.confidence);

    return candidates;
  }

  /**
   * Generate function name from pattern
   */
  private generateFunctionName(pattern: EnhancedPattern): string {
    const keywords = pattern.keywords;
    const domain = this.organism.metadata.specialization;

    // Choose prefix based on keywords
    let prefix = 'analyze';
    if (keywords.includes('treatment') || keywords.includes('therapy')) {
      prefix = 'evaluate';
    } else if (keywords.includes('outcome') || keywords.includes('survival')) {
      prefix = 'predict';
    } else if (keywords.includes('efficacy') || keywords.includes('response')) {
      prefix = 'assess';
    }

    // Construct name
    const mainKeyword = keywords[0];
    return `${prefix}_${mainKeyword}`;
  }

  /**
   * Generate function signature
   */
  private generateSignature(pattern: EnhancedPattern, functionName: string): string {
    const keywords = pattern.keywords;
    const domain = this.organism.metadata.specialization;

    // Generate based on domain and keywords
    if (domain === 'oncology') {
      if (keywords.includes('efficacy')) {
        return `${functionName}(cancer_type: CancerType, drug: Drug, stage: Stage) -> Efficacy`;
      } else if (keywords.includes('outcome')) {
        return `${functionName}(cancer_type: CancerType, treatment: Treatment) -> Outcome`;
      } else if (keywords.includes('trial')) {
        return `${functionName}(cancer_type: CancerType, criteria: Criteria) -> ClinicalTrial[]`;
      }
    }

    // Default signature
    return `${functionName}(input: Input) -> Output`;
  }

  /**
   * Get updated organism with pattern metadata
   */
  public getOrganism(): GlassOrganism {
    // Update organism with enhanced pattern data
    // (stored in a way that's compatible with basic structure)
    return this.organism;
  }

  /**
   * Get pattern analysis summary
   */
  public getSummary(): {
    total_patterns: number;
    emergence_ready: number;
    clusters: number;
    correlations: number;
    emergence_candidates: number;
  } {
    const emergenceReady = Array.from(this.enhancedPatterns.values())
      .filter(p => p.emergence_ready).length;

    const emergenceCandidates = this.identifyEmergenceCandidates().length;

    return {
      total_patterns: this.enhancedPatterns.size,
      emergence_ready: emergenceReady,
      clusters: this.clusters.length,
      correlations: this.correlations.length,
      emergence_candidates: emergenceCandidates
    };
  }
}
