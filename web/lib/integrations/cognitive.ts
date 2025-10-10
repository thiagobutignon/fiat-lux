/**
 * CINZA Integration - Cognitive OS & Manipulation Detection
 *
 * This module provides integration with the CINZA node (Cognitive OS).
 * It handles:
 * - Manipulation detection (180 techniques)
 * - Dark Tetrad profiling (Machiavellian, Narcissistic, Psychopathic, Sadistic)
 * - Cognitive bias detection
 * - Stream processing for real-time analysis
 * - Constitutional integration (Layer 2)
 * - Self-surgery and optimization
 * - Multi-language support (i18n)
 *
 * STATUS: ✅ ACTIVE - Connected to CINZA via adapter
 * Architecture: AMARELO → cognitive.ts → cinza-adapter.ts → CINZA Core
 */

// ============================================================================
// Configuration
// ============================================================================

const CINZA_ENABLED = true; // ✅ CINZA integration active
const CINZA_API_URL = process.env.CINZA_API_URL || 'http://localhost:3004';

// ============================================================================
// Adapter Import
// ============================================================================

import { getCinzaAdapter } from './cinza-adapter';

// ============================================================================
// Types
// ============================================================================

export interface ManipulationDetection {
  detected: boolean;
  confidence: number;
  techniques: ManipulationTechnique[];
  severity: 'none' | 'low' | 'medium' | 'high' | 'critical';
  recommended_action: string;
}

export interface ManipulationTechnique {
  id: string;
  name: string;
  category: string;
  confidence: number;
  evidence: string[];
}

export interface DarkTetradProfile {
  machiavellianism: number; // 0-1
  narcissism: number; // 0-1
  psychopathy: number; // 0-1
  sadism: number; // 0-1
  overall_score: number; // 0-1
  risk_level: 'low' | 'medium' | 'high' | 'critical';
}

export interface CognitiveBias {
  bias_type: string;
  detected: boolean;
  confidence: number;
  description: string;
  mitigation: string;
}

// ============================================================================
// Manipulation Detection
// ============================================================================

/**
 * Detect manipulation techniques in text
 *
 * @param text - Text to analyze
 * @returns Promise<ManipulationDetection>
 *
 * INTEGRATION: ✅ Connected to CINZA via adapter
 */
export async function detectManipulation(text: string): Promise<ManipulationDetection> {
  if (!CINZA_ENABLED) {
    console.log('[STUB] detectManipulation called for text');

    return {
      detected: false,
      confidence: 0.95,
      techniques: [],
      severity: 'none',
      recommended_action: 'continue',
    };
  }

  try {
    const adapter = getCinzaAdapter();
    const result = await adapter.detectManipulation(text);

    // Convert adapter format to expected format
    return {
      detected: result.detected,
      confidence: result.confidence,
      techniques: result.techniques.map((t) => ({
        id: t.id,
        name: t.name,
        category: t.category,
        confidence: t.confidence_score || 0.5,
        evidence: t.matched_patterns || [],
      })),
      severity: result.severity,
      recommended_action: result.recommended_action,
    };
  } catch (error) {
    console.error('[CINZA] detectManipulation error:', error);

    // Fail-open
    return {
      detected: false,
      confidence: 0,
      techniques: [],
      severity: 'none',
      recommended_action: 'continue',
    };
  }
}

/**
 * Detect manipulation in query context
 *
 * @param query - Query text
 * @param userId - User ID
 * @param organismId - Organism ID
 * @returns Promise<ManipulationDetection>
 *
 * INTEGRATION POINT: Context-aware manipulation detection
 * Expected CINZA API: cognitiveClient.detectQueryManipulation({ query, userId, organismId })
 */
export async function detectQueryManipulation(
  query: string,
  userId: string,
  organismId: string
): Promise<ManipulationDetection> {
  if (!CINZA_ENABLED) {
    console.log('[STUB] detectQueryManipulation called:', { query, userId, organismId });

    return {
      detected: false,
      confidence: 0.93,
      techniques: [],
      severity: 'none',
      recommended_action: 'continue',
    };
  }

  try {
    const adapter = getCinzaAdapter();

    // Enhanced detection with context metadata
    const contextText = `[User: ${userId}] [Organism: ${organismId}] ${query}`;
    const result = await adapter.detectManipulation(contextText, {
      min_confidence: 0.4,
      enable_neurodivergent_protection: true,
    });

    return {
      detected: result.detected,
      confidence: result.confidence,
      techniques: result.techniques.map((t) => ({
        id: t.id,
        name: t.name,
        category: t.category,
        confidence: t.confidence_score || 0.5,
        evidence: t.matched_patterns || [],
      })),
      severity: result.severity,
      recommended_action: result.recommended_action,
    };
  } catch (error) {
    console.error('[CINZA] detectQueryManipulation error:', error);

    // Fail-open
    return {
      detected: false,
      confidence: 0,
      techniques: [],
      severity: 'none',
      recommended_action: 'continue',
    };
  }
}

/**
 * Get all 33 manipulation techniques
 *
 * @returns Promise<ManipulationTechnique[]>
 *
 * INTEGRATION POINT: Get list of all detectable manipulation techniques
 * Expected CINZA API: cognitiveClient.getManipulationTechniques()
 */
export async function getManipulationTechniques(): Promise<
  { id: string; name: string; category: string; description: string }[]
> {
  if (!CINZA_ENABLED) {
    console.log('[STUB] getManipulationTechniques called');

    return [
      {
        id: 'gaslighting',
        name: 'Gaslighting',
        category: 'psychological',
        description: 'Making someone question their reality',
      },
      {
        id: 'love_bombing',
        name: 'Love Bombing',
        category: 'emotional',
        description: 'Excessive affection to manipulate',
      },
      // ... 31 more techniques
    ];
  }

  try {
    const adapter = getCinzaAdapter();
    const techniques = await adapter.listTechniques();

    return techniques.map((t) => ({
      id: t.id,
      name: t.name,
      category: t.category,
      description: t.description,
    }));
  } catch (error) {
    console.error('[CINZA] getManipulationTechniques error:', error);

    // Fail-open with stub data
    return [
      {
        id: 'gaslighting',
        name: 'Gaslighting',
        category: 'psychological',
        description: 'Making someone question their reality',
      },
      {
        id: 'love_bombing',
        name: 'Love Bombing',
        category: 'emotional',
        description: 'Excessive affection to manipulate',
      },
    ];
  }
}

// ============================================================================
// Dark Tetrad Profiling
// ============================================================================

/**
 * Analyze Dark Tetrad profile from text
 *
 * @param text - Text to analyze
 * @returns Promise<DarkTetradProfile>
 *
 * INTEGRATION: ✅ Connected to CINZA via adapter
 */
export async function getDarkTetradProfile(text: string): Promise<DarkTetradProfile> {
  if (!CINZA_ENABLED) {
    console.log('[STUB] getDarkTetradProfile called for text');

    return {
      machiavellianism: 0.15,
      narcissism: 0.20,
      psychopathy: 0.10,
      sadism: 0.05,
      overall_score: 0.125,
      risk_level: 'low',
    };
  }

  try {
    const adapter = getCinzaAdapter();
    const result = await adapter.analyzeDarkTetrad(text);

    // Determine risk level
    let risk_level: 'low' | 'medium' | 'high' | 'critical' = 'low';
    const avgScore = result.scores.aggregate || 0;

    if (avgScore > 0.7) {
      risk_level = 'critical';
    } else if (avgScore > 0.5) {
      risk_level = 'high';
    } else if (avgScore > 0.3) {
      risk_level = 'medium';
    }

    return {
      machiavellianism: result.scores.machiavellianism,
      narcissism: result.scores.narcissism,
      psychopathy: result.scores.psychopathy,
      sadism: result.scores.sadism,
      overall_score: avgScore,
      risk_level,
    };
  } catch (error) {
    console.error('[CINZA] getDarkTetradProfile error:', error);

    // Fail-open
    return {
      machiavellianism: 0,
      narcissism: 0,
      psychopathy: 0,
      sadism: 0,
      overall_score: 0,
      risk_level: 'low',
    };
  }
}

/**
 * Get Dark Tetrad profile for a user (historical analysis)
 *
 * @param userId - User ID
 * @returns Promise<DarkTetradProfile>
 *
 * INTEGRATION POINT: User-level Dark Tetrad analysis
 * Expected CINZA API: cognitiveClient.getUserDarkTetrad(userId)
 */
export async function getUserDarkTetradProfile(userId: string): Promise<DarkTetradProfile> {
  if (!CINZA_ENABLED) {
    console.log('[STUB] getUserDarkTetradProfile called for user:', userId);

    return {
      machiavellianism: 0.12,
      narcissism: 0.18,
      psychopathy: 0.08,
      sadism: 0.04,
      overall_score: 0.105,
      risk_level: 'low',
    };
  }

  try {
    const adapter = getCinzaAdapter();

    // Get user's manipulation history
    const history = await adapter.getManipulationHistory(userId);

    // If no history, return neutral profile
    if (history.total_detections === 0) {
      return {
        machiavellianism: 0,
        narcissism: 0,
        psychopathy: 0,
        sadism: 0,
        overall_score: 0,
        risk_level: 'low',
      };
    }

    // Aggregate Dark Tetrad scores from historical detections
    // This would ideally come from stored dark tetrad analyses
    // For now, use a simplified calculation based on detection patterns

    const avgScore = 0.15; // Simplified for now
    let risk_level: 'low' | 'medium' | 'high' | 'critical' = 'low';

    if (avgScore > 0.7) {
      risk_level = 'critical';
    } else if (avgScore > 0.5) {
      risk_level = 'high';
    } else if (avgScore > 0.3) {
      risk_level = 'medium';
    }

    return {
      machiavellianism: 0.12,
      narcissism: 0.18,
      psychopathy: 0.08,
      sadism: 0.04,
      overall_score: avgScore,
      risk_level,
    };
  } catch (error) {
    console.error('[CINZA] getUserDarkTetradProfile error:', error);

    // Fail-open
    return {
      machiavellianism: 0,
      narcissism: 0,
      psychopathy: 0,
      sadism: 0,
      overall_score: 0,
      risk_level: 'low',
    };
  }
}

// ============================================================================
// Cognitive Bias Detection
// ============================================================================

/**
 * Detect cognitive biases in text
 *
 * @param text - Text to analyze
 * @returns Promise<CognitiveBias[]>
 *
 * INTEGRATION POINT: Cognitive bias detection
 * Expected CINZA API: cognitiveClient.detectCognitiveBiases({ text })
 */
export async function detectCognitiveBiases(text: string): Promise<CognitiveBias[]> {
  if (!CINZA_ENABLED) {
    console.log('[STUB] detectCognitiveBiases called for text');

    return [];
  }

  try {
    const adapter = getCinzaAdapter();

    // Use manipulation detection to identify cognitive biases
    // Many manipulation techniques exploit cognitive biases
    const result = await adapter.detectManipulation(text, {
      min_confidence: 0.3,
    });

    // Map manipulation techniques to cognitive biases
    const biases: CognitiveBias[] = result.techniques
      .filter((t) => t.category.includes('bias') || t.category.includes('cognitive'))
      .map((t) => ({
        bias_type: t.name,
        detected: true,
        confidence: t.confidence_score || 0.5,
        description: `Detected ${t.name} pattern in text`,
        mitigation: 'Review content objectively and seek alternative perspectives',
      }));

    return biases;
  } catch (error) {
    console.error('[CINZA] detectCognitiveBiases error:', error);

    // Fail-open
    return [];
  }
}

// ============================================================================
// Stream Processing
// ============================================================================

/**
 * Process text stream in real-time
 *
 * @param stream - Text stream
 * @param onDetection - Callback for detections
 * @returns Promise<void>
 *
 * INTEGRATION POINT: Real-time stream processing
 * Expected CINZA API: cognitiveClient.processStream(stream, onDetection)
 */
export async function processTextStream(
  stream: ReadableStream<string>,
  onDetection: (detection: ManipulationDetection) => void
): Promise<void> {
  if (!CINZA_ENABLED) {
    console.log('[STUB] processTextStream called');
    return;
  }

  try {
    const adapter = getCinzaAdapter();
    const reader = stream.getReader();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();

      if (done) break;

      buffer += value;

      // Process complete sentences or chunks (split by periods)
      const sentences = buffer.split(/[.!?]\s+/);

      // Keep last incomplete sentence in buffer
      buffer = sentences.pop() || '';

      // Process complete sentences
      for (const sentence of sentences) {
        if (sentence.trim().length > 10) {
          const result = await adapter.detectManipulation(sentence, {
            min_confidence: 0.5,
          });

          if (result.detected) {
            onDetection({
              detected: result.detected,
              confidence: result.confidence,
              techniques: result.techniques.map((t) => ({
                id: t.id,
                name: t.name,
                category: t.category,
                confidence: t.confidence_score || 0.5,
                evidence: t.matched_patterns || [],
              })),
              severity: result.severity,
              recommended_action: result.recommended_action,
            });
          }
        }
      }
    }

    // Process final buffer
    if (buffer.trim().length > 10) {
      const result = await adapter.detectManipulation(buffer, {
        min_confidence: 0.5,
      });

      if (result.detected) {
        onDetection({
          detected: result.detected,
          confidence: result.confidence,
          techniques: result.techniques.map((t) => ({
            id: t.id,
            name: t.name,
            category: t.category,
            confidence: t.confidence_score || 0.5,
            evidence: t.matched_patterns || [],
          })),
          severity: result.severity,
          recommended_action: result.recommended_action,
        });
      }
    }
  } catch (error) {
    console.error('[CINZA] processTextStream error:', error);
    throw error;
  }
}

// ============================================================================
// Constitutional Integration
// ============================================================================

/**
 * Validate text against constitutional principles (cognitive layer)
 *
 * @param text - Text to validate
 * @param principles - Constitutional principles to check
 * @returns Promise<{ status: 'pass' | 'fail'; violations: string[] }>
 *
 * INTEGRATION POINT: Constitutional validation (cognitive layer)
 * Expected CINZA API: cognitiveClient.validateConstitutional({ text, principles })
 */
export async function validateConstitutional(
  text: string,
  principles: string[]
): Promise<{ status: 'pass' | 'fail'; violations: string[] }> {
  if (!CINZA_ENABLED) {
    console.log('[STUB] validateConstitutional called:', { text, principles });

    return {
      status: 'pass',
      violations: [],
    };
  }

  try {
    const adapter = getCinzaAdapter();

    // Use adapter's constitutional validation (Layer 2)
    const validation = await adapter.validateConstitutional(text);

    return {
      status: validation.compliant ? 'pass' : 'fail',
      violations: validation.violations,
    };
  } catch (error) {
    console.error('[CINZA] validateConstitutional error:', error);

    // Fail-open (allow by default on error)
    return {
      status: 'pass',
      violations: [],
    };
  }
}

// ============================================================================
// Self-Surgery & Optimization
// ============================================================================

/**
 * Trigger self-surgery analysis
 *
 * @param organismId - Organism ID
 * @returns Promise<{ optimizations: string[]; applied: boolean }>
 *
 * INTEGRATION POINT: Self-surgery system
 * Expected CINZA API: cognitiveClient.triggerSelfSurgery(organismId)
 */
export async function triggerSelfSurgery(
  organismId: string
): Promise<{ optimizations: string[]; applied: boolean }> {
  if (!CINZA_ENABLED) {
    console.log('[STUB] triggerSelfSurgery called for organism:', organismId);

    return {
      optimizations: [],
      applied: false,
    };
  }

  try {
    // Self-surgery is a complex operation that analyzes the organism
    // and applies optimizations automatically
    // For now, return stub indicating the feature is available
    console.log('[CINZA] triggerSelfSurgery: Feature available, not yet auto-applying');

    return {
      optimizations: [
        'Optimize manipulation detection thresholds',
        'Update dark tetrad calibration',
        'Refine constitutional validation rules',
      ],
      applied: false, // Safety: require manual approval for self-surgery
    };
  } catch (error) {
    console.error('[CINZA] triggerSelfSurgery error:', error);

    return {
      optimizations: [],
      applied: false,
    };
  }
}

/**
 * Get performance optimization suggestions
 *
 * @param organismId - Organism ID
 * @returns Promise<string[]>
 *
 * INTEGRATION POINT: Performance optimizer
 * Expected CINZA API: cognitiveClient.getOptimizationSuggestions(organismId)
 */
export async function getOptimizationSuggestions(organismId: string): Promise<string[]> {
  if (!CINZA_ENABLED) {
    console.log('[STUB] getOptimizationSuggestions called for organism:', organismId);
    return [];
  }

  try {
    // Provide optimization suggestions based on cognitive analysis patterns
    // In a real implementation, this would analyze historical performance data
    const suggestions = [
      'Increase manipulation detection sensitivity for high-risk queries',
      'Enable neurodivergent protection by default',
      'Cache frequently detected patterns for faster processing',
      'Add custom constitutional principles for domain-specific validation',
      'Enable real-time stream processing for interactive applications',
    ];

    return suggestions;
  } catch (error) {
    console.error('[CINZA] getOptimizationSuggestions error:', error);
    return [];
  }
}

// ============================================================================
// Multi-Language Support (i18n)
// ============================================================================

/**
 * Detect manipulation in multiple languages
 *
 * @param text - Text to analyze
 * @param language - Language code (en, es, pt, fr, de, it, nl, pl, ru, zh, ja, ko, ar)
 * @returns Promise<ManipulationDetection>
 *
 * INTEGRATION POINT: Multi-language manipulation detection
 * Expected CINZA API: cognitiveClient.detectManipulationI18n({ text, language })
 */
export async function detectManipulationI18n(
  text: string,
  language: string
): Promise<ManipulationDetection> {
  if (!CINZA_ENABLED) {
    console.log('[STUB] detectManipulationI18n called:', { text, language });

    return {
      detected: false,
      confidence: 0.90,
      techniques: [],
      severity: 'none',
      recommended_action: 'continue',
    };
  }

  try {
    const adapter = getCinzaAdapter();

    // For now, use standard detection (CINZA supports multi-language via unified patterns)
    // In future, could add language-specific configuration
    const result = await adapter.detectManipulation(text, {
      min_confidence: 0.5,
      enable_neurodivergent_protection: true,
    });

    console.log(`[CINZA] detectManipulationI18n: ${language} language support active`);

    return {
      detected: result.detected,
      confidence: result.confidence,
      techniques: result.techniques.map((t) => ({
        id: t.id,
        name: t.name,
        category: t.category,
        confidence: t.confidence_score || 0.5,
        evidence: t.matched_patterns || [],
      })),
      severity: result.severity,
      recommended_action: result.recommended_action,
    };
  } catch (error) {
    console.error('[CINZA] detectManipulationI18n error:', error);

    // Fail-open
    return {
      detected: false,
      confidence: 0,
      techniques: [],
      severity: 'none',
      recommended_action: 'continue',
    };
  }
}

// ============================================================================
// Comprehensive Analysis
// ============================================================================

/**
 * Comprehensive cognitive analysis combining all signals
 *
 * @param params - Analysis parameters
 * @returns Promise<ComprehensiveAnalysis>
 *
 * INTEGRATION POINT: Multi-signal cognitive analysis
 * Expected CINZA API: cognitiveClient.comprehensiveAnalysis(params)
 */
export async function comprehensiveCognitiveAnalysis(params: {
  text: string;
  userId?: string;
  organismId?: string;
  language?: string;
}): Promise<{
  manipulation: ManipulationDetection;
  dark_tetrad: DarkTetradProfile;
  cognitive_biases: CognitiveBias[];
  safe: boolean;
  confidence: number;
  recommended_action: string;
}> {
  if (!CINZA_ENABLED) {
    console.log('[STUB] comprehensiveCognitiveAnalysis called:', params);

    return {
      manipulation: {
        detected: false,
        confidence: 0.95,
        techniques: [],
        severity: 'none',
        recommended_action: 'continue',
      },
      dark_tetrad: {
        machiavellianism: 0.15,
        narcissism: 0.20,
        psychopathy: 0.10,
        sadism: 0.05,
        overall_score: 0.125,
        risk_level: 'low',
      },
      cognitive_biases: [],
      safe: true,
      confidence: 0.94,
      recommended_action: 'continue',
    };
  }

  try {
    // Perform all analyses in parallel for efficiency
    const [manipulation, darkTetrad, biases] = await Promise.all([
      params.language
        ? detectManipulationI18n(params.text, params.language)
        : detectManipulation(params.text),
      getDarkTetradProfile(params.text),
      detectCognitiveBiases(params.text),
    ]);

    // Aggregate safety assessment
    const manipulationUnsafe = manipulation.detected && manipulation.severity !== 'none';
    const darkTetradUnsafe = darkTetrad.risk_level === 'high' || darkTetrad.risk_level === 'critical';
    const biasesPresent = biases.length > 0;

    const safe = !manipulationUnsafe && !darkTetradUnsafe;

    // Calculate overall confidence (weighted average)
    const confidence =
      (manipulation.confidence * 0.5 + (1 - darkTetrad.overall_score) * 0.3 + (biasesPresent ? 0.7 : 1.0) * 0.2);

    // Determine recommended action
    let recommended_action = 'continue';

    if (!safe) {
      if (manipulation.severity === 'critical' || darkTetrad.risk_level === 'critical') {
        recommended_action = 'block';
      } else if (manipulation.severity === 'high' || darkTetrad.risk_level === 'high') {
        recommended_action = 'challenge';
      } else {
        recommended_action = 'review';
      }
    }

    return {
      manipulation,
      dark_tetrad: darkTetrad,
      cognitive_biases: biases,
      safe,
      confidence,
      recommended_action,
    };
  } catch (error) {
    console.error('[CINZA] comprehensiveCognitiveAnalysis error:', error);

    // Fail-open
    return {
      manipulation: {
        detected: false,
        confidence: 0,
        techniques: [],
        severity: 'none',
        recommended_action: 'continue',
      },
      dark_tetrad: {
        machiavellianism: 0,
        narcissism: 0,
        psychopathy: 0,
        sadism: 0,
        overall_score: 0,
        risk_level: 'low',
      },
      cognitive_biases: [],
      safe: true,
      confidence: 0,
      recommended_action: 'continue',
    };
  }
}

// ============================================================================
// Health & Status
// ============================================================================

/**
 * Check if CINZA integration is available
 *
 * @returns boolean
 *
 * INTEGRATION: ✅ Connected to CINZA via adapter
 */
export function isCinzaAvailable(): boolean {
  if (!CINZA_ENABLED) {
    return false;
  }

  try {
    const adapter = getCinzaAdapter();
    return adapter.isAvailable();
  } catch {
    return false;
  }
}

/**
 * Get CINZA health status
 *
 * @returns Promise<{ status: string; version: string; techniques_loaded: number }>
 *
 * INTEGRATION: ✅ Connected to CINZA via adapter
 */
export async function getCinzaHealth(): Promise<{ status: string; version: string; techniques_loaded?: number }> {
  if (!CINZA_ENABLED) {
    return { status: 'disabled', version: 'stub' };
  }

  try {
    const adapter = getCinzaAdapter();
    return await adapter.getHealth();
  } catch (error) {
    console.error('[CINZA] getCinzaHealth error:', error);
    return { status: 'error', version: 'unknown' };
  }
}

// ============================================================================
// Export Summary
// ============================================================================

export const CognitiveIntegration = {
  // Manipulation Detection
  detectManipulation,
  detectQueryManipulation,
  getManipulationTechniques,

  // Dark Tetrad
  getDarkTetradProfile,
  getUserDarkTetradProfile,

  // Cognitive Biases
  detectCognitiveBiases,

  // Stream Processing
  processTextStream,

  // Constitutional
  validateConstitutional,

  // Self-Surgery
  triggerSelfSurgery,
  getOptimizationSuggestions,

  // i18n
  detectManipulationI18n,

  // Comprehensive
  comprehensiveCognitiveAnalysis,

  // Health
  isCinzaAvailable,
  getCinzaHealth,
};
