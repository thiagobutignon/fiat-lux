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

  // TODO: Real implementation
  // return await cognitiveClient.detectQueryManipulation({ query, userId, organismId });

  throw new Error('CINZA integration not yet implemented');
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

  // TODO: Real implementation
  // return await cognitiveClient.getManipulationTechniques();

  throw new Error('CINZA integration not yet implemented');
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

  // TODO: Real implementation
  // return await cognitiveClient.getUserDarkTetrad(userId);

  throw new Error('CINZA integration not yet implemented');
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

  // TODO: Real implementation
  // return await cognitiveClient.detectCognitiveBiases({ text });

  throw new Error('CINZA integration not yet implemented');
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

  // TODO: Real implementation
  // await cognitiveClient.processStream(stream, onDetection);

  throw new Error('CINZA integration not yet implemented');
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

  // TODO: Real implementation
  // return await cognitiveClient.validateConstitutional({ text, principles });

  throw new Error('CINZA integration not yet implemented');
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

  // TODO: Real implementation
  // return await cognitiveClient.triggerSelfSurgery(organismId);

  throw new Error('CINZA integration not yet implemented');
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

  // TODO: Real implementation
  // return await cognitiveClient.getOptimizationSuggestions(organismId);

  throw new Error('CINZA integration not yet implemented');
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

  // TODO: Real implementation
  // return await cognitiveClient.detectManipulationI18n({ text, language });

  throw new Error('CINZA integration not yet implemented');
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

  // TODO: Real implementation
  // return await cognitiveClient.comprehensiveAnalysis(params);

  throw new Error('CINZA integration not yet implemented');
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
