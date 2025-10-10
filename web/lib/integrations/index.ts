/**
 * Integration Layer - Index
 *
 * Central export point for all 5 node integrations:
 * - ROXO (glass.ts) - Core .glass organisms & GlassRuntime
 * - VERDE (gvcs.ts) - Genetic Version Control System
 * - VERMELHO (security.ts) - Security & Behavioral Analysis
 * - CINZA (cognitive.ts) - Cognitive OS & Manipulation Detection
 * - LARANJA (sqlo.ts) - O(1) Database
 *
 * Usage:
 * ```typescript
 * import { GlassIntegration, GVCSIntegration, SecurityIntegration, CognitiveIntegration, SQLOIntegration } from '@/lib/integrations';
 *
 * // Execute a query via ROXO
 * const result = await GlassIntegration.executeQuery(organismId, query);
 *
 * // Get version history via VERDE
 * const versions = await GVCSIntegration.getVersionHistory(organismId);
 *
 * // Analyze duress via VERMELHO
 * const duressAnalysis = await SecurityIntegration.analyzeDuress(text, userId);
 *
 * // Detect manipulation via CINZA
 * const manipulation = await CognitiveIntegration.detectManipulation(text);
 *
 * // Store episodic memory via LARANJA
 * await SQLOIntegration.storeEpisodicMemory({ organism_id, query, result, user_id, timestamp, session_id });
 * ```
 *
 * STATUS: All integrations are stubs - Ready for real implementations
 *
 * To enable a node integration:
 * 1. Set the *_ENABLED flag to true in the respective file
 * 2. Configure the *_API_URL environment variable
 * 3. Implement the TODO sections with real API calls
 */

// ============================================================================
// ROXO Integration - Core .glass organisms
// ============================================================================

export { GlassIntegration } from './glass';
export {
  createRuntime,
  loadOrganism,
  executeQuery,
  validateQuery,
  getPatterns,
  detectPatterns,
  getEmergedFunctions,
  synthesizeCode,
  ingestKnowledge,
  getKnowledgeGraph,
  isRoxoAvailable,
  getRoxoHealth,
} from './glass';

// ============================================================================
// VERDE Integration - Genetic Version Control
// ============================================================================

export { GVCSIntegration } from './gvcs';
export {
  getVersionHistory,
  getCurrentVersion,
  getEvolutionData,
  getCanaryStatus,
  deployCanary,
  promoteCanary,
  rollbackCanary,
  rollbackVersion,
  getOldButGoldVersions,
  markOldButGold,
  recordFitness,
  getFitnessTrajectory,
  autoCommit,
  isVerdeAvailable,
  getVerdeHealth,
} from './gvcs';

// ============================================================================
// VERMELHO Integration - Security & Behavioral
// ============================================================================

export { SecurityIntegration } from './security';
export {
  analyzeDuress,
  analyzeQueryDuress,
  getBehavioralProfile,
  updateBehavioralProfile,
  analyzeLinguisticFingerprint,
  analyzeTypingPatterns,
  analyzeEmotionalState,
  compareEmotionalState,
  analyzeTemporalPattern,
  comprehensiveSecurityAnalysis,
  isVermelhoAvailable,
  getVermelhoHealth,
} from './security';

export type {
  DuressAnalysis,
  BehavioralProfile,
  EmotionalState,
  TypingPattern,
} from './security';

// ============================================================================
// CINZA Integration - Cognitive OS & Manipulation Detection
// ============================================================================

export { CognitiveIntegration } from './cognitive';
export {
  detectManipulation,
  detectQueryManipulation,
  getManipulationTechniques,
  getDarkTetradProfile,
  getUserDarkTetradProfile,
  detectCognitiveBiases,
  processTextStream,
  validateConstitutional,
  triggerSelfSurgery,
  getOptimizationSuggestions,
  detectManipulationI18n,
  comprehensiveCognitiveAnalysis,
  isCinzaAvailable,
  getCinzaHealth,
} from './cognitive';

export type {
  ManipulationDetection,
  ManipulationTechnique,
  DarkTetradProfile,
  CognitiveBias,
} from './cognitive';

// ============================================================================
// LARANJA Integration - .sqlo O(1) Database
// ============================================================================

export { SQLOIntegration } from './sqlo';
export {
  getOrganism,
  getAllOrganisms,
  storeOrganism,
  updateOrganism,
  deleteOrganism,
  storeEpisodicMemory,
  getEpisodicMemory,
  getUserQueryHistory,
  storeConstitutionalLog,
  getConstitutionalLogs,
  storeLLMCall,
  getLLMCalls,
  getUserRoles,
  checkPermission,
  createRole,
  assignRole,
  runConsolidation,
  getConsolidationStatus,
  getSQLOMetrics,
  isLaranjaAvailable,
  getLaranjaHealth,
} from './sqlo';

export type {
  EpisodicMemory,
  RBACRole,
  RBACUser,
} from './sqlo';

// ============================================================================
// Health Check - All Nodes
// ============================================================================

/**
 * Check health of all 5 nodes
 *
 * @returns Promise<NodeHealthStatus>
 */
export async function checkAllNodesHealth(): Promise<{
  roxo: { available: boolean; status: string; version: string };
  verde: { available: boolean; status: string; version: string };
  vermelho: { available: boolean; status: string; version: string };
  cinza: { available: boolean; status: string; version: string };
  laranja: { available: boolean; status: string; version: string; performance_us?: number };
}> {
  const [roxo, verde, vermelho, cinza, laranja] = await Promise.all([
    (async () => {
      const { isRoxoAvailable, getRoxoHealth } = await import('./glass');
      return {
        available: isRoxoAvailable(),
        ...(await getRoxoHealth()),
      };
    })(),
    (async () => {
      const { isVerdeAvailable, getVerdeHealth } = await import('./gvcs');
      return {
        available: isVerdeAvailable(),
        ...(await getVerdeHealth()),
      };
    })(),
    (async () => {
      const { isVermelhoAvailable, getVermelhoHealth } = await import('./security');
      return {
        available: isVermelhoAvailable(),
        ...(await getVermelhoHealth()),
      };
    })(),
    (async () => {
      const { isCinzaAvailable, getCinzaHealth } = await import('./cognitive');
      return {
        available: isCinzaAvailable(),
        ...(await getCinzaHealth()),
      };
    })(),
    (async () => {
      const { isLaranjaAvailable, getLaranjaHealth } = await import('./sqlo');
      return {
        available: isLaranjaAvailable(),
        ...(await getLaranjaHealth()),
      };
    })(),
  ]);

  return {
    roxo,
    verde,
    vermelho,
    cinza,
    laranja,
  };
}

// ============================================================================
// Integration Status
// ============================================================================

// Import availability checkers for getIntegrationStatus()
import { isRoxoAvailable as checkRoxo } from './glass';
import { isVerdeAvailable as checkVerde } from './gvcs';
import { isVermelhoAvailable as checkVermelho } from './security';
import { isCinzaAvailable as checkCinza } from './cognitive';
import { isLaranjaAvailable as checkLaranja } from './sqlo';

/**
 * Get integration readiness status
 *
 * @returns Integration readiness summary
 */
export function getIntegrationStatus() {
  const nodes = [
    { name: 'ROXO', available: checkRoxo(), color: '🟣' },
    { name: 'VERDE', available: checkVerde(), color: '🟢' },
    { name: 'VERMELHO', available: checkVermelho(), color: '🔴' },
    { name: 'CINZA', available: checkCinza(), color: '🩶' },
    { name: 'LARANJA', available: checkLaranja(), color: '🟠' },
  ];

  const availableCount = nodes.filter(n => n.available).length;
  const totalCount = nodes.length;
  const progress = (availableCount / totalCount) * 100;

  return {
    nodes,
    available_count: availableCount,
    total_count: totalCount,
    progress_percent: progress,
    ready: availableCount === totalCount,
  };
}
