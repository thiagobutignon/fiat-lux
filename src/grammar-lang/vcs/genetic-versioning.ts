/**
 * Genetic Versioning System - O(1) Version Mutations
 *
 * Applies genetic algorithm to code versioning:
 * - Auto-increment versions (1.0.0 → 1.0.1)
 * - Create mutation files automatically
 * - Track fitness of each version
 * - Natural selection of code
 *
 * Philosophy:
 * - Glass box (100% transparent)
 * - O(1) complexity (deterministic versioning)
 * - Biological evolution applied to code
 * - Never delete (old-but-gold categorization)
 */

import * as fs from 'fs';
import * as path from 'path';
import { vcsConstitutionalValidator } from './constitutional-integration';
import {
  createMutationRequest,
} from '../security/git-operation-guard';
import {
  CognitiveBehaviorGuard,
  shouldProceedWithOperation,
  getCognitiveBehaviorSummary,
  formatCognitiveBehaviorAnalysis,
} from '../security/cognitive-behavior-guard';
import { SecurityStorage } from '../security/security-storage';
import { UserSecurityProfiles } from '../security/types';

// ===== TYPES =====

interface Version {
  major: number;
  minor: number;
  patch: number;
}

interface Mutation {
  original: string;      // Original file path
  mutated: string;       // Mutated file path
  originalVersion: Version;
  mutatedVersion: Version;
  timestamp: number;
  author: 'human' | 'agi';
  fitness: number;       // 0.0 - 1.0
  traffic: number;       // Percentage of traffic (0-100)
}

// ===== STATE (Genetic Pool) =====

// O(1) lookup: version string → mutation
const mutations = new Map<string, Mutation>();

// O(1) lookup: file path → current version
const fileVersions = new Map<string, Version>();

// ===== VERSION PARSING =====

/**
 * Parse semver version string
 * O(1) - fixed format "X.Y.Z"
 */
function parseVersion(versionStr: string): Version {
  const parts = versionStr.split('.');
  return {
    major: parseInt(parts[0]) || 0,
    minor: parseInt(parts[1]) || 0,
    patch: parseInt(parts[2]) || 0
  };
}

/**
 * Stringify version to semver
 * O(1)
 */
function versionToString(version: Version): string {
  return `${version.major}.${version.minor}.${version.patch}`;
}

/**
 * Increment version (genetic mutation)
 * O(1) - simple arithmetic
 */
function incrementVersion(
  version: Version,
  type: 'major' | 'minor' | 'patch' = 'patch'
): Version {
  const newVersion = { ...version };

  switch (type) {
    case 'major':
      newVersion.major += 1;
      newVersion.minor = 0;
      newVersion.patch = 0;
      break;
    case 'minor':
      newVersion.minor += 1;
      newVersion.patch = 0;
      break;
    case 'patch':
      newVersion.patch += 1;
      break;
  }

  return newVersion;
}

// ===== FILE PATH MUTATIONS =====

/**
 * Extract version from file path
 * O(1) - regex match
 *
 * Examples:
 * - "index-1.0.0.gl" → "1.0.0"
 * - "financial-advisor-2.1.3.glass" → "2.1.3"
 */
function extractVersion(filePath: string): Version | null {
  const match = filePath.match(/[-_](\d+\.\d+\.\d+)\.(gl|glass)$/);
  if (match) {
    return parseVersion(match[1]);
  }
  return null;
}

/**
 * Generate mutated file path
 * O(1) - string replacement
 *
 * Examples:
 * - "index-1.0.0.gl" → "index-1.0.1.gl"
 * - "advisor-2.1.3.glass" → "advisor-2.1.4.glass"
 */
function generateMutatedPath(
  originalPath: string,
  newVersion: Version
): string {
  const currentVersion = extractVersion(originalPath);
  if (!currentVersion) {
    // No version in filename, add it
    const ext = path.extname(originalPath);
    const base = path.basename(originalPath, ext);
    const dir = path.dirname(originalPath);
    return path.join(dir, `${base}-${versionToString(newVersion)}${ext}`);
  }

  // Replace existing version
  const oldVersionStr = versionToString(currentVersion);
  const newVersionStr = versionToString(newVersion);
  return originalPath.replace(oldVersionStr, newVersionStr);
}

// ===== GENETIC MUTATION =====

/**
 * Create genetic mutation of a file
 * O(1) - copy + version increment
 *
 * Creates new version with incremented semver:
 * - Copies file content
 * - Increments version
 * - Creates mutation record
 *
 * Constitutional Enforcement: Validates mutation against Layer 1 Constitutional AI
 * before creating mutated file. Blocks mutations that violate universal principles.
 *
 * Dual-Layer Security (VERMELHO + CINZA):
 * - VERMELHO: Validates behavioral biometrics (duress/coercion detection)
 * - CINZA: Validates cognitive integrity (manipulation detection in mutation request)
 * Creates auto-snapshots when suspicious behavior or manipulation detected.
 */
export async function createMutation(
  filePath: string,
  author: 'human' | 'agi' = 'human',
  type: 'major' | 'minor' | 'patch' = 'patch',
  userProfiles?: UserSecurityProfiles,
  userId?: string,
  storage?: SecurityStorage
): Promise<Mutation | null> {
  // Check file exists
  if (!fs.existsSync(filePath)) {
    console.error(`❌ File not found: ${filePath}`);
    return null;
  }

  // Get current version
  let currentVersion = extractVersion(filePath);
  if (!currentVersion) {
    // First version
    currentVersion = { major: 1, minor: 0, patch: 0 };
  }

  // Increment version (genetic mutation)
  const newVersion = incrementVersion(currentVersion, type);

  // Generate mutated file path
  const mutatedPath = generateMutatedPath(filePath, newVersion);

  // ===== CONSTITUTIONAL VALIDATION (BEFORE MUTATION) =====
  // Integration with Layer 1 Constitutional AI System
  // Validates mutation before creating new version
  try {
    const constitutionalResult = await vcsConstitutionalValidator.validateMutation(
      filePath,
      mutatedPath,
      0.5, // Initial fitness (neutral)
      author
    );

    // Block mutation if constitutional violation detected
    if (!constitutionalResult.allowed) {
      console.error('❌ CONSTITUTIONAL VIOLATION - Mutation BLOCKED');
      console.error(vcsConstitutionalValidator.formatVCSReport(constitutionalResult));
      console.error(`   Original: ${filePath}`);
      console.error(`   Mutated: ${mutatedPath}`);
      console.error(`   Author: ${author}`);
      if (constitutionalResult.blockedReason) {
        console.error(`   Reason: ${constitutionalResult.blockedReason}`);
      }
      if (constitutionalResult.suggestedAction) {
        console.error(`   Suggested: ${constitutionalResult.suggestedAction}`);
      }
      return null; // Mutation rejected by constitutional enforcement
    }

    console.log('✅ Constitutional validation passed');
  } catch (constitutionalError) {
    console.error(`⚠️  Constitutional validation error: ${constitutionalError}`);
    console.error('   Proceeding with mutation (fail-open for availability)');
    // Fail-open: if constitutional system is down, allow mutation but log warning
  }

  // ===== DUAL-LAYER SECURITY VALIDATION (VERMELHO + CINZA + VERDE) =====
  // Integration with dual-layer security system
  // - VERMELHO: Behavioral biometrics (duress/coercion)
  // - CINZA: Cognitive manipulation detection
  // Validates user's behavioral + cognitive state before creating mutation
  if (userProfiles && userId && storage) {
    try {
      const cognitiveBehaviorGuard = new CognitiveBehaviorGuard(storage);

      // Create mutation request
      const mutationRequest = createMutationRequest(
        userId,
        filePath,
        author,
        versionToString(currentVersion),
        versionToString(newVersion)
      );

      // Validate dual-layer security (behavioral + cognitive)
      const securityResult = await cognitiveBehaviorGuard.validateGitOperation(
        mutationRequest,
        userProfiles
      );

      console.log('🔒 Dual-layer security validation:');
      console.log(getCognitiveBehaviorSummary(securityResult));

      // Block mutation if security validation failed
      if (!shouldProceedWithOperation(securityResult)) {
        console.error('❌ SECURITY VIOLATION - Mutation BLOCKED');
        console.error(`   Decision: ${securityResult.decision.toUpperCase()}`);
        console.error(`   Reason: ${securityResult.reason}`);
        console.error(`   Original: ${filePath}`);
        console.error(`   Mutated: ${mutatedPath}`);
        console.error(`   Author: ${author}`);

        // Show detailed cognitive-behavior analysis
        if (securityResult.cognitive_analysis) {
          console.error('\n' + formatCognitiveBehaviorAnalysis(securityResult.cognitive_analysis));
        }

        if (securityResult.snapshot_created) {
          console.error(`   📸 Duress snapshot saved: ${securityResult.snapshot_path}`);
        }

        if (securityResult.manipulation_snapshot_created) {
          console.error(`   🧠 Manipulation snapshot saved: ${securityResult.manipulation_snapshot_path}`);
        }

        return null; // Mutation rejected by dual-layer security
      }

      console.log('✅ Dual-layer security validation passed');
    } catch (securityError) {
      console.error(`⚠️  Dual-layer security validation error: ${securityError}`);
      console.error('   Proceeding with mutation (fail-open for availability)');
      // Fail-open: if security system is down, allow mutation but log warning
    }
  }

  // Copy file (genetic replication)
  const content = fs.readFileSync(filePath, 'utf-8');
  fs.writeFileSync(mutatedPath, content, 'utf-8');

  // Create mutation record
  const mutation: Mutation = {
    original: filePath,
    mutated: mutatedPath,
    originalVersion: currentVersion,
    mutatedVersion: newVersion,
    timestamp: Date.now(),
    author,
    fitness: 0.5, // Initial fitness (neutral)
    traffic: 1    // Start with 1% canary traffic
  };

  // Store mutation (O(1))
  const mutationKey = versionToString(newVersion);
  mutations.set(mutationKey, mutation);
  fileVersions.set(filePath, currentVersion);
  fileVersions.set(mutatedPath, newVersion);

  console.log(`🧬 Mutation created:`);
  console.log(`   Original: ${path.basename(filePath)} (v${versionToString(currentVersion)})`);
  console.log(`   Mutated:  ${path.basename(mutatedPath)} (v${versionToString(newVersion)})`);
  console.log(`   Author:   ${author}`);

  return mutation;
}

/**
 * Get current version of a file
 * O(1) - hash map lookup
 */
export function getFileVersion(filePath: string): Version | null {
  return fileVersions.get(filePath) || extractVersion(filePath);
}

/**
 * Get mutation by version
 * O(1) - hash map lookup
 */
export function getMutation(version: string): Mutation | undefined {
  return mutations.get(version);
}

// ===== FITNESS CALCULATION =====

/**
 * Calculate fitness of a version
 * O(1) - based on metrics
 *
 * Fitness = weighted average of:
 * - Performance (latency, throughput)
 * - Accuracy (error rate)
 * - Stability (crash rate)
 */
export function calculateFitness(
  metrics: {
    latency?: number;      // ms
    throughput?: number;   // req/s
    errorRate?: number;    // 0-1
    crashRate?: number;    // 0-1
  }
): number {
  // Normalize metrics to 0-1 scale
  const latencyScore = metrics.latency
    ? Math.max(0, 1 - metrics.latency / 1000) // < 1s is good
    : 0.5;

  const throughputScore = metrics.throughput
    ? Math.min(1, metrics.throughput / 1000) // > 1000 req/s is excellent
    : 0.5;

  const errorScore = metrics.errorRate
    ? 1 - metrics.errorRate // Lower is better
    : 0.5;

  const crashScore = metrics.crashRate
    ? 1 - metrics.crashRate // Lower is better
    : 0.5;

  // Weighted average
  const fitness =
    latencyScore * 0.3 +
    throughputScore * 0.3 +
    errorScore * 0.2 +
    crashScore * 0.2;

  return Math.max(0, Math.min(1, fitness)); // Clamp to [0, 1]
}

/**
 * Update mutation fitness
 * O(1) - hash map update
 */
export function updateFitness(
  version: string,
  metrics: Parameters<typeof calculateFitness>[0]
): boolean {
  const mutation = mutations.get(version);
  if (!mutation) {
    return false;
  }

  mutation.fitness = calculateFitness(metrics);
  mutations.set(version, mutation);

  console.log(`📊 Fitness updated: v${version} = ${mutation.fitness.toFixed(3)}`);
  return true;
}

// ===== NATURAL SELECTION =====

/**
 * Compare two versions and select winner
 * O(1) - fitness comparison
 *
 * Winner gets more traffic (natural selection)
 */
export function selectWinner(
  version1: string,
  version2: string
): string | null {
  const mutation1 = mutations.get(version1);
  const mutation2 = mutations.get(version2);

  if (!mutation1 || !mutation2) {
    return null;
  }

  // Compare fitness
  if (mutation1.fitness > mutation2.fitness) {
    console.log(`🏆 Winner: v${version1} (fitness: ${mutation1.fitness.toFixed(3)})`);
    return version1;
  } else if (mutation2.fitness > mutation1.fitness) {
    console.log(`🏆 Winner: v${version2} (fitness: ${mutation2.fitness.toFixed(3)})`);
    return version2;
  } else {
    console.log(`🤝 Tie: v${version1} = v${version2} (fitness: ${mutation1.fitness.toFixed(3)})`);
    return null; // Tie
  }
}

/**
 * Get all mutations sorted by fitness
 * O(n log n) where n = number of mutations (typically small)
 */
export function getRankedMutations(): Mutation[] {
  return Array.from(mutations.values()).sort((a, b) => b.fitness - a.fitness);
}

// ===== EXPORT STATE (Glass Box) =====

/**
 * Export genetic pool state
 * Glass box - 100% transparent
 */
export function exportGeneticPool(): {
  mutations: Mutation[];
  versions: Record<string, Version>;
  stats: {
    totalMutations: number;
    avgFitness: number;
    bestFitness: number;
    worstFitness: number;
  };
} {
  const mutationArray = Array.from(mutations.values());
  const fitnessValues = mutationArray.map(m => m.fitness);

  return {
    mutations: mutationArray,
    versions: Object.fromEntries(fileVersions),
    stats: {
      totalMutations: mutationArray.length,
      avgFitness: fitnessValues.reduce((a, b) => a + b, 0) / fitnessValues.length || 0,
      bestFitness: Math.max(...fitnessValues, 0),
      worstFitness: Math.min(...fitnessValues, 1)
    }
  };
}
