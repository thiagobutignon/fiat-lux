/**
 * Old-But-Gold Categorization System - O(1) Never Delete
 *
 * Categorizes old versions by fitness instead of deleting:
 * - 90-100%: Highly relevant still
 * - 80-90%: Still useful
 * - 70-80%: Specific cases
 * - 50-70%: Edge cases
 * - <50%: Rarely used
 *
 * Philosophy:
 * - Glass box (100% transparent)
 * - O(1) categorization (fitness-based)
 * - NEVER delete (categorical degradation)
 * - Learn from old versions
 * - Avoid systemic instability
 */

import * as fs from 'fs';
import * as path from 'path';
import { getMutation, exportGeneticPool } from './genetic-versioning';

// ===== TYPES =====

interface CategoryInfo {
  range: string;
  description: string;
  versions: VersionInfo[];
}

interface VersionInfo {
  version: string;
  fitness: number;
  filePath: string;
  timestamp: number;
  author: 'human' | 'agi';
  category: string;
}

// ===== CONSTANTS =====

const CATEGORIES = [
  { min: 0.9, max: 1.0, name: '90-100%', description: 'Highly relevant still' },
  { min: 0.8, max: 0.9, name: '80-90%', description: 'Still useful' },
  { min: 0.7, max: 0.8, name: '70-80%', description: 'Specific cases' },
  { min: 0.5, max: 0.7, name: '50-70%', description: 'Edge cases' },
  { min: 0.0, max: 0.5, name: '<50%', description: 'Rarely used' }
];

// ===== STATE =====

// Track categorized versions
const categorizedVersions = new Map<string, VersionInfo>();

// ===== CATEGORIZATION =====

/**
 * Determine category based on fitness
 * O(1) - simple comparison
 */
export function determineCategory(fitness: number): {
  name: string;
  description: string;
} {
  for (const cat of CATEGORIES) {
    if (fitness >= cat.min && fitness <= cat.max) {
      return { name: cat.name, description: cat.description };
    }
  }

  // Fallback to lowest category
  return { name: '<50%', description: 'Rarely used' };
}

/**
 * Categorize a version by fitness
 * O(1) - hash lookup + file move
 */
export function categorizeVersion(
  version: string,
  filePath: string,
  projectRoot: string = process.cwd()
): boolean {
  // Get mutation info
  const mutation = getMutation(version);

  if (!mutation) {
    console.error(`❌ Version not found: ${version}`);
    return false;
  }

  // Determine category
  const category = determineCategory(mutation.fitness);

  // Create old-but-gold directory structure
  const oldButGoldDir = path.join(projectRoot, 'old-but-gold', category.name);
  if (!fs.existsSync(oldButGoldDir)) {
    fs.mkdirSync(oldButGoldDir, { recursive: true });
  }

  // Generate new path
  const fileName = path.basename(filePath);
  const newPath = path.join(oldButGoldDir, fileName);

  // Move file (preserve, don't delete)
  if (fs.existsSync(filePath)) {
    fs.renameSync(filePath, newPath);
    console.log(`📦 Categorized: ${fileName}`);
    console.log(`   Fitness: ${mutation.fitness.toFixed(3)}`);
    console.log(`   Category: ${category.name} - ${category.description}`);
    console.log(`   Location: ${oldButGoldDir}`);
  } else {
    console.warn(`⚠️  File not found: ${filePath}`);
    return false;
  }

  // Store version info
  const versionInfo: VersionInfo = {
    version,
    fitness: mutation.fitness,
    filePath: newPath,
    timestamp: mutation.timestamp,
    author: mutation.author,
    category: category.name
  };

  categorizedVersions.set(version, versionInfo);

  return true;
}

/**
 * Auto-categorize all versions below a threshold
 * O(n) where n = number of versions (typically small)
 */
export function autoCategorize(
  fitnessThreshold: number = 0.8,
  projectRoot: string = process.cwd()
): {
  categorized: number;
  skipped: number;
} {
  const pool = exportGeneticPool();
  let categorized = 0;
  let skipped = 0;

  console.log(`🔍 Auto-categorizing versions below ${fitnessThreshold} fitness...`);

  for (const mutation of pool.mutations) {
    if (mutation.fitness < fitnessThreshold) {
      const version = `${mutation.mutatedVersion.major}.${mutation.mutatedVersion.minor}.${mutation.mutatedVersion.patch}`;

      // Check if already categorized
      if (categorizedVersions.has(version)) {
        skipped += 1;
        continue;
      }

      // Categorize
      const success = categorizeVersion(version, mutation.mutated, projectRoot);
      if (success) {
        categorized += 1;
      } else {
        skipped += 1;
      }
    }
  }

  console.log(`✅ Auto-categorization complete:`);
  console.log(`   Categorized: ${categorized}`);
  console.log(`   Skipped: ${skipped}`);

  return { categorized, skipped };
}

// ===== RETRIEVAL =====

/**
 * Get all versions in a category
 * O(n) where n = number of categorized versions
 */
export function getVersionsByCategory(categoryName: string): VersionInfo[] {
  return Array.from(categorizedVersions.values()).filter(
    v => v.category === categoryName
  );
}

/**
 * Get all categories with versions
 * O(n) where n = number of categorized versions
 */
export function getAllCategories(): CategoryInfo[] {
  return CATEGORIES.map(cat => ({
    range: cat.name,
    description: cat.description,
    versions: getVersionsByCategory(cat.name)
  }));
}

/**
 * Search old versions by fitness range
 * O(n) where n = number of categorized versions
 */
export function searchByFitness(
  minFitness: number,
  maxFitness: number
): VersionInfo[] {
  return Array.from(categorizedVersions.values()).filter(
    v => v.fitness >= minFitness && v.fitness <= maxFitness
  );
}

/**
 * Find similar old versions for regression analysis
 * O(n) where n = number of categorized versions
 *
 * Useful for understanding why performance degraded
 */
export function findSimilarVersions(
  targetFitness: number,
  tolerance: number = 0.1
): VersionInfo[] {
  const min = targetFitness - tolerance;
  const max = targetFitness + tolerance;

  return searchByFitness(min, max).sort((a, b) => {
    const distA = Math.abs(a.fitness - targetFitness);
    const distB = Math.abs(b.fitness - targetFitness);
    return distA - distB;
  });
}

// ===== INSIGHTS =====

/**
 * Analyze degradation patterns
 * O(n) where n = number of categorized versions
 *
 * Learn from old versions to understand what went wrong
 */
export function analyzeDegradation(): {
  avgFitnessByCategory: Record<string, number>;
  degradationRate: number;
  recommendations: string[];
} {
  const categories = getAllCategories();
  const avgFitnessByCategory: Record<string, number> = {};

  for (const cat of categories) {
    if (cat.versions.length > 0) {
      const avgFitness = cat.versions.reduce((sum, v) => sum + v.fitness, 0) / cat.versions.length;
      avgFitnessByCategory[cat.range] = avgFitness;
    }
  }

  // Calculate degradation rate (fitness decrease over time)
  const sorted = Array.from(categorizedVersions.values()).sort(
    (a, b) => a.timestamp - b.timestamp
  );

  let degradationRate = 0;
  if (sorted.length >= 2) {
    const oldest = sorted[0];
    const newest = sorted[sorted.length - 1];
    const timeDiff = (newest.timestamp - oldest.timestamp) / (1000 * 60 * 60 * 24); // days
    const fitnessDiff = newest.fitness - oldest.fitness;
    degradationRate = timeDiff > 0 ? fitnessDiff / timeDiff : 0;
  }

  // Generate recommendations
  const recommendations: string[] = [];

  if (degradationRate < -0.01) {
    recommendations.push('⚠️  Negative trend: Fitness decreasing over time');
    recommendations.push('💡 Review recent changes and consider rollback');
  }

  const lowFitnessCount = categorizedVersions.size > 0
    ? Array.from(categorizedVersions.values()).filter(v => v.fitness < 0.5).length
    : 0;
  const lowFitnessPercent = categorizedVersions.size > 0
    ? lowFitnessCount / categorizedVersions.size
    : 0;

  if (lowFitnessPercent > 0.3) {
    recommendations.push('⚠️  30%+ versions in low fitness categories');
    recommendations.push('💡 Consider major refactoring or architecture change');
  }

  if (recommendations.length === 0) {
    recommendations.push('✅ Healthy version distribution');
  }

  return {
    avgFitnessByCategory,
    degradationRate,
    recommendations
  };
}

// ===== RESTORE =====

/**
 * Restore an old version from old-but-gold
 * O(1) - file copy
 *
 * Sometimes old versions are better for specific cases
 */
export function restoreVersion(
  version: string,
  targetPath: string
): boolean {
  const versionInfo = categorizedVersions.get(version);

  if (!versionInfo) {
    console.error(`❌ Version not found in old-but-gold: ${version}`);
    return false;
  }

  if (!fs.existsSync(versionInfo.filePath)) {
    console.error(`❌ File not found: ${versionInfo.filePath}`);
    return false;
  }

  // Copy (don't move - preserve old-but-gold)
  fs.copyFileSync(versionInfo.filePath, targetPath);

  console.log(`🔄 Restored version: ${version}`);
  console.log(`   From: ${versionInfo.filePath}`);
  console.log(`   To: ${targetPath}`);
  console.log(`   Fitness: ${versionInfo.fitness.toFixed(3)}`);

  return true;
}

// ===== EXPORT (Glass Box) =====

/**
 * Export categorization state
 * Glass box - 100% transparent
 */
export function exportCategorizationState(): {
  categories: CategoryInfo[];
  analysis: ReturnType<typeof analyzeDegradation>;
  stats: {
    totalCategorized: number;
    avgFitness: number;
    oldestVersion: VersionInfo | null;
    newestVersion: VersionInfo | null;
  };
} {
  const categories = getAllCategories();
  const analysis = analyzeDegradation();

  const sorted = Array.from(categorizedVersions.values()).sort(
    (a, b) => a.timestamp - b.timestamp
  );

  const totalCategorized = categorizedVersions.size;
  const avgFitness = totalCategorized > 0
    ? Array.from(categorizedVersions.values()).reduce((sum, v) => sum + v.fitness, 0) / totalCategorized
    : 0;

  return {
    categories,
    analysis,
    stats: {
      totalCategorized,
      avgFitness,
      oldestVersion: sorted[0] || null,
      newestVersion: sorted[sorted.length - 1] || null
    }
  };
}
