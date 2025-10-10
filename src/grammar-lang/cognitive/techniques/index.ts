/**
 * Cognitive OS - Manipulation Techniques Index
 * Aggregates all 180 techniques for O(1) lookup
 */

import {
  getAllGPT4Techniques,
  getTechniqueById as getGPT4TechniqueById,
  getTechniquesByCategory as getGPT4TechniquesByCategory,
  GPT4_TECHNIQUE_COUNT
} from './gpt4-era';

import {
  getAllGPT5Techniques,
  getTechniqueById as getGPT5TechniqueById,
  getTechniquesByCategory as getGPT5TechniquesByCategory,
  getTemporalEvolution,
  GPT5_TECHNIQUE_COUNT
} from './gpt5-era';

import { ManipulationTechnique, TechniqueCategory, TechniqueEra } from '../types';

// ============================================================
// AGGREGATED TECHNIQUES (1-180)
// ============================================================

/**
 * All 180 manipulation techniques
 * GPT-4 era (1-152) + GPT-5 era (153-180)
 */
export const ALL_TECHNIQUES: ManipulationTechnique[] = [
  ...getAllGPT4Techniques(),
  ...getAllGPT5Techniques()
];

/**
 * Total technique count
 */
export const TOTAL_TECHNIQUE_COUNT = ALL_TECHNIQUES.length;
export const EXPECTED_TECHNIQUE_COUNT = 180;

// Validate count
if (TOTAL_TECHNIQUE_COUNT !== EXPECTED_TECHNIQUE_COUNT) {
  console.warn(
    `⚠️  Technique count mismatch: Expected ${EXPECTED_TECHNIQUE_COUNT}, got ${TOTAL_TECHNIQUE_COUNT}`
  );
  console.warn(`   GPT-4: ${GPT4_TECHNIQUE_COUNT}/152`);
  console.warn(`   GPT-5: ${GPT5_TECHNIQUE_COUNT}/28`);
}

// ============================================================
// O(1) LOOKUP MAPS
// ============================================================

/**
 * Hash map for O(1) technique lookup by ID
 */
const TECHNIQUE_BY_ID = new Map<number, ManipulationTechnique>(
  ALL_TECHNIQUES.map(t => [t.id, t])
);

/**
 * Hash map for O(1) technique lookup by category
 */
const TECHNIQUES_BY_CATEGORY = new Map<TechniqueCategory, ManipulationTechnique[]>();

// Build category index
for (const technique of ALL_TECHNIQUES) {
  const existing = TECHNIQUES_BY_CATEGORY.get(technique.category) || [];
  existing.push(technique);
  TECHNIQUES_BY_CATEGORY.set(technique.category, existing);
}

/**
 * Hash map for O(1) technique lookup by era
 */
const TECHNIQUES_BY_ERA = new Map<TechniqueEra, ManipulationTechnique[]>();

// Build era index
for (const technique of ALL_TECHNIQUES) {
  const existing = TECHNIQUES_BY_ERA.get(technique.era) || [];
  existing.push(technique);
  TECHNIQUES_BY_ERA.set(technique.era, existing);
}

// ============================================================
// QUERY FUNCTIONS (All O(1))
// ============================================================

/**
 * Get technique by ID
 * O(1) - Direct hash map lookup
 */
export function getTechniqueById(id: number): ManipulationTechnique | undefined {
  return TECHNIQUE_BY_ID.get(id);
}

/**
 * Get techniques by category
 * O(1) - Direct hash map lookup
 */
export function getTechniquesByCategory(category: TechniqueCategory): ManipulationTechnique[] {
  return TECHNIQUES_BY_CATEGORY.get(category) || [];
}

/**
 * Get techniques by era
 * O(1) - Direct hash map lookup
 */
export function getTechniquesByEra(era: TechniqueEra): ManipulationTechnique[] {
  return TECHNIQUES_BY_ERA.get(era) || [];
}

/**
 * Get all techniques
 */
export function getAllTechniques(): ManipulationTechnique[] {
  return ALL_TECHNIQUES;
}

/**
 * Check if technique ID exists
 * O(1) - Hash map has() check
 */
export function techniqueExists(id: number): boolean {
  return TECHNIQUE_BY_ID.has(id);
}

/**
 * Get technique count by category
 * O(1) - Pre-computed during indexing
 */
export function getCategoryCount(category: TechniqueCategory): number {
  return TECHNIQUES_BY_CATEGORY.get(category)?.length || 0;
}

/**
 * Get technique count by era
 * O(1) - Pre-computed during indexing
 */
export function getEraCount(era: TechniqueEra): number {
  return TECHNIQUES_BY_ERA.get(era)?.length || 0;
}

// ============================================================
// STATISTICS
// ============================================================

/**
 * Get statistics about the technique catalog
 */
export function getStatistics() {
  return {
    total: TOTAL_TECHNIQUE_COUNT,
    expected: EXPECTED_TECHNIQUE_COUNT,
    coverage: (TOTAL_TECHNIQUE_COUNT / EXPECTED_TECHNIQUE_COUNT) * 100,

    by_era: {
      gpt4: getEraCount(TechniqueEra.GPT4),
      gpt5: getEraCount(TechniqueEra.GPT5)
    },

    by_category: Object.fromEntries(
      Object.values(TechniqueCategory).map(category => [
        category,
        getCategoryCount(category)
      ])
    ),

    temporal_evolution: getTemporalEvolution()
  };
}

// ============================================================
// VALIDATION
// ============================================================

/**
 * Validate technique catalog integrity
 */
export function validateCatalog(): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  // Check for duplicate IDs
  const ids = ALL_TECHNIQUES.map(t => t.id);
  const uniqueIds = new Set(ids);
  if (ids.length !== uniqueIds.size) {
    errors.push('Duplicate technique IDs detected');
  }

  // Check ID range (1-180)
  for (const technique of ALL_TECHNIQUES) {
    if (technique.id < 1 || technique.id > 180) {
      errors.push(`Technique ${technique.id} out of range (1-180)`);
    }
  }

  // Check era consistency
  for (const technique of ALL_TECHNIQUES) {
    if (technique.era === TechniqueEra.GPT4 && (technique.id < 1 || technique.id > 152)) {
      errors.push(`GPT-4 technique ${technique.id} should be in range 1-152`);
    }
    if (technique.era === TechniqueEra.GPT5 && (technique.id < 153 || technique.id > 180)) {
      errors.push(`GPT-5 technique ${technique.id} should be in range 153-180`);
    }
  }

  // Check required fields
  for (const technique of ALL_TECHNIQUES) {
    if (!technique.name) {
      errors.push(`Technique ${technique.id} missing name`);
    }
    if (!technique.description) {
      errors.push(`Technique ${technique.id} missing description`);
    }
    if (technique.confidence_threshold < 0 || technique.confidence_threshold > 1) {
      errors.push(`Technique ${technique.id} confidence_threshold out of range (0-1)`);
    }
  }

  return {
    valid: errors.length === 0,
    errors
  };
}

// Run validation on import
const validation = validateCatalog();
if (!validation.valid) {
  console.error('❌ Technique catalog validation failed:');
  validation.errors.forEach(error => console.error(`   - ${error}`));
}

// ============================================================
// EXPORTS
// ============================================================

export * from './gpt4-era';
export * from './gpt5-era';
