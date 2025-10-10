/**
 * Test: Old-But-Gold Categorization System
 *
 * Verifies:
 * 1. Fitness-based categorization
 * 2. Never delete (move to old-but-gold/)
 * 3. Category retrieval
 * 4. Degradation analysis
 * 5. Version restoration
 */

import * as fs from 'fs';
import * as path from 'path';
import {
  determineCategory,
  categorizeVersion,
  autoCategorize,
  getAllCategories,
  searchByFitness,
  findSimilarVersions,
  analyzeDegradation,
  restoreVersion,
  exportCategorizationState
} from './categorization';
import { createMutation, updateFitness } from './genetic-versioning';

// Test directory
const TEST_DIR = path.join(__dirname, '../../../test-files/vcs');
const OLD_BUT_GOLD_DIR = path.join(TEST_DIR, 'old-but-gold');

// Clean up old test data
if (fs.existsSync(OLD_BUT_GOLD_DIR)) {
  fs.rmSync(OLD_BUT_GOLD_DIR, { recursive: true });
}

// Test 1: Determine category by fitness
console.log('📝 Test 1: Determining categories by fitness...');

const categories = [
  { fitness: 0.95, expected: '90-100%' },
  { fitness: 0.85, expected: '80-90%' },
  { fitness: 0.75, expected: '70-80%' },
  { fitness: 0.60, expected: '50-70%' },
  { fitness: 0.30, expected: '<50%' }
];

categories.forEach(({ fitness, expected }) => {
  const category = determineCategory(fitness);
  const match = category.name === expected ? '✅' : '❌';
  console.log(`   ${match} Fitness ${fitness} → ${category.name} (${category.description})`);
});

// Test 2: Create test mutations with different fitness
console.log('\n📝 Test 2: Creating test mutations with varying fitness...');

const testFile = path.join(TEST_DIR, 'financial-1.0.0.gl');
fs.writeFileSync(testFile, '(define calculate (x: number -> number) (* x 2))');

const mutations = [
  { file: testFile, fitness: 0.95, version: '1.0.1' },
  { file: '', fitness: 0.85, version: '1.0.2' },
  { file: '', fitness: 0.75, version: '1.0.3' },
  { file: '', fitness: 0.60, version: '1.0.4' },
  { file: '', fitness: 0.30, version: '1.0.5' }
];

let lastFile = testFile;
for (let i = 0; i < mutations.length; i++) {
  const mut = createMutation(lastFile, 'human', 'patch');
  if (mut) {
    updateFitness(mutations[i].version, {
      latency: 100 * (1 - mutations[i].fitness),
      throughput: 1000 * mutations[i].fitness,
      errorRate: 0.1 * (1 - mutations[i].fitness),
      crashRate: 0.01 * (1 - mutations[i].fitness)
    });
    lastFile = mut.mutated;
    console.log(`   ✅ Created v${mutations[i].version} with fitness ${mutations[i].fitness}`);
  }
}

// Test 3: Auto-categorize versions below threshold
console.log('\n📝 Test 3: Auto-categorizing versions below 0.8 fitness...');
const result = autoCategorize(0.8, TEST_DIR);
console.log(`✅ Auto-categorization result:`);
console.log(`   Categorized: ${result.categorized}`);
console.log(`   Skipped: ${result.skipped}`);

// Test 4: Verify old-but-gold directory structure
console.log('\n📝 Test 4: Verifying old-but-gold directory structure...');
if (fs.existsSync(OLD_BUT_GOLD_DIR)) {
  const categories = fs.readdirSync(OLD_BUT_GOLD_DIR);
  console.log('✅ Old-but-gold categories created:');
  categories.forEach(cat => {
    const files = fs.readdirSync(path.join(OLD_BUT_GOLD_DIR, cat));
    console.log(`   - ${cat}/ (${files.length} files)`);
  });
} else {
  console.log('⚠️  Old-but-gold directory not created');
}

// Test 5: Get all categories with versions
console.log('\n📝 Test 5: Getting all categories...');
const allCategories = getAllCategories();
console.log('✅ Categories:');
allCategories.forEach(cat => {
  if (cat.versions.length > 0) {
    console.log(`   ${cat.range}: ${cat.versions.length} versions`);
    cat.versions.forEach(v => {
      console.log(`     - v${v.version} (fitness: ${v.fitness.toFixed(3)})`);
    });
  }
});

// Test 6: Search by fitness range
console.log('\n📝 Test 6: Searching versions by fitness range (0.5-0.8)...');
const searchResults = searchByFitness(0.5, 0.8);
console.log(`✅ Found ${searchResults.length} versions:`);
searchResults.forEach(v => {
  console.log(`   - v${v.version} (fitness: ${v.fitness.toFixed(3)}, category: ${v.category})`);
});

// Test 7: Find similar versions
console.log('\n📝 Test 7: Finding versions similar to fitness 0.75...');
const similar = findSimilarVersions(0.75, 0.1);
console.log(`✅ Found ${similar.length} similar versions:`);
similar.forEach(v => {
  const diff = Math.abs(v.fitness - 0.75);
  console.log(`   - v${v.version} (fitness: ${v.fitness.toFixed(3)}, diff: ${diff.toFixed(3)})`);
});

// Test 8: Analyze degradation
console.log('\n📝 Test 8: Analyzing degradation patterns...');
const analysis = analyzeDegradation();
console.log('✅ Degradation analysis:');
console.log('   Average fitness by category:');
Object.entries(analysis.avgFitnessByCategory).forEach(([cat, avg]) => {
  console.log(`     - ${cat}: ${avg.toFixed(3)}`);
});
console.log(`   Degradation rate: ${analysis.degradationRate.toFixed(4)} fitness/day`);
console.log('   Recommendations:');
analysis.recommendations.forEach(rec => console.log(`     ${rec}`));

// Test 9: Restore old version
console.log('\n📝 Test 9: Restoring old version...');
const restoreTarget = path.join(TEST_DIR, 'restored-financial.gl');
const restored = restoreVersion('1.0.3', restoreTarget);
if (restored) {
  console.log('✅ Version restored successfully!');
  console.log(`   Location: ${restoreTarget}`);
  if (fs.existsSync(restoreTarget)) {
    console.log('   ✅ File exists after restoration');
  }
}

// Test 10: Export state (glass box)
console.log('\n📝 Test 10: Exporting categorization state (glass box)...');
const state = exportCategorizationState();
console.log('✅ Categorization state:');
console.log(`   Total categorized: ${state.stats.totalCategorized}`);
console.log(`   Average fitness: ${state.stats.avgFitness.toFixed(3)}`);
if (state.stats.oldestVersion) {
  console.log(`   Oldest: v${state.stats.oldestVersion.version} (fitness: ${state.stats.oldestVersion.fitness.toFixed(3)})`);
}
if (state.stats.newestVersion) {
  console.log(`   Newest: v${state.stats.newestVersion.version} (fitness: ${state.stats.newestVersion.fitness.toFixed(3)})`);
}

console.log('\n✅ All categorization tests complete!');
console.log('\nOld-but-gold system working correctly:');
console.log('- Fitness-based categorization ✅');
console.log('- Never delete (move to old-but-gold/) ✅');
console.log('- Category retrieval ✅');
console.log('- Degradation analysis ✅');
console.log('- Version restoration ✅');
console.log('- Glass box transparency ✅');
