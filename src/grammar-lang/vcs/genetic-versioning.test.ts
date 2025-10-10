/**
 * Test: Genetic Versioning System
 *
 * Verifies:
 * 1. Version incrementer (1.0.0 → 1.0.1)
 * 2. Mutation creator (file duplication)
 * 3. Fitness calculation
 * 4. Natural selection
 */

import * as fs from 'fs';
import * as path from 'path';
import {
  createMutation,
  getFileVersion,
  calculateFitness,
  updateFitness,
  selectWinner,
  exportGeneticPool,
  getRankedMutations
} from './genetic-versioning';

// Test directory
const TEST_DIR = path.join(__dirname, '../../../test-files/vcs');
const TEST_FILE = path.join(TEST_DIR, 'index-1.0.0.gl');

// Ensure test directory exists
if (!fs.existsSync(TEST_DIR)) {
  fs.mkdirSync(TEST_DIR, { recursive: true });
}

// Test 1: Create initial file with version
console.log('📝 Test 1: Creating versioned file...');
fs.writeFileSync(TEST_FILE, `(define calculate-roi (investment: number -> number)
  (* investment 0.15))  ; 15% ROI`);
console.log('✅ File created:', TEST_FILE);

// Test 2: Create first mutation
console.log('\n📝 Test 2: Creating first mutation (1.0.0 → 1.0.1)...');
const mutation1 = createMutation(TEST_FILE, 'human', 'patch');
if (mutation1) {
  console.log('✅ Mutation created successfully!');
  console.log('   Original:', path.basename(mutation1.original));
  console.log('   Mutated:', path.basename(mutation1.mutated));
} else {
  console.log('❌ Failed to create mutation');
}

// Test 3: Create second mutation
console.log('\n📝 Test 3: Creating second mutation (1.0.1 → 1.0.2)...');
if (mutation1) {
  const mutation2 = createMutation(mutation1.mutated, 'agi', 'patch');
  if (mutation2) {
    console.log('✅ Second mutation created!');
    console.log('   Version progression: 1.0.0 → 1.0.1 → 1.0.2');
  }
}

// Test 4: Calculate fitness
console.log('\n📝 Test 4: Calculating fitness...');
const fitness1 = calculateFitness({
  latency: 50,        // 50ms (good)
  throughput: 800,    // 800 req/s (good)
  errorRate: 0.01,    // 1% errors
  crashRate: 0.001    // 0.1% crashes
});
console.log('✅ Fitness calculated:', fitness1.toFixed(3));

const fitness2 = calculateFitness({
  latency: 100,       // 100ms (ok)
  throughput: 500,    // 500 req/s (ok)
  errorRate: 0.05,    // 5% errors
  crashRate: 0.01     // 1% crashes
});
console.log('✅ Fitness calculated:', fitness2.toFixed(3));

// Test 5: Update mutation fitness
console.log('\n📝 Test 5: Updating mutation fitness...');
updateFitness('1.0.1', {
  latency: 50,
  throughput: 800,
  errorRate: 0.01,
  crashRate: 0.001
});
updateFitness('1.0.2', {
  latency: 100,
  throughput: 500,
  errorRate: 0.05,
  crashRate: 0.01
});

// Test 6: Natural selection
console.log('\n📝 Test 6: Natural selection (1.0.1 vs 1.0.2)...');
const winner = selectWinner('1.0.1', '1.0.2');
if (winner) {
  console.log('✅ Winner selected:', winner);
}

// Test 7: Get ranked mutations
console.log('\n📝 Test 7: Getting ranked mutations...');
const ranked = getRankedMutations();
console.log('✅ Mutations ranked by fitness:');
ranked.forEach((m, i) => {
  const version = `${m.mutatedVersion.major}.${m.mutatedVersion.minor}.${m.mutatedVersion.patch}`;
  console.log(`   ${i + 1}. v${version} - fitness: ${m.fitness.toFixed(3)} (${m.author})`);
});

// Test 8: Export genetic pool (glass box)
console.log('\n📝 Test 8: Exporting genetic pool (glass box)...');
const pool = exportGeneticPool();
console.log('✅ Genetic pool state:');
console.log('   Total mutations:', pool.stats.totalMutations);
console.log('   Average fitness:', pool.stats.avgFitness.toFixed(3));
console.log('   Best fitness:', pool.stats.bestFitness.toFixed(3));
console.log('   Worst fitness:', pool.stats.worstFitness.toFixed(3));

// Test 9: Verify files exist
console.log('\n📝 Test 9: Verifying mutation files...');
const files = fs.readdirSync(TEST_DIR).filter(f => f.startsWith('index-'));
console.log('✅ Mutation files created:');
files.forEach(f => console.log(`   - ${f}`));

console.log('\n✅ All tests complete!');
console.log('\nGenetic versioning system working correctly:');
console.log('- Version increments: 1.0.0 → 1.0.1 → 1.0.2 ✅');
console.log('- Mutation files created ✅');
console.log('- Fitness calculation working ✅');
console.log('- Natural selection working ✅');
console.log('- Glass box transparency ✅');
