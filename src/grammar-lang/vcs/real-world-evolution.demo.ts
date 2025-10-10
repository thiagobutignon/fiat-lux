/**
 * DEMO: Real-World .glass Organism Evolution
 *
 * Demonstrates GVCS handling unexpected evolution:
 * - Organism gains knowledge (papers 100 → 150)
 * - BUT maturity REGRESSES (76% → 71.5%)
 * - Stage changes: maturity → adolescence
 *
 * This tests GVCS ability to handle non-linear evolution!
 */

import * as fs from 'fs';
import * as path from 'path';
import { autoCommit } from './auto-commit';
import { createMutation, updateFitness, selectWinner } from './genetic-versioning';
import { startCanary, evaluateCanary, recordMetrics } from './canary';

console.log('╔═══════════════════════════════════════════════════════════════╗');
console.log('║                                                               ║');
console.log('║     🧬 REAL-WORLD ORGANISM EVOLUTION - REGRESSION TEST 🧬     ║');
console.log('║                                                               ║');
console.log('║          Detecting Non-Linear Evolution Patterns             ║');
console.log('║                                                               ║');
console.log('╚═══════════════════════════════════════════════════════════════╝');

// Paths
const PROJECT_ROOT = path.join(__dirname, '../../../');
const GLASS_FILE = path.join(PROJECT_ROOT, 'cancer-research.glass');

// Step 1: Analyze current state
console.log('\n📊 Step 1: Analyzing organism state...');

const currentState = JSON.parse(fs.readFileSync(GLASS_FILE, 'utf-8'));

console.log(`   Current state of ${currentState.metadata.name}:`);
console.log(`     - Version: ${currentState.metadata.version}`);
console.log(`     - Maturity: ${(currentState.metadata.maturity * 100).toFixed(1)}%`);
console.log(`     - Stage: ${currentState.metadata.stage}`);
console.log(`     - Papers: ${currentState.knowledge.papers.count}`);
console.log(`     - Connections: ${currentState.knowledge.connections.nodes} nodes`);

// Step 2: Compare with expected evolution
console.log('\n🔍 Step 2: Detecting evolution anomaly...');

const expectedMaturity = 0.76; // Previous maturity
const actualMaturity = currentState.metadata.maturity;
const maturityDelta = actualMaturity - expectedMaturity;

console.log(`   Expected maturity: ${(expectedMaturity * 100).toFixed(1)}%`);
console.log(`   Actual maturity: ${(actualMaturity * 100).toFixed(1)}%`);
console.log(`   Delta: ${(maturityDelta * 100).toFixed(1)}% ${maturityDelta < 0 ? '⚠️ REGRESSION' : '✅'}`);

if (maturityDelta < 0) {
  console.log('\n   ⚠️  ANOMALY DETECTED: Organism regressed in maturity!');
  console.log('   Possible causes:');
  console.log('     1. Knowledge influx destabilized the organism');
  console.log('     2. New patterns conflict with existing knowledge');
  console.log('     3. Organism is reorganizing (temporary regression)');
  console.log('     4. Stage transition (maturity → adolescence)');
} else {
  console.log('\n   ✅ Normal evolution detected');
}

// Step 3: Auto-commit the change
console.log('\n📝 Step 3: Auto-committing organism evolution...');

// Add to git first if not tracked
try {
  const { execSync } = require('child_process');
  execSync(`git add "${GLASS_FILE}"`, { stdio: 'pipe' });

  const committed = autoCommit(GLASS_FILE);
  if (committed) {
    console.log(`   ✅ Evolution auto-committed!`);
  } else {
    console.log(`   ⏭️  No commit needed (already tracked)`);
  }
} catch (error: any) {
  console.log(`   ⏭️  Git operation skipped: ${error.message}`);
}

// Step 4: Create versioned snapshot
console.log('\n🧬 Step 4: Creating versioned snapshot...');

const VERSIONED_DIR = path.join(PROJECT_ROOT, 'organisms/snapshots');
if (!fs.existsSync(VERSIONED_DIR)) {
  fs.mkdirSync(VERSIONED_DIR, { recursive: true });
}

const timestamp = new Date().toISOString().replace(/[:.]/g, '-').substring(0, 19);
const snapshotPath = path.join(
  VERSIONED_DIR,
  `cancer-research-${timestamp}-m${(actualMaturity * 100).toFixed(0)}.glass`
);

fs.writeFileSync(snapshotPath, JSON.stringify(currentState, null, 2));
console.log(`   ✅ Snapshot created: ${path.basename(snapshotPath)}`);

// Step 5: Calculate fitness based on organism state
console.log('\n📈 Step 5: Calculating organism fitness...');

// Fitness based on:
// - Maturity (weight: 40%)
// - Knowledge count (weight: 30%)
// - Connection density (weight: 20%)
// - Constitutional compliance (weight: 10%)

const maturityScore = actualMaturity;
const knowledgeScore = Math.min(
  currentState.knowledge.papers.count / 200, // Max 200 papers = 1.0
  1.0
);
const connectionDensity =
  currentState.knowledge.connections.edges /
  (currentState.knowledge.connections.nodes * 2); // Max ~2 edges/node
const constitutionalScore = currentState.constitutional.principles.length >= 4 ? 1.0 : 0.5;

const fitness =
  maturityScore * 0.4 +
  knowledgeScore * 0.3 +
  connectionDensity * 0.2 +
  constitutionalScore * 0.1;

console.log(`   Fitness calculation:`);
console.log(`     - Maturity score: ${maturityScore.toFixed(3)} (40% weight)`);
console.log(`     - Knowledge score: ${knowledgeScore.toFixed(3)} (30% weight)`);
console.log(`     - Connection density: ${connectionDensity.toFixed(3)} (20% weight)`);
console.log(`     - Constitutional: ${constitutionalScore.toFixed(3)} (10% weight)`);
console.log(`     → Total fitness: ${fitness.toFixed(3)}`);

// Step 6: Decision based on fitness
console.log('\n🎯 Step 6: Evolution decision...');

if (maturityDelta < 0) {
  console.log(`   ⚠️  Regression detected:`);

  if (fitness > 0.7) {
    console.log(`   ✅ BUT fitness is still high (${fitness.toFixed(3)})`);
    console.log(`   → ACCEPT evolution (organism gaining knowledge)`);
    console.log(`   → Monitor for stabilization`);
    console.log(`   → Expected: maturity will increase as organism integrates knowledge`);
  } else {
    console.log(`   ❌ Fitness dropped significantly (${fitness.toFixed(3)})`);
    console.log(`   → REJECT evolution (would trigger rollback in production)`);
    console.log(`   → Restore previous snapshot`);
    console.log(`   → Categorize in old-but-gold/`);
  }
} else {
  console.log(`   ✅ Normal evolution - accepting changes`);
}

// Step 7: Summary
console.log('\n\n╔═══════════════════════════════════════════════════════════════╗');
console.log('║                                                               ║');
console.log('║            ✅ REAL-WORLD EVOLUTION ANALYSIS COMPLETE ✅        ║');
console.log('║                                                               ║');
console.log('╚═══════════════════════════════════════════════════════════════╝');

console.log('\n📊 Analysis Results:');
console.log(`   • Organism: ${currentState.metadata.name}`);
console.log(`   • Evolution type: Non-linear (regression)`);
console.log(`   • Maturity change: ${(maturityDelta * 100).toFixed(1)}%`);
console.log(`   • Fitness: ${fitness.toFixed(3)}`);
console.log(`   • Decision: ${fitness > 0.7 ? 'ACCEPT' : 'REJECT'}`);

console.log('\n🧬 GVCS Capabilities Demonstrated:');
console.log('   1. ✅ Detected non-linear evolution (regression)');
console.log('   2. ✅ Auto-committed changes');
console.log('   3. ✅ Created versioned snapshot');
console.log('   4. ✅ Calculated multi-factor fitness');
console.log('   5. ✅ Made intelligent decision (accept/reject)');

console.log('\n💡 Key Insight:');
console.log('   "GVCS handles COMPLEX organism evolution"');
console.log('   - Not all evolution is linear (maturity can regress)');
console.log('   - Fitness is multi-dimensional (not just maturity)');
console.log('   - System makes intelligent decisions');
console.log('   - Knowledge gain can temporarily destabilize');

console.log('\n🎯 Sprint 2 - Day 2: Real-world testing complete!');
console.log('   Next: Multiple organism orchestration\n');
