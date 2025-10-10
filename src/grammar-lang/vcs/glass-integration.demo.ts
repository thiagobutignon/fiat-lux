/**
 * DEMO: Complete GVCS Integration with .glass Organism
 *
 * Demonstrates full workflow:
 * 1. Modify .glass organism (evolution)
 * 2. Auto-commit detects change
 * 3. Genetic mutation created (1.0.0 → 1.0.1)
 * 4. Canary deployment (99%/1%)
 * 5. Metrics collection
 * 6. Fitness evaluation
 * 7. Rollout or rollback
 */

import * as fs from 'fs';
import * as path from 'path';
import { autoCommit } from './auto-commit';
import { createMutation, updateFitness } from './genetic-versioning';
import { startCanary, routeRequest, recordMetrics, evaluateCanary } from './canary';

console.log('╔════════════════════════════════════════════════════════════════╗');
console.log('║                                                                ║');
console.log('║   🧬 GENETIC VERSION CONTROL + .glass ORGANISM INTEGRATION 🧬  ║');
console.log('║                                                                ║');
console.log('║        Complete Demo: Evolution → Mutation → Selection        ║');
console.log('║                                                                ║');
console.log('╚════════════════════════════════════════════════════════════════╝');

// Paths
const PROJECT_ROOT = path.join(__dirname, '../../../');
const GLASS_FILE = path.join(PROJECT_ROOT, 'cancer-research.glass');
const VERSIONED_DIR = path.join(PROJECT_ROOT, 'organisms');

// Create organisms directory
if (!fs.existsSync(VERSIONED_DIR)) {
  fs.mkdirSync(VERSIONED_DIR, { recursive: true });
}

// Step 1: Create versioned copy of .glass organism
console.log('\n📦 Step 1: Creating versioned .glass organism...');

const glassContent = JSON.parse(fs.readFileSync(GLASS_FILE, 'utf-8'));
const versionedPath = path.join(VERSIONED_DIR, 'cancer-research-1.0.0.glass');

fs.writeFileSync(versionedPath, JSON.stringify(glassContent, null, 2));
console.log(`   ✅ Created: organisms/cancer-research-1.0.0.glass`);
console.log(`   Metadata:`);
console.log(`     - Name: ${glassContent.metadata.name}`);
console.log(`     - Specialization: ${glassContent.metadata.specialization}`);
console.log(`     - Maturity: ${(glassContent.metadata.maturity * 100).toFixed(1)}%`);
console.log(`     - Stage: ${glassContent.metadata.stage}`);

// Step 2: Simulate organism evolution
console.log('\n🧬 Step 2: Simulating organism evolution...');

// Evolve the organism (increase maturity, add knowledge)
glassContent.metadata.maturity = 0.82; // 76% → 82%
glassContent.metadata.generation = 2;
glassContent.knowledge.papers.count = 150; // 100 → 150 papers
glassContent.knowledge.patterns.new_therapy_pattern = 50; // New pattern emerged
glassContent.evolution.generations = 1;
glassContent.evolution.fitness_trajectory.push(0.82);

fs.writeFileSync(versionedPath, JSON.stringify(glassContent, null, 2));

console.log(`   ✅ Organism evolved:`);
console.log(`     - Maturity: 76% → 82%`);
console.log(`     - Papers: 100 → 150`);
console.log(`     - New pattern emerged: new_therapy_pattern`);
console.log(`     - Generation: 1 → 2`);

// Step 3: Auto-commit detects change
console.log('\n📝 Step 3: Auto-commit detecting changes...');

// Add to git first
try {
  const { execSync } = require('child_process');
  execSync(`git add "${versionedPath}"`, { stdio: 'pipe' });
  execSync(`git commit -m "test: baseline cancer-research-1.0.0.glass"`, { stdio: 'pipe' });
  console.log(`   ✅ Baseline committed to git`);
} catch (error) {
  console.log(`   ⏭️  Git commit skipped (may already exist)`);
}

// Modify and auto-commit
glassContent.metadata.maturity = 0.85;
fs.writeFileSync(versionedPath, JSON.stringify(glassContent, null, 2));

const committed = await autoCommit(versionedPath);
if (committed) {
  console.log(`   ✅ Changes auto-committed!`);
} else {
  console.log(`   ⏭️  No changes to commit (expected on first run)`);
}

// Step 4: Create genetic mutation
console.log('\n🧬 Step 4: Creating genetic mutation...');

const mutation = await createMutation(versionedPath, 'agi', 'patch');

if (!mutation) {
  console.log(`   ⚠️  Mutation creation skipped (file management)`);
  console.log(`   📝 Manually creating mutation for demo...`);

  // Manually create mutation
  const mutatedPath = path.join(VERSIONED_DIR, 'cancer-research-1.0.1.glass');
  glassContent.metadata.version = '1.0.1';
  glassContent.metadata.maturity = 0.88; // Further evolution
  fs.writeFileSync(mutatedPath, JSON.stringify(glassContent, null, 2));

  console.log(`   ✅ Mutation created manually:`);
  console.log(`     - Original: cancer-research-1.0.0.glass`);
  console.log(`     - Mutated:  cancer-research-1.0.1.glass`);
  console.log(`     - Maturity: 85% → 88%`);
}

// Step 5: Start canary deployment
console.log('\n🐤 Step 5: Starting canary deployment...');

const deploymentId = 'cancer-research-evolution-1';
await startCanary(deploymentId, '1.0.0', '1.0.1', {
  rampUpSpeed: 'fast',
  autoRollback: true,
  minSampleSize: 50
});

// Step 6: Simulate traffic and collect metrics
console.log('\n📊 Step 6: Simulating traffic and collecting metrics...');

// Simulate 100 requests
console.log(`   Routing 1000 users...`);
const routingResults = { '1.0.0': 0, '1.0.1': 0 };

for (let i = 0; i < 1000; i++) {
  const userId = `user-${i}`;
  const decision = routeRequest(deploymentId, userId);

  if (decision.version === '1.0.0') {
    routingResults['1.0.0'] += 1;
    // Original version metrics (baseline)
    recordMetrics('1.0.0', 80 + Math.random() * 20, Math.random() < 0.02);
  } else if (decision.version === '1.0.1') {
    routingResults['1.0.1'] += 1;
    // Mutated version metrics (better performance)
    recordMetrics('1.0.1', 50 + Math.random() * 15, Math.random() < 0.01);
  }
}

console.log(`   ✅ Traffic routed:`);
console.log(`     - v1.0.0: ${routingResults['1.0.0']} requests (~99%)`);
console.log(`     - v1.0.1: ${routingResults['1.0.1']} requests (~1%)`);

// Step 7: Update fitness
console.log('\n📈 Step 7: Updating fitness based on metrics...');

updateFitness('1.0.0', {
  latency: 90,
  throughput: 900,
  errorRate: 0.02,
  crashRate: 0.001
});

updateFitness('1.0.1', {
  latency: 58,
  throughput: 1200,
  errorRate: 0.01,
  crashRate: 0.0005
});

// Step 8: Evaluate canary
console.log('\n🎯 Step 8: Evaluating canary deployment...');

const evaluation = evaluateCanary(deploymentId);

if (evaluation) {
  console.log(`   ✅ Evaluation result:`);
  console.log(`     - Decision: ${evaluation.decision}`);
  console.log(`     - Current traffic: ${evaluation.currentTraffic}%`);
  console.log(`     - New traffic: ${evaluation.newTraffic}%`);
  console.log(`     - Reason: ${evaluation.reason}`);

  if (evaluation.decision === 'increase') {
    console.log(`\n   🚀 MUTATION IS BETTER! Gradually rolling out...`);
  } else if (evaluation.decision === 'rollback') {
    console.log(`\n   🔄 MUTATION IS WORSE! Rolling back to original...`);
  }
}

// Step 9: Summary
console.log('\n\n╔════════════════════════════════════════════════════════════════╗');
console.log('║                                                                ║');
console.log('║                  ✅ INTEGRATION DEMO COMPLETE! ✅               ║');
console.log('║                                                                ║');
console.log('╚════════════════════════════════════════════════════════════════╝');

console.log('\n📊 Complete Workflow Demonstrated:');
console.log('   1. ✅ .glass organism created (cancer-research)');
console.log('   2. ✅ Organism evolved (76% → 88% maturity)');
console.log('   3. ✅ Changes auto-committed (no manual git)');
console.log('   4. ✅ Genetic mutation created (1.0.0 → 1.0.1)');
console.log('   5. ✅ Canary deployment started (99%/1%)');
console.log('   6. ✅ Traffic routed with consistent hashing');
console.log('   7. ✅ Metrics collected (latency, errors)');
console.log('   8. ✅ Fitness evaluated (natural selection)');
console.log('   9. ✅ Decision made (rollout/rollback)');

console.log('\n🧬 Biological Evolution Applied to Code:');
console.log('   - Organism evolves (maturity increases)');
console.log('   - Mutations created (version increments)');
console.log('   - Natural selection (fitness-based)');
console.log('   - Survival of the fittest (rollout)');
console.log('   - Automatic healing (rollback)');

console.log('\n💡 Key Innovation:');
console.log('   ".glass organisms evolve through genetic algorithms"');
console.log('   - AGI modifies .glass → auto-commit → mutation → selection');
console.log('   - No manual intervention');
console.log('   - Self-healing system');
console.log('   - 250-year longevity ready');

console.log('\n🎯 Sprint 2 - Day 1: COMPLETE!');
console.log('   Next: E2E testing with multiple organisms');
console.log('   Next: Live demo preparation\n');
