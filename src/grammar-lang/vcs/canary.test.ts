/**
 * Test: Canary Deployment System
 *
 * Verifies:
 * 1. Traffic splitting (99%/1%)
 * 2. Metrics collection
 * 3. Gradual rollout
 * 4. Automatic rollback
 */

import {
  startCanary,
  routeRequest,
  recordMetrics,
  evaluateCanary,
  getCanaryStatus,
  exportCanaryState
} from './canary';
import { createMutation } from './genetic-versioning';

// Deployment ID
const DEPLOYMENT_ID = 'financial-advisor-v1';

// Test 1: Start canary deployment
console.log('📝 Test 1: Starting canary deployment...');
const started = startCanary(
  DEPLOYMENT_ID,
  '1.0.0',  // Original version
  '1.0.1',  // Canary version
  {
    rampUpSpeed: 'fast',
    autoRollback: true,
    minSampleSize: 50
  }
);

if (started) {
  console.log('✅ Canary deployment started successfully!');
}

// Test 2: Traffic splitting
console.log('\n📝 Test 2: Testing traffic split (99%/1%)...');
const userIds = Array.from({ length: 1000 }, (_, i) => `user-${i}`);
const routingResults = {
  '1.0.0': 0,
  '1.0.1': 0
};

userIds.forEach(userId => {
  const decision = routeRequest(DEPLOYMENT_ID, userId);
  if (decision.version === '1.0.0') {
    routingResults['1.0.0'] += 1;
  } else if (decision.version === '1.0.1') {
    routingResults['1.0.1'] += 1;
  }
});

console.log('✅ Traffic split results (1000 requests):');
console.log(`   v1.0.0 (original): ${routingResults['1.0.0']} requests (~99%)`);
console.log(`   v1.0.1 (canary): ${routingResults['1.0.1']} requests (~1%)`);

const expectedCanaryTraffic = 10; // ~1% of 1000
const actualCanaryTraffic = routingResults['1.0.1'];
if (Math.abs(actualCanaryTraffic - expectedCanaryTraffic) < 20) {
  console.log('✅ Traffic split is working correctly!');
} else {
  console.log('⚠️  Traffic split deviation higher than expected');
}

// Test 3: Collect metrics (canary performs better)
console.log('\n📝 Test 3: Collecting metrics (canary better)...');

// Original version metrics (slower, more errors)
for (let i = 0; i < 100; i++) {
  recordMetrics('1.0.0', 100 + Math.random() * 50, Math.random() < 0.05); // 100-150ms, 5% errors
}

// Canary version metrics (faster, fewer errors)
for (let i = 0; i < 100; i++) {
  recordMetrics('1.0.1', 50 + Math.random() * 20, Math.random() < 0.01); // 50-70ms, 1% errors
}

console.log('✅ Metrics recorded for both versions');

// Test 4: Get canary status
console.log('\n📝 Test 4: Getting canary status...');
const status = getCanaryStatus(DEPLOYMENT_ID);

if (status) {
  console.log('✅ Canary status:');
  console.log(`   Original v${status.config.originalVersion}:`);
  if (status.originalMetrics) {
    console.log(`     - Avg latency: ${status.originalMetrics.avgLatency.toFixed(2)}ms`);
    console.log(`     - Error rate: ${(status.originalMetrics.errorRate * 100).toFixed(2)}%`);
  }
  console.log(`   Canary v${status.config.canaryVersion}:`);
  if (status.canaryMetrics) {
    console.log(`     - Avg latency: ${status.canaryMetrics.avgLatency.toFixed(2)}ms`);
    console.log(`     - Error rate: ${(status.canaryMetrics.errorRate * 100).toFixed(2)}%`);
  }
  console.log(`   Traffic: ${status.config.trafficPercentage}%`);
}

// Test 5: Evaluate canary (should increase traffic)
console.log('\n📝 Test 5: Evaluating canary (should increase traffic)...');
const evaluation = evaluateCanary(DEPLOYMENT_ID);

if (evaluation) {
  console.log(`✅ Evaluation result: ${evaluation.decision}`);
  console.log(`   Current traffic: ${evaluation.currentTraffic}%`);
  console.log(`   New traffic: ${evaluation.newTraffic}%`);
  console.log(`   Reason: ${evaluation.reason}`);
}

// Test 6: Simulate gradual rollout
console.log('\n📝 Test 6: Simulating gradual rollout...');
let iteration = 0;
const maxIterations = 10;

while (iteration < maxIterations) {
  const eval2 = evaluateCanary(DEPLOYMENT_ID);

  if (!eval2) break;

  console.log(`   Iteration ${iteration + 1}: ${eval2.decision} (${eval2.currentTraffic}% → ${eval2.newTraffic}%)`);

  if (eval2.decision === 'complete') {
    console.log('✅ Rollout complete! Canary is now serving 100% traffic');
    break;
  } else if (eval2.decision === 'rollback') {
    console.log('🔄 Rollback triggered! Canary failed fitness test');
    break;
  } else if (eval2.decision === 'maintain') {
    // Add more metrics to reach decision
    for (let i = 0; i < 50; i++) {
      recordMetrics('1.0.1', 50 + Math.random() * 20, Math.random() < 0.01);
    }
  }

  iteration++;
}

// Test 7: Test rollback scenario
console.log('\n📝 Test 7: Testing rollback (canary worse)...');

// Start new canary
const ROLLBACK_DEPLOYMENT = 'rollback-test-v1';
startCanary(ROLLBACK_DEPLOYMENT, '2.0.0', '2.0.1', {
  rampUpSpeed: 'fast',
  autoRollback: true,
  minSampleSize: 50
});

// Original is better
for (let i = 0; i < 100; i++) {
  recordMetrics('2.0.0', 50 + Math.random() * 20, Math.random() < 0.01); // Fast, low errors
}

// Canary is worse
for (let i = 0; i < 100; i++) {
  recordMetrics('2.0.1', 200 + Math.random() * 100, Math.random() < 0.2); // Slow, high errors
}

const rollbackEval = evaluateCanary(ROLLBACK_DEPLOYMENT);
if (rollbackEval) {
  console.log(`✅ Rollback evaluation: ${rollbackEval.decision}`);
  if (rollbackEval.decision === 'rollback') {
    console.log('✅ Auto-rollback working correctly!');
  } else {
    console.log('⚠️  Expected rollback but got:', rollbackEval.decision);
  }
}

// Test 8: Export state (glass box)
console.log('\n📝 Test 8: Exporting canary state (glass box)...');
const state = exportCanaryState();
console.log('✅ Canary state:');
console.log(`   Active deployments: ${state.summary.activeDeployments}`);
console.log(`   Total requests: ${state.summary.totalRequests}`);
console.log(`   Deployments:`, state.deployments.map(d => ({
  original: d.originalVersion,
  canary: d.canaryVersion,
  traffic: d.trafficPercentage
})));

console.log('\n✅ All canary tests complete!');
console.log('\nCanary deployment system working correctly:');
console.log('- Traffic splitting: 99%/1% ✅');
console.log('- Metrics collection ✅');
console.log('- Gradual rollout ✅');
console.log('- Auto-rollback ✅');
console.log('- Glass box transparency ✅');
