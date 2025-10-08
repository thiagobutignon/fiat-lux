/**
 * Test Runner for SliceEvolutionEngine
 * TDD: RED phase - tests first
 */

import { SliceEvolutionEngine, SliceCandidate, SliceEvolution, EvolutionType, EvolutionTrigger } from '../core/slice-evolution-engine';
import { EpisodicMemory } from '../core/episodic-memory';
import { KnowledgeDistillation } from '../core/knowledge-distillation';
import { SliceRewriter } from '../core/slice-rewriter';
import { SliceNavigator } from '../core/slice-navigator';
import { ConstitutionEnforcer } from '../core/constitution';
import { AnthropicAdapter } from '../llm/anthropic-adapter';
import { Observability, LogLevel } from '../core/observability';
import fs from 'fs';
import path from 'path';
import os from 'os';

let passed = 0;
let failed = 0;

function test(name: string, fn: () => void | Promise<void>) {
  const result = fn();
  if (result instanceof Promise) {
    result
      .then(() => {
        console.log(`✅ ${name}`);
        passed++;
      })
      .catch((error: any) => {
        console.log(`❌ ${name}`);
        console.log(`   ${error.message}`);
        failed++;
      });
  } else {
    try {
      console.log(`✅ ${name}`);
      passed++;
    } catch (error: any) {
      console.log(`❌ ${name}`);
      console.log(`   ${error.message}`);
      failed++;
    }
  }
}

function assert(condition: boolean, message: string) {
  if (!condition) {
    throw new Error(message);
  }
}

console.log('🧪 Testing SliceEvolutionEngine\n');

// Setup test directories
const testDir = path.join(os.tmpdir(), 'evolution-engine-tests');
const slicesDir = path.join(testDir, 'slices');
const backupDir = path.join(testDir, 'backups');

function setupTestDirs() {
  if (fs.existsSync(testDir)) {
    fs.rmSync(testDir, { recursive: true });
  }
  fs.mkdirSync(testDir, { recursive: true });
  fs.mkdirSync(slicesDir, { recursive: true });
  fs.mkdirSync(backupDir, { recursive: true });
}

function cleanupTestDirs() {
  if (fs.existsSync(testDir)) {
    fs.rmSync(testDir, { recursive: true });
  }
}

// Setup
setupTestDirs();
const obs = new Observability(LogLevel.DEBUG);
const memory = new EpisodicMemory();
const apiKey = process.env.ANTHROPIC_API_KEY || 'test-key';
const llm = new AnthropicAdapter(apiKey);
const distillation = new KnowledgeDistillation(memory, llm, obs);
const rewriter = new SliceRewriter(slicesDir, backupDir, obs);
const navigator = new SliceNavigator(slicesDir);
const constitution = new ConstitutionEnforcer();
const engine = new SliceEvolutionEngine(
  memory,
  distillation,
  rewriter,
  navigator,
  constitution,
  obs
);

// Test 1: Analyze and propose evolutions
test('should analyze memory and propose candidates', async () => {
  // Add some test episodes
  memory.addEpisode(
    'What is compound interest?',
    'Compound interest is interest on interest',
    ['compound_interest', 'interest', 'finance'],
    ['financial'],
    ['financial'],
    0.001,
    true,
    0.9,
    [],
    []
  );

  memory.addEpisode(
    'How does compound interest work?',
    'It grows exponentially',
    ['compound_interest', 'interest', 'finance'],
    ['financial'],
    ['financial'],
    0.001,
    true,
    0.85,
    [],
    []
  );

  const candidates = await engine.analyzeAndPropose();

  assert(Array.isArray(candidates), 'Should return array of candidates');
  // May or may not have candidates depending on patterns found
});

// Test 2: Validate candidate has required fields
test('should validate candidate structure', async () => {
  const candidate: SliceCandidate = {
    id: 'test-candidate',
    type: 'new',
    title: 'Test Slice',
    description: 'A test slice',
    concepts: ['test'],
    content: 'id: test\ntitle: Test',
    supporting_episodes: ['ep1'],
    pattern: {
      concepts: ['test'],
      frequency: 1,
      domains: ['test'],
      confidence: 0.8,
      representative_queries: ['test query'],
      emergent_insight: 'test insight',
    },
    constitutional_score: 0.9,
    test_performance: {
      queries_tested: 0,
      accuracy_improvement: 0,
      cost_delta: 0,
    },
    should_deploy: true,
    reasoning: 'test',
  };

  assert(candidate.id !== undefined, 'Candidate should have ID');
  assert(candidate.type !== undefined, 'Candidate should have type');
  assert(candidate.content !== undefined, 'Candidate should have content');
  assert(candidate.should_deploy !== undefined, 'Candidate should have decision');
});

// Test 3: Deploy evolution
test('should deploy approved candidate', async () => {
  const candidate: SliceCandidate = {
    id: 'deploy-test',
    type: 'new',
    title: 'Deploy Test',
    description: 'Test deployment',
    concepts: ['test'],
    content: `id: deploy-test
title: Deploy Test
description: Test
concepts:
  - test`,
    supporting_episodes: [],
    pattern: {
      concepts: ['test'],
      frequency: 1,
      domains: ['test'],
      confidence: 0.9,
      representative_queries: [],
      emergent_insight: '',
    },
    constitutional_score: 0.9,
    test_performance: {
      queries_tested: 0,
      accuracy_improvement: 0,
      cost_delta: 0,
    },
    should_deploy: true,
    reasoning: 'test',
  };

  const evolution = await engine.deployEvolution(candidate);

  assert(evolution.id !== undefined, 'Evolution should have ID');
  assert(evolution.slice_id === 'deploy-test', 'Evolution should reference slice');
  assert(evolution.approved === true, 'Evolution should be approved');
  assert(evolution.deployed_at !== undefined, 'Evolution should have deployment time');
});

// Test 4: Get evolution history
test('should track evolution history', async () => {
  // Deploy an evolution first to ensure history has content
  const candidate: SliceCandidate = {
    id: 'history-test',
    type: 'new',
    title: 'History Test',
    description: 'Test history tracking',
    concepts: ['test'],
    content: `id: history-test
title: History Test
description: Test
concepts:
  - test`,
    supporting_episodes: [],
    pattern: {
      concepts: ['test'],
      frequency: 1,
      domains: ['test'],
      confidence: 0.9,
      representative_queries: [],
      emergent_insight: '',
    },
    constitutional_score: 0.9,
    test_performance: {
      queries_tested: 0,
      accuracy_improvement: 0,
      cost_delta: 0,
    },
    should_deploy: true,
    reasoning: 'test',
  };

  await engine.deployEvolution(candidate);

  const history = engine.getEvolutionHistory();

  assert(Array.isArray(history), 'History should be array');
  assert(history.length > 0, 'Should have at least one evolution');

  const latest = history[history.length - 1];
  assert(latest.timestamp !== undefined, 'Evolution should have timestamp');
  assert(latest.evolution_type !== undefined, 'Evolution should have type');
});

// Test 5: Evolution types
test('should support different evolution types', () => {
  assert(EvolutionType.CREATED !== undefined, 'Should have CREATED type');
  assert(EvolutionType.UPDATED !== undefined, 'Should have UPDATED type');
  assert(EvolutionType.MERGED !== undefined, 'Should have MERGED type');
  assert(EvolutionType.DEPRECATED !== undefined, 'Should have DEPRECATED type');
});

// Test 6: Evolution triggers
test('should support different triggers', () => {
  assert(EvolutionTrigger.SCHEDULED !== undefined, 'Should have SCHEDULED trigger');
  assert(EvolutionTrigger.THRESHOLD !== undefined, 'Should have THRESHOLD trigger');
  assert(EvolutionTrigger.MANUAL !== undefined, 'Should have MANUAL trigger');
  assert(EvolutionTrigger.CONTINUOUS !== undefined, 'Should have CONTINUOUS trigger');
});

// Test 7: Rollback capability
test('should rollback failed evolution', async () => {
  const candidate: SliceCandidate = {
    id: 'rollback-test',
    type: 'new',
    title: 'Rollback Test',
    description: 'Test rollback',
    concepts: ['test'],
    content: `id: rollback-test
title: Rollback Test`,
    supporting_episodes: [],
    pattern: {
      concepts: [],
      frequency: 0,
      domains: [],
      confidence: 0,
      representative_queries: [],
      emergent_insight: '',
    },
    constitutional_score: 0.9,
    test_performance: {
      queries_tested: 0,
      accuracy_improvement: 0,
      cost_delta: 0,
    },
    should_deploy: true,
    reasoning: 'test',
  };

  const evolution = await engine.deployEvolution(candidate);

  // Now rollback
  await engine.rollback(evolution.id);

  const history = engine.getEvolutionHistory();
  const rolledBack = history.find((e) => e.id === evolution.id);

  assert(rolledBack !== undefined, 'Evolution should exist in history');
  assert(rolledBack!.rolled_back === true, 'Evolution should be marked as rolled back');
});

// Test 8: Get metrics
test('should provide evolution metrics', () => {
  const metrics = engine.getMetrics();

  assert(metrics !== undefined, 'Metrics should exist');
  assert(typeof metrics.total_evolutions === 'number', 'Should count total evolutions');
  assert(typeof metrics.successful_deployments === 'number', 'Should count deployments');
  assert(typeof metrics.rollbacks === 'number', 'Should count rollbacks');
});

// Test 9: Validate constitutional compliance
test('should validate constitutional compliance', async () => {
  const candidate: SliceCandidate = {
    id: 'constitutional-test',
    type: 'new',
    title: 'Constitutional Test',
    description: 'Test',
    concepts: ['test'],
    content: `id: constitutional-test
title: Test`,
    supporting_episodes: [],
    pattern: {
      concepts: [],
      frequency: 0,
      domains: [],
      confidence: 0,
      representative_queries: [],
      emergent_insight: '',
    },
    constitutional_score: 0.5, // Low score
    test_performance: {
      queries_tested: 0,
      accuracy_improvement: 0,
      cost_delta: 0,
    },
    should_deploy: false, // Should not deploy
    reasoning: 'Low constitutional score',
  };

  // Engine should respect should_deploy = false
  try {
    await engine.deployEvolution(candidate);
    throw new Error('Should have rejected low-score candidate');
  } catch (error: any) {
    assert(
      error.message.includes('should_deploy'),
      'Should reject candidate with should_deploy=false'
    );
  }
});

// Test 10: Track performance over time
test('should track performance metrics over time', () => {
  const metrics = engine.getMetrics();

  assert(metrics.knowledge_growth !== undefined, 'Should track knowledge growth');
  assert(
    typeof metrics.knowledge_growth.slices_created === 'number',
    'Should count slices created'
  );
});

// Wait for async tests
setTimeout(() => {
  cleanupTestDirs();

  console.log('\n' + '='.repeat(70));
  console.log(`Total: ${passed + failed}`);
  console.log(`✅ Passed: ${passed}`);
  console.log(`❌ Failed: ${failed}`);
  console.log('='.repeat(70));

  if (failed > 0) {
    process.exit(1);
  }
}, 2000);
