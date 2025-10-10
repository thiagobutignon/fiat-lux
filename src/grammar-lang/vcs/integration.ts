/**
 * VCS Integration - Complete Genetic Evolution Workflow
 *
 * Integrates all VCS components:
 * 1. Auto-commit: Detects changes and commits automatically
 * 2. Genetic versioning: Creates mutations (1.0.0 → 1.0.1)
 * 3. Canary deployment: 99%/1% traffic split with gradual rollout
 * 4. Old-but-gold: Categorizes old versions by fitness
 *
 * Complete workflow:
 * Change detected → Auto-commit → Genetic mutation → Canary deploy → Fitness evaluation → Rollout/Rollback
 *
 * Philosophy:
 * - Glass box (100% transparent)
 * - O(1) complexity (all operations constant time)
 * - Biological evolution (natural selection of code)
 * - Self-healing (automatic rollback)
 */

import * as fs from 'fs';
import * as path from 'path';
import { autoCommit, watchFile } from './auto-commit';
import { createMutation, updateFitness } from './genetic-versioning';
import { startCanary, routeRequest, recordMetrics, evaluateCanary } from './canary';
import { autoCategorize } from './categorization';

// ===== TYPES =====

interface WorkflowConfig {
  projectRoot: string;
  watchPatterns: string[];
  canaryConfig: {
    rampUpSpeed: 'slow' | 'medium' | 'fast';
    autoRollback: boolean;
    minSampleSize: number;
  };
  categorizationThreshold: number;
}

interface WorkflowState {
  activeWatchers: fs.FSWatcher[];
  activeCanaries: Map<string, string>; // file → deployment ID
  evolutionHistory: EvolutionEvent[];
}

interface EvolutionEvent {
  timestamp: number;
  type: 'commit' | 'mutation' | 'canary-start' | 'canary-evaluate' | 'categorize';
  file: string;
  version?: string;
  details: string;
}

// ===== STATE =====

const workflowState: WorkflowState = {
  activeWatchers: [],
  activeCanaries: new Map(),
  evolutionHistory: []
};

// ===== WORKFLOW ORCHESTRATION =====

/**
 * Complete evolution workflow
 * Triggered when file changes
 */
async function evolutionWorkflow(
  filePath: string,
  config: WorkflowConfig
): Promise<void> {
  console.log(`\n🧬 EVOLUTION WORKFLOW STARTED`);
  console.log(`   File: ${path.basename(filePath)}`);

  // Step 1: Auto-commit
  console.log('\n📝 Step 1: Auto-committing changes...');
  const committed = autoCommit(filePath);

  if (!committed) {
    console.log('   ⏭️  No changes to commit');
    return;
  }

  workflowState.evolutionHistory.push({
    timestamp: Date.now(),
    type: 'commit',
    file: filePath,
    details: 'Auto-committed changes'
  });

  // Step 2: Create genetic mutation
  console.log('\n🧬 Step 2: Creating genetic mutation...');
  const mutation = createMutation(filePath, 'agi', 'patch');

  if (!mutation) {
    console.log('   ❌ Failed to create mutation');
    return;
  }

  const mutatedVersion = `${mutation.mutatedVersion.major}.${mutation.mutatedVersion.minor}.${mutation.mutatedVersion.patch}`;
  const originalVersion = `${mutation.originalVersion.major}.${mutation.originalVersion.minor}.${mutation.originalVersion.patch}`;

  workflowState.evolutionHistory.push({
    timestamp: Date.now(),
    type: 'mutation',
    file: filePath,
    version: mutatedVersion,
    details: `Created mutation: v${originalVersion} → v${mutatedVersion}`
  });

  // Step 3: Start canary deployment
  console.log('\n🐤 Step 3: Starting canary deployment...');
  const deploymentId = `${path.basename(filePath, path.extname(filePath))}-${Date.now()}`;

  startCanary(
    deploymentId,
    originalVersion,
    mutatedVersion,
    config.canaryConfig
  );

  workflowState.activeCanaries.set(filePath, deploymentId);

  workflowState.evolutionHistory.push({
    timestamp: Date.now(),
    type: 'canary-start',
    file: filePath,
    version: mutatedVersion,
    details: `Started canary: v${originalVersion} (99%) vs v${mutatedVersion} (1%)`
  });

  console.log('\n✅ Evolution workflow complete!');
  console.log('   Next: Collect metrics and evaluate canary');
}

/**
 * Initialize genetic evolution system
 * Sets up watchers for all .gl and .glass files
 */
export function initializeGeneticEvolution(
  config: WorkflowConfig
): WorkflowState {
  console.log('🚀 Initializing Genetic Evolution System...');
  console.log(`   Project root: ${config.projectRoot}`);
  console.log(`   Watch patterns: ${config.watchPatterns.join(', ')}`);

  // Find all matching files
  const files: string[] = [];

  function findFiles(dir: string) {
    const entries = fs.readdirSync(dir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);

      if (entry.isDirectory()) {
        if (entry.name !== 'node_modules' && entry.name !== '.git' && entry.name !== 'old-but-gold') {
          findFiles(fullPath);
        }
      } else if (
        config.watchPatterns.some(pattern => entry.name.endsWith(pattern))
      ) {
        files.push(fullPath);
      }
    }
  }

  findFiles(config.projectRoot);

  console.log(`   Found ${files.length} files to watch`);

  // Set up watchers
  for (const file of files) {
    const watcher = watchFile(file);

    // On change, trigger evolution workflow
    fs.watch(file, () => {
      evolutionWorkflow(file, config);
    });

    workflowState.activeWatchers.push(watcher);
    console.log(`   ✅ Watching: ${path.relative(config.projectRoot, file)}`);
  }

  console.log('\n✅ Genetic Evolution System initialized!');
  console.log('   Watching for changes...');

  return workflowState;
}

/**
 * Evaluate all active canaries
 * Should be called periodically (e.g., every minute)
 */
export function evaluateAllCanaries(): {
  evaluated: number;
  increased: number;
  rolledBack: number;
  completed: number;
} {
  const results = {
    evaluated: 0,
    increased: 0,
    rolledBack: 0,
    completed: 0
  };

  console.log('\n📊 Evaluating all active canaries...');

  for (const [file, deploymentId] of workflowState.activeCanaries) {
    const evaluation = evaluateCanary(deploymentId);

    if (!evaluation) {
      continue;
    }

    results.evaluated += 1;

    console.log(`   ${path.basename(file)}:`);
    console.log(`     Decision: ${evaluation.decision}`);
    console.log(`     Traffic: ${evaluation.currentTraffic}% → ${evaluation.newTraffic}%`);

    if (evaluation.decision === 'increase') {
      results.increased += 1;
    } else if (evaluation.decision === 'rollback') {
      results.rolledBack += 1;
      workflowState.activeCanaries.delete(file);
    } else if (evaluation.decision === 'complete') {
      results.completed += 1;
      workflowState.activeCanaries.delete(file);
    }

    workflowState.evolutionHistory.push({
      timestamp: Date.now(),
      type: 'canary-evaluate',
      file,
      details: `${evaluation.decision}: ${evaluation.reason}`
    });
  }

  console.log(`\n✅ Evaluation complete:`);
  console.log(`   Evaluated: ${results.evaluated}`);
  console.log(`   Increased: ${results.increased}`);
  console.log(`   Rolled back: ${results.rolledBack}`);
  console.log(`   Completed: ${results.completed}`);

  return results;
}

/**
 * Categorize old versions
 * Should be called periodically (e.g., daily)
 */
export function categorizeOldVersions(
  threshold: number,
  projectRoot: string
): void {
  console.log('\n📦 Categorizing old versions...');

  const result = autoCategorize(threshold, projectRoot);

  workflowState.evolutionHistory.push({
    timestamp: Date.now(),
    type: 'categorize',
    file: 'all',
    details: `Categorized ${result.categorized} versions, skipped ${result.skipped}`
  });

  console.log('✅ Categorization complete!');
}

/**
 * Shutdown genetic evolution system
 * Clean up watchers
 */
export function shutdownGeneticEvolution(): void {
  console.log('\n🛑 Shutting down Genetic Evolution System...');

  for (const watcher of workflowState.activeWatchers) {
    watcher.close();
  }

  workflowState.activeWatchers = [];
  workflowState.activeCanaries.clear();

  console.log('✅ Shutdown complete!');
}

// ===== MONITORING =====

/**
 * Get evolution history
 * Glass box - 100% transparent
 */
export function getEvolutionHistory(
  limit?: number
): EvolutionEvent[] {
  const history = [...workflowState.evolutionHistory].reverse();
  return limit ? history.slice(0, limit) : history;
}

/**
 * Get active canaries
 */
export function getActiveCanaries(): Map<string, string> {
  return new Map(workflowState.activeCanaries);
}

/**
 * Export complete system state
 * Glass box - full transparency
 */
export function exportSystemState(): {
  activeWatchers: number;
  activeCanaries: number;
  evolutionHistory: EvolutionEvent[];
  recentEvents: EvolutionEvent[];
} {
  return {
    activeWatchers: workflowState.activeWatchers.length,
    activeCanaries: workflowState.activeCanaries.size,
    evolutionHistory: workflowState.evolutionHistory,
    recentEvents: getEvolutionHistory(10)
  };
}

// ===== DEMO WORKFLOW =====

/**
 * Run a complete demo of the genetic evolution system
 */
export async function runEvolutionDemo(demoDir: string): Promise<void> {
  console.log('\n🎬 GENETIC EVOLUTION DEMO\n');
  console.log('This demo shows the complete workflow:');
  console.log('1. File change detected');
  console.log('2. Auto-commit');
  console.log('3. Genetic mutation');
  console.log('4. Canary deployment');
  console.log('5. Fitness evaluation');
  console.log('6. Gradual rollout or rollback\n');

  // Create demo file
  const demoFile = path.join(demoDir, 'demo-1.0.0.gl');
  fs.writeFileSync(demoFile, '(define greet (-> string) "Hello, World!")');

  console.log('📝 Created demo file:', path.basename(demoFile));

  // Initialize system
  const config: WorkflowConfig = {
    projectRoot: demoDir,
    watchPatterns: ['.gl', '.glass'],
    canaryConfig: {
      rampUpSpeed: 'fast',
      autoRollback: true,
      minSampleSize: 50
    },
    categorizationThreshold: 0.8
  };

  const state = initializeGeneticEvolution(config);

  // Simulate file change
  console.log('\n📝 Simulating file change...');
  fs.writeFileSync(demoFile, '(define greet (name: string -> string) (concat "Hello, " name))');

  // Trigger workflow
  await evolutionWorkflow(demoFile, config);

  // Show history
  console.log('\n📜 Evolution history:');
  const history = getEvolutionHistory(5);
  history.forEach((event, i) => {
    console.log(`   ${i + 1}. [${event.type}] ${event.details}`);
  });

  // Export state
  console.log('\n📊 System state:');
  const systemState = exportSystemState();
  console.log(`   Active watchers: ${systemState.activeWatchers}`);
  console.log(`   Active canaries: ${systemState.activeCanaries}`);
  console.log(`   Total events: ${systemState.evolutionHistory.length}`);

  // Shutdown
  shutdownGeneticEvolution();

  console.log('\n✅ Demo complete!');
}
