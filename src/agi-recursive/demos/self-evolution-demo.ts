/**
 * Self-Evolution Demo
 *
 * Demonstrates the complete self-evolution cycle:
 * 1. User queries create episodic memories
 * 2. System analyzes patterns in memory
 * 3. System proposes evolution candidates
 * 4. System deploys new/updated slices
 * 5. System learns and improves continuously
 *
 * This is the answer to: "How can the system REWRITE its own slices
 * based on what it learns from episodic memory?"
 */

import { EpisodicMemory } from '../core/episodic-memory';
import { KnowledgeDistillation } from '../core/knowledge-distillation';
import { SliceRewriter } from '../core/slice-rewriter';
import { SliceNavigator } from '../core/slice-navigator';
import { ConstitutionEnforcer } from '../core/constitution';
import { SliceEvolutionEngine, EvolutionTrigger } from '../core/slice-evolution-engine';
import { AnthropicAdapter } from '../llm/anthropic-adapter';
import { Observability, LogLevel } from '../core/observability';
import fs from 'fs';
import path from 'path';
import os from 'os';

// ============================================================================
// Setup
// ============================================================================

console.log('🧬 Self-Evolution Demo\n');
console.log('Demonstrating how the AGI system rewrites its own knowledge slices');
console.log('based on patterns learned from episodic memory.\n');
console.log('='.repeat(70));

// Create demo directories
const demoDir = path.join(os.tmpdir(), 'self-evolution-demo');
const slicesDir = path.join(demoDir, 'slices');
const backupDir = path.join(demoDir, 'backups');

function setupDirs() {
  if (fs.existsSync(demoDir)) {
    fs.rmSync(demoDir, { recursive: true });
  }
  fs.mkdirSync(demoDir, { recursive: true });
  fs.mkdirSync(slicesDir, { recursive: true });
  fs.mkdirSync(backupDir, { recursive: true });
}

setupDirs();

// Initialize components
const obs = new Observability(LogLevel.INFO);
const memory = new EpisodicMemory();

// For demo: use mock LLM if API key not provided
const apiKey = process.env.ANTHROPIC_API_KEY;
if (!apiKey) {
  console.log('⚠️  ANTHROPIC_API_KEY not set - using mock synthesizer for demo\n');
}

const llm = new AnthropicAdapter(apiKey || 'demo-key');
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

// ============================================================================
// Main Demo Function
// ============================================================================

async function runDemo() {
// ============================================================================
// Phase 1: Simulate User Queries (Building Episodic Memory)
// ============================================================================

console.log('\n📝 Phase 1: Building Episodic Memory');
console.log('-'.repeat(70));

console.log('\nUser asks 6 questions about compound interest...\n');

const queries = [
  {
    query: 'What is compound interest?',
    response: 'Compound interest is interest calculated on the initial principal and accumulated interest from previous periods.',
    concepts: ['compound_interest', 'finance', 'interest'],
    success: true,
    confidence: 0.9,
  },
  {
    query: 'How does compound interest work?',
    response: 'Compound interest works by adding earned interest back to the principal, so future interest is calculated on a larger amount, creating exponential growth.',
    concepts: ['compound_interest', 'finance', 'interest'],
    success: true,
    confidence: 0.85,
  },
  {
    query: 'Explain compound interest to me',
    response: 'Compound interest means earning interest on your interest. Your money grows faster over time because returns are reinvested.',
    concepts: ['compound_interest', 'finance', 'interest'],
    success: true,
    confidence: 0.88,
  },
  {
    query: 'What is the compound interest formula?',
    response: 'The formula is A = P(1 + r/n)^(nt), where A is final amount, P is principal, r is annual rate, n is compounds per year, t is years.',
    concepts: ['compound_interest', 'finance', 'interest'],
    success: true,
    confidence: 0.95,
  },
  {
    query: 'Why is compound interest important?',
    response: 'Compound interest is important for wealth building. It creates exponential growth - the earlier you start, the more powerful it becomes.',
    concepts: ['compound_interest', 'finance', 'interest'],
    success: true,
    confidence: 0.92,
  },
  {
    query: 'How can I benefit from compound interest?',
    response: 'Benefit from compound interest by investing early, reinvesting returns, and giving your investments time to grow exponentially.',
    concepts: ['compound_interest', 'finance', 'interest'],
    success: true,
    confidence: 0.91,
  },
];

queries.forEach((q, i) => {
  memory.addEpisode(
    q.query,
    q.response,
    q.concepts,
    ['financial'],
    ['financial_advisor'],
    0.001,
    q.success,
    q.confidence,
    [],
    []
  );

  console.log(`✓ Query ${i + 1}: "${q.query}"`);
  console.log(`  Concepts: ${q.concepts.join(', ')}`);
  console.log(`  Confidence: ${(q.confidence * 100).toFixed(0)}%\n`);
});

// Show memory stats
const memStats = memory.getStats();
console.log('\n📊 Memory Statistics:');
console.log(`  Total Episodes: ${memStats.total_episodes}`);
console.log(`  Total Concepts: ${memStats.total_concepts}`);
console.log(`  Success Rate: ${(memStats.success_rate * 100).toFixed(1)}%`);
console.log(`  Average Confidence: ${(memStats.average_confidence * 100).toFixed(1)}%`);

// ============================================================================
// Phase 2: Pattern Discovery
// ============================================================================

console.log('\n\n🔍 Phase 2: Discovering Patterns in Memory');
console.log('-'.repeat(70));

const patterns = await distillation.discoverPatterns(memory.query({}), 2);

console.log(`\nFound ${patterns.length} recurring patterns:\n`);

patterns.forEach((pattern, i) => {
  console.log(`Pattern ${i + 1}:`);
  console.log(`  Concepts: ${pattern.concepts.join(', ')}`);
  console.log(`  Frequency: ${pattern.frequency} occurrences`);
  console.log(`  Confidence: ${(pattern.confidence * 100).toFixed(1)}%`);
  console.log(`  Domains: ${pattern.domains.join(', ')}`);
  console.log(`  Example Query: "${pattern.representative_queries[0]}"`);
  console.log();
});

// ============================================================================
// Phase 3: Evolution Proposal
// ============================================================================

console.log('\n💡 Phase 3: Proposing Evolution Candidates');
console.log('-'.repeat(70));

let candidates;

// If no API key, manually create candidates for demo
if (!apiKey && patterns.length > 0) {
  console.log('\n(Using manual candidate generation for demo)\n');

  candidates = patterns.map(pattern => ({
    id: pattern.concepts.join('-').replace(/[^a-z0-9-]/gi, '-'),
    type: 'new' as const,
    title: `${pattern.concepts.join(', ')} Knowledge`,
    description: `Knowledge about ${pattern.concepts.join(', ')} from ${pattern.frequency} episodes`,
    concepts: pattern.concepts,
    content: `id: ${pattern.concepts.join('-')}
title: ${pattern.concepts.join(' + ')} Knowledge Slice
description: Auto-generated knowledge from ${pattern.frequency} user queries
concepts:
${pattern.concepts.map(c => `  - ${c}`).join('\n')}
domains:
${pattern.domains.map(d => `  - ${d}`).join('\n')}
content: |
  This knowledge slice was automatically generated from ${pattern.frequency}
  user interactions about ${pattern.concepts.join(', ')}.

  Example questions users asked:
${pattern.representative_queries.slice(0, 3).map((q, i) => `  ${i + 1}. ${q}`).join('\n')}

  This demonstrates the system's ability to learn from experience and
  create new knowledge slices based on recurring patterns.

  Confidence: ${(pattern.confidence * 100).toFixed(1)}%
  Pattern Frequency: ${pattern.frequency} occurrences
`,
    supporting_episodes: [],
    pattern: pattern,
    constitutional_score: 0.9,
    test_performance: {
      queries_tested: 0,
      accuracy_improvement: 0,
      cost_delta: 0,
    },
    should_deploy: pattern.confidence >= 0.7,
    reasoning: `Pattern found ${pattern.frequency} times with ${(pattern.confidence * 100).toFixed(1)}% confidence`,
  }));
} else {
  candidates = await engine.analyzeAndPropose(EvolutionTrigger.THRESHOLD, 2);
}

console.log(`\nProposed ${candidates.length} evolution candidates:\n`);

candidates.forEach((candidate, i) => {
  console.log(`Candidate ${i + 1}:`);
  console.log(`  ID: ${candidate.id}`);
  console.log(`  Type: ${candidate.type}`);
  console.log(`  Title: ${candidate.title}`);
  console.log(`  Concepts: ${candidate.concepts.join(', ')}`);
  console.log(`  Constitutional Score: ${(candidate.constitutional_score * 100).toFixed(1)}%`);
  console.log(`  Should Deploy: ${candidate.should_deploy ? '✅ Yes' : '❌ No'}`);
  console.log(`  Reasoning: ${candidate.reasoning}`);
  console.log();
});

// ============================================================================
// Phase 4: Evolution Deployment
// ============================================================================

console.log('\n🚀 Phase 4: Deploying Approved Evolutions');
console.log('-'.repeat(70));

const evolutions = [];

for (const candidate of candidates) {
  if (candidate.should_deploy) {
    console.log(`\nDeploying: ${candidate.id}...`);

    const evolution = await engine.deployEvolution(candidate, EvolutionTrigger.THRESHOLD);
    evolutions.push(evolution);

    console.log(`✓ Deployed successfully!`);
    console.log(`  Evolution ID: ${evolution.id}`);
    console.log(`  Type: ${evolution.evolution_type}`);
    console.log(`  Slice created at: ${slicesDir}/${candidate.id}.yml`);

    // Show the generated slice content
    const sliceContent = await rewriter.readSlice(candidate.id);
    console.log('\n  Generated Slice Content:');
    console.log('  ' + '-'.repeat(68));
    sliceContent.split('\n').forEach(line => {
      console.log(`  ${line}`);
    });
    console.log('  ' + '-'.repeat(68));
  } else {
    console.log(`\nSkipping: ${candidate.id} (should_deploy = false)`);
  }
}

// ============================================================================
// Phase 5: Evolution Metrics & History
// ============================================================================

console.log('\n\n📈 Phase 5: Evolution Metrics & History');
console.log('-'.repeat(70));

const metrics = engine.getMetrics();

console.log('\nEvolution Metrics:');
console.log(`  Total Evolutions: ${metrics.total_evolutions}`);
console.log(`  Successful Deployments: ${metrics.successful_deployments}`);
console.log(`  Rollbacks: ${metrics.rollbacks}`);
console.log(`  Average Constitutional Score: ${(metrics.avg_constitutional_score * 100).toFixed(1)}%`);

console.log('\nEvolutions by Type:');
Object.entries(metrics.by_type).forEach(([type, count]) => {
  if (count > 0) {
    console.log(`  ${type}: ${count}`);
  }
});

console.log('\nEvolutions by Trigger:');
Object.entries(metrics.by_trigger).forEach(([trigger, count]) => {
  if (count > 0) {
    console.log(`  ${trigger}: ${count}`);
  }
});

console.log('\nKnowledge Growth:');
console.log(`  Slices Created: ${metrics.knowledge_growth.slices_created}`);
console.log(`  Slices Updated: ${metrics.knowledge_growth.slices_updated}`);
console.log(`  Slices Merged: ${metrics.knowledge_growth.slices_merged}`);
console.log(`  Slices Deprecated: ${metrics.knowledge_growth.slices_deprecated}`);

const history = engine.getEvolutionHistory();
console.log(`\n\n📜 Evolution History (${history.length} entries):\n`);

history.forEach((evo, i) => {
  console.log(`${i + 1}. ${evo.slice_id}`);
  console.log(`   Type: ${evo.evolution_type}`);
  console.log(`   Trigger: ${evo.trigger}`);
  console.log(`   Approved: ${evo.approved ? '✅' : '❌'}`);
  console.log(`   Timestamp: ${new Date(evo.timestamp).toISOString()}`);
  if (evo.backup_path) {
    console.log(`   Backup: ${path.basename(evo.backup_path)}`);
  }
  console.log();
});

// ============================================================================
// Phase 6: Demonstration of Continuous Learning
// ============================================================================

console.log('\n🔄 Phase 6: Continuous Learning Cycle');
console.log('-'.repeat(70));

console.log('\nSimulating more user queries to trigger further evolution...\n');

// Add more queries on a new topic
const newQueries = [
  {
    query: 'What is portfolio diversification?',
    response: 'Portfolio diversification means spreading investments across different asset classes to reduce risk. Don\'t put all your eggs in one basket.',
    concepts: ['diversification', 'investing', 'portfolio'],
    success: true,
    confidence: 0.87,
  },
  {
    query: 'Why should I diversify my portfolio?',
    response: 'Diversifying reduces risk because different assets perform differently in various market conditions. It smooths out returns over time.',
    concepts: ['diversification', 'investing', 'portfolio'],
    success: true,
    confidence: 0.90,
  },
  {
    query: 'How do I diversify my investments?',
    response: 'Diversify by investing in different asset classes (stocks, bonds, real estate), sectors, and geographic regions. Consider your risk tolerance and goals.',
    concepts: ['diversification', 'investing', 'portfolio'],
    success: true,
    confidence: 0.89,
  },
];

newQueries.forEach((q, i) => {
  memory.addEpisode(
    q.query,
    q.response,
    q.concepts,
    ['financial'],
    ['financial_advisor'],
    0.001,
    q.success,
    q.confidence,
    [],
    []
  );

  console.log(`✓ New Query ${i + 1}: "${q.query}"`);
});

console.log('\n🔍 Analyzing new patterns...');

const newPatterns = await distillation.discoverPatterns(memory.query({}), 2);

let newCandidates;

// If no API key, manually create candidates for demo
if (!apiKey && newPatterns.length > 0) {
  const unseenPatterns = newPatterns.filter(p =>
    !patterns.some(old => old.concepts.join('|') === p.concepts.join('|'))
  );

  if (unseenPatterns.length > 0) {
    newCandidates = unseenPatterns.map(pattern => ({
      id: pattern.concepts.join('-').replace(/[^a-z0-9-]/gi, '-'),
      type: 'new' as const,
      title: `${pattern.concepts.join(', ')} Knowledge`,
      description: `Knowledge about ${pattern.concepts.join(', ')} from ${pattern.frequency} episodes`,
      concepts: pattern.concepts,
      content: `id: ${pattern.concepts.join('-')}
title: ${pattern.concepts.join(' + ')} Knowledge Slice
description: Auto-generated knowledge from ${pattern.frequency} user queries
concepts:
${pattern.concepts.map(c => `  - ${c}`).join('\n')}
domains:
${pattern.domains.map(d => `  - ${d}`).join('\n')}
content: |
  This knowledge slice was automatically generated from continuous learning.

  Pattern discovered from ${pattern.frequency} user interactions.

  Example questions:
${pattern.representative_queries.slice(0, 3).map((q, i) => `  ${i + 1}. ${q}`).join('\n')}
`,
      supporting_episodes: [],
      pattern: pattern,
      constitutional_score: 0.9,
      test_performance: {
        queries_tested: 0,
        accuracy_improvement: 0,
        cost_delta: 0,
      },
      should_deploy: pattern.confidence >= 0.7,
      reasoning: `New pattern discovered through continuous learning`,
    }));
  } else {
    newCandidates = [];
  }
} else {
  newCandidates = await engine.analyzeAndPropose(EvolutionTrigger.CONTINUOUS, 2);
}

console.log(`\n💡 Proposed ${newCandidates.length} new evolution candidates`);

if (newCandidates.length > 0) {
  console.log('\nThe system continues to learn and evolve! 🌱');
} else {
  console.log('\nNo new patterns detected yet. More data needed for evolution. 📊');
}

// ============================================================================
// Summary
// ============================================================================

console.log('\n\n' + '='.repeat(70));
console.log('🎯 Demo Summary');
console.log('='.repeat(70));

console.log(`
✅ Demonstrated Complete Self-Evolution Cycle:

1. 📝 Episodic Memory: Stored ${memStats.total_episodes} user interactions
2. 🔍 Pattern Discovery: Found ${patterns.length} recurring patterns
3. 💡 Evolution Proposal: Generated ${candidates.length} evolution candidates
4. 🚀 Deployment: Created ${metrics.knowledge_growth.slices_created} new knowledge slices
5. 📈 Tracking: Complete metrics and history maintained
6. 🔄 Continuous Learning: System ready for ongoing evolution

Key Insight: The system now REWRITES ITS OWN SLICES based on learning
from episodic memory, creating a true self-improving AGI system! 🧬

Generated Slices Location: ${slicesDir}
Backups Location: ${backupDir}
`);

console.log('='.repeat(70));
console.log('\n✨ Self-evolution demo complete!\n');

// Cleanup (optional - comment out to inspect files)
// fs.rmSync(demoDir, { recursive: true });
}

// Run the demo
runDemo().catch((error) => {
  console.error('❌ Demo failed:', error);
  process.exit(1);
});
