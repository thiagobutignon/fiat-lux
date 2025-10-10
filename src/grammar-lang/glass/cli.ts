#!/usr/bin/env node
/**
 * Fiat CLI - Create and manage .glass digital organisms
 *
 * Commands:
 * - fiat create <name>         Create nascent organism
 * - fiat status <name>         Show organism status
 * - fiat inspect <name>        Inspect organism (glass box)
 * - fiat ingest <name> <src>   Ingest knowledge (DIA 2)
 * - fiat run <name>            Run organism (DIA 5)
 */

import { createGlassOrganism, loadGlassOrganism, GlassBuilder } from './builder';
import { GlassIngestion, IngestionProgress } from './ingestion';
import { PatternDetectionEngine } from './patterns';
import { CodeEmergenceEngine } from './emergence';
import { GlassRuntime } from './runtime';
import * as path from 'path';
import * as fs from 'fs';
import * as readline from 'readline';

async function main() {
  const args = process.argv.slice(2);
  const command = args[0];

  switch (command) {
    case 'create':
      await cmdCreate(args.slice(1));
      break;

    case 'status':
      await cmdStatus(args.slice(1));
      break;

    case 'inspect':
      await cmdInspect(args.slice(1));
      break;

    case 'ingest':
      await cmdIngest(args.slice(1));
      break;

    case 'analyze':
      await cmdAnalyze(args.slice(1));
      break;

    case 'emerge':
      await cmdEmerge(args.slice(1));
      break;

    case 'run':
      await cmdRun(args.slice(1));
      break;

    default:
      showHelp();
  }
}

/**
 * Create nascent organism
 */
async function cmdCreate(args: string[]) {
  if (args.length === 0) {
    console.error('Error: organism name required');
    console.log('Usage: fiat create <name>');
    process.exit(1);
  }

  const name = args[0];
  const specialization = args[1] || 'general';

  console.log(`Creating ${name}.glass...`);

  const builder = createGlassOrganism({
    name,
    specialization,
    constitutional: [
      'transparency',
      'honesty',
      'privacy',
      'safety'
    ]
  });

  const outputPath = path.join(process.cwd(), `${name}.glass`);
  await builder.save(outputPath);

  console.log('');
  console.log('Organism created! 🧬');
  console.log('');
  console.log('Next steps:');
  console.log(`  fiat status ${name}      # Check organism status`);
  console.log(`  fiat inspect ${name}     # Inspect organism (glass box)`);
  console.log(`  fiat ingest ${name} ...  # Ingest knowledge (coming DIA 2)`);
}

/**
 * Show organism status
 */
async function cmdStatus(args: string[]) {
  if (args.length === 0) {
    console.error('Error: organism name required');
    console.log('Usage: fiat status <name>');
    process.exit(1);
  }

  const name = args[0];
  const inputPath = path.join(process.cwd(), `${name}.glass`);

  const organism = await loadGlassOrganism(inputPath);

  console.log('');
  console.log(`Status: ${organism.metadata.name}.glass`);
  console.log('├── Maturity:', `${(organism.metadata.maturity * 100).toFixed(0)}%`);
  console.log('├── Stage:', organism.metadata.stage);
  console.log('├── Functions emerged:', organism.code.functions.length);
  console.log('├── Patterns detected:', Object.keys(organism.knowledge.patterns).length);
  console.log('├── Knowledge count:', organism.knowledge.papers.count);
  console.log('└── Generation:', organism.metadata.generation);
  console.log('');
}

/**
 * Inspect organism (glass box)
 */
async function cmdInspect(args: string[]) {
  if (args.length === 0) {
    console.error('Error: organism name required');
    console.log('Usage: fiat inspect <name>');
    process.exit(1);
  }

  const name = args[0];
  const inputPath = path.join(process.cwd(), `${name}.glass`);

  const organism = await loadGlassOrganism(inputPath);

  console.log('');
  console.log('🔍 Glass Box Inspection');
  console.log('');
  console.log('METADATA (Cell Identity):');
  console.log('  Format:', organism.metadata.format);
  console.log('  Type:', organism.metadata.type);
  console.log('  Name:', organism.metadata.name);
  console.log('  Version:', organism.metadata.version);
  console.log('  Created:', organism.metadata.created);
  console.log('  Specialization:', organism.metadata.specialization);
  console.log('  Maturity:', `${(organism.metadata.maturity * 100).toFixed(0)}%`);
  console.log('  Stage:', organism.metadata.stage);
  console.log('');

  console.log('MODEL (DNA):');
  console.log('  Architecture:', organism.model.architecture);
  console.log('  Parameters:', organism.model.parameters.toLocaleString());
  console.log('  Quantization:', organism.model.quantization);
  console.log('  Constitutional:', organism.model.constitutional_embedding ? '✅' : '❌');
  console.log('');

  console.log('KNOWLEDGE (RNA):');
  console.log('  Papers:', organism.knowledge.papers.count);
  console.log('  Patterns:', Object.keys(organism.knowledge.patterns).length);
  console.log('  Nodes:', organism.knowledge.connections.nodes);
  console.log('  Edges:', organism.knowledge.connections.edges);
  console.log('');

  console.log('CODE (Proteins):');
  console.log('  Functions:', organism.code.functions.length);
  if (organism.code.functions.length > 0) {
    organism.code.functions.forEach((fn, i) => {
      console.log(`  ${i + 1}. ${fn.name} - confidence: ${(fn.confidence * 100).toFixed(0)}%`);
    });
  } else {
    console.log('  (no functions emerged yet)');
  }
  console.log('');

  console.log('CONSTITUTIONAL (Membrane):');
  console.log('  Principles:', organism.constitutional.principles.join(', '));
  console.log('  Boundaries:');
  Object.entries(organism.constitutional.boundaries).forEach(([key, value]) => {
    console.log(`    ${key}: ${value ? '✅' : '❌'}`);
  });
  console.log('');

  console.log('EVOLUTION (Metabolism):');
  console.log('  Enabled:', organism.evolution.enabled ? '✅' : '❌');
  console.log('  Generations:', organism.evolution.generations);
  console.log('  Fitness:', organism.evolution.fitness_trajectory[organism.evolution.fitness_trajectory.length - 1]);
  console.log('');
}

/**
 * Ingest knowledge
 */
async function cmdIngest(args: string[]) {
  if (args.length < 2) {
    console.error('Error: organism name and source required');
    console.log('Usage: fiat ingest <name> --source <source>');
    console.log('');
    console.log('Sources:');
    console.log('  pubmed:<query>:<count>    e.g., pubmed:cancer+treatment:100');
    console.log('  arxiv:<query>:<count>     e.g., arxiv:oncology:50');
    console.log('  file:<path>               e.g., file:./papers.txt');
    console.log('  text:<content>            e.g., text:"Some text..."');
    process.exit(1);
  }

  const name = args[0];

  // Parse --source argument
  const sourceIndex = args.indexOf('--source');
  if (sourceIndex === -1) {
    console.error('Error: --source required');
    process.exit(1);
  }

  const sourceStr = args[sourceIndex + 1];
  const [type, ...rest] = sourceStr.split(':');

  let source: any = { type };

  switch (type) {
    case 'pubmed':
    case 'arxiv':
      source.query = rest[0];
      source.count = parseInt(rest[1]) || 100;
      break;

    case 'file':
      source.path = rest.join(':');
      break;

    case 'text':
      source.text = rest.join(':');
      break;

    default:
      console.error(`Unknown source type: ${type}`);
      process.exit(1);
  }

  const inputPath = path.join(process.cwd(), `${name}.glass`);

  console.log('');
  console.log(`Ingesting knowledge into ${name}.glass...`);
  console.log(`Source: ${type}`);
  if (source.query) console.log(`Query: ${source.query}`);
  if (source.count) console.log(`Count: ${source.count}`);
  console.log('');

  // Load organism
  const organism = await loadGlassOrganism(inputPath);

  // Create ingestion engine
  const ingestion = new GlassIngestion(organism);

  // Progress bar
  const progressBar = (progress: number) => {
    const total = 40;
    const filled = Math.floor(progress * total);
    const empty = total - filled;
    return '[' + '█'.repeat(filled) + '░'.repeat(empty) + ']';
  };

  // Ingest with progress
  await ingestion.ingest({
    source,
    onProgress: (progress: IngestionProgress) => {
      const percent = Math.floor(progress.maturity * 100);
      console.log(`${progress.stage.padEnd(12)} ${progressBar(progress.maturity)} ${percent}%`);
    }
  });

  // Save updated organism
  const updatedOrganism = ingestion.getOrganism();
  const json = JSON.stringify(updatedOrganism, null, 2);
  fs.writeFileSync(inputPath, json, 'utf-8');

  console.log('');
  console.log('✅ Ingestion complete!');
  console.log('');
  console.log('Updated organism:');
  console.log('├── Maturity:', `${(updatedOrganism.metadata.maturity * 100).toFixed(0)}%`);
  console.log('├── Stage:', updatedOrganism.metadata.stage);
  console.log('├── Papers:', updatedOrganism.knowledge.papers.count);
  console.log('├── Patterns detected:', Object.keys(updatedOrganism.knowledge.patterns).length);
  console.log('├── Knowledge graph:');
  console.log('│   ├── Nodes:', updatedOrganism.knowledge.connections.nodes);
  console.log('│   ├── Edges:', updatedOrganism.knowledge.connections.edges);
  console.log('│   └── Clusters:', updatedOrganism.knowledge.connections.clusters);
  console.log('└── Ready for pattern detection (DIA 3)');
  console.log('');
}

/**
 * Analyze patterns
 */
async function cmdAnalyze(args: string[]) {
  if (args.length === 0) {
    console.error('Error: organism name required');
    console.log('Usage: fiat analyze <name>');
    process.exit(1);
  }

  const name = args[0];
  const inputPath = path.join(process.cwd(), `${name}.glass`);

  console.log('');
  console.log(`🔬 Analyzing patterns in ${name}.glass...`);
  console.log('');

  // Load organism
  const organism = await loadGlassOrganism(inputPath);

  // Create pattern detection engine
  const engine = new PatternDetectionEngine(organism);

  // Analyze
  const analysis = engine.analyze();
  const summary = engine.getSummary();

  console.log('📊 Pattern Analysis Summary');
  console.log('');
  console.log('├── Total patterns:', summary.total_patterns);
  console.log('├── Emergence ready:', summary.emergence_ready);
  console.log('├── Clusters found:', summary.clusters);
  console.log('├── Correlations:', summary.correlations);
  console.log('└── Emergence candidates:', summary.emergence_candidates);
  console.log('');

  // Show enhanced patterns
  if (analysis.enhanced_patterns.length > 0) {
    console.log('🧬 Enhanced Patterns:');
    console.log('');
    for (const pattern of analysis.enhanced_patterns) {
      const readyMark = pattern.emergence_ready ? '🔥' : '  ';
      console.log(`${readyMark} ${pattern.type}`);
      console.log(`   ├── Frequency: ${pattern.frequency}`);
      console.log(`   ├── Confidence: ${(pattern.confidence * 100).toFixed(0)}%`);
      console.log(`   ├── Emergence score: ${(pattern.emergence_score * 100).toFixed(0)}%`);
      console.log(`   ├── Ready: ${pattern.emergence_ready ? 'YES ✅' : 'No'}`);
      console.log(`   └── Cluster: ${pattern.cluster || 'none'}`);
      console.log('');
    }
  }

  // Show clusters
  if (analysis.clusters.length > 0) {
    console.log('🎯 Pattern Clusters:');
    console.log('');
    for (const cluster of analysis.clusters) {
      console.log(`Cluster: ${cluster.name}`);
      console.log(`├── Patterns: ${cluster.patterns.join(', ')}`);
      console.log(`├── Strength: ${(cluster.strength * 100).toFixed(0)}%`);
      console.log(`└── Potential functions:`);
      for (const fn of cluster.potential_functions) {
        console.log(`    - ${fn}()`);
      }
      console.log('');
    }
  }

  // Show emergence candidates
  if (analysis.emergence_candidates.length > 0) {
    console.log('🔥 EMERGENCE CANDIDATES (Ready for DIA 4!):');
    console.log('');
    for (const candidate of analysis.emergence_candidates) {
      console.log(`Function: ${candidate.suggested_function_name}`);
      console.log(`├── Signature: ${candidate.suggested_signature}`);
      console.log(`├── Confidence: ${(candidate.confidence * 100).toFixed(0)}%`);
      console.log(`├── Source pattern: ${candidate.pattern.type} (${candidate.pattern.frequency} occurrences)`);
      console.log(`└── Supporting patterns: ${candidate.supporting_patterns.length > 0 ? candidate.supporting_patterns.join(', ') : 'none'}`);
      console.log('');
    }

    console.log('');
    console.log(`✅ ${analysis.emergence_candidates.length} function(s) ready to emerge on DIA 4!`);
  } else {
    console.log('⏳ No emergence candidates yet. Ingest more knowledge to reach emergence threshold.');
  }

  console.log('');
}

/**
 * Emerge functions from patterns (DIA 4 - CODE EMERGENCE!)
 */
async function cmdEmerge(args: string[]) {
  if (args.length === 0) {
    console.error('Error: organism name required');
    console.log('Usage: fiat emerge <name>');
    process.exit(1);
  }

  // Strip .glass extension if present
  let name = args[0];
  if (name.endsWith('.glass')) {
    name = name.replace('.glass', '');
  }

  const inputPath = path.join(process.cwd(), `${name}.glass`);

  console.log('');
  console.log('🔥🔥🔥 CODE EMERGENCE - THE REVOLUTION! 🔥🔥🔥');
  console.log('');
  console.log(`Triggering emergence in ${name}.glass...`);
  console.log('');

  // Load organism
  const organism = await loadGlassOrganism(inputPath);

  console.log(`Current state:`);
  console.log(`├── Maturity: ${(organism.metadata.maturity * 100).toFixed(0)}%`);
  console.log(`├── Functions: ${organism.code.functions.length}`);
  console.log(`└── Patterns: ${Object.keys(organism.knowledge.patterns).length}`);
  console.log('');

  // Analyze patterns
  const patternEngine = new PatternDetectionEngine(organism);
  const analysis = patternEngine.analyze();

  if (analysis.emergence_candidates.length === 0) {
    console.log('⚠️  No emergence candidates found.');
    console.log('');
    console.log('Reasons:');
    console.log('  - Patterns may not have reached emergence threshold (100+ occurrences)');
    console.log('  - Confidence may be too low (<80%)');
    console.log('');
    console.log('Suggestion: Ingest more knowledge to reach emergence threshold');
    console.log('');
    process.exit(0);
  }

  console.log(`Found ${analysis.emergence_candidates.length} emergence candidate(s):`);
  for (const candidate of analysis.emergence_candidates) {
    console.log(`  🔥 ${candidate.suggested_function_name} (${(candidate.confidence * 100).toFixed(0)}% confidence)`);
  }
  console.log('');

  console.log('🧬 Beginning emergence process...');
  console.log('');

  // Create emergence engine
  const emergenceEngine = new CodeEmergenceEngine(organism);

  // EMERGE!
  const results = await emergenceEngine.emerge(analysis.emergence_candidates);

  console.log('');
  console.log('═══════════════════════════════════════════════════════');
  console.log('');
  console.log('🎉 CODE EMERGENCE COMPLETE!');
  console.log('');
  console.log(`✅ ${results.length} function(s) emerged:`);
  console.log('');

  for (const result of results) {
    const fn = result.function;
    console.log(`📦 ${fn.name}`);
    console.log(`   ├── Signature: ${fn.signature}`);
    console.log(`   ├── Confidence: ${(fn.confidence * 100).toFixed(0)}%`);
    console.log(`   ├── Accuracy: ${(fn.accuracy * 100).toFixed(0)}%`);
    console.log(`   ├── Constitutional: ${fn.constitutional ? '✅' : '❌'}`);
    console.log(`   ├── Lines of code: ${fn.implementation.split('\n').length}`);
    console.log(`   ├── Tests passed: ${result.test_results.passed}/${result.test_results.passed + result.test_results.failed}`);
    console.log(`   └── Emerged at: ${fn.emerged_at}`);
    console.log('');
  }

  // Save updated organism
  const updatedOrganism = emergenceEngine.getOrganism();
  const json = JSON.stringify(updatedOrganism, null, 2);
  fs.writeFileSync(inputPath, json, 'utf-8');

  console.log('Updated organism:');
  console.log(`├── Maturity: ${(updatedOrganism.metadata.maturity * 100).toFixed(0)}% (increased!)`);
  console.log(`├── Functions: ${updatedOrganism.code.functions.length} (emerged!)`);
  console.log(`├── Generation: ${updatedOrganism.evolution.generations}`);
  console.log(`└── Fitness: ${updatedOrganism.evolution.fitness_trajectory[updatedOrganism.evolution.fitness_trajectory.length - 1].toFixed(2)}`);
  console.log('');

  console.log('🎊 CODE EMERGED FROM KNOWLEDGE - NOT PROGRAMMED! 🎊');
  console.log('');
  console.log('This is the revolution: functions that emerge organically');
  console.log('from patterns in knowledge, fully transparent and auditable.');
  console.log('');
}

/**
 * Run organism (DIA 5 - GLASS RUNTIME!)
 */
async function cmdRun(args: string[]) {
  if (args.length === 0) {
    console.error('Error: organism name required');
    console.log('Usage: fiat run <name> [--query "your question"]');
    console.log('');
    console.log('Examples:');
    console.log('  fiat run cancer-research --query "Best treatment for lung cancer?"');
    console.log('  fiat run cancer-research  # Interactive mode');
    process.exit(1);
  }

  // Strip .glass extension if present
  let name = args[0];
  if (name.endsWith('.glass')) {
    name = name.replace('.glass', '');
  }

  const inputPath = path.join(process.cwd(), `${name}.glass`);

  console.log('');
  console.log('🚀🚀🚀 GLASS RUNTIME - EXECUTING ORGANISM! 🚀🚀🚀');
  console.log('');

  // Load organism
  const organism = await loadGlassOrganism(inputPath);

  console.log(`Loaded: ${organism.metadata.name}.glass`);
  console.log(`├── Specialization: ${organism.metadata.specialization}`);
  console.log(`├── Maturity: ${(organism.metadata.maturity * 100).toFixed(0)}%`);
  console.log(`├── Functions: ${organism.code.functions.length}`);
  console.log(`└── Knowledge: ${organism.knowledge.papers.count} papers`);
  console.log('');

  // Check if organism has functions
  if (organism.code.functions.length === 0) {
    console.error('❌ Error: Organism has no emerged functions yet!');
    console.log('');
    console.log('Run these commands first:');
    console.log(`  fiat ingest ${name} --source pubmed:${organism.metadata.specialization}:250`);
    console.log(`  fiat analyze ${name}`);
    console.log(`  fiat emerge ${name}`);
    console.log('');
    process.exit(1);
  }

  // Create runtime
  const runtime = new GlassRuntime(organism, 0.5);

  // Check if --query provided
  const queryIndex = args.indexOf('--query');
  if (queryIndex !== -1 && args[queryIndex + 1]) {
    // Single query mode
    const query = args[queryIndex + 1];
    await executeSingleQuery(runtime, query);
  } else {
    // Interactive mode
    await executeInteractive(runtime, name);
  }

  // Save updated organism (with memory updates)
  const json = JSON.stringify(runtime.getStats().organism, null, 2);
  // Note: Can't save directly, would need to extract organism from runtime
  // For now, just show stats
  console.log('');
  console.log('📊 Runtime Statistics:');
  const stats = runtime.getStats();
  console.log(`├── Total cost: $${stats.runtime.total_cost.toFixed(4)}`);
  console.log(`├── Queries processed: ${stats.runtime.queries_processed}`);
  console.log(`└── Attention tracked: ${stats.runtime.attention_tracked} knowledge sources`);
  console.log('');
}

/**
 * Execute single query
 */
async function executeSingleQuery(runtime: GlassRuntime, query: string) {
  const result = await runtime.query({ query });
  console.log(GlassRuntime.formatResult(result));
}

/**
 * Execute interactive mode
 */
async function executeInteractive(runtime: GlassRuntime, name: string) {
  console.log('🎮 Interactive Mode');
  console.log('Type your questions below. Type "exit" to quit.');
  console.log('');

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: `${name}> `
  });

  rl.prompt();

  rl.on('line', async (line: string) => {
    const query = line.trim();

    if (query.toLowerCase() === 'exit') {
      console.log('');
      console.log('👋 Goodbye!');
      rl.close();
      process.exit(0);
    }

    if (!query) {
      rl.prompt();
      return;
    }

    try {
      const result = await runtime.query({ query });
      console.log('');
      console.log(`📝 ${result.answer}`);
      console.log('');
      console.log(`💡 Confidence: ${(result.confidence * 100).toFixed(0)}% | Functions: ${result.functions_used.join(', ')} | Cost: $${result.cost_usd.toFixed(4)}`);
      console.log('');
    } catch (error: any) {
      console.error('❌ Error:', error.message);
      console.log('');
    }

    rl.prompt();
  }).on('close', () => {
    console.log('');
    console.log('Session ended.');
    process.exit(0);
  });
}

/**
 * Show help
 */
function showHelp() {
  console.log('');
  console.log('Fiat CLI - Create and manage .glass digital organisms');
  console.log('');
  console.log('Usage:');
  console.log('  fiat create <name> [specialization]   Create nascent organism');
  console.log('  fiat status <name>                    Show organism status');
  console.log('  fiat inspect <name>                   Inspect organism (glass box)');
  console.log('  fiat ingest <name> --source <src>     Ingest knowledge (DIA 2)');
  console.log('  fiat analyze <name>                   Analyze patterns (DIA 3)');
  console.log('  fiat emerge <name>                    🔥 Trigger code emergence (DIA 4)');
  console.log('  fiat run <name> [--query "q"]         🚀 Run organism (DIA 5)');
  console.log('');
  console.log('Examples:');
  console.log('  fiat create cancer-research oncology');
  console.log('  fiat ingest cancer-research --source pubmed:cancer:250');
  console.log('  fiat analyze cancer-research');
  console.log('  fiat emerge cancer-research  # 🔥 CODE EMERGENCE!');
  console.log('  fiat run cancer-research --query "Best treatment for lung cancer?"');
  console.log('  fiat run cancer-research  # Interactive mode');
  console.log('');
}

// Run CLI
main().catch(console.error);
