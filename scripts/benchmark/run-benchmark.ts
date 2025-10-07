#!/usr/bin/env tsx

/**
 * CLI Script: Run Benchmark
 * Executes the Deterministic Intelligence Benchmark
 */

import 'dotenv/config'; // Load environment variables from .env
import { BenchmarkOrchestrator } from '../../src/benchmark/domain/use-cases/benchmark-orchestrator';
import * as fs from 'fs';
import * as path from 'path';

async function main() {
  const args = process.argv.slice(2);
  const testCount = args[0] ? parseInt(args[0], 10) : 1000;

  if (isNaN(testCount) || testCount <= 0) {
    console.error('Error: Test count must be a positive number');
    console.error('Usage: npm run benchmark [testCount]');
    process.exit(1);
  }

  console.log('Starting Deterministic Intelligence Benchmark...\n');

  const orchestrator = new BenchmarkOrchestrator();

  try {
    const summary = await orchestrator.runFullBenchmark(testCount);

    // Display results
    orchestrator.displayResults(summary);

    // Export to JSON
    const outputDir = path.join(process.cwd(), 'benchmark-results');
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const outputPath = path.join(outputDir, `benchmark-${timestamp}.json`);

    const jsonOutput = orchestrator.exportToJSON(summary);
    fs.writeFileSync(outputPath, jsonOutput);

    console.log(`\n💾 Results saved to: ${outputPath}`);
    console.log('\n✅ Benchmark complete!\n');

    process.exit(0);
  } catch (error) {
    console.error('\n❌ Benchmark failed:', error);
    process.exit(1);
  }
}

main();
