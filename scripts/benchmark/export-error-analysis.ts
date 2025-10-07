#!/usr/bin/env tsx

/**
 * Export error analysis for visualization
 */

import 'dotenv/config';
import { BenchmarkOrchestrator } from '../../src/benchmark/domain/use-cases/benchmark-orchestrator';
import * as fs from 'fs';
import * as path from 'path';

async function main() {
  const testCount = 100;

  console.log('Running benchmark with error analysis...\n');

  const orchestrator = new BenchmarkOrchestrator();
  const summary = await orchestrator.runFullBenchmark(testCount);

  // Export error analysis
  const errorData: any = {
    timestamp: new Date().toISOString(),
    testCount,
    systems: [],
  };

  summary.results.forEach(result => {
    const errorAnalysis = (result as any).errorAnalysis;
    if (errorAnalysis) {
      const builder = (errorAnalysis as any).constructor.name === 'ErrorAnalysis'
        ? null
        : errorAnalysis;

      errorData.systems.push({
        name: result.systemName,
        accuracy: result.metrics.accuracy,
        falsePositiveRate: result.metrics.falsePositives / (result.metrics.falsePositives + result.metrics.trueNegatives),
        falseNegativeRate: result.metrics.falseNegatives / (result.metrics.falseNegatives + result.metrics.truePositives),
        confusionMatrix: Array.from(errorAnalysis.confusionMatrix.entries()).map(([expected, predictedMap]) => ({
          expected,
          predicted: Array.from(predictedMap.entries()).map(([predicted, count]) => ({
            signal: predicted,
            count,
          })),
        })),
        mostCommonError: errorAnalysis.getMostCommonError(),
        worstPattern: errorAnalysis.getWorstPattern(),
      });
    }
  });

  // Save to file
  const outputDir = path.join(process.cwd(), 'benchmark-results');
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const outputPath = path.join(outputDir, `error-analysis-${timestamp}.json`);

  fs.writeFileSync(outputPath, JSON.stringify(errorData, null, 2));

  console.log(`\n💾 Error analysis exported to: ${outputPath}`);
  console.log('\n✅ Export complete!\n');

  process.exit(0);
}

main();
