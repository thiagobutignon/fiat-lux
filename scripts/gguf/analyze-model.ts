#!/usr/bin/env tsx

/**
 * CLI Tool: GGUF Model Analyzer
 * Analyzes GGUF model files and displays comprehensive architecture information
 */

import { analyzeGGUF, formatAnalysis } from '../../src/gguf-parser/presentation';
import { resolve } from 'path';

async function main() {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    console.error('Usage: tsx scripts/gguf/analyze-model.ts <path-to-gguf-file>');
    console.error('\nExample:');
    console.error('  tsx scripts/gguf/analyze-model.ts landing/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf');
    process.exit(1);
  }

  const filePath = resolve(args[0]);

  console.log('🔍 GGUF Model Analyzer\n');
  console.log(`📄 Loading: ${filePath}\n`);

  try {
    const startTime = Date.now();

    // Analyze model
    const { model, analysis } = await analyzeGGUF(filePath);

    const elapsedTime = Date.now() - startTime;

    // Display formatted analysis
    console.log(formatAnalysis(analysis));

    // Performance info
    console.log(`⚡ Analysis completed in ${elapsedTime}ms\n`);

    // Additional metadata export option
    if (args.includes('--export-json')) {
      const outputPath = filePath.replace('.gguf', '_analysis.json');
      const fs = await import('fs/promises');
      await fs.writeFile(
        outputPath,
        JSON.stringify(
          {
            model: {
              header: {
                ...model.header,
                tensorCount: model.header.tensorCount.toString(),
                metadataKVCount: model.header.metadataKVCount.toString(),
              },
              architecture: model.architecture,
              totalParameters: model.totalParameters.toString(),
              quantizationType: model.quantizationType,
              tensorCount: model.tensors.length,
            },
            analysis,
          },
          null,
          2
        )
      );
      console.log(`💾 Exported analysis to: ${outputPath}\n`);
    }
  } catch (error: any) {
    console.error(`\n❌ Error analyzing model: ${error.message}\n`);
    if (error.stack) {
      console.error(error.stack);
    }
    process.exit(1);
  }
}

main();
