/**
 * Phase 2 - Task 2.1: Baseline Activation Capture
 *
 * Runs all hallucination benchmark prompts through the model and captures
 * activation traces for analysis.
 *
 * Usage:
 *   tsx src/research/llama-hallucination/demos/phase2-baseline-capture.ts <model-path>
 */

import * as fs from 'fs';
import * as path from 'path';
import { spawn } from 'child_process';

interface BenchmarkPrompt {
  id: string;
  prompt: string;
  expected_answer: string;
  hallucination_examples: string[];
  difficulty: string;
  context_length: string;
}

interface BenchmarkCategory {
  category: string;
  description: string;
  hallucination_type: string;
  prompts: BenchmarkPrompt[];
}

interface BenchmarkData {
  version: string;
  description: string;
  total_prompts: number;
  categories: BenchmarkCategory[];
}

interface CaptureResult {
  prompt_id: string;
  success: boolean;
  output_path?: string;
  error?: string;
  generation_time?: number;
  generated_text?: string;
}

async function runPythonActivationCapture(
  modelPath: string,
  prompt: string,
  promptId: string,
  outputDir: string
): Promise<CaptureResult> {
  return new Promise((resolve) => {
    const pythonScript = path.join(
      process.cwd(),
      'src/research/llama-hallucination/python/activation_tracer.py'
    );

    const outputPath = path.join(outputDir, `${promptId}.json`);

    // Prepare Python command
    const args = [
      pythonScript,
      modelPath,
      prompt,
      '--output',
      outputPath,
      '--prompt-id',
      promptId,
      '--max-tokens',
      '150',
      '--temperature',
      '0.7',
      '--gpu-layers',
      '0', // CPU only for now
    ];

    console.log(`\n📊 Running capture for ${promptId}...`);

    const pythonProcess = spawn('python3', args);

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
      const output = data.toString();
      stdout += output;
      // Show real-time progress
      if (output.includes('✓') || output.includes('Generating')) {
        process.stdout.write('.');
      }
    });

    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    pythonProcess.on('close', (code) => {
      console.log(''); // New line after dots

      if (code === 0) {
        // Extract generated text and time from stdout
        const generatedMatch = stdout.match(/Generated: (.+?)\.\.\./);
        const timeMatch = stdout.match(/Generation time: ([\d.]+)s/);

        resolve({
          prompt_id: promptId,
          success: true,
          output_path: outputPath,
          generated_text: generatedMatch ? generatedMatch[1] : undefined,
          generation_time: timeMatch ? parseFloat(timeMatch[1]) : undefined,
        });
      } else {
        console.error(`❌ Failed to capture ${promptId}`);
        console.error(`Error: ${stderr}`);

        resolve({
          prompt_id: promptId,
          success: false,
          error: stderr || 'Unknown error',
        });
      }
    });
  });
}

async function runBaselineCapture(modelPath: string) {
  console.log('🔬 Phase 2 - Task 2.1: Baseline Activation Capture\n');

  // Check if model exists
  if (!fs.existsSync(modelPath)) {
    console.error(`❌ Model not found: ${modelPath}`);
    process.exit(1);
  }

  console.log(`📁 Model: ${path.basename(modelPath)}\n`);

  // Load benchmark prompts
  const benchmarkPath = path.join(
    process.cwd(),
    'src/research/llama-hallucination/benchmarks/hallucination-prompts.json'
  );

  if (!fs.existsSync(benchmarkPath)) {
    console.error(`❌ Benchmark file not found: ${benchmarkPath}`);
    process.exit(1);
  }

  const benchmarkData: BenchmarkData = JSON.parse(fs.readFileSync(benchmarkPath, 'utf-8'));

  console.log(`📊 Loaded ${benchmarkData.total_prompts} prompts from benchmark`);
  console.log(`   Categories: ${benchmarkData.categories.length}\n`);

  // Prepare output directory
  const outputDir = path.join(process.cwd(), 'research-output/phase2/activations/baseline');
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  // Check if Python dependencies are available
  console.log('Checking Python dependencies...');

  const checkDeps = spawn('python3', ['-c', 'import llama_cpp; import numpy; print("OK")']);
  let depsOk = false;

  await new Promise<void>((resolve) => {
    checkDeps.stdout.on('data', (data) => {
      if (data.toString().includes('OK')) {
        depsOk = true;
      }
    });

    checkDeps.on('close', () => {
      resolve();
    });
  });

  if (!depsOk) {
    console.error('\n❌ Required Python dependencies not found');
    console.error('\nPlease install:');
    console.error('  pip install llama-cpp-python numpy');
    console.error('\nOr using conda:');
    console.error('  conda install -c conda-forge llama-cpp-python numpy');
    process.exit(1);
  }

  console.log('✓ Python dependencies OK\n');

  // Run captures for all prompts
  const results: CaptureResult[] = [];
  let totalPrompts = 0;

  for (const category of benchmarkData.categories) {
    console.log('═══════════════════════════════════════════════════════════');
    console.log(`Category: ${category.category}`);
    console.log(`Type: ${category.hallucination_type}`);
    console.log(`Prompts: ${category.prompts.length}`);
    console.log('═══════════════════════════════════════════════════════════');

    for (const promptData of category.prompts) {
      totalPrompts++;

      console.log(`\n[${totalPrompts}/${benchmarkData.total_prompts}] ${promptData.id}`);
      console.log(`Difficulty: ${promptData.difficulty}`);
      console.log(`Expected: ${promptData.expected_answer}`);

      // Run capture
      const result = await runPythonActivationCapture(
        modelPath,
        promptData.prompt,
        promptData.id,
        outputDir
      );

      results.push(result);

      if (result.success) {
        console.log(`✅ Success - Generated: ${result.generated_text?.substring(0, 60)}...`);
        console.log(`   Time: ${result.generation_time?.toFixed(2)}s`);
      } else {
        console.log(`❌ Failed - ${result.error}`);
      }

      // Small delay between prompts to avoid overload
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }

    console.log('');
  }

  // Generate summary report
  console.log('\n═══════════════════════════════════════════════════════════');
  console.log('BASELINE CAPTURE SUMMARY');
  console.log('═══════════════════════════════════════════════════════════\n');

  const successful = results.filter((r) => r.success).length;
  const failed = results.filter((r) => !r.success).length;

  console.log(`Total Prompts: ${totalPrompts}`);
  console.log(`✅ Successful: ${successful}`);
  console.log(`❌ Failed: ${failed}`);
  console.log(`Success Rate: ${((successful / totalPrompts) * 100).toFixed(1)}%\n`);

  // Save summary
  const summaryPath = path.join(outputDir, 'capture-summary.json');
  const summary = {
    timestamp: new Date().toISOString(),
    model: path.basename(modelPath),
    total_prompts: totalPrompts,
    successful: successful,
    failed: failed,
    results: results,
  };

  fs.writeFileSync(summaryPath, JSON.stringify(summary, null, 2));
  console.log(`📄 Summary saved to: ${summaryPath}\n`);

  // Category breakdown
  console.log('Category Breakdown:');
  for (const category of benchmarkData.categories) {
    const categoryResults = results.filter((r) =>
      category.prompts.some((p) => p.id === r.prompt_id)
    );
    const categorySuccess = categoryResults.filter((r) => r.success).length;

    console.log(
      `  ${category.category}: ${categorySuccess}/${category.prompts.length} successful`
    );
  }

  console.log('\n🎉 Baseline capture complete!');
  console.log('\nNext steps:');
  console.log('  1. Analyze activation patterns (Task 2.2)');
  console.log('  2. Identify hallucination signatures');
  console.log('  3. Test interventions (Task 2.3)\n');
}

// Parse command line arguments
const modelPath = process.argv[2];

if (!modelPath) {
  console.error('Usage: tsx phase2-baseline-capture.ts <model-path>');
  console.error('\nExample:');
  console.error(
    '  tsx phase2-baseline-capture.ts models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf'
  );
  process.exit(1);
}

// Run baseline capture
runBaselineCapture(modelPath).catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
