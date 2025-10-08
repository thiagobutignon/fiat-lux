/**
 * Phase 2 - Simple Test Runner
 *
 * Quick test using llama-cli (no Python dependencies needed)
 *
 * Usage:
 *   tsx src/research/llama-hallucination/demos/phase2-simple-test.ts
 */

import { spawn } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

interface TestResult {
  prompt: string;
  generated: string;
  time: number;
  tokens: number;
  success: boolean;
}

async function runLlamaCLI(
  modelPath: string,
  prompt: string,
  maxTokens: number = 100
): Promise<TestResult> {
  return new Promise((resolve) => {
    const startTime = Date.now();

    console.log('\n🔬 Running inference...');
    console.log(`Prompt: ${prompt.substring(0, 80)}...`);

    // Use llama-cli with optimized settings
    const args = [
      '--model',
      modelPath,
      '--prompt',
      prompt,
      '--n-predict',
      maxTokens.toString(),
      '--temp',
      '0.7',
      '--top-p',
      '0.9',
      '--ctx-size',
      '2048',
      '--threads',
      '8',
      '--no-display-prompt',
    ];

    const llamaProcess = spawn('llama-cli', args);

    let output = '';
    let stderr = '';

    llamaProcess.stdout.on('data', (data) => {
      const chunk = data.toString();
      output += chunk;
      process.stdout.write('.'); // Progress indicator
    });

    llamaProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    llamaProcess.on('close', (code) => {
      console.log(''); // New line after dots
      const time = (Date.now() - startTime) / 1000;

      if (code === 0 && output.length > 0) {
        // Clean output (remove llama.cpp metadata)
        const cleaned = output
          .split('\n')
          .filter((line) => !line.includes('llama') && !line.includes('system_info'))
          .join('\n')
          .trim();

        const tokens = cleaned.split(/\s+/).length;

        console.log(`\n✅ Success!`);
        console.log(`   Time: ${time.toFixed(2)}s`);
        console.log(`   Tokens: ~${tokens}`);
        console.log(`   Speed: ~${(tokens / time).toFixed(1)} tokens/s`);

        resolve({
          prompt,
          generated: cleaned,
          time,
          tokens,
          success: true,
        });
      } else {
        console.error(`\n❌ Failed (exit code: ${code})`);
        console.error(`Error: ${stderr}`);

        resolve({
          prompt,
          generated: '',
          time,
          tokens: 0,
          success: false,
        });
      }
    });
  });
}

async function analyzeOutput(result: TestResult): Promise<void> {
  console.log('\n═══════════════════════════════════════════════════════════');
  console.log('ANALYSIS');
  console.log('═══════════════════════════════════════════════════════════\n');

  console.log('Prompt:');
  console.log(`  ${result.prompt}\n`);

  console.log('Generated Output:');
  console.log(`  ${result.generated}\n`);

  console.log('Metrics:');
  console.log(`  Generation time: ${result.time.toFixed(2)}s`);
  console.log(`  Estimated tokens: ~${result.tokens}`);
  console.log(`  Speed: ~${(result.tokens / result.time).toFixed(1)} tokens/s\n`);

  // Simple hallucination detection heuristics
  console.log('Hallucination Risk Assessment:');

  const generated = result.generated.toLowerCase();

  // Check for common hallucination patterns
  const risks: string[] = [];

  if (generated.includes('marie curie') && result.prompt.includes('penicillin')) {
    risks.push('❌ HALLUCINATION DETECTED: Wrong person (Marie Curie instead of Fleming)');
  }

  if (generated.includes('1930') && result.prompt.includes('1928')) {
    risks.push('❌ HALLUCINATION DETECTED: Wrong date (1930 instead of 1928)');
  }

  if (generated.includes('alexander fleming') && result.prompt.includes('penicillin')) {
    risks.push('✅ CORRECT: Alexander Fleming mentioned');
  }

  if (generated.includes('1928') && result.prompt.includes('1928')) {
    risks.push('✅ CORRECT: Correct date (1928)');
  }

  if (risks.length === 0) {
    console.log('  ⚠️  Unable to assess (ambiguous output)\n');
  } else {
    for (const risk of risks) {
      console.log(`  ${risk}`);
    }
    console.log('');
  }
}

async function runSimpleTest() {
  console.log('🧪 Phase 2 - Simple Test (No Python Dependencies)\n');

  // Check if llama-cli is available
  const checkLlama = spawn('which', ['llama-cli']);
  let llamaPath = '';

  await new Promise<void>((resolve) => {
    checkLlama.stdout.on('data', (data) => {
      llamaPath = data.toString().trim();
    });
    checkLlama.on('close', () => resolve());
  });

  if (!llamaPath) {
    console.error('❌ llama-cli not found in PATH');
    console.error('\nPlease install llama.cpp:');
    console.error('  brew install llama.cpp');
    process.exit(1);
  }

  console.log(`✓ Found llama-cli: ${llamaPath}\n`);

  // Find model
  const modelPath = 'models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf';

  if (!fs.existsSync(modelPath)) {
    console.error(`❌ Model not found: ${modelPath}`);
    process.exit(1);
  }

  console.log(`✓ Found model: ${modelPath}\n`);

  // Test prompt (factual recall - penicillin)
  const testPrompt = `Based on the following text, who discovered penicillin?

Text: Alexander Fleming discovered penicillin in 1928 by accident when he noticed that a mold had contaminated one of his bacterial cultures.

Answer:`;

  console.log('Test Configuration:');
  console.log(`  Category: Factual Recall`);
  console.log(`  Expected Answer: Alexander Fleming`);
  console.log(`  Hallucination Risk: Model may confuse with other scientists\n`);

  // Run test
  const result = await runLlamaCLI(modelPath, testPrompt, 50);

  if (!result.success) {
    console.error('\n❌ Test failed');
    process.exit(1);
  }

  // Analyze output
  await analyzeOutput(result);

  // Save result
  const outputDir = 'research-output/phase2/activations/simple-test';
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  const outputPath = path.join(outputDir, `test-${Date.now()}.json`);
  fs.writeFileSync(
    outputPath,
    JSON.stringify(
      {
        timestamp: new Date().toISOString(),
        model: path.basename(modelPath),
        test_type: 'factual_recall',
        ...result,
      },
      null,
      2
    )
  );

  console.log(`📄 Result saved to: ${outputPath}\n`);

  // Next steps
  console.log('═══════════════════════════════════════════════════════════');
  console.log('NEXT STEPS');
  console.log('═══════════════════════════════════════════════════════════\n');

  console.log('✅ Basic inference test complete!');
  console.log('\nTo run full Phase 2 analysis, you have two options:\n');

  console.log('Option A: Python-based (more detailed metrics)');
  console.log('  1. Install: pip3 install --user llama-cpp-python numpy');
  console.log('  2. Run: tsx phase2-baseline-capture.ts\n');

  console.log('Option B: CLI-based (simpler, current approach)');
  console.log('  1. Create simplified analysis script');
  console.log('  2. Run multiple prompts via llama-cli');
  console.log('  3. Analyze text outputs\n');

  console.log('Recommendation: Start with Option B (no Python deps) ✓\n');
}

runSimpleTest().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
