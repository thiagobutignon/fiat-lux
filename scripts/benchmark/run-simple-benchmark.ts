#!/usr/bin/env tsx

/**
 * Simplified AGI Benchmark - Quick Test
 */

import 'dotenv/config';
import { MetaAgent } from '../../src/agi-recursive/core/meta-agent';
import { FinancialAgent } from '../../src/agi-recursive/agents/financial-agent';
import { createAdapter } from '../../src/agi-recursive/llm/anthropic-adapter';

async function runSimpleBenchmark(): Promise<void> {
  console.log('═'.repeat(70));
  console.log('  SIMPLE AGI BENCHMARK - QUICK TEST');
  console.log('═'.repeat(70));
  console.log();

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    console.error('❌ ANTHROPIC_API_KEY not found');
    process.exit(1);
  }

  // Initialize AGI with only 1 agent and low limits
  const metaAgent = new MetaAgent(
    apiKey,
    2,   // max depth: 2 (low)
    3,   // max invocations: 3 (low)
    0.50 // max cost: $0.50
  );

  metaAgent.registerAgent('financial', new FinancialAgent(apiKey));

  console.log('✅ AGI initialized with 1 agent (financial)\n');

  const query = 'What is compound interest?';

  // Test 1: AGI
  console.log('🧠 Testing AGI Multi-Agent...');
  const agiStart = Date.now();
  const agiStartCost = metaAgent.getTotalCost();

  try {
    const agiResult = await metaAgent.process(query);
    const agiTime = Date.now() - agiStart;
    const agiCost = metaAgent.getTotalCost() - agiStartCost;

    console.log(`✅ AGI Response: ${agiResult.final_answer.substring(0, 150)}...`);
    console.log(`   Cost: $${agiCost.toFixed(4)}`);
    console.log(`   Time: ${(agiTime / 1000).toFixed(2)}s`);
    console.log(`   Emergent insights: ${agiResult.emergent_insights.length}`);
  } catch (e: any) {
    console.error(`❌ AGI failed: ${e.message}`);
  }

  // Test 2: Monolithic LLM
  console.log('\n🤖 Testing Monolithic LLM...');
  const llmAdapter = createAdapter(apiKey);
  const llmStart = Date.now();
  const llmStartCost = llmAdapter.getTotalCost();

  try {
    const llmResult = await llmAdapter.invoke(
      'You are a helpful assistant.',
      query,
      {
        model: 'claude-sonnet-4-5',
        max_tokens: 1000,
        temperature: 0.7
      }
    );

    const llmTime = Date.now() - llmStart;
    const llmCost = llmAdapter.getTotalCost() - llmStartCost;

    console.log(`✅ LLM Response: ${llmResult.text.substring(0, 150)}...`);
    console.log(`   Cost: $${llmCost.toFixed(4)}`);
    console.log(`   Time: ${(llmTime / 1000).toFixed(2)}s`);
  } catch (e: any) {
    console.error(`❌ LLM failed: ${e.message}`);
  }

  console.log('\n✅ Simple benchmark complete!\n');
}

runSimpleBenchmark().catch(console.error);
