/**
 * Anthropic Adapter Demo
 *
 * Demonstrates centralized LLM integration with cost tracking and model selection.
 *
 * Shows:
 * 1. Model selection (Opus 4 vs Sonnet 4.5)
 * 2. Automatic cost tracking
 * 3. Token usage monitoring
 * 4. Cost comparison between models
 * 5. Streaming support
 * 6. Cost estimation
 */

import 'dotenv/config';
import { createAdapter, getRecommendedModel } from '../llm/anthropic-adapter';

console.log('═══════════════════════════════════════════════════════════════');
console.log('🤖 Anthropic Adapter Demo - Centralized LLM Integration');
console.log('═══════════════════════════════════════════════════════════════\n');

async function main() {
  // Get API key
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    console.error('❌ ANTHROPIC_API_KEY environment variable not set');
    process.exit(1);
  }

  // Create adapter instance
  console.log('📋 Initializing Anthropic Adapter...\n');
  const adapter = createAdapter(apiKey);

  console.log('───────────────────────────────────────────────────────────────\n');

  // ============================================================================
  // TEST 1: Model Recommendations
  // ============================================================================

  console.log('📋 TEST 1: Model Recommendations');
  console.log('─────────────────────────────────────────────────────────────\n');

  const tasks = ['reasoning', 'creative', 'fast', 'cheap'] as const;
  console.log('🎯 Recommended models by task:\n');

  for (const task of tasks) {
    const model = getRecommendedModel(task);
    console.log(`   ${task.padEnd(12)} → ${model}`);
  }

  console.log('\n───────────────────────────────────────────────────────────────\n');

  // ============================================================================
  // TEST 2: Cost Estimation
  // ============================================================================

  console.log('📋 TEST 2: Cost Estimation');
  console.log('─────────────────────────────────────────────────────────────\n');

  const systemPrompt = 'You are a helpful AI assistant.';
  const query = 'Explain quantum computing in simple terms.';

  console.log('📝 Query: "Explain quantum computing in simple terms."\n');

  const opusEstimate = adapter.estimateCost(systemPrompt, query, 'claude-opus-4');
  const sonnetEstimate = adapter.estimateCost(systemPrompt, query, 'claude-sonnet-4-5');

  console.log('💰 Estimated costs (without making API call):\n');
  console.log(`   Opus 4:      $${opusEstimate.estimated_cost.toFixed(6)}`);
  console.log(`   Sonnet 4.5:  $${sonnetEstimate.estimated_cost.toFixed(6)}`);
  console.log(`   Note: ${opusEstimate.note}\n`);

  console.log('───────────────────────────────────────────────────────────────\n');

  // ============================================================================
  // TEST 3: Actual API Call with Sonnet 4.5 (Fast & Cheap)
  // ============================================================================

  console.log('📋 TEST 3: Sonnet 4.5 Invocation');
  console.log('─────────────────────────────────────────────────────────────\n');

  console.log('🚀 Calling Claude Sonnet 4.5...\n');

  const sonnetResponse = await adapter.invoke(systemPrompt, query, {
    model: 'claude-sonnet-4-5',
    max_tokens: 500,
    temperature: 0.5,
  });

  console.log('📖 Response (first 200 chars):');
  console.log(`   "${sonnetResponse.text.substring(0, 200)}..."\n`);

  console.log('📊 Usage Statistics:');
  console.log(`   Input tokens:  ${sonnetResponse.usage.input_tokens}`);
  console.log(`   Output tokens: ${sonnetResponse.usage.output_tokens}`);
  console.log(`   Actual cost:   $${sonnetResponse.usage.cost_usd.toFixed(6)}`);
  console.log(`   Model:         ${sonnetResponse.model}`);
  console.log(`   Stop reason:   ${sonnetResponse.stop_reason}\n`);

  console.log('───────────────────────────────────────────────────────────────\n');

  // ============================================================================
  // TEST 4: Cost Comparison Between Models
  // ============================================================================

  console.log('📋 TEST 4: Cost Comparison');
  console.log('─────────────────────────────────────────────────────────────\n');

  console.log('💵 Actual cost comparison (same query):\n');

  const costs = adapter.compareCosts(
    sonnetResponse.usage.input_tokens,
    sonnetResponse.usage.output_tokens
  );

  console.log(`   Opus 4:      $${costs['claude-opus-4'].toFixed(6)}`);
  console.log(`   Sonnet 4.5:  $${costs['claude-sonnet-4-5'].toFixed(6)}\n`);

  const savings = costs['claude-opus-4'] - costs['claude-sonnet-4-5'];
  const savingsPercent = (savings / costs['claude-opus-4']) * 100;

  console.log(`   💰 Savings by using Sonnet: $${savings.toFixed(6)} (${savingsPercent.toFixed(1)}% cheaper)\n`);

  console.log('───────────────────────────────────────────────────────────────\n');

  // ============================================================================
  // TEST 5: Streaming Response
  // ============================================================================

  console.log('📋 TEST 5: Streaming Response');
  console.log('─────────────────────────────────────────────────────────────\n');

  console.log('🌊 Streaming query: "Count from 1 to 5 slowly."\n');
  console.log('📡 Stream output: ');

  process.stdout.write('   "');

  let streamedText = '';
  const streamQuery = 'Count from 1 to 5, explaining each number in one sentence.';

  const streamUsage = await (async () => {
    let usage;
    for await (const chunk of adapter.invokeStream(systemPrompt, streamQuery, {
      model: 'claude-sonnet-4-5',
      max_tokens: 300,
      temperature: 0.7,
    })) {
      if (typeof chunk === 'string') {
        process.stdout.write(chunk);
        streamedText += chunk;
      } else {
        usage = chunk;
      }
    }
    return usage;
  })();

  process.stdout.write('"\n\n');

  console.log('📊 Stream Usage:');
  console.log(`   Input tokens:  ${streamUsage?.input_tokens || 0}`);
  console.log(`   Output tokens: ${streamUsage?.output_tokens || 0}`);
  console.log(`   Cost:          $${streamUsage?.cost_usd.toFixed(6) || '0.000000'}\n`);

  console.log('───────────────────────────────────────────────────────────────\n');

  // ============================================================================
  // TEST 6: Cumulative Cost Tracking
  // ============================================================================

  console.log('📋 TEST 6: Cumulative Cost Tracking');
  console.log('─────────────────────────────────────────────────────────────\n');

  console.log('📈 Total adapter statistics:\n');
  console.log(`   Total requests: ${adapter.getTotalRequests()}`);
  console.log(`   Total cost:     $${adapter.getTotalCost().toFixed(6)}\n`);

  console.log('───────────────────────────────────────────────────────────────\n');

  // ============================================================================
  // TEST 7: Multiple Requests Cost Accumulation
  // ============================================================================

  console.log('📋 TEST 7: Cost Accumulation');
  console.log('─────────────────────────────────────────────────────────────\n');

  console.log('🔁 Making 3 quick requests to demonstrate accumulation...\n');

  const queries = [
    'What is 2+2?',
    'Name one planet.',
    'What color is the sky?',
  ];

  for (let i = 0; i < queries.length; i++) {
    console.log(`   Request ${i + 1}: "${queries[i]}"`);

    const response = await adapter.invoke(systemPrompt, queries[i], {
      model: 'claude-sonnet-4-5',
      max_tokens: 50,
      temperature: 0.3,
    });

    console.log(`   → Answer: "${response.text.substring(0, 50)}..."`);
    console.log(`   → Cost: $${response.usage.cost_usd.toFixed(6)}`);
    console.log(`   → Cumulative: $${adapter.getTotalCost().toFixed(6)}\n`);
  }

  console.log('───────────────────────────────────────────────────────────────\n');

  // Final summary
  console.log('📊 FINAL SUMMARY');
  console.log('─────────────────────────────────────────────────────────────\n');

  console.log(`   Total API Requests:  ${adapter.getTotalRequests()}`);
  console.log(`   Total Cost:          $${adapter.getTotalCost().toFixed(6)}`);
  console.log(`   Average Cost/Req:    $${(adapter.getTotalCost() / adapter.getTotalRequests()).toFixed(6)}\n`);

  console.log('═══════════════════════════════════════════════════════════════');
  console.log('🎓 KEY INSIGHTS');
  console.log('═══════════════════════════════════════════════════════════════\n');

  console.log(`The Anthropic Adapter provides:

✅ CENTRALIZED LLM INTEGRATION
   - Single point for all Claude API calls
   - Consistent interface across the system
   - Easy to swap models or add new ones

✅ AUTOMATIC COST TRACKING
   - Every request tracked with actual costs
   - No more guesswork or manual calculation
   - Real-time cumulative cost monitoring

✅ MODEL SELECTION
   - Opus 4: Best reasoning/creative (5x more expensive)
   - Sonnet 4.5: Fast, cost-effective (recommended default)
   - Easy comparison and switching

✅ STREAMING SUPPORT
   - Real-time response generation
   - Same cost tracking as non-streaming
   - Better UX for long responses

✅ COST OPTIMIZATION
   - Estimate before making calls
   - Compare costs between models
   - Make informed decisions on model selection

RESULT:
  - Transparent cost tracking
  - Intelligent model selection
  - Production-ready LLM integration
`);

  console.log('═══════════════════════════════════════════════════════════════\n');
}

// Run the demo
main().catch((error) => {
  console.error('❌ Error:', error);
  process.exit(1);
});
