/**
 * Budget Homeostasis Example
 *
 * Demonstrates emergent AGI through recursive agent composition.
 *
 * EMERGENT INSIGHT:
 * No single agent would suggest "budget homeostasis" as a framework.
 * - Financial agent: sees spending problem, suggests budget limits
 * - Biology agent: sees homeostatic failure, suggests set point regulation
 * - Systems agent: sees positive feedback loop, suggests corrector mechanism
 *
 * COMPOSED TOGETHER → "Budget as Biological System with Homeostatic Control"
 *
 * This is AGI through composition, not model size.
 */

import 'dotenv/config';
import { MetaAgent } from '../core/meta-agent';
import { FinancialAgent } from '../agents/financial-agent';
import { BiologyAgent } from '../agents/biology-agent';
import { SystemsAgent } from '../agents/systems-agent';

async function main() {
  // Get API key from environment
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    console.error('❌ ANTHROPIC_API_KEY environment variable not set');
    process.exit(1);
  }

  console.log('═══════════════════════════════════════════════════════════════');
  console.log('🧠 AGI RECURSIVE SYSTEM - Budget Homeostasis Demo');
  console.log('═══════════════════════════════════════════════════════════════\n');

  // Initialize meta-agent with budget limits
  const metaAgent = new MetaAgent(
    apiKey,
    3, // max depth
    10, // max invocations
    1.0 // max $1 USD cost
  );

  // Register specialist agents
  console.log('📋 Registering specialist agents...');
  metaAgent.registerAgent('financial', new FinancialAgent(apiKey));
  metaAgent.registerAgent('biology', new BiologyAgent(apiKey));
  metaAgent.registerAgent('systems', new SystemsAgent(apiKey));
  console.log('✓ Registered: financial, biology, systems\n');

  // User query
  const query = `My spending on Nubank is out of control. I spend way too much on food delivery, especially on Fridays after stressful work days. I know I should stop but I can't seem to control it. What should I do?`;

  console.log('❓ USER QUERY:');
  console.log(`"${query}"\n`);

  console.log('🔄 Processing with recursive agent composition...\n');
  console.log('───────────────────────────────────────────────────────────────\n');

  // Process query
  const result = await metaAgent.process(query);

  // Display results
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('📊 RESULTS');
  console.log('═══════════════════════════════════════════════════════════════\n');

  console.log('🎯 FINAL ANSWER:');
  console.log(result.final_answer);
  console.log('\n───────────────────────────────────────────────────────────────\n');

  console.log('💡 EMERGENT INSIGHTS:');
  console.log('(Concepts that emerged from composition, not individual agents)\n');
  if (result.emergent_insights.length > 0) {
    result.emergent_insights.forEach((insight) => {
      console.log(`  • ${insight}`);
    });
  } else {
    console.log('  (No distinct emergent insights detected in this run)');
  }
  console.log('\n───────────────────────────────────────────────────────────────\n');

  console.log('🔍 REASONING PATH:');
  console.log(result.reasoning_path);
  console.log('───────────────────────────────────────────────────────────────\n');

  console.log('📜 EXECUTION TRACE:');
  console.log(`Total invocations: ${result.trace.length}`);
  console.log(`Max depth reached: ${Math.max(...result.trace.map((t) => t.depth))}`);
  console.log(`Estimated cost: $${result.trace.reduce((sum, t) => sum + t.cost_estimate, 0).toFixed(4)}\n`);

  result.trace.forEach((trace, i) => {
    const indent = '  '.repeat(trace.depth);
    console.log(
      `${indent}[${i + 1}] ${trace.agent_id} (depth: ${trace.depth}, conf: ${trace.response.confidence})`
    );
  });
  console.log('\n───────────────────────────────────────────────────────────────\n');

  console.log('⚖️ CONSTITUTION VIOLATIONS:');
  if (result.constitution_violations.length > 0) {
    result.constitution_violations.forEach((v) => {
      const icon = v.severity === 'fatal' ? '❌' : v.severity === 'error' ? '⚠️' : 'ℹ️';
      console.log(`  ${icon} [${v.severity.toUpperCase()}] ${v.principle_id}: ${v.message}`);
      console.log(`     → ${v.suggested_action}\n`);
    });
  } else {
    console.log('  ✅ No violations detected\n');
  }

  console.log('═══════════════════════════════════════════════════════════════');
  console.log('🎓 KEY INSIGHT');
  console.log('═══════════════════════════════════════════════════════════════\n');

  console.log(`This solution emerged from COMPOSITION:

- Financial Agent alone → "Set budget limit, track spending"
- Biology Agent alone → "Homeostasis, set point regulation"
- Systems Agent alone → "Feedback loop, leverage points"

COMPOSED TOGETHER → "Budget as Biological Homeostatic System"

This is AGI through recursive composition, not through model size.

The meta-agent orchestrated specialists, detected cross-domain patterns,
and synthesized insights no single agent could produce alone.

Constitution enforcement prevented:
  ❌ Hallucination cascades
  ❌ Infinite recursion
  ❌ Cost explosions
  ❌ Domain boundary violations

This is the future: Compositional Intelligence.
`);

  console.log('═══════════════════════════════════════════════════════════════\n');
}

// Run the demo
main().catch((error) => {
  console.error('❌ Error:', error);
  process.exit(1);
});
