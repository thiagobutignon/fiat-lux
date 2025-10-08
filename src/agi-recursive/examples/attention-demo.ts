/**
 * Attention Tracking Demo
 *
 * Demonstrates the interpretability layer for AGI decisions.
 *
 * This demo shows:
 * 1. How to track which concepts influence decisions
 * 2. How to visualize attention patterns
 * 3. How to export data for auditing
 * 4. How to debug reasoning chains
 *
 * USE CASES:
 * - Developer: "Why did the system give this answer?"
 * - Auditor: "Which knowledge influenced this financial advice?"
 * - Researcher: "What patterns emerge in multi-domain queries?"
 * - Debugger: "Where did the reasoning go wrong?"
 */

import dotenv from 'dotenv';
import path from 'path';
import { MetaAgent } from '../core/meta-agent';
import { FinancialAgent } from '../agents/financial-agent';
import { BiologyAgent } from '../agents/biology-agent';
import { SystemsAgent } from '../agents/systems-agent';
import {
  visualizeAttention,
  visualizeStats,
  saveAttentionReport,
  compareAttentions,
} from '../core/attention-visualizer';

dotenv.config();

interface DemoQuery {
  id: string;
  query: string;
  expected_domains: string[];
  description: string;
}

const DEMO_QUERIES: DemoQuery[] = [
  {
    id: 'simple_financial',
    query: 'What is compound interest and how does it work?',
    expected_domains: ['financial'],
    description: 'Simple single-domain query',
  },
  {
    id: 'cross_domain',
    query:
      'How can I apply biological evolution principles to optimize my investment portfolio diversification strategy?',
    expected_domains: ['financial', 'biology', 'systems'],
    description: 'Complex cross-domain query requiring synthesis',
  },
  {
    id: 'systems_thinking',
    query:
      'What are feedback loops and how do they appear in both biological systems and financial markets?',
    expected_domains: ['biology', 'financial', 'systems'],
    description: 'Pattern recognition across domains',
  },
];

async function runAttentionDemo() {
  console.log('═'.repeat(80));
  console.log('🧠 ATTENTION TRACKING DEMO');
  console.log('═'.repeat(80));
  console.log();
  console.log('This demo showcases the interpretability layer of the AGI system.');
  console.log('We will track EXACTLY which concepts influence each decision.');
  console.log();
  console.log('═'.repeat(80));
  console.log();

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    console.error('❌ ANTHROPIC_API_KEY not found in environment');
    process.exit(1);
  }

  // Create meta-agent
  const metaAgent = new MetaAgent(
    apiKey,
    5, // max depth
    10, // max invocations
    0.5 // max cost: $0.50
  );

  // Register specialized agents
  metaAgent.registerAgent('financial', new FinancialAgent(apiKey));
  metaAgent.registerAgent('biology', new BiologyAgent(apiKey));
  metaAgent.registerAgent('systems', new SystemsAgent(apiKey));

  // Initialize
  const slicesDir = path.join(__dirname, '..', 'slices');
  await metaAgent.initialize(slicesDir);

  console.log('✅ MetaAgent initialized with 3 specialized agents');
  console.log();

  // Run demo queries
  const results: Array<{
    query: DemoQuery;
    queryId: string;
  }> = [];

  for (let i = 0; i < DEMO_QUERIES.length; i++) {
    const demoQuery = DEMO_QUERIES[i];
    console.log('\n\n');
    console.log('═'.repeat(80));
    console.log(`📋 QUERY ${i + 1}/${DEMO_QUERIES.length}: ${demoQuery.id}`);
    console.log('═'.repeat(80));
    console.log(`Description: ${demoQuery.description}`);
    console.log(`Expected domains: ${demoQuery.expected_domains.join(', ')}`);
    console.log();
    console.log(`Query: "${demoQuery.query}"`);
    console.log();
    console.log('Processing...');

    const startTime = Date.now();
    const startCost = metaAgent.getTotalCost();

    try {
      const result = await metaAgent.process(demoQuery.query);
      const endTime = Date.now();
      const queryCost = metaAgent.getTotalCost() - startCost;

      console.log();
      console.log('✅ Query processed successfully!');
      console.log();
      console.log('📊 Execution Stats:');
      console.log(`   Time: ${((endTime - startTime) / 1000).toFixed(2)}s`);
      console.log(`   Cost: $${queryCost.toFixed(4)}`);
      console.log(`   Traces: ${result.trace.length}`);
      console.log();

      // Show answer (first 300 chars)
      console.log('💬 Answer:');
      const answerPreview =
        result.final_answer.length > 300
          ? result.final_answer.substring(0, 300) + '...'
          : result.final_answer;
      console.log(`   ${answerPreview}`);
      console.log();

      // Visualize attention
      if (result.attention) {
        console.log('🎯 ATTENTION ANALYSIS:');
        console.log();
        console.log(visualizeAttention(result.attention));

        results.push({
          query: demoQuery,
          queryId: result.attention.query_id,
        });
      } else {
        console.log('⚠️  No attention data available for this query');
      }
    } catch (error: any) {
      console.error(`\n❌ Error processing query: ${error.message}`);
    }
  }

  // Show aggregate statistics
  console.log('\n\n');
  console.log('═'.repeat(80));
  console.log('📈 AGGREGATE ATTENTION STATISTICS');
  console.log('═'.repeat(80));
  console.log();

  const stats = metaAgent.getAttentionStats();
  console.log(visualizeStats(stats));

  // Compare attentions between queries
  if (results.length >= 2) {
    console.log('\n\n');
    console.log('═'.repeat(80));
    console.log('🔄 ATTENTION COMPARISON');
    console.log('═'.repeat(80));
    console.log();

    const tracker = metaAgent.getAttentionTracker();
    const attention1 = tracker.getQueryAttention(results[0].queryId);
    const attention2 = tracker.getQueryAttention(results[1].queryId);

    if (attention1 && attention2) {
      console.log(compareAttentions(attention1, attention2));
    }
  }

  // Export reports
  console.log('\n\n');
  console.log('═'.repeat(80));
  console.log('💾 EXPORTING REPORTS');
  console.log('═'.repeat(80));
  console.log();

  const outputDir = path.join(__dirname, '..', '..', '..', 'attention-reports');
  const tracker = metaAgent.getAttentionTracker();
  const allAttentions = tracker.getAllAttentions();

  try {
    const savedFiles = await saveAttentionReport(allAttentions, stats, outputDir, 'all');
    console.log('✅ Reports saved successfully:');
    savedFiles.forEach((file) => {
      console.log(`   📄 ${file}`);
    });
    console.log();
  } catch (error: any) {
    console.error(`❌ Error saving reports: ${error.message}`);
  }

  // Show audit export example
  console.log('─'.repeat(80));
  console.log('📋 AUDIT EXPORT (Regulatory Compliance)');
  console.log('─'.repeat(80));
  console.log();

  const auditData = metaAgent.exportAttentionForAudit();
  console.log('Audit export structure:');
  console.log(JSON.stringify(auditData, null, 2).substring(0, 500) + '...');
  console.log();

  // Show query explanation example
  if (results.length > 0) {
    console.log('─'.repeat(80));
    console.log('🔍 DETAILED QUERY EXPLANATION');
    console.log('─'.repeat(80));
    console.log();

    const explanation = metaAgent.explainQuery(results[0].queryId);
    console.log(explanation);
  }

  // Final summary
  console.log('\n\n');
  console.log('═'.repeat(80));
  console.log('🎯 DEMO SUMMARY');
  console.log('═'.repeat(80));
  console.log();
  console.log('Attention tracking provides:');
  console.log();
  console.log('1. ✅ INTERPRETABILITY');
  console.log('   → See EXACTLY which concepts influenced each decision');
  console.log('   → Understand the reasoning chain step-by-step');
  console.log();
  console.log('2. ✅ DEBUGGING');
  console.log('   → Identify where reasoning went wrong');
  console.log('   → Trace back to specific knowledge sources');
  console.log();
  console.log('3. ✅ AUDITING');
  console.log('   → Export complete decision traces for compliance');
  console.log('   → Track which data influenced financial/medical advice');
  console.log();
  console.log('4. ✅ META-LEARNING');
  console.log('   → Discover which concepts are most influential');
  console.log('   → Identify patterns in cross-domain reasoning');
  console.log();
  console.log(`Total cost: $${metaAgent.getTotalCost().toFixed(4)}`);
  console.log(`Total queries: ${results.length}`);
  console.log(`Total traces recorded: ${stats.total_traces}`);
  console.log();
  console.log('═'.repeat(80));
  console.log();
  console.log('🏆 BLACK BOX → GLASS BOX TRANSFORMATION COMPLETE');
  console.log();
  console.log('The AGI system is now fully interpretable.');
  console.log('Every decision can be traced back to its knowledge sources.');
  console.log();
  console.log('═'.repeat(80));
}

// Run demo
runAttentionDemo().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
