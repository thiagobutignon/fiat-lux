/**
 * Slice Navigation Demo
 *
 * Demonstrates dynamic knowledge discovery and navigation through slices.
 *
 * Shows:
 * 1. Slice indexing and search
 * 2. Loading slice content on demand
 * 3. Cross-domain navigation
 * 4. Connection discovery
 * 5. Knowledge composition
 */

import 'dotenv/config';
import path from 'path';
import { SliceNavigator } from '../core/slice-navigator';

console.log('═══════════════════════════════════════════════════════════════');
console.log('🧭 Slice Navigation Demo - Dynamic Knowledge Discovery');
console.log('═══════════════════════════════════════════════════════════════\n');

async function main() {
  // Initialize navigator
  const slicesDir = path.join(__dirname, '..', 'slices');
  const navigator = new SliceNavigator(slicesDir);

  console.log('📂 Initializing Slice Navigator...');
  console.log(`   Slices Directory: ${slicesDir}\n`);

  await navigator.initialize();

  // Get stats
  const stats = navigator.getStats();
  console.log('📊 Index Statistics:');
  console.log(`   Total Slices: ${stats.total_slices}`);
  console.log(`   Total Concepts: ${stats.total_concepts}`);
  console.log(`   Domains: ${stats.domains}`);
  console.log(`   Cache Size: ${stats.cache_size}\n`);

  console.log('───────────────────────────────────────────────────────────────\n');

  // ============================================================================
  // TEST 1: Search by Concept
  // ============================================================================

  console.log('📋 TEST 1: Search by Concept');
  console.log('─────────────────────────────────────────────────────────────\n');

  console.log('🔍 Searching for "homeostasis"...\n');

  const homeostasisResults = await navigator.search('homeostasis');

  console.log(`Found ${homeostasisResults.length} slices:\n`);

  for (const result of homeostasisResults) {
    console.log(`   [${result.relevance_score.toFixed(2)}] ${result.metadata.id}`);
    console.log(`       Domain: ${result.metadata.domain}`);
    console.log(`       Title: ${result.metadata.title}`);
    console.log(`       Matched: ${result.matched_concepts.join(', ')}\n`);
  }

  console.log('───────────────────────────────────────────────────────────────\n');

  // ============================================================================
  // TEST 2: Load Full Slice Content
  // ============================================================================

  console.log('📋 TEST 2: Load Full Slice Content');
  console.log('─────────────────────────────────────────────────────────────\n');

  if (homeostasisResults.length > 0) {
    const firstSliceId = homeostasisResults[0].slice_id;
    console.log(`📖 Loading slice: ${firstSliceId}\n`);

    const sliceContext = await navigator.loadSlice(firstSliceId);

    console.log(`   Title: ${sliceContext.slice.metadata.title}`);
    console.log(`   Domain: ${sliceContext.slice.metadata.domain}`);
    console.log(`   Concepts: [${sliceContext.slice.metadata.concepts.join(', ')}]`);
    console.log(`   Version: ${sliceContext.slice.metadata.version}\n`);

    console.log('   Knowledge Preview:');
    const knowledgePreview = sliceContext.slice.knowledge.split('\n').slice(0, 10).join('\n');
    console.log(knowledgePreview.split('\n').map((line) => `     ${line}`).join('\n'));
    console.log(`     ... (${sliceContext.slice.knowledge.length} chars total)\n`);

    if (sliceContext.slice.principles && sliceContext.slice.principles.length > 0) {
      console.log('   Principles:');
      sliceContext.slice.principles.slice(0, 3).forEach((p) => {
        console.log(`     • ${p}`);
      });
      console.log('');
    }

    if (sliceContext.related_slices.length > 0) {
      console.log('   Related Slices:');
      sliceContext.related_slices.forEach((related) => {
        console.log(`     → ${related.id} (${related.domain})`);
      });
      console.log('');
    }
  }

  console.log('───────────────────────────────────────────────────────────────\n');

  // ============================================================================
  // TEST 3: Search by Domain
  // ============================================================================

  console.log('📋 TEST 3: Search by Domain');
  console.log('─────────────────────────────────────────────────────────────\n');

  const domains = ['financial', 'biology', 'systems'];

  for (const domain of domains) {
    const slices = await navigator.searchByDomain(domain);
    console.log(`   ${domain.toUpperCase()}: ${slices.length} slices`);

    if (slices.length > 0) {
      slices.forEach((slice) => {
        console.log(`     • ${slice.id}: ${slice.title}`);
      });
    }

    console.log('');
  }

  console.log('───────────────────────────────────────────────────────────────\n');

  // ============================================================================
  // TEST 4: Find Cross-Domain Connections
  // ============================================================================

  console.log('📋 TEST 4: Find Cross-Domain Connections');
  console.log('─────────────────────────────────────────────────────────────\n');

  console.log('🔗 Finding connections between domains...\n');

  const allSlices = navigator.getAllSlices();

  if (allSlices.length >= 2) {
    // Try to find connection between first two slices from different domains
    const slice1 = allSlices[0];
    const slice2 = allSlices.find((s) => s.domain !== slice1.domain);

    if (slice2) {
      console.log(`   From: ${slice1.id} (${slice1.domain})`);
      console.log(`   To: ${slice2.id} (${slice2.domain})\n`);

      const connection = await navigator.findConnections(slice1.id, slice2.id);

      if (connection) {
        console.log('   ✅ Connection Found!');
        console.log(`   Path: ${connection.path.join(' → ')}`);
        console.log(`   Shared Concepts: [${connection.shared_concepts.join(', ')}]\n`);
      } else {
        console.log('   ℹ️ No direct connection found\n');
      }
    }
  }

  console.log('───────────────────────────────────────────────────────────────\n');

  // ============================================================================
  // TEST 5: Concept-Based Navigation (Simulating Agent Discovery)
  // ============================================================================

  console.log('📋 TEST 5: Agent Knowledge Discovery Simulation');
  console.log('─────────────────────────────────────────────────────────────\n');

  console.log('🤖 Scenario: Financial agent needs to understand budget regulation\n');

  // Step 1: Search for relevant concept
  console.log('   Step 1: Search for "budget"...');
  const budgetResults = await navigator.search('budget');

  if (budgetResults.length > 0) {
    const budgetSlice = budgetResults[0];
    console.log(`   ✓ Found: ${budgetSlice.slice_id}\n`);

    // Step 2: Load slice content
    console.log('   Step 2: Load slice content...');
    const budgetContext = await navigator.loadSlice(budgetSlice.slice_id);
    console.log(`   ✓ Loaded: ${budgetContext.slice.metadata.title}\n`);

    // Step 3: Discover connections
    console.log('   Step 3: Discover cross-domain connections...');
    const connections = budgetContext.slice.metadata.connects_to;

    if (Object.keys(connections).length > 0) {
      console.log(`   ✓ Found ${Object.keys(connections).length} connections:\n`);

      for (const [domain, sliceId] of Object.entries(connections)) {
        console.log(`     → ${domain}: ${sliceId}`);

        // Step 4: Load connected slice
        const connectedContext = await navigator.loadSlice(sliceId);
        console.log(`       "${connectedContext.slice.metadata.title}"`);
        console.log(
          `       Concepts: [${connectedContext.slice.metadata.concepts.slice(0, 3).join(', ')}...]`
        );
        console.log('');
      }
    }

    // Step 5: Compose knowledge
    console.log('   Step 4: Compose cross-domain knowledge...');
    console.log(`   ✓ Agent now has integrated understanding:`);
    console.log(`     • Budget mechanics (financial domain)`);
    console.log(`     • Homeostasis principles (biology domain)`);
    console.log(`     • Feedback loop structure (systems domain)`);
    console.log(`     → EMERGENT: "Budget as Biological System" 🎯\n`);
  }

  console.log('───────────────────────────────────────────────────────────────\n');

  // ============================================================================
  // TEST 6: Cache Performance
  // ============================================================================

  console.log('📋 TEST 6: Cache Performance');
  console.log('─────────────────────────────────────────────────────────────\n');

  if (allSlices.length > 0) {
    const testSliceId = allSlices[0].id;

    console.log(`   Loading slice "${testSliceId}" (first time)...`);
    const start1 = performance.now();
    await navigator.loadSlice(testSliceId);
    const time1 = performance.now() - start1;
    console.log(`   Time: ${time1.toFixed(2)}ms (disk read)\n`);

    console.log(`   Loading same slice (cached)...`);
    const start2 = performance.now();
    await navigator.loadSlice(testSliceId);
    const time2 = performance.now() - start2;
    console.log(`   Time: ${time2.toFixed(2)}ms (cache hit)\n`);

    const speedup = time1 / time2;
    console.log(`   💨 Speedup: ${speedup.toFixed(1)}x faster from cache\n`);
  }

  console.log('───────────────────────────────────────────────────────────────\n');

  // Final stats
  const finalStats = navigator.getStats();
  console.log('📊 Final Statistics:');
  console.log(`   Total Slices Indexed: ${finalStats.total_slices}`);
  console.log(`   Unique Concepts: ${finalStats.total_concepts}`);
  console.log(`   Domains Covered: ${finalStats.domains}`);
  console.log(`   Slices in Cache: ${finalStats.cache_size}\n`);

  console.log('═══════════════════════════════════════════════════════════════');
  console.log('🎓 KEY INSIGHTS');
  console.log('═══════════════════════════════════════════════════════════════\n');

  console.log(`The Slice Navigator enables:

✅ DYNAMIC KNOWLEDGE LOADING
   - Agents don't need all knowledge upfront
   - Load only relevant slices on demand
   - Scales to unlimited knowledge base

✅ DISCOVERABLE CONNECTIONS
   - Slices explicitly define cross-domain links
   - Agents navigate between related concepts
   - Enables emergent cross-domain insights

✅ STRUCTURED KNOWLEDGE
   - Each slice has metadata, content, examples, principles
   - Consistent structure across domains
   - Machine-readable and human-readable

✅ PERFORMANCE OPTIMIZATION
   - In-memory caching of frequently used slices
   - Inverted index for fast concept search
   - Domain-based filtering

✅ COMPOSABLE UNDERSTANDING
   - Financial agent loads budget-homeostasis slice
   - Discovers connection to cellular-homeostasis
   - Loads biology slice
   - Composes integrated knowledge
   - Result: "Budget as Biological System" 🧬💰

This is knowledge-as-a-graph, not knowledge-as-a-monolith.
Agents explore, discover, and compose understanding dynamically.
`);

  console.log('═══════════════════════════════════════════════════════════════\n');
}

// Run the demo
main().catch((error) => {
  console.error('❌ Error:', error);
  process.exit(1);
});
