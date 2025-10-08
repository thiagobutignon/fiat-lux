/**
 * Advanced Caching Demo
 *
 * Demonstrates how to achieve >95% cache hit rate
 */

import { CachedMetaAgent } from '../packages/core/src/ilp/meta-agent-cached.js';

async function demonstrateAdvancedCaching() {
  console.log('🚀 Advanced Caching Demo\n');

  // ========================================================================
  // 1. Create Cached Meta-Agent
  // ========================================================================

  const agent = new CachedMetaAgent({
    apiKey: process.env.ANTHROPIC_API_KEY || 'test-key',
    maxDepth: 5,
    maxInvocations: 10,
    maxCostUSD: 1.0,
    cacheConfig: {
      maxSize: 10000,
      ttlMs: 3600000, // 1 hour
      similarityThreshold: 0.85, // 85% similarity for semantic matching
      enableSemanticCache: true,
      enableTemplateCache: true,
      enableLRU: true,
    },
  });

  await agent.initialize();

  // ========================================================================
  // 2. Pre-warm Cache with Common Queries
  // ========================================================================

  console.log('📝 Step 1: Pre-warming cache with common queries...\n');

  const commonQueries = [
    { query: 'What is compound interest?', expectedAnswer: 'Compound interest is...' },
    { query: 'How do I calculate ROI?', expectedAnswer: 'ROI is calculated by...' },
    { query: 'What is diversification?', expectedAnswer: 'Diversification is...' },
    { query: 'Explain the time value of money', expectedAnswer: 'Time value of money means...' },
    { query: 'What is dollar cost averaging?', expectedAnswer: 'Dollar cost averaging is...' },
  ];

  const warmed = await agent.preWarmCache(commonQueries);
  console.log(`✅ Pre-warmed cache with ${warmed} queries\n`);

  // ========================================================================
  // 3. Demonstrate Exact Matching
  // ========================================================================

  console.log('📊 Step 2: Exact matching (normalized)\n');

  // These will all hit the same cache entry
  const exactVariations = [
    'What is compound interest?',           // Original
    'what is compound interest',            // Lowercase
    'What is compound interest',            // No punctuation
    '  what   is  compound  interest  ',  // Extra whitespace
    'What is compound interest?',           // Exact duplicate
  ];

  for (const query of exactVariations) {
    const result = await agent.process(query);
    console.log(`Query: "${query}"`);
    console.log(`Cache Hit: ${result.cache_hit}`);
    console.log(`Cache Type: ${result.cache_type || 'exact'}\n`);
  }

  let stats = agent.getCacheStats();
  console.log(`Exact Matching Hit Rate: ${(stats.hitRate * 100).toFixed(1)}%\n`);

  // ========================================================================
  // 4. Demonstrate Template Matching
  // ========================================================================

  console.log('📊 Step 3: Template matching (abstracted parameters)\n');

  // Add a template-based query
  await agent.process('I want to invest $1000 for 5 years');

  // These will match the template (numbers abstracted)
  const templateVariations = [
    'I want to invest $5000 for 10 years',
    'I want to invest $500 for 2 years',
    'I want to invest $10000 for 20 years',
  ];

  for (const query of templateVariations) {
    const result = await agent.process(query);
    console.log(`Query: "${query}"`);
    console.log(`Cache Hit: ${result.cache_hit}`);
    console.log(`Cache Type: ${result.cache_type}\n`);
  }

  stats = agent.getCacheStats();
  console.log(`After Template Matching Hit Rate: ${(stats.hitRate * 100).toFixed(1)}%\n`);

  // ========================================================================
  // 5. Demonstrate Semantic Matching
  // ========================================================================

  console.log('📊 Step 4: Semantic matching (similar queries)\n');

  // Add base query
  await agent.process('What is machine learning?');

  // Similar queries (high word overlap)
  const semanticVariations = [
    'What is deep learning?',           // Similar domain
    'What is supervised learning?',      // Similar concepts
    'What is reinforcement learning?',   // Related topic
  ];

  for (const query of semanticVariations) {
    const result = await agent.process(query);
    const similarity = agent.calculateSimilarity('What is machine learning?', query);

    console.log(`Query: "${query}"`);
    console.log(`Similarity: ${(similarity * 100).toFixed(1)}%`);
    console.log(`Cache Hit: ${result.cache_hit}`);
    console.log(`Cache Type: ${result.cache_type}\n`);
  }

  stats = agent.getCacheStats();
  console.log(`After Semantic Matching Hit Rate: ${(stats.hitRate * 100).toFixed(1)}%\n`);

  // ========================================================================
  // 6. Show Detailed Statistics
  // ========================================================================

  console.log('📊 Step 5: Final Cache Statistics\n');

  const finalStats = agent.getCacheStats();

  console.log('=== Overall Performance ===');
  console.log(`Total Queries: ${finalStats.hits + finalStats.misses}`);
  console.log(`Cache Hits: ${finalStats.hits}`);
  console.log(`Cache Misses: ${finalStats.misses}`);
  console.log(`Hit Rate: ${(finalStats.hitRate * 100).toFixed(1)}%\n`);

  console.log('=== Hit Type Breakdown ===');
  console.log(`Exact Hits: ${finalStats.exactHits} (${(finalStats.hitRateBreakdown.exact * 100).toFixed(1)}%)`);
  console.log(`Semantic Hits: ${finalStats.semanticHits} (${(finalStats.hitRateBreakdown.semantic * 100).toFixed(1)}%)`);
  console.log(`Template Hits: ${finalStats.templateHits} (${(finalStats.hitRateBreakdown.template * 100).toFixed(1)}%)\n`);

  console.log('=== Cache Health ===');
  console.log(`Cache Size: ${finalStats.size}/${finalStats.maxSize}`);
  console.log(`Evictions: ${finalStats.evictions}`);
  console.log(`Avg Hits per Query: ${finalStats.avgHitCount.toFixed(1)}\n`);

  console.log('=== Top 5 Popular Queries ===');
  finalStats.popularQueries.slice(0, 5).forEach((q, i) => {
    console.log(`${i + 1}. "${q.query}" - ${q.hits} hits`);
  });
  console.log();

  if (finalStats.recommendations.length > 0) {
    console.log('=== Recommendations ===');
    finalStats.recommendations.forEach((rec, i) => {
      console.log(`${i + 1}. ${rec}`);
    });
    console.log();
  }

  // ========================================================================
  // 7. Demonstrate Cache Tuning
  // ========================================================================

  console.log('📊 Step 6: Dynamic Cache Tuning\n');

  // If hit rate is low, try adjusting similarity threshold
  if (finalStats.hitRate < 0.9) {
    console.log('⚠️  Hit rate below 90%, lowering similarity threshold...\n');

    agent.updateCacheConfig({
      similarityThreshold: 0.75, // More lenient matching
    });

    // Process some queries to see improvement
    await agent.process('What is neural networks?');
    await agent.process('What is artificial neural networks?');

    const newStats = agent.getCacheStats();
    console.log(`New Hit Rate: ${(newStats.hitRate * 100).toFixed(1)}%\n`);
  }

  // ========================================================================
  // 8. Show Normalized Queries (Debugging)
  // ========================================================================

  console.log('📊 Step 7: Query Normalization Examples\n');

  const debugQueries = [
    'What is AI?',
    "  What's  AI?  ",
    'what is ai',
    'Could you please tell me what AI is?',
  ];

  console.log('Original → Normalized:');
  debugQueries.forEach((q) => {
    const normalized = agent.normalizeQuery(q);
    console.log(`"${q}" → "${normalized}"`);
  });
  console.log();

  // ========================================================================
  // 9. Show Template Extraction (Debugging)
  // ========================================================================

  console.log('📊 Step 8: Template Extraction Examples\n');

  const templateQueries = [
    'I need 5 apples',
    'I need 10 oranges',
    'It costs $100',
    'It costs $500',
  ];

  console.log('Original → Template:');
  templateQueries.forEach((q) => {
    const template = agent.getQueryTemplate(q);
    console.log(`"${q}" → "${template}"`);
  });
  console.log();

  // ========================================================================
  // 10. Cost Savings Calculation
  // ========================================================================

  console.log('💰 Step 9: Cost Savings Estimate\n');

  const avgCostPerQuery = 0.05; // $0.05 per query without cache
  const totalQueries = finalStats.hits + finalStats.misses;
  const costWithoutCache = totalQueries * avgCostPerQuery;
  const costWithCache = finalStats.misses * avgCostPerQuery; // Only pay for misses
  const savings = costWithoutCache - costWithCache;
  const savingsPercent = (savings / costWithoutCache) * 100;

  console.log(`Queries Processed: ${totalQueries}`);
  console.log(`Cost without cache: $${costWithoutCache.toFixed(2)}`);
  console.log(`Cost with cache: $${costWithCache.toFixed(2)}`);
  console.log(`Savings: $${savings.toFixed(2)} (${savingsPercent.toFixed(1)}%)\n`);

  // ========================================================================
  // Summary
  // ========================================================================

  console.log('✅ Summary\n');
  console.log(`Final Cache Hit Rate: ${(finalStats.hitRate * 100).toFixed(1)}%`);
  console.log(`Cost Reduction: ${savingsPercent.toFixed(1)}%`);
  console.log(`Queries Cached: ${finalStats.size}`);
  console.log('\n🎉 Advanced caching demo complete!');
}

// Run the demo
if (import.meta.url === `file://${process.argv[1]}`) {
  demonstrateAdvancedCaching().catch(console.error);
}

export { demonstrateAdvancedCaching };
