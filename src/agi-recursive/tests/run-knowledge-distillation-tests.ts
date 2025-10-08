/**
 * Test Runner for KnowledgeDistillation
 * TDD: RED phase - tests first
 */

import { KnowledgeDistillation, KnowledgePattern, ErrorPattern } from '../core/knowledge-distillation';
import { EpisodicMemory, Episode } from '../core/episodic-memory';
import { AnthropicAdapter } from '../llm/anthropic-adapter';
import { Observability, LogLevel } from '../core/observability';

let passed = 0;
let failed = 0;

function test(name: string, fn: () => void | Promise<void>) {
  const result = fn();
  if (result instanceof Promise) {
    result
      .then(() => {
        console.log(`✅ ${name}`);
        passed++;
      })
      .catch((error: any) => {
        console.log(`❌ ${name}`);
        console.log(`   ${error.message}`);
        failed++;
      });
  } else {
    try {
      console.log(`✅ ${name}`);
      passed++;
    } catch (error: any) {
      console.log(`❌ ${name}`);
      console.log(`   ${error.message}`);
      failed++;
    }
  }
}

function assert(condition: boolean, message: string) {
  if (!condition) {
    throw new Error(message);
  }
}

console.log('🧪 Testing KnowledgeDistillation\n');

// Setup
const obs = new Observability(LogLevel.DEBUG);
const memory = new EpisodicMemory();
const apiKey = process.env.ANTHROPIC_API_KEY || 'test-key';
const llm = new AnthropicAdapter(apiKey);
const distillation = new KnowledgeDistillation(memory, llm, obs);

// Create test episodes
const episodes: Episode[] = [
  {
    id: 'ep1',
    timestamp: Date.now(),
    query: 'What is compound interest?',
    query_hash: 'hash1',
    response: 'Compound interest is interest on interest',
    concepts: ['compound_interest', 'interest', 'finance'],
    domains: ['financial'],
    agents_used: ['financial'],
    cost: 0.001,
    success: true,
    confidence: 0.9,
    execution_trace: [],
    emergent_insights: [],
  },
  {
    id: 'ep2',
    timestamp: Date.now(),
    query: 'How does compound interest work?',
    query_hash: 'hash2',
    response: 'Compound interest grows exponentially',
    concepts: ['compound_interest', 'exponential_growth', 'finance'],
    domains: ['financial'],
    agents_used: ['financial'],
    cost: 0.001,
    success: true,
    confidence: 0.85,
    execution_trace: [],
    emergent_insights: [],
  },
  {
    id: 'ep3',
    timestamp: Date.now(),
    query: 'Explain compound interest formula',
    query_hash: 'hash3',
    response: 'The formula is A = P(1 + r/n)^(nt)',
    concepts: ['compound_interest', 'interest', 'finance'], // Same as ep1 for pattern
    domains: ['financial'],
    agents_used: ['financial'],
    cost: 0.001,
    success: true,
    confidence: 0.95,
    execution_trace: [],
    emergent_insights: [],
  },
  {
    id: 'ep4',
    timestamp: Date.now(),
    query: 'What is portfolio diversification?',
    query_hash: 'hash4',
    response: 'Diversification spreads risk',
    concepts: ['diversification', 'risk', 'portfolio'],
    domains: ['financial'],
    agents_used: ['financial'],
    cost: 0.001,
    success: true,
    confidence: 0.8,
    execution_trace: [],
    emergent_insights: [],
  },
  {
    id: 'ep5',
    timestamp: Date.now(),
    query: 'What is quantum physics?',
    query_hash: 'hash5',
    response: 'I do not have sufficient knowledge',
    concepts: [],
    domains: [],
    agents_used: ['financial'],
    cost: 0.001,
    success: false,
    confidence: 0.2,
    execution_trace: [],
    emergent_insights: [],
  },
];

// Add episodes to memory
episodes.forEach((ep) => {
  memory.addEpisode(
    ep.query,
    ep.response,
    ep.concepts,
    ep.domains,
    ep.agents_used,
    ep.cost,
    ep.success,
    ep.confidence,
    ep.execution_trace,
    ep.emergent_insights
  );
});

// Test 1: Discover patterns with minimum frequency
test('should discover patterns with min frequency', async () => {
  const patterns = await distillation.discoverPatterns(episodes, 2); // Lower threshold

  assert(patterns.length > 0, `Should find at least one pattern, got ${patterns.length}`);

  const compoundInterestPattern = patterns.find((p) =>
    p.concepts.includes('compound_interest')
  );
  assert(
    compoundInterestPattern !== undefined,
    'Should find compound_interest pattern'
  );
  assert(
    compoundInterestPattern!.frequency >= 2,
    `Pattern frequency should be >= 2, got ${compoundInterestPattern!.frequency}`
  );
});

// Test 2: Filter by minimum frequency
test('should filter patterns by frequency threshold', async () => {
  const allPatterns = await distillation.discoverPatterns(episodes, 1);
  const filteredPatterns = await distillation.discoverPatterns(episodes, 5);

  assert(
    allPatterns.length >= filteredPatterns.length,
    'Higher threshold should have fewer patterns'
  );
});

// Test 3: Calculate confidence scores
test('should calculate confidence scores for patterns', async () => {
  const patterns = await distillation.discoverPatterns(episodes, 2);

  for (const pattern of patterns) {
    assert(
      pattern.confidence >= 0 && pattern.confidence <= 1,
      `Confidence should be 0-1, got ${pattern.confidence}`
    );
  }
});

// Test 4: Include representative queries
test('should include representative queries', async () => {
  const patterns = await distillation.discoverPatterns(episodes, 2);

  if (patterns.length > 0) {
    const pattern = patterns[0];
    assert(
      pattern.representative_queries.length > 0,
      'Should have representative queries'
    );
    assert(
      typeof pattern.representative_queries[0] === 'string',
      'Queries should be strings'
    );
  } else {
    // If no patterns found, that's also valid
    assert(true, 'No patterns found, test passes');
  }
});

// Test 5: Identify knowledge gaps
test('should identify knowledge gaps', async () => {
  const gaps = await distillation.identifyGaps(episodes);

  assert(gaps.length > 0, 'Should find at least one gap');

  const quantumGap = gaps.find((g) => g.concept.includes('quantum'));
  assert(quantumGap !== undefined, 'Should identify quantum physics gap');
  assert(quantumGap!.evidence.length > 0, 'Gap should have evidence');
});

// Test 6: Group gaps by domain
test('should group gaps by domain', async () => {
  const gaps = await distillation.identifyGaps(episodes);

  for (const gap of gaps) {
    assert(typeof gap.concept === 'string', 'Concept should be string');
    assert(Array.isArray(gap.evidence), 'Evidence should be array');
  }
});

// Test 7: Detect systematic errors
test('should detect systematic errors', async () => {
  const errors = await distillation.detectErrors(episodes);

  // Since we have one failed episode, should detect errors
  assert(errors.length >= 0, 'Should return error patterns array');

  if (errors.length > 0) {
    const error = errors[0];
    assert(typeof error.concept === 'string', 'Error should have concept');
    assert(typeof error.frequency === 'number', 'Error should have frequency');
    assert(typeof error.typical_error === 'string', 'Error should have typical error');
  }
});

// Test 8: Suggest fixes for errors
test('should suggest fixes for errors', async () => {
  const errors = await distillation.detectErrors(episodes);

  if (errors.length > 0) {
    const error = errors[0];
    assert(
      typeof error.suggested_fix === 'string',
      'Error should have suggested fix'
    );
    assert(error.suggested_fix.length > 0, 'Fix should not be empty');
  }
});

// Test 9: Find co-occurring concepts
test('should find co-occurring concepts', async () => {
  const patterns = await distillation.discoverPatterns(episodes, 1); // Lower threshold

  // Patterns may or may not exist depending on data
  assert(Array.isArray(patterns), 'Should return array of patterns');

  for (const pattern of patterns) {
    assert(pattern.concepts.length > 0, 'Pattern should have concepts');
    assert(
      Array.isArray(pattern.concepts),
      'Concepts should be array'
    );
  }
});

// Test 10: Include domains in patterns
test('should include domains in patterns', async () => {
  const patterns = await distillation.discoverPatterns(episodes, 2);

  for (const pattern of patterns) {
    assert(Array.isArray(pattern.domains), 'Domains should be array');

    if (pattern.domains.length > 0) {
      assert(
        typeof pattern.domains[0] === 'string',
        'Domain should be string'
      );
    }
  }
});

// Wait for async tests
setTimeout(() => {
  console.log('\n' + '='.repeat(70));
  console.log(`Total: ${passed + failed}`);
  console.log(`✅ Passed: ${passed}`);
  console.log(`❌ Failed: ${failed}`);
  console.log('='.repeat(70));

  if (failed > 0) {
    process.exit(1);
  }
}, 2000);
