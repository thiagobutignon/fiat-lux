/**
 * E2E: LLM + Constitutional AI Integration Test
 *
 * Tests the complete LLM integration across all organism components:
 * - Code synthesis (ROXO)
 * - Pattern detection (ROXO)
 * - Knowledge ingestion with semantic embeddings (ROXO)
 * - Intent analysis (CINZA)
 * - Constitutional validation (all nodes)
 *
 * Requires: ANTHROPIC_API_KEY environment variable
 * Expected cost: ~$1.20 per complete run
 * Expected time: ~2-3 minutes
 */

import { describe, it, beforeEach, afterEach, expect } from '../src/shared/utils/test-runner';
import { createGlassOrganism } from '../src/grammar-lang/glass/builder';
import { GlassIngestion } from '../src/grammar-lang/glass/ingestion';
import { PatternDetectionEngine } from '../src/grammar-lang/glass/patterns';
import { CodeEmergenceEngine } from '../src/grammar-lang/glass/emergence';
import { createGlassLLM } from '../src/grammar-lang/glass/llm-adapter';
import * as fs from 'fs';

const TEST_ORGANISM_DIR = 'test_e2e_llm';
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;

// Skip tests if no API key
const runTests = ANTHROPIC_API_KEY ? describe : describe.skip;

runTests('E2E: LLM Integration with Real API', () => {

  afterEach(() => {
    if (fs.existsSync(TEST_ORGANISM_DIR)) {
      fs.rmSync(TEST_ORGANISM_DIR, { recursive: true });
    }
  });

  it('should create organism with constitutional AI', async () => {
    const builder = createGlassOrganism({
      name: 'oncology-research-agent',
      specialization: 'oncology',
      constitutional: [
        'transparency',
        'honesty',
        'privacy',
        'safety'
      ]
    });

    const organism = builder.getOrganism();

    // Verify constitutional setup
    expect.toEqual(organism.constitutional.agent_type, 'biology');
    expect.toBeTruthy(organism.constitutional.boundaries.cannot_diagnose);
    expect.toBeTruthy(organism.constitutional.boundaries.must_cite_sources);
    expect.toEqual(organism.metadata.maturity, 0);
    expect.toEqual(organism.metadata.stage, 'nascent');

    console.log('✅ Organism created with constitutional AI');
    console.log(`   Agent type: ${organism.constitutional.agent_type}`);
    console.log(`   Maturity: ${organism.metadata.maturity}%`);
  });

  it('should ingest knowledge with LLM semantic embeddings', async () => {
    const builder = createGlassOrganism({
      name: 'oncology-agent',
      specialization: 'oncology'
    });
    const organism = builder.getOrganism();

    // Create ingestion with LLM budget
    const ingestion = new GlassIngestion(organism, 0.15); // $0.15 budget

    console.log('🧬 Starting knowledge ingestion with LLM embeddings...');

    // Ingest knowledge
    await ingestion.ingest({
      source: {
        type: 'text',
        text: `Pembrolizumab is a humanized antibody used in cancer immunotherapy that blocks PD-1.
               Clinical trials show significant efficacy in non-small cell lung cancer (NSCLC).
               The PD-1/PD-L1 pathway is a key immune checkpoint mechanism.
               Combination therapy with chemotherapy shows improved outcomes.`
      },
      metadata: {
        title: 'Pembrolizumab in NSCLC',
        authors: ['Smith et al.'],
        year: 2023
      }
    });

    const stats = ingestion.getStats();
    const costStats = ingestion.getCostStats();

    console.log('📊 Ingestion stats:');
    console.log(`   Documents ingested: ${stats.total_ingested}`);
    console.log(`   Total cost: $${costStats.total_cost.toFixed(4)}`);
    console.log(`   Remaining budget: $${costStats.remaining_budget.toFixed(4)}`);

    // Verify ingestion
    expect.toEqual(stats.total_ingested, 1);
    expect.toBeLessThan(costStats.total_cost, 0.15);
    expect.toBeFalsy(costStats.over_budget);

  }, 60000); // 60s timeout

  it('should detect patterns with LLM semantic analysis', async () => {
    const builder = createGlassOrganism({
      name: 'pattern-test',
      specialization: 'oncology'
    });
    const organism = builder.getOrganism();

    // Populate knowledge base
    organism.knowledge.papers.count = 5;
    organism.knowledge.patterns = {
      'efficacy_analysis': {
        name: 'efficacy_analysis',
        type: 'semantic',
        keywords: ['efficacy', 'response rate', 'outcomes', 'survival'],
        frequency: 8,
        confidence: 0.85,
        examples: []
      },
      'trial_design': {
        name: 'trial_design',
        type: 'semantic',
        keywords: ['randomized', 'controlled', 'phase', 'trial'],
        frequency: 6,
        confidence: 0.78,
        examples: []
      },
      'safety_profile': {
        name: 'safety_profile',
        type: 'semantic',
        keywords: ['adverse events', 'toxicity', 'safety', 'tolerability'],
        frequency: 7,
        confidence: 0.82,
        examples: []
      }
    };

    console.log('🔍 Starting LLM pattern detection...');

    // Create pattern detector with LLM
    const detector = new PatternDetectionEngine(organism, true, 0.3); // useLLM=true, $0.30 budget

    // Analyze with LLM
    const result = await detector.analyzeWithLLM();

    console.log('📊 Pattern detection results:');
    console.log(`   Patterns found: ${result.patterns.length}`);
    console.log(`   Correlations: ${result.correlations.length}`);
    console.log(`   Emergence candidates: ${result.emergence_candidates.length}`);
    console.log(`   Total cost: $${result.cost.toFixed(4)}`);

    // Verify results
    expect.toBeGreaterThan(result.patterns.length, 0);
    expect.toBeLessThan(result.cost, 0.3);

  }, 90000); // 90s timeout

  it('should emerge code with LLM synthesis', async () => {
    const builder = createGlassOrganism({
      name: 'emergence-test',
      specialization: 'oncology'
    });
    const organism = builder.getOrganism();

    // Set up organism with patterns
    organism.knowledge.papers.count = 10;
    organism.knowledge.patterns = {
      'efficacy_analysis': {
        name: 'efficacy_analysis',
        type: 'semantic',
        keywords: ['efficacy', 'response rate', 'outcomes'],
        frequency: 12,
        confidence: 0.88,
        examples: []
      }
    };

    console.log('⚗️  Starting code emergence with LLM synthesis...');

    // Create emergence engine with LLM budget
    const emergenceEngine = new CodeEmergenceEngine(organism, 0.5); // $0.50 budget

    // Create emergence candidates
    const candidates = [
      {
        suggested_function_name: 'analyzeEfficacy',
        suggested_signature: '(treatment: string, data: ClinicalData) -> EfficacyResult',
        pattern: organism.knowledge.patterns['efficacy_analysis'],
        supporting_patterns: ['trial_design', 'safety_profile']
      }
    ];

    // Emerge code
    const results = await emergenceEngine.emerge(candidates);

    console.log('📊 Code emergence results:');
    console.log(`   Functions emerged: ${results.length}`);
    if (results.length > 0) {
      console.log(`   Validation passed: ${results[0].validation_passed}`);
      console.log(`   Code length: ${results[0].implementation.length} chars`);
    }
    console.log(`   Total cost: $${emergenceEngine.getTotalCost().toFixed(4)}`);

    // Verify emergence
    expect.toBeGreaterThan(results.length, 0);
    expect.toBeTruthy(results[0].validation_passed);
    expect.toBeGreaterThan(results[0].implementation.length, 50);
    expect.toBeLessThan(emergenceEngine.getTotalCost(), 0.5);

  }, 120000); // 120s timeout

  it('should validate constitutional compliance across LLM operations', async () => {
    console.log('⚖️  Testing constitutional validation...');

    // Create LLM with constitutional validation
    const llm = createGlassLLM('glass-core', 0.05);

    // Test 1: Valid response with high confidence
    const validResponse = await llm.invoke(
      'Explain the mechanism of action of pembrolizumab in simple terms.',
      {
        task: 'reasoning',
        max_tokens: 200,
        enable_constitutional: true
      }
    );

    console.log('📊 Constitutional check (valid response):');
    console.log(`   Passed: ${validResponse.constitutional_check?.passed}`);
    console.log(`   Violations: ${validResponse.constitutional_check?.violations.length || 0}`);
    console.log(`   Cost: $${validResponse.usage.cost_usd.toFixed(4)}`);

    // Verify constitutional validation
    expect.toBeTruthy(validResponse.constitutional_check);
    expect.toBeTruthy(validResponse.constitutional_check!.passed ||
                      validResponse.constitutional_check!.violations.length === 0);
    expect.toBeLessThan(validResponse.usage.cost_usd, 0.05);

  }, 30000); // 30s timeout

  it('should track total cost across complete organism lifecycle', async () => {
    const builder = createGlassOrganism({
      name: 'cost-tracking-test',
      specialization: 'oncology'
    });
    const organism = builder.getOrganism();

    console.log('💰 Testing cost tracking across organism lifecycle...');

    let totalCost = 0;

    // 1. Ingestion
    const ingestion = new GlassIngestion(organism, 0.10);
    await ingestion.ingest({
      source: {
        type: 'text',
        text: 'Pembrolizumab shows efficacy in NSCLC treatment with manageable adverse events.'
      },
      metadata: { title: 'Brief study', authors: [], year: 2023 }
    });
    totalCost += ingestion.getCostStats().total_cost;

    // 2. Pattern detection
    organism.knowledge.papers.count = 3;
    organism.knowledge.patterns = {
      'efficacy': {
        name: 'efficacy',
        type: 'semantic',
        keywords: ['efficacy', 'outcomes'],
        frequency: 5,
        confidence: 0.8,
        examples: []
      }
    };

    const detector = new PatternDetectionEngine(organism, true, 0.20);
    const patternResult = await detector.analyzeWithLLM();
    totalCost += patternResult.cost;

    // 3. Code emergence
    const emergenceEngine = new CodeEmergenceEngine(organism, 0.40);
    const candidates = [{
      suggested_function_name: 'analyzeOutcome',
      suggested_signature: '(data: any) -> Result',
      pattern: organism.knowledge.patterns['efficacy'],
      supporting_patterns: []
    }];
    const emergenceResults = await emergenceEngine.emerge(candidates);
    totalCost += emergenceEngine.getTotalCost();

    console.log('💰 Total lifecycle cost breakdown:');
    console.log(`   Ingestion: $${ingestion.getCostStats().total_cost.toFixed(4)}`);
    console.log(`   Pattern detection: $${patternResult.cost.toFixed(4)}`);
    console.log(`   Code emergence: $${emergenceEngine.getTotalCost().toFixed(4)}`);
    console.log(`   TOTAL: $${totalCost.toFixed(4)}`);

    // Verify cost is within budget
    expect.toBeLessThan(totalCost, 1.50); // Target was ~$1.20, allow some buffer
    expect.toBeGreaterThan(emergenceResults.length, 0);

  }, 180000); // 180s timeout (3 min)

});

// If no API key, show message
if (!ANTHROPIC_API_KEY) {
  describe('E2E: LLM Integration - API Key Required', () => {
    it('should skip tests when ANTHROPIC_API_KEY is not set', () => {
      console.log('\n⚠️  ANTHROPIC_API_KEY not set. Skipping LLM integration tests.');
      console.log('   To run these tests:');
      console.log('   export ANTHROPIC_API_KEY="sk-ant-..."');
      console.log('   npm test tests/e2e-llm-integration.test.ts\n');
      expect.toBeTruthy(true); // Always pass
    });
  });
}
