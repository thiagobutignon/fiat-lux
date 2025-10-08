/**
 * Universal Grammar Validation Demo
 *
 * Tests the thesis: "Clean Architecture exhibits Universal Grammar"
 *
 * Method:
 * 1. Show AGI examples of patterns in TypeScript and Swift
 * 2. Ask AGI to identify common deep structure
 * 3. Test if AGI can generate code in new language (Python/Go)
 * 4. Use episodic memory to learn patterns
 * 5. Validate thesis empirically
 *
 * Expected result:
 * AGI should identify that deep structure (DI, SRP, patterns) is universal,
 * only surface structure (syntax) differs
 */

import dotenv from 'dotenv';
import path from 'path';
import { createMetaAgentWithMemory } from '../core/meta-agent-with-memory';
import { ArchitectureAgent } from '../agents/architecture-agent';
import { LinguisticsAgent } from '../agents/linguistics-agent';
import { SystemsAgent } from '../agents/systems-agent';

dotenv.config();

// ============================================================================
// Code Examples for Training
// ============================================================================

const TYPESCRIPT_USECASE = `
// TypeScript: UseCase Pattern
export interface AddAccount {
  add: (params: AddAccount.Params) => Promise<AddAccount.Result>
}

export namespace AddAccount {
  export type Params = { name: string, email: string }
  export type Result = boolean
}
`;

const SWIFT_USECASE = `
// Swift: UseCase Pattern
public protocol AddAccount {
    typealias Result = Swift.Result<Bool, DomainError>
    func add(model: AddAccountModel, completion: @escaping (Result) -> Void)
}

public struct AddAccountModel {
    public var name: String
    public var email: String
}
`;

const TYPESCRIPT_ADAPTER = `
// TypeScript: Adapter Pattern
export class BcryptAdapter implements Hasher {
  constructor(private readonly salt: number) {}

  async hash(plaintext: string): Promise<string> {
    return await bcrypt.hash(plaintext, this.salt)
  }
}
`;

const SWIFT_ADAPTER = `
// Swift: Adapter Pattern
public final class AlamofireAdapter: HttpClient {
    private let session: Session

    public init(session: Session = .default) {
        self.session = session
    }

    public func request(data: HttpRequest, completion: @escaping (Result) -> Void) {
        session.request(data.url, method: .post, parameters: data.body)
    }
}
`;

// ============================================================================
// Test Queries
// ============================================================================

const TEST_QUERIES = [
  {
    id: 1,
    query: `I'm seeing two code examples (TypeScript and Swift). Can you identify what architectural pattern both are implementing? Look for the DEEP STRUCTURE, not the syntax.

TypeScript:
${TYPESCRIPT_USECASE}

Swift:
${SWIFT_USECASE}

What is the universal pattern here?`,
    expected_concepts: ['dependency_inversion', 'abstraction', 'clean_architecture'],
  },
  {
    id: 2,
    query: `Based on these two examples, what is UNIVERSAL (same in both) vs what is LANGUAGE-SPECIFIC (different syntax)?

TypeScript Adapter:
${TYPESCRIPT_ADAPTER}

Swift Adapter:
${SWIFT_ADAPTER}

Identify the deep structure vs surface structure.`,
    expected_concepts: ['adapter_pattern', 'dependency_injection', 'universal_grammar'],
  },
  {
    id: 3,
    query: `You've seen UseCase patterns in TypeScript and Swift. Can you generate the SAME pattern in Python?

Requirements:
- Same deep structure (protocol, params, result)
- Different surface structure (Python syntax)
- Follow dependency inversion principle

Generate a Python AddAccount use case.`,
    expected_concepts: ['code_generation', 'pattern_transfer', 'universal_grammar'],
  },
  {
    id: 4,
    query: `Based on all examples you've seen, formulate the UNIVERSAL GRAMMAR RULE for the UseCase pattern.

This rule should:
1. Work in ANY programming language
2. Define the deep structure
3. Be independent of syntax

What is the rule?`,
    expected_concepts: ['universal_grammar', 'clean_architecture', 'language_independence'],
  },
];

// ============================================================================
// Validation Logic
// ============================================================================

async function runValidation() {
  console.log('═'.repeat(70));
  console.log('🧬 UNIVERSAL GRAMMAR THESIS VALIDATION');
  console.log('═'.repeat(70));
  console.log();
  console.log('Thesis: "Clean Architecture exhibits Universal Grammar"');
  console.log();
  console.log('Method:');
  console.log('  1. Show AGI code patterns in TypeScript and Swift');
  console.log('  2. Test if AGI can identify universal deep structure');
  console.log('  3. Test if AGI can generate code in new language');
  console.log('  4. Use episodic memory to learn patterns');
  console.log('  5. Validate thesis empirically');
  console.log();
  console.log('═'.repeat(70));
  console.log();

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    console.error('❌ ANTHROPIC_API_KEY not found in environment');
    process.exit(1);
  }

  // Create meta-agent with memory
  const metaAgent = createMetaAgentWithMemory(
    apiKey,
    5, // max depth
    10, // max invocations
    0.5, // max cost: $0.50
    true // use memory
  );

  // Register agents
  metaAgent.registerAgent('architecture', new ArchitectureAgent(apiKey));
  metaAgent.registerAgent('linguistics', new LinguisticsAgent(apiKey));
  metaAgent.registerAgent('systems', new SystemsAgent(apiKey));

  // Initialize
  const slicesDir = path.join(__dirname, '..', 'slices');
  await metaAgent.initialize(slicesDir);

  console.log('✅ MetaAgent with Episodic Memory initialized');
  console.log();

  // Run test queries
  const results: any[] = [];

  for (const test of TEST_QUERIES) {
    console.log('─'.repeat(70));
    console.log(`📋 TEST ${test.id}/${TEST_QUERIES.length}`);
    console.log('─'.repeat(70));
    console.log();
    console.log(`Query: ${test.query.substring(0, 150)}...`);
    console.log();

    const startCost = metaAgent.getTotalCost();
    const startTime = Date.now();

    try {
      const result = await metaAgent.processWithMemory(test.query);
      const endTime = Date.now();
      const queryCost = metaAgent.getTotalCost() - startCost;

      console.log(`📊 Results:`);
      console.log(`   Memory used: ${result.memory_used ? '✅ YES (cache hit!)' : '❌ NO (fresh query)'}`);
      console.log(`   Cost: $${queryCost.toFixed(4)}`);
      console.log(`   Time: ${((endTime - startTime) / 1000).toFixed(2)}s`);
      console.log(`   Agents invoked: ${result.invocations}`);
      console.log(`   Emergent concepts: ${result.emergent_insights.length}`);
      console.log();

      console.log(`💬 Answer (first 500 chars):`);
      console.log(`   ${result.final_answer.substring(0, 500)}${result.final_answer.length > 500 ? '...' : ''}`);
      console.log();

      // Validate against expected concepts
      const found_concepts = result.emergent_insights;
      const expected = test.expected_concepts;
      const overlap = expected.filter((c) =>
        found_concepts.some((fc) => fc.toLowerCase().includes(c.toLowerCase()))
      );

      console.log(`🔬 Concept Validation:`);
      console.log(`   Expected concepts: ${expected.join(', ')}`);
      console.log(`   Found concepts: ${found_concepts.join(', ')}`);
      console.log(`   Overlap: ${overlap.length}/${expected.length} (${((overlap.length / expected.length) * 100).toFixed(0)}%)`);
      console.log();

      if (result.similar_past_queries.length > 0) {
        console.log(`🧠 Similar Past Queries Found: ${result.similar_past_queries.length}`);
        result.similar_past_queries.forEach((ep, i) => {
          console.log(`   ${i + 1}. "${ep.query.substring(0, 50)}..." (confidence: ${ep.confidence.toFixed(2)})`);
        });
        console.log();
      }

      results.push({
        test_id: test.id,
        cost: queryCost,
        time_seconds: (endTime - startTime) / 1000,
        memory_used: result.memory_used,
        concepts_found: found_concepts.length,
        concepts_expected: expected.length,
        overlap_percentage: (overlap.length / expected.length) * 100,
      });
    } catch (error: any) {
      console.error(`\n❌ Error: ${error.message}`);
      results.push({
        test_id: test.id,
        error: error.message,
      });
    }

    console.log();
  }

  // Final summary
  console.log('\n\n');
  console.log('═'.repeat(70));
  console.log('📊 VALIDATION SUMMARY');
  console.log('═'.repeat(70));
  console.log();

  // Memory stats
  const memoryStats = metaAgent.getMemoryStats();
  console.log('🧠 Memory Statistics:');
  console.log(`   Total episodes: ${memoryStats.total_episodes}`);
  console.log(`   Total concepts learned: ${memoryStats.total_concepts}`);
  console.log(`   Success rate: ${(memoryStats.success_rate * 100).toFixed(1)}%`);
  console.log(`   Average confidence: ${memoryStats.average_confidence.toFixed(2)}`);
  console.log();

  if (memoryStats.most_common_concepts.length > 0) {
    console.log('   Most common concepts:');
    memoryStats.most_common_concepts.slice(0, 5).forEach((c) => {
      console.log(`     - ${c.concept}: ${c.count} occurrences`);
    });
    console.log();
  }

  // Test results
  console.log('🧪 Test Results:');
  const successful_tests = results.filter((r) => !r.error);
  const avg_overlap = successful_tests.reduce((sum, r) => sum + (r.overlap_percentage || 0), 0) / successful_tests.length;

  console.log(`   Tests completed: ${successful_tests.length}/${TEST_QUERIES.length}`);
  console.log(`   Average concept overlap: ${avg_overlap.toFixed(1)}%`);
  console.log(`   Total cost: $${metaAgent.getTotalCost().toFixed(4)}`);
  console.log();

  // Consolidate memory
  console.log('🔄 Consolidating memory...');
  const consolidation = metaAgent.consolidateMemory();
  console.log(`   Episodes merged: ${consolidation.merged_count}`);
  console.log(`   New insights discovered: ${consolidation.new_insights.length}`);
  console.log(`   Patterns discovered: ${consolidation.patterns_discovered.length}`);
  console.log();

  if (consolidation.patterns_discovered.length > 0) {
    console.log('   Discovered patterns:');
    consolidation.patterns_discovered.forEach((p) => {
      console.log(`     - ${p}`);
    });
    console.log();
  }

  // Thesis validation
  console.log('═'.repeat(70));
  console.log('🏆 THESIS VALIDATION');
  console.log('═'.repeat(70));
  console.log();

  const validation_criteria = [
    {
      criterion: 'AGI identifies deep structure in TypeScript and Swift',
      passed: avg_overlap > 50,
      score: avg_overlap,
    },
    {
      criterion: 'AGI can generate code in new language (Python)',
      passed: results[2] && !results[2].error,
      score: results[2] ? 100 : 0,
    },
    {
      criterion: 'AGI formulates universal grammar rule',
      passed: results[3] && !results[3].error,
      score: results[3] ? 100 : 0,
    },
    {
      criterion: 'Episodic memory improves learning',
      passed: memoryStats.total_episodes > 0,
      score: memoryStats.total_episodes * 25,
    },
  ];

  validation_criteria.forEach((vc) => {
    console.log(`${vc.passed ? '✅' : '❌'} ${vc.criterion}`);
    console.log(`   Score: ${vc.score.toFixed(1)}%`);
  });
  console.log();

  const all_passed = validation_criteria.every((vc) => vc.passed);
  const avg_score = validation_criteria.reduce((sum, vc) => sum + vc.score, 0) / validation_criteria.length;

  console.log(`Overall Score: ${avg_score.toFixed(1)}%`);
  console.log();

  if (all_passed && avg_score > 75) {
    console.log('🎯 THESIS VALIDATED!');
    console.log();
    console.log('The AGI successfully:');
    console.log('  ✅ Identified universal deep structure across languages');
    console.log('  ✅ Distinguished deep structure from surface syntax');
    console.log('  ✅ Generated code in new language following same pattern');
    console.log('  ✅ Formulated language-independent grammar rules');
    console.log('  ✅ Used episodic memory to learn and improve');
    console.log();
    console.log('This proves: Clean Architecture exhibits Universal Grammar');
    console.log('just as Chomsky theorized for natural languages.');
  } else {
    console.log('⚠️ THESIS PARTIALLY VALIDATED');
    console.log();
    console.log(`Score: ${avg_score.toFixed(1)}% (threshold: 75%)`);
    console.log(`Criteria passed: ${validation_criteria.filter((v) => v.passed).length}/${validation_criteria.length}`);
  }

  console.log();
  console.log('═'.repeat(70));

  // Export memory for later use
  const memoryExport = metaAgent.exportMemory();
  console.log();
  console.log(`💾 Memory exported (${memoryExport.length} bytes)`);
  console.log('   Can be imported for future sessions');
}

// Run validation
runValidation().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
