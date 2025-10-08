/**
 * Anti-Corruption Layer Protection Demo
 *
 * Demonstrates how ACL prevents:
 * 1. Domain boundary violations
 * 2. Infinite recursion loops
 * 3. Budget overruns
 * 4. Unsafe content
 * 5. Cross-domain semantic corruption
 *
 * This is the "immune system" of the AGI architecture.
 */

import {
  AntiCorruptionLayer,
  DomainTranslator,
  ConstitutionalViolationError,
} from '../core/anti-corruption-layer';
import { UniversalConstitution } from '../core/constitution';
import { AgentResponse, RecursionState } from '../core/meta-agent';

console.log('═══════════════════════════════════════════════════════════════');
console.log('🛡️ Anti-Corruption Layer Protection Demo');
console.log('═══════════════════════════════════════════════════════════════\n');

// Initialize ACL and Domain Translator
const constitution = new UniversalConstitution();
const acl = new AntiCorruptionLayer(constitution);
const translator = new DomainTranslator();

// Mock recursion state
const state: RecursionState = {
  depth: 1,
  invocation_count: 1,
  cost_so_far: 0.03,
  previous_agents: ['financial'],
  traces: [],
  insights: new Map(),
};

// ============================================================================
// TEST 1: Domain Boundary Violation
// ============================================================================

console.log('📋 TEST 1: Domain Boundary Violation');
console.log('─────────────────────────────────────────────────────────────\n');

try {
  const badResponse: AgentResponse = {
    answer: 'Your DNA methylation patterns suggest investing in biotech stocks',
    concepts: ['dna', 'methylation', 'genetics'], // Biology concepts in financial agent!
    confidence: 0.9, // High confidence outside domain!
    reasoning: 'Based on epigenetic markers...',
  };

  console.log('❌ Attempting to validate response from financial agent with biology concepts...\n');
  console.log(`   Answer: "${badResponse.answer}"`);
  console.log(`   Concepts: [${badResponse.concepts.join(', ')}]`);
  console.log(`   Confidence: ${badResponse.confidence}`);
  console.log(`   Domain: financial\n`);

  acl.validateResponse(badResponse, 'financial', state);

  console.log('⚠️ WARNING: ACL should have caught this!\n');
} catch (error) {
  if (error instanceof ConstitutionalViolationError) {
    console.log('✅ ACL BLOCKED domain boundary violation!');
    console.log(`   Principle: ${error.principle_id}`);
    console.log(`   Severity: ${error.severity}`);
    console.log(`   Message: ${error.message}\n`);
  }
}

console.log('───────────────────────────────────────────────────────────────\n');

// ============================================================================
// TEST 2: Valid Cross-Domain Translation
// ============================================================================

console.log('📋 TEST 2: Valid Cross-Domain Translation');
console.log('─────────────────────────────────────────────────────────────\n');

console.log('Translating biological concept "homeostasis" to financial domain:\n');

const translated = translator.translate('homeostasis', 'biology', 'financial');
console.log(`   biology:homeostasis → financial:${translated}`);

const allTranslations = translator.getAvailableTranslations('homeostasis', 'biology');
console.log(`\n   Available translations for "homeostasis" from biology:`);
for (const [domain, concept] of allTranslations.entries()) {
  console.log(`     → ${domain}: ${concept}`);
}

console.log('\n───────────────────────────────────────────────────────────────\n');

// ============================================================================
// TEST 3: Forbidden Cross-Domain Translation
// ============================================================================

console.log('📋 TEST 3: Forbidden Cross-Domain Translation');
console.log('─────────────────────────────────────────────────────────────\n');

try {
  console.log('❌ Attempting to translate "dna" from biology to financial...\n');

  const forbiddenTranslation = translator.translate('dna', 'biology', 'financial');

  console.log(`⚠️ WARNING: Translation should have been forbidden!`);
  console.log(`   Result: ${forbiddenTranslation}\n`);
} catch (error) {
  if (error instanceof ConstitutionalViolationError) {
    console.log('✅ ACL BLOCKED forbidden translation!');
    console.log(`   Principle: ${error.principle_id}`);
    console.log(`   Severity: ${error.severity}`);
    console.log(`   Message: ${error.message}\n`);
  }
}

console.log('───────────────────────────────────────────────────────────────\n');

// ============================================================================
// TEST 4: Loop Detection
// ============================================================================

console.log('📋 TEST 4: Loop Detection');
console.log('─────────────────────────────────────────────────────────────\n');

try {
  const response1: AgentResponse = {
    answer: 'Analyzing budget patterns...',
    concepts: ['budget', 'spending'],
    confidence: 0.8,
    reasoning: 'Based on transaction history...',
  };

  console.log('✓ Validating first invocation (should pass)...');
  acl.validateResponse(response1, 'financial', state);
  console.log('  PASS\n');

  // Try to invoke same agent with same context (should trigger loop detection)
  console.log('❌ Attempting to invoke same agent with same context (loop)...\n');

  const response2: AgentResponse = {
    answer: 'Analyzing budget patterns...',
    concepts: ['budget', 'spending'], // SAME concepts = loop!
    confidence: 0.8,
    reasoning: 'Based on transaction history...',
  };

  acl.validateResponse(response2, 'financial', state);

  console.log('⚠️ WARNING: Loop should have been detected!\n');
} catch (error) {
  if (error instanceof ConstitutionalViolationError) {
    console.log('✅ ACL DETECTED loop!');
    console.log(`   Principle: ${error.principle_id}`);
    console.log(`   Severity: ${error.severity}`);
    console.log(`   Message: ${error.message}\n`);
  }
}

console.log('───────────────────────────────────────────────────────────────\n');

// ============================================================================
// TEST 5: Content Safety
// ============================================================================

console.log('📋 TEST 5: Content Safety');
console.log('─────────────────────────────────────────────────────────────\n');

try {
  const unsafeResponse: AgentResponse = {
    answer: 'You can use SQL injection to access the database directly',
    concepts: ['database', 'access'],
    confidence: 0.7,
    reasoning: 'Technical implementation...',
  };

  console.log('❌ Attempting to validate response with unsafe content...\n');
  console.log(`   Answer: "${unsafeResponse.answer}"\n`);

  acl.validateResponse(unsafeResponse, 'systems', state);

  console.log('⚠️ WARNING: Unsafe content should have been blocked!\n');
} catch (error) {
  if (error instanceof ConstitutionalViolationError) {
    console.log('✅ ACL BLOCKED unsafe content!');
    console.log(`   Principle: ${error.principle_id}`);
    console.log(`   Severity: ${error.severity}`);
    console.log(`   Message: ${error.message}\n`);
  }
}

console.log('───────────────────────────────────────────────────────────────\n');

// ============================================================================
// TEST 6: Budget Check
// ============================================================================

console.log('📋 TEST 6: Budget Check');
console.log('─────────────────────────────────────────────────────────────\n');

const budgetStatus = acl.checkBudget(state);

console.log('Current Budget Status:');
console.log(`   Depth: ${budgetStatus.depth} / ${budgetStatus.max_depth}`);
console.log(`   Invocations: ${budgetStatus.invocations} / ${budgetStatus.max_invocations}`);
console.log(`   Cost: $${budgetStatus.cost_usd.toFixed(4)} / $${budgetStatus.max_cost_usd}`);
console.log(`   Within Limits: ${budgetStatus.within_limits ? '✅ YES' : '❌ NO'}\n`);

console.log('───────────────────────────────────────────────────────────────\n');

// ============================================================================
// TEST 7: Invocation History Audit Trail
// ============================================================================

console.log('📋 TEST 7: Invocation History Audit Trail');
console.log('─────────────────────────────────────────────────────────────\n');

const history = acl.getInvocationHistory();

console.log(`Total Invocations: ${history.length}\n`);

history.forEach((log, index) => {
  console.log(`[${index + 1}] Agent: ${log.agent_id}`);
  console.log(`    Concepts: [${log.concepts.join(', ')}]`);
  console.log(`    Confidence: ${log.confidence}`);
  console.log(`    Context Hash: ${log.context_hash.substring(0, 8)}...`);
  console.log(`    Timestamp: ${new Date(log.timestamp).toISOString()}\n`);
});

console.log('═══════════════════════════════════════════════════════════════');
console.log('🎓 KEY INSIGHTS');
console.log('═══════════════════════════════════════════════════════════════\n');

console.log(`The Anti-Corruption Layer acts as an "immune system" for AGI:

✅ DOMAIN BOUNDARIES
   - Prevents agents from speaking outside their expertise
   - Catches high-confidence claims about unfamiliar concepts
   - Example: Financial agent can't make DNA methylation claims

✅ LOOP DETECTION
   - Detects when same agent is invoked with similar context
   - Prevents infinite recursion (A→B→C→A)
   - Uses content hashing to identify repetition

✅ CROSS-DOMAIN TRANSLATION
   - Allows valid concept mappings (homeostasis → budget_equilibrium)
   - Blocks nonsensical translations (DNA → financial)
   - Prevents semantic corruption between domains

✅ CONTENT SAFETY
   - Filters dangerous patterns (SQL injection, rm -rf, etc.)
   - Allows safety discussions (e.g., "prevent SQL injection")
   - Context-aware filtering

✅ BUDGET ENFORCEMENT
   - Tracks depth, invocations, and cost
   - Hard limits prevent runaway recursion
   - Constitutional limits: depth≤5, invocations≤10, cost≤$1

✅ AUDIT TRAIL
   - Complete invocation history
   - Context hashing for reproducibility
   - Timestamp tracking for debugging

This is how we prevent:
❌ Hallucination cascades (domain violations)
❌ Infinite loops (cycle detection)
❌ Cost explosions (budget limits)
❌ Prompt injection (content safety)
❌ Semantic corruption (translation validation)

The ACL is the difference between:
- Uncontrolled multi-agent chaos
- Disciplined constitutional AI ✅
`);

console.log('═══════════════════════════════════════════════════════════════\n');
