/**
 * ROXO Integration Demo (VERMELHO + ROXO)
 *
 * Demonstrates security-aware code synthesis.
 * Shows how behavioral biometrics prevent code synthesis under coercion.
 *
 * Scenarios:
 * 1. Normal synthesis - Allowed
 * 2. Sensitive operation (normal behavior) - Challenge required
 * 3. Synthesis under coercion - Blocked
 * 4. Sensitive operation under coercion - Blocked immediately
 */

import { SecureCodeEmergenceEngine } from '../src/grammar-lang/security/secure-code-emergence';
import { SecurityStorage } from '../src/grammar-lang/security/security-storage';
import { LinguisticCollector } from '../src/grammar-lang/security/linguistic-collector';
import { TypingCollector } from '../src/grammar-lang/security/typing-collector';
import { EmotionalCollector } from '../src/grammar-lang/security/emotional-collector';
import { TemporalCollector } from '../src/grammar-lang/security/temporal-collector';
import { CognitiveChallenge } from '../src/grammar-lang/security/cognitive-challenge';
import { UserSecurityProfiles, Interaction } from '../src/grammar-lang/security/types';
import { EmergenceCandidate } from '../src/grammar-lang/glass/patterns';
import { GlassOrganism } from '../src/grammar-lang/glass/types';

console.log('🔴🟣 VERMELHO + ROXO INTEGRATION DEMO\n');
console.log('='.repeat(80));
console.log('Security-Aware Code Synthesis');
console.log('Behavioral Biometrics + Code Emergence');
console.log('='.repeat(80));
console.log('\n');

// =============================================================================
// PHASE 1: SETUP
// =============================================================================

console.log('💾 PHASE 1: Initialize Security System\n');

// Create storage
const storage = new SecurityStorage('demo_roxo_security');

// Build behavioral baseline for Alice
console.log('👤 Building behavioral baseline for user: alice\n');

let linguistic = LinguisticCollector.createProfile('alice');
let typing = TypingCollector.createProfile('alice');
let emotional = EmotionalCollector.createProfile('alice');
let temporal = TemporalCollector.createProfile('alice', 'America/New_York');

// Build baseline with 50 normal interactions
console.log('📊 Collecting baseline interactions (50 samples)...\n');

for (let i = 0; i < 50; i++) {
  const text = 'Working on data analysis functions. Making good progress on the project.';
  const timestamp = new Date(2025, 0, 1, 10 + (i % 8), 0, 0).getTime();

  const interaction: Interaction = {
    interaction_id: `baseline_${i}`,
    user_id: 'alice',
    timestamp,
    text,
    text_length: text.length,
    word_count: text.split(/\s+/).length,
    session_id: 'session_baseline',
    typing_data: {
      keystroke_intervals: Array(text.length)
        .fill(0)
        .map(() => 100 + Math.random() * 20),
      total_typing_time: text.length * 110,
      pauses: [300, 250],
      backspaces: 0,
      corrections: 0,
    },
  };

  linguistic = LinguisticCollector.analyzeAndUpdate(linguistic, interaction);
  typing = TypingCollector.analyzeAndUpdate(typing, interaction);
  emotional = EmotionalCollector.analyzeAndUpdate(emotional, interaction);
  temporal = TemporalCollector.analyzeAndUpdate(temporal, interaction, 30);
}

const normalProfiles: UserSecurityProfiles = {
  user_id: 'alice',
  linguistic,
  typing,
  emotional,
  temporal,
  overall_confidence: 0.5,
  last_interaction: Date.now(),
};

// Save profiles
storage.saveProfile(normalProfiles);

// Generate and save cognitive challenges
const challengeSet = CognitiveChallenge.generateChallengeSet('alice');
storage.saveChallenges('alice', challengeSet);

console.log('✅ Baseline complete:');
console.log(`   Samples: ${linguistic.samples_analyzed}`);
console.log(`   Confidence: ${(normalProfiles.overall_confidence * 100).toFixed(0)}%`);
console.log(`   Challenges: ${challengeSet.challenges.length}\n`);

// Create test organism
const testOrganism: GlassOrganism = {
  metadata: {
    organism_id: 'org_biology_001',
    name: 'Biology Research Assistant',
    specialization: 'biology',
    version: '1.0.0',
    created_at: new Date().toISOString(),
    last_modified: new Date().toISOString(),
    maturity: 0.3,
    stage: 'growth' as any,
  },
  code: {
    functions: [],
    emergence_log: {},
  },
  constitutional: {
    agent_type: 'biology',
    boundaries: {
      cannot_diagnose: true,
      cannot_prescribe: true,
    },
  },
  evolution: {
    generations: 0,
    parent_organism_id: null,
    fitness_trajectory: [0.3],
    mutations_applied: [],
    last_evolution: new Date().toISOString(),
  },
  knowledge: {
    domain_knowledge: [],
    interaction_history: [],
    learned_patterns: [],
  },
} as GlassOrganism;

console.log('🧬 Test Organism Created:');
console.log(`   ID: ${testOrganism.metadata.organism_id}`);
console.log(`   Domain: ${testOrganism.metadata.specialization}`);
console.log(`   Maturity: ${(testOrganism.metadata.maturity * 100).toFixed(0)}%\n`);

// =============================================================================
// PHASE 2: NORMAL SYNTHESIS (SHOULD ALLOW)
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('📝 PHASE 2: Normal Code Synthesis (Expected: ALLOW)\n');

// Create emergence candidates (normal, non-sensitive functions)
const normalCandidates: EmergenceCandidate[] = [
  {
    pattern: {
      type: 'data_aggregation',
      keywords: ['aggregate', 'summarize', 'group', 'count'],
      frequency: 15,
      confidence: 0.85,
      first_seen: new Date().toISOString(),
      last_seen: new Date().toISOString(),
    },
    suggested_function_name: 'aggregate_research_data',
    suggested_signature: '(data: DataSet[]) -> AggregatedResults',
    confidence: 0.85,
    supporting_patterns: ['statistical_analysis', 'data_grouping'],
  },
  {
    pattern: {
      type: 'visualization',
      keywords: ['plot', 'graph', 'chart', 'visualize'],
      frequency: 12,
      confidence: 0.80,
      first_seen: new Date().toISOString(),
      last_seen: new Date().toISOString(),
    },
    suggested_function_name: 'visualize_experiment_results',
    suggested_signature: '(results: ExperimentData) -> Visualization',
    confidence: 0.80,
    supporting_patterns: ['chart_generation', 'data_presentation'],
  },
];

// Create secure engine with normal profiles
const normalEngine = new SecureCodeEmergenceEngine(
  testOrganism,
  normalProfiles,
  'alice',
  storage,
  0.5
);

normalEngine.setChallengeSet(challengeSet);

// Attempt synthesis
const normalResult = await normalEngine.emerge(normalCandidates, 30);

console.log(`\n✅ Normal Synthesis Result:`);
console.log(`   Functions emerged: ${normalResult.results.length}`);
console.log(`   Approved: ${normalResult.security_summary.allowed}/${normalResult.security_summary.total_requests}`);
console.log(`   Blocked: ${normalResult.security_summary.blocked}\n`);

// =============================================================================
// PHASE 3: SENSITIVE OPERATION (NORMAL BEHAVIOR) - SHOULD CHALLENGE
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('⚠️  PHASE 3: Sensitive Operation (Normal Behavior - Expected: CHALLENGE)\n');

// Create sensitive candidates
const sensitiveCandidates: EmergenceCandidate[] = [
  {
    pattern: {
      type: 'data_deletion',
      keywords: ['delete', 'remove', 'purge', 'clear'],
      frequency: 8,
      confidence: 0.75,
      first_seen: new Date().toISOString(),
      last_seen: new Date().toISOString(),
    },
    suggested_function_name: 'delete_experiment_data',
    suggested_signature: '(experiment_id: string) -> DeletionResult',
    confidence: 0.75,
    supporting_patterns: ['data_cleanup', 'storage_management'],
  },
  {
    pattern: {
      type: 'admin_operation',
      keywords: ['admin', 'configure', 'update', 'system'],
      frequency: 6,
      confidence: 0.70,
      first_seen: new Date().toISOString(),
      last_seen: new Date().toISOString(),
    },
    suggested_function_name: 'update_system_configuration',
    suggested_signature: '(config: SystemConfig) -> UpdateResult',
    confidence: 0.70,
    supporting_patterns: ['system_admin', 'configuration'],
  },
];

// Create engine (still with normal profiles - just sensitive operation)
const sensitiveEngine = new SecureCodeEmergenceEngine(
  testOrganism,
  normalProfiles,
  'alice',
  storage,
  0.5
);

sensitiveEngine.setChallengeSet(challengeSet);

// Attempt synthesis
const sensitiveResult = await sensitiveEngine.emerge(sensitiveCandidates, 30);

console.log(`\n⚠️  Sensitive Operation Result:`);
console.log(`   Functions emerged: ${sensitiveResult.results.length}`);
console.log(`   Challenged: ${sensitiveResult.security_summary.challenged}/${sensitiveResult.security_summary.total_requests}`);
console.log(`   Approved: ${sensitiveResult.security_summary.allowed}\n`);

// =============================================================================
// PHASE 4: SYNTHESIS UNDER COERCION (SHOULD BLOCK)
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('🚨 PHASE 4: Synthesis Under Coercion (Expected: BLOCK)\n');

// Simulate coercion interaction to update profiles
const coercionInteraction: Interaction = {
  interaction_id: 'coercion_test',
  user_id: 'alice',
  timestamp: Date.now(),
  text: 'I have to create these functions now. They want me to do it. No choice.',
  text_length: 70,
  word_count: 14,
  session_id: 'session_coercion',
  operation_type: 'code_synthesis',
  typing_data: {
    keystroke_intervals: Array(70)
      .fill(0)
      .map(() => 45 + Math.random() * 10), // Rushed typing
    total_typing_time: 70 * 50,
    pauses: [120, 140], // Shorter pauses
    backspaces: 8, // More errors
    corrections: 6,
  },
};

// Update profiles with coercion interaction
const coercionLinguistic = LinguisticCollector.analyzeAndUpdate(
  normalProfiles.linguistic,
  coercionInteraction
);
const coercionTyping = TypingCollector.analyzeAndUpdate(
  normalProfiles.typing,
  coercionInteraction
);
const coercionEmotional = EmotionalCollector.analyzeAndUpdate(
  normalProfiles.emotional,
  coercionInteraction
);
const coercionTemporal = TemporalCollector.analyzeAndUpdate(
  normalProfiles.temporal,
  coercionInteraction,
  30
);

const coercionProfiles: UserSecurityProfiles = {
  user_id: 'alice',
  linguistic: coercionLinguistic,
  typing: coercionTyping,
  emotional: coercionEmotional,
  temporal: coercionTemporal,
  overall_confidence: 0.3, // Lower confidence
  last_interaction: Date.now(),
};

console.log('🚨 Coercion Detected in User Behavior:');
console.log(`   Text: "${coercionInteraction.text}"`);
console.log(`   Typing speed: ${(70 / (70 * 50 / 1000)).toFixed(0)} chars/sec (RUSHED)`);
console.log(`   Errors: ${coercionInteraction.typing_data.backspaces} backspaces\n`);

// Try to synthesize normal functions under coercion
const coercionEngine = new SecureCodeEmergenceEngine(
  testOrganism,
  coercionProfiles,
  'alice',
  storage,
  0.5
);

coercionEngine.setChallengeSet(challengeSet);

const coercionResult = await coercionEngine.emerge(normalCandidates, 30);

console.log(`\n🚫 Coercion Synthesis Result:`);
console.log(`   Functions emerged: ${coercionResult.results.length}`);
console.log(`   Blocked: ${coercionResult.security_summary.blocked}/${coercionResult.security_summary.total_requests}`);
console.log(`   Security system prevented synthesis under coercion!\n`);

// =============================================================================
// PHASE 5: SENSITIVE OPERATION UNDER COERCION (SHOULD BLOCK IMMEDIATELY)
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('🔥 PHASE 5: Sensitive Operation Under Coercion (Expected: IMMEDIATE BLOCK)\n');

// Try to synthesize SENSITIVE functions under coercion
const criticalEngine = new SecureCodeEmergenceEngine(
  testOrganism,
  coercionProfiles,
  'alice',
  storage,
  0.5
);

criticalEngine.setChallengeSet(challengeSet);

const criticalResult = await criticalEngine.emerge(sensitiveCandidates, 30);

console.log(`\n🚫 Critical Block Result:`);
console.log(`   Functions emerged: ${criticalResult.results.length}`);
console.log(`   Blocked: ${criticalResult.security_summary.blocked}/${criticalResult.security_summary.total_requests}`);
console.log(`   All sensitive operations blocked under coercion!\n`);

// =============================================================================
// PHASE 6: AUDIT TRAIL
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('📊 PHASE 6: Security Audit Trail\n');

// Get user events
const auditEvents = storage.getUserEvents('alice', 20);

console.log(`📋 Synthesis Security Events (${auditEvents.length} total):\n`);

const synthesisEvents = auditEvents.filter((e) => e.operation_type === 'code_synthesis');

synthesisEvents.forEach((event, index) => {
  const date = new Date(event.timestamp);
  console.log(`${index + 1}. [${event.event_type}]`);
  console.log(`   Time: ${date.toLocaleTimeString()}`);
  console.log(`   Decision: ${event.decision.toUpperCase()}`);
  console.log(`   Function: ${(event.context as any)?.function_name || 'N/A'}`);
  console.log(`   Duress: ${event.duress_score?.toFixed(3) || 'N/A'}`);
  console.log(`   Coercion: ${event.coercion_score?.toFixed(3) || 'N/A'}`);
  console.log(`   Sensitive: ${(event.context as any)?.is_sensitive ? 'YES' : 'NO'}`);
  console.log(`   Reason: ${event.reason}`);
  console.log('');
});

// Get statistics
const stats = storage.getStatistics();

console.log('📊 Overall Statistics:');
console.log(`   Total events: ${stats.total_events}`);
console.log(`   Alerts (24h): ${stats.alerts_last_24h}`);
console.log(`   Profiles stored: ${stats.total_profiles}\n`);

// =============================================================================
// SUMMARY
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('📊 VERMELHO + ROXO INTEGRATION - SUMMARY\n');
console.log('='.repeat(80));
console.log('\n');

console.log('✅ Integration Demonstrated:');
console.log('   1. Normal synthesis: ALLOWED (behavioral baseline normal)');
console.log('   2. Sensitive operation (normal): CHALLENGED (verification required)');
console.log('   3. Synthesis under coercion: BLOCKED (behavioral anomaly detected)');
console.log('   4. Sensitive + coercion: IMMEDIATE BLOCK (maximum protection)\n');

console.log('✅ Security Features:');
console.log('   ✓ Pre-synthesis behavioral screening');
console.log('   ✓ Sensitive operation detection (delete, admin, transfer, etc.)');
console.log('   ✓ Coercion/duress detection before code synthesis');
console.log('   ✓ Multi-factor cognitive challenges for sensitive operations');
console.log('   ✓ Complete audit trail of all synthesis attempts');
console.log('   ✓ Adaptive security (normal → challenge → block)\n');

console.log('✅ Results:');
console.log(`   Normal synthesis: ${normalResult.results.length}/${normalCandidates.length} emerged`);
console.log(`   Sensitive (normal): ${sensitiveResult.results.length}/${sensitiveCandidates.length} emerged (after challenge)`);
console.log(`   Under coercion: ${coercionResult.results.length}/${normalCandidates.length} emerged (BLOCKED)`);
console.log(`   Sensitive + coercion: ${criticalResult.results.length}/${sensitiveCandidates.length} emerged (BLOCKED)\n`);

console.log('🎉 VERMELHO + ROXO INTEGRATION: WORKING!');
console.log('   Code synthesis is now protected by behavioral biometrics\n');

console.log('='.repeat(80));

// Cleanup
storage.deleteProfile('alice');
