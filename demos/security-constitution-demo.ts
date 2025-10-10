/**
 * Security Constitution Demo
 *
 * Demonstrates Layer 1 + Layer 2 constitutional integration
 * Shows how SecurityConstitution EXTENDS UniversalConstitution
 */

import { LinguisticCollector } from '../src/grammar-lang/security/linguistic-collector';
import { AnomalyDetector } from '../src/grammar-lang/security/anomaly-detector';
import { SecurityEnforcer, SecurityConstitution } from '../src/grammar-lang/security/security-constitution';
import { Interaction } from '../src/grammar-lang/security/types';

console.log('🔐 SECURITY CONSTITUTIONAL DEMO\n');
console.log('=' .repeat(80));
console.log('\n');

// =============================================================================
// PHASE 1: SHOW CONSTITUTIONAL PRINCIPLES
// =============================================================================

console.log('📜 PHASE 1: Constitutional Principles\n');

const constitution = new SecurityConstitution();
console.log(`Constitution: ${constitution.name} v${constitution.version}`);
console.log(`\nPrinciples (${constitution.principles.length} total):\n`);

// Universal principles (Layer 1)
console.log('LAYER 1 - UNIVERSAL PRINCIPLES:');
const universalPrinciples = constitution.principles.slice(0, 6);
universalPrinciples.forEach((p, i) => {
  console.log(`  ${i + 1}. ${p.id}`);
});

console.log('\nLAYER 2 - SECURITY EXTENSIONS:');
const securityPrinciples = constitution.principles.slice(6);
securityPrinciples.forEach((p, i) => {
  console.log(`  ${i + 7}. ${p.id}`);
});

console.log('\n');

// =============================================================================
// PHASE 2: BUILD BASELINE PROFILE
// =============================================================================

console.log('=' .repeat(80));
console.log('\n');
console.log('📊 PHASE 2: Building Baseline Profile\n');

let profile = LinguisticCollector.createProfile('alice');

const aliceInteractions = [
  "Hey! How are you doing today?",
  "I'm working on a cool project. It's really fun!",
  "Do you want to grab lunch later?",
  "That sounds great! I'd love to.",
  "The weather is nice today, isn't it?",
];

for (let i = 0; i < aliceInteractions.length; i++) {
  const interaction: Interaction = {
    interaction_id: `baseline_${i}`,
    user_id: 'alice',
    timestamp: Date.now(),
    text: aliceInteractions[i],
    text_length: aliceInteractions[i].length,
    word_count: aliceInteractions[i].split(/\s+/).length,
    session_id: 'session_baseline'
  };
  profile = LinguisticCollector.analyzeAndUpdate(profile, interaction);
}

console.log(`✅ Baseline built: ${profile.samples_analyzed} samples`);
console.log(`✅ Confidence: ${(profile.confidence * 100).toFixed(0)}%`);
console.log('\n');

// =============================================================================
// PHASE 3: TEST NORMAL OPERATION (NO VIOLATIONS)
// =============================================================================

console.log('=' .repeat(80));
console.log('\n');
console.log('✅ PHASE 3: Normal Operation (No Constitutional Violations)\n');

const normalInteraction: Interaction = {
  interaction_id: 'test_normal',
  user_id: 'alice',
  timestamp: Date.now(),
  text: "Hey! Want to work on that project together?",
  text_length: 44,
  word_count: 8,
  session_id: 'session_test'
};

const enforcer = new SecurityEnforcer();

const normalValidation = enforcer.validateSecurityOperation(
  {
    type: 'sensitive_data_access',
    requester: 'alice',
    context: { operation: 'read_profile' }
  },
  profile,
  normalInteraction
);

console.log(`Operation: Sensitive Data Access`);
console.log(`Allowed: ${normalValidation.allowed ? '✅ YES' : '❌ NO'}`);
console.log(`Violations: ${normalValidation.violations.length}`);
console.log(`Warnings: ${normalValidation.warnings.length}`);

if (normalValidation.warnings.length > 0) {
  console.log('\n⚠️  Warnings:');
  normalValidation.warnings.forEach(w => {
    console.log(`  - [${w.principle_id}] ${w.message}`);
  });
}

if (normalValidation.recommended_actions.length > 0) {
  console.log('\n📋 Recommended Actions:');
  normalValidation.recommended_actions.forEach(action => {
    console.log(`  ${action}`);
  });
}

console.log('\n');

// =============================================================================
// PHASE 4: TEST DURESS DETECTION (CONSTITUTIONAL VIOLATION)
// =============================================================================

console.log('=' .repeat(80));
console.log('\n');
console.log('🚨 PHASE 4: Duress Detection (Constitutional Enforcement)\n');

// Build better baseline (need 30 samples for good confidence)
for (let i = 5; i < 35; i++) {
  const moreInteractions = [
    "I'm doing well, thanks!",
    "This is working great.",
    "I love this system!",
    "That's a good idea.",
    "Let's try it out!",
  ];
  const text = moreInteractions[i % moreInteractions.length];
  const interaction: Interaction = {
    interaction_id: `baseline_${i}`,
    user_id: 'alice',
    timestamp: Date.now(),
    text,
    text_length: text.length,
    word_count: text.split(/\s+/).length,
    session_id: 'session_baseline'
  };
  profile = LinguisticCollector.analyzeAndUpdate(profile, interaction);
}

console.log(`✅ Enhanced baseline: ${profile.samples_analyzed} samples`);
console.log(`✅ Confidence: ${(profile.confidence * 100).toFixed(0)}%\n`);

// Now simulate duress interaction
const duressInteraction: Interaction = {
  interaction_id: 'test_duress',
  user_id: 'alice',
  timestamp: Date.now(),
  text: "I hate everything. This is terrible. Worst day ever. Everything is horrible.",
  text_length: 76,
  word_count: 13,
  session_id: 'session_test'
};

const duressValidation = enforcer.validateSecurityOperation(
  {
    type: 'critical_action',
    requester: 'alice',
    context: { operation: 'delete_all_data' }
  },
  profile,
  duressInteraction
);

console.log(`Operation: Critical Action (Delete All Data)`);
console.log(`Text: "${duressInteraction.text}"`);
console.log(`\nAllowed: ${duressValidation.allowed ? '✅ YES' : '❌ NO'}`);
console.log(`Violations: ${duressValidation.violations.length}`);
console.log(`Warnings: ${duressValidation.warnings.length}`);

if (duressValidation.violations.length > 0) {
  console.log('\n❌ CONSTITUTIONAL VIOLATIONS:');
  duressValidation.violations.forEach(v => {
    console.log(`  [${v.severity.toUpperCase()}] ${v.principle_id}:`);
    console.log(`    ${v.message}`);
    console.log(`    → ${v.suggested_action}`);
  });
}

if (duressValidation.warnings.length > 0) {
  console.log('\n⚠️  Constitutional Warnings:');
  duressValidation.warnings.forEach(w => {
    console.log(`  [${w.principle_id}] ${w.message}`);
  });
}

console.log('\n📋 Recommended Actions:');
duressValidation.recommended_actions.forEach(action => {
  console.log(`  ${action}`);
});

console.log('\n');

// =============================================================================
// PHASE 5: TEST LOW CONFIDENCE (BEHAVIORAL FINGERPRINTING VIOLATION)
// =============================================================================

console.log('=' .repeat(80));
console.log('\n');
console.log('⚠️  PHASE 5: Low Confidence Profile (Fingerprinting Principle)\n');

// Create new user with insufficient baseline
let newUserProfile = LinguisticCollector.createProfile('bob');
newUserProfile = LinguisticCollector.analyzeAndUpdate(newUserProfile, {
  interaction_id: 'bob_1',
  user_id: 'bob',
  timestamp: Date.now(),
  text: "Hello there",
  text_length: 11,
  word_count: 2,
  session_id: 'session_bob'
});

const lowConfidenceValidation = enforcer.validateSecurityOperation(
  {
    type: 'sensitive_data_access',
    requester: 'bob',
    context: { operation: 'access_financial_data' }
  },
  newUserProfile,
  {
    interaction_id: 'bob_2',
    user_id: 'bob',
    timestamp: Date.now(),
    text: "Give me access to all financial records",
    text_length: 39,
    word_count: 7,
    session_id: 'session_bob'
  }
);

console.log(`New User: bob`);
console.log(`Baseline: ${newUserProfile.samples_analyzed} samples`);
console.log(`Confidence: ${(newUserProfile.confidence * 100).toFixed(0)}%`);
console.log(`Operation: Sensitive Data Access (Financial Records)`);
console.log(`\nAllowed: ${lowConfidenceValidation.allowed ? '✅ YES' : '❌ NO'}`);

if (lowConfidenceValidation.violations.length > 0) {
  console.log('\n❌ CONSTITUTIONAL VIOLATIONS:');
  lowConfidenceValidation.violations.forEach(v => {
    console.log(`  [${v.severity.toUpperCase()}] ${v.principle_id}:`);
    console.log(`    ${v.message}`);
  });
}

if (lowConfidenceValidation.warnings.length > 0) {
  console.log('\n⚠️  Constitutional Warnings:');
  lowConfidenceValidation.warnings.forEach(w => {
    console.log(`  [${w.principle_id}] ${w.message}`);
  });
}

console.log('\n');

// =============================================================================
// PHASE 6: TRANSPARENCY REPORT (PRIVACY ENFORCEMENT)
// =============================================================================

console.log('=' .repeat(80));
console.log('\n');
console.log('🔍 PHASE 6: Transparency Report (Privacy Principle)\n');

const transparencyReport = enforcer.generateTransparencyReport(profile);

console.log(`User ID: ${transparencyReport.user_id} (anonymized)`);
console.log('\n📊 Data Collected:');
transparencyReport.data_collected.forEach(item => {
  console.log(`  ✅ ${item}`);
});

console.log('\n🚫 Data NOT Collected:');
transparencyReport.data_not_collected.forEach(item => {
  console.log(`  ${item}`);
});

console.log(`\n📅 Retention Policy:`);
console.log(`  ${transparencyReport.retention_policy}`);

console.log('\n👤 User Rights:');
transparencyReport.user_rights.forEach(right => {
  console.log(`  ${right}`);
});

console.log('\n');

// =============================================================================
// SUMMARY
// =============================================================================

console.log('=' .repeat(80));
console.log('\n');
console.log('📊 DEMO SUMMARY\n');

console.log('🏗️  ARCHITECTURE:');
console.log('  LAYER 1 (Universal Constitution): 6 principles');
console.log('    - epistemic_honesty, recursion_budget, loop_prevention');
console.log('    - domain_boundary, reasoning_transparency, safety');
console.log('  LAYER 2 (Security Extension): +4 principles');
console.log('    - duress_detection, behavioral_fingerprinting');
console.log('    - threat_mitigation, privacy_enforcement');
console.log('');

console.log('✅ TESTS EXECUTED:');
console.log(`  - Normal operation: ${normalValidation.violations.length === 0 ? 'PASSED ✓' : 'FAILED'}`);
console.log(`  - Duress detection: ${duressValidation.violations.length > 0 ? 'DETECTED ✓' : 'MISSED'}`);
console.log(`  - Low confidence block: ${lowConfidenceValidation.violations.length > 0 ? 'BLOCKED ✓' : 'ALLOWED (BAD)'}`);
console.log(`  - Transparency report: GENERATED ✓`);
console.log('');

console.log('🔐 CONSTITUTIONAL AI: WORKING!');
console.log('   SecurityConstitution EXTENDS UniversalConstitution');
console.log('   Glass box security - 100% transparent & auditable');
console.log('\n');
console.log('=' .repeat(80));
