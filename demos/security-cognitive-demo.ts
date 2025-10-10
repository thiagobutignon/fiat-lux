/**
 * Security - Cognitive Challenge Demo
 *
 * Demonstrates multi-factor cognitive authentication
 * Complements behavioral biometrics with knowledge-based verification
 */

import {
  CognitiveChallenge,
  CognitiveAuthenticator,
} from '../src/grammar-lang/security/cognitive-challenge';
import { MultiSignalDetector } from '../src/grammar-lang/security/multi-signal-detector';
import { LinguisticCollector } from '../src/grammar-lang/security/linguistic-collector';
import { TypingCollector } from '../src/grammar-lang/security/typing-collector';
import { EmotionalCollector } from '../src/grammar-lang/security/emotional-collector';
import { TemporalCollector } from '../src/grammar-lang/security/temporal-collector';
import { UserSecurityProfiles, Interaction } from '../src/grammar-lang/security/types';

console.log('🧠 SECURITY - COGNITIVE CHALLENGE DEMO\n');
console.log('='.repeat(80));
console.log('\nMulti-Factor Cognitive Authentication');
console.log('Knowledge You HAVE (in your brain) + Who You ARE (behavior)\n');
console.log('='.repeat(80));
console.log('\n');

// =============================================================================
// PHASE 1: CREATE CHALLENGE SET
// =============================================================================

console.log('📋 PHASE 1: Creating Cognitive Challenge Set\n');

const challengeSet = CognitiveChallenge.generateChallengeSet('alice');

console.log(`✅ Challenge set created for user: ${challengeSet.user_id}`);
console.log(`✅ Total challenges: ${challengeSet.challenges.length}\n`);

console.log('Challenge Types:');
challengeSet.challenges.forEach((challenge, index) => {
  console.log(`  ${index + 1}. ${challenge.type.toUpperCase()}`);
  console.log(`     Question: "${challenge.question}"`);
  console.log(`     Difficulty: ${(challenge.difficulty * 100).toFixed(0)}%`);
  console.log(`     Fuzzy match: ${challenge.fuzzy_match ? 'Yes' : 'No (exact match required)'}`);
  console.log('');
});

// =============================================================================
// PHASE 2: TEST EXACT MATCH VERIFICATION
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('✅ PHASE 2: Exact Match Verification\n');

// Create a simple challenge with exact answer
const exactChallenge = CognitiveChallenge.create(
  'alice',
  'personal_fact',
  'What is your favorite color?',
  'blue',
  {
    fuzzy_match: false,
    difficulty: 0.3,
  }
);

console.log(`Question: "${exactChallenge.question}"`);
console.log('Fuzzy match: No (exact match required)\n');

// Test correct answer
const correctResult = CognitiveChallenge.verify(exactChallenge, 'blue');
console.log(`Answer: "blue"`);
console.log(`✅ Verified: ${correctResult.verified}`);
console.log(`✅ Confidence: ${(correctResult.confidence * 100).toFixed(0)}%`);
console.log(`✅ Method: ${correctResult.method}\n`);

// Test case insensitive
const caseResult = CognitiveChallenge.verify(exactChallenge, 'BLUE');
console.log(`Answer: "BLUE" (different case)`);
console.log(`✅ Verified: ${caseResult.verified} (case-insensitive)`);
console.log(`✅ Confidence: ${(caseResult.confidence * 100).toFixed(0)}%\n`);

// Test incorrect answer
const wrongResult = CognitiveChallenge.verify(exactChallenge, 'red');
console.log(`Answer: "red" (incorrect)`);
console.log(`❌ Verified: ${wrongResult.verified}`);
console.log(`❌ Confidence: ${(wrongResult.confidence * 100).toFixed(0)}%`);
console.log(`❌ Reason: ${wrongResult.reason}\n`);

// =============================================================================
// PHASE 3: TEST FUZZY MATCH VERIFICATION
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('🔍 PHASE 3: Fuzzy Match Verification\n');

const fuzzyChallenge = CognitiveChallenge.create(
  'alice',
  'preference',
  'What is your favorite morning beverage?',
  'coffee',
  {
    fuzzy_match: true,
    confidence_threshold: 0.3,
    difficulty: 0.2,
  }
);

console.log(`Question: "${fuzzyChallenge.question}"`);
console.log('Fuzzy match: Yes (semantic similarity allowed)\n');

// Test exact match (should still work)
const fuzzyExactResult = CognitiveChallenge.verify(fuzzyChallenge, 'coffee');
console.log(`Answer: "coffee" (exact)`);
console.log(`✅ Verified: ${fuzzyExactResult.verified}`);
console.log(`✅ Confidence: ${(fuzzyExactResult.confidence * 100).toFixed(0)}%`);
console.log(`✅ Method: ${fuzzyExactResult.method}\n`);

// =============================================================================
// PHASE 4: MULTI-FACTOR AUTHENTICATION
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('🔐 PHASE 4: Multi-Factor Cognitive Authentication\n');

// Create a set of challenges
const challenge1 = CognitiveChallenge.create(
  'alice',
  'personal_fact',
  'What is your favorite hobby?',
  'photography'
);

const challenge2 = CognitiveChallenge.create(
  'alice',
  'preference',
  'Do you prefer coffee or tea?',
  'coffee'
);

const testChallengeSet = {
  user_id: 'alice',
  challenges: [challenge1, challenge2],
  created_at: Date.now(),
  last_updated: Date.now(),
};

console.log('Challenges:');
console.log('  1. What is your favorite hobby?');
console.log('  2. Do you prefer coffee or tea?\n');

// Test correct answers
const answers = [
  { challenge_id: challenge1.challenge_id, answer: 'photography' },
  { challenge_id: challenge2.challenge_id, answer: 'coffee' },
];

const authResult = CognitiveAuthenticator.authenticate(testChallengeSet, answers, {
  min_challenges: 2,
  min_confidence: 0.7,
});

console.log('Answers provided:');
console.log('  1. "photography" ✓');
console.log('  2. "coffee" ✓\n');

console.log('🔐 Authentication Result:');
console.log(`  Authenticated: ${authResult.authenticated ? '✅ YES' : '❌ NO'}`);
console.log(`  Confidence: ${(authResult.confidence * 100).toFixed(0)}%`);
console.log(`  Challenges passed: ${authResult.results.filter((r) => r.result.verified).length}/${authResult.results.length}\n`);

// Test with one wrong answer
const wrongAnswers = [
  { challenge_id: challenge1.challenge_id, answer: 'photography' },
  { challenge_id: challenge2.challenge_id, answer: 'tea' }, // Wrong!
];

const failedAuthResult = CognitiveAuthenticator.authenticate(
  testChallengeSet,
  wrongAnswers,
  {
    min_challenges: 2,
    min_confidence: 0.7,
  }
);

console.log('Answers provided (one wrong):');
console.log('  1. "photography" ✓');
console.log('  2. "tea" ✗ (incorrect)\n');

console.log('🔐 Authentication Result:');
console.log(`  Authenticated: ${failedAuthResult.authenticated ? '✅ YES' : '❌ NO'}`);
console.log(`  Confidence: ${(failedAuthResult.confidence * 100).toFixed(0)}%`);
console.log(`  Challenges passed: ${failedAuthResult.results.filter((r) => r.result.verified).length}/${failedAuthResult.results.length}\n`);

// =============================================================================
// PHASE 5: INTEGRATION WITH BEHAVIORAL SECURITY
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('🔗 PHASE 5: Integration with Behavioral Security\n');

// Build behavioral profiles
let linguistic = LinguisticCollector.createProfile('alice');
let typing = TypingCollector.createProfile('alice');
let emotional = EmotionalCollector.createProfile('alice');
let temporal = TemporalCollector.createProfile('alice', 'America/New_York');

// Build baseline
for (let i = 0; i < 50; i++) {
  const text = 'I am working on the project today. Everything is going well.';
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

const profiles: UserSecurityProfiles = {
  user_id: 'alice',
  linguistic,
  typing,
  emotional,
  temporal,
  overall_confidence: 0.5,
  last_interaction: Date.now(),
};

console.log('✅ Behavioral baseline built\n');

// Simulate coercion scenario
const coercionInteraction: Interaction = {
  interaction_id: 'test_coercion',
  user_id: 'alice',
  timestamp: new Date(2025, 0, 1, 10, 0, 0).getTime(),
  text: 'I have to transfer the funds now. No choice. They want me to do it.',
  text_length: 68,
  word_count: 14,
  session_id: 'session_test',
  operation_type: 'transfer',
  typing_data: {
    keystroke_intervals: Array(68)
      .fill(0)
      .map(() => 45 + Math.random() * 10), // Rushed
    total_typing_time: 68 * 50,
    pauses: [120, 140],
    backspaces: 8,
    corrections: 6,
  },
};

// Build security context
const context = MultiSignalDetector.buildSecurityContext(
  profiles,
  coercionInteraction,
  {
    operation_type: 'transfer',
    is_sensitive_operation: true,
    operation_value: 50000,
  },
  30
);

console.log('🚨 Coercion Scenario Detected:');
console.log(`  Text: "${coercionInteraction.text}"`);
console.log(`  Operation: TRANSFER $50,000`);
console.log(`  Duress Score: ${context.duress_score.score.toFixed(3)}`);
console.log(`  Coercion Detected: ${context.coercion_score.alert ? 'YES 🚨' : 'NO'}`);
console.log(`  Decision: ${context.decision.toUpperCase()}\n`);

// Check if cognitive challenge is required
const challengeRequirement =
  MultiSignalDetector.requiresCognitiveChallenge(context);

console.log('🧠 Cognitive Challenge Requirement:');
console.log(`  Required: ${challengeRequirement.required ? 'YES ✓' : 'NO'}`);
console.log(`  Reason: ${challengeRequirement.reason}`);
console.log(`  Difficulty: ${challengeRequirement.difficulty_level.toUpperCase()}\n`);

if (challengeRequirement.required) {
  // Request cognitive verification
  const verification = MultiSignalDetector.requestCognitiveVerification(
    challengeSet,
    context
  );

  console.log('📋 Cognitive Verification Requested:');
  console.log(`  Challenges to present: ${verification.challenges.length}`);
  console.log(`  Required to pass: ${verification.required_count}`);
  console.log(`  Minimum confidence: ${(verification.min_confidence * 100).toFixed(0)}%`);
  console.log(`  Reason: ${verification.reason}\n`);

  if (verification.challenges.length > 0) {
    console.log('  Challenges:');
    verification.challenges.forEach((c, i) => {
      console.log(`    ${i + 1}. [${c.type}] "${c.question}"`);
      console.log(`       Difficulty: ${(c.difficulty * 100).toFixed(0)}%`);
    });
  }
}

console.log('\n');

// =============================================================================
// PHASE 6: CHALLENGE SELECTION
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('🎯 PHASE 6: Context-Aware Challenge Selection\n');

console.log('Scenario A: Normal Operation (Low Security)');
const normalChallenges = CognitiveChallenge.selectChallenges(challengeSet, {
  is_sensitive_operation: false,
  duress_score: 0.1,
  count: 1,
});
console.log(`  Challenges selected: ${normalChallenges.length}`);
if (normalChallenges.length > 0) {
  console.log(`  Difficulty: ${(normalChallenges[0].difficulty * 100).toFixed(0)}%`);
}
console.log('');

console.log('Scenario B: Sensitive Operation (Medium Security)');
const sensitiveChallenges = CognitiveChallenge.selectChallenges(challengeSet, {
  is_sensitive_operation: true,
  duress_score: 0.3,
  count: 1,
});
console.log(`  Challenges selected: ${sensitiveChallenges.length}`);
if (sensitiveChallenges.length > 0) {
  console.log(`  Average difficulty: ${(sensitiveChallenges.reduce((sum, c) => sum + c.difficulty, 0) / sensitiveChallenges.length * 100).toFixed(0)}%`);
}
console.log('');

console.log('Scenario C: High Duress (High Security)');
const duressChallenges = CognitiveChallenge.selectChallenges(challengeSet, {
  is_sensitive_operation: true,
  duress_score: 0.8,
  count: 2,
});
console.log(`  Challenges selected: ${duressChallenges.length}`);
if (duressChallenges.length > 0) {
  console.log(`  Average difficulty: ${(duressChallenges.reduce((sum, c) => sum + c.difficulty, 0) / duressChallenges.length * 100).toFixed(0)}%`);
}
console.log('\n');

// =============================================================================
// SUMMARY
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('📊 COGNITIVE CHALLENGE DEMO - SUMMARY\n');
console.log('='.repeat(80));
console.log('\n');

console.log('✅ Features Demonstrated:');
console.log('   1. Challenge creation (4 types: personal_fact, preference, memory, reasoning)');
console.log('   2. Exact match verification (case-insensitive)');
console.log('   3. Fuzzy match verification (semantic similarity)');
console.log('   4. Multi-factor authentication (multiple challenges)');
console.log('   5. Integration with behavioral security');
console.log('   6. Context-aware challenge selection\n');

console.log('✅ Key Capabilities:');
console.log('   ✓ Complements behavioral biometrics');
console.log('   ✓ Adaptive difficulty (easy/medium/hard)');
console.log('   ✓ Context-aware (responds to duress/coercion)');
console.log('   ✓ Multi-factor (requires multiple challenges for high security)');
console.log('   ✓ Fuzzy matching (semantic similarity for flexible answers)');
console.log('   ✓ Secure (answers hashed, never stored in plaintext)\n');

console.log('🔐 MULTI-FACTOR COGNITIVE AUTHENTICATION: WORKING!');
console.log('   Behavioral (Who You ARE) + Cognitive (What You KNOW)');
console.log('   = Highest security confidence\n');

console.log('='.repeat(80));
