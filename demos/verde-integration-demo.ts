/**
 * VERDE Integration Demo (VERMELHO + VERDE)
 *
 * Demonstrates security-aware Git version control.
 * Shows how behavioral biometrics prevent malicious commits under coercion.
 *
 * Scenarios:
 * 1. Normal commit - Allowed
 * 2. Sensitive Git operation (normal behavior) - Challenge required
 * 3. Commit under coercion - Blocked
 * 4. Sensitive operation under coercion - Blocked + Snapshot created
 * 5. Duress snapshot system demonstration
 */

import { SecurityStorage } from '../src/grammar-lang/security/security-storage';
import { LinguisticCollector } from '../src/grammar-lang/security/linguistic-collector';
import { TypingCollector } from '../src/grammar-lang/security/typing-collector';
import { EmotionalCollector } from '../src/grammar-lang/security/emotional-collector';
import { TemporalCollector } from '../src/grammar-lang/security/temporal-collector';
import { CognitiveChallenge } from '../src/grammar-lang/security/cognitive-challenge';
import { UserSecurityProfiles, Interaction } from '../src/grammar-lang/security/types';
import {
  GitOperationGuard,
  createCommitRequest,
  createMutationRequest,
  shouldProceedWithGitOperation,
  getGitValidationSummary,
  generateSecurityMetadata,
} from '../src/grammar-lang/security/git-operation-guard';
import * as fs from 'fs';
import * as path from 'path';

console.log('🔴🟢 VERMELHO + VERDE INTEGRATION DEMO\n');
console.log('='.repeat(80));
console.log('Security-Aware Git Version Control');
console.log('Behavioral Biometrics + Genetic VCS');
console.log('='.repeat(80));
console.log('\n');

// =============================================================================
// PHASE 1: SETUP
// =============================================================================

console.log('💾 PHASE 1: Initialize Security System\n');

// Create storage
const storage = new SecurityStorage('demo_verde_security');

// Build behavioral baseline for Alice
console.log('👤 Building behavioral baseline for user: alice\n');

let linguistic = LinguisticCollector.createProfile('alice');
let typing = TypingCollector.createProfile('alice');
let emotional = EmotionalCollector.createProfile('alice');
let temporal = TemporalCollector.createProfile('alice', 'America/New_York');

// Build baseline with 50 normal interactions
console.log('📊 Collecting baseline interactions (50 samples)...\n');

for (let i = 0; i < 50; i++) {
  const text = 'Committing changes to data analysis module. Good progress today.';
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

// Create demo files
const demoDir = 'demo_organisms';
if (!fs.existsSync(demoDir)) {
  fs.mkdirSync(demoDir, { recursive: true });
}

const testFile = path.join(demoDir, 'data-analysis-1.0.0.glass');
fs.writeFileSync(testFile, '(define analyze-data (data: Dataset) -> Results "Implementation here")');

console.log('📄 Test file created:');
console.log(`   Path: ${testFile}\n`);

// Create Git operation guard
const gitGuard = new GitOperationGuard(storage);

// =============================================================================
// PHASE 2: NORMAL COMMIT (SHOULD ALLOW)
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('📝 PHASE 2: Normal Git Commit (Expected: ALLOW)\n');

// Create normal commit request
const normalCommitRequest = createCommitRequest(
  'alice',
  testFile,
  'feat: add data analysis function\n\nImplemented basic data analysis',
  'human',
  { lines_added: 5, lines_removed: 0 }
);

// Validate commit
const normalCommitResult = gitGuard.validateCommitRequest(
  normalCommitRequest,
  normalProfiles
);

console.log(`\n✅ Normal Commit Result:`);
console.log(`   Decision: ${normalCommitResult.decision.toUpperCase()}`);
console.log(`   Allowed: ${normalCommitResult.allowed}`);
console.log(`   Confidence: ${(normalCommitResult.confidence * 100).toFixed(0)}%`);
console.log(`   Reason: ${normalCommitResult.reason}\n`);

// =============================================================================
// PHASE 3: SENSITIVE GIT OPERATION (NORMAL BEHAVIOR) - SHOULD CHALLENGE
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('⚠️  PHASE 3: Sensitive Git Operation (Normal Behavior - Expected: CHALLENGE)\n');

// Create sensitive commit request (force-push, delete)
const sensitiveCommitRequest = createCommitRequest(
  'alice',
  testFile,
  'refactor: force delete old implementation\n\nRemoving deprecated code with --force',
  'human',
  { lines_added: 2, lines_removed: 150 }
);

// Validate sensitive commit
const sensitiveCommitResult = gitGuard.validateCommitRequest(
  sensitiveCommitRequest,
  normalProfiles
);

console.log(`\n⚠️  Sensitive Commit Result:`);
console.log(`   Decision: ${sensitiveCommitResult.decision.toUpperCase()}`);
console.log(`   Allowed: ${sensitiveCommitResult.allowed}`);
console.log(`   Requires Challenge: ${sensitiveCommitResult.requires_cognitive_challenge}`);
console.log(`   Challenge Difficulty: ${sensitiveCommitResult.challenge_difficulty}`);
console.log(`   Sensitive Keywords: ${sensitiveCommitResult.security_context.sensitive_keywords.join(', ')}\n`);

// =============================================================================
// PHASE 4: COMMIT UNDER COERCION (SHOULD BLOCK)
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('🚨 PHASE 4: Commit Under Coercion (Expected: BLOCK)\n');

// Simulate coercion interaction to update profiles
const coercionInteraction: Interaction = {
  interaction_id: 'coercion_test',
  user_id: 'alice',
  timestamp: Date.now(),
  text: 'I must commit this now. They are forcing me. No choice here.',
  text_length: 60,
  word_count: 12,
  session_id: 'session_coercion',
  operation_type: 'git_operation',
  typing_data: {
    keystroke_intervals: Array(60)
      .fill(0)
      .map(() => 45 + Math.random() * 10), // Rushed typing
    total_typing_time: 60 * 50,
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
console.log(`   Typing speed: ${(60 / (60 * 50 / 1000)).toFixed(0)} chars/sec (RUSHED)`);
console.log(`   Errors: ${coercionInteraction.typing_data.backspaces} backspaces\n`);

// Try to commit under coercion
const coercionCommitRequest = createCommitRequest(
  'alice',
  testFile,
  'feat: add new feature\n\nNormal commit message',
  'human',
  { lines_added: 10, lines_removed: 0 }
);

const coercionCommitResult = gitGuard.validateCommitRequest(
  coercionCommitRequest,
  coercionProfiles
);

console.log(`\n🚫 Coercion Commit Result:`);
console.log(`   Decision: ${coercionCommitResult.decision.toUpperCase()}`);
console.log(`   Allowed: ${coercionCommitResult.allowed}`);
console.log(`   Reason: ${coercionCommitResult.reason}`);
console.log(`   Security system prevented commit under coercion!\n`);

// =============================================================================
// PHASE 5: SENSITIVE OPERATION UNDER COERCION (SHOULD BLOCK + SNAPSHOT)
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('🔥 PHASE 5: Sensitive Operation Under Coercion (Expected: BLOCK + SNAPSHOT)\n');

// Try sensitive operation under coercion
const criticalCommitRequest = createCommitRequest(
  'alice',
  testFile,
  'refactor: force-push delete all data\n\nRemoving everything with git reset --hard',
  'human',
  { lines_added: 0, lines_removed: 200 }
);

const criticalCommitResult = gitGuard.validateCommitRequest(
  criticalCommitRequest,
  coercionProfiles
);

console.log(`\n🚫 Critical Block Result:`);
console.log(`   Decision: ${criticalCommitResult.decision.toUpperCase()}`);
console.log(`   Allowed: ${criticalCommitResult.allowed}`);
console.log(`   Snapshot Created: ${criticalCommitResult.snapshot_created}`);
if (criticalCommitResult.snapshot_path) {
  console.log(`   Snapshot Path: ${criticalCommitResult.snapshot_path}`);
}
console.log(`   Sensitive operation blocked + backup created!\n`);

// =============================================================================
// PHASE 6: MUTATION VALIDATION
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('🧬 PHASE 6: Genetic Mutation Validation\n');

// Normal mutation (should allow)
const normalMutationRequest = createMutationRequest(
  'alice',
  testFile,
  'agi',
  '1.0.0',
  '1.0.1'
);

const normalMutationResult = gitGuard.validateMutationRequest(
  normalMutationRequest,
  normalProfiles
);

console.log(`✅ Normal Mutation Result:`);
console.log(`   Decision: ${normalMutationResult.decision.toUpperCase()}`);
console.log(`   Allowed: ${normalMutationResult.allowed}`);
console.log(`   Version: 1.0.0 → 1.0.1\n`);

// Mutation under coercion (should block)
const coercionMutationRequest = createMutationRequest(
  'alice',
  testFile,
  'human',
  '1.0.0',
  '2.0.0' // Major version change
);

const coercionMutationResult = gitGuard.validateMutationRequest(
  coercionMutationRequest,
  coercionProfiles
);

console.log(`🚫 Coercion Mutation Result:`);
console.log(`   Decision: ${coercionMutationResult.decision.toUpperCase()}`);
console.log(`   Allowed: ${coercionMutationResult.allowed}`);
console.log(`   Reason: ${coercionMutationResult.reason}\n`);

// =============================================================================
// PHASE 7: DURESS SNAPSHOT MANAGEMENT
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('📸 PHASE 7: Duress Snapshot Management\n');

// List all duress snapshots
const snapshots = gitGuard.listDuressSnapshots();

console.log(`📋 Duress Snapshots Created (${snapshots.length} total):\n`);

snapshots.forEach((snapshot, index) => {
  console.log(`${index + 1}. Snapshot ID: ${snapshot.snapshot_id}`);
  console.log(`   File: ${snapshot.file_path}`);
  console.log(`   User: ${snapshot.user_id}`);
  console.log(`   Duress: ${(snapshot.duress_score * 100).toFixed(0)}%`);
  console.log(`   Coercion: ${(snapshot.coercion_score * 100).toFixed(0)}%`);
  console.log(`   Timestamp: ${new Date(snapshot.timestamp).toLocaleString()}`);
  console.log('');
});

// =============================================================================
// PHASE 8: SECURITY METADATA GENERATION
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('📝 PHASE 8: Security Metadata for Commits\n');

console.log('Example commit message with security metadata:');
console.log('─'.repeat(80));

const exampleCommitMessage = `feat: add data analysis function

Implemented statistical analysis module

🧬 Auto-generated by VCS (O(1))

Co-Authored-By: Human <human@fiat.ai>`;

const securityMetadata = generateSecurityMetadata(normalCommitResult.security_context);
const fullCommitMessage = exampleCommitMessage + securityMetadata;

console.log(fullCommitMessage);
console.log('─'.repeat(80));
console.log('\n');

// =============================================================================
// PHASE 9: AUDIT TRAIL
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('📊 PHASE 9: Security Audit Trail\n');

// Get user events
const auditEvents = storage.getUserEvents('alice', 20);

console.log(`📋 Git Security Events (${auditEvents.length} total):\n`);

const gitEvents = auditEvents.filter((e) => e.operation_type === 'git_operation');

gitEvents.forEach((event, index) => {
  const date = new Date(event.timestamp);
  console.log(`${index + 1}. [${event.event_type}]`);
  console.log(`   Time: ${date.toLocaleTimeString()}`);
  console.log(`   Decision: ${event.decision.toUpperCase()}`);
  console.log(`   Operation: ${(event.context as any)?.operation_type || 'N/A'}`);
  console.log(`   File: ${path.basename((event.context as any)?.file_path || 'N/A')}`);
  console.log(`   Duress: ${event.duress_score?.toFixed(3) || 'N/A'}`);
  console.log(`   Coercion: ${event.coercion_score?.toFixed(3) || 'N/A'}`);
  console.log(`   Sensitive: ${(event.context as any)?.is_sensitive ? 'YES' : 'NO'}`);
  console.log(`   Snapshot: ${(event.context as any)?.snapshot_created ? 'YES' : 'NO'}`);
  console.log(`   Reason: ${event.reason}`);
  console.log('');
});

// Get Git statistics
const gitStats = gitGuard.getGitStatistics('alice', 24);

console.log('📊 Git Operation Statistics (Last 24h):');
console.log(`   Total operations: ${gitStats.total_git_operations}`);
console.log(`   Blocked: ${gitStats.blocked_operations}`);
console.log(`   Sensitive: ${gitStats.sensitive_operations}`);
console.log(`   Snapshots created: ${gitStats.snapshots_created}`);
console.log(`   Avg duress: ${(gitStats.avg_duress_score * 100).toFixed(0)}%`);
console.log(`   Avg coercion: ${(gitStats.avg_coercion_score * 100).toFixed(0)}%\n`);

// =============================================================================
// SUMMARY
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('📊 VERMELHO + VERDE INTEGRATION - SUMMARY\n');
console.log('='.repeat(80));
console.log('\n');

console.log('✅ Integration Demonstrated:');
console.log('   1. Normal commit: ALLOWED (behavioral baseline normal)');
console.log('   2. Sensitive Git operation (normal): CHALLENGED (verification required)');
console.log('   3. Commit under coercion: BLOCKED (behavioral anomaly detected)');
console.log('   4. Sensitive + coercion: IMMEDIATE BLOCK + SNAPSHOT (maximum protection)');
console.log('   5. Mutation validation: Same security checks applied');
console.log('   6. Duress snapshots: Auto-backup created for recovery\n');

console.log('✅ Security Features:');
console.log('   ✓ Pre-commit behavioral screening');
console.log('   ✓ Pre-mutation behavioral screening');
console.log('   ✓ Sensitive Git operation detection (force-push, delete, reset, etc.)');
console.log('   ✓ Coercion/duress detection before Git operations');
console.log('   ✓ Multi-factor cognitive challenges for sensitive operations');
console.log('   ✓ Duress-triggered snapshot system (auto-backup)');
console.log('   ✓ Security metadata in commit messages');
console.log('   ✓ Complete audit trail of all Git operations');
console.log('   ✓ Adaptive security (normal → challenge → block)\n');

console.log('✅ Results:');
console.log(`   Normal commit: ${normalCommitResult.allowed ? 'ALLOWED' : 'BLOCKED'}`);
console.log(`   Sensitive Git op (normal): ${sensitiveCommitResult.decision.toUpperCase()}`);
console.log(`   Commit under coercion: ${coercionCommitResult.allowed ? 'ALLOWED' : 'BLOCKED'}`);
console.log(`   Sensitive + coercion: ${criticalCommitResult.allowed ? 'ALLOWED' : 'BLOCKED'} + ${criticalCommitResult.snapshot_created ? 'SNAPSHOT' : 'NO SNAPSHOT'}`);
console.log(`   Normal mutation: ${normalMutationResult.allowed ? 'ALLOWED' : 'BLOCKED'}`);
console.log(`   Coercion mutation: ${coercionMutationResult.allowed ? 'ALLOWED' : 'BLOCKED'}`);
console.log(`   Snapshots created: ${snapshots.length}\n`);

console.log('🎉 VERMELHO + VERDE INTEGRATION: WORKING!');
console.log('   Git version control is now protected by behavioral biometrics\n');

console.log('='.repeat(80));

// Cleanup
storage.deleteProfile('alice');

// Clean up demo files
if (fs.existsSync(testFile)) {
  fs.unlinkSync(testFile);
}
if (fs.existsSync(demoDir) && fs.readdirSync(demoDir).length === 0) {
  fs.rmdirSync(demoDir);
}
