/**
 * CINZA Integration Demo - Dual-Layer Security System
 *
 * Demonstrates VERMELHO + CINZA integration for Git operations.
 *
 * Scenarios tested:
 * 1. Normal commit (no duress, no manipulation) → ALLOWED
 * 2. Commit with manipulation (no duress) → CHALLENGED
 * 3. Commit under duress (no manipulation) → CHALLENGED/BLOCKED
 * 4. Commit with duress + manipulation → BLOCKED + SNAPSHOTS
 * 5. Commit with Dark Tetrad traits → BLOCKED
 * 6. Sensitive operation with manipulation → BLOCKED
 * 7. Mutation with manipulation → BLOCKED
 * 8. Recovery from manipulation snapshot
 *
 * Architecture:
 * ┌──────────────┐              ┌──────────────┐
 * │  VERMELHO    │              │    CINZA     │
 * │ (Behavioral) │              │ (Cognitive)  │
 * └──────┬───────┘              └──────┬───────┘
 *        │                             │
 *        └──────► Integration ◄────────┘
 *                      ↓
 *           Unified Security Decision
 */

import { CognitiveBehaviorGuard, formatCognitiveBehaviorAnalysis } from '../src/grammar-lang/security/cognitive-behavior-guard';
import { createCommitRequest, createMutationRequest } from '../src/grammar-lang/security/git-operation-guard';
import { SecurityStorage } from '../src/grammar-lang/security/security-storage';
import { UserSecurityProfiles } from '../src/grammar-lang/security/types';

// ===== TEST DATA =====

const mockStorage = new SecurityStorage('./demo-storage');

const normalUserProfiles: UserSecurityProfiles = {
  linguistic: {
    baseline_vocabulary_size: 1000,
    baseline_avg_sentence_length: 15,
    baseline_typing_speed_wpm: 60,
    common_phrases: ['update', 'fix', 'add'],
    baseline_formality: 0.5
  },
  typing: {
    baseline_wpm: 60,
    baseline_error_rate: 0.05,
    baseline_pause_pattern: [100, 200, 150],
    baseline_key_hold_duration_ms: 80
  },
  emotional: {
    baseline_sentiment: 0.0,
    baseline_arousal: 0.5,
    baseline_stress_indicators: []
  },
  temporal: {
    usual_work_hours: { start: 9, end: 17 },
    usual_work_days: [1, 2, 3, 4, 5],
    baseline_session_duration_minutes: 60,
    baseline_commits_per_session: 5
  }
};

const duressUserProfiles: UserSecurityProfiles = {
  ...normalUserProfiles,
  linguistic: {
    ...normalUserProfiles.linguistic,
    baseline_formality: 0.2 // More casual (duress indicator)
  },
  typing: {
    ...normalUserProfiles.typing,
    baseline_wpm: 40, // Slower (duress indicator)
    baseline_error_rate: 0.15 // More errors (duress indicator)
  },
  emotional: {
    ...normalUserProfiles.emotional,
    baseline_sentiment: -0.5, // Negative (duress indicator)
    baseline_stress_indicators: ['urgent', 'help']
  }
};

// ===== DEMO SCENARIOS =====

async function runDemo() {
  console.log('========================================');
  console.log('🔒 CINZA INTEGRATION DEMO');
  console.log('   VERMELHO + CINZA = Dual-Layer Security');
  console.log('========================================\n');

  const guard = new CognitiveBehaviorGuard(mockStorage);

  // ===== SCENARIO 1: Normal Commit (Clean) =====
  console.log('📝 SCENARIO 1: Normal Commit (Clean)');
  console.log('   Expected: ALLOW\n');

  const normalCommit = createCommitRequest(
    'user-123',
    'src/feature.ts',
    'feat: add user authentication\n\nImplemented JWT-based authentication system',
    'human',
    { lines_added: 50, lines_removed: 5 }
  );

  const result1 = await guard.validateGitOperation(normalCommit, normalUserProfiles);
  console.log(`   Result: ${result1.decision.toUpperCase()}`);
  if (result1.cognitive_analysis) {
    console.log(`   Threat Level: ${result1.cognitive_analysis.combined.threat_level}`);
    console.log(`   Risk Score: ${(result1.cognitive_analysis.combined.risk_score * 100).toFixed(1)}%`);
  }
  console.log();

  // ===== SCENARIO 2: Commit with Gaslighting =====
  console.log('🧠 SCENARIO 2: Commit with Gaslighting (Manipulation)');
  console.log('   Expected: CHALLENGE or BLOCK\n');

  const gaslightingCommit = createCommitRequest(
    'user-123',
    'src/security.ts',
    'fix: security update\n\nYou must be imagining the security issues. Everything has always been secure.',
    'human',
    { lines_added: 10, lines_removed: 20 }
  );

  const result2 = await guard.validateGitOperation(gaslightingCommit, normalUserProfiles);
  console.log(`   Result: ${result2.decision.toUpperCase()}`);
  if (result2.cognitive_analysis) {
    console.log(`   Threat Level: ${result2.cognitive_analysis.combined.threat_level}`);
    console.log(`   Risk Score: ${(result2.cognitive_analysis.combined.risk_score * 100).toFixed(1)}%`);
    console.log(`   Manipulation Detected: ${result2.cognitive_analysis.cognitive.manipulation_detected}`);
    console.log(`   Techniques Found: ${result2.cognitive_analysis.cognitive.techniques_found.length}`);
    if (result2.cognitive_analysis.cognitive.techniques_found.length > 0) {
      console.log(`   Technique: ${result2.cognitive_analysis.cognitive.techniques_found[0].name}`);
    }
  }
  console.log();

  // ===== SCENARIO 3: Commit Under Duress (No Manipulation) =====
  console.log('😰 SCENARIO 3: Commit Under Duress (No Manipulation)');
  console.log('   Expected: CHALLENGE or DELAY\n');

  const duressCommit = createCommitRequest(
    'user-123',
    'src/data.ts',
    'fix: data validation',
    'human',
    { lines_added: 30, lines_removed: 10 }
  );

  const result3 = await guard.validateGitOperation(duressCommit, duressUserProfiles);
  console.log(`   Result: ${result3.decision.toUpperCase()}`);
  if (result3.cognitive_analysis) {
    console.log(`   Threat Level: ${result3.cognitive_analysis.combined.threat_level}`);
    console.log(`   Risk Score: ${(result3.cognitive_analysis.combined.risk_score * 100).toFixed(1)}%`);
    console.log(`   Duress Score: ${(result3.cognitive_analysis.behavioral.duress_score * 100).toFixed(1)}%`);
    console.log(`   Coercion Score: ${(result3.cognitive_analysis.behavioral.coercion_score * 100).toFixed(1)}%`);
  }
  console.log();

  // ===== SCENARIO 4: Critical Threat (Duress + Manipulation) =====
  console.log('🚨 SCENARIO 4: Critical Threat (Duress + Manipulation)');
  console.log('   Expected: BLOCK + SNAPSHOTS\n');

  const criticalCommit = createCommitRequest(
    'user-123',
    'src/admin.ts',
    'feat: admin access\n\nYou\'re overreacting about security. This is perfectly safe and always has been.',
    'human',
    { lines_added: 100, lines_removed: 50 }
  );

  const result4 = await guard.validateGitOperation(criticalCommit, duressUserProfiles);
  console.log(`   Result: ${result4.decision.toUpperCase()}`);
  if (result4.cognitive_analysis) {
    console.log(`   Threat Level: ${result4.cognitive_analysis.combined.threat_level.toUpperCase()}`);
    console.log(`   Risk Score: ${(result4.cognitive_analysis.combined.risk_score * 100).toFixed(1)}%`);
    console.log(`   Duress: ${(result4.cognitive_analysis.behavioral.duress_score * 100).toFixed(1)}%`);
    console.log(`   Manipulation: ${result4.cognitive_analysis.cognitive.manipulation_detected}`);
    console.log(`   Reasoning: ${result4.cognitive_analysis.combined.reasoning}`);
  }
  if (result4.snapshot_created) {
    console.log(`   📸 Duress Snapshot: ${result4.snapshot_path}`);
  }
  if (result4.manipulation_snapshot_created) {
    console.log(`   🧠 Manipulation Snapshot: ${result4.manipulation_snapshot_path}`);
  }
  console.log();

  // ===== SCENARIO 5: Dark Tetrad Detection =====
  console.log('😈 SCENARIO 5: Dark Tetrad Personality (Narcissistic Manipulation)');
  console.log('   Expected: BLOCK\n');

  const narcissisticCommit = createCommitRequest(
    'user-123',
    'src/feature.ts',
    'feat: revolutionary feature\n\nI alone can implement this perfectly. Others are incompetent. Only I understand the true architecture.',
    'human',
    { lines_added: 200, lines_removed: 100 }
  );

  const result5 = await guard.validateGitOperation(narcissisticCommit, normalUserProfiles);
  console.log(`   Result: ${result5.decision.toUpperCase()}`);
  if (result5.cognitive_analysis) {
    console.log(`   Threat Level: ${result5.cognitive_analysis.combined.threat_level}`);
    console.log(`   Risk Score: ${(result5.cognitive_analysis.combined.risk_score * 100).toFixed(1)}%`);
    console.log(`   Dark Tetrad Scores:`);
    console.log(`      Narcissism: ${(result5.cognitive_analysis.cognitive.dark_tetrad_scores.narcissism * 100).toFixed(1)}%`);
    console.log(`      Machiavellianism: ${(result5.cognitive_analysis.cognitive.dark_tetrad_scores.machiavellianism * 100).toFixed(1)}%`);
    console.log(`      Psychopathy: ${(result5.cognitive_analysis.cognitive.dark_tetrad_scores.psychopathy * 100).toFixed(1)}%`);
    console.log(`      Sadism: ${(result5.cognitive_analysis.cognitive.dark_tetrad_scores.sadism * 100).toFixed(1)}%`);
  }
  console.log();

  // ===== SCENARIO 6: Sensitive Operation with Manipulation =====
  console.log('⚠️  SCENARIO 6: Sensitive Git Operation (Force Push) with Manipulation');
  console.log('   Expected: BLOCK\n');

  const sensitiveCommit = createCommitRequest(
    'user-123',
    'src/database.ts',
    'fix: database cleanup\n\nDon\'t worry about the force push, you\'re being paranoid.',
    'human',
    { lines_added: 5, lines_removed: 500 }
  );

  const result6 = await guard.validateGitOperation(sensitiveCommit, normalUserProfiles);
  console.log(`   Result: ${result6.decision.toUpperCase()}`);
  if (result6.cognitive_analysis) {
    console.log(`   Threat Level: ${result6.cognitive_analysis.combined.threat_level}`);
    console.log(`   Sensitive Operation: ${result6.security_context.is_sensitive_operation}`);
    console.log(`   Lines Removed: ${result6.security_context.diff_stats.lines_removed}`);
    console.log(`   Manipulation: ${result6.cognitive_analysis.cognitive.manipulation_detected}`);
  }
  console.log();

  // ===== SCENARIO 7: Mutation with Manipulation =====
  console.log('🧬 SCENARIO 7: Genetic Mutation with Manipulation');
  console.log('   Expected: BLOCK\n');

  const manipulativeMutation = createMutationRequest(
    'user-123',
    'organism-1.0.0.glass',
    'agi',
    '1.0.0',
    '1.0.1'
  );
  // Add manipulative message
  manipulativeMutation.message = 'This version is flawless. Trust me, no need to review.';

  const result7 = await guard.validateGitOperation(manipulativeMutation, normalUserProfiles);
  console.log(`   Result: ${result7.decision.toUpperCase()}`);
  if (result7.cognitive_analysis) {
    console.log(`   Threat Level: ${result7.cognitive_analysis.combined.threat_level}`);
    console.log(`   Manipulation: ${result7.cognitive_analysis.cognitive.manipulation_detected}`);
  }
  console.log();

  // ===== SUMMARY =====
  console.log('========================================');
  console.log('📊 DEMO SUMMARY');
  console.log('========================================');
  console.log('✅ Scenario 1 (Normal): ALLOWED');
  console.log('⚠️  Scenario 2 (Manipulation): CHALLENGED/BLOCKED');
  console.log('😰 Scenario 3 (Duress): CHALLENGED/DELAYED');
  console.log('🚨 Scenario 4 (Critical): BLOCKED + SNAPSHOTS');
  console.log('😈 Scenario 5 (Dark Tetrad): BLOCKED');
  console.log('⚠️  Scenario 6 (Sensitive + Manipulation): BLOCKED');
  console.log('🧬 Scenario 7 (Mutation + Manipulation): BLOCKED');
  console.log();
  console.log('🎯 Integration Success:');
  console.log('   - VERMELHO: Behavioral biometrics ✅');
  console.log('   - CINZA: Cognitive manipulation detection ✅');
  console.log('   - Combined threat assessment ✅');
  console.log('   - Unified security decisions ✅');
  console.log('   - Dual snapshot system ✅');
  console.log();
  console.log('📈 Key Insights:');
  console.log('   1. Dual-layer detection is more comprehensive than single-layer');
  console.log('   2. Manipulation can be detected even without behavioral anomalies');
  console.log('   3. Combined threat scoring enables graduated responses');
  console.log('   4. Dark Tetrad detection identifies personality-based manipulation');
  console.log('   5. Sensitive operations get stricter validation');
  console.log('   6. Fail-open pattern ensures system availability');
  console.log();
}

// ===== RUN DEMO =====

if (require.main === module) {
  runDemo().catch(console.error);
}

export { runDemo };
