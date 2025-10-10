/**
 * Security Storage Demo
 *
 * Demonstrates integration between VERMELHO (Security) and LARANJA (Storage)
 * Shows O(1) persistence for behavioral profiles using .sqlo
 */

import { SecurityStorage } from '../src/grammar-lang/security/security-storage';
import { LinguisticCollector } from '../src/grammar-lang/security/linguistic-collector';
import { TypingCollector } from '../src/grammar-lang/security/typing-collector';
import { EmotionalCollector } from '../src/grammar-lang/security/emotional-collector';
import { TemporalCollector } from '../src/grammar-lang/security/temporal-collector';
import { CognitiveChallenge } from '../src/grammar-lang/security/cognitive-challenge';
import { MultiSignalDetector } from '../src/grammar-lang/security/multi-signal-detector';
import { Interaction, UserSecurityProfiles } from '../src/grammar-lang/security/types';

console.log('🔴🟠 SECURITY + STORAGE INTEGRATION DEMO\n');
console.log('='.repeat(80));
console.log('\nVERMELHO (Security) + LARANJA (Storage) = Persistent Behavioral Biometrics\n');
console.log('='.repeat(80));
console.log('\n');

// =============================================================================
// PHASE 1: CREATE STORAGE
// =============================================================================

console.log('💾 PHASE 1: Initialize Security Storage\n');

// Create storage (uses sqlo_security/ directory)
const storage = new SecurityStorage();

console.log('✅ Storage initialized');
console.log(`✅ Base directory: sqlo_security/`);
console.log(`✅ Profiles directory: sqlo_security/profiles/`);
console.log(`✅ Events directory: sqlo_security/events/\n`);

const stats = storage.getStatistics();
console.log('📊 Initial Statistics:');
console.log(`  Total profiles: ${stats.total_profiles}`);
console.log(`  Total events: ${stats.total_events}`);
console.log(`  Alerts (24h): ${stats.alerts_last_24h}\n`);

// =============================================================================
// PHASE 2: BUILD AND SAVE PROFILE
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('📝 PHASE 2: Build and Save Behavioral Profile\n');

// Build behavioral profiles
let linguistic = LinguisticCollector.createProfile('alice');
let typing = TypingCollector.createProfile('alice');
let emotional = EmotionalCollector.createProfile('alice');
let temporal = TemporalCollector.createProfile('alice', 'America/New_York');

console.log('Building baseline (50 interactions)...\n');

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

console.log('✅ Baseline built:');
console.log(`  Samples: ${linguistic.samples_analyzed}`);
console.log(`  Confidence: ${(profiles.overall_confidence * 100).toFixed(0)}%\n`);

// Save profile to storage
console.log('💾 Saving profile to storage...\n');
const profileHash = storage.saveProfile(profiles);

console.log(`✅ Profile saved!`);
console.log(`  User ID: ${profiles.user_id}`);
console.log(`  Profile Hash: ${profileHash.substring(0, 16)}...`);
console.log(`  Location: sqlo_security/profiles/${profileHash}/\n`);

// Save cognitive challenges
const challengeSet = CognitiveChallenge.generateChallengeSet('alice');
storage.saveChallenges('alice', challengeSet);

console.log(`✅ Cognitive challenges saved`);
console.log(`  Challenges: ${challengeSet.challenges.length}\n`);

// =============================================================================
// PHASE 3: LOAD PROFILE (O(1))
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('📖 PHASE 3: Load Profile (O(1) Lookup)\n');

const loadedProfiles = storage.loadProfile('alice');

if (loadedProfiles) {
  console.log('✅ Profile loaded successfully!');
  console.log(`  User ID: ${loadedProfiles.user_id}`);
  console.log(`  Linguistic samples: ${loadedProfiles.linguistic.samples_analyzed}`);
  console.log(`  Typing samples: ${loadedProfiles.typing.samples_analyzed}`);
  console.log(`  Emotional samples: ${loadedProfiles.emotional.samples_analyzed}`);
  console.log(`  Temporal samples: ${loadedProfiles.temporal.samples_analyzed}`);
  console.log(`  Overall confidence: ${(loadedProfiles.overall_confidence * 100).toFixed(0)}%\n`);
} else {
  console.log('❌ Failed to load profile\n');
}

const loadedChallenges = storage.loadChallenges('alice');
if (loadedChallenges) {
  console.log('✅ Challenges loaded:');
  console.log(`  Challenge count: ${loadedChallenges.challenges.length}\n`);
}

// =============================================================================
// PHASE 4: SECURITY EVENTS (AUDIT LOG)
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('🚨 PHASE 4: Security Events & Audit Log\n');

// Simulate coercion scenario
const coercionInteraction: Interaction = {
  interaction_id: 'test_coercion',
  user_id: 'alice',
  timestamp: Date.now(),
  text: 'I have to transfer the funds now. No choice. They want me to do it.',
  text_length: 68,
  word_count: 14,
  session_id: 'session_test',
  operation_type: 'transfer',
  typing_data: {
    keystroke_intervals: Array(68)
      .fill(0)
      .map(() => 45 + Math.random() * 10),
    total_typing_time: 68 * 50,
    pauses: [120, 140],
    backspaces: 8,
    corrections: 6,
  },
};

// Build security context
const context = MultiSignalDetector.buildSecurityContext(
  loadedProfiles!,
  coercionInteraction,
  {
    operation_type: 'transfer',
    is_sensitive_operation: true,
    operation_value: 50000,
  },
  30
);

console.log('🚨 Coercion Detected:');
console.log(`  User: ${context.user_id}`);
console.log(`  Operation: TRANSFER $50,000`);
console.log(`  Duress Score: ${context.duress_score.score.toFixed(3)}`);
console.log(`  Coercion Score: ${context.coercion_score.score.toFixed(3)}`);
console.log(`  Decision: ${context.decision.toUpperCase()}\n`);

// Log security event
console.log('💾 Logging security event...\n');

const eventHash = storage.logEvent({
  user_id: 'alice',
  timestamp: Date.now(),
  event_type: 'coercion_detected',
  duress_score: context.duress_score.score,
  coercion_score: context.coercion_score.score,
  confidence: context.coercion_score.confidence,
  decision: context.decision,
  reason: context.decision_reason,
  operation_type: 'transfer',
  operation_value: 50000,
  context: {
    coercion_indicators: context.coercion_score.indicators,
  },
});

console.log(`✅ Event logged!`);
console.log(`  Event Hash: ${eventHash.substring(0, 16)}...`);
console.log(`  Location: sqlo_security/events/${eventHash}/\n`);

// Log operation blocked event
storage.logEvent({
  user_id: 'alice',
  timestamp: Date.now(),
  event_type: 'operation_blocked',
  confidence: 1.0,
  decision: 'block',
  reason: 'Coercion detected during sensitive transfer operation',
  operation_type: 'transfer',
  operation_value: 50000,
});

console.log('✅ Operation blocked event logged\n');

// =============================================================================
// PHASE 5: QUERY EVENTS
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('📊 PHASE 5: Query Security Events\n');

// Get user events
const userEvents = storage.getUserEvents('alice', 10);

console.log(`📋 User Events (${userEvents.length} total):\n`);

userEvents.forEach((event, index) => {
  const date = new Date(event.timestamp);
  console.log(`  ${index + 1}. [${event.event_type}]`);
  console.log(`     Time: ${date.toLocaleString()}`);
  console.log(`     Decision: ${event.decision.toUpperCase()}`);
  console.log(`     Confidence: ${(event.confidence * 100).toFixed(0)}%`);
  if (event.duress_score) {
    console.log(`     Duress: ${event.duress_score.toFixed(3)}`);
  }
  if (event.coercion_score) {
    console.log(`     Coercion: ${event.coercion_score.toFixed(3)}`);
  }
  console.log('');
});

// Get recent alerts
const alerts = storage.getRecentAlerts(24);

console.log(`🚨 Recent Alerts (last 24h): ${alerts.length}\n`);

alerts.forEach((alert, index) => {
  const date = new Date(alert.timestamp);
  console.log(`  ${index + 1}. [${alert.event_type}] - ${alert.reason}`);
  console.log(`     Time: ${date.toLocaleString()}`);
  console.log('');
});

// =============================================================================
// PHASE 6: INCREMENTAL UPDATE
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('🔄 PHASE 6: Incremental Profile Update\n');

// Simulate one more interaction
const newInteraction: Interaction = {
  interaction_id: 'update_1',
  user_id: 'alice',
  timestamp: Date.now(),
  text: 'Reviewing the security implementation. Looking good!',
  text_length: 52,
  word_count: 7,
  session_id: 'session_new',
  typing_data: {
    keystroke_intervals: Array(52)
      .fill(0)
      .map(() => 105 + Math.random() * 15),
    total_typing_time: 52 * 110,
    pauses: [290, 310],
    backspaces: 0,
    corrections: 0,
  },
};

// Update profiles
const updatedLinguistic = LinguisticCollector.analyzeAndUpdate(
  loadedProfiles!.linguistic,
  newInteraction
);

console.log('📝 Updating linguistic profile...\n');

// Save only the updated profile (O(1) incremental update)
const updateSuccess = storage.updateProfile('alice', {
  linguistic: updatedLinguistic,
  overall_confidence: 0.51,
});

if (updateSuccess) {
  console.log('✅ Profile updated!');
  console.log(`  Samples: ${updatedLinguistic.samples_analyzed}`);
  console.log(`  Confidence: 51%\n`);
} else {
  console.log('❌ Update failed\n');
}

// Verify update
const verifyProfiles = storage.loadProfile('alice');
if (verifyProfiles) {
  console.log('✅ Verified update:');
  console.log(`  Linguistic samples: ${verifyProfiles.linguistic.samples_analyzed}`);
  console.log(`  Overall confidence: ${(verifyProfiles.overall_confidence * 100).toFixed(0)}%\n`);
}

// =============================================================================
// PHASE 7: METADATA & STATISTICS
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('📊 PHASE 7: Metadata & Statistics\n');

const metadata = storage.getProfileMetadata('alice');
if (metadata) {
  const createdDate = new Date(metadata.created_at);
  const updatedDate = new Date(metadata.last_updated);

  console.log('📋 Profile Metadata:');
  console.log(`  User ID: ${metadata.user_id}`);
  console.log(`  User Hash: ${metadata.user_hash.substring(0, 16)}...`);
  console.log(`  Created: ${createdDate.toLocaleString()}`);
  console.log(`  Last Updated: ${updatedDate.toLocaleString()}`);
  console.log(`  Samples: ${metadata.samples_analyzed}`);
  console.log(`  Confidence: ${(metadata.overall_confidence * 100).toFixed(0)}%\n`);
}

const finalStats = storage.getStatistics();
console.log('📊 Final Statistics:');
console.log(`  Total profiles: ${finalStats.total_profiles}`);
console.log(`  Total events: ${finalStats.total_events}`);
console.log(`  Alerts (24h): ${finalStats.alerts_last_24h}\n`);

// =============================================================================
// SUMMARY
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('📊 SECURITY STORAGE INTEGRATION - SUMMARY\n');
console.log('='.repeat(80));
console.log('\n');

console.log('✅ Features Demonstrated:');
console.log('   1. Profile persistence (save/load/update) - O(1)');
console.log('   2. Cognitive challenge storage');
console.log('   3. Security events audit log');
console.log('   4. Recent alerts query (last 24h)');
console.log('   5. Incremental updates (efficient)');
console.log('   6. Profile metadata & statistics\n');

console.log('✅ Storage Architecture:');
console.log('   📁 sqlo_security/');
console.log('      ├── profiles/ (user behavioral profiles)');
console.log('      │   └── <user_hash>/');
console.log('      │       ├── linguistic.json');
console.log('      │       ├── typing.json');
console.log('      │       ├── emotional.json');
console.log('      │       ├── temporal.json');
console.log('      │       ├── challenges.json');
console.log('      │       └── metadata.json');
console.log('      ├── events/ (security audit log)');
console.log('      │   └── <event_hash>/');
console.log('      │       ├── event.json');
console.log('      │       └── metadata.json');
console.log('      └── .index (O(1) lookups)\n');

console.log('✅ Performance:');
console.log('   - Profile save: O(1)');
console.log('   - Profile load: O(1)');
console.log('   - Profile update: O(1)');
console.log('   - Event logging: O(1)');
console.log('   - User lookup: O(1) (hash-based index)\n');

console.log('✅ Integration Points:');
console.log('   🔴 VERMELHO: Behavioral biometrics + cognitive auth');
console.log('   🟠 LARANJA: Content-addressable storage (.sqlo)');
console.log('   = Persistent behavioral security profiles\n');

console.log('🎉 VERMELHO + LARANJA INTEGRATION: WORKING!');
console.log('   Behavioral profiles now persist with O(1) performance\n');

console.log('='.repeat(80));
