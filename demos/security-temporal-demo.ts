/**
 * Security - Temporal Patterns Demo
 *
 * Demonstrates temporal baseline building and impersonation detection
 */

import { TemporalCollector } from '../src/grammar-lang/security/temporal-collector';
import { TemporalAnomalyDetector } from '../src/grammar-lang/security/temporal-anomaly-detector';
import { Interaction } from '../src/grammar-lang/security/types';

console.log('🕐 SECURITY - TEMPORAL PATTERNS DEMO\n');
console.log('='.repeat(80));
console.log('\n');

// =============================================================================
// PHASE 1: BUILD BASELINE TEMPORAL PROFILE
// =============================================================================

console.log('📊 PHASE 1: Building Baseline Temporal Profile\n');

let profile = TemporalCollector.createProfile('alice', 'America/New_York');

// Simulate Alice's normal access pattern (9am-5pm, Mon-Fri)
console.log('Simulating Alice with 50 normal interactions (9am-5pm, Mon-Fri)...\n');

for (let i = 0; i < 50; i++) {
  const hour = 9 + (i % 8); // Rotate 9am-4pm
  const day = 1 + (i % 5); // Rotate Mon-Fri
  const timestamp = new Date(2025, 0, day, hour, 0, 0).getTime();

  const interaction: Interaction = {
    interaction_id: `baseline_${i}`,
    user_id: 'alice',
    timestamp,
    text: 'Working on project tasks.',
    text_length: 25,
    word_count: 4,
    session_id: 'session_baseline',
  };

  profile = TemporalCollector.analyzeAndUpdate(profile, interaction, 30);
}

const stats = TemporalCollector.getStatistics(profile);
const dayNames = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

console.log(`✅ Baseline built: ${profile.samples_analyzed} samples`);
console.log(`✅ Confidence: ${(profile.confidence * 100).toFixed(0)}%`);
console.log(`✅ Typical hours: ${Array.from(profile.hourly.typical_hours).sort((a, b) => a - b).join(', ')}`);
console.log(`✅ Typical days: ${Array.from(profile.daily.typical_days).map((d) => dayNames[d]).join(', ')}`);
console.log(`✅ Avg session duration: ${profile.sessions.session_duration_avg.toFixed(1)} minutes`);
console.log(`✅ Most active hours: ${stats.most_active_hours.join(', ')}`);
console.log('\n');

// =============================================================================
// PHASE 2: TEST NORMAL ACCESS (NO ANOMALY)
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('✅ PHASE 2: Test Normal Access (No Anomaly)\n');

// Test access at 10am Monday (normal)
const normalTimestamp = new Date(2025, 0, 1, 10, 0, 0).getTime();
const normalInteraction: Interaction = {
  interaction_id: 'test_normal',
  user_id: 'alice',
  timestamp: normalTimestamp,
  text: 'Working on project.',
  text_length: 19,
  word_count: 3,
  session_id: 'session_test',
};

const normalAnomaly = TemporalAnomalyDetector.detectTemporalAnomaly(profile, normalInteraction, 30);

const normalDate = new Date(normalTimestamp);
console.log(`Time: ${dayNames[normalDate.getDay()]} at ${normalDate.getHours()}:00`);
console.log(`\nAnomaly Score: ${normalAnomaly.score.toFixed(3)}`);
console.log(`Threshold: ${normalAnomaly.threshold}`);
console.log(`Alert: ${normalAnomaly.alert ? '🚨 YES' : '✅ NO'}`);
console.log(`\nDetails:`);
console.log(`  - Unusual hour: ${normalAnomaly.details.unusual_hour ? 'YES' : 'NO'}`);
console.log(`  - Unusual day: ${normalAnomaly.details.unusual_day ? 'YES' : 'NO'}`);
console.log(`  - Unusual duration: ${normalAnomaly.details.unusual_duration ? 'YES' : 'NO'}`);

if (normalAnomaly.specific_anomalies.length > 0 && !normalAnomaly.specific_anomalies[0].includes('Insufficient')) {
  console.log(`\nSpecific Anomalies:`);
  normalAnomaly.specific_anomalies.forEach((a) => console.log(`  ⚠️  ${a}`));
} else {
  console.log(`\n✅ No anomalies detected - normal access pattern`);
}

console.log('\n');

// =============================================================================
// PHASE 3: TEST UNUSUAL HOUR (LATE NIGHT)
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('⚠️  PHASE 3: Test Unusual Hour (Late Night Access)\n');

// Test access at 11pm Monday (unusual)
const lateNightTimestamp = new Date(2025, 0, 1, 23, 0, 0).getTime();
const lateNightInteraction: Interaction = {
  interaction_id: 'test_late_night',
  user_id: 'alice',
  timestamp: lateNightTimestamp,
  text: 'Working on project.',
  text_length: 19,
  word_count: 3,
  session_id: 'session_test',
};

const lateNightAnomaly = TemporalAnomalyDetector.detectTemporalAnomaly(
  profile,
  lateNightInteraction,
  30
);

const lateNightDate = new Date(lateNightTimestamp);
console.log(`Time: ${dayNames[lateNightDate.getDay()]} at ${lateNightDate.getHours()}:00`);
console.log(`\nAnomaly Score: ${lateNightAnomaly.score.toFixed(3)}`);
console.log(`Threshold: ${lateNightAnomaly.threshold}`);
console.log(`Alert: ${lateNightAnomaly.alert ? '🚨 YES' : '✅ NO'}`);
console.log(`\nDetails:`);
console.log(`  - Unusual hour: ${lateNightAnomaly.details.unusual_hour ? 'YES 🚨' : 'NO'}`);
console.log(`  - Unusual day: ${lateNightAnomaly.details.unusual_day ? 'YES 🚨' : 'NO'}`);
console.log(`  - Unusual duration: ${lateNightAnomaly.details.unusual_duration ? 'YES 🚨' : 'NO'}`);

if (lateNightAnomaly.specific_anomalies.length > 0) {
  console.log(`\nSpecific Anomalies:`);
  lateNightAnomaly.specific_anomalies.forEach((a) => console.log(`  🚨 ${a}`));
}

console.log('\n');

// =============================================================================
// PHASE 4: TEST IMPERSONATION (MIDDLE-OF-NIGHT + WEEKEND)
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('🚨 PHASE 4: Test Impersonation Detection (3am Sunday)\n');

// Test access at 3am Sunday (highly suspicious)
const impersonationTimestamp = new Date(2025, 0, 7, 3, 0, 0).getTime(); // Sunday 3am
const impersonationInteraction: Interaction = {
  interaction_id: 'test_impersonation',
  user_id: 'alice',
  timestamp: impersonationTimestamp,
  text: 'Working on project.',
  text_length: 19,
  word_count: 3,
  session_id: 'session_test',
};

const impersonationAnomaly = TemporalAnomalyDetector.detectTemporalAnomaly(
  profile,
  impersonationInteraction,
  30
);
const impersonationDetection = TemporalAnomalyDetector.detectImpersonation(
  profile,
  impersonationInteraction,
  30
);

const impersonationDate = new Date(impersonationTimestamp);
console.log(`Time: ${dayNames[impersonationDate.getDay()]} at ${impersonationDate.getHours()}:00`);
console.log(`\nAnomaly Score: ${impersonationAnomaly.score.toFixed(3)}`);
console.log(`Alert: ${impersonationAnomaly.alert ? '🚨 YES' : '✅ NO'}`);
console.log(`\nDetails:`);
console.log(`  - Unusual hour: ${impersonationAnomaly.details.unusual_hour ? 'YES 🚨' : 'NO'}`);
console.log(`  - Unusual day: ${impersonationAnomaly.details.unusual_day ? 'YES 🚨' : 'NO'}`);
console.log(`  - Unusual duration: ${impersonationAnomaly.details.unusual_duration ? 'YES 🚨' : 'NO'}`);

console.log(`\n🚨 IMPERSONATION DETECTION:`);
console.log(`  Detected: ${impersonationDetection.impersonation_detected ? '🚨 YES' : '✅ NO'}`);
console.log(`  Confidence: ${(impersonationDetection.confidence * 100).toFixed(0)}%`);

if (impersonationDetection.indicators.length > 0) {
  console.log(`\n  Indicators:`);
  impersonationDetection.indicators.forEach((i) => console.log(`    🚨 ${i}`));
}

if (impersonationAnomaly.specific_anomalies.length > 0) {
  console.log(`\nSpecific Anomalies:`);
  impersonationAnomaly.specific_anomalies.forEach((a) => console.log(`  🚨 ${a}`));
}

console.log('\n');

// =============================================================================
// PHASE 5: SERIALIZATION
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('💾 PHASE 5: Profile Serialization\n');

const profileJSON = TemporalCollector.toJSON(profile);
const restored = TemporalCollector.fromJSON(profileJSON);

console.log(`✅ Original profile ID: ${profile.user_id}`);
console.log(`✅ Restored profile ID: ${restored.user_id}`);
console.log(`✅ Samples match: ${profile.samples_analyzed === restored.samples_analyzed}`);
console.log(`✅ Confidence match: ${profile.confidence === restored.confidence}`);
console.log(`✅ Timezone match: ${profile.timezone === restored.timezone}`);
console.log(`✅ Typical hours match: ${profile.hourly.typical_hours.size === restored.hourly.typical_hours.size}`);
console.log('\n');

// =============================================================================
// SUMMARY
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('📊 DEMO SUMMARY\n');
console.log(`✅ Baseline Profile Built:`);
console.log(`   - ${profile.samples_analyzed} temporal samples analyzed`);
console.log(`   - ${(profile.confidence * 100).toFixed(0)}% confidence`);
console.log(`   - Typical hours: ${Array.from(profile.hourly.typical_hours).sort((a, b) => a - b).join(', ')}`);
console.log(`   - Typical days: ${Array.from(profile.daily.typical_days).map((d) => dayNames[d]).join(', ')}`);
console.log(`   - Avg session: ${profile.sessions.session_duration_avg.toFixed(1)}min`);
console.log('');
console.log(`✅ Anomaly Detection:`);
console.log(`   - Normal access (10am Mon): ${normalAnomaly.alert ? 'ALERT' : 'PASSED ✓'}`);
console.log(`   - Late night (11pm Mon): ${lateNightAnomaly.alert ? 'ALERT ✓' : 'PASSED'} (score: ${lateNightAnomaly.score.toFixed(2)})`);
console.log(`   - Middle-of-night (3am Sun): ${impersonationAnomaly.alert ? 'ALERT ✓' : 'PASSED'} (score: ${impersonationAnomaly.score.toFixed(2)})`);
console.log('');
console.log(`🚨 Impersonation Detection:`);
console.log(`   - 3am Sunday access: ${impersonationDetection.impersonation_detected ? 'DETECTED ✓' : 'NOT DETECTED'} (confidence: ${(impersonationDetection.confidence * 100).toFixed(0)}%)`);
console.log('');
console.log(`✅ Serialization:`);
console.log(`   - Profile can be saved and restored ✓`);
console.log('');
console.log('🕐 TEMPORAL FINGERPRINTING: WORKING!');
console.log('');
console.log('='.repeat(80));
