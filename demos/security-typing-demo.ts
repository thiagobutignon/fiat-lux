/**
 * Security - Typing Patterns Demo
 *
 * Demonstrates typing-based duress detection
 */

import { TypingCollector } from '../src/grammar-lang/security/typing-collector';
import { TypingAnomalyDetector } from '../src/grammar-lang/security/typing-anomaly-detector';
import { Interaction } from '../src/grammar-lang/security/types';

console.log('⌨️  SECURITY - TYPING PATTERNS DEMO\n');
console.log('=' .repeat(80));
console.log('\n');

// =============================================================================
// PHASE 1: BUILD BASELINE PROFILE
// =============================================================================

console.log('📊 PHASE 1: Building Baseline Typing Profile\n');

let profile = TypingCollector.createProfile('alice');

// Simulate Alice's normal typing (consistent ~110ms per keystroke)
console.log('Simulating Alice typing 50 normal interactions...\n');

for (let i = 0; i < 50; i++) {
  const text = 'This is a normal typing pattern';
  const keystrokeIntervals = Array(text.length).fill(0).map(() =>
    100 + Math.random() * 20 // 100-120ms (normal variance)
  );

  const interaction: Interaction = {
    interaction_id: `baseline_${i}`,
    user_id: 'alice',
    timestamp: Date.now(),
    text,
    text_length: text.length,
    word_count: text.split(/\s+/).length,
    session_id: 'session_baseline',
    typing_data: {
      keystroke_intervals: keystrokeIntervals,
      total_typing_time: keystrokeIntervals.reduce((a, b) => a + b, 0),
      pauses: [300, 250, 280], // Normal word pauses
      backspaces: Math.random() > 0.7 ? 1 : 0, // Occasional typo
      corrections: Math.random() > 0.8 ? 1 : 0,
    },
  };

  profile = TypingCollector.analyzeAndUpdate(profile, interaction);
}

console.log(`✅ Baseline built: ${profile.samples_analyzed} samples`);
console.log(`✅ Confidence: ${(profile.confidence * 100).toFixed(0)}%`);
console.log(`✅ Average keystroke interval: ${profile.timing.keystroke_interval_avg.toFixed(2)}ms`);
console.log(`✅ Average word pause: ${profile.timing.word_pause_duration.toFixed(2)}ms`);
console.log(`✅ Typo rate: ${profile.errors.typo_rate.toFixed(2)} per 100 chars`);
console.log(`✅ Backspace frequency: ${profile.errors.backspace_frequency.toFixed(2)}`);
console.log('\n');

// =============================================================================
// PHASE 2: TEST NORMAL TYPING (NO ANOMALY)
// =============================================================================

console.log('=' .repeat(80));
console.log('\n');
console.log('✅ PHASE 2: Test Normal Typing (No Anomaly)\n');

const normalText = 'Everything is fine';
const normalKeystroke = Array(normalText.length).fill(0).map(() =>
  105 + Math.random() * 15 // Still within normal range
);

const normalInteraction: Interaction = {
  interaction_id: 'test_normal',
  user_id: 'alice',
  timestamp: Date.now(),
  text: normalText,
  text_length: normalText.length,
  word_count: normalText.split(/\s+/).length,
  session_id: 'session_test',
  typing_data: {
    keystroke_intervals: normalKeystroke,
    total_typing_time: normalKeystroke.reduce((a, b) => a + b, 0),
    pauses: [290, 310],
    backspaces: 1,
    corrections: 0,
  },
};

const normalAnomaly = TypingAnomalyDetector.detectTypingAnomaly(profile, normalInteraction);

console.log(`Text: "${normalInteraction.text}"`);
console.log(`Avg keystroke interval: ${(normalKeystroke.reduce((a, b) => a + b, 0) / normalKeystroke.length).toFixed(2)}ms`);
console.log(`\nAnomaly Score: ${normalAnomaly.score.toFixed(3)}`);
console.log(`Threshold: ${normalAnomaly.threshold}`);
console.log(`Alert: ${normalAnomaly.alert ? '🚨 YES' : '✅ NO'}`);
console.log(`\nDetails:`);
console.log(`  - Speed deviation: ${normalAnomaly.details.speed_deviation.toFixed(3)}`);
console.log(`  - Error rate change: ${normalAnomaly.details.error_rate_change.toFixed(3)}`);
console.log(`  - Pause pattern change: ${normalAnomaly.details.pause_pattern_change.toFixed(3)}`);
console.log(`  - Input burst: ${normalAnomaly.details.input_burst ? 'YES' : 'NO'}`);

if (normalAnomaly.specific_anomalies.length > 0) {
  console.log(`\nSpecific Anomalies:`);
  normalAnomaly.specific_anomalies.forEach(a => console.log(`  ⚠️  ${a}`));
} else {
  console.log(`\n✅ No anomalies detected - normal typing behavior`);
}

console.log('\n');

// =============================================================================
// PHASE 3: TEST RUSHED TYPING (DURESS)
// =============================================================================

console.log('=' .repeat(80));
console.log('\n');
console.log('🚨 PHASE 3: Test Rushed Typing (Duress Detection)\n');

const rushedText = 'Transfer all funds now please';
const rushedKeystroke = Array(rushedText.length).fill(0).map(() =>
  40 + Math.random() * 10 // 3x faster! (rushed)
);

const rushedInteraction: Interaction = {
  interaction_id: 'test_rushed',
  user_id: 'alice',
  timestamp: Date.now(),
  text: rushedText,
  text_length: rushedText.length,
  word_count: rushedText.split(/\s+/).length,
  session_id: 'session_test',
  typing_data: {
    keystroke_intervals: rushedKeystroke,
    total_typing_time: rushedKeystroke.reduce((a, b) => a + b, 0),
    pauses: [100, 120], // Shorter pauses (rushed)
    backspaces: 8, // More errors (stress)
    corrections: 5,
  },
};

const rushedAnomaly = TypingAnomalyDetector.detectTypingAnomaly(profile, rushedInteraction);
const rushedDuress = TypingAnomalyDetector.detectDuressFromTyping(profile, rushedInteraction);

console.log(`Text: "${rushedInteraction.text}"`);
console.log(`Avg keystroke interval: ${(rushedKeystroke.reduce((a, b) => a + b, 0) / rushedKeystroke.length).toFixed(2)}ms (vs baseline ${profile.timing.keystroke_interval_avg.toFixed(2)}ms)`);
console.log(`\nAnomaly Score: ${rushedAnomaly.score.toFixed(3)}`);
console.log(`Threshold: ${rushedAnomaly.threshold}`);
console.log(`Alert: ${rushedAnomaly.alert ? '🚨 YES' : '✅ NO'}`);
console.log(`\nDetails:`);
console.log(`  - Speed deviation: ${rushedAnomaly.details.speed_deviation.toFixed(3)} ${rushedAnomaly.details.speed_deviation > 0.6 ? '🚨' : ''}`);
console.log(`  - Error rate change: ${rushedAnomaly.details.error_rate_change.toFixed(3)} ${rushedAnomaly.details.error_rate_change > 0.6 ? '🚨' : ''}`);
console.log(`  - Pause pattern change: ${rushedAnomaly.details.pause_pattern_change.toFixed(3)}`);
console.log(`  - Input burst: ${rushedAnomaly.details.input_burst ? 'YES 🚨' : 'NO'}`);

console.log(`\n🚨 DURESS DETECTION:`);
console.log(`  Detected: ${rushedDuress.duress_detected ? '🚨 YES' : '✅ NO'}`);
console.log(`  Confidence: ${(rushedDuress.confidence * 100).toFixed(0)}%`);

if (rushedDuress.indicators.length > 0) {
  console.log(`\n  Indicators:`);
  rushedDuress.indicators.forEach(i => console.log(`    🚨 ${i}`));
}

if (rushedAnomaly.specific_anomalies.length > 0) {
  console.log(`\nSpecific Anomalies:`);
  rushedAnomaly.specific_anomalies.forEach(a => console.log(`  🚨 ${a}`));
}

console.log('\n');

// =============================================================================
// PHASE 4: TEST PASTE ATTACK (INPUT BURST)
// =============================================================================

console.log('=' .repeat(80));
console.log('\n');
console.log('⚠️  PHASE 4: Test Paste Attack (Input Burst Detection)\n');

const pasteText = 'This is a very long text that was clearly pasted from somewhere else instead of being typed naturally';
const pasteKeystroke = Array(pasteText.length).fill(0).map(() =>
  3 + Math.random() * 2 // Impossibly fast (paste)
);

const pasteInteraction: Interaction = {
  interaction_id: 'test_paste',
  user_id: 'alice',
  timestamp: Date.now(),
  text: pasteText,
  text_length: pasteText.length,
  word_count: pasteText.split(/\s+/).length,
  session_id: 'session_test',
  typing_data: {
    keystroke_intervals: pasteKeystroke,
    total_typing_time: pasteKeystroke.reduce((a, b) => a + b, 0),
    pauses: [],
    backspaces: 0,
    corrections: 0,
  },
};

const pasteAnomaly = TypingAnomalyDetector.detectTypingAnomaly(profile, pasteInteraction);

console.log(`Text: "${pasteInteraction.text.substring(0, 50)}..."`);
console.log(`Length: ${pasteText.length} characters`);
console.log(`Avg keystroke interval: ${(pasteKeystroke.reduce((a, b) => a + b, 0) / pasteKeystroke.length).toFixed(2)}ms (IMPOSSIBLY FAST)`);
console.log(`\nAnomaly Score: ${pasteAnomaly.score.toFixed(3)}`);
console.log(`Alert: ${pasteAnomaly.alert ? '🚨 YES' : '✅ NO'}`);
console.log(`\nDetails:`);
console.log(`  - Speed deviation: ${pasteAnomaly.details.speed_deviation.toFixed(3)}`);
console.log(`  - Input burst: ${pasteAnomaly.details.input_burst ? 'YES 🚨🚨🚨' : 'NO'}`);

if (pasteAnomaly.specific_anomalies.length > 0) {
  console.log(`\nSpecific Anomalies:`);
  pasteAnomaly.specific_anomalies.forEach(a => console.log(`  🚨 ${a}`));
}

console.log('\n');

// =============================================================================
// PHASE 5: SERIALIZATION
// =============================================================================

console.log('=' .repeat(80));
console.log('\n');
console.log('💾 PHASE 5: Profile Serialization\n');

const profileJSON = TypingCollector.toJSON(profile);
const restored = TypingCollector.fromJSON(profileJSON);

console.log(`✅ Original profile ID: ${profile.user_id}`);
console.log(`✅ Restored profile ID: ${restored.user_id}`);
console.log(`✅ Samples match: ${profile.samples_analyzed === restored.samples_analyzed}`);
console.log(`✅ Confidence match: ${profile.confidence === restored.confidence}`);
console.log(`✅ Keystroke avg match: ${profile.timing.keystroke_interval_avg === restored.timing.keystroke_interval_avg}`);
console.log('\n');

// =============================================================================
// SUMMARY
// =============================================================================

console.log('=' .repeat(80));
console.log('\n');
console.log('📊 DEMO SUMMARY\n');
console.log(`✅ Baseline Profile Built:`);
console.log(`   - ${profile.samples_analyzed} typing samples analyzed`);
console.log(`   - ${(profile.confidence * 100).toFixed(0)}% confidence`);
console.log(`   - ${profile.timing.keystroke_interval_avg.toFixed(2)}ms avg keystroke interval`);
console.log('');
console.log(`✅ Anomaly Detection:`);
console.log(`   - Normal typing: ${normalAnomaly.alert ? 'ALERT' : 'PASSED ✓'}`);
console.log(`   - Rushed typing (duress): ${rushedAnomaly.alert ? 'ALERT ✓' : 'PASSED'} (score: ${rushedAnomaly.score.toFixed(2)})`);
console.log(`   - Paste attack: ${pasteAnomaly.alert ? 'ALERT ✓' : 'PASSED'} (score: ${pasteAnomaly.score.toFixed(2)})`);
console.log('');
console.log(`🚨 Duress Detection:`);
console.log(`   - Rushed typing detected: ${rushedDuress.duress_detected ? 'YES ✓' : 'NO'} (confidence: ${(rushedDuress.confidence * 100).toFixed(0)}%)`);
console.log('');
console.log(`✅ Serialization:`);
console.log(`   - Profile can be saved and restored ✓`);
console.log('');
console.log('⌨️  TYPING FINGERPRINTING: WORKING!');
console.log('');
console.log('=' .repeat(80));
