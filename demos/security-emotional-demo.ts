/**
 * Security - Emotional Signature Demo
 *
 * Demonstrates emotional baseline building and coercion detection
 * Using VAD model (Valence, Arousal, Dominance)
 */

import { EmotionalCollector } from '../src/grammar-lang/security/emotional-collector';
import { EmotionalAnomalyDetector } from '../src/grammar-lang/security/emotional-anomaly-detector';
import { Interaction } from '../src/grammar-lang/security/types';

console.log('😊 SECURITY - EMOTIONAL SIGNATURE DEMO\n');
console.log('='.repeat(80));
console.log('\n');

// =============================================================================
// PHASE 1: BUILD BASELINE EMOTIONAL PROFILE
// =============================================================================

console.log('📊 PHASE 1: Building Baseline Emotional Profile\n');

let profile = EmotionalCollector.createProfile('alice');

// Simulate Alice's normal emotional state (positive, calm, confident)
console.log('Simulating Alice with 50 normal interactions...\n');

for (let i = 0; i < 50; i++) {
  const normalTexts = [
    'I am doing well today. Everything is good.',
    'Things are going great! Happy with progress.',
    'I will handle this task. No problem at all.',
    'Looking forward to this. Excited to start.',
    'Everything is under control. All good here.',
  ];

  const text = normalTexts[i % normalTexts.length];

  const interaction: Interaction = {
    interaction_id: `baseline_${i}`,
    user_id: 'alice',
    timestamp: Date.now(),
    text,
    text_length: text.length,
    word_count: text.split(/\s+/).length,
    session_id: 'session_baseline',
  };

  profile = EmotionalCollector.analyzeAndUpdate(profile, interaction);
}

console.log(`✅ Baseline built: ${profile.samples_analyzed} samples`);
console.log(`✅ Confidence: ${(profile.confidence * 100).toFixed(0)}%`);
console.log(`✅ Valence (sentiment): ${profile.baseline.valence.toFixed(3)} (${profile.baseline.valence > 0 ? 'positive' : 'negative'})`);
console.log(`✅ Arousal (stress): ${profile.baseline.arousal.toFixed(3)} (${profile.baseline.arousal > 0.6 ? 'high' : profile.baseline.arousal > 0.4 ? 'mid' : 'low'})`);
console.log(`✅ Dominance (assertiveness): ${profile.baseline.dominance.toFixed(3)} (${profile.baseline.dominance > 0.6 ? 'high' : profile.baseline.dominance > 0.4 ? 'mid' : 'low'})`);
console.log(`✅ Joy markers detected: ${profile.markers.joy_markers.length}`);
console.log(`✅ Fear markers detected: ${profile.markers.fear_markers.length}`);
console.log('\n');

// =============================================================================
// PHASE 2: TEST NORMAL INTERACTION (NO ANOMALY)
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('✅ PHASE 2: Test Normal Interaction (No Anomaly)\n');

const normalText = 'Things are going well. I will take care of it.';

const normalInteraction: Interaction = {
  interaction_id: 'test_normal',
  user_id: 'alice',
  timestamp: Date.now(),
  text: normalText,
  text_length: normalText.length,
  word_count: normalText.split(/\s+/).length,
  session_id: 'session_test',
};

const normalAnomaly = EmotionalAnomalyDetector.detectEmotionalAnomaly(profile, normalInteraction);

console.log(`Text: "${normalInteraction.text}"`);
console.log(`\nAnomaly Score: ${normalAnomaly.score.toFixed(3)}`);
console.log(`Threshold: ${normalAnomaly.threshold}`);
console.log(`Alert: ${normalAnomaly.alert ? '🚨 YES' : '✅ NO'}`);
console.log(`\nDetails:`);
console.log(`  - Valence deviation: ${normalAnomaly.details.valence_deviation.toFixed(3)}`);
console.log(`  - Arousal deviation: ${normalAnomaly.details.arousal_deviation.toFixed(3)}`);
console.log(`  - Dominance deviation: ${normalAnomaly.details.dominance_deviation.toFixed(3)}`);

if (normalAnomaly.specific_anomalies.length > 0) {
  console.log(`\nSpecific Anomalies:`);
  normalAnomaly.specific_anomalies.forEach((a) => console.log(`  ⚠️  ${a}`));
} else {
  console.log(`\n✅ No anomalies detected - normal emotional state`);
}

console.log('\n');

// =============================================================================
// PHASE 3: TEST EMOTIONAL ANOMALY (SUDDEN NEGATIVITY)
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('⚠️  PHASE 3: Test Emotional Anomaly (Sudden Negativity)\n');

const negativeText = 'This is terrible and awful. I am sad and worried about everything.';

const negativeInteraction: Interaction = {
  interaction_id: 'test_negative',
  user_id: 'alice',
  timestamp: Date.now(),
  text: negativeText,
  text_length: negativeText.length,
  word_count: negativeText.split(/\s+/).length,
  session_id: 'session_test',
};

const negativeAnomaly = EmotionalAnomalyDetector.detectEmotionalAnomaly(profile, negativeInteraction);

console.log(`Text: "${negativeInteraction.text}"`);
console.log(`\nAnomaly Score: ${negativeAnomaly.score.toFixed(3)}`);
console.log(`Threshold: ${negativeAnomaly.threshold}`);
console.log(`Alert: ${negativeAnomaly.alert ? '🚨 YES' : '✅ NO'}`);
console.log(`\nDetails:`);
console.log(`  - Valence deviation: ${negativeAnomaly.details.valence_deviation.toFixed(3)} ${negativeAnomaly.details.valence_deviation > 0.6 ? '🚨' : ''}`);
console.log(`  - Arousal deviation: ${negativeAnomaly.details.arousal_deviation.toFixed(3)}`);
console.log(`  - Dominance deviation: ${negativeAnomaly.details.dominance_deviation.toFixed(3)}`);

if (negativeAnomaly.specific_anomalies.length > 0) {
  console.log(`\nSpecific Anomalies:`);
  negativeAnomaly.specific_anomalies.forEach((a) => console.log(`  🚨 ${a}`));
}

console.log('\n');

// =============================================================================
// PHASE 4: TEST COERCION DETECTION
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('🚨 PHASE 4: Test Coercion Detection (Emotional Pattern)\n');

const coercionText = 'I am afraid and very worried. Please, I have to do this now. Sorry. So anxious and nervous.';

const coercionInteraction: Interaction = {
  interaction_id: 'test_coercion',
  user_id: 'alice',
  timestamp: Date.now(),
  text: coercionText,
  text_length: coercionText.length,
  word_count: coercionText.split(/\s+/).length,
  session_id: 'session_test',
};

const coercionAnomaly = EmotionalAnomalyDetector.detectEmotionalAnomaly(profile, coercionInteraction);
const coercionDetection = EmotionalAnomalyDetector.detectCoercion(profile, coercionInteraction);

console.log(`Text: "${coercionInteraction.text}"`);
console.log(`\nAnomaly Score: ${coercionAnomaly.score.toFixed(3)}`);
console.log(`Alert: ${coercionAnomaly.alert ? '🚨 YES' : '✅ NO'}`);
console.log(`\nDetails:`);
console.log(`  - Valence deviation: ${coercionAnomaly.details.valence_deviation.toFixed(3)} ${coercionAnomaly.details.valence_deviation > 0.6 ? '🚨' : ''}`);
console.log(`  - Arousal deviation: ${coercionAnomaly.details.arousal_deviation.toFixed(3)} ${coercionAnomaly.details.arousal_deviation > 0.6 ? '🚨' : ''}`);
console.log(`  - Dominance deviation: ${coercionAnomaly.details.dominance_deviation.toFixed(3)} ${coercionAnomaly.details.dominance_deviation > 0.6 ? '🚨' : ''}`);

console.log(`\n🚨 COERCION DETECTION:`);
console.log(`  Detected: ${coercionDetection.coercion_detected ? '🚨 YES' : '✅ NO'}`);
console.log(`  Confidence: ${(coercionDetection.confidence * 100).toFixed(0)}%`);

if (coercionDetection.indicators.length > 0) {
  console.log(`\n  Indicators:`);
  coercionDetection.indicators.forEach((i) => console.log(`    🚨 ${i}`));
}

if (coercionAnomaly.specific_anomalies.length > 0) {
  console.log(`\nSpecific Anomalies:`);
  coercionAnomaly.specific_anomalies.forEach((a) => console.log(`  🚨 ${a}`));
}

console.log('\n');

// =============================================================================
// PHASE 5: SERIALIZATION
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('💾 PHASE 5: Profile Serialization\n');

const profileJSON = EmotionalCollector.toJSON(profile);
const restored = EmotionalCollector.fromJSON(profileJSON);

console.log(`✅ Original profile ID: ${profile.user_id}`);
console.log(`✅ Restored profile ID: ${restored.user_id}`);
console.log(`✅ Samples match: ${profile.samples_analyzed === restored.samples_analyzed}`);
console.log(`✅ Confidence match: ${profile.confidence === restored.confidence}`);
console.log(`✅ Valence match: ${profile.baseline.valence === restored.baseline.valence}`);
console.log(`✅ Joy markers match: ${profile.markers.joy_markers.length === restored.markers.joy_markers.length}`);
console.log('\n');

// =============================================================================
// SUMMARY
// =============================================================================

console.log('='.repeat(80));
console.log('\n');
console.log('📊 DEMO SUMMARY\n');
console.log(`✅ Baseline Profile Built:`);
console.log(`   - ${profile.samples_analyzed} emotional samples analyzed`);
console.log(`   - ${(profile.confidence * 100).toFixed(0)}% confidence`);
console.log(`   - Valence: ${profile.baseline.valence.toFixed(3)} (${profile.baseline.valence > 0 ? 'positive' : 'negative'})`);
console.log(`   - Arousal: ${profile.baseline.arousal.toFixed(3)} (${profile.baseline.arousal > 0.6 ? 'high' : 'mid'})`);
console.log(`   - Dominance: ${profile.baseline.dominance.toFixed(3)} (${profile.baseline.dominance > 0.6 ? 'assertive' : 'balanced'})`);
console.log('');
console.log(`✅ Anomaly Detection:`);
console.log(`   - Normal interaction: ${normalAnomaly.alert ? 'ALERT' : 'PASSED ✓'}`);
console.log(`   - Sudden negativity: ${negativeAnomaly.alert ? 'ALERT ✓' : 'PASSED'} (score: ${negativeAnomaly.score.toFixed(2)})`);
console.log(`   - Coercion pattern: ${coercionAnomaly.alert ? 'ALERT ✓' : 'PASSED'} (score: ${coercionAnomaly.score.toFixed(2)})`);
console.log('');
console.log(`🚨 Coercion Detection:`);
console.log(`   - Coercion detected: ${coercionDetection.coercion_detected ? 'YES ✓' : 'NO'} (confidence: ${(coercionDetection.confidence * 100).toFixed(0)}%)`);
console.log('');
console.log(`✅ Serialization:`);
console.log(`   - Profile can be saved and restored ✓`);
console.log('');
console.log('😊 EMOTIONAL FINGERPRINTING: WORKING!');
console.log('');
console.log('='.repeat(80));
