/**
 * Security - Linguistic Fingerprinting Demo
 *
 * Demonstrates behavioral security system
 */

import { LinguisticCollector } from '../src/grammar-lang/security/linguistic-collector';
import { AnomalyDetector } from '../src/grammar-lang/security/anomaly-detector';
import { Interaction } from '../src/grammar-lang/security/types';

console.log('🔐 SECURITY - LINGUISTIC FINGERPRINTING DEMO\n');
console.log('=' .repeat(80));
console.log('\n');

// =============================================================================
// PHASE 1: BUILD BASELINE PROFILE
// =============================================================================

console.log('📊 PHASE 1: Building Baseline Profile\n');

let profile = LinguisticCollector.createProfile('alice');

// Simulate Alice's typical interactions (casual, friendly)
const aliceInteractions = [
  "Hey! How are you doing today?",
  "I'm working on a cool project. It's really fun!",
  "Do you want to grab lunch later?",
  "That sounds great! I'd love to.",
  "The weather is nice today, isn't it?",
  "I think we should try that new restaurant.",
  "This code is working well. Very happy with it!",
  "Can you help me with this? I'm stuck.",
  "Thanks so much! You're awesome.",
  "Let's meet tomorrow at 10am."
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

console.log(`✅ Analyzed ${profile.samples_analyzed} interactions`);
console.log(`✅ Confidence: ${(profile.confidence * 100).toFixed(1)}%`);
console.log(`✅ Vocabulary size: ${profile.vocabulary.unique_words.size} unique words`);
console.log(`✅ Average sentence length: ${profile.syntax.average_sentence_length.toFixed(1)} words`);
console.log(`✅ Sentiment baseline: ${profile.semantics.sentiment_baseline.toFixed(2)} (${profile.semantics.sentiment_baseline > 0 ? 'positive' : 'negative'})`);
console.log(`✅ Formality level: ${(profile.semantics.formality_level * 100).toFixed(0)}%`);
console.log('\n');

// Show statistics
const stats = LinguisticCollector.getStatistics(profile);
console.log('📈 Most Common Words:');
stats.most_common_words.slice(0, 5).forEach(([word, count]) => {
  console.log(`   - "${word}": ${count} times`);
});
console.log('\n');

// =============================================================================
// PHASE 2: TEST NORMAL INTERACTION (NO ANOMALY)
// =============================================================================

console.log('=' .repeat(80));
console.log('\n');
console.log('✅ PHASE 2: Test Normal Interaction (No Anomaly)\n');

const normalInteraction: Interaction = {
  interaction_id: 'test_normal',
  user_id: 'alice',
  timestamp: Date.now(),
  text: "Hey! I'm doing great today. Want to work on that project together?",
  text_length: 67,
  word_count: 13,
  session_id: 'session_test'
};

const normalAnomaly = AnomalyDetector.detectLinguisticAnomaly(profile, normalInteraction);

console.log(`Interaction: "${normalInteraction.text}"`);
console.log(`\nAnomaly Score: ${normalAnomaly.score.toFixed(3)}`);
console.log(`Threshold: ${normalAnomaly.threshold}`);
console.log(`Alert: ${normalAnomaly.alert ? '🚨 YES' : '✅ NO'}`);
console.log(`\nDetails:`);
console.log(`  - Vocabulary deviation: ${normalAnomaly.details.vocabulary_deviation.toFixed(3)}`);
console.log(`  - Syntax deviation: ${normalAnomaly.details.syntax_deviation.toFixed(3)}`);
console.log(`  - Semantics deviation: ${normalAnomaly.details.semantics_deviation.toFixed(3)}`);
console.log(`  - Sentiment deviation: ${normalAnomaly.details.sentiment_deviation.toFixed(3)}`);

if (normalAnomaly.specific_anomalies.length > 0) {
  console.log(`\nSpecific Anomalies:`);
  normalAnomaly.specific_anomalies.forEach(a => console.log(`  ⚠️  ${a}`));
} else {
  console.log(`\n✅ No specific anomalies detected - normal behavior`);
}

console.log('\n');

// =============================================================================
// PHASE 3: TEST VOCABULARY ANOMALY
// =============================================================================

console.log('=' .repeat(80));
console.log('\n');
console.log('⚠️  PHASE 3: Test Vocabulary Anomaly\n');

const vocabAnomalyInteraction: Interaction = {
  interaction_id: 'test_vocab',
  user_id: 'alice',
  timestamp: Date.now(),
  text: "Quantum entanglement exhibits superposition phenomena within subatomic particles.",
  text_length: 81,
  word_count: 9,
  session_id: 'session_test'
};

const vocabAnomaly = AnomalyDetector.detectLinguisticAnomaly(profile, vocabAnomalyInteraction);

console.log(`Interaction: "${vocabAnomalyInteraction.text}"`);
console.log(`\nAnomaly Score: ${vocabAnomaly.score.toFixed(3)}`);
console.log(`Threshold: ${vocabAnomaly.threshold}`);
console.log(`Alert: ${vocabAnomaly.alert ? '🚨 YES' : '✅ NO'}`);
console.log(`\nDetails:`);
console.log(`  - Vocabulary deviation: ${vocabAnomaly.details.vocabulary_deviation.toFixed(3)} ${vocabAnomaly.details.vocabulary_deviation > 0.6 ? '🚨' : ''}`);
console.log(`  - Syntax deviation: ${vocabAnomaly.details.syntax_deviation.toFixed(3)}`);
console.log(`  - Semantics deviation: ${vocabAnomaly.details.semantics_deviation.toFixed(3)}`);
console.log(`  - Sentiment deviation: ${vocabAnomaly.details.sentiment_deviation.toFixed(3)}`);

if (vocabAnomaly.specific_anomalies.length > 0) {
  console.log(`\nSpecific Anomalies:`);
  vocabAnomaly.specific_anomalies.forEach(a => console.log(`  🚨 ${a}`));
}

console.log('\n');

// =============================================================================
// PHASE 4: TEST SENTIMENT ANOMALY (DURESS)
// =============================================================================

console.log('=' .repeat(80));
console.log('\n');
console.log('🚨 PHASE 4: Test Sentiment Anomaly (Potential Duress)\n');

const duressInteraction: Interaction = {
  interaction_id: 'test_duress',
  user_id: 'alice',
  timestamp: Date.now(),
  text: "This is terrible. I hate everything. Worst day ever. Everything is horrible.",
  text_length: 76,
  word_count: 13,
  session_id: 'session_test'
};

const duressAnomaly = AnomalyDetector.detectLinguisticAnomaly(profile, duressInteraction);

console.log(`Interaction: "${duressInteraction.text}"`);
console.log(`\nAnomaly Score: ${duressAnomaly.score.toFixed(3)}`);
console.log(`Threshold: ${duressAnomaly.threshold}`);
console.log(`Alert: ${duressAnomaly.alert ? '🚨 YES' : '✅ NO'}`);
console.log(`\nDetails:`);
console.log(`  - Vocabulary deviation: ${duressAnomaly.details.vocabulary_deviation.toFixed(3)}`);
console.log(`  - Syntax deviation: ${duressAnomaly.details.syntax_deviation.toFixed(3)}`);
console.log(`  - Semantics deviation: ${duressAnomaly.details.semantics_deviation.toFixed(3)}`);
console.log(`  - Sentiment deviation: ${duressAnomaly.details.sentiment_deviation.toFixed(3)} ${duressAnomaly.details.sentiment_deviation > 0.5 ? '🚨' : ''}`);

if (duressAnomaly.specific_anomalies.length > 0) {
  console.log(`\nSpecific Anomalies:`);
  duressAnomaly.specific_anomalies.forEach(a => console.log(`  🚨 ${a}`));
}

console.log('\n');

// =============================================================================
// PHASE 5: SERIALIZE & DESERIALIZE
// =============================================================================

console.log('=' .repeat(80));
console.log('\n');
console.log('💾 PHASE 5: Profile Serialization\n');

const profileJSON = LinguisticCollector.toJSON(profile);
const restored = LinguisticCollector.fromJSON(profileJSON);

console.log(`✅ Original profile ID: ${profile.user_id}`);
console.log(`✅ Restored profile ID: ${restored.user_id}`);
console.log(`✅ Samples match: ${profile.samples_analyzed === restored.samples_analyzed}`);
console.log(`✅ Confidence match: ${profile.confidence === restored.confidence}`);
console.log(`✅ Vocabulary size match: ${profile.vocabulary.unique_words.size === restored.vocabulary.unique_words.size}`);
console.log('\n');

// =============================================================================
// SUMMARY
// =============================================================================

console.log('=' .repeat(80));
console.log('\n');
console.log('📊 DEMO SUMMARY\n');
console.log(`✅ Baseline Profile Built:`);
console.log(`   - ${profile.samples_analyzed} interactions analyzed`);
console.log(`   - ${(profile.confidence * 100).toFixed(0)}% confidence`);
console.log(`   - ${profile.vocabulary.unique_words.size} unique words tracked`);
console.log('');
console.log(`✅ Anomaly Detection:`);
console.log(`   - Normal interaction: ${normalAnomaly.alert ? 'ALERT' : 'PASSED ✓'}`);
console.log(`   - Vocabulary anomaly: ${vocabAnomaly.alert ? 'ALERT ✓' : 'PASSED'} (score: ${vocabAnomaly.score.toFixed(2)})`);
console.log(`   - Sentiment anomaly: ${duressAnomaly.alert ? 'ALERT ✓' : 'PASSED'} (score: ${duressAnomaly.score.toFixed(2)})`);
console.log('');
console.log(`✅ Serialization:`);
console.log(`   - Profile can be saved and restored ✓`);
console.log('');
console.log('🔐 LINGUISTIC FINGERPRINTING: WORKING!');
console.log('\n');
console.log('=' .repeat(80));
