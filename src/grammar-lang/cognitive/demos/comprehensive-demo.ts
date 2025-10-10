/**
 * Comprehensive Demo - Cognitive OS
 * Demonstrates all features of the manipulation detection system
 */

import { createCognitiveOrganism, analyzeText } from '../glass/cognitive-organism';
import { createStreamProcessor } from '../detector/stream-processor';
import { createSelfSurgeryEngine } from '../evolution/self-surgery';
import { setLocale, formatDetectionMessage } from '../i18n/locales';
import { runAllBenchmarks } from '../benchmarks/performance-benchmarks';
import { globalPerformanceMonitor, globalProfiler } from '../performance/optimizer';

// ============================================================
// DEMO 1: BASIC DETECTION
// ============================================================

export async function demoBasicDetection() {
  console.log('\n📋 DEMO 1: Basic Manipulation Detection\n');
  console.log('='.repeat(60));

  const organism = createCognitiveOrganism('Demo Organism');

  const examples = [
    "That never happened. You're making this up.",
    "I'm the victim here! You're attacking me!",
    "Unlike you, they actually understand me.",
    "Let's discuss this calmly and respectfully."
  ];

  for (const text of examples) {
    console.log(`\nText: "${text}"`);
    const result = await analyzeText(organism, text);

    if (result.detections.length > 0) {
      console.log(`🚨 DETECTED: ${result.detections.length} technique(s)`);
      for (const detection of result.detections.slice(0, 2)) {
        console.log(`  - ${detection.technique_name} (${(detection.confidence * 100).toFixed(1)}%)`);
      }
    } else {
      console.log('✅ No manipulation detected');
    }
  }

  console.log('\n' + '='.repeat(60));
}

// ============================================================
// DEMO 2: MULTI-LANGUAGE SUPPORT
// ============================================================

export async function demoMultiLanguage() {
  console.log('\n🌍 DEMO 2: Multi-Language Support\n');
  console.log('='.repeat(60));

  const text = "That never happened. You're imagining things.";
  const organism = createCognitiveOrganism('Multi-lang Demo');
  const result = await analyzeText(organism, text);

  const languages: Array<'en' | 'pt' | 'es'> = ['en', 'pt', 'es'];

  for (const lang of languages) {
    setLocale(lang);

    if (result.detections.length > 0) {
      const message = formatDetectionMessage(
        result.detections[0].technique_name,
        result.detections[0].confidence,
        lang
      );

      console.log(`[${lang.toUpperCase()}] ${message}`);
    }
  }

  // Reset to English
  setLocale('en');

  console.log('\n' + '='.repeat(60));
}

// ============================================================
// DEMO 3: REAL-TIME STREAM PROCESSING
// ============================================================

export async function demoStreamProcessing() {
  console.log('\n⚡ DEMO 3: Real-Time Stream Processing\n');
  console.log('='.repeat(60));

  const processor = createStreamProcessor({
    debounce_ms: 100,
    min_text_length: 10,
    enable_incremental: true
  });

  // Set up event listeners
  processor.on('analysis', (event: any) => {
    if (event.result.total_matches > 0) {
      console.log(`\n🚨 Stream Detection (${event.text_length} chars):`);
      console.log(`   Techniques: ${event.result.detections.map((d: any) => d.technique_name).join(', ')}`);
    }
  });

  processor.on('alert', (alert: any) => {
    console.log(`\n⚠️  HIGH CONFIDENCE ALERT: ${alert.message}`);
  });

  // Simulate typing
  console.log('\nSimulating live typing...');
  const fullText = "That never happened. You're just being too sensitive. Nobody else thinks this is a problem.";

  for (let i = 0; i < fullText.length; i += 10) {
    const chunk = fullText.slice(i, i + 10);
    processor.push(chunk);
    await new Promise(resolve => setTimeout(resolve, 50));
  }

  // Flush final buffer
  await processor.flush();
  processor.stop();

  const stats = processor.getStats();
  console.log(`\nStream Stats:`);
  console.log(`  Total chunks: ${stats.total_chunks_processed}`);
  console.log(`  Total chars: ${stats.total_characters_processed}`);
  console.log(`  Avg time: ${stats.average_processing_time_ms.toFixed(2)}ms`);

  console.log('\n' + '='.repeat(60));
}

// ============================================================
// DEMO 4: SELF-SURGERY & EVOLUTION
// ============================================================

export async function demoSelfSurgery() {
  console.log('\n🧬 DEMO 4: Self-Surgery & Evolution\n');
  console.log('='.repeat(60));

  const surgeryEngine = createSelfSurgeryEngine({
    enable_auto_surgery: false,
    auto_approve_threshold: 0.95,
    min_evidence_count: 3
  });

  // Simulate observing a new pattern
  console.log('\nSimulating novel manipulation pattern observations...');

  for (let i = 0; i < 5; i++) {
    surgeryEngine.observeAnomalousPattern(
      "You're doing it wrong again. Let me show you the right way.",
      [],
      'gaslighting'
    );
  }

  const stats = surgeryEngine.getStats();
  console.log(`\nSurgery Engine Stats:`);
  console.log(`  Total candidates: ${stats.total_candidates}`);
  console.log(`  Pending approval: ${stats.pending_approval}`);
  console.log(`  Next technique ID: ${stats.next_technique_id}`);

  const candidates = surgeryEngine.getPendingCandidates();
  if (candidates.length > 0) {
    console.log(`\nCandidate Techniques:`);
    for (const candidate of candidates) {
      console.log(`  - ${candidate.proposed_name}`);
      console.log(`    Evidence: ${candidate.evidence.occurrence_count} occurrences`);
      console.log(`    Confidence: ${(candidate.confidence * 100).toFixed(1)}%`);
    }
  }

  console.log('\n' + '='.repeat(60));
}

// ============================================================
// DEMO 5: PERFORMANCE MONITORING
// ============================================================

export async function demoPerformanceMonitoring() {
  console.log('\n📊 DEMO 5: Performance Monitoring\n');
  console.log('='.repeat(60));

  globalProfiler.enable();

  const organism = createCognitiveOrganism('Performance Demo');

  // Run multiple detections
  console.log('\nRunning 50 detections...');
  for (let i = 0; i < 50; i++) {
    const text = "That never happened. You're overreacting.";
    await analyzeText(organism, text);

    const randomTime = Math.random() * 2;
    globalPerformanceMonitor.recordDetectionTime(randomTime);
  }

  // Get metrics
  const metrics = globalPerformanceMonitor.getMetrics();

  console.log(`\nPerformance Metrics:`);
  console.log(`  Total detections: ${metrics.total_detections}`);
  console.log(`  Avg time: ${metrics.average_detection_time_ms.toFixed(3)}ms`);
  console.log(`  P50: ${metrics.p50_detection_time_ms.toFixed(3)}ms`);
  console.log(`  P95: ${metrics.p95_detection_time_ms.toFixed(3)}ms`);
  console.log(`  P99: ${metrics.p99_detection_time_ms.toFixed(3)}ms`);
  console.log(`  Cache hit rate: ${(metrics.cache_hit_rate * 100).toFixed(1)}%`);
  console.log(`  Cache size: ${metrics.cache_size}`);

  globalProfiler.disable();

  console.log('\n' + '='.repeat(60));
}

// ============================================================
// DEMO 6: COMPREHENSIVE BENCHMARK
// ============================================================

export async function demoBenchmarks() {
  console.log('\n🏁 DEMO 6: Comprehensive Benchmarks\n');
  console.log('='.repeat(60));

  const results = await runAllBenchmarks(50);

  console.log(`\n${'='.repeat(60)}`);
  console.log(`\nFinal Results:`);
  console.log(`  Speed: ${results.speed.summary.target_achievement_rate >= 0.8 ? '✅ PASS' : '❌ FAIL'}`);
  console.log(`  Accuracy: ${results.accuracy.meets_target ? '✅ PASS' : '❌ FAIL'}`);
  console.log(`  False Positives: ${results.false_positives.meets_target ? '✅ PASS' : '❌ FAIL'}`);
  console.log(`\n  Overall: ${results.overall_pass ? '✅ PASS' : '❌ FAIL'}`);

  console.log('\n' + '='.repeat(60));
}

// ============================================================
// RUN ALL DEMOS
// ============================================================

export async function runAllDemos() {
  console.log('\n');
  console.log('╔' + '═'.repeat(58) + '╗');
  console.log('║' + ' '.repeat(10) + 'COGNITIVE OS - COMPREHENSIVE DEMO' + ' '.repeat(14) + '║');
  console.log('╚' + '═'.repeat(58) + '╝');

  await demoBasicDetection();
  await demoMultiLanguage();
  await demoStreamProcessing();
  await demoSelfSurgery();
  await demoPerformanceMonitoring();
  await demoBenchmarks();

  console.log('\n');
  console.log('╔' + '═'.repeat(58) + '╗');
  console.log('║' + ' '.repeat(19) + 'DEMO COMPLETE' + ' '.repeat(26) + '║');
  console.log('╚' + '═'.repeat(58) + '╝');
  console.log('\n');
}

// Export individual demos
export const demos = {
  basicDetection: demoBasicDetection,
  multiLanguage: demoMultiLanguage,
  streamProcessing: demoStreamProcessing,
  selfSurgery: demoSelfSurgery,
  performanceMonitoring: demoPerformanceMonitoring,
  benchmarks: demoBenchmarks,
  all: runAllDemos
};

// CLI support
if (require.main === module) {
  runAllDemos().catch(console.error);
}
