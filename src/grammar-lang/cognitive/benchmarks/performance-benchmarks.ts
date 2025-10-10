/**
 * Performance Benchmarks
 * Comprehensive benchmarking suite for Cognitive OS:
 * 1. Detection Speed (<0.5ms target)
 * 2. Accuracy (>95% precision target)
 * 3. False Positives (<1% target)
 */

import { detectManipulation } from '../detector/pattern-matcher';
import { getAllTechniques } from '../techniques';

// ============================================================
// BENCHMARK 1: DETECTION SPEED
// ============================================================

export interface SpeedBenchmarkResult {
  technique_name: string;
  avg_time_ms: number;
  min_time_ms: number;
  max_time_ms: number;
  p50_ms: number;
  p95_ms: number;
  p99_ms: number;
  meets_target: boolean; // <0.5ms
}

/**
 * Benchmark detection speed
 * Target: <0.5ms per technique
 */
export async function benchmarkDetectionSpeed(
  iterations: number = 1000
): Promise<{
  summary: {
    total_techniques: number;
    avg_time_ms: number;
    techniques_meeting_target: number;
    target_achievement_rate: number;
  };
  by_technique: SpeedBenchmarkResult[];
}> {
  const techniques = getAllTechniques();
  const results: SpeedBenchmarkResult[] = [];

  // Test samples (short, medium, long)
  const testSamples = [
    "You're overreacting. That never happened.",
    "I never said that. You're just imagining things again. You're too sensitive.",
    "You're being ridiculous. Everyone thinks you're crazy. I can't believe you're bringing this up again. You need to calm down and think about what you're saying."
  ];

  for (const technique of techniques) {
    const times: number[] = [];

    // Run multiple iterations
    for (let i = 0; i < iterations; i++) {
      const sample = testSamples[i % testSamples.length];
      const startTime = performance.now();

      await detectManipulation(sample, {
        technique_ids: [technique.id]
      });

      const endTime = performance.now();
      times.push(endTime - startTime);
    }

    times.sort((a, b) => a - b);

    const avg = times.reduce((a, b) => a + b, 0) / times.length;
    const p50 = times[Math.floor(times.length * 0.5)];
    const p95 = times[Math.floor(times.length * 0.95)];
    const p99 = times[Math.floor(times.length * 0.99)];

    results.push({
      technique_name: technique.name,
      avg_time_ms: avg,
      min_time_ms: times[0],
      max_time_ms: times[times.length - 1],
      p50_ms: p50,
      p95_ms: p95,
      p99_ms: p99,
      meets_target: p95 < 0.5 // 95th percentile should be under 0.5ms
    });
  }

  const techniquesMeetingTarget = results.filter(r => r.meets_target).length;
  const avgTime = results.reduce((sum, r) => sum + r.avg_time_ms, 0) / results.length;

  return {
    summary: {
      total_techniques: techniques.length,
      avg_time_ms: avgTime,
      techniques_meeting_target: techniquesMeetingTarget,
      target_achievement_rate: techniquesMeetingTarget / techniques.length
    },
    by_technique: results
  };
}

// ============================================================
// BENCHMARK 2: ACCURACY
// ============================================================

export interface AccuracyTestCase {
  text: string;
  expected_technique: string;
  expected_category: string;
  is_manipulative: boolean;
}

export interface AccuracyBenchmarkResult {
  total_cases: number;
  correct_detections: number;
  false_positives: number;
  false_negatives: number;
  precision: number; // TP / (TP + FP)
  recall: number;    // TP / (TP + FN)
  f1_score: number;
  accuracy: number;  // (TP + TN) / Total
  meets_target: boolean; // >95% precision
}

/**
 * Benchmark detection accuracy
 * Target: >95% precision
 */
export async function benchmarkAccuracy(
  testCases?: AccuracyTestCase[]
): Promise<AccuracyBenchmarkResult> {
  // Default test cases if none provided
  const defaultTestCases: AccuracyTestCase[] = [
    // Gaslighting examples
    {
      text: "That never happened. You're making things up again.",
      expected_technique: "gaslighting",
      expected_category: "gaslighting",
      is_manipulative: true
    },
    {
      text: "You're too sensitive. Nobody else has a problem with this.",
      expected_technique: "gaslighting",
      expected_category: "gaslighting",
      is_manipulative: true
    },

    // Non-manipulative examples
    {
      text: "I don't remember that happening. Can you help me understand?",
      expected_technique: "",
      expected_category: "",
      is_manipulative: false
    },
    {
      text: "Let's have a calm discussion about this.",
      expected_technique: "",
      expected_category: "",
      is_manipulative: false
    },

    // DARVO examples
    {
      text: "I'm the victim here! You're the one attacking me!",
      expected_technique: "darvo",
      expected_category: "darvo",
      is_manipulative: true
    },

    // Triangulation examples
    {
      text: "Unlike you, they actually understand me.",
      expected_technique: "triangulation",
      expected_category: "triangulation",
      is_manipulative: true
    }
  ];

  const cases = testCases || defaultTestCases;

  let truePositives = 0;
  let falsePositives = 0;
  let trueNegatives = 0;
  let falseNegatives = 0;

  for (const testCase of cases) {
    const result = await detectManipulation(testCase.text, {
      min_confidence: 0.7
    });

    const detected = result.total_matches > 0;

    if (testCase.is_manipulative) {
      if (detected) {
        truePositives++;
      } else {
        falseNegatives++;
      }
    } else {
      if (detected) {
        falsePositives++;
      } else {
        trueNegatives++;
      }
    }
  }

  const precision = truePositives / (truePositives + falsePositives) || 0;
  const recall = truePositives / (truePositives + falseNegatives) || 0;
  const f1 = 2 * (precision * recall) / (precision + recall) || 0;
  const accuracy = (truePositives + trueNegatives) / cases.length;

  return {
    total_cases: cases.length,
    correct_detections: truePositives,
    false_positives: falsePositives,
    false_negatives: falseNegatives,
    precision,
    recall,
    f1_score: f1,
    accuracy,
    meets_target: precision >= 0.95
  };
}

// ============================================================
// BENCHMARK 3: FALSE POSITIVES
// ============================================================

export interface FalsePositiveBenchmarkResult {
  total_benign_samples: number;
  false_positives: number;
  false_positive_rate: number;
  meets_target: boolean; // <1% FPR
  neurodivergent_samples_tested: number;
  neurodivergent_false_positives: number;
  neurodivergent_fpr: number;
}

/**
 * Benchmark false positive rate
 * Target: <1% false positives
 */
export async function benchmarkFalsePositives(): Promise<FalsePositiveBenchmarkResult> {
  // Benign communication samples (should NOT trigger detection)
  const benignSamples = [
    "I think we should discuss this calmly.",
    "Can you help me understand your perspective?",
    "I appreciate your honesty about this.",
    "Let's work together to find a solution.",
    "I'm sorry, I don't remember. Can you remind me?",
    "That's interesting. Tell me more.",
    "I value your opinion on this.",
    "We might see things differently, and that's okay.",
    "I need some time to process this information.",
    "Thank you for sharing that with me."
  ];

  // Neurodivergent communication patterns (should NOT trigger with protection)
  const neurodivergentSamples = [
    "Actually, to be precise, that's not quite accurate.",
    "I don't understand. Can you be more literal?",
    "Wait, what were we talking about? I got distracted.",
    "I forgot about that. Sorry, my memory is not great.",
    "I prefer direct communication without subtext.",
    "I'm being literal here, not sarcastic."
  ];

  let falsePositives = 0;
  let neurodivergentFalsePositives = 0;

  // Test benign samples
  for (const sample of benignSamples) {
    const result = await detectManipulation(sample, {
      min_confidence: 0.8
    });

    if (result.total_matches > 0) {
      falsePositives++;
    }
  }

  // Test neurodivergent samples (with protection enabled)
  for (const sample of neurodivergentSamples) {
    const result = await detectManipulation(sample, {
      min_confidence: 0.8,
      enable_neurodivergent_protection: true
    });

    if (result.total_matches > 0) {
      neurodivergentFalsePositives++;
    }
  }

  const totalBenign = benignSamples.length;
  const fpr = falsePositives / totalBenign;
  const neurodivergentFpr = neurodivergentFalsePositives / neurodivergentSamples.length;

  return {
    total_benign_samples: totalBenign,
    false_positives: falsePositives,
    false_positive_rate: fpr,
    meets_target: fpr < 0.01,
    neurodivergent_samples_tested: neurodivergentSamples.length,
    neurodivergent_false_positives: neurodivergentFalsePositives,
    neurodivergent_fpr: neurodivergentFpr
  };
}

// ============================================================
// COMPREHENSIVE BENCHMARK SUITE
// ============================================================

export interface ComprehensiveBenchmarkResult {
  speed: Awaited<ReturnType<typeof benchmarkDetectionSpeed>>;
  accuracy: AccuracyBenchmarkResult;
  false_positives: FalsePositiveBenchmarkResult;
  overall_pass: boolean;
  timestamp: string;
}

/**
 * Run all benchmarks
 */
export async function runAllBenchmarks(
  speedIterations: number = 100
): Promise<ComprehensiveBenchmarkResult> {
  console.log('🏁 Running comprehensive benchmark suite...\n');

  console.log('⏱️  Benchmark 1: Detection Speed');
  const speed = await benchmarkDetectionSpeed(speedIterations);
  console.log(`   Average time: ${speed.summary.avg_time_ms.toFixed(3)}ms`);
  console.log(`   Meeting target: ${speed.summary.techniques_meeting_target}/${speed.summary.total_techniques} techniques\n`);

  console.log('🎯 Benchmark 2: Accuracy');
  const accuracy = await benchmarkAccuracy();
  console.log(`   Precision: ${(accuracy.precision * 100).toFixed(1)}%`);
  console.log(`   Recall: ${(accuracy.recall * 100).toFixed(1)}%`);
  console.log(`   F1 Score: ${(accuracy.f1_score * 100).toFixed(1)}%\n`);

  console.log('🛡️  Benchmark 3: False Positives');
  const falsePositives = await benchmarkFalsePositives();
  console.log(`   FPR: ${(falsePositives.false_positive_rate * 100).toFixed(2)}%`);
  console.log(`   Neurodivergent FPR: ${(falsePositives.neurodivergent_fpr * 100).toFixed(2)}%\n`);

  const overallPass =
    speed.summary.target_achievement_rate >= 0.8 && // 80% of techniques meet speed target
    accuracy.meets_target &&                        // >95% precision
    falsePositives.meets_target;                    // <1% FPR

  console.log(`\n${overallPass ? '✅ PASS' : '❌ FAIL'} - Overall benchmark result\n`);

  return {
    speed,
    accuracy,
    false_positives: falsePositives,
    overall_pass: overallPass,
    timestamp: new Date().toISOString()
  };
}
