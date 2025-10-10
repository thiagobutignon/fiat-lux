/**
 * Typing Anomaly Detection
 *
 * Detects deviations from established typing patterns
 * Focus on duress detection through typing behavior changes
 */

import { TypingProfile, Interaction, TypingAnomaly } from './types';

// ============================================================================
// TYPING ANOMALY DETECTOR
// ============================================================================

export class TypingAnomalyDetector {
  /**
   * Detect typing anomalies in interaction
   * Compares current typing behavior against established profile
   */
  static detectTypingAnomaly(
    profile: TypingProfile,
    interaction: Interaction
  ): TypingAnomaly {
    // Minimum confidence required (need sufficient baseline)
    if (profile.confidence < 0.3) {
      return {
        score: 0,
        threshold: 0.7,
        alert: false,
        details: {
          speed_deviation: 0,
          error_rate_change: 0,
          pause_pattern_change: 0,
          input_burst: false,
        },
        specific_anomalies: ['Insufficient baseline data - building typing profile'],
      };
    }

    // If no typing data in current interaction, can't detect
    if (!interaction.typing_data) {
      return {
        score: 0,
        threshold: 0.7,
        alert: false,
        details: {
          speed_deviation: 0,
          error_rate_change: 0,
          pause_pattern_change: 0,
          input_burst: false,
        },
        specific_anomalies: ['No typing data available in current interaction'],
      };
    }

    // Analyze current typing behavior
    const currentBehavior = this.analyzeTypingBehavior(interaction.typing_data);

    // Calculate deviations
    const speedDev = this.calculateSpeedDeviation(profile, currentBehavior);
    const errorDev = this.calculateErrorDeviation(profile, currentBehavior);
    const pauseDev = this.calculatePauseDeviation(profile, currentBehavior);
    const burstDetected = this.detectInputBurst(currentBehavior, interaction.text_length);

    // Weighted anomaly score
    const anomalyScore =
      speedDev * 0.35 +          // Speed changes are strong duress indicator
      errorDev * 0.30 +           // Error rate changes indicate stress
      pauseDev * 0.25 +           // Unusual pauses suggest hesitation/coercion
      (burstDetected ? 0.5 : 0);  // Input burst is VERY suspicious (paste attack)

    const threshold = 0.7;
    const alert = anomalyScore > threshold;

    // Identify specific anomalies
    const specificAnomalies: string[] = [];

    if (speedDev > 0.6) {
      const faster = currentBehavior.avg_interval < profile.timing.keystroke_interval_avg;
      specificAnomalies.push(
        faster
          ? 'Typing significantly faster than usual (possible duress/rush)'
          : 'Typing significantly slower than usual (possible hesitation/coercion)'
      );
    }

    if (errorDev > 0.6) {
      const moreErrors = currentBehavior.error_rate > profile.errors.typo_rate;
      specificAnomalies.push(
        moreErrors
          ? 'Error rate significantly higher (possible stress/duress)'
          : 'Error rate unusually low (possible careful/forced input)'
      );
    }

    if (pauseDev > 0.6) {
      specificAnomalies.push('Unusual pause patterns detected (possible hesitation under pressure)');
    }

    if (burstDetected) {
      specificAnomalies.push('Input burst detected (possible paste attack or impersonation)');
    }

    return {
      score: Math.min(anomalyScore, 1.0), // Cap at 1.0
      threshold,
      alert,
      details: {
        speed_deviation: speedDev,
        error_rate_change: errorDev,
        pause_pattern_change: pauseDev,
        input_burst: burstDetected,
      },
      specific_anomalies: specificAnomalies,
    };
  }

  // ==========================================================================
  // DEVIATION CALCULATIONS
  // ==========================================================================

  /**
   * Calculate speed deviation
   * Measures typing speed changes (faster = rushed, slower = hesitant)
   */
  private static calculateSpeedDeviation(
    profile: TypingProfile,
    current: TypingBehavior
  ): number {
    const baselineInterval = profile.timing.keystroke_interval_avg;
    const currentInterval = current.avg_interval;

    // Calculate percentage difference
    const diff = Math.abs(currentInterval - baselineInterval);
    const percentChange = diff / baselineInterval;

    // Normalize to 0-1 (clip at 2x change = 1.0)
    return Math.min(percentChange / 2, 1.0);
  }

  /**
   * Calculate error rate deviation
   * More errors = stress, fewer errors = careful/forced
   */
  private static calculateErrorDeviation(
    profile: TypingProfile,
    current: TypingBehavior
  ): number {
    const baselineErrorRate = profile.errors.typo_rate;
    const currentErrorRate = current.error_rate;

    // If baseline is 0, use small epsilon
    const baseline = baselineErrorRate || 0.01;

    // Calculate ratio
    const ratio = Math.abs(currentErrorRate - baseline) / baseline;

    // Normalize to 0-1 (clip at 3x change = 1.0)
    return Math.min(ratio / 3, 1.0);
  }

  /**
   * Calculate pause pattern deviation
   * Unusual pauses suggest hesitation or thinking under pressure
   */
  private static calculatePauseDeviation(
    profile: TypingProfile,
    current: TypingBehavior
  ): number {
    // Compare average pause duration
    const baselinePause = profile.timing.thinking_pause_duration || 500;
    const currentPause = current.avg_pause;

    const diff = Math.abs(currentPause - baselinePause);
    const percentChange = diff / baselinePause;

    // Normalize to 0-1
    return Math.min(percentChange / 2, 1.0);
  }

  /**
   * Detect input burst (paste attack)
   * Very fast typing + long text = likely paste
   */
  private static detectInputBurst(
    behavior: TypingBehavior,
    textLength: number
  ): boolean {
    // If average interval < 10ms and text > 50 chars, it's a paste
    return behavior.avg_interval < 10 && textLength > 50;
  }

  // ==========================================================================
  // TYPING BEHAVIOR ANALYSIS
  // ==========================================================================

  /**
   * Analyze current typing behavior
   */
  private static analyzeTypingBehavior(
    typingData: NonNullable<Interaction['typing_data']>
  ): TypingBehavior {
    const { keystroke_intervals, pauses, backspaces, corrections, total_typing_time } = typingData;

    // Calculate average keystroke interval
    const avgInterval = keystroke_intervals.length > 0
      ? keystroke_intervals.reduce((a, b) => a + b, 0) / keystroke_intervals.length
      : 0;

    // Calculate average pause duration
    const avgPause = pauses.length > 0
      ? pauses.reduce((a, b) => a + b, 0) / pauses.length
      : 0;

    // Calculate error rate (errors per 100 chars)
    const errorRate = total_typing_time > 0
      ? (corrections / total_typing_time) * 100
      : 0;

    // Calculate backspace frequency
    const backspaceFreq = backspaces;

    return {
      avg_interval: avgInterval,
      avg_pause: avgPause,
      error_rate: errorRate,
      backspace_frequency: backspaceFreq,
      total_corrections: corrections,
    };
  }

  /**
   * Detect duress through typing patterns
   * Combines multiple signals for high-confidence detection
   */
  static detectDuressFromTyping(
    profile: TypingProfile,
    interaction: Interaction
  ): {
    duress_detected: boolean;
    confidence: number;
    indicators: string[];
  } {
    const anomaly = this.detectTypingAnomaly(profile, interaction);

    // Duress indicators
    const indicators: string[] = [];
    let duressScore = 0;

    // Check for rush (very fast typing)
    if (interaction.typing_data) {
      const currentSpeed = interaction.typing_data.keystroke_intervals.reduce((a, b) => a + b, 0) /
        interaction.typing_data.keystroke_intervals.length;
      const baselineSpeed = profile.timing.keystroke_interval_avg;

      if (currentSpeed < baselineSpeed * 0.5) {
        // Typing 2x faster
        indicators.push('Typing significantly faster (rushed under duress)');
        duressScore += 0.4;
      }
    }

    // Check for high error rate (stress)
    if (anomaly.details.error_rate_change > 0.7) {
      indicators.push('Error rate very high (stress/duress indicator)');
      duressScore += 0.3;
    }

    // Check for unusual pauses (hesitation)
    if (anomaly.details.pause_pattern_change > 0.7) {
      indicators.push('Unusual pause patterns (hesitation under coercion)');
      duressScore += 0.2;
    }

    // Check for input burst (impersonation)
    if (anomaly.details.input_burst) {
      indicators.push('Input burst detected (possible impersonation/paste attack)');
      duressScore += 0.5; // Very suspicious
    }

    const duressDetected = duressScore > 0.6; // 60% confidence threshold
    const confidence = Math.min(duressScore, 1.0);

    return {
      duress_detected: duressDetected,
      confidence,
      indicators,
    };
  }
}

// ============================================================================
// TYPES
// ============================================================================

interface TypingBehavior {
  avg_interval: number;          // Average keystroke interval (ms)
  avg_pause: number;              // Average pause duration (ms)
  error_rate: number;             // Errors per 100 characters
  backspace_frequency: number;    // Number of backspaces
  total_corrections: number;      // Total corrections made
}
