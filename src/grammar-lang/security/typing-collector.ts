/**
 * Typing Pattern Collector
 *
 * Analyzes keystroke timing and error patterns to build behavioral profile
 * Detects duress through typing anomalies (rushed, delayed, burst input)
 */

import { TypingProfile, Interaction, ProfileStatistics } from './types';

// ============================================================================
// TYPING COLLECTOR
// ============================================================================

export class TypingCollector {
  /**
   * Create initial empty typing profile
   */
  static createProfile(userId: string): TypingProfile {
    return {
      user_id: userId,
      created_at: Date.now(),
      last_updated: Date.now(),

      timing: {
        keystroke_intervals: [],
        keystroke_interval_avg: 0,
        keystroke_interval_variance: 0,
        word_pause_duration: 0,
        thinking_pause_duration: 0,
        sentence_pause_duration: 0,
      },

      errors: {
        typo_rate: 0,
        correction_patterns: [],
        backspace_frequency: 0,
        delete_frequency: 0,
        common_typos: new Map(),
      },

      input: {
        copy_paste_frequency: 0,
        input_burst_detected: false,
        edit_distance_avg: 0,
        session_length_avg: 0,
      },

      device: {
        keyboard_layout: 'unknown',
        typical_device_type: 'desktop',
      },

      samples_analyzed: 0,
      confidence: 0,
    };
  }

  /**
   * Analyze interaction and update typing profile
   */
  static analyzeAndUpdate(
    profile: TypingProfile,
    interaction: Interaction
  ): TypingProfile {
    profile.last_updated = Date.now();
    profile.samples_analyzed++;

    // If no typing data available, skip timing analysis
    if (!interaction.typing_data) {
      // Still update confidence (lower weight)
      profile.confidence = Math.min(profile.samples_analyzed / 150, 1.0); // Need more samples
      return profile;
    }

    const { typing_data } = interaction;

    // Update timing patterns
    this.updateTimingPatterns(profile, typing_data);

    // Update error patterns
    this.updateErrorPatterns(profile, typing_data);

    // Update input behavior
    this.updateInputBehavior(profile, typing_data, interaction);

    // Update device fingerprint
    if (interaction.device_type) {
      this.updateDeviceFingerprint(profile, interaction.device_type);
    }

    // Update confidence (more samples = higher confidence)
    profile.confidence = Math.min(profile.samples_analyzed / 100, 1.0); // 100 samples = 100%

    return profile;
  }

  // ==========================================================================
  // TIMING PATTERN ANALYSIS
  // ==========================================================================

  private static updateTimingPatterns(
    profile: TypingProfile,
    typingData: NonNullable<Interaction['typing_data']>
  ): void {
    const { keystroke_intervals, pauses } = typingData;

    // Update keystroke intervals
    if (keystroke_intervals.length > 0) {
      // Add to historical data (keep last 1000)
      profile.timing.keystroke_intervals.push(...keystroke_intervals);
      if (profile.timing.keystroke_intervals.length > 1000) {
        profile.timing.keystroke_intervals = profile.timing.keystroke_intervals.slice(-1000);
      }

      // Recalculate average
      const sum = profile.timing.keystroke_intervals.reduce((a, b) => a + b, 0);
      profile.timing.keystroke_interval_avg = sum / profile.timing.keystroke_intervals.length;

      // Calculate variance
      profile.timing.keystroke_interval_variance = this.calculateVariance(
        profile.timing.keystroke_intervals
      );
    }

    // Update pause durations
    if (pauses.length > 0) {
      // Classify pauses
      const shortPauses = pauses.filter(p => p < 500); // < 500ms = word pause
      const mediumPauses = pauses.filter(p => p >= 500 && p < 2000); // 500-2000ms = thinking
      const longPauses = pauses.filter(p => p >= 2000); // > 2000ms = sentence pause

      // Update running averages
      if (shortPauses.length > 0) {
        const avgShort = shortPauses.reduce((a, b) => a + b, 0) / shortPauses.length;
        profile.timing.word_pause_duration =
          (profile.timing.word_pause_duration * (profile.samples_analyzed - 1) + avgShort) /
          profile.samples_analyzed;
      }

      if (mediumPauses.length > 0) {
        const avgMedium = mediumPauses.reduce((a, b) => a + b, 0) / mediumPauses.length;
        profile.timing.thinking_pause_duration =
          (profile.timing.thinking_pause_duration * (profile.samples_analyzed - 1) + avgMedium) /
          profile.samples_analyzed;
      }

      if (longPauses.length > 0) {
        const avgLong = longPauses.reduce((a, b) => a + b, 0) / longPauses.length;
        profile.timing.sentence_pause_duration =
          (profile.timing.sentence_pause_duration * (profile.samples_analyzed - 1) + avgLong) /
          profile.samples_analyzed;
      }
    }
  }

  // ==========================================================================
  // ERROR PATTERN ANALYSIS
  // ==========================================================================

  private static updateErrorPatterns(
    profile: TypingProfile,
    typingData: NonNullable<Interaction['typing_data']>
  ): void {
    const { backspaces, corrections } = typingData;

    // Update typo rate (typos per 100 characters)
    const currentTypoRate = (corrections / typingData.total_typing_time) * 100;
    profile.errors.typo_rate =
      (profile.errors.typo_rate * (profile.samples_analyzed - 1) + currentTypoRate) /
      profile.samples_analyzed;

    // Update backspace frequency
    profile.errors.backspace_frequency =
      (profile.errors.backspace_frequency * (profile.samples_analyzed - 1) + backspaces) /
      profile.samples_analyzed;

    // Track correction patterns (keep last 100)
    // In real implementation, would track actual correction sequences
    if (profile.errors.correction_patterns.length > 100) {
      profile.errors.correction_patterns = profile.errors.correction_patterns.slice(-100);
    }
  }

  // ==========================================================================
  // INPUT BEHAVIOR ANALYSIS
  // ==========================================================================

  private static updateInputBehavior(
    profile: TypingProfile,
    typingData: NonNullable<Interaction['typing_data']>,
    interaction: Interaction
  ): void {
    // Detect input burst (paste)
    const avgInterval = typingData.keystroke_intervals.reduce((a, b) => a + b, 0) /
      typingData.keystroke_intervals.length;

    // If average interval is very small (< 10ms), likely a paste
    const isBurst = avgInterval < 10 && interaction.text_length > 50;

    if (isBurst) {
      // Increment copy/paste frequency
      profile.input.copy_paste_frequency++;
      profile.input.input_burst_detected = true;
    } else {
      profile.input.input_burst_detected = false;
    }

    // Update edit distance (based on backspaces/corrections)
    const editDistance = typingData.backspaces + typingData.corrections;
    profile.input.edit_distance_avg =
      (profile.input.edit_distance_avg * (profile.samples_analyzed - 1) + editDistance) /
      profile.samples_analyzed;
  }

  // ==========================================================================
  // DEVICE FINGERPRINT
  // ==========================================================================

  private static updateDeviceFingerprint(
    profile: TypingProfile,
    deviceType: 'mobile' | 'desktop' | 'tablet'
  ): void {
    // Track most common device type
    // In real implementation, would use more sophisticated device fingerprinting
    profile.device.typical_device_type = deviceType;
  }

  // ==========================================================================
  // UTILITY FUNCTIONS
  // ==========================================================================

  private static calculateVariance(numbers: number[]): number {
    if (numbers.length === 0) return 0;

    const mean = numbers.reduce((a, b) => a + b, 0) / numbers.length;
    const squaredDiffs = numbers.map(n => Math.pow(n - mean, 2));
    return squaredDiffs.reduce((a, b) => a + b, 0) / numbers.length;
  }

  /**
   * Get typing profile statistics
   */
  static getStatistics(profile: TypingProfile): {
    avg_keystroke_interval: number;
    avg_word_pause: number;
    avg_thinking_pause: number;
    typo_rate: number;
    backspace_frequency: number;
    copy_paste_frequency: number;
    total_samples: number;
    confidence: number;
  } {
    return {
      avg_keystroke_interval: profile.timing.keystroke_interval_avg,
      avg_word_pause: profile.timing.word_pause_duration,
      avg_thinking_pause: profile.timing.thinking_pause_duration,
      typo_rate: profile.errors.typo_rate,
      backspace_frequency: profile.errors.backspace_frequency,
      copy_paste_frequency: profile.input.copy_paste_frequency,
      total_samples: profile.samples_analyzed,
      confidence: profile.confidence,
    };
  }

  /**
   * Serialize typing profile to JSON
   */
  static toJSON(profile: TypingProfile): any {
    return {
      user_id: profile.user_id,
      created_at: profile.created_at,
      last_updated: profile.last_updated,
      timing: {
        keystroke_intervals: profile.timing.keystroke_intervals,
        keystroke_interval_avg: profile.timing.keystroke_interval_avg,
        keystroke_interval_variance: profile.timing.keystroke_interval_variance,
        word_pause_duration: profile.timing.word_pause_duration,
        thinking_pause_duration: profile.timing.thinking_pause_duration,
        sentence_pause_duration: profile.timing.sentence_pause_duration,
      },
      errors: {
        typo_rate: profile.errors.typo_rate,
        correction_patterns: profile.errors.correction_patterns,
        backspace_frequency: profile.errors.backspace_frequency,
        delete_frequency: profile.errors.delete_frequency,
        common_typos: Object.fromEntries(profile.errors.common_typos),
      },
      input: {
        copy_paste_frequency: profile.input.copy_paste_frequency,
        input_burst_detected: profile.input.input_burst_detected,
        edit_distance_avg: profile.input.edit_distance_avg,
        session_length_avg: profile.input.session_length_avg,
      },
      device: profile.device,
      samples_analyzed: profile.samples_analyzed,
      confidence: profile.confidence,
    };
  }

  /**
   * Deserialize typing profile from JSON
   */
  static fromJSON(data: any): TypingProfile {
    return {
      user_id: data.user_id,
      created_at: data.created_at,
      last_updated: data.last_updated,
      timing: data.timing,
      errors: {
        ...data.errors,
        common_typos: new Map(Object.entries(data.errors.common_typos)),
      },
      input: data.input,
      device: data.device,
      samples_analyzed: data.samples_analyzed,
      confidence: data.confidence,
    };
  }
}
