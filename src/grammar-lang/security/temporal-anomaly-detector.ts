/**
 * Temporal Anomaly Detection
 *
 * Detects unusual access times and patterns
 * Focus on impersonation detection through temporal analysis
 */

import { TemporalProfile, Interaction, TemporalAnomaly } from './types';

// =============================================================================
// TEMPORAL ANOMALY DETECTOR
// =============================================================================

export class TemporalAnomalyDetector {
  /**
   * Detect temporal anomalies in interaction
   * Compares current access time/pattern against established baseline
   */
  static detectTemporalAnomaly(
    profile: TemporalProfile,
    interaction: Interaction,
    currentSessionDurationMinutes?: number
  ): TemporalAnomaly {
    // Minimum confidence required (need sufficient baseline)
    if (profile.confidence < 0.3) {
      return {
        score: 0,
        threshold: 0.7,
        alert: false,
        details: {
          unusual_hour: false,
          unusual_day: false,
          unusual_duration: false,
          unusual_frequency: false,
        },
        specific_anomalies: ['Insufficient baseline data - building temporal profile'],
      };
    }

    // Extract temporal features
    const date = new Date(interaction.timestamp);
    const hour = date.getHours(); // 0-23
    const day = date.getDay(); // 0-6

    // Check anomalies
    const unusualHour = this.isUnusualHour(profile, hour);
    const unusualDay = this.isUnusualDay(profile, day);
    const unusualDuration = currentSessionDurationMinutes
      ? this.isUnusualDuration(profile, currentSessionDurationMinutes)
      : false;
    const unusualFrequency = false; // Would need interaction history to calculate

    // Weighted anomaly score
    const anomalyScore =
      (unusualHour ? 0.4 : 0) + // Hour is very important (40%)
      (unusualDay ? 0.3 : 0) + // Day is important (30%)
      (unusualDuration ? 0.2 : 0) + // Duration is somewhat important (20%)
      (unusualFrequency ? 0.1 : 0); // Frequency is least important (10%)

    const threshold = 0.7;
    const alert = anomalyScore > threshold;

    // Identify specific anomalies
    const specificAnomalies: string[] = [];

    if (unusualHour) {
      const typicalHours = Array.from(profile.hourly.typical_hours).sort((a, b) => a - b);
      specificAnomalies.push(
        `Access at unusual hour: ${hour}:00 (typical hours: ${typicalHours.join(', ')})`
      );
    }

    if (unusualDay) {
      const dayNames = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
      const typicalDays = Array.from(profile.daily.typical_days)
        .map((d) => dayNames[d])
        .join(', ');
      specificAnomalies.push(`Access on unusual day: ${dayNames[day]} (typical: ${typicalDays})`);
    }

    if (unusualDuration && currentSessionDurationMinutes) {
      const expected = profile.sessions.session_duration_avg.toFixed(1);
      specificAnomalies.push(
        `Unusual session duration: ${currentSessionDurationMinutes}min (expected: ${expected}min)`
      );
    }

    return {
      score: Math.min(anomalyScore, 1.0), // Cap at 1.0
      threshold,
      alert,
      details: {
        unusual_hour: unusualHour,
        unusual_day: unusualDay,
        unusual_duration: unusualDuration,
        unusual_frequency: unusualFrequency,
      },
      specific_anomalies: specificAnomalies,
    };
  }

  // ===========================================================================
  // ANOMALY CHECKS
  // ===========================================================================

  /**
   * Check if hour is unusual
   */
  private static isUnusualHour(profile: TemporalProfile, hour: number): boolean {
    // If no typical hours established yet, can't determine
    if (profile.hourly.typical_hours.size === 0) {
      return false;
    }

    // Check if hour is in typical hours
    return !profile.hourly.typical_hours.has(hour);
  }

  /**
   * Check if day is unusual
   */
  private static isUnusualDay(profile: TemporalProfile, day: number): boolean {
    // If no typical days established yet, can't determine
    if (profile.daily.typical_days.size === 0) {
      return false;
    }

    // Check if day is in typical days
    return !profile.daily.typical_days.has(day);
  }

  /**
   * Check if session duration is unusual
   */
  private static isUnusualDuration(
    profile: TemporalProfile,
    currentDurationMinutes: number
  ): boolean {
    // If no baseline duration yet, can't determine
    if (profile.sessions.session_duration_avg === 0) {
      return false;
    }

    const baseline = profile.sessions.session_duration_avg;
    const variance = profile.sessions.session_duration_variance;
    const stdDev = Math.sqrt(variance) || 1; // Fallback to prevent division by 0

    // Check if current duration is more than 2 standard deviations away
    const diff = Math.abs(currentDurationMinutes - baseline);
    return diff > 2 * stdDev;
  }

  // ===========================================================================
  // IMPERSONATION DETECTION
  // ===========================================================================

  /**
   * Detect impersonation through temporal patterns
   * Multiple temporal anomalies suggest impersonation
   */
  static detectImpersonation(
    profile: TemporalProfile,
    interaction: Interaction,
    currentSessionDurationMinutes?: number
  ): {
    impersonation_detected: boolean;
    confidence: number;
    indicators: string[];
  } {
    const anomaly = this.detectTemporalAnomaly(profile, interaction, currentSessionDurationMinutes);

    // Impersonation indicators
    const indicators: string[] = [];
    let impersonationScore = 0;

    // Unusual hour (strong indicator)
    if (anomaly.details.unusual_hour) {
      indicators.push('Access at highly unusual hour for this user');
      impersonationScore += 0.4;
    }

    // Unusual day
    if (anomaly.details.unusual_day) {
      indicators.push('Access on unusual day for this user');
      impersonationScore += 0.3;
    }

    // Very unusual duration (too short or too long)
    if (anomaly.details.unusual_duration) {
      indicators.push('Session duration very different from baseline');
      impersonationScore += 0.2;
    }

    // Check for middle-of-night access (2am-5am) if user never accesses at that time
    const date = new Date(interaction.timestamp);
    const hour = date.getHours();
    const isMiddleOfNight = hour >= 2 && hour <= 5;
    const neverAccessedAtNight = !Array.from(profile.hourly.typical_hours).some(
      (h) => h >= 2 && h <= 5
    );

    if (isMiddleOfNight && neverAccessedAtNight) {
      indicators.push('Middle-of-night access (user never accesses at this time)');
      impersonationScore += 0.5; // Very strong indicator
    }

    const impersonationDetected = impersonationScore > 0.6; // 60% confidence threshold
    const confidence = Math.min(impersonationScore, 1.0);

    return {
      impersonation_detected: impersonationDetected,
      confidence,
      indicators,
    };
  }

  // ===========================================================================
  // TIMING ANALYSIS
  // ===========================================================================

  /**
   * Analyze if timing is consistent with user's timezone
   */
  static isTimezoneConsistent(
    profile: TemporalProfile,
    interactionTimestamp: number,
    reportedTimezone?: string
  ): {
    consistent: boolean;
    expected_timezone: string;
    reported_timezone: string;
    hour_difference: number;
  } {
    const expectedTimezone = profile.timezone;
    const actualTimezone = reportedTimezone || 'UTC';

    // This is a simplified check - in production would use proper timezone library
    // For now, just check if reported timezone matches profile
    const consistent = expectedTimezone === actualTimezone;

    return {
      consistent,
      expected_timezone: expectedTimezone,
      reported_timezone: actualTimezone,
      hour_difference: consistent ? 0 : 1, // Simplified
    };
  }
}
