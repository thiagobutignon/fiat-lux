/**
 * Temporal Pattern Collection
 *
 * Builds temporal baseline: when user typically interacts
 * Focus on detecting unusual access times (possible impersonation)
 */

import { TemporalProfile, Interaction } from './types';

// =============================================================================
// TEMPORAL COLLECTOR
// =============================================================================

export class TemporalCollector {
  /**
   * Create new temporal profile for user
   */
  static createProfile(userId: string, timezone: string = 'UTC'): TemporalProfile {
    return {
      user_id: userId,
      created_at: Date.now(),
      last_updated: Date.now(),
      samples_analyzed: 0,
      confidence: 0,

      // Hourly patterns (0-23)
      hourly: {
        typical_hours: new Set(),
        hour_distribution: new Map(),
      },

      // Daily patterns (0-6, Sunday-Saturday)
      daily: {
        typical_days: new Set(),
        day_distribution: new Map(),
      },

      // Session patterns
      sessions: {
        session_duration_avg: 0,
        session_duration_variance: 0,
        interactions_per_day_avg: 0,
        interactions_per_week_avg: 0,
      },

      // Offline patterns
      offline: {
        typical_offline_periods: [],
        longest_offline_duration: 0,
      },

      // Timezone
      timezone,
    };
  }

  /**
   * Analyze interaction and update temporal profile
   */
  static analyzeAndUpdate(
    profile: TemporalProfile,
    interaction: Interaction,
    sessionDurationMinutes?: number
  ): TemporalProfile {
    profile.last_updated = Date.now();
    profile.samples_analyzed++;

    // Extract temporal features
    const date = new Date(interaction.timestamp);
    const hour = date.getHours(); // 0-23
    const day = date.getDay(); // 0-6 (Sunday-Saturday)

    // Update hourly distribution
    this.updateHourlyPatterns(profile, hour);

    // Update daily distribution
    this.updateDailyPatterns(profile, day);

    // Update session patterns
    if (sessionDurationMinutes !== undefined) {
      this.updateSessionPatterns(profile, sessionDurationMinutes);
    }

    // Update confidence (100 samples = 100% confidence)
    profile.confidence = Math.min(profile.samples_analyzed / 100, 1.0);

    return profile;
  }

  // ===========================================================================
  // HOURLY PATTERNS
  // ===========================================================================

  /**
   * Update hourly patterns
   */
  private static updateHourlyPatterns(profile: TemporalProfile, hour: number): void {
    // Update hour distribution
    const currentCount = profile.hourly.hour_distribution.get(hour) || 0;
    profile.hourly.hour_distribution.set(hour, currentCount + 1);

    // Update typical hours (threshold: at least 10% of interactions)
    const threshold = profile.samples_analyzed * 0.1;
    profile.hourly.typical_hours.clear();

    for (const [h, count] of profile.hourly.hour_distribution.entries()) {
      if (count >= threshold) {
        profile.hourly.typical_hours.add(h);
      }
    }
  }

  // ===========================================================================
  // DAILY PATTERNS
  // ===========================================================================

  /**
   * Update daily patterns
   */
  private static updateDailyPatterns(profile: TemporalProfile, day: number): void {
    // Update day distribution
    const currentCount = profile.daily.day_distribution.get(day) || 0;
    profile.daily.day_distribution.set(day, currentCount + 1);

    // Update typical days (threshold: at least 5% of interactions)
    const threshold = profile.samples_analyzed * 0.05;
    profile.daily.typical_days.clear();

    for (const [d, count] of profile.daily.day_distribution.entries()) {
      if (count >= threshold) {
        profile.daily.typical_days.add(d);
      }
    }
  }

  // ===========================================================================
  // SESSION PATTERNS
  // ===========================================================================

  /**
   * Update session patterns
   */
  private static updateSessionPatterns(
    profile: TemporalProfile,
    sessionDurationMinutes: number
  ): void {
    const n = profile.samples_analyzed;

    // Update average session duration
    const oldAvg = profile.sessions.session_duration_avg;
    profile.sessions.session_duration_avg = (oldAvg * (n - 1) + sessionDurationMinutes) / n;

    // Update session duration variance
    const diff = sessionDurationMinutes - profile.sessions.session_duration_avg;
    profile.sessions.session_duration_variance =
      (profile.sessions.session_duration_variance * (n - 1) + diff * diff) / n;
  }

  /**
   * Update interactions per day/week
   */
  static updateInteractionFrequency(
    profile: TemporalProfile,
    interactionsInLastDay: number,
    interactionsInLastWeek: number
  ): void {
    const n = profile.samples_analyzed;

    // Running average of interactions per day
    const oldDayAvg = profile.sessions.interactions_per_day_avg;
    profile.sessions.interactions_per_day_avg = (oldDayAvg * (n - 1) + interactionsInLastDay) / n;

    // Running average of interactions per week
    const oldWeekAvg = profile.sessions.interactions_per_week_avg;
    profile.sessions.interactions_per_week_avg = (oldWeekAvg * (n - 1) + interactionsInLastWeek) / n;
  }

  // ===========================================================================
  // OFFLINE PERIODS
  // ===========================================================================

  /**
   * Detect and record offline periods
   */
  static recordOfflinePeriod(
    profile: TemporalProfile,
    offlineDurationMinutes: number,
    startHour: number,
    endHour: number
  ): void {
    // Update longest offline duration
    if (offlineDurationMinutes > profile.offline.longest_offline_duration) {
      profile.offline.longest_offline_duration = offlineDurationMinutes;
    }

    // Check if this offline period is typical
    const isTypical = offlineDurationMinutes > 60; // At least 1 hour

    if (isTypical) {
      // Check if we already have a similar period
      const existingPeriod = profile.offline.typical_offline_periods.find(
        (p) => Math.abs(p.start_hour - startHour) <= 1 && Math.abs(p.end_hour - endHour) <= 1
      );

      if (existingPeriod) {
        // Increase confidence
        existingPeriod.confidence = Math.min(existingPeriod.confidence + 0.1, 1.0);
      } else {
        // Add new typical period
        profile.offline.typical_offline_periods.push({
          start_hour: startHour,
          end_hour: endHour,
          confidence: 0.5,
        });
      }
    }
  }

  // ===========================================================================
  // STATISTICS
  // ===========================================================================

  /**
   * Get temporal statistics
   */
  static getStatistics(profile: TemporalProfile): {
    most_active_hours: number[];
    most_active_days: number[];
    avg_session_duration: number;
    typical_offline_hours: { start: number; end: number }[];
  } {
    // Sort hours by frequency
    const hourCounts = Array.from(profile.hourly.hour_distribution.entries());
    hourCounts.sort((a, b) => b[1] - a[1]);
    const mostActiveHours = hourCounts.slice(0, 5).map((h) => h[0]);

    // Sort days by frequency
    const dayCounts = Array.from(profile.daily.day_distribution.entries());
    dayCounts.sort((a, b) => b[1] - a[1]);
    const mostActiveDays = dayCounts.slice(0, 3).map((d) => d[0]);

    // Offline hours
    const typicalOfflineHours = profile.offline.typical_offline_periods.map((p) => ({
      start: p.start_hour,
      end: p.end_hour,
    }));

    return {
      most_active_hours: mostActiveHours,
      most_active_days: mostActiveDays,
      avg_session_duration: profile.sessions.session_duration_avg,
      typical_offline_hours: typicalOfflineHours,
    };
  }

  // ===========================================================================
  // SERIALIZATION
  // ===========================================================================

  /**
   * Convert profile to JSON
   */
  static toJSON(profile: TemporalProfile): string {
    const serializable = {
      ...profile,
      hourly: {
        typical_hours: Array.from(profile.hourly.typical_hours),
        hour_distribution: Array.from(profile.hourly.hour_distribution.entries()),
      },
      daily: {
        typical_days: Array.from(profile.daily.typical_days),
        day_distribution: Array.from(profile.daily.day_distribution.entries()),
      },
    };
    return JSON.stringify(serializable, null, 2);
  }

  /**
   * Restore profile from JSON
   */
  static fromJSON(json: string): TemporalProfile {
    const data = JSON.parse(json);
    return {
      ...data,
      hourly: {
        typical_hours: new Set(data.hourly.typical_hours),
        hour_distribution: new Map(data.hourly.hour_distribution),
      },
      daily: {
        typical_days: new Set(data.daily.typical_days),
        day_distribution: new Map(data.daily.day_distribution),
      },
    };
  }
}
