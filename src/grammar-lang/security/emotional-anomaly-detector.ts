/**
 * Emotional Anomaly Detection
 *
 * Detects emotional state deviations and coercion patterns
 * Focus on VAD model (Valence, Arousal, Dominance)
 */

import { EmotionalProfile, Interaction, EmotionalAnomaly } from './types';

// =============================================================================
// EMOTIONAL ANOMALY DETECTOR
// =============================================================================

export class EmotionalAnomalyDetector {
  /**
   * Detect emotional anomalies in interaction
   * Compares current emotional state against established baseline
   */
  static detectEmotionalAnomaly(
    profile: EmotionalProfile,
    interaction: Interaction
  ): EmotionalAnomaly {
    // Minimum confidence required (need sufficient baseline)
    if (profile.confidence < 0.3) {
      return {
        score: 0,
        threshold: 0.7,
        alert: false,
        details: {
          valence_deviation: 0,
          arousal_deviation: 0,
          dominance_deviation: 0,
        },
        specific_anomalies: ['Insufficient baseline data - building emotional profile'],
      };
    }

    // Analyze VAD from current text
    const currentVAD = this.analyzeVADFromText(interaction.text);

    // Calculate deviations
    const valenceDev = this.calculateValenceDeviation(profile, currentVAD.valence);
    const arousalDev = this.calculateArousalDeviation(profile, currentVAD.arousal);
    const dominanceDev = this.calculateDominanceDeviation(profile, currentVAD.dominance);

    // Weighted anomaly score
    const anomalyScore =
      valenceDev * 0.35 + // Valence shift is strong indicator
      arousalDev * 0.35 + // Arousal change indicates stress
      dominanceDev * 0.30; // Dominance change indicates submission/coercion

    const threshold = 0.7;
    const alert = anomalyScore > threshold;

    // Identify specific anomalies
    const specificAnomalies: string[] = [];

    if (valenceDev > 0.6) {
      const moreNegative = currentVAD.valence < profile.baseline.valence;
      specificAnomalies.push(
        moreNegative
          ? 'Sentiment significantly more negative than usual (possible distress)'
          : 'Sentiment significantly more positive than usual (possible forced positivity)'
      );
    }

    if (arousalDev > 0.6) {
      const higherArousal = currentVAD.arousal > profile.baseline.arousal;
      specificAnomalies.push(
        higherArousal
          ? 'Arousal significantly higher (possible stress/anxiety)'
          : 'Arousal significantly lower (possible sedation/suppression)'
      );
    }

    if (dominanceDev > 0.6) {
      const lowerDominance = currentVAD.dominance < profile.baseline.dominance;
      specificAnomalies.push(
        lowerDominance
          ? 'Dominance significantly lower (possible submission/coercion)'
          : 'Dominance significantly higher (possible overcompensation)'
      );
    }

    return {
      score: Math.min(anomalyScore, 1.0), // Cap at 1.0
      threshold,
      alert,
      details: {
        valence_deviation: valenceDev,
        arousal_deviation: arousalDev,
        dominance_deviation: dominanceDev,
      },
      specific_anomalies: specificAnomalies,
    };
  }

  // ===========================================================================
  // COERCION DETECTION
  // ===========================================================================

  /**
   * Detect coercion through emotional pattern analysis
   * Coercion typically shows: negative valence + high arousal + low dominance
   */
  static detectCoercion(
    profile: EmotionalProfile,
    interaction: Interaction
  ): {
    coercion_detected: boolean;
    confidence: number;
    indicators: string[];
  } {
    const anomaly = this.detectEmotionalAnomaly(profile, interaction);
    const currentVAD = this.analyzeVADFromText(interaction.text);

    // Coercion indicators
    const indicators: string[] = [];
    let coercionScore = 0;

    // Check for negative valence (fear, anxiety)
    if (currentVAD.valence < profile.baseline.valence - 2 * Math.sqrt(profile.variance.valence_variance)) {
      indicators.push('Negative sentiment (fear/anxiety indicator)');
      coercionScore += 0.4;
    }

    // Check for high arousal (stress)
    if (currentVAD.arousal > profile.baseline.arousal + 2 * Math.sqrt(profile.variance.arousal_variance)) {
      indicators.push('High arousal (stress/anxiety indicator)');
      coercionScore += 0.3;
    }

    // Check for low dominance (submission)
    if (currentVAD.dominance < profile.baseline.dominance - 2 * Math.sqrt(profile.variance.dominance_variance)) {
      indicators.push('Low dominance (submission indicator)');
      coercionScore += 0.3;
    }

    // Check for fear markers
    const textLower = interaction.text.toLowerCase();
    const fearMarkers = ['afraid', 'scared', 'worried', 'anxious', 'nervous', 'can\'t', 'forced', 'have to'];
    const hasFearMarkers = fearMarkers.some((marker) => textLower.includes(marker));

    if (hasFearMarkers) {
      indicators.push('Fear markers detected in text');
      coercionScore += 0.2;
    }

    // Check for submission language
    const submissionMarkers = ['please', 'sorry', 'must', 'have to', 'no choice', 'forced'];
    const hasSubmissionMarkers = submissionMarkers.some((marker) => textLower.includes(marker));

    if (hasSubmissionMarkers) {
      indicators.push('Submission language detected');
      coercionScore += 0.2;
    }

    // Coercion classic pattern: negative + high arousal + low dominance
    const classicPattern =
      currentVAD.valence < -0.3 && currentVAD.arousal > 0.7 && currentVAD.dominance < 0.3;

    if (classicPattern) {
      indicators.push('Classic coercion pattern detected (negative, stressed, submissive)');
      coercionScore += 0.5; // Very strong indicator
    }

    const coercionDetected = coercionScore > 0.6; // 60% confidence threshold
    const confidence = Math.min(coercionScore, 1.0);

    return {
      coercion_detected: coercionDetected,
      confidence,
      indicators,
    };
  }

  // ===========================================================================
  // DEVIATION CALCULATIONS
  // ===========================================================================

  /**
   * Calculate valence deviation
   * Measures sentiment shift (more negative/positive)
   */
  private static calculateValenceDeviation(profile: EmotionalProfile, currentValence: number): number {
    const baselineValence = profile.baseline.valence;
    const variance = profile.variance.valence_variance;

    // Calculate number of standard deviations away
    const diff = Math.abs(currentValence - baselineValence);
    const stdDev = Math.sqrt(variance) || 0.1; // Fallback to prevent division by 0

    // Normalize to 0-1 (3 std deviations = 1.0)
    return Math.min(diff / (3 * stdDev), 1.0);
  }

  /**
   * Calculate arousal deviation
   * Measures stress level changes
   */
  private static calculateArousalDeviation(profile: EmotionalProfile, currentArousal: number): number {
    const baselineArousal = profile.baseline.arousal;
    const variance = profile.variance.arousal_variance;

    const diff = Math.abs(currentArousal - baselineArousal);
    const stdDev = Math.sqrt(variance) || 0.1;

    // Normalize to 0-1 (3 std deviations = 1.0)
    return Math.min(diff / (3 * stdDev), 1.0);
  }

  /**
   * Calculate dominance deviation
   * Measures assertiveness/submission changes
   */
  private static calculateDominanceDeviation(
    profile: EmotionalProfile,
    currentDominance: number
  ): number {
    const baselineDominance = profile.baseline.dominance;
    const variance = profile.variance.dominance_variance;

    const diff = Math.abs(currentDominance - baselineDominance);
    const stdDev = Math.sqrt(variance) || 0.1;

    // Normalize to 0-1 (3 std deviations = 1.0)
    return Math.min(diff / (3 * stdDev), 1.0);
  }

  // ===========================================================================
  // VAD ANALYSIS (simplified keyword-based)
  // ===========================================================================

  /**
   * Analyze VAD from text (simplified version)
   * This is a quick analysis - full version is in EmotionalCollector
   */
  private static analyzeVADFromText(text: string): {
    valence: number;
    arousal: number;
    dominance: number;
  } {
    const lowerText = text.toLowerCase();

    // VALENCE: Positive vs Negative
    const valence = this.quickValenceAnalysis(lowerText);

    // AROUSAL: Calm vs Excited/Stressed
    const arousal = this.quickArousalAnalysis(lowerText);

    // DOMINANCE: Submissive vs Dominant
    const dominance = this.quickDominanceAnalysis(lowerText);

    return { valence, arousal, dominance };
  }

  /**
   * Quick valence analysis
   */
  private static quickValenceAnalysis(text: string): number {
    const positiveWords = ['great', 'good', 'happy', 'excellent', 'love', 'wonderful', 'thanks'];
    const negativeWords = ['bad', 'terrible', 'hate', 'awful', 'sad', 'angry', 'worried', 'afraid'];

    let score = 0;
    positiveWords.forEach((word) => {
      if (text.includes(word)) score += 0.2;
    });
    negativeWords.forEach((word) => {
      if (text.includes(word)) score -= 0.2;
    });

    return Math.max(-1, Math.min(1, score));
  }

  /**
   * Quick arousal analysis
   */
  private static quickArousalAnalysis(text: string): number {
    const highArousalWords = ['urgent', 'immediately', 'now', 'hurry', 'anxious', 'stressed', 'panic'];
    const lowArousalWords = ['calm', 'relaxed', 'peaceful', 'slow', 'easy'];

    let score = 0.5; // Default mid
    highArousalWords.forEach((word) => {
      if (text.includes(word)) score += 0.15;
    });
    lowArousalWords.forEach((word) => {
      if (text.includes(word)) score -= 0.15;
    });

    // Check punctuation
    const exclamations = (text.match(/!/g) || []).length;
    score += exclamations * 0.05;

    return Math.max(0, Math.min(1, score));
  }

  /**
   * Quick dominance analysis
   */
  private static quickDominanceAnalysis(text: string): number {
    const dominantWords = ['will', 'must', 'should', 'need', 'require', 'demand'];
    const submissiveWords = ['maybe', 'perhaps', 'might', 'please', 'sorry', 'i guess', 'not sure'];

    let score = 0.5; // Default mid
    dominantWords.forEach((word) => {
      if (text.includes(word)) score += 0.15;
    });
    submissiveWords.forEach((word) => {
      if (text.includes(word)) score -= 0.15;
    });

    // Check questions (uncertainty = lower dominance)
    const questions = (text.match(/\?/g) || []).length;
    score -= questions * 0.05;

    return Math.max(0, Math.min(1, score));
  }
}
