/**
 * Emotional Signature Collection
 *
 * Builds emotional baseline using VAD model (Valence, Arousal, Dominance)
 * Focus on detecting coercion through emotional state changes
 */

import { EmotionalProfile, Interaction } from './types';

// =============================================================================
// EMOTIONAL COLLECTOR
// =============================================================================

export class EmotionalCollector {
  /**
   * Create new emotional profile for user
   */
  static createProfile(userId: string): EmotionalProfile {
    return {
      user_id: userId,
      created_at: Date.now(),
      last_updated: Date.now(),
      samples_analyzed: 0,
      confidence: 0,

      // VAD baseline (Valence, Arousal, Dominance)
      baseline: {
        valence: 0, // -1 (negative) to +1 (positive)
        arousal: 0.5, // 0 (calm) to 1 (excited/stressed) - default mid
        dominance: 0.5, // 0 (submissive) to 1 (dominant) - default mid
      },

      // Normal variance
      variance: {
        valence_variance: 0,
        arousal_variance: 0,
        dominance_variance: 0,
      },

      // Contextual signatures
      contexts: {
        work_mode: { valence: 0, arousal: 0.5, dominance: 0.5, timestamp: Date.now() },
        casual_mode: { valence: 0, arousal: 0.5, dominance: 0.5, timestamp: Date.now() },
        stress_mode: { valence: 0, arousal: 0.5, dominance: 0.5, timestamp: Date.now() },
      },

      // Emotion markers
      markers: {
        joy_markers: [],
        fear_markers: [],
        anger_markers: [],
        sadness_markers: [],
      },
    };
  }

  /**
   * Analyze interaction and update emotional profile
   */
  static analyzeAndUpdate(
    profile: EmotionalProfile,
    interaction: Interaction
  ): EmotionalProfile {
    profile.last_updated = Date.now();
    profile.samples_analyzed++;

    // Analyze VAD from text
    const vad = this.analyzeVAD(interaction.text);

    // Update VAD baseline
    this.updateVADBaseline(profile, vad);

    // Analyze emotions and update markers
    this.analyzeAndUpdateEmotionMarkers(profile, interaction.text);

    // Update confidence (100 samples = 100% confidence)
    profile.confidence = Math.min(profile.samples_analyzed / 100, 1.0);

    return profile;
  }

  // ===========================================================================
  // VAD ANALYSIS (Valence, Arousal, Dominance)
  // ===========================================================================

  /**
   * Analyze Valence, Arousal, Dominance from text
   * Uses keyword-based analysis (can be enhanced with LLM)
   */
  private static analyzeVAD(text: string): {
    valence: number;
    arousal: number;
    dominance: number;
  } {
    const lowerText = text.toLowerCase();

    // VALENCE: Positive vs Negative sentiment
    const valence = this.calculateValence(lowerText);

    // AROUSAL: Calm vs Excited/Stressed
    const arousal = this.calculateArousal(lowerText);

    // DOMINANCE: Submissive vs Dominant
    const dominance = this.calculateDominance(lowerText);

    return { valence, arousal, dominance };
  }

  /**
   * Calculate Valence (sentiment: -1 negative to +1 positive)
   */
  private static calculateValence(text: string): number {
    // Positive words
    const positiveWords = [
      'great',
      'good',
      'happy',
      'excellent',
      'love',
      'wonderful',
      'amazing',
      'perfect',
      'fantastic',
      'awesome',
      'excited',
      'glad',
      'pleased',
      'joy',
      'delighted',
      'thanks',
      'thank you',
    ];

    // Negative words
    const negativeWords = [
      'bad',
      'terrible',
      'hate',
      'awful',
      'horrible',
      'sad',
      'angry',
      'upset',
      'frustrated',
      'annoyed',
      'worried',
      'anxious',
      'afraid',
      'scared',
      'fear',
      'unfortunately',
      'sorry',
      'problem',
      'issue',
      'wrong',
    ];

    let positiveScore = 0;
    let negativeScore = 0;

    positiveWords.forEach((word) => {
      if (text.includes(word)) positiveScore++;
    });

    negativeWords.forEach((word) => {
      if (text.includes(word)) negativeScore++;
    });

    // Normalize to -1 to +1
    const total = positiveScore + negativeScore;
    if (total === 0) return 0;

    return (positiveScore - negativeScore) / total;
  }

  /**
   * Calculate Arousal (0 calm to 1 excited/stressed)
   */
  private static calculateArousal(text: string): number {
    // High arousal indicators
    const highArousalWords = [
      'urgent',
      'immediately',
      'now',
      'hurry',
      'quick',
      'fast',
      'asap',
      'emergency',
      'critical',
      'important',
      'anxious',
      'stressed',
      'excited',
      'nervous',
      'worried',
      'panic',
    ];

    // Low arousal indicators
    const lowArousalWords = [
      'calm',
      'relaxed',
      'peaceful',
      'slow',
      'eventually',
      'whenever',
      'easy',
      'chill',
      'no rush',
      'take your time',
    ];

    let highScore = 0;
    let lowScore = 0;

    highArousalWords.forEach((word) => {
      if (text.includes(word)) highScore++;
    });

    lowArousalWords.forEach((word) => {
      if (text.includes(word)) lowScore++;
    });

    // Check for punctuation indicators
    const exclamationCount = (text.match(/!/g) || []).length;
    const capsRatio = (text.match(/[A-Z]/g) || []).length / Math.max(text.length, 1);

    highScore += exclamationCount * 0.5;
    highScore += capsRatio * 5;

    // Normalize to 0-1 (bias towards mid-range 0.5 if no indicators)
    const total = highScore + lowScore;
    if (total === 0) return 0.5;

    return highScore / (highScore + lowScore);
  }

  /**
   * Calculate Dominance (0 submissive to 1 dominant)
   */
  private static calculateDominance(text: string): number {
    // High dominance (assertive, commanding)
    const dominantWords = [
      'will',
      'must',
      'should',
      'need to',
      'have to',
      'require',
      'demand',
      'insist',
      'definitely',
      'certainly',
      'absolutely',
      'command',
      'order',
    ];

    // Low dominance (submissive, uncertain)
    const submissiveWords = [
      'maybe',
      'perhaps',
      'might',
      'could',
      'possibly',
      'if you want',
      'would you',
      'please',
      'sorry',
      'excuse me',
      'i think',
      'i guess',
      'not sure',
      'uncertain',
      "i don't know",
    ];

    let dominantScore = 0;
    let submissiveScore = 0;

    dominantWords.forEach((word) => {
      if (text.includes(word)) dominantScore++;
    });

    submissiveWords.forEach((word) => {
      if (text.includes(word)) submissiveScore++;
    });

    // Check for question marks (uncertainty = lower dominance)
    const questionCount = (text.match(/\?/g) || []).length;
    submissiveScore += questionCount * 0.5;

    // Normalize to 0-1 (bias towards mid-range 0.5 if no indicators)
    const total = dominantScore + submissiveScore;
    if (total === 0) return 0.5;

    return dominantScore / (dominantScore + submissiveScore);
  }

  // ===========================================================================
  // VAD BASELINE UPDATE
  // ===========================================================================

  /**
   * Update VAD baseline with running average
   */
  private static updateVADBaseline(
    profile: EmotionalProfile,
    vad: { valence: number; arousal: number; dominance: number }
  ): void {
    const n = profile.samples_analyzed;

    // Update valence
    const oldValenceAvg = profile.baseline.valence;
    profile.baseline.valence = (oldValenceAvg * (n - 1) + vad.valence) / n;

    // Update valence variance
    const valenceDiff = vad.valence - profile.baseline.valence;
    profile.variance.valence_variance =
      (profile.variance.valence_variance * (n - 1) + valenceDiff * valenceDiff) / n;

    // Update arousal
    const oldArousalAvg = profile.baseline.arousal;
    profile.baseline.arousal = (oldArousalAvg * (n - 1) + vad.arousal) / n;

    // Update arousal variance
    const arousalDiff = vad.arousal - profile.baseline.arousal;
    profile.variance.arousal_variance =
      (profile.variance.arousal_variance * (n - 1) + arousalDiff * arousalDiff) / n;

    // Update dominance
    const oldDominanceAvg = profile.baseline.dominance;
    profile.baseline.dominance = (oldDominanceAvg * (n - 1) + vad.dominance) / n;

    // Update dominance variance
    const dominanceDiff = vad.dominance - profile.baseline.dominance;
    profile.variance.dominance_variance =
      (profile.variance.dominance_variance * (n - 1) + dominanceDiff * dominanceDiff) / n;
  }

  // ===========================================================================
  // EMOTION MARKERS ANALYSIS
  // ===========================================================================

  /**
   * Analyze and update emotion markers from text
   */
  private static analyzeAndUpdateEmotionMarkers(profile: EmotionalProfile, text: string): void {
    const lowerText = text.toLowerCase();

    // Joy markers
    const joyWords = [
      'haha',
      'lol',
      'lmao',
      ':)',
      '😊',
      '😄',
      'happy',
      'joy',
      'excited',
      'glad',
      'delighted',
      'wonderful',
      'great',
      'awesome',
      'yay',
      'woohoo',
    ];

    // Fear markers
    const fearWords = [
      'afraid',
      'scared',
      'worried',
      'anxious',
      'nervous',
      'fearful',
      'terrified',
      'panic',
      'dread',
      'concern',
      'uncertain',
      'uneasy',
    ];

    // Anger markers
    const angerWords = [
      'angry',
      'mad',
      'furious',
      'upset',
      'frustrated',
      'annoyed',
      'irritated',
      'outraged',
      'pissed',
      'hate',
      'damn',
    ];

    // Sadness markers
    const sadnessWords = [
      'sad',
      'unhappy',
      'depressed',
      'down',
      'miserable',
      'disappointed',
      'hopeless',
      'crying',
      'tears',
      ':(',
      '😢',
      '😭',
    ];

    // Detect and add markers
    joyWords.forEach((word) => {
      if (lowerText.includes(word) && !profile.markers.joy_markers.includes(word)) {
        profile.markers.joy_markers.push(word);
      }
    });

    fearWords.forEach((word) => {
      if (lowerText.includes(word) && !profile.markers.fear_markers.includes(word)) {
        profile.markers.fear_markers.push(word);
      }
    });

    angerWords.forEach((word) => {
      if (lowerText.includes(word) && !profile.markers.anger_markers.includes(word)) {
        profile.markers.anger_markers.push(word);
      }
    });

    sadnessWords.forEach((word) => {
      if (lowerText.includes(word) && !profile.markers.sadness_markers.includes(word)) {
        profile.markers.sadness_markers.push(word);
      }
    });
  }

  // ===========================================================================
  // SERIALIZATION
  // ===========================================================================

  /**
   * Convert profile to JSON
   */
  static toJSON(profile: EmotionalProfile): string {
    return JSON.stringify(profile, null, 2);
  }

  /**
   * Restore profile from JSON
   */
  static fromJSON(json: string): EmotionalProfile {
    return JSON.parse(json);
  }
}
