/**
 * Linguistic Anomaly Detection
 *
 * Detects deviations from established linguistic profiles
 * Multi-signal behavioral security
 */

import {
  LinguisticProfile,
  Interaction,
  LinguisticAnomaly,
  AnomalyScore
} from './types';
import { LinguisticCollector } from './linguistic-collector';

// ============================================================================
// ANOMALY DETECTOR
// ============================================================================

export class AnomalyDetector {
  /**
   * Detect linguistic anomalies in interaction
   * Compares current interaction against established profile
   */
  static detectLinguisticAnomaly(
    profile: LinguisticProfile,
    interaction: Interaction
  ): LinguisticAnomaly {
    // Minimum confidence required (need sufficient baseline)
    if (profile.confidence < 0.3) {
      return {
        score: 0,
        threshold: 0.7,
        alert: false,
        details: {
          vocabulary_deviation: 0,
          syntax_deviation: 0,
          semantics_deviation: 0,
          sentiment_deviation: 0
        },
        specific_anomalies: ['Insufficient baseline data - building profile']
      };
    }

    // Analyze current interaction
    const currentAnalysis = this.analyzeInteraction(interaction);

    // Calculate deviations
    const vocabularyDev = this.calculateVocabularyDeviation(profile, currentAnalysis);
    const syntaxDev = this.calculateSyntaxDeviation(profile, currentAnalysis);
    const semanticsDev = this.calculateSemanticsDeviation(profile, currentAnalysis);
    const sentimentDev = Math.abs(currentAnalysis.sentiment - profile.semantics.sentiment_baseline);

    // Weighted anomaly score
    const anomalyScore =
      vocabularyDev * 0.3 +
      syntaxDev * 0.25 +
      semanticsDev * 0.25 +
      sentimentDev * 0.2;

    const threshold = 0.7;
    const alert = anomalyScore > threshold;

    // Identify specific anomalies
    const specificAnomalies: string[] = [];

    if (vocabularyDev > 0.6) {
      specificAnomalies.push('Unusual vocabulary - words not typically used');
    }
    if (syntaxDev > 0.6) {
      specificAnomalies.push('Unusual sentence structure or length');
    }
    if (semanticsDev > 0.6) {
      specificAnomalies.push('Atypical formality or hedging patterns');
    }
    if (sentimentDev > 0.5) {
      specificAnomalies.push(`Sentiment shift: ${profile.semantics.sentiment_baseline.toFixed(2)} → ${currentAnalysis.sentiment.toFixed(2)}`);
    }

    return {
      score: anomalyScore,
      threshold,
      alert,
      details: {
        vocabulary_deviation: vocabularyDev,
        syntax_deviation: syntaxDev,
        semantics_deviation: semanticsDev,
        sentiment_deviation: sentimentDev
      },
      specific_anomalies: specificAnomalies
    };
  }

  // ==========================================================================
  // DEVIATION CALCULATIONS
  // ==========================================================================

  /**
   * Calculate vocabulary deviation
   * Measures how different word usage is from profile
   */
  private static calculateVocabularyDeviation(
    profile: LinguisticProfile,
    current: InteractionAnalysis
  ): number {
    // Word overlap: how many current words are in profile
    const profileWords = profile.vocabulary.unique_words;
    const currentWords = current.words;

    let overlapCount = 0;
    for (const word of currentWords) {
      if (profileWords.has(word)) {
        overlapCount++;
      }
    }

    const overlapRatio = currentWords.size > 0
      ? overlapCount / currentWords.size
      : 1.0;

    // Word length deviation
    const lengthDiff = Math.abs(current.avg_word_length - profile.vocabulary.average_word_length);
    const lengthDeviation = Math.min(lengthDiff / 5, 1.0);  // Normalize to 0-1

    // Rare words deviation
    const rareWordsDiff = Math.abs(current.rare_words_ratio - profile.vocabulary.rare_words_frequency);

    // Combined score (0 = identical, 1 = completely different)
    return (
      (1 - overlapRatio) * 0.5 +        // Word overlap
      lengthDeviation * 0.3 +            // Word length
      rareWordsDiff * 0.2                // Rare words
    );
  }

  /**
   * Calculate syntax deviation
   * Measures sentence structure differences
   */
  private static calculateSyntaxDeviation(
    profile: LinguisticProfile,
    current: InteractionAnalysis
  ): number {
    // Sentence length deviation
    const lengthDiff = Math.abs(current.avg_sentence_length - profile.syntax.average_sentence_length);
    const lengthDeviation = Math.min(lengthDiff / 10, 1.0);  // Normalize

    // Punctuation pattern deviation
    const punctuationDev = this.comparePunctuationPatterns(
      profile.syntax.punctuation_patterns,
      current.punctuation
    );

    // Passive voice deviation
    const passiveDiff = Math.abs(current.passive_voice_ratio - profile.syntax.passive_voice_frequency);

    // Question frequency deviation
    const questionDiff = Math.abs(current.question_ratio - profile.syntax.question_frequency);

    return (
      lengthDeviation * 0.4 +
      punctuationDev * 0.3 +
      passiveDiff * 0.2 +
      questionDiff * 0.1
    );
  }

  /**
   * Calculate semantics deviation
   * Measures formality and hedging differences
   */
  private static calculateSemanticsDeviation(
    profile: LinguisticProfile,
    current: InteractionAnalysis
  ): number {
    // Formality deviation
    const formalityDiff = Math.abs(current.formality - profile.semantics.formality_level);

    // Hedging deviation
    const hedgingDiff = Math.abs(current.hedging_ratio - profile.semantics.hedging_frequency);

    return (
      formalityDiff * 0.6 +
      hedgingDiff * 0.4
    );
  }

  /**
   * Compare punctuation usage patterns
   */
  private static comparePunctuationPatterns(
    profilePunctuation: Map<string, number>,
    currentPunctuation: Map<string, number>
  ): number {
    // Normalize to frequencies
    const profileTotal = Array.from(profilePunctuation.values()).reduce((a, b) => a + b, 0) || 1;
    const currentTotal = Array.from(currentPunctuation.values()).reduce((a, b) => a + b, 0) || 1;

    const profileFreq = new Map<string, number>();
    const currentFreq = new Map<string, number>();

    for (const [char, count] of profilePunctuation) {
      profileFreq.set(char, count / profileTotal);
    }

    for (const [char, count] of currentPunctuation) {
      currentFreq.set(char, count / currentTotal);
    }

    // Calculate difference
    const allChars = new Set([...profileFreq.keys(), ...currentFreq.keys()]);
    let totalDiff = 0;

    for (const char of allChars) {
      const profileVal = profileFreq.get(char) || 0;
      const currentVal = currentFreq.get(char) || 0;
      totalDiff += Math.abs(profileVal - currentVal);
    }

    return Math.min(totalDiff / 2, 1.0);  // Normalize to 0-1
  }

  // ==========================================================================
  // INTERACTION ANALYSIS
  // ==========================================================================

  /**
   * Analyze current interaction
   * Extract same features as profile collector
   */
  private static analyzeInteraction(interaction: Interaction): InteractionAnalysis {
    const text = interaction.text.toLowerCase();
    const words = this.tokenize(text);
    const sentences = this.splitSentences(text);

    // Word analysis
    const uniqueWords = new Set(words);
    const totalWordLength = words.reduce((sum, word) => sum + word.length, 0);
    const avgWordLength = words.length > 0 ? totalWordLength / words.length : 0;

    // Rare words (words that appear only once)
    const wordFreq = new Map<string, number>();
    for (const word of words) {
      wordFreq.set(word, (wordFreq.get(word) || 0) + 1);
    }
    const rareWords = Array.from(wordFreq.entries()).filter(([_, count]) => count === 1);
    const rareWordsRatio = words.length > 0 ? rareWords.length / words.length : 0;

    // Sentence analysis
    const sentenceLengths = sentences.map(s => this.tokenize(s).length);
    const avgSentenceLength = sentenceLengths.length > 0
      ? sentenceLengths.reduce((a, b) => a + b, 0) / sentenceLengths.length
      : 0;

    // Punctuation
    const punctuation = new Map<string, number>();
    for (const char of text) {
      if (/[.!?,;:\-()[\]{}"]/.test(char)) {
        punctuation.set(char, (punctuation.get(char) || 0) + 1);
      }
    }

    // Passive voice
    const passiveCount = this.countPassiveVoice(text);
    const passiveVoiceRatio = sentences.length > 0 ? passiveCount / sentences.length : 0;

    // Questions
    const questionCount = sentences.filter(s => s.trim().endsWith('?')).length;
    const questionRatio = sentences.length > 0 ? questionCount / sentences.length : 0;

    // Sentiment
    const sentiment = this.calculateSentiment(words);

    // Formality (based on contractions)
    const contractions = this.countContractions(text);
    const formality = 1.0 - (contractions / words.length);

    // Hedging
    const hedgingCount = this.countHedging(text);
    const hedgingRatio = words.length > 0 ? hedgingCount / words.length : 0;

    return {
      words: uniqueWords,
      avg_word_length: avgWordLength,
      rare_words_ratio: rareWordsRatio,
      avg_sentence_length: avgSentenceLength,
      punctuation,
      passive_voice_ratio: passiveVoiceRatio,
      question_ratio: questionRatio,
      sentiment,
      formality,
      hedging_ratio: hedgingRatio
    };
  }

  // ==========================================================================
  // UTILITY FUNCTIONS
  // ==========================================================================

  private static tokenize(text: string): string[] {
    const STOP_WORDS = new Set([
      'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
      'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
      'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
      'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
    ]);

    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(word => word.length > 0 && !STOP_WORDS.has(word));
  }

  private static splitSentences(text: string): string[] {
    return text
      .split(/[.!?]+/)
      .map(s => s.trim())
      .filter(s => s.length > 0);
  }

  private static countPassiveVoice(text: string): number {
    const patterns = [
      /was \w+ed/gi,
      /were \w+ed/gi,
      /been \w+ed/gi,
      /is \w+ed/gi,
      /are \w+ed/gi
    ];

    let count = 0;
    for (const pattern of patterns) {
      const matches = text.match(pattern);
      if (matches) count += matches.length;
    }
    return count;
  }

  private static countContractions(text: string): number {
    const patterns = [
      /\w+n't/gi,
      /\w+'ll/gi,
      /\w+'re/gi,
      /\w+'ve/gi,
      /\w+'d/gi,
      /\w+'s/gi
    ];

    let count = 0;
    for (const pattern of patterns) {
      const matches = text.match(pattern);
      if (matches) count += matches.length;
    }
    return count;
  }

  private static countHedging(text: string): number {
    const HEDGING_WORDS = [
      'maybe', 'perhaps', 'possibly', 'probably', 'likely', 'might', 'may',
      'could', 'seems', 'appears', 'i think', 'i guess', 'i suppose', 'kind of', 'sort of'
    ];

    let count = 0;
    const textLower = text.toLowerCase();
    for (const hedge of HEDGING_WORDS) {
      if (textLower.includes(hedge)) count++;
    }
    return count;
  }

  private static calculateSentiment(words: string[]): number {
    const POSITIVE_WORDS = new Set([
      'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'happy',
      'love', 'like', 'enjoy', 'appreciate', 'perfect', 'best', 'awesome', 'nice'
    ]);

    const NEGATIVE_WORDS = new Set([
      'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'worst',
      'poor', 'sad', 'angry', 'frustrated', 'annoyed', 'disappointed', 'ugly'
    ]);

    let positiveCount = 0;
    let negativeCount = 0;

    for (const word of words) {
      if (POSITIVE_WORDS.has(word)) positiveCount++;
      if (NEGATIVE_WORDS.has(word)) negativeCount++;
    }

    const total = positiveCount + negativeCount;
    return total > 0 ? (positiveCount - negativeCount) / total : 0;
  }

  /**
   * Combine multiple anomaly signals into overall score
   */
  static combineAnomalySignals(
    linguistic: LinguisticAnomaly,
    confidence: number = 1.0
  ): AnomalyScore {
    const score = linguistic.score * confidence;
    const threshold = 0.7;

    return {
      score,
      threshold,
      alert: score > threshold,
      confidence,
      components: {
        linguistic: linguistic.score
      },
      anomalies_detected: linguistic.specific_anomalies,
      reason: linguistic.alert
        ? `Linguistic anomaly detected (score: ${score.toFixed(2)})`
        : 'Normal linguistic patterns'
    };
  }
}

// ============================================================================
// TYPES
// ============================================================================

interface InteractionAnalysis {
  words: Set<string>;
  avg_word_length: number;
  rare_words_ratio: number;
  avg_sentence_length: number;
  punctuation: Map<string, number>;
  passive_voice_ratio: number;
  question_ratio: number;
  sentiment: number;
  formality: number;
  hedging_ratio: number;
}
