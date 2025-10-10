/**
 * Linguistic Fingerprinting - Pattern Collector
 *
 * Analyzes user interactions to build unique linguistic profiles
 * O(1) amortized complexity - hash-based analysis
 *
 * Now supports LLM-powered sentiment analysis for contextual understanding.
 */

import {
  LinguisticProfile,
  Interaction,
  GrammarPattern,
  ProfileStatistics
} from './types';
import { createGlassLLM, GlassLLM } from '../glass/llm-adapter';

// ============================================================================
// CONSTANTS
// ============================================================================

// Common stop words (for filtering)
const STOP_WORDS = new Set([
  'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
  'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
  'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
  'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
]);

// Punctuation marks
const PUNCTUATION = new Set(['.', '!', '?', ',', ';', ':', '-', '—', '(', ')', '[', ']', '{', '}', '"', "'", '`']);

// Hedging words (uncertainty markers)
const HEDGING_WORDS = new Set([
  'maybe', 'perhaps', 'possibly', 'probably', 'likely', 'might', 'may',
  'could', 'seems', 'appears', 'i think', 'i guess', 'i suppose', 'kind of', 'sort of'
]);

// Sentiment lexicon (simple)
const POSITIVE_WORDS = new Set([
  'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'happy',
  'love', 'like', 'enjoy', 'appreciate', 'perfect', 'best', 'awesome', 'nice'
]);

const NEGATIVE_WORDS = new Set([
  'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'worst',
  'poor', 'sad', 'angry', 'frustrated', 'annoyed', 'disappointed', 'ugly'
]);

// ============================================================================
// LINGUISTIC COLLECTOR
// ============================================================================

export class LinguisticCollector {
  /**
   * Create initial empty linguistic profile
   */
  static createProfile(userId: string): LinguisticProfile {
    return {
      user_id: userId,
      created_at: Date.now(),
      last_updated: Date.now(),

      vocabulary: {
        distribution: new Map(),
        unique_words: new Set(),
        average_word_length: 0,
        rare_words_frequency: 0
      },

      syntax: {
        average_sentence_length: 0,
        sentence_length_variance: 0,
        punctuation_patterns: new Map(),
        grammar_preferences: [],
        passive_voice_frequency: 0,
        question_frequency: 0
      },

      semantics: {
        topic_distribution: new Map(),
        sentiment_baseline: 0,
        sentiment_variance: 0,
        formality_level: 0.5,
        hedging_frequency: 0
      },

      samples_analyzed: 0,
      confidence: 0
    };
  }

  /**
   * Analyze interaction and update linguistic profile
   * O(n) where n = text length (unavoidable for text analysis)
   * But amortized O(1) per update via hash maps
   */
  static analyzeAndUpdate(
    profile: LinguisticProfile,
    interaction: Interaction
  ): LinguisticProfile {
    const text = interaction.text.toLowerCase();

    // Update last_updated
    profile.last_updated = Date.now();
    profile.samples_analyzed++;

    // Lexical analysis
    this.updateVocabulary(profile, text);

    // Syntactic analysis
    this.updateSyntax(profile, text);

    // Semantic analysis
    this.updateSemantics(profile, text);

    // Update confidence (more samples = higher confidence)
    profile.confidence = Math.min(
      profile.samples_analyzed / 100,  // 100 samples = 100% confidence
      1.0
    );

    return profile;
  }

  /**
   * Analyze interaction with LLM-powered sentiment analysis
   * Provides deeper contextual understanding of emotional state
   */
  static async analyzeAndUpdateWithLLM(
    profile: LinguisticProfile,
    interaction: Interaction,
    llm: GlassLLM
  ): Promise<{
    profile: LinguisticProfile;
    sentiment_details: {
      primary_emotion: string;
      intensity: number;
      secondary_emotions: string[];
      reasoning: string;
    };
  }> {
    const text = interaction.text;

    // Update last_updated
    profile.last_updated = Date.now();
    profile.samples_analyzed++;

    // Lexical analysis
    this.updateVocabulary(profile, text.toLowerCase());

    // Syntactic analysis
    this.updateSyntax(profile, text.toLowerCase());

    // LLM-powered semantic analysis
    const sentimentDetails = await this.analyzeSentimentWithLLM(text, llm);

    // Update semantics with LLM results
    profile.semantics.sentiment_baseline =
      (profile.semantics.sentiment_baseline * (profile.samples_analyzed - 1) + sentimentDetails.intensity) /
      profile.samples_analyzed;

    // Update topic distribution from LLM analysis
    const topics = this.extractTopics(text);
    for (const topic of topics) {
      const count = profile.semantics.topic_distribution.get(topic) || 0;
      profile.semantics.topic_distribution.set(topic, count + 1);
    }

    // Update confidence
    profile.confidence = Math.min(
      profile.samples_analyzed / 100,
      1.0
    );

    return {
      profile,
      sentiment_details: sentimentDetails
    };
  }

  /**
   * Analyze sentiment using LLM for contextual understanding
   */
  private static async analyzeSentimentWithLLM(
    text: string,
    llm: GlassLLM
  ): Promise<{
    primary_emotion: string;
    intensity: number;
    secondary_emotions: string[];
    reasoning: string;
  }> {
    const prompt = `Analyze the emotional state and sentiment of this text beyond simple keywords:

**Text**: "${text}"

**Task**: Perform nuanced sentiment analysis considering:
1. Primary emotion (anger, fear, joy, sadness, disgust, surprise, neutral)
2. Emotional intensity (0.0 to 1.0)
3. Secondary emotions present
4. Contextual factors (sarcasm, irony, mixed emotions)

Return JSON:
\`\`\`json
{
  "primary_emotion": "anger" | "fear" | "joy" | "sadness" | "disgust" | "surprise" | "neutral",
  "intensity": 0.7,
  "secondary_emotions": ["frustration", "disappointment"],
  "reasoning": "Brief explanation of emotional analysis"
}
\`\`\``;

    try {
      const response = await llm.invoke(prompt, {
        task: 'sentiment-analysis',
        max_tokens: 500,
        enable_constitutional: false  // Skip for speed
      });

      // Parse LLM response
      const jsonMatch = response.text.match(/```(?:json)?\n([\s\S]*?)\n```/);
      if (jsonMatch) {
        const data = JSON.parse(jsonMatch[1]);

        return {
          primary_emotion: data.primary_emotion || 'neutral',
          intensity: data.intensity || 0.5,
          secondary_emotions: data.secondary_emotions || [],
          reasoning: data.reasoning || 'LLM sentiment analysis'
        };
      }
    } catch (error) {
      console.warn('⚠️  LLM sentiment analysis failed:', error);
    }

    // Fallback to simple keyword-based analysis
    return this.fallbackSentimentAnalysis(text);
  }

  /**
   * Fallback sentiment analysis using keywords
   */
  private static fallbackSentimentAnalysis(text: string): {
    primary_emotion: string;
    intensity: number;
    secondary_emotions: string[];
    reasoning: string;
  } {
    const words = this.tokenize(text.toLowerCase());

    let positiveCount = 0;
    let negativeCount = 0;

    for (const word of words) {
      if (POSITIVE_WORDS.has(word)) positiveCount++;
      if (NEGATIVE_WORDS.has(word)) negativeCount++;
    }

    const totalSentiment = positiveCount + negativeCount;
    const sentimentScore = totalSentiment > 0
      ? (positiveCount - negativeCount) / totalSentiment
      : 0;

    let primaryEmotion = 'neutral';
    if (sentimentScore > 0.3) primaryEmotion = 'joy';
    else if (sentimentScore < -0.3) primaryEmotion = 'sadness';

    return {
      primary_emotion: primaryEmotion,
      intensity: Math.abs(sentimentScore),
      secondary_emotions: [],
      reasoning: 'Fallback keyword-based analysis'
    };
  }

  // ==========================================================================
  // LEXICAL ANALYSIS
  // ==========================================================================

  private static updateVocabulary(profile: LinguisticProfile, text: string): void {
    const words = this.tokenize(text);

    // Update word distribution
    let totalWordLength = 0;
    let rareWordCount = 0;

    for (const word of words) {
      if (word.length === 0) continue;

      // Update distribution (frequency)
      const currentCount = profile.vocabulary.distribution.get(word) || 0;
      profile.vocabulary.distribution.set(word, currentCount + 1);

      // Update unique words
      profile.vocabulary.unique_words.add(word);

      // Track word length
      totalWordLength += word.length;

      // Track rare words (words used only once or twice)
      if (currentCount <= 1) {
        rareWordCount++;
      }
    }

    // Update average word length (running average)
    const newAvgLength = totalWordLength / words.length;
    profile.vocabulary.average_word_length =
      (profile.vocabulary.average_word_length * (profile.samples_analyzed - 1) + newAvgLength) /
      profile.samples_analyzed;

    // Update rare words frequency
    const rareWordsRatio = rareWordCount / words.length;
    profile.vocabulary.rare_words_frequency =
      (profile.vocabulary.rare_words_frequency * (profile.samples_analyzed - 1) + rareWordsRatio) /
      profile.samples_analyzed;
  }

  // ==========================================================================
  // SYNTACTIC ANALYSIS
  // ==========================================================================

  private static updateSyntax(profile: LinguisticProfile, text: string): void {
    const sentences = this.splitSentences(text);

    // Sentence length analysis
    const sentenceLengths = sentences.map(s => this.tokenize(s).length);
    const avgLength = sentenceLengths.reduce((a, b) => a + b, 0) / sentenceLengths.length || 0;

    // Update average sentence length (running average)
    profile.syntax.average_sentence_length =
      (profile.syntax.average_sentence_length * (profile.samples_analyzed - 1) + avgLength) /
      profile.samples_analyzed;

    // Update variance
    const variance = this.calculateVariance(sentenceLengths);
    profile.syntax.sentence_length_variance =
      (profile.syntax.sentence_length_variance * (profile.samples_analyzed - 1) + variance) /
      profile.samples_analyzed;

    // Punctuation patterns
    for (const char of text) {
      if (PUNCTUATION.has(char)) {
        const count = profile.syntax.punctuation_patterns.get(char) || 0;
        profile.syntax.punctuation_patterns.set(char, count + 1);
      }
    }

    // Passive voice detection (simple heuristic)
    const passiveCount = this.countPassiveVoice(text);
    const passiveRatio = passiveCount / sentences.length;
    profile.syntax.passive_voice_frequency =
      (profile.syntax.passive_voice_frequency * (profile.samples_analyzed - 1) + passiveRatio) /
      profile.samples_analyzed;

    // Question frequency
    const questionCount = sentences.filter(s => s.trim().endsWith('?')).length;
    const questionRatio = questionCount / sentences.length;
    profile.syntax.question_frequency =
      (profile.syntax.question_frequency * (profile.samples_analyzed - 1) + questionRatio) /
      profile.samples_analyzed;
  }

  // ==========================================================================
  // SEMANTIC ANALYSIS
  // ==========================================================================

  private static updateSemantics(profile: LinguisticProfile, text: string): void {
    const words = this.tokenize(text);

    // Sentiment analysis
    let positiveCount = 0;
    let negativeCount = 0;
    let hedgingCount = 0;

    const textLower = text.toLowerCase();

    for (const word of words) {
      if (POSITIVE_WORDS.has(word)) positiveCount++;
      if (NEGATIVE_WORDS.has(word)) negativeCount++;
    }

    // Check for hedging phrases
    for (const hedge of HEDGING_WORDS) {
      if (textLower.includes(hedge)) hedgingCount++;
    }

    // Calculate sentiment (-1 to +1)
    const totalSentiment = positiveCount + negativeCount;
    const sentiment = totalSentiment > 0
      ? (positiveCount - negativeCount) / totalSentiment
      : 0;

    // Update sentiment baseline (running average)
    profile.semantics.sentiment_baseline =
      (profile.semantics.sentiment_baseline * (profile.samples_analyzed - 1) + sentiment) /
      profile.samples_analyzed;

    // Hedging frequency
    const hedgingRatio = hedgingCount / words.length;
    profile.semantics.hedging_frequency =
      (profile.semantics.hedging_frequency * (profile.samples_analyzed - 1) + hedgingRatio) /
      profile.samples_analyzed;

    // Formality level (simple heuristic based on contractions and slang)
    const contractions = this.countContractions(text);
    const contractionsRatio = contractions / words.length;
    const formality = 1.0 - contractionsRatio;  // Fewer contractions = more formal

    profile.semantics.formality_level =
      (profile.semantics.formality_level * (profile.samples_analyzed - 1) + formality) /
      profile.samples_analyzed;

    // Topic distribution (extract main nouns)
    const topics = this.extractTopics(text);
    for (const topic of topics) {
      const count = profile.semantics.topic_distribution.get(topic) || 0;
      profile.semantics.topic_distribution.set(topic, count + 1);
    }
  }

  // ==========================================================================
  // UTILITY FUNCTIONS
  // ==========================================================================

  /**
   * Tokenize text into words
   * O(n) where n = text length
   */
  private static tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')  // Remove punctuation
      .split(/\s+/)
      .filter(word => word.length > 0 && !STOP_WORDS.has(word));
  }

  /**
   * Split text into sentences
   * O(n) where n = text length
   */
  private static splitSentences(text: string): string[] {
    return text
      .split(/[.!?]+/)
      .map(s => s.trim())
      .filter(s => s.length > 0);
  }

  /**
   * Calculate variance of an array
   * O(n) where n = array length
   */
  private static calculateVariance(numbers: number[]): number {
    if (numbers.length === 0) return 0;

    const mean = numbers.reduce((a, b) => a + b, 0) / numbers.length;
    const squaredDiffs = numbers.map(n => Math.pow(n - mean, 2));
    return squaredDiffs.reduce((a, b) => a + b, 0) / numbers.length;
  }

  /**
   * Count passive voice constructions
   * Simple heuristic: "was/were/been + [verb]ed"
   */
  private static countPassiveVoice(text: string): number {
    const passivePatterns = [
      /was \w+ed/gi,
      /were \w+ed/gi,
      /been \w+ed/gi,
      /is \w+ed/gi,
      /are \w+ed/gi
    ];

    let count = 0;
    for (const pattern of passivePatterns) {
      const matches = text.match(pattern);
      if (matches) count += matches.length;
    }

    return count;
  }

  /**
   * Count contractions (informal language)
   */
  private static countContractions(text: string): number {
    const contractionPatterns = [
      /\w+n't/gi,      // don't, can't, won't
      /\w+'ll/gi,      // I'll, you'll
      /\w+'re/gi,      // we're, they're
      /\w+'ve/gi,      // I've, we've
      /\w+'d/gi,       // I'd, he'd
      /\w+'s/gi        // it's, that's
    ];

    let count = 0;
    for (const pattern of contractionPatterns) {
      const matches = text.match(pattern);
      if (matches) count += matches.length;
    }

    return count;
  }

  /**
   * Extract main topics (simple noun extraction)
   */
  private static extractTopics(text: string): string[] {
    // Simple heuristic: capitalized words that aren't sentence-initial
    const words = text.split(/\s+/);
    const topics: string[] = [];

    for (let i = 1; i < words.length; i++) {  // Skip first word
      const word = words[i];
      // If word starts with capital and isn't after punctuation
      if (/^[A-Z][a-z]+/.test(word) && !/[.!?]$/.test(words[i-1])) {
        topics.push(word.toLowerCase());
      }
    }

    return topics;
  }

  /**
   * Get profile statistics
   */
  static getStatistics(profile: LinguisticProfile): {
    vocabulary_size: number;
    most_common_words: [string, number][];
    most_common_punctuation: [string, number][];
    most_common_topics: [string, number][];
  } {
    // Sort by frequency
    const sortedWords = Array.from(profile.vocabulary.distribution.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 20);

    const sortedPunctuation = Array.from(profile.syntax.punctuation_patterns.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10);

    const sortedTopics = Array.from(profile.semantics.topic_distribution.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10);

    return {
      vocabulary_size: profile.vocabulary.unique_words.size,
      most_common_words: sortedWords,
      most_common_punctuation: sortedPunctuation,
      most_common_topics: sortedTopics
    };
  }

  /**
   * Serialize profile to JSON
   */
  static toJSON(profile: LinguisticProfile): any {
    return {
      user_id: profile.user_id,
      created_at: profile.created_at,
      last_updated: profile.last_updated,
      vocabulary: {
        distribution: Object.fromEntries(profile.vocabulary.distribution),
        unique_words: Array.from(profile.vocabulary.unique_words),
        average_word_length: profile.vocabulary.average_word_length,
        rare_words_frequency: profile.vocabulary.rare_words_frequency
      },
      syntax: {
        average_sentence_length: profile.syntax.average_sentence_length,
        sentence_length_variance: profile.syntax.sentence_length_variance,
        punctuation_patterns: Object.fromEntries(profile.syntax.punctuation_patterns),
        grammar_preferences: profile.syntax.grammar_preferences,
        passive_voice_frequency: profile.syntax.passive_voice_frequency,
        question_frequency: profile.syntax.question_frequency
      },
      semantics: {
        topic_distribution: Object.fromEntries(profile.semantics.topic_distribution),
        sentiment_baseline: profile.semantics.sentiment_baseline,
        sentiment_variance: profile.semantics.sentiment_variance,
        formality_level: profile.semantics.formality_level,
        hedging_frequency: profile.semantics.hedging_frequency
      },
      samples_analyzed: profile.samples_analyzed,
      confidence: profile.confidence
    };
  }

  /**
   * Deserialize profile from JSON
   */
  static fromJSON(data: any): LinguisticProfile {
    return {
      user_id: data.user_id,
      created_at: data.created_at,
      last_updated: data.last_updated,
      vocabulary: {
        distribution: new Map(Object.entries(data.vocabulary.distribution)),
        unique_words: new Set(data.vocabulary.unique_words),
        average_word_length: data.vocabulary.average_word_length,
        rare_words_frequency: data.vocabulary.rare_words_frequency
      },
      syntax: {
        average_sentence_length: data.syntax.average_sentence_length,
        sentence_length_variance: data.syntax.sentence_length_variance,
        punctuation_patterns: new Map(Object.entries(data.syntax.punctuation_patterns)),
        grammar_preferences: data.syntax.grammar_preferences || [],
        passive_voice_frequency: data.syntax.passive_voice_frequency,
        question_frequency: data.syntax.question_frequency
      },
      semantics: {
        topic_distribution: new Map(Object.entries(data.semantics.topic_distribution)),
        sentiment_baseline: data.semantics.sentiment_baseline,
        sentiment_variance: data.semantics.sentiment_variance,
        formality_level: data.semantics.formality_level,
        hedging_frequency: data.semantics.hedging_frequency
      },
      samples_analyzed: data.samples_analyzed,
      confidence: data.confidence
    };
  }
}
