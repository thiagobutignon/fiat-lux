/**
 * Embedding Adapter for SQLO Database
 *
 * Local embedding generation using transformers.js
 * - Zero cost (no API calls)
 * - Fast (<50ms per embedding)
 * - 384-dimensional vectors
 * - Semantic similarity search
 *
 * Model: Xenova/all-MiniLM-L6-v2 (22MB, sentence-transformers)
 * - Optimized for semantic similarity
 * - Multilingual support
 * - 384 dimensions
 * - Mean pooling
 */

import { pipeline } from '@xenova/transformers';

// ============================================================================
// Types
// ============================================================================

export type Embedding = number[]; // 384-dimensional vector

export interface EmbeddingResult {
  embedding: Embedding;
  dimensions: number;
  model: string;
  time_ms: number;
}

export interface SimilarityResult {
  similarity: number; // Cosine similarity [0-1]
  distance: number;   // Euclidean distance
}

// ============================================================================
// Embedding Adapter
// ============================================================================

export class EmbeddingAdapter {
  private pipeline: any | null = null;
  private model: string = 'Xenova/all-MiniLM-L6-v2';
  private dimensions: number = 384;
  private totalEmbeddings: number = 0;
  private totalTimeMs: number = 0;

  constructor() {
    // Pipeline is lazy-loaded on first use
  }

  /**
   * Initialize the embedding pipeline (lazy)
   */
  private async initialize(): Promise<void> {
    if (this.pipeline) return;

    try {
      // Load feature extraction pipeline
      this.pipeline = await pipeline('feature-extraction', this.model);
    } catch (error) {
      throw new Error(`Failed to initialize embedding model: ${error}`);
    }
  }

  /**
   * Generate embedding for text
   *
   * @param text - Text to embed
   * @returns Embedding result with vector and metadata
   */
  async embed(text: string): Promise<EmbeddingResult> {
    if (!text || text.trim().length === 0) {
      throw new Error('Text cannot be empty');
    }

    // Initialize pipeline if needed
    await this.initialize();

    const startTime = Date.now();

    try {
      // Generate embedding
      const output = await this.pipeline!(text, {
        pooling: 'mean', // Mean pooling for sentence embeddings
        normalize: true, // Normalize vectors for cosine similarity
      });

      // Extract embedding array
      const embedding = Array.from(output.data) as Embedding;

      const timeMs = Date.now() - startTime;

      // Track stats
      this.totalEmbeddings++;
      this.totalTimeMs += timeMs;

      return {
        embedding,
        dimensions: this.dimensions,
        model: this.model,
        time_ms: timeMs,
      };
    } catch (error) {
      throw new Error(`Failed to generate embedding: ${error}`);
    }
  }

  /**
   * Generate embeddings for multiple texts (batched)
   *
   * @param texts - Array of texts to embed
   * @returns Array of embeddings
   */
  async embedBatch(texts: string[]): Promise<EmbeddingResult[]> {
    if (!texts || texts.length === 0) {
      return [];
    }

    // Process in parallel for speed
    const results = await Promise.all(
      texts.map(text => this.embed(text))
    );

    return results;
  }

  /**
   * Calculate cosine similarity between two embeddings
   *
   * Cosine similarity: [-1, 1]
   * - 1.0 = identical
   * - 0.0 = orthogonal (no similarity)
   * - -1.0 = opposite
   *
   * We normalize to [0, 1] for easier interpretation:
   * - 1.0 = identical
   * - 0.5 = orthogonal
   * - 0.0 = opposite
   */
  cosineSimilarity(embedding1: Embedding, embedding2: Embedding): number {
    if (embedding1.length !== embedding2.length) {
      throw new Error('Embeddings must have same dimensions');
    }

    // Dot product
    let dotProduct = 0;
    for (let i = 0; i < embedding1.length; i++) {
      dotProduct += embedding1[i] * embedding2[i];
    }

    // Magnitudes (should be 1.0 if normalized)
    let mag1 = 0;
    let mag2 = 0;
    for (let i = 0; i < embedding1.length; i++) {
      mag1 += embedding1[i] * embedding1[i];
      mag2 += embedding2[i] * embedding2[i];
    }
    mag1 = Math.sqrt(mag1);
    mag2 = Math.sqrt(mag2);

    // Cosine similarity
    const similarity = dotProduct / (mag1 * mag2);

    // Normalize to [0, 1]
    return (similarity + 1) / 2;
  }

  /**
   * Calculate Euclidean distance between two embeddings
   *
   * Lower distance = more similar
   */
  euclideanDistance(embedding1: Embedding, embedding2: Embedding): number {
    if (embedding1.length !== embedding2.length) {
      throw new Error('Embeddings must have same dimensions');
    }

    let sum = 0;
    for (let i = 0; i < embedding1.length; i++) {
      const diff = embedding1[i] - embedding2[i];
      sum += diff * diff;
    }

    return Math.sqrt(sum);
  }

  /**
   * Find most similar embeddings to query
   *
   * @param queryEmbedding - Query embedding
   * @param candidateEmbeddings - Array of candidate embeddings
   * @param limit - Max results to return
   * @returns Indices of most similar embeddings, sorted by similarity
   */
  findMostSimilar(
    queryEmbedding: Embedding,
    candidateEmbeddings: Embedding[],
    limit: number = 5
  ): Array<{ index: number; similarity: number }> {
    // Calculate similarities
    const similarities = candidateEmbeddings.map((embedding, index) => ({
      index,
      similarity: this.cosineSimilarity(queryEmbedding, embedding),
    }));

    // Sort by similarity (descending)
    similarities.sort((a, b) => b.similarity - a.similarity);

    // Return top k
    return similarities.slice(0, limit);
  }

  /**
   * Get statistics
   */
  getStats(): {
    total_embeddings: number;
    total_time_ms: number;
    avg_time_ms: number;
    model: string;
    dimensions: number;
  } {
    return {
      total_embeddings: this.totalEmbeddings,
      total_time_ms: this.totalTimeMs,
      avg_time_ms: this.totalEmbeddings > 0 ? this.totalTimeMs / this.totalEmbeddings : 0,
      model: this.model,
      dimensions: this.dimensions,
    };
  }

  /**
   * Reset statistics
   */
  resetStats(): void {
    this.totalEmbeddings = 0;
    this.totalTimeMs = 0;
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let globalEmbeddingAdapter: EmbeddingAdapter | null = null;

/**
 * Get global embedding adapter (singleton)
 */
export function getGlobalEmbeddingAdapter(): EmbeddingAdapter {
  if (!globalEmbeddingAdapter) {
    globalEmbeddingAdapter = new EmbeddingAdapter();
  }
  return globalEmbeddingAdapter;
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Quick embed helper
 */
export async function embed(text: string): Promise<Embedding> {
  const adapter = getGlobalEmbeddingAdapter();
  const result = await adapter.embed(text);
  return result.embedding;
}

/**
 * Quick similarity helper
 */
export function similarity(embedding1: Embedding, embedding2: Embedding): number {
  const adapter = getGlobalEmbeddingAdapter();
  return adapter.cosineSimilarity(embedding1, embedding2);
}
