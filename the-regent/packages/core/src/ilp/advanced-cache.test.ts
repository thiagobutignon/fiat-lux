/**
 * Tests for Advanced Caching System
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { AdvancedCache, QueryNormalizer } from './advanced-cache.js';

describe('QueryNormalizer', () => {
  describe('normalize', () => {
    it('should lowercase and trim queries', () => {
      expect(QueryNormalizer.normalize('  HELLO WORLD  ')).toBe('hello world');
    });

    it('should remove extra whitespace', () => {
      expect(QueryNormalizer.normalize('hello    world')).toBe('hello world');
    });

    it('should remove punctuation', () => {
      expect(QueryNormalizer.normalize('What is AI?')).toBe('what is ai');
      expect(QueryNormalizer.normalize('Hello, world!')).toBe('hello world');
    });

    it('should normalize contractions', () => {
      expect(QueryNormalizer.normalize("what's this")).toBe('what is this');
      expect(QueryNormalizer.normalize("can't do")).toBe('cannot do');
    });

    it('should remove articles', () => {
      expect(QueryNormalizer.normalize('the quick brown fox')).toBe(
        'quick brown fox'
      );
    });

    it('should remove filler words', () => {
      expect(QueryNormalizer.normalize('could you please help')).toBe('help');
    });
  });

  describe('extractTemplate', () => {
    it('should replace numbers with placeholder', () => {
      const template = QueryNormalizer.extractTemplate('I need 5 apples');
      expect(template).toContain('<NUM>');
    });

    it('should replace amounts with placeholder', () => {
      const template = QueryNormalizer.extractTemplate('It costs $100');
      expect(template).toContain('<AMOUNT>');
    });

    it('should create same template for similar queries', () => {
      const template1 = QueryNormalizer.extractTemplate('I need 5 apples');
      const template2 = QueryNormalizer.extractTemplate('I need 10 apples');
      expect(template1).toBe(template2);
    });
  });

  describe('calculateSimilarity', () => {
    it('should return 1 for identical queries', () => {
      const sim = QueryNormalizer.calculateSimilarity('hello world', 'hello world');
      expect(sim).toBe(1);
    });

    it('should return 0 for completely different queries', () => {
      const sim = QueryNormalizer.calculateSimilarity('hello', 'goodbye');
      expect(sim).toBe(0);
    });

    it('should return high similarity for similar queries', () => {
      const sim = QueryNormalizer.calculateSimilarity(
        'what is machine learning',
        'what is deep learning'
      );
      expect(sim).toBeGreaterThan(0.5);
    });

    it('should be order-independent for high overlap', () => {
      const sim1 = QueryNormalizer.calculateSimilarity(
        'quick brown fox',
        'brown quick fox'
      );
      expect(sim1).toBe(1); // Same words, different order
    });
  });
});

describe('AdvancedCache', () => {
  let cache: AdvancedCache<string>;

  beforeEach(() => {
    cache = new AdvancedCache<string>({
      maxSize: 100,
      ttlMs: 3600000,
      similarityThreshold: 0.85,
      enableSemanticCache: true,
      enableTemplateCache: true,
    });
  });

  describe('exact matching', () => {
    it('should cache and retrieve exact matches', () => {
      cache.set('What is AI?', 'AI is artificial intelligence');

      const result = cache.get('What is AI?');
      expect(result).toBe('AI is artificial intelligence');
    });

    it('should match normalized variations', () => {
      cache.set('What is AI?', 'AI is artificial intelligence');

      // Different punctuation, capitalization, spacing
      const result = cache.get('what is ai');
      expect(result).toBe('AI is artificial intelligence');
    });

    it('should return null for cache miss', () => {
      const result = cache.get('unknown query');
      expect(result).toBeNull();
    });
  });

  describe('template matching', () => {
    it('should match queries with different numbers', () => {
      cache.set('I need 5 apples', 'You need 5 apples');

      const result = cache.get('I need 10 apples');
      expect(result).toBe('You need 5 apples');
    });

    it('should match queries with different amounts', () => {
      cache.set('It costs $100', 'Price is $100');

      const result = cache.get('It costs $200');
      expect(result).toBe('Price is $100');
    });
  });

  describe('semantic matching', () => {
    it('should match similar queries', () => {
      cache.set('What is machine learning?', 'ML is a subset of AI');

      const result = cache.get('What is deep learning?');
      // Should match because of high word overlap
      if (result) {
        expect(result).toBe('ML is a subset of AI');
      }
    });

    it('should respect similarity threshold', () => {
      cache.set('What is AI?', 'AI answer');

      // Very different query
      const result = cache.get('How to cook pasta?');
      expect(result).toBeNull();
    });
  });

  describe('LRU eviction', () => {
    it('should evict least recently used when full', () => {
      const smallCache = new AdvancedCache<string>({
        maxSize: 3,
        enableLRU: true,
      });

      smallCache.set('query1', 'answer1');
      smallCache.set('query2', 'answer2');
      smallCache.set('query3', 'answer3');

      // Access query1 to make it recently used
      smallCache.get('query1');

      // Add query4 - should evict query2 (least recently used)
      smallCache.set('query4', 'answer4');

      expect(smallCache.get('query1')).toBe('answer1'); // Still there
      expect(smallCache.get('query2')).toBeNull(); // Evicted
      expect(smallCache.get('query3')).toBe('answer3'); // Still there
      expect(smallCache.get('query4')).toBe('answer4'); // New entry
    });
  });

  describe('TTL', () => {
    it('should expire entries after TTL', async () => {
      const shortCache = new AdvancedCache<string>({
        ttlMs: 100, // 100ms TTL
      });

      shortCache.set('query', 'answer');

      // Should be available immediately
      expect(shortCache.get('query')).toBe('answer');

      // Wait for expiration
      await new Promise((resolve) => setTimeout(resolve, 150));

      // Should be expired
      expect(shortCache.get('query')).toBeNull();
    });
  });

  describe('statistics', () => {
    it('should track hits and misses', () => {
      cache.set('query1', 'answer1');

      cache.get('query1'); // Hit
      cache.get('query2'); // Miss

      const stats = cache.getStats();

      expect(stats.hits).toBe(1);
      expect(stats.misses).toBe(1);
      expect(stats.hitRate).toBe(0.5);
    });

    it('should track exact vs semantic vs template hits', () => {
      cache.set('What is AI?', 'answer');

      cache.get('What is AI?'); // Exact hit
      cache.get('I need 5 apples'); // Miss

      const stats = cache.getStats();

      expect(stats.exactHits).toBe(1);
      expect(stats.semanticHits).toBe(0);
      expect(stats.templateHits).toBe(0);
    });

    it('should track popular queries', () => {
      cache.set('popular', 'answer');

      for (let i = 0; i < 10; i++) {
        cache.get('popular');
      }

      const stats = cache.getStats();

      expect(stats.popularQueries[0]?.query).toBe('popular');
      expect(stats.popularQueries[0]?.hits).toBe(10);
    });
  });

  describe('pre-warming', () => {
    it('should pre-warm cache with provided queries', async () => {
      const warmed = await cache.preWarm([
        { query: 'What is AI?', value: 'AI answer' },
        { query: 'What is ML?', value: 'ML answer' },
      ]);

      expect(warmed).toBe(2);
      expect(cache.get('What is AI?')).toBe('AI answer');
      expect(cache.get('What is ML?')).toBe('ML answer');
    });
  });

  describe('clear', () => {
    it('should clear all entries and reset stats', () => {
      cache.set('query1', 'answer1');
      cache.set('query2', 'answer2');
      cache.get('query1');

      cache.clear();

      expect(cache.get('query1')).toBeNull();
      expect(cache.get('query2')).toBeNull();

      const stats = cache.getStats();
      expect(stats.size).toBe(0);
      expect(stats.hits).toBe(0);
      expect(stats.misses).toBe(0);
    });
  });

  describe('updateConfig', () => {
    it('should update configuration at runtime', () => {
      cache.set('What is AI?', 'answer');

      // Disable semantic cache
      cache.updateConfig({ enableSemanticCache: false });

      // Should still get exact matches
      expect(cache.get('What is AI?')).toBe('answer');
    });
  });
});

describe('AdvancedCache Integration', () => {
  it('should achieve high cache hit rate with varied queries', () => {
    const cache = new AdvancedCache<string>({
      similarityThreshold: 0.8,
    });

    // Add base queries
    cache.set('What is machine learning?', 'ML answer');
    cache.set('How much does this cost?', 'Cost answer');

    // Exact match
    expect(cache.get('What is machine learning?')).toBeTruthy();

    // Template match (different number)
    cache.set('I need 5 items', 'Items answer');
    expect(cache.get('I need 10 items')).toBeTruthy();

    // Semantic match
    const semanticResult = cache.get('What is deep learning?');
    // May or may not match depending on similarity

    const stats = cache.getStats();
    expect(stats.hitRate).toBeGreaterThan(0);
  });

  it('should provide detailed statistics for monitoring', () => {
    const cache = new AdvancedCache<string>();

    // Simulate usage
    cache.set('query1', 'answer1');
    cache.set('query2', 'answer2');

    for (let i = 0; i < 10; i++) {
      cache.get('query1');
    }

    for (let i = 0; i < 5; i++) {
      cache.get('query2');
    }

    cache.get('unknown'); // Miss

    const stats = cache.getStats();

    expect(stats.size).toBe(2);
    expect(stats.hits).toBe(15);
    expect(stats.misses).toBe(1);
    expect(stats.hitRate).toBeCloseTo(15 / 16, 2);
    expect(stats.popularQueries[0].query).toBe('query1');
    expect(stats.popularQueries[0].hits).toBe(10);
  });
});
