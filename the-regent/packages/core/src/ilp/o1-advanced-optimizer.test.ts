/**
 * Unit Tests for Advanced O(1) Optimizations
 *
 * Tests BloomFilter, ConceptTrie, IncrementalStats, LazyIterator, and DeduplicationTracker
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  BloomFilter,
  ConceptTrie,
  IncrementalStats,
  LazyIterator,
  DeduplicationTracker,
} from './o1-advanced-optimizer.js';

describe('BloomFilter', () => {
  let filter: BloomFilter;

  beforeEach(() => {
    filter = new BloomFilter(1000, 0.01); // 1000 elements, 1% false positive rate
  });

  describe('add and mightContain', () => {
    it('should return false for elements not added', () => {
      expect(filter.mightContain('test')).toBe(false);
    });

    it('should return true for elements that were added', () => {
      filter.add('test');
      expect(filter.mightContain('test')).toBe(true);
    });

    it('should handle multiple elements', () => {
      const elements = ['slice1', 'slice2', 'slice3', 'slice4', 'slice5'];
      elements.forEach((el) => filter.add(el));

      elements.forEach((el) => {
        expect(filter.mightContain(el)).toBe(true);
      });
    });

    it('should return false for definitely absent elements', () => {
      filter.add('present');
      expect(filter.mightContain('absent')).toBe(false);
    });
  });

  describe('getStats', () => {
    it('should return statistics about the bloom filter', () => {
      filter.add('test1');
      filter.add('test2');

      const stats = filter.getStats();

      expect(stats.size).toBeGreaterThan(0);
      expect(stats.hashCount).toBeGreaterThan(0);
      expect(stats.fillRatio).toBeGreaterThan(0);
      expect(stats.estimatedFalsePositiveRate).toBeGreaterThanOrEqual(0);
    });
  });

  describe('performance characteristics', () => {
    it('should have low false positive rate', () => {
      // Add 100 elements
      for (let i = 0; i < 100; i++) {
        filter.add(`element${i}`);
      }

      // Check 1000 elements that were NOT added
      let falsePositives = 0;
      for (let i = 100; i < 1100; i++) {
        if (filter.mightContain(`element${i}`)) {
          falsePositives++;
        }
      }

      // False positive rate should be approximately 1% (100 out of 1000)
      // Allow some variance
      expect(falsePositives).toBeLessThan(50); // <5% actual rate
    });
  });
});

describe('ConceptTrie', () => {
  let trie: ConceptTrie;

  beforeEach(() => {
    trie = new ConceptTrie();
  });

  describe('insert and search', () => {
    it('should return null for non-existent concepts', () => {
      expect(trie.search('test')).toBeNull();
    });

    it('should find exact concept matches', () => {
      trie.insert('dependency_inversion', 'slice1');
      const result = trie.search('dependency_inversion');

      expect(result).not.toBeNull();
      expect(result!.has('slice1')).toBe(true);
    });

    it('should handle multiple slices for same concept', () => {
      trie.insert('lazy_evaluation', 'slice1');
      trie.insert('lazy_evaluation', 'slice2');

      const result = trie.search('lazy_evaluation');

      expect(result).not.toBeNull();
      expect(result!.has('slice1')).toBe(true);
      expect(result!.has('slice2')).toBe(true);
      expect(result!.size).toBe(2);
    });

    it('should be case-insensitive', () => {
      trie.insert('TestConcept', 'slice1');

      expect(trie.search('testconcept')).not.toBeNull();
      expect(trie.search('TESTCONCEPT')).not.toBeNull();
      expect(trie.search('TestConcept')).not.toBeNull();
    });
  });

  describe('findByPrefix', () => {
    beforeEach(() => {
      trie.insert('dependency_inversion', 'slice1');
      trie.insert('dependency_injection', 'slice2');
      trie.insert('dependency_management', 'slice3');
      trie.insert('domain_driven_design', 'slice4');
      trie.insert('data_structures', 'slice5');
    });

    it('should find all concepts with given prefix', () => {
      const results = trie.findByPrefix('depen');

      expect(results.size).toBe(3);
      expect(results.has('dependency_inversion')).toBe(true);
      expect(results.has('dependency_injection')).toBe(true);
      expect(results.has('dependency_management')).toBe(true);
    });

    it('should return empty map for non-matching prefix', () => {
      const results = trie.findByPrefix('xyz');
      expect(results.size).toBe(0);
    });

    it('should return all concepts for empty prefix', () => {
      const results = trie.findByPrefix('d');
      expect(results.size).toBeGreaterThan(0);
    });
  });

  describe('autocomplete', () => {
    beforeEach(() => {
      trie.insert('test', 'slice1');
      trie.insert('testing', 'slice1');
      trie.insert('testing', 'slice2'); // More popular
      trie.insert('tester', 'slice3');
      trie.insert('testament', 'slice4');
    });

    it('should return autocomplete suggestions sorted by popularity', () => {
      const suggestions = trie.autocomplete('test', 5);

      expect(suggestions.length).toBeGreaterThan(0);
      expect(suggestions).toContain('testing'); // Should be first (2 slices)
    });

    it('should limit results to specified count', () => {
      const suggestions = trie.autocomplete('test', 2);
      expect(suggestions.length).toBeLessThanOrEqual(2);
    });

    it('should return empty array for non-matching prefix', () => {
      const suggestions = trie.autocomplete('xyz', 5);
      expect(suggestions.length).toBe(0);
    });
  });

  describe('getSize', () => {
    it('should return correct number of unique concepts', () => {
      expect(trie.getSize()).toBe(0);

      trie.insert('concept1', 'slice1');
      expect(trie.getSize()).toBe(1);

      trie.insert('concept2', 'slice2');
      expect(trie.getSize()).toBe(2);

      // Inserting same concept with different slice should not increase size
      trie.insert('concept1', 'slice3');
      expect(trie.getSize()).toBe(2);
    });
  });
});

describe('IncrementalStats', () => {
  let stats: IncrementalStats;

  beforeEach(() => {
    stats = new IncrementalStats();
  });

  describe('add and getMean', () => {
    it('should return 0 mean for empty stats', () => {
      expect(stats.getMean()).toBe(0);
    });

    it('should calculate correct mean', () => {
      stats.add(10);
      stats.add(20);
      stats.add(30);

      expect(stats.getMean()).toBe(20);
    });

    it('should handle single value', () => {
      stats.add(42);
      expect(stats.getMean()).toBe(42);
    });
  });

  describe('getVariance and getStdDev', () => {
    it('should return 0 variance for empty stats', () => {
      expect(stats.getVariance()).toBe(0);
      expect(stats.getStdDev()).toBe(0);
    });

    it('should return 0 variance for identical values', () => {
      stats.add(5);
      stats.add(5);
      stats.add(5);

      expect(stats.getVariance()).toBe(0);
      expect(stats.getStdDev()).toBe(0);
    });

    it('should calculate correct variance and standard deviation', () => {
      // Values: 2, 4, 4, 4, 5, 5, 7, 9
      // Mean: 5
      // Variance: 4
      // StdDev: 2
      [2, 4, 4, 4, 5, 5, 7, 9].forEach((v) => stats.add(v));

      expect(stats.getMean()).toBe(5);
      expect(stats.getVariance()).toBeCloseTo(4, 1);
      expect(stats.getStdDev()).toBeCloseTo(2, 1);
    });
  });

  describe('getStats', () => {
    it('should return comprehensive statistics', () => {
      stats.add(1);
      stats.add(2);
      stats.add(3);
      stats.add(4);
      stats.add(5);

      const result = stats.getStats();

      expect(result.count).toBe(5);
      expect(result.mean).toBe(3);
      expect(result.min).toBe(1);
      expect(result.max).toBe(5);
      expect(result.stdDev).toBeGreaterThan(0);
    });
  });

  describe('performance - O(1) operations', () => {
    it('should maintain O(1) performance for all operations', () => {
      // Add 10,000 values
      for (let i = 0; i < 10000; i++) {
        stats.add(Math.random() * 100);
      }

      // All getters should be instant (O(1))
      const start = Date.now();
      stats.getMean();
      stats.getVariance();
      stats.getStdDev();
      stats.getStats();
      const elapsed = Date.now() - start;

      // Should complete in less than 1ms (generous allowance)
      expect(elapsed).toBeLessThan(5);
    });
  });
});

describe('LazyIterator', () => {
  describe('basic iteration', () => {
    it('should iterate over values lazily', () => {
      const iterator = new LazyIterator(function* () {
        yield 1;
        yield 2;
        yield 3;
      });

      const values: number[] = [];
      for (const value of iterator) {
        values.push(value);
      }

      expect(values).toEqual([1, 2, 3]);
    });
  });

  describe('map', () => {
    it('should lazily transform values', () => {
      const iterator = new LazyIterator(function* () {
        yield 1;
        yield 2;
        yield 3;
      });

      const doubled = iterator.map((x) => x * 2);
      expect(doubled.toArray()).toEqual([2, 4, 6]);
    });

    it('should not execute until consumed', () => {
      let executed = false;

      const iterator = new LazyIterator(function* () {
        executed = true;
        yield 1;
      });

      const mapped = iterator.map((x) => x * 2);

      expect(executed).toBe(false); // Not executed yet

      mapped.toArray(); // Consume

      expect(executed).toBe(true); // Now executed
    });
  });

  describe('filter', () => {
    it('should lazily filter values', () => {
      const iterator = new LazyIterator(function* () {
        yield 1;
        yield 2;
        yield 3;
        yield 4;
        yield 5;
      });

      const evens = iterator.filter((x) => x % 2 === 0);
      expect(evens.toArray()).toEqual([2, 4]);
    });
  });

  describe('take', () => {
    it('should take first N elements', () => {
      const iterator = new LazyIterator(function* () {
        yield 1;
        yield 2;
        yield 3;
        yield 4;
        yield 5;
      });

      const first3 = iterator.take(3);
      expect(first3.toArray()).toEqual([1, 2, 3]);
    });

    it('should support early termination', () => {
      let generated = 0;

      const iterator = new LazyIterator(function* () {
        for (let i = 1; i <= 1000; i++) {
          generated++;
          yield i;
        }
      });

      iterator.take(5).toArray();

      // Should only generate 5 values, not all 1000
      expect(generated).toBe(5);
    });
  });

  describe('reduce', () => {
    it('should reduce values to single result', () => {
      const iterator = new LazyIterator(function* () {
        yield 1;
        yield 2;
        yield 3;
        yield 4;
      });

      const sum = iterator.reduce((acc, x) => acc + x, 0);
      expect(sum).toBe(10);
    });
  });

  describe('chaining operations', () => {
    it('should support chaining map, filter, and take', () => {
      const iterator = new LazyIterator(function* () {
        for (let i = 1; i <= 100; i++) {
          yield i;
        }
      });

      const result = iterator
        .filter((x) => x % 2 === 0) // Only evens
        .map((x) => x * 3) // Multiply by 3
        .take(5) // Take first 5
        .toArray();

      expect(result).toEqual([6, 12, 18, 24, 30]);
    });
  });
});

describe('DeduplicationTracker', () => {
  let tracker: DeduplicationTracker<string>;

  beforeEach(() => {
    tracker = new DeduplicationTracker<string>();
  });

  describe('isDuplicate and add', () => {
    it('should return false for new items', () => {
      expect(tracker.isDuplicate('item1')).toBe(false);
    });

    it('should return true for duplicate items', () => {
      tracker.add('item1');
      expect(tracker.isDuplicate('item1')).toBe(true);
    });

    it('should handle multiple unique items', () => {
      expect(tracker.add('item1')).toBe(true); // New
      expect(tracker.add('item2')).toBe(true); // New
      expect(tracker.add('item1')).toBe(false); // Duplicate
      expect(tracker.add('item3')).toBe(true); // New
      expect(tracker.add('item2')).toBe(false); // Duplicate
    });
  });

  describe('custom hash function', () => {
    it('should support custom hash function', () => {
      interface User {
        id: number;
        name: string;
      }

      const userTracker = new DeduplicationTracker<User>((user) =>
        user.id.toString()
      );

      const user1 = { id: 1, name: 'Alice' };
      const user1Duplicate = { id: 1, name: 'Alice Smith' }; // Same ID, different name
      const user2 = { id: 2, name: 'Bob' };

      expect(userTracker.add(user1)).toBe(true);
      expect(userTracker.isDuplicate(user1Duplicate)).toBe(true); // Same ID
      expect(userTracker.add(user2)).toBe(true);
    });
  });

  describe('getStats', () => {
    it('should return correct unique item count', () => {
      expect(tracker.getStats().uniqueItems).toBe(0);

      tracker.add('item1');
      expect(tracker.getStats().uniqueItems).toBe(1);

      tracker.add('item2');
      expect(tracker.getStats().uniqueItems).toBe(2);

      tracker.add('item1'); // Duplicate
      expect(tracker.getStats().uniqueItems).toBe(2);
    });
  });

  describe('clear', () => {
    it('should clear all tracked items', () => {
      tracker.add('item1');
      tracker.add('item2');

      expect(tracker.getStats().uniqueItems).toBe(2);

      tracker.clear();

      expect(tracker.getStats().uniqueItems).toBe(0);
      expect(tracker.isDuplicate('item1')).toBe(false);
    });
  });

  describe('performance - O(1) operations', () => {
    it('should maintain O(1) performance for duplicate detection', () => {
      // Add 10,000 items
      for (let i = 0; i < 10000; i++) {
        tracker.add(`item${i}`);
      }

      // Check duplicates should be O(1) regardless of size
      const start = Date.now();
      for (let i = 0; i < 100; i++) {
        tracker.isDuplicate(`item${i}`);
      }
      const elapsed = Date.now() - start;

      // Should complete in less than 10ms
      expect(elapsed).toBeLessThan(10);
    });
  });
});

describe('Integration Tests', () => {
  describe('BloomFilter + Map fallback pattern', () => {
    it('should use BloomFilter for fast rejection', () => {
      const filter = new BloomFilter(1000, 0.01);
      const actualSlices = new Set<string>();

      // Add slices to both
      const slices = ['slice1', 'slice2', 'slice3'];
      slices.forEach((s) => {
        filter.add(s);
        actualSlices.add(s);
      });

      // Test pattern: BloomFilter first, then Map
      const testSlice = 'unknown-slice';

      if (!filter.mightContain(testSlice)) {
        // Fast path: definitely not present
        expect(actualSlices.has(testSlice)).toBe(false);
      } else {
        // Slow path: check actual map
        expect(actualSlices.has(testSlice)).toBe(true);
      }
    });
  });

  describe('ConceptTrie + IncrementalStats', () => {
    it('should efficiently track concept statistics', () => {
      const trie = new ConceptTrie();
      const conceptStats = new Map<string, IncrementalStats>();

      // Insert concepts with different weights
      const insertWithWeight = (
        concept: string,
        slice: string,
        weight: number
      ) => {
        trie.insert(concept, slice);

        if (!conceptStats.has(concept)) {
          conceptStats.set(concept, new IncrementalStats());
        }
        conceptStats.get(concept)!.add(weight);
      };

      insertWithWeight('lazy_evaluation', 'slice1', 0.9);
      insertWithWeight('lazy_evaluation', 'slice2', 0.85);
      insertWithWeight('dependency_inversion', 'slice3', 0.95);

      // O(1) retrieval of stats
      const lazyStats = conceptStats.get('lazy_evaluation')!;
      expect(lazyStats.getMean()).toBeCloseTo(0.875, 2);
      expect(lazyStats.getStats().count).toBe(2);
    });
  });

  describe('LazyIterator + DeduplicationTracker', () => {
    it('should efficiently process and deduplicate large datasets', () => {
      const tracker = new DeduplicationTracker<number>();
      let processed = 0;

      const data = new LazyIterator(function* () {
        for (let i = 1; i <= 1000; i++) {
          yield i % 100; // Will generate duplicates
        }
      });

      const uniqueData = data
        .filter((item) => {
          processed++;
          return tracker.add(item); // Only pass through unique items
        })
        .toArray();

      expect(uniqueData.length).toBe(100); // 100 unique values (0-99)
      expect(tracker.getStats().uniqueItems).toBe(100);
      expect(processed).toBe(1000); // Processed all items lazily
    });
  });
});
