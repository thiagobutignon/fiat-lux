/**
 * Advanced O(1) Optimizations for The Regent
 *
 * This module provides advanced data structures and algorithms
 * to achieve constant-time or near-constant-time operations.
 *
 * Key Optimizations:
 * 1. Bloom Filters - Fast negative lookups (O(k) where k << n)
 * 2. Trie Structure - Prefix-based concept search
 * 3. Pre-computed Aggregations - Stats without iteration
 * 4. Lazy Iterators - Avoid materializing large arrays
 * 5. Hash-based Deduplication - O(1) duplicate detection
 */

// ============================================================================
// Bloom Filter - Probabilistic Set Membership
// ============================================================================

/**
 * Bloom Filter for fast "definitely not in set" checks
 *
 * Use Case: Quickly reject irrelevant slices without checking indexes
 *
 * Complexity:
 * - Add: O(k) where k = number of hash functions (typically 3-5)
 * - Check: O(k) - MUCH faster than O(n) linear search
 * - Space: O(m) where m = bit array size (very compact)
 *
 * Trade-off:
 * - False positives possible (says "maybe" when should be "no")
 * - False negatives NEVER (if says "no", definitely not there)
 */
export class BloomFilter {
  private bitArray: Uint8Array;
  private size: number;
  private hashCount: number;

  constructor(expectedElements: number = 1000, falsePositiveRate: number = 0.01) {
    // Optimal bit array size: m = -(n * ln(p)) / (ln(2)^2)
    this.size = Math.ceil(
      -(expectedElements * Math.log(falsePositiveRate)) / (Math.LN2 * Math.LN2)
    );

    // Optimal number of hash functions: k = (m/n) * ln(2)
    this.hashCount = Math.ceil((this.size / expectedElements) * Math.LN2);

    // Bit array stored as bytes (8 bits per byte)
    this.bitArray = new Uint8Array(Math.ceil(this.size / 8));
  }

  /**
   * Add element to bloom filter (O(k) where k = hash count)
   */
  add(element: string): void {
    const hashes = this.getHashes(element);
    for (const hash of hashes) {
      const bitIndex = hash % this.size;
      const byteIndex = Math.floor(bitIndex / 8);
      const bitOffset = bitIndex % 8;
      this.bitArray[byteIndex] |= 1 << bitOffset;
    }
  }

  /**
   * Check if element MIGHT be in set (O(k))
   * Returns:
   * - false: DEFINITELY not in set
   * - true: MAYBE in set (need to check actual index)
   */
  mightContain(element: string): boolean {
    const hashes = this.getHashes(element);
    for (const hash of hashes) {
      const bitIndex = hash % this.size;
      const byteIndex = Math.floor(bitIndex / 8);
      const bitOffset = bitIndex % 8;
      if ((this.bitArray[byteIndex] & (1 << bitOffset)) === 0) {
        return false; // Definitely not in set
      }
    }
    return true; // Maybe in set
  }

  /**
   * Generate k hash values for element
   */
  private getHashes(element: string): number[] {
    const hashes: number[] = [];

    // Use two independent hash functions and combine them
    // This is more efficient than computing k independent hashes
    const hash1 = this.hash(element, 0);
    const hash2 = this.hash(element, hash1);

    for (let i = 0; i < this.hashCount; i++) {
      // Double hashing: h_i = h1 + i * h2
      hashes.push(Math.abs(hash1 + i * hash2));
    }

    return hashes;
  }

  /**
   * Simple hash function (FNV-1a)
   */
  private hash(str: string, seed: number = 0): number {
    let hash = 2166136261 ^ seed; // FNV offset basis
    for (let i = 0; i < str.length; i++) {
      hash ^= str.charCodeAt(i);
      hash += (hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24);
    }
    return hash >>> 0; // Convert to unsigned 32-bit integer
  }

  /**
   * Get statistics about bloom filter
   */
  getStats() {
    let setBits = 0;
    for (let i = 0; i < this.bitArray.length; i++) {
      // Count set bits in byte
      setBits += this.popCount(this.bitArray[i]);
    }

    const fillRatio = setBits / this.size;
    const estimatedFPR = Math.pow(fillRatio, this.hashCount);

    return {
      size: this.size,
      hashCount: this.hashCount,
      fillRatio,
      estimatedFalsePositiveRate: estimatedFPR,
    };
  }

  /**
   * Count set bits in byte (population count)
   */
  private popCount(byte: number): number {
    byte = byte - ((byte >> 1) & 0x55);
    byte = (byte & 0x33) + ((byte >> 2) & 0x33);
    return ((byte + (byte >> 4)) & 0x0f);
  }
}

// ============================================================================
// Trie - Prefix-based Search Tree
// ============================================================================

/**
 * Trie (Prefix Tree) for fast concept lookups
 *
 * Use Case: Find all concepts starting with "depen..." in O(m) where m = prefix length
 *
 * Complexity:
 * - Insert: O(m) where m = word length
 * - Search: O(m)
 * - Prefix search: O(m + k) where k = number of results
 *
 * Much better than:
 * - Linear scan: O(n * m) where n = total concepts
 */
class TrieNode {
  children: Map<string, TrieNode> = new Map();
  isEndOfWord: boolean = false;
  sliceIds: Set<string> = new Set(); // Slices containing this concept
  metadata?: any; // Additional data (frequency, etc.)
}

export class ConceptTrie {
  private root: TrieNode = new TrieNode();
  private size: number = 0;

  /**
   * Insert concept with associated slice IDs (O(m))
   */
  insert(concept: string, sliceId: string): void {
    concept = concept.toLowerCase();
    let node = this.root;

    for (const char of concept) {
      if (!node.children.has(char)) {
        node.children.set(char, new TrieNode());
      }
      node = node.children.get(char)!;
      node.sliceIds.add(sliceId); // All prefixes know about this slice
    }

    if (!node.isEndOfWord) {
      node.isEndOfWord = true;
      this.size++;
    }
  }

  /**
   * Search for exact concept (O(m))
   */
  search(concept: string): Set<string> | null {
    concept = concept.toLowerCase();
    const node = this.findNode(concept);
    return node?.isEndOfWord ? node.sliceIds : null;
  }

  /**
   * Find all concepts with given prefix (O(m + k))
   */
  findByPrefix(prefix: string): Map<string, Set<string>> {
    prefix = prefix.toLowerCase();
    const node = this.findNode(prefix);
    if (!node) return new Map();

    const results = new Map<string, Set<string>>();
    this.collectWords(node, prefix, results);
    return results;
  }

  /**
   * Find node for given string
   */
  private findNode(str: string): TrieNode | null {
    let node = this.root;
    for (const char of str) {
      if (!node.children.has(char)) {
        return null;
      }
      node = node.children.get(char)!;
    }
    return node;
  }

  /**
   * Collect all words from node (DFS)
   */
  private collectWords(
    node: TrieNode,
    prefix: string,
    results: Map<string, Set<string>>
  ): void {
    if (node.isEndOfWord) {
      results.set(prefix, node.sliceIds);
    }

    for (const [char, childNode] of node.children.entries()) {
      this.collectWords(childNode, prefix + char, results);
    }
  }

  /**
   * Autocomplete suggestions (top N by frequency)
   */
  autocomplete(prefix: string, limit: number = 5): string[] {
    const matches = this.findByPrefix(prefix);

    // Sort by number of slices (popularity) and take top N
    return Array.from(matches.keys())
      .sort((a, b) => matches.get(b)!.size - matches.get(a)!.size)
      .slice(0, limit);
  }

  getSize(): number {
    return this.size;
  }
}

// ============================================================================
// Pre-computed Aggregations
// ============================================================================

/**
 * Incremental Statistics Tracker
 *
 * Maintains statistics in O(1) by updating incrementally
 * instead of recomputing from scratch.
 *
 * Trade-off: Slightly more complex add/remove, but O(1) reads
 */
export class IncrementalStats {
  private count: number = 0;
  private sum: number = 0;
  private sumSquares: number = 0;
  private min: number = Infinity;
  private max: number = -Infinity;

  /**
   * Add value to statistics (O(1))
   */
  add(value: number): void {
    this.count++;
    this.sum += value;
    this.sumSquares += value * value;
    this.min = Math.min(this.min, value);
    this.max = Math.max(this.max, value);
  }

  /**
   * Get mean (O(1))
   */
  getMean(): number {
    return this.count > 0 ? this.sum / this.count : 0;
  }

  /**
   * Get variance (O(1))
   */
  getVariance(): number {
    if (this.count === 0) return 0;
    const mean = this.getMean();
    return this.sumSquares / this.count - mean * mean;
  }

  /**
   * Get standard deviation (O(1))
   */
  getStdDev(): number {
    return Math.sqrt(this.getVariance());
  }

  /**
   * Get all stats (O(1))
   */
  getStats() {
    return {
      count: this.count,
      mean: this.getMean(),
      stdDev: this.getStdDev(),
      min: this.min,
      max: this.max,
    };
  }
}

// ============================================================================
// Lazy Iterator - Avoid Materializing Large Arrays
// ============================================================================

/**
 * Lazy iterator that yields values on demand
 *
 * Use Case: Process millions of slices without loading all into memory
 *
 * Benefit: Constant memory usage regardless of dataset size
 */
export class LazyIterator<T> implements Iterable<T> {
  constructor(private generator: () => Generator<T>) {}

  /**
   * Make iterable (can use in for...of loops)
   */
  *[Symbol.iterator](): Generator<T> {
    yield* this.generator();
  }

  /**
   * Map operation (lazy)
   */
  map<U>(fn: (item: T) => U): LazyIterator<U> {
    const self = this;
    return new LazyIterator(function* () {
      for (const item of self) {
        yield fn(item);
      }
    });
  }

  /**
   * Filter operation (lazy)
   */
  filter(predicate: (item: T) => boolean): LazyIterator<T> {
    const self = this;
    return new LazyIterator(function* () {
      for (const item of self) {
        if (predicate(item)) {
          yield item;
        }
      }
    });
  }

  /**
   * Take first N elements (early termination)
   */
  take(n: number): LazyIterator<T> {
    const self = this;
    return new LazyIterator(function* () {
      let count = 0;
      for (const item of self) {
        if (count >= n) break;
        yield item;
        count++;
      }
    });
  }

  /**
   * Materialize to array (only when needed!)
   */
  toArray(): T[] {
    return Array.from(this);
  }

  /**
   * Reduce operation (terminal)
   */
  reduce<U>(fn: (acc: U, item: T) => U, initial: U): U {
    let acc = initial;
    for (const item of this) {
      acc = fn(acc, item);
    }
    return acc;
  }
}

// ============================================================================
// Hash-based Deduplication
// ============================================================================

/**
 * Deduplication tracker using hashes
 *
 * Use Case: Detect duplicate queries/responses in O(1)
 *
 * Benefit: No need to compare every element with every other
 */
export class DeduplicationTracker<T> {
  private seen: Set<string> = new Set();
  private hashFn: (item: T) => string;

  constructor(hashFn: (item: T) => string = (item) => JSON.stringify(item)) {
    this.hashFn = hashFn;
  }

  /**
   * Check if item is duplicate (O(1))
   */
  isDuplicate(item: T): boolean {
    const hash = this.hashFn(item);
    return this.seen.has(hash);
  }

  /**
   * Add item to tracker (O(1))
   */
  add(item: T): boolean {
    const hash = this.hashFn(item);
    if (this.seen.has(hash)) {
      return false; // Already seen
    }
    this.seen.add(hash);
    return true; // New item
  }

  /**
   * Get statistics
   */
  getStats() {
    return {
      uniqueItems: this.seen.size,
    };
  }

  clear(): void {
    this.seen.clear();
  }
}

// ============================================================================
// Usage Examples
// ============================================================================

/*
// Example 1: Bloom Filter for Slice Filtering
const sliceFilter = new BloomFilter(10000, 0.01);

// Add all slice IDs
for (const sliceId of allSliceIds) {
  sliceFilter.add(sliceId);
}

// Quick negative check (O(k) instead of O(n))
if (!sliceFilter.mightContain('unknown-slice')) {
  // Definitely not a valid slice, skip expensive index lookup
  return null;
}

// Example 2: Trie for Concept Search
const conceptTrie = new ConceptTrie();

// Build index
conceptTrie.insert('dependency_inversion', 'architecture/solid.yml');
conceptTrie.insert('dependency_injection', 'architecture/di.yml');
conceptTrie.insert('domain_driven_design', 'architecture/ddd.yml');

// Fast prefix search (O(m + k))
const matches = conceptTrie.findByPrefix('depen'); // Returns both dependency_*

// Autocomplete
const suggestions = conceptTrie.autocomplete('dep', 5);
// ['dependency_inversion', 'dependency_injection', ...]

// Example 3: Incremental Stats for Attention Weights
const weightStats = new IncrementalStats();

// Add weights as they come (O(1) each)
weightStats.add(0.85);
weightStats.add(0.92);
weightStats.add(0.73);

// Get stats instantly (O(1))
console.log(weightStats.getStats());
// { mean: 0.833, stdDev: 0.095, min: 0.73, max: 0.92, count: 3 }

// Example 4: Lazy Iterator for Large Datasets
const allSlices = new LazyIterator(function* () {
  for (const file of sliceFiles) {
    yield loadSlice(file); // Only loaded when iterated
  }
});

// Process without loading everything into memory
const relevantSlices = allSlices
  .filter(slice => slice.domain === 'architecture')
  .map(slice => slice.concepts)
  .take(10) // Only process first 10!
  .toArray();

// Example 5: Deduplication of Queries
const queryDedup = new DeduplicationTracker<string>();

if (!queryDedup.isDuplicate(query)) {
  // First time seeing this query
  const result = await processQuery(query);
  queryDedup.add(query);
  return result;
} else {
  // Duplicate query, use cache
  return getCachedResult(query);
}
*/

export {
  // Re-export for convenience
  BloomFilter,
  ConceptTrie,
  IncrementalStats,
  LazyIterator,
  DeduplicationTracker,
};
