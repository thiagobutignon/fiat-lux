/**
 * Slice Navigator
 *
 * Dynamic knowledge navigation system for AGI agents.
 * Instead of loading all knowledge upfront, agents discover and load
 * relevant "slices" on demand.
 *
 * Key Concepts:
 * - SLICE: Self-contained piece of knowledge with metadata
 * - NAVIGATOR: Indexes and retrieves slices dynamically
 * - CONNECTIONS: Slices link to related concepts in other domains
 *
 * Benefits:
 * - Scalable: Add knowledge without modifying agent prompts
 * - Discoverable: Agents find relevant knowledge through search
 * - Composable: Slices connect across domains
 * - Cacheable: Frequently used slices stay in memory
 */

import fs from 'fs';
import path from 'path';
import yaml from 'yaml';
import { BloomFilter, ConceptTrie } from './o1-advanced-optimizer.js';

// ============================================================================
// Types
// ============================================================================

export interface SliceMetadata {
  id: string;
  domain: string;
  title: string;
  description: string;
  concepts: string[];
  connects_to: Record<string, string>; // { domain: slice_id }
  tags: string[];
  version: string;
  author?: string;
}

export interface SliceContent {
  metadata: SliceMetadata;
  knowledge: string;
  examples?: string[];
  references?: string[];
  formulas?: Record<string, string>;
  principles?: string[];
}

export interface SliceContext {
  slice: SliceContent;
  related_slices: SliceMetadata[];
  connection_graph: Map<string, string[]>;
}

export interface SearchResult {
  slice_id: string;
  relevance_score: number;
  matched_concepts: string[];
  metadata: SliceMetadata;
}

export interface ConnectionPath {
  from: string;
  to: string;
  path: string[];
  shared_concepts: string[];
}

// ============================================================================
// Slice Navigator
// ============================================================================

export class SliceNavigator {
  private index: Map<string, SliceMetadata>;
  private sliceCache: Map<string, SliceContent>;
  private slicesDirectory: string;
  private conceptIndex: Map<string, Set<string>>; // concept -> slice_ids
  private domainIndex: Map<string, Set<string>>; // domain -> slice_ids
  private sliceBloomFilter: BloomFilter; // O(k) slice existence check
  private conceptTrie: ConceptTrie; // O(m+k) prefix-based concept search

  constructor(slicesDirectory: string) {
    this.index = new Map();
    this.sliceCache = new Map();
    this.slicesDirectory = slicesDirectory;
    this.conceptIndex = new Map();
    this.domainIndex = new Map();
    this.sliceBloomFilter = new BloomFilter(10000, 0.01); // 10k slices, 1% FPR
    this.conceptTrie = new ConceptTrie();
  }

  /**
   * Initialize navigator by scanning slices directory
   */
  async initialize(): Promise<void> {
    await this.scanSlicesDirectory();
    this.buildConceptIndex();
    this.buildDomainIndex();
  }

  /**
   * Scan directory for .slice.yaml files
   */
  private async scanSlicesDirectory(): Promise<void> {
    const scanDir = (dir: string) => {
      if (!fs.existsSync(dir)) {
        console.warn(`Slices directory not found: ${dir}`);
        return;
      }

      const entries = fs.readdirSync(dir, { withFileTypes: true });

      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);

        if (entry.isDirectory()) {
          scanDir(fullPath);
        } else if (entry.name.endsWith('.slice.yaml') || entry.name.endsWith('.slice.yml')) {
          this.loadSliceMetadata(fullPath);
        }
      }
    };

    scanDir(this.slicesDirectory);
  }

  /**
   * Load slice metadata (not full content) for indexing
   */
  private loadSliceMetadata(filePath: string): void {
    try {
      const content = fs.readFileSync(filePath, 'utf8');
      const parsed = yaml.parse(content);

      if (parsed.metadata) {
        const metadata: SliceMetadata = {
          id: parsed.metadata.id,
          domain: parsed.metadata.domain,
          title: parsed.metadata.title || '',
          description: parsed.metadata.description || '',
          concepts: parsed.metadata.concepts || [],
          connects_to: parsed.metadata.connects_to || {},
          tags: parsed.metadata.tags || [],
          version: parsed.metadata.version || '1.0',
          author: parsed.metadata.author,
        };

        this.index.set(metadata.id, metadata);

        // O(k) bloom filter insertion for fast existence checks
        this.sliceBloomFilter.add(metadata.id);

        // O(m*c) trie insertion for prefix-based concept search
        for (const concept of metadata.concepts) {
          this.conceptTrie.insert(concept, metadata.id);
        }
      }
    } catch (error) {
      console.error(`Failed to load slice metadata from ${filePath}:`, error);
    }
  }

  /**
   * Build inverted index: concept -> slice_ids
   */
  private buildConceptIndex(): void {
    for (const [sliceId, metadata] of this.index.entries()) {
      for (const concept of metadata.concepts) {
        const normalizedConcept = concept.toLowerCase();

        if (!this.conceptIndex.has(normalizedConcept)) {
          this.conceptIndex.set(normalizedConcept, new Set());
        }

        this.conceptIndex.get(normalizedConcept)!.add(sliceId);
      }
    }
  }

  /**
   * Build domain index: domain -> slice_ids
   */
  private buildDomainIndex(): void {
    for (const [sliceId, metadata] of this.index.entries()) {
      const domain = metadata.domain.toLowerCase();

      if (!this.domainIndex.has(domain)) {
        this.domainIndex.set(domain, new Set());
      }

      this.domainIndex.get(domain)!.add(sliceId);
    }
  }

  /**
   * Fast slice existence check using Bloom Filter (O(k))
   *
   * Returns false if slice DEFINITELY doesn't exist.
   * Returns true if slice MIGHT exist (need to check index).
   */
  mightHaveSlice(sliceId: string): boolean {
    return this.sliceBloomFilter.mightContain(sliceId);
  }

  /**
   * Load full slice content (with caching)
   */
  async loadSlice(sliceId: string): Promise<SliceContext> {
    // O(k) Bloom filter check - fast rejection of invalid slices
    if (!this.mightHaveSlice(sliceId)) {
      throw new Error(`Slice not found: ${sliceId}`);
    }

    // Check cache first
    if (this.sliceCache.has(sliceId)) {
      const slice = this.sliceCache.get(sliceId)!;
      return this.buildSliceContext(slice);
    }

    // Load from disk
    const metadata = this.index.get(sliceId);
    if (!metadata) {
      throw new Error(`Slice not found: ${sliceId}`);
    }

    const slicePath = this.findSliceFile(sliceId);
    if (!slicePath) {
      throw new Error(`Slice file not found for: ${sliceId}`);
    }

    const content = fs.readFileSync(slicePath, 'utf8');
    const parsed = yaml.parse(content);

    const slice: SliceContent = {
      metadata,
      knowledge: parsed.knowledge || '',
      examples: parsed.examples || [],
      references: parsed.references || [],
      formulas: parsed.formulas || {},
      principles: parsed.principles || [],
    };

    // Cache it
    this.sliceCache.set(sliceId, slice);

    return this.buildSliceContext(slice);
  }

  /**
   * Find slice file path
   */
  private findSliceFile(sliceId: string): string | null {
    const findFile = (dir: string): string | null => {
      if (!fs.existsSync(dir)) return null;

      const entries = fs.readdirSync(dir, { withFileTypes: true });

      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);

        if (entry.isDirectory()) {
          const found = findFile(fullPath);
          if (found) return found;
        } else if (
          (entry.name.endsWith('.slice.yaml') || entry.name.endsWith('.slice.yml')) &&
          entry.name.includes(sliceId)
        ) {
          return fullPath;
        }
      }

      return null;
    };

    return findFile(this.slicesDirectory);
  }

  /**
   * Build slice context with related slices
   */
  private buildSliceContext(slice: SliceContent): SliceContext {
    const relatedSlices: SliceMetadata[] = [];
    const connectionGraph = new Map<string, string[]>();

    // Get slices this slice connects to
    for (const [domain, targetSliceId] of Object.entries(slice.metadata.connects_to)) {
      const targetMetadata = this.index.get(targetSliceId);
      if (targetMetadata) {
        relatedSlices.push(targetMetadata);

        if (!connectionGraph.has(slice.metadata.id)) {
          connectionGraph.set(slice.metadata.id, []);
        }
        connectionGraph.get(slice.metadata.id)!.push(targetSliceId);
      }
    }

    // Get slices that connect to this one
    for (const [otherSliceId, otherMetadata] of this.index.entries()) {
      if (Object.values(otherMetadata.connects_to).includes(slice.metadata.id)) {
        relatedSlices.push(otherMetadata);

        if (!connectionGraph.has(otherSliceId)) {
          connectionGraph.set(otherSliceId, []);
        }
        connectionGraph.get(otherSliceId)!.push(slice.metadata.id);
      }
    }

    return {
      slice,
      related_slices: relatedSlices,
      connection_graph: connectionGraph,
    };
  }

  /**
   * Search for slices by concept
   *
   * Enhanced with O(m+k) Trie-based prefix search
   */
  async search(concept: string): Promise<SearchResult[]> {
    const normalizedConcept = concept.toLowerCase();
    const results: SearchResult[] = [];

    // Exact concept match (O(1) hash lookup)
    const exactMatches = this.conceptIndex.get(normalizedConcept);
    if (exactMatches) {
      for (const sliceId of exactMatches) {
        const metadata = this.index.get(sliceId)!;
        results.push({
          slice_id: sliceId,
          relevance_score: 1.0,
          matched_concepts: [concept],
          metadata,
        });
      }
    }

    // O(m + k) Trie-based prefix search (much faster than O(n) linear scan)
    const prefixMatches = this.conceptTrie.findByPrefix(normalizedConcept);
    for (const [matchedConcept, sliceIds] of prefixMatches.entries()) {
      for (const sliceId of sliceIds) {
        // Avoid duplicates from exact match
        if (results.some((r) => r.slice_id === sliceId)) continue;

        const metadata = this.index.get(sliceId)!;
        results.push({
          slice_id: sliceId,
          relevance_score: 0.8,
          matched_concepts: [matchedConcept],
          metadata,
        });
      }
    }

    // Sort by relevance
    results.sort((a, b) => b.relevance_score - a.relevance_score);

    return results;
  }

  /**
   * Get autocomplete suggestions for concept prefix (O(m + k))
   *
   * Uses Trie for efficient prefix matching
   */
  autocomplete(prefix: string, limit: number = 5): string[] {
    return this.conceptTrie.autocomplete(prefix, limit);
  }

  /**
   * Search concepts by exact prefix (O(m + k))
   *
   * Returns all concepts starting with the given prefix
   */
  searchByPrefix(prefix: string): Map<string, Set<string>> {
    return this.conceptTrie.findByPrefix(prefix);
  }

  /**
   * Search slices by domain
   */
  async searchByDomain(domain: string): Promise<SliceMetadata[]> {
    const normalizedDomain = domain.toLowerCase();
    const sliceIds = this.domainIndex.get(normalizedDomain);

    if (!sliceIds) return [];

    return Array.from(sliceIds).map((id) => this.index.get(id)!);
  }

  /**
   * Find connection path between two slices
   */
  async findConnections(sliceA: string, sliceB: string): Promise<ConnectionPath | null> {
    const metadataA = this.index.get(sliceA);
    const metadataB = this.index.get(sliceB);

    if (!metadataA || !metadataB) return null;

    // Direct connection?
    if (Object.values(metadataA.connects_to).includes(sliceB)) {
      return {
        from: sliceA,
        to: sliceB,
        path: [sliceA, sliceB],
        shared_concepts: this.findSharedConcepts(metadataA, metadataB),
      };
    }

    // BFS to find shortest path
    const visited = new Set<string>();
    const queue: { id: string; path: string[] }[] = [{ id: sliceA, path: [sliceA] }];

    while (queue.length > 0) {
      const { id, path } = queue.shift()!;

      if (id === sliceB) {
        return {
          from: sliceA,
          to: sliceB,
          path,
          shared_concepts: this.findSharedConcepts(metadataA, metadataB),
        };
      }

      if (visited.has(id)) continue;
      visited.add(id);

      const metadata = this.index.get(id);
      if (!metadata) continue;

      for (const connectedId of Object.values(metadata.connects_to)) {
        if (!visited.has(connectedId)) {
          queue.push({ id: connectedId, path: [...path, connectedId] });
        }
      }
    }

    return null;
  }

  /**
   * Find shared concepts between two slices
   */
  private findSharedConcepts(metadataA: SliceMetadata, metadataB: SliceMetadata): string[] {
    const conceptsA = new Set(metadataA.concepts.map((c) => c.toLowerCase()));
    const conceptsB = new Set(metadataB.concepts.map((c) => c.toLowerCase()));

    return Array.from(conceptsA).filter((c) => conceptsB.has(c));
  }

  /**
   * Get all slices in index
   */
  getAllSlices(): SliceMetadata[] {
    return Array.from(this.index.values());
  }

  /**
   * Get statistics
   */
  getStats(): {
    total_slices: number;
    total_concepts: number;
    domains: number;
    cache_size: number;
  } {
    return {
      total_slices: this.index.size,
      total_concepts: this.conceptIndex.size,
      domains: this.domainIndex.size,
      cache_size: this.sliceCache.size,
    };
  }

  /**
   * Clear cache
   */
  clearCache(): void {
    this.sliceCache.clear();
  }
}
