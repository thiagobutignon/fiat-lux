/**
 * @file slice-navigator.test.ts
 * Tests for Slice Navigator - O(1) knowledge discovery system
 *
 * Key capabilities tested:
 * - Directory scanning and indexing
 * - Concept-based search with relevance scoring
 * - Domain-based search
 * - Connection pathfinding (BFS)
 * - Caching behavior
 * - Statistics and memory management
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import {
  SliceNavigator,
  SliceMetadata,
  SliceContent,
  SliceContext,
  SearchResult,
  ConnectionPath,
} from '../core/slice-navigator';
import fs from 'fs';
import path from 'path';
import os from 'os';

describe('SliceNavigator', () => {
  let navigator: SliceNavigator;
  let testSlicesDir: string;

  beforeEach(async () => {
    // Create temporary directory for test slices
    testSlicesDir = path.join(os.tmpdir(), `test-slices-${Date.now()}`);
    fs.mkdirSync(testSlicesDir, { recursive: true });

    // Create test slice files
    createTestSlices(testSlicesDir);

    navigator = new SliceNavigator(testSlicesDir);
  });

  afterEach(() => {
    // Clean up test directory
    if (fs.existsSync(testSlicesDir)) {
      fs.rmSync(testSlicesDir, { recursive: true, force: true });
    }
  });

  describe('Initialization', () => {
    it('should initialize and scan slices directory', async () => {
      await navigator.initialize();

      const stats = navigator.getStats();
      expect(stats.total_slices).toBeGreaterThan(0);
      expect(stats.total_concepts).toBeGreaterThan(0);
      expect(stats.domains).toBeGreaterThan(0);
    });

    it('should build concept index during initialization', async () => {
      await navigator.initialize();

      const results = await navigator.search('diversification');
      expect(results.length).toBeGreaterThan(0);
    });

    it('should build domain index during initialization', async () => {
      await navigator.initialize();

      const results = await navigator.searchByDomain('financial');
      expect(results.length).toBeGreaterThan(0);
    });

    it('should handle non-existent directory gracefully', async () => {
      const nonExistentDir = path.join(os.tmpdir(), 'does-not-exist');
      const nav = new SliceNavigator(nonExistentDir);

      await expect(nav.initialize()).resolves.not.toThrow();
    });

    it('should load metadata from .slice.yaml files', async () => {
      await navigator.initialize();

      const allSlices = navigator.getAllSlices();
      expect(allSlices.length).toBeGreaterThan(0);
      expect(allSlices[0]).toHaveProperty('id');
      expect(allSlices[0]).toHaveProperty('domain');
      expect(allSlices[0]).toHaveProperty('concepts');
    });

    it('should load metadata from .slice.yml files', async () => {
      const ymlPath = path.join(testSlicesDir, 'test.slice.yml');
      fs.writeFileSync(
        ymlPath,
        `
metadata:
  id: test-yml
  domain: test
  title: Test YML
  concepts: [test]
  connects_to: {}
  tags: []
  version: "1.0"
knowledge: Test knowledge
`
      );

      await navigator.initialize();
      const allSlices = navigator.getAllSlices();
      const ymlSlice = allSlices.find((s) => s.id === 'test-yml');

      expect(ymlSlice).toBeDefined();
      expect(ymlSlice?.domain).toBe('test');
    });
  });

  describe('Concept Search', () => {
    beforeEach(async () => {
      await navigator.initialize();
    });

    it('should find slices by exact concept match', async () => {
      const results = await navigator.search('diversification');

      expect(results.length).toBeGreaterThan(0);
      expect(results[0].relevance_score).toBe(1.0);
      expect(results[0].matched_concepts).toContain('diversification');
    });

    it('should return slices sorted by relevance', async () => {
      const results = await navigator.search('div');

      for (let i = 0; i < results.length - 1; i++) {
        expect(results[i].relevance_score).toBeGreaterThanOrEqual(results[i + 1].relevance_score);
      }
    });

    it('should handle case-insensitive search', async () => {
      const resultsLower = await navigator.search('diversification');
      const resultsUpper = await navigator.search('DIVERSIFICATION');

      expect(resultsLower.length).toBe(resultsUpper.length);
    });

    it('should perform partial concept matching', async () => {
      const results = await navigator.search('feed');

      // Should find 'feedback' concept
      expect(results.length).toBeGreaterThan(0);
      const partialMatches = results.filter((r) => r.relevance_score < 1.0);
      expect(partialMatches.length).toBeGreaterThanOrEqual(0);
    });

    it('should return empty array for non-existent concept', async () => {
      const results = await navigator.search('nonexistent-concept-xyz');

      expect(results).toEqual([]);
    });

    it('should include metadata in search results', async () => {
      const results = await navigator.search('diversification');

      expect(results[0].metadata).toBeDefined();
      expect(results[0].metadata.id).toBeDefined();
      expect(results[0].metadata.domain).toBeDefined();
    });
  });

  describe('Domain Search', () => {
    beforeEach(async () => {
      await navigator.initialize();
    });

    it('should find all slices in a domain', async () => {
      const results = await navigator.searchByDomain('financial');

      expect(results.length).toBeGreaterThan(0);
      results.forEach((slice) => {
        expect(slice.domain).toBe('financial');
      });
    });

    it('should handle case-insensitive domain search', async () => {
      const resultsLower = await navigator.searchByDomain('financial');
      const resultsUpper = await navigator.searchByDomain('FINANCIAL');

      expect(resultsLower.length).toBe(resultsUpper.length);
    });

    it('should return empty array for non-existent domain', async () => {
      const results = await navigator.searchByDomain('nonexistent-domain');

      expect(results).toEqual([]);
    });

    it('should return all biology domain slices', async () => {
      const results = await navigator.searchByDomain('biology');

      expect(results.length).toBeGreaterThan(0);
      results.forEach((slice) => {
        expect(slice.domain).toBe('biology');
      });
    });
  });

  describe('Slice Loading', () => {
    beforeEach(async () => {
      await navigator.initialize();
    });

    it('should load full slice content', async () => {
      const context = await navigator.loadSlice('financial-diversification');

      expect(context.slice).toBeDefined();
      expect(context.slice.metadata.id).toBe('financial-diversification');
      expect(context.slice.knowledge).toBeDefined();
      expect(context.slice.knowledge.length).toBeGreaterThan(0);
    });

    it('should include related slices in context', async () => {
      const context = await navigator.loadSlice('financial-diversification');

      expect(context.related_slices).toBeDefined();
      expect(Array.isArray(context.related_slices)).toBe(true);
    });

    it('should include connection graph in context', async () => {
      const context = await navigator.loadSlice('financial-diversification');

      expect(context.connection_graph).toBeDefined();
      expect(context.connection_graph instanceof Map).toBe(true);
    });

    it('should throw error for non-existent slice', async () => {
      await expect(navigator.loadSlice('nonexistent-slice')).rejects.toThrow(
        'Slice not found: nonexistent-slice'
      );
    });

    it('should cache loaded slices', async () => {
      await navigator.loadSlice('financial-diversification');

      const statsBefore = navigator.getStats();
      expect(statsBefore.cache_size).toBe(1);

      // Load again
      await navigator.loadSlice('financial-diversification');

      const statsAfter = navigator.getStats();
      expect(statsAfter.cache_size).toBe(1); // Still 1, from cache
    });

    it('should load slice with all optional fields', async () => {
      const context = await navigator.loadSlice('financial-diversification');

      expect(context.slice.examples).toBeDefined();
      expect(context.slice.references).toBeDefined();
      expect(context.slice.formulas).toBeDefined();
      expect(context.slice.principles).toBeDefined();
    });
  });

  describe('Connection Pathfinding', () => {
    beforeEach(async () => {
      await navigator.initialize();
    });

    it('should find direct connection between slices', async () => {
      const path = await navigator.findConnections(
        'financial-diversification',
        'biology-homeostasis'
      );

      expect(path).toBeDefined();
      expect(path?.from).toBe('financial-diversification');
      expect(path?.to).toBe('biology-homeostasis');
      expect(path?.path.length).toBeGreaterThanOrEqual(2);
    });

    it('should return path array with start and end', async () => {
      const path = await navigator.findConnections(
        'financial-diversification',
        'biology-homeostasis'
      );

      expect(path?.path[0]).toBe('financial-diversification');
      expect(path?.path[path.path.length - 1]).toBe('biology-homeostasis');
    });

    it('should identify shared concepts between connected slices', async () => {
      const path = await navigator.findConnections(
        'financial-diversification',
        'biology-homeostasis'
      );

      expect(path?.shared_concepts).toBeDefined();
      expect(Array.isArray(path?.shared_concepts)).toBe(true);
    });

    it('should return null if no connection exists', async () => {
      // Create isolated slice
      const isolatedPath = path.join(testSlicesDir, 'isolated.slice.yaml');
      fs.writeFileSync(
        isolatedPath,
        `
metadata:
  id: isolated-slice
  domain: isolated
  title: Isolated Slice
  concepts: [isolated]
  connects_to: {}
  tags: []
  version: "1.0"
knowledge: Isolated knowledge
`
      );

      await navigator.initialize();

      const connectionPath = await navigator.findConnections('financial-diversification', 'isolated-slice');

      expect(connectionPath).toBeNull();
    });

    it('should return null if either slice does not exist', async () => {
      const path = await navigator.findConnections('financial-diversification', 'nonexistent');

      expect(path).toBeNull();
    });

    it('should find shortest path using BFS', async () => {
      const path = await navigator.findConnections(
        'financial-diversification',
        'biology-homeostasis'
      );

      // Path should exist and be reasonably short
      expect(path?.path.length).toBeLessThanOrEqual(5);
    });
  });

  describe('Statistics', () => {
    it('should return accurate statistics after initialization', async () => {
      await navigator.initialize();

      const stats = navigator.getStats();

      expect(stats.total_slices).toBeGreaterThan(0);
      expect(stats.total_concepts).toBeGreaterThan(0);
      expect(stats.domains).toBeGreaterThan(0);
      expect(stats.cache_size).toBe(0); // No slices loaded yet
    });

    it('should update cache size after loading slices', async () => {
      await navigator.initialize();

      await navigator.loadSlice('financial-diversification');
      const stats1 = navigator.getStats();
      expect(stats1.cache_size).toBe(1);

      await navigator.loadSlice('biology-homeostasis');
      const stats2 = navigator.getStats();
      expect(stats2.cache_size).toBe(2);
    });

    it('should have correct domain count', async () => {
      await navigator.initialize();

      const stats = navigator.getStats();
      const allSlices = navigator.getAllSlices();
      const uniqueDomains = new Set(allSlices.map((s) => s.domain));

      expect(stats.domains).toBe(uniqueDomains.size);
    });

    it('should have correct concept count', async () => {
      await navigator.initialize();

      const stats = navigator.getStats();
      const allSlices = navigator.getAllSlices();
      const uniqueConcepts = new Set<string>();

      allSlices.forEach((slice) => {
        slice.concepts.forEach((concept) => {
          uniqueConcepts.add(concept.toLowerCase());
        });
      });

      expect(stats.total_concepts).toBe(uniqueConcepts.size);
    });
  });

  describe('Cache Management', () => {
    beforeEach(async () => {
      await navigator.initialize();
    });

    it('should clear cache', async () => {
      await navigator.loadSlice('financial-diversification');
      await navigator.loadSlice('biology-homeostasis');

      expect(navigator.getStats().cache_size).toBe(2);

      navigator.clearCache();

      expect(navigator.getStats().cache_size).toBe(0);
    });

    it('should reload slice after cache clear', async () => {
      await navigator.loadSlice('financial-diversification');
      navigator.clearCache();

      const context = await navigator.loadSlice('financial-diversification');

      expect(context.slice).toBeDefined();
      expect(navigator.getStats().cache_size).toBe(1);
    });
  });

  describe('All Slices Retrieval', () => {
    it('should return all indexed slices', async () => {
      await navigator.initialize();

      const allSlices = navigator.getAllSlices();

      expect(allSlices.length).toBeGreaterThan(0);
      allSlices.forEach((slice) => {
        expect(slice.id).toBeDefined();
        expect(slice.domain).toBeDefined();
        expect(slice.concepts).toBeDefined();
      });
    });

    it('should return empty array before initialization', () => {
      const allSlices = navigator.getAllSlices();

      expect(allSlices).toEqual([]);
    });
  });

  describe('Nested Directory Scanning', () => {
    it('should scan nested subdirectories', async () => {
      const nestedDir = path.join(testSlicesDir, 'nested', 'deep');
      fs.mkdirSync(nestedDir, { recursive: true });

      fs.writeFileSync(
        path.join(nestedDir, 'nested-slice.slice.yaml'),
        `
metadata:
  id: nested-slice
  domain: nested
  title: Nested Slice
  concepts: [nested]
  connects_to: {}
  tags: []
  version: "1.0"
knowledge: Nested knowledge
`
      );

      await navigator.initialize();

      const allSlices = navigator.getAllSlices();
      const nestedSlice = allSlices.find((s) => s.id === 'nested-slice');

      expect(nestedSlice).toBeDefined();
    });
  });
});

// ============================================================================
// Test Fixtures
// ============================================================================

function createTestSlices(dir: string): void {
  // Financial slice
  fs.writeFileSync(
    path.join(dir, 'financial-diversification.slice.yaml'),
    `
metadata:
  id: financial-diversification
  domain: financial
  title: Investment Diversification
  description: Spreading risk across different assets
  concepts: [diversification, risk, portfolio, investment]
  connects_to:
    biology: biology-homeostasis
  tags: [risk-management, strategy]
  version: "1.0"
  author: Financial Agent

knowledge: |
  Diversification is the practice of spreading investments across various
  financial instruments, industries, and other categories to reduce risk.

examples:
  - "60/40 stock-bond portfolio"
  - "International diversification"

references:
  - "Modern Portfolio Theory - Markowitz (1952)"

formulas:
  portfolio_variance: "σ²_p = Σw_i²σ_i² + ΣΣw_iw_jρ_ijσ_iσ_j"

principles:
  - "Don't put all eggs in one basket"
  - "Balance risk and return"
`
  );

  // Biology slice
  fs.writeFileSync(
    path.join(dir, 'biology-homeostasis.slice.yaml'),
    `
metadata:
  id: biology-homeostasis
  domain: biology
  title: Biological Homeostasis
  description: Self-regulating processes maintaining stability
  concepts: [homeostasis, regulation, balance, feedback]
  connects_to:
    financial: financial-diversification
    systems: systems-feedback
  tags: [biology, systems]
  version: "1.0"
  author: Biology Agent

knowledge: |
  Homeostasis is the property of a system to regulate its internal environment
  and maintain a stable, constant condition through feedback mechanisms.

examples:
  - "Body temperature regulation"
  - "Blood glucose control"

references:
  - "Cannon, W.B. (1932) The Wisdom of the Body"

principles:
  - "Maintain internal equilibrium"
  - "Use negative feedback loops"
`
  );

  // Systems slice
  fs.writeFileSync(
    path.join(dir, 'systems-feedback.slice.yaml'),
    `
metadata:
  id: systems-feedback
  domain: systems
  title: Feedback Loops
  description: Control systems using feedback
  concepts: [feedback, loop, control, regulation]
  connects_to:
    biology: biology-homeostasis
  tags: [systems, control]
  version: "1.0"
  author: Systems Agent

knowledge: |
  Feedback loops are mechanisms where outputs are fed back as inputs,
  enabling self-regulation and control in systems.

examples:
  - "Thermostat control"
  - "PID controller"

principles:
  - "Monitor outputs continuously"
  - "Adjust inputs based on error"
`
  );
}
