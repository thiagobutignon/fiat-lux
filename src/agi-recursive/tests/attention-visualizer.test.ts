/**
 * @file attention-visualizer.test.ts
 * Tests for Attention Visualizer
 *
 * Key capabilities tested:
 * - ASCII bar chart rendering
 * - Attention visualization
 * - Statistics visualization
 * - CSV export
 * - HTML report generation
 * - File saving (JSON, CSV, HTML)
 * - Attention comparison
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import {
  renderAttentionBar,
  visualizeAttention,
  visualizeStats,
  exportToCSV,
  generateHTMLReport,
  saveAttentionReport,
  compareAttentions,
} from '../core/attention-visualizer';
import { QueryAttention, AttentionTrace, AttentionStats } from '../core/attention-tracker';
import fs from 'fs';
import path from 'path';
import os from 'os';

describe('AttentionVisualizer', () => {
  let testAttention: QueryAttention;
  let testStats: AttentionStats;
  let testDir: string;

  beforeEach(() => {
    // Create test attention data
    const traces: AttentionTrace[] = [
      {
        concept: 'concept1',
        slice: 'slice1',
        weight: 0.9,
        reasoning: 'High importance concept',
        timestamp: Date.now(),
      },
      {
        concept: 'concept2',
        slice: 'slice2',
        weight: 0.7,
        reasoning: 'Medium importance',
        timestamp: Date.now(),
      },
      {
        concept: 'concept3',
        slice: 'slice1',
        weight: 0.5,
        reasoning: 'Lower importance',
        timestamp: Date.now(),
      },
    ];

    testAttention = {
      query_id: 'test-query-1',
      query: 'Test query about concepts',
      timestamp: Date.now(),
      traces,
      total_concepts: 3,
      top_influencers: traces.slice(0, 2),
      decision_path: ['Decision 1', 'Decision 2', 'Decision 3'],
    };

    testStats = {
      total_queries: 10,
      total_traces: 50,
      average_traces_per_query: 5,
      most_influential_concepts: [
        { concept: 'concept1', count: 8, average_weight: 0.85 },
        { concept: 'concept2', count: 6, average_weight: 0.72 },
      ],
      most_used_slices: [
        { slice: 'slice1', count: 12, average_weight: 0.78 },
        { slice: 'slice2', count: 8, average_weight: 0.65 },
      ],
      high_confidence_patterns: [
        { concepts: ['concept1', 'concept2'], frequency: 5 },
      ],
    };

    // Create temp directory for file tests
    testDir = path.join(os.tmpdir(), `attention-viz-test-${Date.now()}`);
  });

  afterEach(() => {
    // Cleanup test directory
    if (fs.existsSync(testDir)) {
      fs.rmSync(testDir, { recursive: true, force: true });
    }
  });

  describe('renderAttentionBar', () => {
    it('should render full bar for weight 1.0', () => {
      const bar = renderAttentionBar(1.0, 10);

      expect(bar).toBe('██████████');
      expect(bar.length).toBe(10);
    });

    it('should render empty bar for weight 0.0', () => {
      const bar = renderAttentionBar(0.0, 10);

      expect(bar).toBe('░░░░░░░░░░');
      expect(bar.length).toBe(10);
    });

    it('should render half bar for weight 0.5', () => {
      const bar = renderAttentionBar(0.5, 10);

      expect(bar).toContain('█');
      expect(bar).toContain('░');
      expect(bar.length).toBe(10);
    });

    it('should respect custom max width', () => {
      const bar = renderAttentionBar(0.5, 20);

      expect(bar.length).toBe(20);
    });

    it('should handle edge cases', () => {
      expect(renderAttentionBar(0.01, 100).length).toBe(100);
      expect(renderAttentionBar(0.99, 100).length).toBe(100);
    });

    it('should use filled character for filled portion', () => {
      const bar = renderAttentionBar(0.3, 10);

      expect(bar).toMatch(/^█+░+$/);
    });
  });

  describe('visualizeAttention', () => {
    it('should include query information', () => {
      const viz = visualizeAttention(testAttention);

      expect(viz).toContain(testAttention.query);
      expect(viz).toContain(testAttention.query_id);
    });

    it('should include decision path', () => {
      const viz = visualizeAttention(testAttention);

      expect(viz).toContain('DECISION PATH');
      expect(viz).toContain('Decision 1');
      expect(viz).toContain('Decision 2');
      expect(viz).toContain('Decision 3');
    });

    it('should include top influencers', () => {
      const viz = visualizeAttention(testAttention);

      expect(viz).toContain('TOP INFLUENCERS');
      expect(viz).toContain('concept1');
      expect(viz).toContain('concept2');
      expect(viz).toContain('90.0%'); // 0.9 * 100
    });

    it('should group traces by slice', () => {
      const viz = visualizeAttention(testAttention);

      expect(viz).toContain('ALL TRACES (Grouped by Source)');
      expect(viz).toContain('slice1');
      expect(viz).toContain('slice2');
    });

    it('should include visual separators', () => {
      const viz = visualizeAttention(testAttention);

      expect(viz).toContain('═'.repeat(80));
      expect(viz).toContain('─'.repeat(80));
    });

    it('should include timestamp', () => {
      const viz = visualizeAttention(testAttention);

      expect(viz).toContain('Timestamp:');
    });

    it('should handle empty decision path', () => {
      const attention = { ...testAttention, decision_path: [] };
      const viz = visualizeAttention(attention);

      expect(viz).toBeDefined();
      expect(viz).not.toContain('DECISION PATH');
    });

    it('should show percentage for weights', () => {
      const viz = visualizeAttention(testAttention);

      expect(viz).toContain('%');
    });
  });

  describe('visualizeStats', () => {
    it('should include summary statistics', () => {
      const viz = visualizeStats(testStats);

      expect(viz).toContain('Total Queries: 10');
      expect(viz).toContain('Total Traces: 50');
      expect(viz).toContain('Average Traces per Query: 5.00');
    });

    it('should include most influential concepts', () => {
      const viz = visualizeStats(testStats);

      expect(viz).toContain('MOST INFLUENTIAL CONCEPTS');
      expect(viz).toContain('concept1');
      expect(viz).toContain('85.0%'); // 0.85 * 100
      expect(viz).toContain('used 8 times');
    });

    it('should include most used slices', () => {
      const viz = visualizeStats(testStats);

      expect(viz).toContain('MOST USED SLICES');
      expect(viz).toContain('slice1');
      expect(viz).toContain('Used: 12 times');
    });

    it('should include concept patterns', () => {
      const viz = visualizeStats(testStats);

      expect(viz).toContain('COMMON CONCEPT PATTERNS');
      expect(viz).toContain('concept1 + concept2');
      expect(viz).toContain('[5 times]');
    });

    it('should handle empty patterns', () => {
      const stats = { ...testStats, high_confidence_patterns: [] };
      const viz = visualizeStats(stats);

      expect(viz).toBeDefined();
      expect(viz).not.toContain('COMMON CONCEPT PATTERNS');
    });

    it('should include visual bars', () => {
      const viz = visualizeStats(testStats);

      expect(viz).toContain('█');
    });
  });

  describe('exportToCSV', () => {
    it('should include CSV header', () => {
      const csv = exportToCSV([testAttention]);

      expect(csv).toContain('query_id,query,concept,slice,weight,reasoning,timestamp');
    });

    it('should export all traces', () => {
      const csv = exportToCSV([testAttention]);
      const lines = csv.split('\n');

      // Header + 3 traces
      expect(lines.length).toBe(4);
    });

    it('should escape quotes in query', () => {
      const attention = { ...testAttention, query: 'Test "quoted" query' };
      const csv = exportToCSV([attention]);

      expect(csv).toContain('""quoted""');
    });

    it('should escape quotes in reasoning', () => {
      const traces = [
        {
          concept: 'test',
          slice: 'slice',
          weight: 0.5,
          reasoning: 'Reasoning with "quotes"',
          timestamp: Date.now(),
        },
      ];
      const attention = { ...testAttention, traces };
      const csv = exportToCSV([attention]);

      expect(csv).toContain('""quotes""');
    });

    it('should format weights with 4 decimals', () => {
      const csv = exportToCSV([testAttention]);

      expect(csv).toContain('0.9000');
      expect(csv).toContain('0.7000');
    });

    it('should include ISO timestamp', () => {
      const csv = exportToCSV([testAttention]);

      expect(csv).toMatch(/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/);
    });

    it('should handle multiple queries', () => {
      const attention2 = { ...testAttention, query_id: 'query2' };
      const csv = exportToCSV([testAttention, attention2]);
      const lines = csv.split('\n');

      // Header + (3 traces × 2 queries)
      expect(lines.length).toBe(7);
    });
  });

  describe('generateHTMLReport', () => {
    it('should generate valid HTML', () => {
      const html = generateHTMLReport([testAttention], testStats);

      expect(html).toContain('<!DOCTYPE html>');
      expect(html).toContain('<html');
      expect(html).toContain('</html>');
    });

    it('should include title', () => {
      const html = generateHTMLReport([testAttention], testStats);

      expect(html).toContain('<title>AGI Attention Report</title>');
      expect(html).toContain('AGI Attention Analysis Report');
    });

    it('should include CSS styles', () => {
      const html = generateHTMLReport([testAttention], testStats);

      expect(html).toContain('<style>');
      expect(html).toContain('</style>');
    });

    it('should include summary statistics', () => {
      const html = generateHTMLReport([testAttention], testStats);

      expect(html).toContain('Total Queries');
      expect(html).toContain('10');
      expect(html).toContain('Total Traces');
      expect(html).toContain('50');
    });

    it('should include tables for concepts and slices', () => {
      const html = generateHTMLReport([testAttention], testStats);

      expect(html).toContain('<table');
      expect(html).toContain('Most Influential Concepts');
      expect(html).toContain('Most Used Slices');
    });

    it('should include query details', () => {
      const html = generateHTMLReport([testAttention], testStats);

      expect(html).toContain(testAttention.query);
      expect(html).toContain(testAttention.query_id);
    });

    it('should include decision path', () => {
      const html = generateHTMLReport([testAttention], testStats);

      expect(html).toContain('Decision Path');
      expect(html).toContain('Decision 1');
    });

    it('should include top influences', () => {
      const html = generateHTMLReport([testAttention], testStats);

      expect(html).toContain('Top Influences');
      expect(html).toContain('concept1');
      expect(html).toContain('90.0%');
    });

    it('should handle empty decision path', () => {
      const attention = { ...testAttention, decision_path: [] };
      const html = generateHTMLReport([attention], testStats);

      expect(html).toBeDefined();
      expect(html).not.toContain('Decision Path:');
    });

    it('should format timestamps', () => {
      const html = generateHTMLReport([testAttention], testStats);

      expect(html).toMatch(/Generated:.*\d{4}-\d{2}-\d{2}/);
    });
  });

  describe('saveAttentionReport', () => {
    it('should save JSON format', async () => {
      const files = await saveAttentionReport([testAttention], testStats, testDir, 'json');

      expect(files.length).toBe(1);
      expect(files[0]).toContain('.json');
      expect(fs.existsSync(files[0])).toBe(true);
    });

    it('should save CSV format', async () => {
      const files = await saveAttentionReport([testAttention], testStats, testDir, 'csv');

      expect(files.length).toBe(1);
      expect(files[0]).toContain('.csv');
      expect(fs.existsSync(files[0])).toBe(true);
    });

    it('should save HTML format', async () => {
      const files = await saveAttentionReport([testAttention], testStats, testDir, 'html');

      expect(files.length).toBe(1);
      expect(files[0]).toContain('.html');
      expect(fs.existsSync(files[0])).toBe(true);
    });

    it('should save all formats', async () => {
      const files = await saveAttentionReport([testAttention], testStats, testDir, 'all');

      expect(files.length).toBe(3);
      expect(files.some((f) => f.endsWith('.json'))).toBe(true);
      expect(files.some((f) => f.endsWith('.csv'))).toBe(true);
      expect(files.some((f) => f.endsWith('.html'))).toBe(true);
    });

    it('should create output directory if not exists', async () => {
      const nonExistentDir = path.join(testDir, 'new-dir');
      await saveAttentionReport([testAttention], testStats, nonExistentDir, 'json');

      expect(fs.existsSync(nonExistentDir)).toBe(true);
    });

    it('should include timestamp in filename', async () => {
      const files = await saveAttentionReport([testAttention], testStats, testDir, 'json');

      expect(files[0]).toMatch(/attention-report-\d{4}-\d{2}-\d{2}/);
    });

    it('should save valid JSON data', async () => {
      const files = await saveAttentionReport([testAttention], testStats, testDir, 'json');
      const data = JSON.parse(fs.readFileSync(files[0], 'utf-8'));

      expect(data.statistics).toBeDefined();
      expect(data.queries).toBeDefined();
      expect(data.generated).toBeDefined();
    });

    it('should save valid CSV data', async () => {
      const files = await saveAttentionReport([testAttention], testStats, testDir, 'csv');
      const content = fs.readFileSync(files[0], 'utf-8');

      expect(content).toContain('query_id,query,concept');
      expect(content.split('\n').length).toBeGreaterThan(1);
    });

    it('should save valid HTML data', async () => {
      const files = await saveAttentionReport([testAttention], testStats, testDir, 'html');
      const content = fs.readFileSync(files[0], 'utf-8');

      expect(content).toContain('<!DOCTYPE html>');
      expect(content).toContain('</html>');
    });
  });

  describe('compareAttentions', () => {
    let attention2: QueryAttention;

    beforeEach(() => {
      const traces2: AttentionTrace[] = [
        {
          concept: 'concept1',
          slice: 'slice1',
          weight: 0.8,
          reasoning: 'Different weight',
          timestamp: Date.now(),
        },
        {
          concept: 'concept3',
          slice: 'slice3',
          weight: 0.6,
          reasoning: 'Common concept',
          timestamp: Date.now(),
        },
        {
          concept: 'concept4',
          slice: 'slice4',
          weight: 0.4,
          reasoning: 'Unique to query 2',
          timestamp: Date.now(),
        },
      ];

      attention2 = {
        query_id: 'test-query-2',
        query: 'Second test query',
        timestamp: Date.now(),
        traces: traces2,
        total_concepts: 3,
        top_influencers: traces2.slice(0, 2),
        decision_path: [],
      };
    });

    it('should show both queries', () => {
      const comparison = compareAttentions(testAttention, attention2);

      expect(comparison).toContain('Query 1:');
      expect(comparison).toContain('Query 2:');
      expect(comparison).toContain(testAttention.query);
      expect(comparison).toContain(attention2.query);
    });

    it('should identify common concepts', () => {
      const comparison = compareAttentions(testAttention, attention2);

      expect(comparison).toContain('Common concepts: 2'); // concept1 and concept3
      expect(comparison).toContain('COMMON CONCEPTS');
      expect(comparison).toContain('concept1');
      expect(comparison).toContain('concept3');
    });

    it('should identify unique concepts', () => {
      const comparison = compareAttentions(testAttention, attention2);

      expect(comparison).toContain('Unique to Query 1: 1'); // concept2
      expect(comparison).toContain('Unique to Query 2: 1'); // concept4
    });

    it('should show weight changes for common concepts', () => {
      const comparison = compareAttentions(testAttention, attention2);

      // concept1: 0.9 → 0.8 = -0.1 = -10.0%
      expect(comparison).toContain('concept1:');
      expect(comparison).toContain('90.0%');
      expect(comparison).toContain('80.0%');
      expect(comparison).toContain('-10.0%');
    });

    it('should show positive weight changes', () => {
      // Modify to have positive change
      const traces2: AttentionTrace[] = [
        {
          concept: 'concept1',
          slice: 'slice1',
          weight: 0.95, // Higher than 0.9
          reasoning: 'Increased weight',
          timestamp: Date.now(),
        },
      ];

      const attention2Modified = {
        ...attention2,
        traces: traces2,
      };

      const comparison = compareAttentions(testAttention, attention2Modified);

      expect(comparison).toContain('+5.0%');
    });

    it('should handle no common concepts', () => {
      const traces2: AttentionTrace[] = [
        {
          concept: 'different_concept',
          slice: 'slice',
          weight: 0.5,
          reasoning: 'Different',
          timestamp: Date.now(),
        },
      ];

      const attention2Different = {
        ...attention2,
        traces: traces2,
      };

      const comparison = compareAttentions(testAttention, attention2Different);

      expect(comparison).toContain('Common concepts: 0');
    });

    it('should include visual separators', () => {
      const comparison = compareAttentions(testAttention, attention2);

      expect(comparison).toContain('═'.repeat(80));
      expect(comparison).toContain('─'.repeat(80));
    });
  });
});
