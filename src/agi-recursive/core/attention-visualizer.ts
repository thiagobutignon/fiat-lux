/**
 * Attention Visualizer - Tools for visualizing and exporting attention data
 *
 * PURPOSE:
 * - Generate human-readable visualizations of attention patterns
 * - Export data in multiple formats (JSON, CSV, HTML)
 * - Create regulatory compliance reports
 * - Debug reasoning chains visually
 */

import { QueryAttention, AttentionTrace, AttentionStats } from './attention-tracker';
import fs from 'fs';
import path from 'path';

/**
 * ASCII bar chart for attention weights
 */
export function renderAttentionBar(weight: number, maxWidth: number = 50): string {
  const filled = Math.round(weight * maxWidth);
  const empty = maxWidth - filled;
  return '█'.repeat(filled) + '░'.repeat(empty);
}

/**
 * Generate ASCII visualization of attention traces
 */
export function visualizeAttention(attention: QueryAttention): string {
  const lines: string[] = [];

  lines.push('═'.repeat(80));
  lines.push('ATTENTION VISUALIZATION');
  lines.push('═'.repeat(80));
  lines.push('');
  lines.push(`Query: "${attention.query}"`);
  lines.push(`Query ID: ${attention.query_id}`);
  lines.push(`Timestamp: ${new Date(attention.timestamp).toISOString()}`);
  lines.push(`Total Concepts: ${attention.total_concepts}`);
  lines.push('');

  // Decision path
  if (attention.decision_path.length > 0) {
    lines.push('─'.repeat(80));
    lines.push('DECISION PATH');
    lines.push('─'.repeat(80));
    attention.decision_path.forEach((decision, i) => {
      lines.push(`${i + 1}. ${decision}`);
    });
    lines.push('');
  }

  // Top influencers
  lines.push('─'.repeat(80));
  lines.push('TOP INFLUENCERS');
  lines.push('─'.repeat(80));
  attention.top_influencers.forEach((trace, i) => {
    const percentage = (trace.weight * 100).toFixed(1);
    const bar = renderAttentionBar(trace.weight, 30);
    lines.push(`${i + 1}. [${percentage}%] ${bar}`);
    lines.push(`   Concept: ${trace.concept}`);
    lines.push(`   Source: ${trace.slice}`);
    lines.push(`   Reasoning: ${trace.reasoning}`);
    lines.push('');
  });

  // All traces grouped by slice
  lines.push('─'.repeat(80));
  lines.push('ALL TRACES (Grouped by Source)');
  lines.push('─'.repeat(80));

  const tracesBySlice = groupTracesBySlice(attention.traces);

  for (const [slice, traces] of Object.entries(tracesBySlice)) {
    const totalWeight = traces.reduce((sum, t) => sum + t.weight, 0);
    const avgWeight = totalWeight / traces.length;
    const percentage = (avgWeight * 100).toFixed(1);

    lines.push(`📁 ${slice} (${traces.length} concepts, avg: ${percentage}%)`);

    traces.forEach((trace) => {
      const tracePercentage = (trace.weight * 100).toFixed(1);
      const bar = renderAttentionBar(trace.weight, 20);
      lines.push(`   • [${tracePercentage}%] ${bar} ${trace.concept}`);
    });
    lines.push('');
  }

  lines.push('═'.repeat(80));

  return lines.join('\n');
}

/**
 * Group traces by their source slice
 */
function groupTracesBySlice(traces: AttentionTrace[]): Record<string, AttentionTrace[]> {
  const grouped: Record<string, AttentionTrace[]> = {};

  for (const trace of traces) {
    if (!grouped[trace.slice]) {
      grouped[trace.slice] = [];
    }
    grouped[trace.slice].push(trace);
  }

  // Sort each group by weight
  for (const slice in grouped) {
    grouped[slice].sort((a, b) => b.weight - a.weight);
  }

  return grouped;
}

/**
 * Generate statistics visualization
 */
export function visualizeStats(stats: AttentionStats): string {
  const lines: string[] = [];

  lines.push('═'.repeat(80));
  lines.push('ATTENTION STATISTICS');
  lines.push('═'.repeat(80));
  lines.push('');
  lines.push(`Total Queries: ${stats.total_queries}`);
  lines.push(`Total Traces: ${stats.total_traces}`);
  lines.push(`Average Traces per Query: ${stats.average_traces_per_query.toFixed(2)}`);
  lines.push('');

  // Most influential concepts
  lines.push('─'.repeat(80));
  lines.push('MOST INFLUENTIAL CONCEPTS (Top 10)');
  lines.push('─'.repeat(80));
  stats.most_influential_concepts.forEach((item, i) => {
    const percentage = (item.average_weight * 100).toFixed(1);
    const bar = renderAttentionBar(item.average_weight, 30);
    lines.push(`${i + 1}. [${percentage}%] ${bar}`);
    lines.push(`   ${item.concept} (used ${item.count} times)`);
  });
  lines.push('');

  // Most used slices
  lines.push('─'.repeat(80));
  lines.push('MOST USED SLICES (Top 10)');
  lines.push('─'.repeat(80));
  stats.most_used_slices.forEach((item, i) => {
    const percentage = (item.average_weight * 100).toFixed(1);
    lines.push(`${i + 1}. ${item.slice}`);
    lines.push(`   Used: ${item.count} times, Avg Weight: ${percentage}%`);
  });
  lines.push('');

  // High confidence patterns
  if (stats.high_confidence_patterns.length > 0) {
    lines.push('─'.repeat(80));
    lines.push('COMMON CONCEPT PATTERNS');
    lines.push('─'.repeat(80));
    stats.high_confidence_patterns.forEach((pattern, i) => {
      lines.push(`${i + 1}. [${pattern.frequency} times] ${pattern.concepts.join(' + ')}`);
    });
    lines.push('');
  }

  lines.push('═'.repeat(80));

  return lines.join('\n');
}

/**
 * Export attention data to CSV format
 */
export function exportToCSV(attentions: QueryAttention[]): string {
  const lines: string[] = [];

  // Header
  lines.push('query_id,query,concept,slice,weight,reasoning,timestamp');

  // Data rows
  for (const attention of attentions) {
    for (const trace of attention.traces) {
      const row = [
        attention.query_id,
        `"${attention.query.replace(/"/g, '""')}"`, // Escape quotes
        trace.concept,
        trace.slice,
        trace.weight.toFixed(4),
        `"${trace.reasoning.replace(/"/g, '""')}"`,
        new Date(trace.timestamp).toISOString(),
      ];
      lines.push(row.join(','));
    }
  }

  return lines.join('\n');
}

/**
 * Generate HTML report with interactive visualization
 */
export function generateHTMLReport(
  attentions: QueryAttention[],
  stats: AttentionStats
): string {
  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AGI Attention Report</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      line-height: 1.6;
      color: #333;
      background: #f5f5f5;
      padding: 20px;
    }
    .container {
      max-width: 1200px;
      margin: 0 auto;
      background: white;
      padding: 40px;
      border-radius: 8px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    h1 {
      color: #2c3e50;
      margin-bottom: 10px;
      font-size: 2.5em;
    }
    h2 {
      color: #34495e;
      margin-top: 30px;
      margin-bottom: 15px;
      padding-bottom: 10px;
      border-bottom: 2px solid #3498db;
    }
    h3 {
      color: #7f8c8d;
      margin-top: 20px;
      margin-bottom: 10px;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 20px;
      margin: 20px 0;
    }
    .stat-card {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 20px;
      border-radius: 8px;
      text-align: center;
    }
    .stat-value {
      font-size: 2.5em;
      font-weight: bold;
      margin-bottom: 5px;
    }
    .stat-label {
      font-size: 0.9em;
      opacity: 0.9;
    }
    .query-card {
      background: #f8f9fa;
      border-left: 4px solid #3498db;
      padding: 20px;
      margin: 20px 0;
      border-radius: 4px;
    }
    .query-text {
      font-size: 1.1em;
      font-weight: 500;
      color: #2c3e50;
      margin-bottom: 10px;
    }
    .trace-item {
      background: white;
      padding: 15px;
      margin: 10px 0;
      border-radius: 4px;
      border-left: 3px solid #95a5a6;
    }
    .weight-bar {
      height: 8px;
      background: linear-gradient(90deg, #3498db 0%, #2ecc71 100%);
      border-radius: 4px;
      margin: 10px 0;
    }
    .weight-value {
      font-weight: bold;
      color: #3498db;
    }
    .concept {
      font-weight: 600;
      color: #2c3e50;
    }
    .slice {
      color: #7f8c8d;
      font-size: 0.9em;
    }
    .reasoning {
      color: #555;
      margin-top: 8px;
      font-size: 0.95em;
    }
    .decision-path {
      background: #fff3cd;
      border-left: 4px solid #ffc107;
      padding: 15px;
      margin: 15px 0;
      border-radius: 4px;
    }
    .decision-step {
      padding: 5px 0;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin: 20px 0;
    }
    th {
      background: #34495e;
      color: white;
      padding: 12px;
      text-align: left;
    }
    td {
      padding: 12px;
      border-bottom: 1px solid #ddd;
    }
    tr:hover {
      background: #f5f5f5;
    }
    .timestamp {
      color: #95a5a6;
      font-size: 0.85em;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>🧠 AGI Attention Analysis Report</h1>
    <p class="timestamp">Generated: ${new Date().toISOString()}</p>

    <h2>📊 Summary Statistics</h2>
    <div class="stats">
      <div class="stat-card">
        <div class="stat-value">${stats.total_queries}</div>
        <div class="stat-label">Total Queries</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">${stats.total_traces}</div>
        <div class="stat-label">Total Traces</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">${stats.average_traces_per_query.toFixed(1)}</div>
        <div class="stat-label">Avg Traces/Query</div>
      </div>
    </div>

    <h2>🎯 Most Influential Concepts</h2>
    <table>
      <thead>
        <tr>
          <th>Rank</th>
          <th>Concept</th>
          <th>Count</th>
          <th>Avg Weight</th>
          <th>Visualization</th>
        </tr>
      </thead>
      <tbody>
        ${stats.most_influential_concepts
          .map(
            (item, i) => `
        <tr>
          <td>${i + 1}</td>
          <td class="concept">${item.concept}</td>
          <td>${item.count}</td>
          <td class="weight-value">${(item.average_weight * 100).toFixed(1)}%</td>
          <td>
            <div class="weight-bar" style="width: ${item.average_weight * 100}%"></div>
          </td>
        </tr>
        `
          )
          .join('')}
      </tbody>
    </table>

    <h2>📁 Most Used Slices</h2>
    <table>
      <thead>
        <tr>
          <th>Rank</th>
          <th>Slice</th>
          <th>Usage Count</th>
          <th>Avg Weight</th>
        </tr>
      </thead>
      <tbody>
        ${stats.most_used_slices
          .map(
            (item, i) => `
        <tr>
          <td>${i + 1}</td>
          <td class="slice">${item.slice}</td>
          <td>${item.count}</td>
          <td class="weight-value">${(item.average_weight * 100).toFixed(1)}%</td>
        </tr>
        `
          )
          .join('')}
      </tbody>
    </table>

    <h2>🔍 Query Details</h2>
    ${attentions
      .map(
        (attention) => `
    <div class="query-card">
      <div class="query-text">"${attention.query}"</div>
      <p class="timestamp">Query ID: ${attention.query_id} | ${new Date(attention.timestamp).toLocaleString()}</p>

      ${
        attention.decision_path.length > 0
          ? `
      <div class="decision-path">
        <h3>Decision Path:</h3>
        ${attention.decision_path.map((d, i) => `<div class="decision-step">${i + 1}. ${d}</div>`).join('')}
      </div>
      `
          : ''
      }

      <h3>Top Influences:</h3>
      ${attention.top_influencers
        .map(
          (trace) => `
      <div class="trace-item">
        <div class="concept">${trace.concept}</div>
        <div class="slice">📍 ${trace.slice}</div>
        <div class="weight-bar" style="width: ${trace.weight * 100}%"></div>
        <div class="weight-value">${(trace.weight * 100).toFixed(1)}%</div>
        <div class="reasoning">${trace.reasoning}</div>
      </div>
      `
        )
        .join('')}
    </div>
    `
      )
      .join('')}
  </div>
</body>
</html>`;

  return html;
}

/**
 * Save attention data to file
 */
export async function saveAttentionReport(
  attentions: QueryAttention[],
  stats: AttentionStats,
  outputDir: string,
  format: 'json' | 'csv' | 'html' | 'all' = 'all'
): Promise<string[]> {
  const savedFiles: string[] = [];

  // Ensure output directory exists
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  const timestamp = new Date().toISOString().replace(/:/g, '-').split('.')[0];

  // JSON export
  if (format === 'json' || format === 'all') {
    const jsonPath = path.join(outputDir, `attention-report-${timestamp}.json`);
    const jsonData = {
      generated: new Date().toISOString(),
      statistics: stats,
      queries: attentions,
    };
    fs.writeFileSync(jsonPath, JSON.stringify(jsonData, null, 2));
    savedFiles.push(jsonPath);
  }

  // CSV export
  if (format === 'csv' || format === 'all') {
    const csvPath = path.join(outputDir, `attention-traces-${timestamp}.csv`);
    const csvData = exportToCSV(attentions);
    fs.writeFileSync(csvPath, csvData);
    savedFiles.push(csvPath);
  }

  // HTML export
  if (format === 'html' || format === 'all') {
    const htmlPath = path.join(outputDir, `attention-report-${timestamp}.html`);
    const htmlData = generateHTMLReport(attentions, stats);
    fs.writeFileSync(htmlPath, htmlData);
    savedFiles.push(htmlPath);
  }

  return savedFiles;
}

/**
 * Generate comparison between two query attentions
 */
export function compareAttentions(
  attention1: QueryAttention,
  attention2: QueryAttention
): string {
  const lines: string[] = [];

  lines.push('═'.repeat(80));
  lines.push('ATTENTION COMPARISON');
  lines.push('═'.repeat(80));
  lines.push('');

  lines.push(`Query 1: "${attention1.query}"`);
  lines.push(`Query 2: "${attention2.query}"`);
  lines.push('');

  // Find common concepts
  const concepts1 = new Set(attention1.traces.map((t) => t.concept));
  const concepts2 = new Set(attention2.traces.map((t) => t.concept));
  const common = [...concepts1].filter((c) => concepts2.has(c));
  const unique1 = [...concepts1].filter((c) => !concepts2.has(c));
  const unique2 = [...concepts2].filter((c) => !concepts1.has(c));

  lines.push(`Common concepts: ${common.length}`);
  lines.push(`Unique to Query 1: ${unique1.length}`);
  lines.push(`Unique to Query 2: ${unique2.length}`);
  lines.push('');

  if (common.length > 0) {
    lines.push('─'.repeat(80));
    lines.push('COMMON CONCEPTS');
    lines.push('─'.repeat(80));
    common.forEach((concept) => {
      const trace1 = attention1.traces.find((t) => t.concept === concept);
      const trace2 = attention2.traces.find((t) => t.concept === concept);
      if (trace1 && trace2) {
        const diff = trace2.weight - trace1.weight;
        const diffStr = diff > 0 ? `+${(diff * 100).toFixed(1)}%` : `${(diff * 100).toFixed(1)}%`;
        lines.push(
          `• ${concept}: ${(trace1.weight * 100).toFixed(1)}% → ${(trace2.weight * 100).toFixed(1)}% (${diffStr})`
        );
      }
    });
    lines.push('');
  }

  lines.push('═'.repeat(80));

  return lines.join('\n');
}
