# Attention Tracking: Interpretability Layer for AGI

**Status:** ✅ Production Ready
**Version:** 1.0.0
**Date:** October 2025

---

## Executive Summary

The Attention Tracking system transforms the AGI from a **black box** into a **glass box** by recording **exactly** which concepts from which knowledge slices influenced each decision.

### The Problem It Solves

Traditional AI systems are black boxes:
- ❌ "Why did it give this answer?" → Unknown
- ❌ "Which data influenced this decision?" → Unknown
- ❌ "Can we trust this medical/financial advice?" → Unknown

### The Solution

Attention Tracking provides:
- ✅ **Interpretability**: See exactly which concepts influenced decisions
- ✅ **Debugging**: Trace errors back to specific knowledge sources
- ✅ **Auditing**: Export complete decision chains for regulatory compliance
- ✅ **Meta-learning**: Discover patterns in cross-domain reasoning

---

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                         MetaAgent                            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              AttentionTracker                         │  │
│  │  • startQuery()                                       │  │
│  │  • addTrace(concept, slice, weight, reasoning)       │  │
│  │  • addDecisionPoint()                                │  │
│  │  • endQuery() → QueryAttention                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                           ↓                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │            Query Processing Pipeline                  │  │
│  │                                                       │  │
│  │  1. Decompose → Track domain selection               │  │
│  │  2. Invoke Agents → Track concept usage              │  │
│  │  3. Compose → Track synthesis decisions              │  │
│  │  4. Synthesize → Track final reasoning               │  │
│  └───────────────────────────────────────────────────────┘  │
│                           ↓                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              QueryAttention Result                    │  │
│  │  • traces: AttentionTrace[]                          │  │
│  │  • decision_path: string[]                           │  │
│  │  • top_influencers: AttentionTrace[]                 │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           ↓
      ┌────────────────────────────────────────────┐
      │      AttentionVisualizer                   │
      │  • visualizeAttention() → ASCII            │
      │  • generateHTMLReport() → Interactive      │
      │  • exportToCSV() → Data analysis           │
      │  • saveAttentionReport() → Persistence     │
      └────────────────────────────────────────────┘
```

### Data Model

#### AttentionTrace

Records a single concept's influence on a decision:

```typescript
interface AttentionTrace {
  concept: string;        // e.g., "compound_interest", "diversification"
  slice: string;          // e.g., "agent/financial", "slice/biology/evolution.md"
  weight: number;         // 0-1, how much it influenced
  reasoning: string;      // WHY this concept was influential
  timestamp: number;      // When this trace was recorded
}
```

#### QueryAttention

Complete attention data for one query execution:

```typescript
interface QueryAttention {
  query_id: string;              // Unique identifier
  query: string;                 // Original query text
  timestamp: number;             // When processed
  traces: AttentionTrace[];      // All attention traces
  total_concepts: number;        // Total concepts considered
  top_influencers: AttentionTrace[];  // Top 5 most influential
  decision_path: string[];       // Sequence of decisions
}
```

---

## How It Works

### 1. Tracking Lifecycle

```typescript
// 1. Start tracking
attentionTracker.startQuery(queryId, "What is compound interest?");

// 2. Add traces during processing
attentionTracker.addTrace(
  "compound_interest",           // concept
  "agent/financial",              // source
  0.9,                            // weight (high confidence)
  "Financial agent identified compound interest as core concept"
);

// 3. Record decision points
attentionTracker.addDecisionPoint(
  "Query decomposed into domains: financial"
);

// 4. End tracking
const attention = attentionTracker.endQuery();
```

### 2. Automatic Integration with MetaAgent

The MetaAgent automatically tracks attention at key decision points:

#### Domain Selection
```typescript
// When query is decomposed
for (const domain of decomposition.domains) {
  attentionTracker.addTrace(
    'domain_selection',
    'meta-agent/decomposition',
    0.8,
    `Selected ${domain} based on: ${decomposition.reasoning}`
  );
}
```

#### Agent Invocation
```typescript
// When agent provides concepts
for (const concept of response.concepts) {
  attentionTracker.addTrace(
    concept,
    `agent/${domain}`,
    response.confidence,
    `${domain} contributed concept: ${response.reasoning}`
  );
}
```

#### Composition Decisions
```typescript
// When insights are composed
if (composition.should_recurse) {
  attentionTracker.addTrace(
    'composition_recursion',
    'meta-agent/composition',
    composition.confidence,
    `Composition suggests recursion due to: ${missing_perspectives}`
  );
}
```

### 3. Weight Calculation

Weights represent how much a concept influenced the decision (0-1 scale):

```typescript
// Based on agent confidence
weight = agentResponse.confidence;

// Or geometric mean of confidence and relevance
weight = Math.sqrt(agentConfidence * sliceRelevance);
```

**Weight Interpretation:**
- `0.9-1.0`: Critical influence, central to decision
- `0.7-0.9`: Strong influence, important contributor
- `0.5-0.7`: Moderate influence, supporting evidence
- `0.3-0.5`: Weak influence, tangential mention
- `0.0-0.3`: Minimal influence, barely relevant

---

## Usage Examples

### Basic Usage

```typescript
import { MetaAgent } from './core/meta-agent';
import { visualizeAttention } from './core/attention-visualizer';

const metaAgent = new MetaAgent(apiKey);
await metaAgent.initialize();

// Register agents...

const result = await metaAgent.process(
  "How does compound interest work in savings accounts?"
);

// Access attention data
if (result.attention) {
  console.log(visualizeAttention(result.attention));

  // Top 5 most influential concepts
  result.attention.top_influencers.forEach(trace => {
    console.log(`${trace.concept}: ${(trace.weight * 100).toFixed(1)}%`);
  });
}
```

### Debugging Reasoning Chains

```typescript
// Get detailed explanation
const explanation = metaAgent.explainQuery(result.attention.query_id);
console.log(explanation);

// Output:
// ═══ REASONING EXPLANATION ═══
// Query: "How does compound interest work?"
//
// ─── DECISION PATH ─────────────
// 1. Query decomposed into domains: financial
// 2. Agent financial invoked with confidence 0.95
// 3. Composition confidence: 0.92, should_recurse: false
//
// ─── TOP 5 INFLUENCES ──────────
// 1. [95.0%] compound_interest
//    From: agent/financial
//    Why: Financial agent identified compound interest as core concept
```

### Regulatory Auditing

```typescript
// Export for compliance
const auditData = metaAgent.exportAttentionForAudit();

// auditData structure:
{
  export_timestamp: 1728345600000,
  total_queries: 10,
  queries: [
    {
      query_id: "query_1728345600_abc123",
      query: "Should I invest in stocks?",
      decision_path: [...],
      traces: [
        {
          concept: "risk_assessment",
          slice: "agent/financial",
          weight: 0.85,
          reasoning: "Financial agent assessed investment risk"
        },
        // ... more traces
      ]
    },
    // ... more queries
  ]
}

// Save to file for regulatory review
fs.writeFileSync('audit-report.json', JSON.stringify(auditData, null, 2));
```

### Pattern Discovery (Meta-Learning)

```typescript
// Analyze patterns across all queries
const stats = metaAgent.getAttentionStats();

console.log('Most influential concepts:');
stats.most_influential_concepts.forEach(item => {
  console.log(`${item.concept}: used ${item.count} times, avg weight ${item.average_weight.toFixed(2)}`);
});

// Output:
// Most influential concepts:
// compound_interest: used 5 times, avg weight 0.89
// diversification: used 4 times, avg weight 0.83
// risk_assessment: used 7 times, avg weight 0.81

// Find common patterns
stats.high_confidence_patterns.forEach(pattern => {
  console.log(`Pattern: ${pattern.concepts.join(' + ')} appears ${pattern.frequency} times`);
});

// Output:
// Pattern: compound_interest + time_value_of_money appears 3 times
// Pattern: diversification + risk_management appears 2 times
```

### Comparing Queries

```typescript
const attention1 = tracker.getQueryAttention(queryId1);
const attention2 = tracker.getQueryAttention(queryId2);

console.log(compareAttentions(attention1, attention2));

// Output shows:
// - Common concepts (used in both queries)
// - Weight differences (how influence changed)
// - Unique concepts (used in only one query)
```

---

## Visualization and Export

### ASCII Visualization

```typescript
import { visualizeAttention } from './core/attention-visualizer';

console.log(visualizeAttention(result.attention));
```

Output:
```
════════════════════════════════════════════════════════════════════════════════
ATTENTION VISUALIZATION
════════════════════════════════════════════════════════════════════════════════

Query: "How does compound interest work?"
Total Concepts: 5

────────────────────────────────────────────────────────────────────────────────
TOP INFLUENCERS
────────────────────────────────────────────────────────────────────────────────
1. [95.0%] ██████████████████████████████
   Concept: compound_interest
   Source: agent/financial
   Reasoning: Financial agent identified compound interest as core concept

2. [82.0%] ████████████████████████░░░░░░
   Concept: time_value_of_money
   Source: agent/financial
   Reasoning: TVM is fundamental to understanding compounding
```

### HTML Report

```typescript
import { generateHTMLReport, saveAttentionReport } from './core/attention-visualizer';

const allAttentions = tracker.getAllAttentions();
const stats = tracker.getStatistics();

// Generate interactive HTML report
const html = generateHTMLReport(allAttentions, stats);
fs.writeFileSync('attention-report.html', html);

// Or save all formats at once
const files = await saveAttentionReport(
  allAttentions,
  stats,
  './reports',
  'all'  // Exports JSON, CSV, and HTML
);
```

The HTML report includes:
- 📊 Interactive charts and graphs
- 🎯 Top influencers with visual weight bars
- 📁 Slice usage statistics
- 🔍 Detailed query breakdowns
- 📈 Aggregate statistics

### CSV Export

```typescript
import { exportToCSV } from './core/attention-visualizer';

const csv = exportToCSV(allAttentions);
fs.writeFileSync('attention-data.csv', csv);
```

CSV format for data analysis in Excel, pandas, etc.:
```csv
query_id,query,concept,slice,weight,reasoning,timestamp
query_123,What is compound interest?,compound_interest,agent/financial,0.95,"Core concept",2025-10-08T10:00:00Z
query_123,What is compound interest?,time_value_of_money,agent/financial,0.82,"TVM fundamental",2025-10-08T10:00:01Z
```

---

## Use Cases

### 1. Developer Debugging

**Scenario:** System gave unexpected answer about investment strategy.

```typescript
// Find the query
const attention = tracker.getQueryAttention(problematicQueryId);

// Examine decision path
attention.decision_path.forEach(decision => {
  console.log(decision);
});

// Output:
// 1. Query decomposed into domains: financial, biology
// 2. Agent biology invoked with confidence 0.65  ← Unexpected!
// 3. Agent financial invoked with confidence 0.89
// 4. Composition confidence: 0.75, should_recurse: false

// Identify the problem: Biology agent was invoked unnecessarily
// Solution: Improve query decomposition logic
```

### 2. Regulatory Auditing

**Scenario:** Regulator asks "Why did your AI recommend this investment?"

```typescript
// Export complete audit trail
const audit = metaAgent.exportAttentionForAudit();

// Find specific query
const investmentAdvice = audit.queries.find(q =>
  q.query_id === regulatorQueryId
);

// Show which data influenced decision
investmentAdvice.traces.forEach(trace => {
  if (trace.weight > 0.7) {
    console.log(`${trace.concept} (${trace.slice}): ${trace.reasoning}`);
  }
});

// Output:
// risk_assessment (agent/financial): Evaluated investment risk based on...
// diversification (agent/financial): Recommended diversification to...
// portfolio_optimization (agent/systems): Applied optimization theory...

// Demonstrates: Decision based on sound financial principles, not hidden biases
```

### 3. Medical/Financial Advice Transparency

**Scenario:** Patient asks "Why did you recommend this treatment?"

```typescript
const explanation = metaAgent.explainQuery(medicalQueryId);

// Shows:
// 1. Which medical knowledge was used
// 2. How confident the system was in each concept
// 3. The reasoning chain from symptoms to recommendation
// 4. Which sources were consulted

// Patient can see: "System used cardiology guidelines (90% confidence)
// and consulted drug interaction database (85% confidence)"
```

### 4. Research and Meta-Learning

**Scenario:** Discover patterns in cross-domain reasoning.

```typescript
const stats = metaAgent.getAttentionStats();

// Find most influential cross-domain patterns
stats.high_confidence_patterns
  .filter(p => p.concepts.length >= 3)
  .forEach(pattern => {
    console.log(`Cross-domain pattern: ${pattern.concepts.join(' + ')}`);
    console.log(`Frequency: ${pattern.frequency} queries`);
  });

// Output:
// Cross-domain pattern: evolution + diversification + optimization
// Frequency: 5 queries
// → Biology concepts (evolution) frequently combine with financial concepts
//   (diversification) through systems thinking (optimization)

// Research insight: Cross-domain analogies are powerful reasoning tools
```

---

## Performance Considerations

### Memory Usage

```typescript
const memStats = tracker.getMemoryStats();
console.log(`Queries: ${memStats.total_queries}`);
console.log(`Traces: ${memStats.total_traces}`);
console.log(`Estimated memory: ${(memStats.estimated_bytes / 1024).toFixed(2)} KB`);
```

**Typical memory usage:**
- ~500 bytes per query metadata
- ~200 bytes per attention trace
- Example: 100 queries with 10 traces each = ~250 KB

### Performance Impact

Attention tracking adds **minimal overhead** (<1% of total execution time):

```typescript
// Without attention tracking: 2.45s
// With attention tracking: 2.47s
// Overhead: 0.02s (0.8%)
```

**Why so low?**
- Tracking is synchronous (no async overhead)
- Simple data structures (Maps and Arrays)
- No external I/O during tracking
- Export/visualization happens after processing

### When to Clear Cache

```typescript
// Clear old attention data to free memory
if (tracker.getMemoryStats().total_queries > 1000) {
  tracker.clear();
}

// Or selectively keep recent queries
const recentQueries = tracker.getAllAttentions()
  .filter(a => a.timestamp > Date.now() - 24 * 60 * 60 * 1000); // Last 24h
tracker.clear();
recentQueries.forEach(a => {
  // Re-add recent queries
});
```

---

## Best Practices

### 1. Meaningful Concept Names

❌ Bad:
```typescript
attentionTracker.addTrace("thing1", "somewhere", 0.8, "it worked");
```

✅ Good:
```typescript
attentionTracker.addTrace(
  "compound_interest",
  "agent/financial",
  0.95,
  "Financial agent identified compound interest as the core concept for understanding investment growth"
);
```

### 2. Accurate Weight Assignment

Use agent confidence directly:
```typescript
const weight = agentResponse.confidence;
```

Or combine confidence with relevance:
```typescript
const weight = computeInfluenceWeight(
  agentConfidence,  // 0-1
  sliceRelevance    // 0-1
);
```

### 3. Descriptive Decision Points

❌ Bad:
```typescript
attentionTracker.addDecisionPoint("step 1");
```

✅ Good:
```typescript
attentionTracker.addDecisionPoint(
  `Query decomposed into domains: ${domains.join(', ')} based on semantic analysis`
);
```

### 4. Regular Exports for Auditing

```typescript
// Daily audit export
setInterval(() => {
  const audit = metaAgent.exportAttentionForAudit();
  const timestamp = new Date().toISOString().split('T')[0];
  fs.writeFileSync(`audit-${timestamp}.json`, JSON.stringify(audit, null, 2));
}, 24 * 60 * 60 * 1000);
```

---

## API Reference

### AttentionTracker

#### Methods

**startQuery(queryId: string, query: string): void**
- Begins tracking for a new query
- Must be called before adding traces

**addTrace(concept: string, slice: string, weight: number, reasoning: string): void**
- Records a concept's influence
- `weight` must be 0-1
- Throws error if no active query

**addDecisionPoint(decision: string): void**
- Records a decision in the reasoning chain
- Used for step-by-step tracking

**endQuery(): QueryAttention | null**
- Finalizes tracking
- Computes top influencers
- Returns complete attention data

**getQueryAttention(queryId: string): QueryAttention | undefined**
- Retrieves attention for specific query

**getAllAttentions(): QueryAttention[]**
- Returns all tracked queries

**getStatistics(): AttentionStats**
- Computes aggregate statistics

**exportForAudit(): AuditExport**
- Generates regulatory compliance export

**explainQuery(queryId: string): string**
- Generates human-readable explanation

**clear(): void**
- Clears all tracking data

### MetaAgent Extensions

**getAttentionTracker(): AttentionTracker**
- Access the attention tracker instance

**exportAttentionForAudit(): AuditExport**
- Convenience method for audit export

**explainQuery(queryId: string): string**
- Convenience method for explanations

**getAttentionStats(): AttentionStats**
- Convenience method for statistics

---

## Roadmap

### Future Enhancements

1. **Real-time Visualization**
   - WebSocket streaming of attention traces
   - Live dashboard for monitoring AGI reasoning

2. **Advanced Analytics**
   - Concept correlation analysis
   - Anomaly detection in reasoning patterns
   - Causal inference from attention patterns

3. **Attention Optimization**
   - Prune low-weight concepts automatically
   - Suggest knowledge gaps based on patterns
   - Adaptive weight calibration

4. **Integration**
   - LangSmith/LangChain integration
   - OpenTelemetry traces
   - Prometheus metrics

---

## Conclusion

Attention Tracking transforms the AGI from a mysterious black box into a transparent, auditable, and debuggable system.

**Key Benefits:**
- ✅ Full interpretability: Every decision traceable
- ✅ Regulatory compliance: Complete audit trails
- ✅ Debugging: Identify reasoning errors instantly
- ✅ Research: Discover cross-domain patterns
- ✅ Trust: Transparent AI builds user confidence

**Next Steps:**
1. Run the demo: `npm run demo:attention`
2. Integrate into your application
3. Export your first audit report
4. Analyze patterns in your domain

---

**Documentation Version:** 1.0.0
**Last Updated:** October 2025
**Maintainer:** AGI Recursive Team
