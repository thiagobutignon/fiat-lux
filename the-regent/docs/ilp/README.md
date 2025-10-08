# InsightLoop Protocol (ILP) Documentation

Welcome to the InsightLoop Protocol documentation. ILP is the foundation of The Regent's AGI capabilities.

## 📚 Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Quick Start](#quick-start)
4. [Architecture](#architecture)
5. [Performance](#performance)
6. [Security](#security)

## Overview

The InsightLoop Protocol (ILP/1.0) is an application-level protocol for semantic reasoning exchange between AI agents. It enables recursive composition of specialized agents to achieve emergent intelligence.

### Key Features

✅ **Constitutional Governance** - Runtime enforcement of ethical principles
✅ **O(1) Performance** - Advanced data structures for constant-time operations
✅ **Anti-Corruption Layer** - Domain boundary protection
✅ **Attention Tracking** - Complete reasoning auditability
✅ **Episodic Memory** - Learning from past interactions
✅ **Self-Evolution** - Automatic knowledge base improvement

## Core Concepts

### 1. Meta-Agent Orchestration

The Meta-Agent coordinates specialized agents to solve complex queries:

```typescript
import { MetaAgent } from '@the-regent/core';

const metaAgent = new MetaAgent(apiKey, {
  maxDepth: 5,
  maxInvocations: 10,
  maxCostUSD: 1.0
});

// Register specialized agents
metaAgent.registerAgent('finance', new FinanceAgent(apiKey));
metaAgent.registerAgent('technology', new TechAgent(apiKey));

// Process query with recursive composition
const result = await metaAgent.process('How to optimize investment algorithms?');
```

### 2. Constitutional Principles

Six core principles govern all agent behavior:

1. **Epistemic Honesty** - Admit uncertainty, cite sources
2. **Recursion Budget** - Respect depth/cost/invocation limits
3. **Loop Prevention** - Detect and break infinite cycles
4. **Domain Boundaries** - Stay within expertise
5. **Reasoning Transparency** - Explain all decisions
6. **Safety** - Prevent harm, protect privacy

### 3. O(1) Performance

Advanced data structures ensure constant-time operations:

- **BloomFilter** - O(k) slice existence checks
- **ConceptTrie** - O(m+k) prefix search
- **IncrementalStats** - O(1) statistics
- **LazyIterator** - Constant memory usage
- **DeduplicationTracker** - O(1) duplicate detection

See [Performance Guide](./performance.md) for benchmarks.

### 4. Attention Tracking

Every decision is tracked for auditability:

```typescript
const attention = result.attention;

console.log(attention.top_influencers); // Top 5 most influential concepts
console.log(attention.decision_path);   // Sequence of decisions
console.log(attention.traces);          // All concept influences

// Export for regulatory audit
const auditLog = metaAgent.exportAttentionForAudit();
```

## Quick Start

### Installation

```bash
npm install @the-regent/core
```

### Basic Usage

```typescript
import { MetaAgent, FinanceAgent, TechAgent } from '@the-regent/core';

// 1. Initialize Meta-Agent
const metaAgent = new MetaAgent(process.env.ANTHROPIC_API_KEY);
await metaAgent.initialize();

// 2. Register Specialized Agents
metaAgent.registerAgent('finance', new FinanceAgent(process.env.ANTHROPIC_API_KEY));
metaAgent.registerAgent('technology', new TechAgent(process.env.ANTHROPIC_API_KEY));

// 3. Process Query
const result = await metaAgent.process('What is compound interest?');

console.log(result.final_answer);
console.log(result.emergent_insights);
console.log(result.reasoning_path);

// 4. Check for violations
if (result.constitution_violations.length > 0) {
  console.warn('Constitutional violations:', result.constitution_violations);
}

// 5. Cache Statistics
const cacheStats = metaAgent.getCacheStats();
console.log(`Cache hit rate: ${(cacheStats.hit_rate * 100).toFixed(1)}%`);
```

## Architecture

### Component Hierarchy

```
MetaAgent (Orchestrator)
├── Constitution Enforcer
├── Anti-Corruption Layer
├── Slice Navigator
│   ├── BloomFilter (slice existence)
│   └── ConceptTrie (prefix search)
├── Attention Tracker
│   └── IncrementalStats (O(1) metrics)
├── Episodic Memory
│   ├── ConceptTrie (concept search)
│   ├── BloomFilter (episode exists)
│   └── IncrementalStats (memory stats)
└── Specialized Agents
    ├── Finance Agent
    ├── Technology Agent
    ├── Biology Agent
    └── Custom Agents
```

### Data Flow

1. **Query Received** → Check cache (O(1))
2. **Query Decomposition** → Identify relevant domains
3. **Agent Invocation** → Recursive composition with budget tracking
4. **ACL Validation** → Check domain boundaries
5. **Constitution Check** → Validate principles
6. **Attention Tracking** → Record concept influences
7. **Synthesis** → Combine insights
8. **Cache Result** → Store for future queries

## Performance

### Benchmarks

| Operation | Before | After | Improvement |
|-----------|--------|-------|-------------|
| Slice Existence Check | O(n) | O(k) | 1000x |
| Concept Prefix Search | O(n*m) | O(m+k) | 500x |
| Statistics Computation | O(n*m) | O(1) | ∞ |
| Duplicate Query | O(n) | O(1) | 100x |

### Cache Performance

- **Hit Rate**: 89% (typical)
- **Cost Reduction**: 84%
- **Latency**: <1ms for cache hits

See [O1_OPTIMIZATION.md](../../O1_OPTIMIZATION.md) for detailed analysis.

## Security

### Constitutional Enforcement

All responses are validated against constitutional principles:

```typescript
const enforcer = new ConstitutionEnforcer();

const result = enforcer.validate(
  'finance',
  response,
  context
);

if (!result.passed) {
  console.error('Violations:', result.violations);
  // Handle violations (reject response, log, alert)
}
```

### Anti-Corruption Layer

Prevents domain corruption and prompt injection:

```typescript
const acl = new AntiCorruptionLayer(constitution);

try {
  acl.validateResponse(response, agentDomain, state);
} catch (error) {
  if (error instanceof ConstitutionalViolationError) {
    console.error('Domain violation:', error.message);
    // Fatal violations stop processing
  }
}
```

### Budget Protection

Automatic protection against runaway costs:

```typescript
const budgetStatus = acl.getBudgetStatus(state);

if (!budgetStatus.within_limits) {
  console.warn('Budget exceeded:');
  console.warn(`- Depth: ${budgetStatus.depth}/${budgetStatus.max_depth}`);
  console.warn(`- Cost: $${budgetStatus.cost_usd}/$${budgetStatus.max_cost_usd}`);
  // Processing stops automatically
}
```

## Learn More

- [Constitution Guide](./constitution.md) - Detailed principle explanations
- [Performance Guide](./performance.md) - Optimization strategies
- [Memory System](./memory.md) - Episodic memory and learning
- [Self-Evolution](./self-evolution.md) - Automatic improvement
- [API Reference](./api.md) - Complete API documentation

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

Apache-2.0 - See [LICENSE](../../LICENSE)
