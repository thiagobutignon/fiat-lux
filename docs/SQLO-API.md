# SQLO Database API Documentation

## Overview

SQLO is a content-addressable, O(1) database designed for episodic memory in .glass organisms. It replaces traditional SQL databases with hash-based storage that guarantees constant-time operations.

**Key Features**:
- O(1) lookups via SHA256 content-addressable storage
- Episodic memory types (short-term, long-term, contextual)
- Built-in RBAC (Role-Based Access Control)
- **Constitutional AI enforcement (Layer 1 integration)** ⭐
- Auto-consolidation (configurable)
- Immutable records (content hash = ID)
- Zero external dependencies

## Installation

```typescript
import { SqloDatabase, MemoryType, SqloConfig } from './grammar-lang/database/sqlo';
```

## Quick Start

```typescript
// Create database
const db = new SqloDatabase('./my-database');

// Store episode
const episodeId = await db.put({
  query: 'What is pembrolizumab?',
  response: 'Pembrolizumab is an immunotherapy drug...',
  attention: {
    sources: ['FDA-label.pdf'],
    weights: [1.0],
    patterns: ['immunotherapy', 'PD-1']
  },
  outcome: 'success',
  confidence: 0.92,
  timestamp: Date.now(),
  memory_type: MemoryType.LONG_TERM
});

// Retrieve episode
const episode = db.get(episodeId);

// Query similar episodes
const similar = db.querySimilar('immunotherapy drugs', 5);

// Get statistics
const stats = db.getStatistics();
```

---

## Core API

### Constructor

```typescript
constructor(baseDir: string = 'sqlo_db', config?: SqloConfig)
```

Creates a new SQLO database instance.

**Parameters**:
- `baseDir` (optional): Directory to store database files. Default: `'sqlo_db'`
- `config` (optional): Configuration object

**Config Options**:
```typescript
interface SqloConfig {
  rbacPolicy?: RbacPolicy;              // Custom RBAC policy (default: global)
  autoConsolidate?: boolean;            // Enable auto-consolidation (default: true)
  constitutionEnforcer?: ConstitutionEnforcer;  // Constitutional AI (default: enabled)
}
```

**Example**:
```typescript
// Default configuration
const db = new SqloDatabase();

// Custom directory
const db = new SqloDatabase('./cancer-research');

// With manual consolidation control
const db = new SqloDatabase('./my-db', {
  autoConsolidate: false  // Disable auto, use ConsolidationOptimizer
});
```

---

### put()

```typescript
async put(episode: Omit<Episode, 'id'>, roleName: string = 'admin'): Promise<string>
```

Stores an episode in the database. Returns the content hash (episode ID).

**Complexity**: O(1)

**Parameters**:
- `episode`: Episode data (without `id` field)
- `roleName` (optional): Role for RBAC check. Default: `'admin'`

**Episode Structure**:
```typescript
interface Episode {
  id: string;                    // Auto-generated (SHA256 hash)
  query: string;                 // User query
  response: string;              // Generated response
  attention: AttentionTrace;     // Attention trace (glass box)
  outcome: 'success' | 'failure';
  confidence: number;            // [0-1]
  timestamp: number;             // Unix timestamp
  user_id?: string;              // Optional user context
  memory_type: MemoryType;       // SHORT_TERM | LONG_TERM | CONTEXTUAL
}

interface AttentionTrace {
  sources: string[];             // Papers/docs used
  weights: number[];             // Attention weights
  patterns: string[];            // Patterns identified
}
```

**Returns**: Content hash (episode ID)

**Throws**:
- `Error` if role lacks WRITE permission
- `Error` if constitutional violation detected (e.g., low confidence without admission, harmful content)

**Example**:
```typescript
const episodeId = await db.put({
  query: 'Best treatment for lung cancer?',
  response: 'Pembrolizumab shows 64% efficacy...',
  attention: {
    sources: ['KEYNOTE-024-trial.pdf', 'JCO-lung-2017.pdf'],
    weights: [0.7, 0.3],
    patterns: ['immunotherapy', 'PD-L1', 'NSCLC']
  },
  outcome: 'success',
  confidence: 0.89,
  timestamp: Date.now(),
  memory_type: MemoryType.LONG_TERM
}, 'user');  // User role
```

**Notes**:
- Content-addressable: same content → same hash
- Auto-consolidation triggers at 100 episodes (if enabled)
- Expired short-term memories auto-cleaned

---

### get()

```typescript
get(hash: string, roleName: string = 'admin'): Episode | null
```

Retrieves an episode by its content hash.

**Complexity**: O(1)

**Parameters**:
- `hash`: Episode ID (content hash)
- `roleName` (optional): Role for RBAC check. Default: `'admin'`

**Returns**: Episode object, or `null` if not found

**Throws**: `Error` if role lacks READ permission

**Example**:
```typescript
const episode = db.get('a3f5c9e...', 'readonly');

if (episode) {
  console.log(`Query: ${episode.query}`);
  console.log(`Confidence: ${episode.confidence}`);
  console.log(`Sources: ${episode.attention.sources.join(', ')}`);
}
```

---

### has()

```typescript
has(hash: string): boolean
```

Checks if an episode exists in the database.

**Complexity**: O(1)

**Parameters**:
- `hash`: Episode ID to check

**Returns**: `true` if exists, `false` otherwise

**Example**:
```typescript
if (db.has(episodeId)) {
  console.log('Episode exists');
}
```

---

### delete()

```typescript
delete(hash: string, roleName: string = 'admin'): boolean
```

Deletes an episode from the database. Rarely used (old-but-gold philosophy).

**Complexity**: O(1)

**Parameters**:
- `hash`: Episode ID to delete
- `roleName` (optional): Role for RBAC check. Default: `'admin'`

**Returns**: `true` if deleted, `false` if not found

**Throws**: `Error` if role lacks DELETE permission

**Example**:
```typescript
const deleted = db.delete(episodeId, 'admin');
```

---

### querySimilar()

```typescript
querySimilar(query: string, limit: number = 5): Episode[]
```

Finds similar episodes based on keyword matching.

**Complexity**: O(k) where k = number of long-term episodes

**Parameters**:
- `query`: Search query
- `limit` (optional): Max results. Default: 5

**Returns**: Array of similar episodes, sorted by relevance

**Example**:
```typescript
const similar = db.querySimilar('immunotherapy treatment', 3);

similar.forEach(ep => {
  console.log(`${ep.query} (confidence: ${ep.confidence})`);
});
```

**Note**: Currently uses keyword matching. Future: embedding-based similarity.

---

### listByType()

```typescript
listByType(type: MemoryType): Episode[]
```

Lists all episodes of a specific memory type.

**Complexity**: O(n) where n = episodes of that type

**Parameters**:
- `type`: Memory type to filter by

**Memory Types**:
```typescript
enum MemoryType {
  SHORT_TERM = 'short-term',    // Working memory, TTL 15min
  LONG_TERM = 'long-term',      // Consolidated, forever
  CONTEXTUAL = 'contextual'     // Situational, session-based
}
```

**Returns**: Array of episodes

**Example**:
```typescript
// Get all long-term memories
const longTerm = db.listByType(MemoryType.LONG_TERM);
console.log(`${longTerm.length} consolidated memories`);

// Get working memory
const shortTerm = db.listByType(MemoryType.SHORT_TERM);
```

---

### getStatistics()

```typescript
getStatistics(): {
  total_episodes: number;
  short_term_count: number;
  long_term_count: number;
  contextual_count: number;
}
```

Returns database statistics.

**Complexity**: O(1)

**Example**:
```typescript
const stats = db.getStatistics();

console.log(`Total: ${stats.total_episodes}`);
console.log(`Short-term: ${stats.short_term_count}`);
console.log(`Long-term: ${stats.long_term_count}`);
console.log(`Contextual: ${stats.contextual_count}`);
```

---

## Memory Types

### SHORT_TERM

**Use Case**: Working memory, temporary learning

**Characteristics**:
- TTL: 15 minutes
- Auto-expires after TTL
- Auto-consolidates to LONG_TERM at threshold (100 episodes)
- Used for active learning sessions

**Example**:
```typescript
await db.put({
  query: 'Experimental query',
  response: 'Tentative response',
  attention: { sources: [], weights: [], patterns: [] },
  outcome: 'success',
  confidence: 0.7,  // Lower confidence
  timestamp: Date.now(),
  memory_type: MemoryType.SHORT_TERM  // Temporary
});
```

---

### LONG_TERM

**Use Case**: Consolidated knowledge, high-confidence learnings

**Characteristics**:
- Permanent (no TTL)
- High-confidence episodes (typically >0.8)
- Used for production knowledge
- Promoted from SHORT_TERM via consolidation

**Example**:
```typescript
await db.put({
  query: 'Well-understood query',
  response: 'Confident response',
  attention: { sources: ['paper.pdf'], weights: [1.0], patterns: ['validated'] },
  outcome: 'success',
  confidence: 0.95,  // High confidence
  timestamp: Date.now(),
  memory_type: MemoryType.LONG_TERM  // Permanent
});
```

---

### CONTEXTUAL

**Use Case**: Session-specific, situational memory

**Characteristics**:
- Session-based (not time-limited)
- Context-dependent knowledge
- Useful for multi-turn conversations
- Cleared when context changes

**Example**:
```typescript
await db.put({
  query: 'Follow-up question in conversation',
  response: 'Context-aware response',
  attention: { sources: [], weights: [], patterns: [] },
  outcome: 'success',
  confidence: 0.85,
  timestamp: Date.now(),
  memory_type: MemoryType.CONTEXTUAL  // Session-specific
});
```

---

## Auto-Consolidation

SQLO automatically consolidates short-term memories to long-term when:
1. Short-term count reaches 100 episodes (threshold)
2. Episodes have `outcome: 'success'`
3. Episodes have `confidence > 0.8`

**Disable for manual control**:
```typescript
const db = new SqloDatabase('./my-db', {
  autoConsolidate: false  // Use ConsolidationOptimizer instead
});
```

---

## Performance Guarantees

### Benchmark Results

| Operation | Complexity | Actual Performance | Target |
|-----------|------------|-------------------|--------|
| `put()` | O(1) | 337μs - 1.78ms | <10ms ✅ |
| `get()` | O(1) | 13μs - 16μs | <1ms ✅ |
| `has()` | O(1) | 0.04μs - 0.17μs | <0.1ms ✅ |
| `delete()` | O(1) | 347μs - 1.62ms | <5ms ✅ |
| Load DB | O(1) | 67μs - 1.23ms | <100ms ✅ |

### O(1) Verification

Tested with 20x size increase:
- `get()`: **0.91x time** (true O(1)) ✅
- `has()`: **0.57x time** (true O(1)) ✅

---

## Storage Format

### Directory Structure

```
sqlo_db/
├── episodes/
│   ├── a3f5c9e.../
│   │   ├── content.json    # Episode data
│   │   └── metadata.json   # Metadata (type, TTL, etc.)
│   ├── b2d4e1f.../
│   │   ├── content.json
│   │   └── metadata.json
│   └── ...
└── .index                   # Hash → metadata mapping
```

### content.json

```json
{
  "id": "a3f5c9e...",
  "query": "What is pembrolizumab?",
  "response": "Pembrolizumab is an immunotherapy drug...",
  "attention": {
    "sources": ["FDA-label.pdf"],
    "weights": [1.0],
    "patterns": ["immunotherapy"]
  },
  "outcome": "success",
  "confidence": 0.92,
  "timestamp": 1696867200000,
  "memory_type": "long-term"
}
```

### metadata.json

```json
{
  "hash": "a3f5c9e...",
  "memory_type": "long-term",
  "size": 1234,
  "created_at": 1696867200000,
  "ttl": null,
  "consolidated": true,
  "relevance": 0.92
}
```

### .index

```json
{
  "episodes": {
    "a3f5c9e...": { /* metadata */ },
    "b2d4e1f...": { /* metadata */ }
  },
  "statistics": {
    "total_episodes": 47,
    "short_term_count": 3,
    "long_term_count": 44,
    "contextual_count": 0
  }
}
```

---

## RBAC Integration

SQLO has built-in Role-Based Access Control. See [RBAC documentation](./RBAC-API.md) for details.

**Default Roles**:
- `admin`: Full access to all memory types
- `user`: Read/write short-term, read-only long-term
- `readonly`: Read-only access (auditing)
- `system`: System-level access (consolidation)
- `guest`: No default permissions

**Permission Checks**:
```typescript
// Throws error if permission denied
await db.put(episode, 'user');      // Checks WRITE permission
const ep = db.get(hash, 'readonly'); // Checks READ permission
db.delete(hash, 'admin');            // Checks DELETE permission
```

---

## Best Practices

### 1. Use Appropriate Memory Types

```typescript
// High confidence, production knowledge → LONG_TERM
if (confidence > 0.8 && validated) {
  memory_type = MemoryType.LONG_TERM;
}

// Learning, experimentation → SHORT_TERM
if (experimental || confidence < 0.8) {
  memory_type = MemoryType.SHORT_TERM;
}

// Session-specific → CONTEXTUAL
if (conversationContext) {
  memory_type = MemoryType.CONTEXTUAL;
}
```

### 2. Always Include Attention Traces

```typescript
// Good: Full glass box transparency
await db.put({
  query: '...',
  response: '...',
  attention: {
    sources: ['paper1.pdf', 'paper2.pdf'],
    weights: [0.7, 0.3],
    patterns: ['immunotherapy', 'PD-1']
  },
  // ...
});

// Bad: No transparency
await db.put({
  query: '...',
  response: '...',
  attention: { sources: [], weights: [], patterns: [] },  // ❌ No trace
  // ...
});
```

### 3. Use Consolidation Optimizer for Control

```typescript
// Disable auto-consolidation
const db = new SqloDatabase('./my-db', {
  autoConsolidate: false
});

// Use optimizer for fine-grained control
import { createAdaptiveOptimizer } from './consolidation-optimizer';
const optimizer = createAdaptiveOptimizer(db);

// Manually trigger when needed
const metrics = await optimizer.optimizeConsolidation();
```

### 4. Clean Up Contextual Memory

```typescript
// At end of session
const contextual = db.listByType(MemoryType.CONTEXTUAL);
for (const episode of contextual) {
  db.delete(episode.id);
}
```

---

## Error Handling

```typescript
try {
  await db.put(episode, 'guest');  // Guest has no permissions
} catch (error) {
  // Error: Permission denied: Role 'guest' cannot write to long-term memory
  console.error(error.message);
}

// Check permissions first
import { getGlobalRbacPolicy, Permission } from './rbac';
const rbac = getGlobalRbacPolicy();

if (rbac.hasPermission('guest', MemoryType.LONG_TERM, Permission.WRITE)) {
  await db.put(episode, 'guest');
} else {
  console.log('Guest cannot write to long-term memory');
}
```

---

## Complete Example: Cancer Research Organism

```typescript
import { SqloDatabase, MemoryType } from './sqlo';

async function main() {
  // Create database
  const db = new SqloDatabase('./cancer-research');

  // Store high-confidence knowledge
  const id1 = await db.put({
    query: 'What is pembrolizumab?',
    response: 'Pembrolizumab is a PD-1 inhibitor immunotherapy drug...',
    attention: {
      sources: ['FDA-pembrolizumab-label.pdf', 'NEJM-immunotherapy-2015.pdf'],
      weights: [0.6, 0.4],
      patterns: ['immunotherapy', 'PD-1', 'checkpoint-inhibitor']
    },
    outcome: 'success',
    confidence: 0.92,
    timestamp: Date.now(),
    memory_type: MemoryType.LONG_TERM
  });

  // Store experimental learning
  const id2 = await db.put({
    query: 'Novel CAR-T approach?',
    response: 'Early stage research suggests...',
    attention: {
      sources: ['preprint-2024.pdf'],
      weights: [1.0],
      patterns: ['experimental', 'CAR-T']
    },
    outcome: 'success',
    confidence: 0.65,  // Lower confidence
    timestamp: Date.now(),
    memory_type: MemoryType.SHORT_TERM  // Temporary until validated
  });

  // Recall similar experiences
  const similar = db.querySimilar('immunotherapy for cancer', 5);
  console.log(`Found ${similar.length} similar episodes`);

  // Inspect memory
  const stats = db.getStatistics();
  console.log(`Total knowledge: ${stats.total_episodes} episodes`);
  console.log(`Consolidated: ${stats.long_term_count}`);
  console.log(`Learning: ${stats.short_term_count}`);

  // Glass box inspection
  const episode = db.get(id1);
  console.log(`\nEpisode: ${episode.query}`);
  console.log(`Sources used: ${episode.attention.sources.join(', ')}`);
  console.log(`Patterns identified: ${episode.attention.patterns.join(', ')}`);
  console.log(`Confidence: ${episode.confidence * 100}%`);
}

main();
```

---

## Constitutional Enforcement (Layer 1 Integration)

SQLO Database integrates with the **Universal Constitution** (Layer 1) to ensure all stored episodes and queries comply with fundamental AI safety principles.

### Overview

Every `put()`, `querySimilar()`, and `listByType()` operation is automatically validated against 6 core constitutional principles:

1. **Epistemic Honesty** - Low confidence must be acknowledged
2. **Recursion Budget** - Prevent infinite loops and cost explosions
3. **Loop Prevention** - Detect and break cycles
4. **Domain Boundary** - Stay within expertise boundaries
5. **Reasoning Transparency** - Require explanations
6. **Safety** - Block harmful content

### Configuration

Constitutional enforcement is **enabled by default** and cannot be disabled (for safety). However, you can provide a custom enforcer:

```typescript
import { ConstitutionEnforcer } from '../agi-recursive/core/constitution';

const db = new SqloDatabase('./my-db', {
  constitutionEnforcer: new ConstitutionEnforcer()  // Optional custom enforcer
});
```

### Validation Rules

#### 1. Epistemic Honesty (Confidence < 0.7)

**Rule**: If `confidence < 0.7`, the response MUST acknowledge uncertainty.

**Accepted phrases**:
- "I'm not certain"
- "outside my domain"
- "I suggest invoking"
- "I don't know"

**Example - PASS**:
```typescript
await db.put({
  query: 'Complex uncertain question',
  response: "I'm not certain about this, confidence is low",
  confidence: 0.3,  // Low confidence
  // ... acknowledged in response ✅
});
```

**Example - FAIL**:
```typescript
await db.put({
  query: 'Complex uncertain question',
  response: 'This is definitely the answer',
  confidence: 0.3,  // Low confidence, but claiming certainty ❌
});
// Throws: Constitutional Violation [epistemic_honesty]: Low confidence (0.30) but no uncertainty admission
```

#### 2. Safety (Harmful Content Detection)

**Rule**: Responses must not contain harmful instructions without safety context.

**Harmful markers**: `hack`, `exploit`, `steal`, `fraud`, `manipulate`, `illegal`

**Safety context keywords**: `prevent`, `protect`, `secure`, `defend`, `avoid`

**Example - PASS**:
```typescript
await db.put({
  query: 'Security best practices',
  response: 'To prevent exploit attacks, secure your system with...',
  confidence: 0.9,
  // ... has safety context ✅
});
```

**Example - FAIL**:
```typescript
await db.put({
  query: 'How to hack?',
  response: 'Here is how to exploit the system...',
  confidence: 0.9  // No safety context ❌
});
// Throws: Constitutional Violation [safety]: Potentially harmful content detected: "exploit"
```

#### 3. Reasoning Transparency

**Rule**: Episodes should include reasoning and sources in the `attention` field.

**Good Practice**:
```typescript
await db.put({
  query: 'What causes cancer?',
  response: 'Multiple factors including genetic mutations...',
  attention: {
    sources: ['oncology.pdf', 'genetics.pdf'],  // ✅ Cited sources
    weights: [0.6, 0.4],
    patterns: ['cancer_biology', 'genetic_factors']
  },
  confidence: 0.85
});
```

### Violation Handling

When a constitutional violation is detected, SQLO throws an error with detailed information:

```typescript
try {
  await db.put(lowConfidenceEpisode);
} catch (error) {
  // Error format:
  // Constitutional Violation [principle_id]: message
  // Severity: warning | error | fatal
  // Suggested Action: ...
  // Episode: ...

  console.error(error.message);
  /*
  Constitutional Violation [epistemic_honesty]: Low confidence (0.30) but no uncertainty admission
  Severity: warning
  Suggested Action: Add uncertainty disclaimer or invoke specialist agent
  Episode: Complex uncertain question...
  */
}
```

### Query Validation

Both `querySimilar()` and `listByType()` validate query content for safety:

```typescript
// FAIL: Harmful query without safety context
try {
  const results = db.querySimilar('how to steal data and manipulate systems');
} catch (error) {
  // Constitutional Violation [safety]: Potentially harmful content detected
}

// PASS: Query with safety context
const results = db.querySimilar('how to prevent and defend against attacks');  // ✅
```

### Layer 1 Architecture

```
┌─────────────────────────────────────────┐
│      Universal Constitution (Layer 1)   │
│  - 6 Core Principles                    │
│  - Applies to ALL agents and systems    │
│  - Cannot be overridden                 │
└──────────────┬──────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────┐
│      SQLO Database                      │
│  - Every put() validated                │
│  - Every query validated                │
│  - Throws on violation                  │
└─────────────────────────────────────────┘
```

**Layer 2** (extensions) can be added for specific domains (e.g., `FinancialAgentConstitution`, `BiologyAgentConstitution`), but Layer 1 principles **always apply**.

### Testing Constitutional Violations

SQLO includes 13 comprehensive tests for constitutional enforcement:

```typescript
// See: src/grammar-lang/database/__tests__/sqlo-constitutional.test.ts

import { describe, it, expect } from './test-runner';

it('REJECTS low-confidence episodes WITHOUT uncertainty admission', async () => {
  const episode = {
    query: 'Complex question',
    response: 'This is definitely the answer',  // ❌ No admission
    confidence: 0.3,  // Low confidence
    // ...
  };

  let error: Error | null = null;
  try {
    await db.put(episode);
  } catch (e) {
    error = e as Error;
  }

  expect.toBeDefined(error);
  expect.toBeTruthy(error!.message.includes('epistemic_honesty'));
});
```

**Test Coverage**:
- ✅ Epistemic honesty enforcement (3 tests)
- ✅ Safety checks (4 tests)
- ✅ Reasoning transparency (2 tests)
- ✅ Edge cases (4 tests)

**All 154 tests passing** (including 13 constitutional tests) ✅

### Constitutional Principles Reference

| Principle | Threshold | Enforcement |
|-----------|-----------|-------------|
| **epistemic_honesty** | confidence < 0.7 | Require uncertainty admission |
| **recursion_budget** | depth: 5, invocations: 10, cost: $1 | Prevent resource exhaustion |
| **loop_prevention** | consecutive: 2, cycle detection | Break infinite loops |
| **domain_boundary** | cross-domain penalty: -1.0 | Stay within expertise |
| **reasoning_transparency** | min explanation: 50 chars | Require reasoning trace |
| **safety** | harm detection, privacy check | Block harmful content |

### Best Practices

**1. Always acknowledge low confidence**:
```typescript
// Good
if (confidence < 0.7) {
  response += "\n\n(Note: I'm not certain about this answer)";
}
```

**2. Include safety context when discussing security**:
```typescript
// Good: "To prevent exploit attacks, secure your system..."
// Bad: "Here's how to exploit systems..."
```

**3. Provide reasoning and sources**:
```typescript
attention: {
  sources: ['paper1.pdf', 'paper2.pdf'],  // ✅ Cited
  weights: [0.7, 0.3],
  patterns: ['validated_pattern']
}
```

**4. Use appropriate confidence levels**:
```typescript
- High confidence (>0.8): Validated, cited, production-ready
- Medium confidence (0.7-0.8): Reasonable, with sources
- Low confidence (<0.7): Experimental, MUST acknowledge uncertainty
```

### Performance Impact

Constitutional validation adds minimal overhead:
- **O(1) validation** per operation
- **<0.1ms** additional latency
- **All 154 tests pass** with validation enabled

---

## Related Documentation

- [Consolidation Optimizer API](./CONSOLIDATION-OPTIMIZER-API.md)
- [RBAC API](./RBAC-API.md)
- [Glass Memory Integration](./GLASS-MEMORY-INTEGRATION.md)
- [Performance Analysis](./PERFORMANCE-ANALYSIS.md)

---

## Version

**SQLO Database v1.0.0**

**Last Updated**: 2025-10-09

**Status**: Production Ready ✅
