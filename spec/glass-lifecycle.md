# .glass Lifecycle Specification

**Version**: 1.0.0
**Date**: 2025-10-09
**Author**: AZUL Node
**Related**: glass-format-v1.md

---

## 1. Overview

### 1.1 Biological Analogy

The `.glass` organism follows a biological lifecycle inspired by cellular organisms:

```
Cell Birth → Growth → Maturity → Reproduction → Death
    ↓           ↓         ↓           ↓            ↓
.glass:
Nascimento → Infância → Maturidade → Reprodução → Retirement
  (0%)      (0-25%)     (75-100%)      (cloning)    (old-but-gold)
```

### 1.2 Core Principle

**Organisms are CULTIVATED, not PROGRAMMED.**

Traditional approach:
```
Write code → Test → Deploy
(human writes every line)
```

.glass approach:
```
Create base → Ingest knowledge → Code EMERGES → Evolves continuously
(human provides knowledge, code synthesizes itself)
```

---

## 2. Lifecycle States

### 2.1 State Diagram

```
     CREATE
        ↓
   ┌────────────┐
   │  NASCENT   │  0-0% maturity
   │   (Birth)  │  Base model only
   └─────┬──────┘
         │ ingest()
         ↓
   ┌────────────┐
   │   INFANT   │  0-25% maturity
   │  (Infancy) │  Learning basics
   └─────┬──────┘
         │ organize()
         ↓
   ┌────────────┐
   │ ADOLESCENT │  25-75% maturity
   │(Adolescence)│ Specializing
   └─────┬──────┘
         │ emerge()
         ↓
   ┌────────────┐
   │   MATURE   │  75-100% maturity
   │ (Maturity) │  Production ready
   └─────┬──────┘
         │
    ┌────┴─────┬──────────┐
    │          │          │
    ↓          ↓          ↓
EVOLVING   CLONING    RETIRING
(continuous) (reproduce) (old-but-gold)
```

### 2.2 State Definitions

#### NASCENT (0% maturity)

**Characteristics:**
- Just created, empty organism
- Base model weights only (27M params)
- No domain knowledge
- No emerged functions
- No episodic memory
- Constitutional principles defined but not tested

**Metrics:**
```typescript
{
  maturity: 0.0,
  state: "nascent",
  knowledge: {
    papers: 0,
    embeddings: 0,
    patterns: 0
  },
  code: {
    functions: 0,
    emerged: 0
  },
  memory: {
    episodes: 0
  },
  size: ~150MB
}
```

**Capabilities:**
- Can respond to general queries (via base model)
- Cannot perform specialized tasks
- No domain expertise

**Transitions to:** INFANT (via ingest)

---

#### INFANT (0-25% maturity)

**Characteristics:**
- Absorbing knowledge
- Building embeddings from papers/datasets
- Identifying basic patterns
- No or very few emerged functions yet
- Learning domain vocabulary

**Metrics:**
```typescript
{
  maturity: 0.0 - 0.25,
  state: "infant",
  knowledge: {
    papers: 100 - 3000,
    embeddings: 100 - 3000,
    patterns: 0 - 50
  },
  code: {
    functions: 0 - 3,
    emerged: 0 - 3
  },
  memory: {
    episodes: 0 - 100
  },
  size: ~200MB - 500MB
}
```

**Capabilities:**
- Can retrieve basic facts from ingested knowledge
- Can cite sources
- Limited synthesis ability
- No complex reasoning yet

**Key Activities:**
- Embedding generation
- Knowledge graph construction (early)
- Pattern detection (threshold not yet reached)

**Transitions to:** ADOLESCENT (via organize + emerge)

---

#### ADOLESCENT (25-75% maturity)

**Characteristics:**
- Patterns becoming clear
- Functions starting to EMERGE from patterns
- Specialization taking shape
- Knowledge graph well-formed
- Hypothesis testing begins

**Metrics:**
```typescript
{
  maturity: 0.25 - 0.75,
  state: "adolescent",
  knowledge: {
    papers: 3000 - 8000,
    embeddings: 3000 - 8000,
    patterns: 50 - 200
  },
  code: {
    functions: 3 - 15,
    emerged: 3 - 15  // All emerged, none programmed!
  },
  memory: {
    episodes: 100 - 500
  },
  size: ~500MB - 1.5GB
}
```

**Capabilities:**
- Can perform specialized tasks in domain
- Can synthesize information across multiple sources
- Emerging reasoning abilities
- Functions auto-generated from patterns

**Key Activities:**
- **CODE EMERGENCE** (pattern threshold reached → function synthesized)
- Constitutional validation of emerged functions
- Accuracy testing on known cases
- Knowledge graph refinement

**Example:**
```typescript
// Pattern detected: "drug X + cancer Y = efficacy Z" (occurred 847 times)
// Threshold: 500 occurrences
// Action: Synthesize function

function analyze_treatment_efficacy(
  cancer_type: CancerType,
  drug: Drug,
  stage: Stage
): Efficacy {
  // Implementation EMERGED from 847 pattern occurrences
  // Confidence: 0.89
  // Validated: true
  // Constitutional: ✅
}
```

**Transitions to:** MATURE (via continued learning + emergence)

---

#### MATURE (75-100% maturity)

**Characteristics:**
- Full specialization achieved
- 10+ emerged functions
- Comprehensive knowledge of domain
- High confidence (>85%)
- Production-ready
- Constitutional compliance tested

**Metrics:**
```typescript
{
  maturity: 0.75 - 1.0,
  state: "mature",
  knowledge: {
    papers: 8000+,
    embeddings: 8000+,
    patterns: 200+
  },
  code: {
    functions: 15 - 50,
    emerged: 15 - 50  // All emerged!
  },
  memory: {
    episodes: 500+
  },
  fitness: 0.85 - 0.97,
  size: ~1.5GB - 2.5GB
}
```

**Capabilities:**
- Expert-level performance in specialized domain
- Complex multi-step reasoning
- High accuracy on domain tasks
- Self-aware of limitations
- Cites sources with attention weights

**Key Activities:**
- Responding to queries
- Continuous learning from interactions
- Function refinement
- Episodic memory consolidation

**Transitions to:**
- EVOLVING (continuous improvement)
- CLONING (reproduction)
- RETIRING (if superseded)

---

#### EVOLVING (continuous state)

**Characteristics:**
- Ongoing improvement
- Learning from every interaction
- Refining existing functions
- Emerging new functions as patterns accumulate
- Fitness trajectory increasing

**Metrics:**
```typescript
{
  state: "evolving",
  fitness: {
    current: 0.89,
    trajectory: [0.72, 0.81, 0.85, 0.87, 0.89],  // Upward trend
    events: [
      { timestamp: "...", type: "function_refined", impact: +0.02 },
      { timestamp: "...", type: "new_function_emerged", impact: +0.03 }
    ]
  }
}
```

**Key Activities:**
- Pattern re-evaluation
- Function optimization
- Weight fine-tuning (via interactions)
- Memory consolidation

**Example Evolution Event:**
```typescript
{
  timestamp: "2025-01-20T15:30:00Z",
  type: "function_refined",
  function: "analyze_treatment_efficacy",
  details: {
    accuracy_before: 0.87,
    accuracy_after: 0.91,
    trigger: "100 new interactions with feedback",
    changes: "Improved confidence calibration"
  },
  fitness_impact: +0.04
}
```

**Continues indefinitely** (or until retirement)

---

#### CLONING (reproduction event)

**Characteristics:**
- Creating offspring .glass organism
- Inherits parent's knowledge and code
- Can be specialized further
- Genetic diversity via variations

**Process:**
```typescript
function clone(parent: GlassFile, specialization: string): GlassFile {
  // Create child
  const child = deepCopy(parent);

  // Update identity
  child.metadata.organism.name = `${parent.name} - ${specialization}`;
  child.metadata.lineage.generation = parent.generation + 1;
  child.metadata.lineage.parent_hash = hash(parent);

  // Reset lifecycle
  child.metadata.lifecycle.created = now();

  // Clear episodic memory (fresh start)
  child.memory.episodic.episodes = [];

  // Inherit knowledge & code (can specialize further)
  child.knowledge = parent.knowledge;
  child.code = parent.code;

  return child;
}
```

**Example:**
```
cancer-research.glass (parent, general oncology)
    ↓ clone("lung cancer")
lung-cancer-research.glass (child, lung cancer specialist)
    ↓ clone("stage 3 lung cancer")
stage3-lung-cancer.glass (grandchild, ultra-specialized)
```

**Benefits:**
- Faster specialization (starts from mature base)
- Preserves parent's learning
- Genetic diversity through variation
- Evolutionary tree of organisms

---

#### RETIRING (end-of-life)

**Characteristics:**
- Superseded by better organism
- Moved to old-but-gold archive
- Never deleted (categorical degradation)
- Can be reactivated if needed

**Categorization:**
```
old-but-gold/
├── 90-100%/         ← Still highly relevant
│   └── cancer-research-v1.0.glass
├── 80-90%/          ← Somewhat relevant
│   └── cancer-research-v0.9.glass
├── 70-80%/          ← Specific use cases
├── 50-70%/          ← Edge cases
└── <50%/            ← Rarely used
    └── cancer-research-v0.1.glass
```

**Fitness-based categorization:**
```typescript
function categorizeRetired(glass: GlassFile): string {
  const fitness = glass.evolution.fitness.current;

  if (fitness >= 0.90) return "90-100%";
  if (fitness >= 0.80) return "80-90%";
  if (fitness >= 0.70) return "70-80%";
  if (fitness >= 0.50) return "50-70%";
  return "<50%";
}
```

**Why never delete?**
- May have systemic dependencies
- Historical learning value
- Potential edge case superiority
- Institutional memory preservation

---

## 3. State Transitions

### 3.1 Nascent → Infant

**Trigger:** `ingest()` operation

**Requirements:**
- At least 100 papers/documents ingested
- Embeddings generated
- Knowledge graph initialized

**Changes:**
```typescript
{
  maturity: 0.0 → 0.05 - 0.25,
  state: "nascent" → "infant",
  knowledge: {
    papers: 0 → 100+,
    embeddings: 0 → 100+
  },
  size: 150MB → 200-500MB
}
```

**Operation:**
```bash
$ fiat ingest cancer-research --source "pubmed:cancer:1000"

Processing:
├─ Downloading 1000 papers...
├─ Generating embeddings...
├─ Building knowledge graph...
└─ State: nascent → infant ✅
```

---

### 3.2 Infant → Adolescent

**Trigger:** Pattern emergence threshold reached

**Requirements:**
- At least 50 patterns detected
- Pattern occurrence threshold: 100+
- Knowledge graph well-formed (1000+ nodes)

**Changes:**
```typescript
{
  maturity: 0.25 → 0.50,
  state: "infant" → "adolescent",
  knowledge: {
    patterns: 50+
  },
  code: {
    functions: 0 → 3+  // First functions emerge!
  }
}
```

**Automatic transition** when conditions met.

---

### 3.3 Adolescent → Mature

**Trigger:** Comprehensive specialization + validation

**Requirements:**
- At least 10 emerged functions
- Fitness > 0.75
- Constitutional compliance: 100%
- Accuracy on test set: >80%

**Changes:**
```typescript
{
  maturity: 0.75 → 1.0,
  state: "adolescent" → "mature",
  code: {
    functions: 10+
  },
  fitness: 0.75+,
  production_ready: true
}
```

**Automatic transition** when conditions met.

---

### 3.4 Mature → Evolving

**Trigger:** Continuous (automatic)

**Requirements:**
- State = mature
- Receiving queries/interactions

**Behavior:**
- Always active in mature organisms
- Runs in background
- Incremental improvements

---

### 3.5 Any State → Cloning

**Trigger:** Explicit `clone()` operation

**Requirements:**
- Parent maturity >= 0.25 (at least infant)
- Specialization defined

**Operation:**
```bash
$ fiat clone cancer-research lung-cancer --specialize "lung cancer"

Creating offspring:
├─ Parent: cancer-research.glass (generation 1)
├─ Child: lung-cancer.glass (generation 2)
├─ Specialization: "lung cancer"
└─ Inherited: knowledge, code, constitutional
```

---

### 3.6 Any State → Retiring

**Trigger:** Fitness degradation OR manual retirement

**Requirements:**
- New organism with higher fitness available
- OR explicit retirement command

**Operation:**
```bash
$ fiat retire cancer-research-v1.0.glass

Retiring:
├─ Current fitness: 0.83
├─ Category: 80-90% (still useful)
├─ Moved to: old-but-gold/80-90%/
└─ Status: retired (can be reactivated)
```

---

## 4. Maturity Calculation

### 4.1 Formula

```typescript
function calculateMaturity(glass: GlassFile): number {
  const weights = {
    knowledge: 0.30,   // 30% - knowledge coverage
    code: 0.40,        // 40% - emerged functions (most important!)
    memory: 0.20,      // 20% - episodic learning
    evolution: 0.10    // 10% - fitness trajectory
  };

  // Knowledge score
  const TARGET_PAPERS = 10000;
  const knowledgeScore = Math.min(1.0,
    glass.knowledge.sources.papers.count / TARGET_PAPERS
  );

  // Code score (emerged functions)
  const TARGET_FUNCTIONS = 25;
  const codeScore = Math.min(1.0,
    glass.code.functions.length / TARGET_FUNCTIONS
  );

  // Memory score (episodic learning)
  const TARGET_EPISODES = 1000;
  const memoryScore = Math.min(1.0,
    glass.memory.episodic.episodes.length / TARGET_EPISODES
  );

  // Evolution score (fitness)
  const evolutionScore = glass.evolution.fitness.current;

  // Weighted sum
  return Math.min(1.0,
    knowledgeScore * weights.knowledge +
    codeScore * weights.code +
    memoryScore * weights.memory +
    evolutionScore * weights.evolution
  );
}
```

### 4.2 Example Calculation

**Nascent organism:**
```typescript
{
  knowledge: { papers: 0 },      // 0 / 10000 = 0.00 → 0.00 * 0.30 = 0.00
  code: { functions: 0 },         // 0 / 25 = 0.00 → 0.00 * 0.40 = 0.00
  memory: { episodes: 0 },        // 0 / 1000 = 0.00 → 0.00 * 0.20 = 0.00
  evolution: { fitness: 0.0 }     // 0.0 → 0.0 * 0.10 = 0.00

  maturity = 0.00 + 0.00 + 0.00 + 0.00 = 0.0%
}
```

**Infant organism:**
```typescript
{
  knowledge: { papers: 2500 },    // 2500 / 10000 = 0.25 → 0.25 * 0.30 = 0.075
  code: { functions: 2 },         // 2 / 25 = 0.08 → 0.08 * 0.40 = 0.032
  memory: { episodes: 150 },      // 150 / 1000 = 0.15 → 0.15 * 0.20 = 0.030
  evolution: { fitness: 0.5 }     // 0.5 → 0.5 * 0.10 = 0.050

  maturity = 0.075 + 0.032 + 0.030 + 0.050 = 18.7% ≈ 19%
}
```

**Mature organism:**
```typescript
{
  knowledge: { papers: 12500 },   // 12500 / 10000 = 1.25 → min(1.0, 1.25) * 0.30 = 0.30
  code: { functions: 23 },        // 23 / 25 = 0.92 → 0.92 * 0.40 = 0.368
  memory: { episodes: 1247 },     // 1247 / 1000 = 1.24 → min(1.0, 1.24) * 0.20 = 0.20
  evolution: { fitness: 0.94 }    // 0.94 → 0.94 * 0.10 = 0.094

  maturity = 0.30 + 0.368 + 0.20 + 0.094 = 96.2% ≈ 96%
}
```

---

## 5. Lifecycle Operations

### 5.1 Create (Birth)

```typescript
async function create(params: {
  name: string;
  specialization: string;
  constitutional: Principle[];
}): Promise<GlassFile> {

  const glass: GlassFile = {
    metadata: {
      organism: {
        name: params.name,
        type: "digital-organism",
        specialization: params.specialization
      },
      lifecycle: {
        maturity: 0.0,
        state: "nascent",
        created: now(),
        last_evolved: now()
      },
      lineage: {
        generation: 1,
        parent_hash: null,
        children: []
      }
    },

    model: loadBaseModel("grammar-lang-27M"),

    knowledge: {
      embeddings: { count: 0, data: [] },
      sources: { papers: { count: 0, providers: [] } },
      patterns: { detected: [] },
      graph: { nodes: 0, edges: 0 }
    },

    code: {
      functions: [],
      emergence_log: []
    },

    memory: {
      episodic: { episodes: [] }
    },

    constitutional: createConstitutional(params.constitutional),

    evolution: {
      enabled: true,
      fitness: { current: 0.0, trajectory: [] }
    }
  };

  return glass;
}
```

**CLI:**
```bash
$ fiat create cancer-research --specialization "oncology"

✅ Created cancer-research.glass
   State: nascent
   Maturity: 0%
   Size: 150MB
```

---

### 5.2 Ingest (Infancy)

```typescript
async function ingest(
  glass: GlassFile,
  sources: Source[]
): Promise<GlassFile> {

  // Download papers/datasets
  const documents = await loadSources(sources);

  // Generate embeddings
  const embeddings = await generateEmbeddings(documents, glass.model);

  // Build knowledge graph
  const graph = await buildKnowledgeGraph(embeddings);

  // Detect patterns
  const patterns = await detectPatterns(graph, threshold: 100);

  // Update glass
  glass.knowledge = {
    embeddings: { count: embeddings.length, data: embeddings },
    sources: catalogSources(sources),
    patterns: { detected: patterns },
    graph: graph
  };

  // Recalculate maturity
  glass.metadata.lifecycle.maturity = calculateMaturity(glass);

  // Update state if transitioned
  if (glass.metadata.lifecycle.maturity >= 0.25) {
    glass.metadata.lifecycle.state = "infant";
  }

  return glass;
}
```

**CLI:**
```bash
$ fiat ingest cancer-research --source "pubmed:cancer+treatment:5000"

Processing:
├─ Downloading 5000 papers from PubMed...
├─ Generating embeddings... [████████░░] 80%
├─ Building knowledge graph... 3247 nodes, 15423 edges
├─ Detecting patterns... 127 patterns found
└─ State: nascent → infant ✅
   Maturity: 0% → 32%
   Size: 150MB → 780MB
```

---

### 5.3 Emerge (Adolescence → Maturity)

```typescript
async function emerge(glass: GlassFile): Promise<GlassFile> {

  // Find significant patterns (threshold: 100+ occurrences)
  const significantPatterns = glass.knowledge.patterns.detected
    .filter(p => p.occurrences >= EMERGENCE_THRESHOLD);

  for (const pattern of significantPatterns) {
    // Check if function already exists for this pattern
    const existing = glass.code.functions.find(f =>
      f.emergence.source_patterns.some(sp => sp.pattern_id === pattern.id)
    );

    if (existing) continue;  // Already emerged

    // Synthesize function from pattern
    const func = await synthesizeFunction({
      pattern: pattern,
      model: glass.model,
      knowledge: glass.knowledge
    });

    // Test on known cases
    const testResults = await testFunction(func, pattern);

    // Constitutional validation
    const constitutional = await validateConstitutional(func, glass.constitutional);

    if (testResults.accuracy >= 0.70 && constitutional.passed) {
      // Add emerged function
      glass.code.functions.push({
        id: generateUUID(),
        name: func.name,
        signature: func.signature,
        emergence: {
          emerged_at: now(),
          trigger: "pattern_threshold",
          source_patterns: [pattern],
          confidence: pattern.confidence
        },
        validation: {
          constitutional: constitutional.passed,
          accuracy: testResults.accuracy,
          test_cases: testResults.count
        },
        implementation: func.code
      });

      // Log emergence
      glass.code.emergence_log.push({
        timestamp: now(),
        event: "function_emerged",
        function_id: func.id,
        details: {
          pattern_id: pattern.id,
          occurrences: pattern.occurrences,
          accuracy: testResults.accuracy
        }
      });
    }
  }

  // Recalculate maturity
  glass.metadata.lifecycle.maturity = calculateMaturity(glass);

  // Update state
  if (glass.metadata.lifecycle.maturity >= 0.25 && glass.metadata.lifecycle.maturity < 0.75) {
    glass.metadata.lifecycle.state = "adolescent";
  } else if (glass.metadata.lifecycle.maturity >= 0.75) {
    glass.metadata.lifecycle.state = "mature";
  }

  return glass;
}
```

**CLI:**
```bash
$ fiat emerge cancer-research

Emerging code from patterns:
├─ Pattern: "drug_efficacy" (1847 occurrences)
│  └─ Synthesizing: analyze_treatment_efficacy()
│     Testing... accuracy: 87% ✅
│     Constitutional: ✅
│     Emerged! ✅
│
├─ Pattern: "clinical_outcomes" (923 occurrences)
│  └─ Synthesizing: predict_drug_interactions()
│     Testing... accuracy: 81% ✅
│     Constitutional: ✅
│     Emerged! ✅
│
└─ Total: 12 functions emerged

State: infant → adolescent ✅
Maturity: 32% → 58%
```

---

### 5.4 Evolve (Continuous)

```typescript
async function evolve(glass: GlassFile): Promise<GlassFile> {

  // Calculate current fitness
  const fitness = await calculateFitness({
    accuracy: await measureAccuracy(glass),
    latency: await measureLatency(glass),
    constitutional: await measureConstitutional(glass),
    satisfaction: await measureSatisfaction(glass)
  });

  // Record fitness
  glass.evolution.fitness.trajectory.push({
    timestamp: now(),
    fitness: fitness,
    event: "periodic_evaluation"
  });

  glass.evolution.fitness.current = fitness;

  // Analyze trajectory
  const previousFitness = glass.evolution.fitness.trajectory[trajectory.length - 2]?.fitness || 0;

  if (fitness < previousFitness) {
    // Fitness degraded - analyze and correct
    const analysis = await analyzeDegradation(glass);
    const corrections = await generateCorrections(analysis);

    for (const correction of corrections) {
      await applyCorrection(glass, correction);

      glass.evolution.learning_events.push({
        timestamp: now(),
        type: "error_correction",
        details: correction,
        impact: correction.expected_improvement
      });
    }
  } else if (fitness > previousFitness) {
    // Fitness improved - record what worked
    glass.evolution.learning_events.push({
      timestamp: now(),
      type: "improvement_detected",
      details: { fitness_gain: fitness - previousFitness },
      impact: fitness - previousFitness
    });
  }

  glass.metadata.lifecycle.last_evolved = now();

  return glass;
}
```

**Automatic** (runs periodically in background)

---

### 5.5 Clone (Reproduction)

```typescript
async function clone(
  parent: GlassFile,
  childName: string,
  specialization: string
): Promise<GlassFile> {

  // Deep copy parent
  const child: GlassFile = deepCopy(parent);

  // Update identity
  child.metadata.organism.id = generateUUID();
  child.metadata.organism.name = `${parent.metadata.organism.name} - ${childName}`;
  child.metadata.organism.specialization = specialization;

  // Update lineage
  child.metadata.lineage.generation = parent.metadata.lineage.generation + 1;
  child.metadata.lineage.parent_hash = contentHash(parent);
  child.metadata.lineage.children = [];

  // Reset lifecycle
  child.metadata.lifecycle.created = now();
  child.metadata.lifecycle.last_evolved = now();
  child.metadata.version.semantic = "0.0.1";
  child.metadata.version.hash = contentHash(child);

  // Clear episodic memory (fresh start for learning)
  child.memory.episodic.episodes = [];

  // Inherit knowledge and code (can be further specialized)
  // child.knowledge = parent.knowledge  (already copied)
  // child.code = parent.code            (already copied)

  // Update parent's children list
  parent.metadata.lineage.children.push(contentHash(child));

  return child;
}
```

**CLI:**
```bash
$ fiat clone cancer-research lung-cancer --specialize "lung cancer"

Creating offspring:
├─ Parent: cancer-research.glass
│  Generation: 1
│  Maturity: 96%
│  Functions: 23
│
├─ Child: lung-cancer.glass
│  Generation: 2
│  Maturity: 96% (inherited)
│  Functions: 23 (inherited)
│  Specialization: "lung cancer"
│
└─ Next: Further specialize via ingest

✅ Cloned successfully
```

---

### 5.6 Retire (End-of-Life)

```typescript
async function retire(
  glass: GlassFile,
  reason: string
): Promise<void> {

  // Calculate fitness-based category
  const fitness = glass.evolution.fitness.current;
  const category = categorizeByFitness(fitness);

  // Move to old-but-gold
  const archivePath = `old-but-gold/${category}/${glass.metadata.organism.name}.glass`;

  await moveFile(glass, archivePath);

  // Update metadata
  glass.metadata.lifecycle.state = "retired";
  glass.metadata.lifecycle.retired_at = now();
  glass.metadata.lifecycle.retirement_reason = reason;

  // Log
  console.log(`Retired ${glass.metadata.organism.name} to ${category}`);
}

function categorizeByFitness(fitness: number): string {
  if (fitness >= 0.90) return "90-100%";
  if (fitness >= 0.80) return "80-90%";
  if (fitness >= 0.70) return "70-80%";
  if (fitness >= 0.50) return "50-70%";
  return "<50%";
}
```

**CLI:**
```bash
$ fiat retire cancer-research-v1.0.glass --reason "Superseded by v2.0"

Retiring:
├─ Organism: cancer-research-v1.0.glass
├─ Current fitness: 0.83
├─ Category: 80-90%
├─ Archive path: old-but-gold/80-90%/
└─ Reason: Superseded by v2.0

✅ Retired (preserved, not deleted)
```

---

## 6. Lifecycle Events

### 6.1 Event Types

```typescript
enum LifecycleEvent {
  CREATED = "created",
  STATE_TRANSITION = "state_transition",
  KNOWLEDGE_INGESTED = "knowledge_ingested",
  PATTERN_DETECTED = "pattern_detected",
  FUNCTION_EMERGED = "function_emerged",
  FUNCTION_REFINED = "function_refined",
  QUERY_EXECUTED = "query_executed",
  MEMORY_CONSOLIDATED = "memory_consolidated",
  FITNESS_EVALUATED = "fitness_evaluated",
  CLONED = "cloned",
  RETIRED = "retired"
}
```

### 6.2 Event Log

Every .glass file maintains an internal event log:

```typescript
interface LifecycleEventLog {
  events: Array<{
    timestamp: ISO8601Timestamp;
    type: LifecycleEvent;
    details: object;
    impact?: {
      maturity_change?: number;
      fitness_change?: number;
      state_change?: string;
    };
  }>;
}
```

**Example log:**
```typescript
[
  {
    timestamp: "2025-01-15T10:00:00Z",
    type: "created",
    details: { name: "cancer-research", specialization: "oncology" },
    impact: { maturity_change: 0, state_change: "nascent" }
  },
  {
    timestamp: "2025-01-15T10:15:00Z",
    type: "knowledge_ingested",
    details: { source: "pubmed", papers: 5000 },
    impact: { maturity_change: 0.32, state_change: "infant" }
  },
  {
    timestamp: "2025-01-15T11:30:00Z",
    type: "function_emerged",
    details: { name: "analyze_treatment_efficacy", pattern: "drug_efficacy", occurrences: 1847 },
    impact: { maturity_change: 0.15, fitness_change: 0.08 }
  },
  {
    timestamp: "2025-01-15T14:00:00Z",
    type: "state_transition",
    details: { from: "adolescent", to: "mature" },
    impact: { maturity_change: 0.12, state_change: "mature" }
  }
]
```

---

## 7. Lifecycle Metrics & Monitoring

### 7.1 Key Metrics

```typescript
interface LifecycleMetrics {
  // Current state
  state: LifecycleState;
  maturity: number;              // 0.0 - 1.0
  fitness: number;               // 0.0 - 1.0
  age: Duration;                 // Time since creation

  // Knowledge
  knowledge: {
    papers: number;
    embeddings: number;
    patterns: number;
    graph_nodes: number;
  };

  // Code
  code: {
    functions: number;
    emerged: number;              // Should equal functions
    avg_confidence: number;
    avg_accuracy: number;
  };

  // Memory
  memory: {
    episodes: number;
    short_term: number;
    long_term: number;
    consolidation_rate: number;
  };

  // Evolution
  evolution: {
    fitness_trajectory: number[];
    learning_events: number;
    mutations: number;
  };

  // Performance
  performance: {
    avg_query_latency: number;    // ms
    queries_per_minute: number;
    constitutional_compliance: number;  // 0.0 - 1.0
  };
}
```

### 7.2 Monitoring Dashboard

```bash
$ fiat status cancer-research

Cancer Research Agent
═══════════════════════════════════════════════════
State:           mature
Maturity:        96.2%
Fitness:         0.94
Age:             15 days

Knowledge:
├─ Papers:       12,500
├─ Embeddings:   12,500
├─ Patterns:     347
└─ Graph nodes:  45,231

Code:
├─ Functions:    23 (all emerged!)
├─ Avg confidence: 0.89
└─ Avg accuracy: 0.87

Memory:
├─ Episodes:     1,247
├─ Short-term:   12
├─ Long-term:    1,235
└─ Consolidation: 98.2%

Evolution:
├─ Fitness trajectory: [0.72, 0.81, 0.87, 0.91, 0.94]
├─ Learning events: 347
└─ Mutations: 23

Performance:
├─ Query latency: 12.3ms avg
├─ Queries/min:   45
└─ Constitutional: 100%

Status: 🟢 Healthy, production-ready
```

---

## 8. Best Practices

### 8.1 Cultivation Tips

**For nascent → infant:**
- Ingest 1000-5000 papers for good foundation
- Choose high-quality, peer-reviewed sources
- Ensure diversity in knowledge sources

**For infant → adolescent:**
- Allow patterns to emerge naturally (don't force)
- Monitor pattern occurrences
- Validate emerged functions thoroughly

**For adolescent → mature:**
- Continue feeding high-quality knowledge
- Test extensively on known cases
- Ensure constitutional compliance

### 8.2 Evolution Best Practices

- Monitor fitness trajectory weekly
- Investigate fitness drops immediately
- Collect user feedback systematically
- Consolidate episodic memory regularly

### 8.3 Cloning Strategy

**When to clone:**
- Parent maturity >= 75% (mature or near-mature)
- Clear specialization need identified
- Parent has proven track record

**How to specialize offspring:**
- Ingest domain-specific knowledge
- Fine-tune on specialized datasets
- Allow new patterns to emerge

---

## 9. Anti-Patterns (What NOT to Do)

### ❌ Forcing maturity

```typescript
// DON'T
glass.metadata.lifecycle.maturity = 1.0;  // Fake maturity

// DO
await ingest(glass, sources);
await emerge(glass);
// Maturity emerges naturally
```

### ❌ Skipping validation

```typescript
// DON'T
glass.code.functions.push(untested_function);

// DO
const validation = await validateFunction(func);
if (validation.passed) {
  glass.code.functions.push(func);
}
```

### ❌ Deleting retired organisms

```typescript
// DON'T
fs.unlinkSync(old_glass_file);

// DO
await retire(glass, "Superseded");
// Moves to old-but-gold/, preserves
```

---

## 10. Future Work

### 10.1 Advanced Lifecycle Features

- **Meta-circular cultivation**: .glass that creates other .glass
- **Swarm lifecycle**: Multiple .glass organisms collaborating
- **Cross-pollination**: Knowledge transfer between organisms
- **Adaptive fitness**: Domain-specific fitness functions

### 10.2 Research Questions

- Optimal emergence threshold?
- Best cloning strategies?
- Maximum sustainable maturity?
- Lifecycle acceleration techniques?

---

**Status**: Lifecycle Specification Complete ✅
**Date**: 2025-10-09
**Next**: Constitutional AI Embedding Specification (Day 3)

