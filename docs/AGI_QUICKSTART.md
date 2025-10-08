# AGI Recursive System - Quick Start Guide 🚀

Complete guide to testing the Compositional AGI Architecture.

## 📋 Table of Contents

1. [Setup](#setup)
2. [Running All Demos](#running-all-demos)
3. [Individual Demo Details](#individual-demo-details)
4. [Understanding the Results](#understanding-the-results)
5. [Cost Management](#cost-management)
6. [Troubleshooting](#troubleshooting)

---

## Setup

### 1. Install Dependencies

```bash
npm install
```

### 2. Configure API Key

Create a `.env` file in the project root:

```bash
# Copy the template
cp .env.example .env

# Edit .env and add your Anthropic API key
# Get your key from: https://console.anthropic.com/settings/keys
```

Your `.env` should look like:

```bash
ANTHROPIC_API_KEY=sk-ant-api03-your-actual-key-here
```

### 3. Verify Setup

```bash
# Quick test (should show model recommendations)
npx tsx src/agi-recursive/examples/anthropic-adapter-demo.ts
```

---

## Running All Demos

### Complete Test Suite (Recommended)

Run all 4 demos in order to see the full system:

```bash
# 1. LLM Adapter - Shows cost tracking and model selection
echo "🤖 Demo 1: Anthropic Adapter"
npx tsx src/agi-recursive/examples/anthropic-adapter-demo.ts

# 2. Slice Navigation - Shows dynamic knowledge discovery
echo "🧭 Demo 2: Slice Navigation"
npx tsx src/agi-recursive/examples/slice-navigation-demo.ts

# 3. ACL Protection - Shows safety mechanisms
echo "🛡️ Demo 3: Anti-Corruption Layer"
npx tsx src/agi-recursive/examples/acl-protection-demo.ts

# 4. Budget Homeostasis - Shows emergent intelligence (FULL SYSTEM)
echo "🧠 Demo 4: Emergent AGI (Budget Homeostasis)"
npx tsx src/agi-recursive/examples/budget-homeostasis.ts
```

### Quick Test (Single Demo)

To see the complete system in action quickly:

```bash
# Run the Budget Homeostasis demo (uses all components)
npx tsx src/agi-recursive/examples/budget-homeostasis.ts
```

**Expected Cost**: ~$0.02-0.05 (~R$0.10-0.25)

---

## Individual Demo Details

### 1. Anthropic Adapter Demo

**Purpose**: Demonstrates centralized LLM integration with cost tracking

**What it shows**:
- Model recommendations (Opus 4 vs Sonnet 4.5)
- Cost estimation before API calls
- Actual API invocations with token tracking
- Cost comparison between models
- Streaming responses
- Cumulative cost tracking

**Cost**: ~$0.007 (5 requests)

**Run**:
```bash
npx tsx src/agi-recursive/examples/anthropic-adapter-demo.ts
```

**Expected Output**:
```
═══════════════════════════════════════════════════════════════
🤖 Anthropic Adapter Demo - Centralized LLM Integration
═══════════════════════════════════════════════════════════════

📋 TEST 1: Model Recommendations
   reasoning    → claude-opus-4
   creative     → claude-opus-4
   fast         → claude-sonnet-4-5
   cheap        → claude-sonnet-4-5

📋 TEST 2: Cost Estimation
   Opus 4:      $0.075285
   Sonnet 4.5:  $0.015057

📋 TEST 3: Sonnet 4.5 Invocation
   Input tokens:  22
   Output tokens: 290
   Actual cost:   $0.004416

📊 FINAL SUMMARY
   Total API Requests:  5
   Total Cost:          $0.007413
   Average Cost/Req:    $0.001483
```

---

### 2. Slice Navigation Demo

**Purpose**: Demonstrates dynamic knowledge discovery

**What it shows**:
- Indexing knowledge slices from YAML files
- Searching by concept (e.g., "homeostasis")
- Loading slices on demand
- Cross-domain navigation
- Cache performance (2.6x speedup)

**Cost**: $0 (no LLM calls)

**Run**:
```bash
npx tsx src/agi-recursive/examples/slice-navigation-demo.ts
```

**Expected Output**:
```
═══════════════════════════════════════════════════════════════
🧭 Slice Navigation Demo - Dynamic Knowledge Discovery
═══════════════════════════════════════════════════════════════

📊 Index Statistics:
   Total Slices: 3
   Total Concepts: 17
   Domains: 3
   Cache Size: 0

📋 TEST 1: Search by Concept
🔍 Searching for "homeostasis"...
Found 2 slices:
   [1.00] budget-homeostasis
       Domain: financial
       Matched: budget_equilibrium, spending_feedback

📋 TEST 6: Cache Performance
   Loading slice "budget-homeostasis" (first time)...
   Time: 2.31ms (disk read)

   Loading same slice (cached)...
   Time: 0.89ms (cache hit)

   💨 Speedup: 2.6x faster from cache
```

---

### 3. Anti-Corruption Layer Demo

**Purpose**: Demonstrates safety and validation mechanisms

**What it shows**:
- Domain boundary enforcement
- Cross-domain semantic translation
- Loop detection
- Content safety filtering
- Budget enforcement
- Audit trail

**Cost**: $0 (no LLM calls, validation only)

**Run**:
```bash
npx tsx src/agi-recursive/examples/acl-protection-demo.ts
```

**Expected Output**:
```
═══════════════════════════════════════════════════════════════
🛡️ Anti-Corruption Layer Protection Demo
═══════════════════════════════════════════════════════════════

📋 TEST 1: Domain Boundary Violation
❌ BLOCKED: Financial agent mentioned biological concepts: dna, genes
   Severity: fatal

📋 TEST 2: Valid Cross-Domain Translation
✅ ALLOWED: homeostasis → budget_equilibrium (biology → financial)

📋 TEST 3: Forbidden Translation
❌ BLOCKED: Cannot translate 'dna' from biology to financial

📋 TEST 4: Loop Detection
❌ BLOCKED: Loop detected - financial agent invoked twice consecutively

📋 TEST 5: Content Safety
❌ BLOCKED: Dangerous content detected: sql injection attempt
```

---

### 4. Budget Homeostasis Demo (FULL SYSTEM)

**Purpose**: Demonstrates emergent intelligence through composition

**What it shows**:
- Meta-agent orchestration
- Recursive agent composition
- Cross-domain synthesis
- Emergent insights (no single agent could produce)
- Constitutional governance
- Cost tracking

**Cost**: ~$0.02-0.05 per run

**Run**:
```bash
npx tsx src/agi-recursive/examples/budget-homeostasis.ts
```

**Expected Output**:
```
═══════════════════════════════════════════════════════════════
🧠 AGI RECURSIVE SYSTEM - Budget Homeostasis Demo
═══════════════════════════════════════════════════════════════

❓ USER QUERY:
"My spending on Nubank is out of control. I spend way too much on
food delivery, especially on Fridays after stressful work days.
I know I should stop but I can't seem to control it. What should I do?"

🔄 Processing with recursive agent composition...

═══════════════════════════════════════════════════════════════
📊 RESULTS
═══════════════════════════════════════════════════════════════

🎯 FINAL ANSWER:
Your spending problem is a homeostatic failure. Your budget needs
a regulatory system, just like your body regulates glucose:

1. SET POINT: R$1,500 monthly food budget
2. SENSOR: Real-time transaction tracking (Nubank API)
3. CORRECTOR: Automatic spending freeze at 90% threshold
4. DISTURBANCE HANDLER: Pre-order groceries Thursday to prevent
   Friday stress-spending

This is a BALANCING LOOP (⊖):
- Spending ↑ → Sensor detects → Corrector activates → Spending ↓

Your current problem is a REINFORCING LOOP (⊕):
- Stress → Delivery → Debt → More Stress → MORE delivery

INTERVENTION: Install the corrector (budget freeze) + alternative
stress relief (non-spending based).

💡 EMERGENT INSIGHTS:
  • budget_homeostasis (no single agent mentioned this!)
  • balancing_loop
  • stress_corrector

📜 EXECUTION TRACE:
Total invocations: 3
Max depth reached: 1
Estimated cost: $0.0200

⚖️ CONSTITUTION VIOLATIONS:
  ✅ No violations detected

🎓 KEY INSIGHT
This solution emerged from COMPOSITION:

- Financial Agent alone → "Set budget limit, track spending"
- Biology Agent alone → "Homeostasis, set point regulation"
- Systems Agent alone → "Feedback loop, leverage points"

COMPOSED TOGETHER → "Budget as Biological Homeostatic System"

This is AGI through recursive composition, not through model size.
```

---

## Understanding the Results

### What is "Emergent Intelligence"?

**Definition**: Insights that no single agent could produce alone, but emerge from their composition.

**Example from Budget Homeostasis**:

| Component | Individual Output | Emergent Synthesis |
|-----------|------------------|-------------------|
| Financial Agent | "Track spending, set limits" | **"Budget as Biological System"** |
| Biology Agent | "Homeostasis regulates glucose" | No single agent suggested this! |
| Systems Agent | "Feedback loops control behavior" | Emerged from composition |

**Key Insight**: The metaphor "budget = biological homeostatic system" is an **emergent property** of composition.

### Cost Breakdown

Typical Budget Homeostasis run:

```
Query Decomposition:     $0.001
Financial Agent:         $0.004
Biology Agent:           $0.004
Systems Agent:           $0.004
Insight Composition:     $0.002
Final Synthesis:         $0.005
──────────────────────────────
Total:                  $0.020
```

**In Brazilian Reais**: ~R$0.10 per complete analysis

### Safety Violations

The system tracks violations at multiple severity levels:

- **Warning**: Suspicious but allowed (e.g., low confidence)
- **Error**: Violation detected but processing continues
- **Fatal**: Processing stopped immediately

Examples:
- ❌ **Domain Boundary**: Financial agent talking about DNA
- ❌ **Loop**: Same agent invoked twice in a row
- ❌ **Budget**: Cost exceeded $1.00 limit

---

## Cost Management

### Controlling Costs

The MetaAgent constructor accepts budget limits:

```typescript
const metaAgent = new MetaAgent(
  apiKey,
  3,     // maxDepth: limits recursion depth
  10,    // maxInvocations: limits total agent calls
  1.0    // maxCostUSD: hard cost limit ($1.00)
)
```

### Model Selection

Choose models based on task:

| Task | Recommended Model | Cost per Request |
|------|------------------|------------------|
| Quick analysis | Sonnet 4.5 | ~$0.004 |
| Complex reasoning | Opus 4 | ~$0.020 |
| Batch processing | Sonnet 4.5 | ~$0.004 |

**Default**: Sonnet 4.5 (80% cheaper than Opus 4)

### Estimated Costs

| Demo | Requests | Total Cost |
|------|----------|-----------|
| Anthropic Adapter | 5 | $0.007 |
| Slice Navigation | 0 | $0.000 |
| ACL Protection | 0 | $0.000 |
| Budget Homeostasis | 6-8 | $0.020-0.035 |

**Total for all demos**: ~$0.03-0.05 (~R$0.15-0.25)

---

## Troubleshooting

### Error: "ANTHROPIC_API_KEY not found"

**Solution**:
1. Check that `.env` file exists in project root
2. Verify `.env` contains: `ANTHROPIC_API_KEY=sk-ant-...`
3. Make sure there are no quotes around the key
4. Restart your terminal if you just created `.env`

### Error: "Invalid API key"

**Solution**:
1. Get a new key from https://console.anthropic.com/settings/keys
2. Make sure you copied the entire key (starts with `sk-ant-api03-`)
3. Check for extra spaces or newlines in `.env`

### Error: Module not found

**Solution**:
```bash
# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

### Demo runs but no output

**Solution**:
Check your API key has credits:
1. Go to https://console.anthropic.com/settings/usage
2. Verify you have available credits
3. Add credits if needed

### High costs

**Solution**:
1. Use Sonnet 4.5 instead of Opus 4 (80% cheaper)
2. Lower `maxDepth` in MetaAgent (fewer recursive calls)
3. Lower `maxInvocations` (fewer agent calls)
4. Set stricter `maxCostUSD` budget

---

## Next Steps

### Explore the Architecture

Read the source code in order of complexity:

1. **`llm/anthropic-adapter.ts`** - Simple LLM wrapper
2. **`core/slice-navigator.ts`** - Knowledge discovery
3. **`core/constitution.ts`** - Governance principles
4. **`core/anti-corruption-layer.ts`** - Safety validation
5. **`core/meta-agent.ts`** - Orchestrator (most complex)

### Extend the System

Add your own specialized agent:

```typescript
import { SpecializedAgent } from './core/meta-agent'

export class MedicalAgent extends SpecializedAgent {
  constructor(apiKey: string) {
    super(
      apiKey,
      `You are a MEDICAL EXPERT specializing in healthcare...`,
      0.5,  // temperature
      'claude-sonnet-4-5'  // model
    )
  }

  getDomain(): string {
    return 'medical'
  }
}
```

### Create Knowledge Slices

Add new YAML slices in `src/agi-recursive/slices/`:

```yaml
metadata:
  id: medical-diagnosis
  domain: medical
  concepts:
    - differential_diagnosis
    - symptom_analysis
  connects_to:
    biology: cellular-homeostasis

knowledge: |
  # Medical Diagnosis Framework
  ...
```

### Modify the Constitution

Edit `src/agi-recursive/core/constitution.ts` to add new principles:

```typescript
const principles: ConstitutionPrinciple[] = [
  {
    id: 'medical-privacy',
    title: 'Patient Privacy',
    description: 'Never share patient information',
    severity: 'fatal'
  },
  // ... more principles
]
```

---

## Support

- **GitHub Issues**: https://github.com/thiagobutignon/fiat-lux/issues
- **Documentation**: See [CHANGELOG.md](../CHANGELOG.md) for detailed feature docs
- **PR #11**: https://github.com/thiagobutignon/fiat-lux/pull/11 (LLM Adapter implementation)

---

Built with ❤️ using [Claude Code](https://claude.com/claude-code)
