# LLM Integration Guide

## Overview

The Grammar Language AGI system now features **deep LLM integration** powered by Anthropic Claude, while maintaining **O(1) performance** and **Constitutional AI validation** at every layer.

## Architecture

### Layer 0: Constitutional Foundation
All LLM operations are validated against the **UniversalConstitution**:
- Epistemic honesty (must admit uncertainty)
- Recursion budget (max depth, cost limits)
- Loop prevention (no infinite recursion)
- Domain boundaries (cannot diagnose/prescribe)
- Reasoning transparency (must explain)
- Safety (no harmful outputs)

**Location**: `/src/agi-recursive/core/constitution.ts`

### Layer 1: LLM Adapter
Centralized Anthropic Claude integration with task-specific model selection:
- **Opus 4**: Complex reasoning (code synthesis, intent analysis)
- **Sonnet 4.5**: Fast analysis (patterns, embeddings, sentiment)

**Location**: `/src/grammar-lang/glass/llm-adapter.ts`

**Features**:
- Automatic cost tracking
- Budget enforcement
- Task-specific prompting
- Streaming support
- Constitutional validation

### Layer 2: Domain Integration

#### ROXO (Code Generation)
**Files**:
- `llm-code-synthesis.ts` - LLM-powered .gl code synthesis
- `ingestion.ts` - LLM semantic embeddings
- `patterns.ts` - LLM semantic pattern detection

**What Changed**:
- ❌ Before: Hardcoded templates
- ✅ After: LLM synthesizes .gl code from patterns

#### CINZA (Cognitive Defense)
**Files**:
- `llm-intent-detector.ts` - LLM pragmatic intent analysis
- `pragmatics.ts` - detectIntentWithLLM(), parsePragmaticsWithLLM()
- `semantics.ts` - parseSemanticsWithLLM()

**What Changed**:
- ❌ Before: Rule-based if-else trees
- ✅ After: Deep LLM semantic analysis (implicit meanings, subtext)

#### VERMELHO (Security)
**Files**:
- `linguistic-collector.ts` - LLM sentiment analysis

**What Changed**:
- ❌ Before: Simple word lists (POSITIVE_WORDS, NEGATIVE_WORDS)
- ✅ After: Contextual LLM sentiment analysis

## Usage Examples

### 1. Code Synthesis with LLM

```typescript
import { CodeEmergenceEngine } from './glass/emergence';
import { GlassOrganism } from './glass/types';

// Create organism
const organism: GlassOrganism = /* ... */;

// Create emergence engine with $0.50 budget
const engine = new CodeEmergenceEngine(organism, 0.5);

// Get emergence candidates from patterns
const candidates = /* ... */;

// Emerge code using LLM
const results = await engine.emerge(candidates);

// Results contain:
results.forEach(result => {
  console.log(result.function.name);
  console.log(result.function.implementation); // .gl code
  console.log(result.validation_passed); // Constitutional check
});

// Check cost
const stats = engine.getCostStats();
console.log(`Cost: $${stats.total_cost}`);
```

### 2. Pattern Detection with LLM

```typescript
import { PatternDetectionEngine } from './glass/patterns';

const organism: GlassOrganism = /* ... */;

// Enable LLM with $0.30 budget
const engine = new PatternDetectionEngine(organism, true, 0.3);

// Analyze patterns semantically
const analysis = await engine.analyzeWithLLM();

console.log(`Patterns: ${analysis.enhanced_patterns.length}`);
console.log(`Correlations: ${analysis.correlations.length}`);
console.log(`Emergence candidates: ${analysis.emergence_candidates.length}`);

// Correlations are semantic, not keyword-based!
analysis.correlations.forEach(corr => {
  console.log(`${corr.pattern_a} ↔ ${corr.pattern_b}: ${corr.strength}`);
});
```

### 3. Intent Analysis with LLM

```typescript
import { parsePragmaticsWithLLM } from './cognitive/parser/pragmatics';
import { createGlassLLM } from './glass/llm-adapter';

const text = "You're remembering that wrong. I never said that.";

// Parse linguistic features
const morphemes = parseMorphemes(text);
const syntax = parseSyntax(text);
const semantics = parseSemantics(text);

// Analyze intent with LLM
const llm = createGlassLLM('cognitive', 0.2);
const result = await parsePragmaticsWithLLM(
  text,
  morphemes,
  syntax,
  semantics,
  llm
);

console.log(result.pragmatics.intent); // 'manipulate'
console.log(result.pragmatics.power_dynamic); // 'exploit' | 'reverse'
console.log(result.confidence); // 0.85
console.log(result.reasoning); // ["Step 1: ...", "Step 2: ..."]
```

### 4. Semantic Embeddings with LLM

```typescript
import { GlassIngestion } from './glass/ingestion';

const organism: GlassOrganism = /* ... */;

// Create ingestion with $0.10 budget
const ingestion = new GlassIngestion(organism, 0.1);

// Ingest with LLM embeddings
await ingestion.ingest({
  source: {
    type: 'text',
    text: 'Clinical trial shows improved survival rates...'
  }
});

// LLM extracts semantic features:
// - topics, domain, concepts, methodology, findings
// - Converts to 384-dim embeddings (deterministic, hash-based)

const stats = ingestion.getCostStats();
console.log(`Cost: $${stats.total_cost}`);
```

## Budget Tracking

All LLM operations track costs:

```typescript
// Each component has getCostStats()
const stats = component.getCostStats();

console.log(`Total: $${stats.total_cost}`);
console.log(`Budget: $${stats.max_budget}`);
console.log(`Remaining: $${stats.remaining_budget}`);
console.log(`Over budget: ${stats.over_budget}`);
```

**Budget per organism (nascimento → maturidade)**:
- Code synthesis: ~$0.50
- Embeddings: ~$0.10
- Pattern detection: ~$0.30
- Intent analysis: ~$0.20
- Sentiment analysis: ~$0.10
- **Total**: ~**$1.20**

## Task-Specific Model Selection

The `GlassLLM` adapter automatically selects the best model:

| Task | Model | Use Case |
|------|-------|----------|
| `'code-synthesis'` | Opus 4 | Complex .gl code generation |
| `'intent-analysis'` | Opus 4 | Deep pragmatic analysis |
| `'semantic-analysis'` | Opus 4/Sonnet 4.5 | Meaning extraction |
| `'pattern-detection'` | Sonnet 4.5 | Fast pattern correlations |
| `'sentiment-analysis'` | Sonnet 4.5 | Emotion detection |

## Constitutional Validation

Every LLM response is validated:

```typescript
const response = await llm.invoke(query, {
  task: 'code-synthesis',
  enable_constitutional: true // Default
});

// response.constitutional_check contains:
if (!response.constitutional_check.passed) {
  console.log('Violations:', response.constitutional_check.violations);
  // [
  //   {
  //     principle_id: 'epistemic_honesty',
  //     severity: 'high',
  //     message: 'Response claims certainty without evidence',
  //     suggested_action: 'Reject and retry with lower confidence'
  //   }
  // ]
}
```

## Fallback Mechanisms

All LLM integrations have fallbacks:

```typescript
// If LLM fails, system falls back to rule-based methods
try {
  const result = await llm.invoke(query);
  return result.text;
} catch (error) {
  console.warn('LLM failed, using fallback');
  return ruleBasedFallback(query);
}
```

**Fallbacks**:
- Code synthesis → Template-based generation
- Pattern detection → Keyword Jaccard similarity
- Intent analysis → Rule-based if-else trees
- Embeddings → Deterministic hash-based vectors

## Performance Guarantees

✅ **O(1) operations maintained**:
- Database put/get/delete: O(1)
- Constitutional validation: O(1)
- Hash lookups: O(1)
- Pattern threshold checks: O(1)

❌ **LLM calls are NOT O(1)**:
- LLM synthesis: O(n) where n = response tokens
- But LLM is used strategically (emergence, not every operation)

## Testing

Run E2E tests:

```bash
npm test tests/e2e-llm-integration.test.ts
```

Run performance benchmarks:

```bash
npm test tests/performance-benchmarks.test.ts
```

## Environment Setup

Required environment variable:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Or pass directly:

```typescript
import { createGlassLLM } from './glass/llm-adapter';

const llm = createGlassLLM('cognitive', 0.5, 'sk-ant-...');
```

## Migration Guide

### Before (Templates)

```typescript
// Old: Hardcoded template
function generateEfficacyFunction(name, params, returnType) {
  return `function ${name}(...) -> ...: /* hardcoded logic */`;
}
```

### After (LLM)

```typescript
// New: LLM synthesis
const llm = createGlassLLM('glass-core', 0.5);
const code = await llm.invoke(buildPrompt(candidate), {
  task: 'code-synthesis'
});
```

## Best Practices

1. **Always set budgets**: Prevent runaway costs
   ```typescript
   const llm = createGlassLLM('cognitive', 0.2); // $0.20 max
   ```

2. **Enable constitutional validation**: Catch hallucinations
   ```typescript
   const result = await llm.invoke(query, {
     enable_constitutional: true
   });
   ```

3. **Check cost after operations**:
   ```typescript
   const stats = llm.getCostStats();
   if (stats.over_budget) {
     console.warn('Budget exceeded!');
   }
   ```

4. **Use task-specific models**: Let adapter choose
   ```typescript
   // Opus 4 for complex reasoning
   llm.invoke(query, { task: 'code-synthesis' });

   // Sonnet 4.5 for fast analysis
   llm.invoke(query, { task: 'pattern-detection' });
   ```

5. **Implement fallbacks**: Never rely solely on LLM
   ```typescript
   try {
     return await llmAnalysis(text);
   } catch (error) {
     return ruleBasedAnalysis(text);
   }
   ```

## Troubleshooting

### LLM calls timing out

Increase timeout:
```typescript
const result = await llm.invoke(query, {
  task: 'code-synthesis',
  max_tokens: 3000 // Increase if needed
});
```

### Budget exceeded errors

Check and reset:
```typescript
const stats = llm.getCostStats();
console.log(`Used: $${stats.total_cost} / $${stats.max_budget}`);

// Create new instance with higher budget
const newLLM = createGlassLLM('cognitive', 1.0);
```

### Constitutional violations

Review the violation and adjust:
```typescript
if (!result.constitutional_check.passed) {
  const violation = result.constitutional_check.violations[0];
  console.log(violation.principle_id); // Which principle failed
  console.log(violation.suggested_action); // What to do
}
```

## Future Enhancements

- [ ] Response caching (avoid duplicate LLM calls)
- [ ] Batch processing for embeddings
- [ ] Streaming for long code synthesis
- [ ] Custom model selection per organism
- [ ] Multi-LLM support (OpenAI, Gemini, etc.)

## Support

For issues or questions:
- Check `/docs/ARCHITECTURE.md`
- Review test files in `/tests/`
- See constitutional docs in `/spec/constitutional-embedding.md`
