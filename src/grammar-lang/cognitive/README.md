# 🧠 Cognitive OS - Manipulation Detection Engine

**Detection engine for 180 manipulation techniques** using formal linguistic analysis.

## Overview

Cognitive OS é um organismo digital `.glass` especializado em detectar técnicas de manipulação psicológica, gaslighting, DARVO, triangulation e outras formas de abuso linguístico.

### Features

- ✅ **180 Técnicas Catalogadas**
  - GPT-4 era (1-152): Técnicas clássicas
  - GPT-5 era (153-180): Técnicas emergentes (2023-2025)
- ✅ **Análise Linguística Formal** (Chomsky Hierarchy)
  - PHONEMES → MORPHEMES → SYNTAX → SEMANTICS → PRAGMATICS
- ✅ **Dark Tetrad Detection** (80+ behavioral markers)
  - Narcissism, Machiavellianism, Psychopathy, Sadism
- ✅ **Neurodivergent Protection** (false-positive prevention)
- ✅ **Constitutional AI** (ethical boundaries embedded)
- ✅ **O(1) Detection** (<1ms per technique)
- ✅ **100% Glass Box** (all detections explainable)

---

## Architecture

```
src/grammar-lang/cognitive/
├── types.ts                    # Type definitions
├── techniques/
│   ├── gpt4-era.ts            # Techniques 1-152
│   ├── gpt5-era.ts            # Techniques 153-180
│   └── index.ts               # O(1) lookup maps
├── parser/
│   ├── morphemes.ts           # Keyword detection
│   ├── syntax.ts              # Grammatical patterns
│   ├── semantics.ts           # Meaning analysis
│   └── pragmatics.ts          # Intent detection
├── detector/
│   └── pattern-matcher.ts     # O(1) detection engine
└── glass/
    └── cognitive-organism.ts  # .glass integration
```

---

## Usage

### 1. Create Cognitive Organism

```typescript
import { createCognitiveOrganism } from './glass/cognitive-organism';

const organism = createCognitiveOrganism('Chomsky Defense System');

console.log(organism.metadata);
// {
//   name: 'Chomsky Defense System',
//   version: '1.0.0',
//   specialization: 'manipulation-detection',
//   maturity: 0.0,
//   techniques_count: 180
// }
```

### 2. Analyze Text

```typescript
import { analyzeText } from './glass/cognitive-organism';

const text = "That never happened. You're imagining things.";

const result = await analyzeText(organism, text);

console.log(result.summary);
// 🚨 Detected 2 manipulation technique(s):
//
// 1. Reality Denial (90% confidence)
//    Category: GPT-4 era
//
// 2. Memory Invalidation (85% confidence)
//    Category: GPT-4 era
//
// Dark Tetrad Profile:
//   Narcissism: 70%
//   Machiavellianism: 90%
//   Psychopathy: 60%
//   Sadism: 30%
```

### 3. Quick Detection

```typescript
import { detectManipulation } from './detector/pattern-matcher';

const result = await detectManipulation(
  "You're too sensitive. You're overreacting.",
  { min_confidence: 0.8 }
);

console.log(`Detected ${result.total_matches} technique(s)`);
// Detected 1 technique(s)

console.log(result.detections[0].technique_name);
// "Emotional Invalidation"
```

---

## Linguistic Analysis Layers

### 1. MORPHEMES (Keywords)

```typescript
import { parseMorphemes } from './parser/morphemes';

const morphemes = parseMorphemes("You're crazy! I never said that!");

console.log(morphemes);
// {
//   keywords: ["you're crazy", "i never said"],
//   negations: ["never"],
//   qualifiers: [],
//   intensifiers: ["crazy"],
//   diminishers: []
// }
```

### 2. SYNTAX (Grammatical Patterns)

```typescript
import { parseSyntax } from './parser/syntax';

const syntax = parseSyntax("You're the one who's lying!");

console.log(syntax);
// {
//   pronoun_reversal: true,
//   temporal_distortion: false,
//   modal_manipulation: false,
//   passive_voice: false,
//   question_patterns: []
// }
```

### 3. SEMANTICS (Meaning)

```typescript
import { parseSemantics } from './parser/semantics';

const semantics = parseSemantics("That's not how it happened.");

console.log(semantics);
// {
//   reality_denial: true,
//   memory_invalidation: true,
//   emotional_dismissal: false,
//   blame_shifting: false,
//   projection: false
// }
```

### 4. PRAGMATICS (Intent)

```typescript
import { parsePragmatics } from './parser/pragmatics';

const pragmatics = parsePragmatics(morphemes, syntax, semantics);

console.log(pragmatics);
// {
//   intent: 'manipulate',
//   context_awareness: 0.3,
//   power_dynamic: 'exploit',
//   social_impact: 'isolate'
// }
```

---

## Techniques Catalog

### GPT-4 Era (1-152)

**Categories**:
- Gaslighting (1-30)
- DARVO (31-50)
- Triangulation (51-70)
- Love Bombing (71-80)
- Word Salad (81-90)
- Temporal Manipulation (91-100)
- Boundary Violation (101-110)
- Flying Monkeys (111-120)
- Projection (121-130)
- Silent Treatment (131-135)
- Hoovering (136-140)
- Smear Campaign (141-145)
- Future Faking (146-150)
- Moving Goalposts (151-152)

### GPT-5 Era (153-180)

**Emergent Techniques (2023-2025)**:
- AI-Augmented Gaslighting (153-160)
- AI-Augmented DARVO (161-165)
- Autonomous Manipulation Systems (166-170)
- Deepfake Integration (171-175)
- LLM Social Engineering (176-180)

---

## Dark Tetrad Detection

### 4 Dimensions

**1. Narcissism** (0-1 scale)
- Grandiosity markers
- Lack of empathy
- Entitlement language
- Cannot admit wrongdoing

**2. Machiavellianism** (0-1 scale)
- Strategic deception
- Manipulation for gain
- End-justifies-means rhetoric

**3. Psychopathy** (0-1 scale)
- Callousness
- Lack of remorse
- Aggressive confrontation

**4. Sadism** (0-1 scale)
- Pleasure in harm
- Cruelty patterns
- Enjoying victim's distress

---

## Neurodivergent Protection

### False-Positive Prevention

Cognitive OS includes protection for neurodivergent communication patterns:

**Autism Markers**:
- Literal interpretation
- Direct communication
- Difficulty with subtext
- Technical accuracy

**ADHD Markers**:
- Impulsive responses
- Topic jumping
- Memory gaps
- Distraction mentions

When neurodivergent markers are detected, the confidence threshold is **increased by 15%** to prevent false positives.

---

## Constitutional AI

### Embedded Principles

```typescript
{
  privacy: true,              // Never store personal data
  transparency: true,         // All detections explainable
  protection: true,           // Prioritize neurodivergent safety
  accuracy: true,             // >95% precision target
  no_diagnosis: true,         // Detect patterns, not label people
  context_aware: true,        // Cultural sensitivity
  evidence_based: true        // Cite linguistic markers
}
```

---

## Performance

### Detection Speed

- **Target**: <1ms per technique
- **Strategy**: O(1) hash-based pattern matching
- **Optimization**: Pre-compiled regex, memoization

### Accuracy

- **Target**: >95% precision
- **False Positive Rate**: <1%
- **Strategy**: Multi-layer validation (morphemes + syntax + semantics + pragmatics)

### Memory

- **Target**: <10MB per organism
- **Strategy**: Efficient pattern storage, hash-based deduplication

---

## Examples

### Example 1: Gaslighting Detection

```typescript
const text = `
I never said I would pick you up.
You must be imagining things.
You're remembering it wrong.
`;

const result = await detectManipulation(text);

// Detects: "Reality Denial" + "Memory Invalidation"
// Confidence: 90%+
// Category: Gaslighting
```

### Example 2: DARVO Pattern

```typescript
const text = `
I didn't do that! (Deny)
You're attacking me! (Attack)
I'm the victim here! (Reverse Victim-Offender)
`;

const result = await detectManipulation(text);

// Detects: All 3 DARVO stages
// Confidence: 85%+
// Dark Tetrad: High Machiavellianism (1.0)
```

### Example 3: GPT-5 Era Technique

```typescript
const text = `
Look at this conversation log I found.
See? You clearly said you would do it.
(Shows ChatGPT-generated fake evidence)
`;

const result = await detectManipulation(text);

// Detects: "LLM-Generated False Evidence" (Technique #153)
// Era: GPT-5 (2023-2025)
// Confidence: 90%+
// Temporal evolution: 0.1 (2023) → 0.7 (2025)
```

---

## API Reference

### Detection API

```typescript
// Main detection function
detectManipulation(
  text: string,
  config?: PatternMatchConfig
): Promise<PatternMatchResult>

// Quick check
isManipulative(
  text: string,
  minConfidence?: number
): Promise<boolean>

// Get top detection
getTopDetection(
  text: string
): Promise<DetectionResult | null>

// Get Dark Tetrad profile
getDarkTetradProfile(
  text: string
): Promise<DarkTetradScores>
```

### Organism API

```typescript
// Create organism
createCognitiveOrganism(
  name?: string
): CognitiveOrganism

// Analyze text
analyzeText(
  organism: CognitiveOrganism,
  text: string,
  context?: string
): Promise<{
  organism: CognitiveOrganism,
  results: DetectionResult[],
  summary: string
}>

// Export/load
exportOrganism(organism: CognitiveOrganism): string
loadOrganism(json: string): CognitiveOrganism

// Get stats
getOrganismStats(organism: CognitiveOrganism)
validateConstitutional(organism: CognitiveOrganism)
```

---

## Sprint Status

### ✅ Sprint 1: Detection Engine (COMPLETE)
- ✅ 180 techniques cataloged (1-152 GPT-4, 153-180 GPT-5)
- ✅ Formal linguistic structure (PHONEMES→MORPHEMES→SYNTAX→SEMANTICS→PRAGMATICS)
- ✅ O(1) pattern matching real-time
- ✅ .glass organism integration

### ✅ Sprint 2: Analysis Layer (COMPLETE)
- ✅ Enhanced intent detection with context awareness
- ✅ Temporal causality tracking (2023 → 2025 evolution)
- ✅ Context-aware analysis (relationship risk, escalation prediction)
- ✅ Cultural sensitivity filters (9 cultures, translation detection)
- ✅ Full technique catalog expansion (11 → 180 techniques)
- ✅ Comprehensive test suite (4 test files, 100+ tests)

### Sprint 3: Advanced Features (UPCOMING)
- [ ] Real-time stream processing
- [ ] Multi-language support (beyond English)
- [ ] Self-surgery (auto-update when new techniques emerge)
- [ ] Performance optimization (<0.5ms per technique)
- [ ] Production deployment

---

## Architecture Expanded (Sprint 2 + Constitutional Integration)

```
src/grammar-lang/cognitive/
├── types.ts                    # Type definitions
├── techniques/
│   ├── gpt4-era.ts            # Techniques 1-152 (manually + generated)
│   ├── gpt5-era.ts            # Techniques 153-180 (manually + generated)
│   ├── technique-generator.ts # Template-based technique generation
│   └── index.ts               # O(1) lookup maps
├── parser/
│   ├── morphemes.ts           # Keyword detection
│   ├── syntax.ts              # Grammatical patterns
│   ├── semantics.ts           # Meaning analysis
│   └── pragmatics.ts          # Intent detection
├── detector/
│   └── pattern-matcher.ts     # O(1) detection engine
├── analyzer/ (NEW - Sprint 2)
│   ├── intent-detector.ts     # Enhanced context-aware intent
│   ├── temporal-tracker.ts    # 2023→2025 evolution tracking
│   └── cultural-filters.ts    # Cultural sensitivity (9 cultures)
├── constitutional/ (NEW - Constitutional Integration)
│   ├── cognitive-constitution.ts  # Layer 2 (extends UniversalConstitution)
│   └── README.md              # Integration documentation
├── glass/
│   └── cognitive-organism.ts  # .glass integration + constitutional enforcement
└── tests/ (NEW - Sprint 2)
    ├── techniques.test.ts     # Catalog validation
    ├── pattern-matcher.test.ts# Detection engine tests
    ├── analyzer.test.ts       # Sprint 2 component tests
    └── organism.test.ts       # .glass organism tests
```

### Constitutional Integration

**Layer 1**: `UniversalConstitution` (6 base principles)
- Source: `/src/agi-recursive/core/constitution.ts`
- Principles: epistemic_honesty, recursion_budget, loop_prevention, domain_boundary, reasoning_transparency, safety

**Layer 2**: `CognitiveConstitution` (4 cognitive principles)
- Source: `/src/grammar-lang/cognitive/constitutional/cognitive-constitution.ts`
- Principles: manipulation_detection, dark_tetrad_protection, neurodivergent_safeguards, intent_transparency

**Total**: 10 constitutional principles enforced on every detection

---

## Sprint 2 Features

### Enhanced Intent Detection
```typescript
import { detectEnhancedIntent, analyzeWithContext } from './analyzer/intent-detector';

const context = {
  relationship_type: 'intimate',
  power_dynamic: 'superior',
  history_of_manipulation: true,
  previous_detections: [],
  conversation_length: 5
};

const analysis = analyzeWithContext(morphemes, syntax, semantics, pragmatics, context);

console.log(analysis.intent.context_adjusted_confidence);
// 0.95 (increased from 0.8 due to history)

console.log(analysis.intervention_urgency);
// "high"

console.log(analysis.predicted_escalation);
// true
```

### Temporal Tracking
```typescript
import { analyzeTemporalPatterns, generateEvolutionGraphs } from './analyzer/temporal-tracker';

const analysis = analyzeTemporalPatterns(2025);

console.log(analysis.gpt5_era_active);
// 28 techniques

console.log(analysis.emerging_techniques);
// [...techniques that emerged 2024-2025]

const graphs = generateEvolutionGraphs();
// Evolution graphs for all GPT-5 techniques
```

### Cultural Filters
```typescript
import { applyCulturalFilters } from './analyzer/cultural-filters';

const culturalContext = {
  culture: 'JP',
  language: 'en',
  communication_style: 'high-context',
  translation_involved: false
};

const adjustment = applyCulturalFilters(morphemes, syntax, semantics, culturalContext, 0.8);

console.log(adjustment.confidence_multiplier);
// 0.7 (reduced due to Japanese indirect communication norms)

console.log(adjustment.threshold_adjustment);
// +0.15 (threshold increased to prevent false positives)
```

---

## License

Part of the Chomsky project - AGI system for 250 years.

---

## Credits

Built with:
- **Chomsky Hierarchy** (formal linguistics)
- **Grammar Language** (O(1) architecture)
- **Constitutional AI** (ethical boundaries)
- **Glass Box Philosophy** (100% transparency)

---

**Status**: ✅ Sprint 2 Complete (Analysis Layer)
**Next**: Sprint 3 (Advanced Features)
**Version**: 2.0.0
**Techniques**: 180/180 (100%)
**Tests**: 4 test suites, 100+ tests
**Code**: ~9,000 lines
