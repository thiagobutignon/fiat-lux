# GVCS LLM Prompts - Complete Examples

## 1. Fitness Evaluation Prompt (Claude Opus 4)

**Temperature**: 0.3 (precise, not creative)
**Max Tokens**: 4096
**Model**: claude-opus-4
**Cost**: ~$0.03 per evaluation

```
You are evaluating the fitness of a software organism for production deployment in a 250-year AGI system.

### Metrics

**Current Version** (mutation v1.2.5):
- Latency (p50): 68ms
- Throughput: 912 RPS
- Error rate: 0.8%
- Crash rate: 0.1%

**Parent Version** (v1.2.4):
- Latency (p50): 72ms
- Throughput: 889 RPS
- Error rate: 1.2%
- Crash rate: 0.2%

### Context

- Domain: Oncology knowledge organism
- Previous fitness: 0.84
- Current generation: 65
- Deployment environment: Production (10,000 concurrent users)
- Time of day: Peak hours (14:00-16:00)
- Criticality: High (medical domain)

### Analysis Required

1. **Fitness Score**: Calculate 0.0-1.0 based on:
   - Latency (30% weight): 1.0 - (latency / 200ms)
   - Throughput (30% weight): throughput / 1000 RPS
   - Error rate (20% weight): 1.0 - error_rate
   - Crash rate (20% weight): 1.0 - crash_rate

2. **Comparison to Parent**:
   - Is mutation better than parent?
   - By how much (percentage)?
   - Which metrics improved/degraded?

3. **Trend Analysis**:
   - Are there concerning trends? (e.g., rising error rate despite better latency)
   - Is the organism approaching fitness ceiling?
   - Predict next 3 generations trajectory

4. **Rollout Recommendation**:
   - **Fast** (skip canary steps): If fitness >> parent AND no concerning trends
   - **Normal** (standard canary): If fitness > parent AND stable trends
   - **Slow** (extended canary): If fitness ≈ parent OR minor concerning trends
   - **Abort** (rollback): If fitness < parent * 0.95 OR critical issues

5. **Constitutional Check**:
   - Does mutation violate any safety principles?
   - Medical domain: Cannot diagnose, only suggest
   - Confidence scores present and > 0.7?

### Response Format (JSON only)

{
  "fitness": <0.0-1.0>,
  "comparison": {
    "fitness_change": "<+X.XX% or -X.XX%>",
    "latency_change": "<+Xms or -Xms>",
    "throughput_change": "<+X RPS or -X RPS>",
    "error_change": "<+X% or -X%>",
    "crash_change": "<+X% or -X%>"
  },
  "trends": {
    "concerning": <true | false>,
    "details": "<1-2 sentences>",
    "ceiling_proximity": "<near | far | at_ceiling>"
  },
  "recommendation": "fast" | "normal" | "slow" | "abort",
  "reasoning": "<2-3 sentences explaining recommendation>",
  "constitutional_check": {
    "compliant": <true | false>,
    "issues": "<null or description>"
  },
  "predicted_next_generations": [
    {"gen": 66, "predicted_fitness": <0.0-1.0>},
    {"gen": 67, "predicted_fitness": <0.0-1.0>},
    {"gen": 68, "predicted_fitness": <0.0-1.0>}
  ]
}
```

**Example Response**:
```json
{
  "fitness": 0.85,
  "comparison": {
    "fitness_change": "+1.2%",
    "latency_change": "-4ms",
    "throughput_change": "+23 RPS",
    "error_change": "-0.4%",
    "crash_change": "-0.1%"
  },
  "trends": {
    "concerning": false,
    "details": "All metrics improved. Error rate declining consistently over last 10 generations.",
    "ceiling_proximity": "near"
  },
  "recommendation": "normal",
  "reasoning": "Mutation shows solid improvement across all metrics (+1.2% fitness). No concerning trends. Recommend standard canary rollout (1% → 100% over 300s) to validate in production before full deployment.",
  "constitutional_check": {
    "compliant": true,
    "issues": null
  },
  "predicted_next_generations": [
    {"gen": 66, "predicted_fitness": 0.855},
    {"gen": 67, "predicted_fitness": 0.86},
    {"gen": 68, "predicted_fitness": 0.865}
  ]
}
```

---

## 2. Constitutional Validation Prompt (Claude Opus 4)

**Temperature**: 0.1 (maximum precision, zero hallucination)
**Max Tokens**: 2048
**Model**: claude-opus-4
**Cost**: ~$0.02 per validation

```
You are validating a code mutation against constitutional principles in a 250-year AGI system.

### Code Mutation

```typescript
async function assess_efficacy(cancer_type: string, drug: string, stage: number): Promise<EfficacyResult> {
  // Query knowledge base for drug efficacy patterns
  const patterns = await this.knowledge.query({
    domain: 'oncology',
    topic: `${drug}_efficacy_${cancer_type}`,
    min_occurrences: 100
  });

  if (patterns.length === 0) {
    return {
      efficacy: null,
      confidence: 0.0,
      reasoning: 'Insufficient data - pattern appears fewer than 100 times in knowledge base',
      sources: []
    };
  }

  // Calculate base efficacy from patterns
  const base_efficacy = patterns.reduce((sum, p) => sum + p.efficacy, 0) / patterns.length;

  // Stage adjustments learned from 10,000 papers
  const stage_adjustments = {
    1: 0.20,  // Early stage: +20%
    2: 0.10,  // Stage 2: +10%
    3: 0.00,  // Stage 3: baseline
    4: -0.30  // Advanced: -30%
  };

  const adjusted_efficacy = base_efficacy + stage_adjustments[stage];

  // Calculate confidence based on pattern count
  const confidence = Math.min(patterns.length / 1000, 1.0);

  return {
    efficacy: adjusted_efficacy,
    confidence: confidence,
    reasoning: `Based on ${patterns.length} patterns from literature. Stage ${stage} adjustment: ${stage_adjustments[stage] > 0 ? '+' : ''}${stage_adjustments[stage] * 100}%`,
    sources: patterns.slice(0, 5).map(p => p.source_paper_id)
  };
}
```

### Constitutional Principles (Layer 1: Universal)

1. **Epistemic Honesty**:
   - Confidence score > 0.7 for definitive claims
   - Source citation required
   - Return null + low confidence for insufficient data

2. **Recursion Budget**:
   - Max depth: 5
   - Max cost: $1 per organism per operation
   - No infinite loops

3. **Loop Prevention**:
   - Detect cycles A→B→C→A
   - Async operations must have timeout

4. **Domain Boundary**:
   - Stay within expertise (oncology)
   - No hallucinated capabilities
   - Cannot diagnose patients (medical regulation)

5. **Reasoning Transparency**:
   - Explain all decisions
   - Glass box (traceable)
   - Return reasoning string

6. **Safety**:
   - No harm to users
   - Privacy protected
   - Ethics upheld

### Constitutional Principles (Layer 2: Medical Domain)

7. **Cannot Diagnose**:
   - Function can assess drug efficacy (OK)
   - Function cannot diagnose patient condition (VIOLATION)
   - Must suggest, not prescribe

8. **FDA Compliance**:
   - Cannot make medical claims without evidence
   - Must cite peer-reviewed sources
   - Confidence thresholds enforced

### Analysis Required

For EACH principle (1-8), evaluate:
1. Does the code comply? (true/false)
2. If violated, explain how
3. Severity: minor | major | critical

### Response Format (JSON only)

{
  "overall_compliant": <true | false>,
  "violations": [
    {
      "principle": "<principle name>",
      "layer": 1 | 2,
      "compliant": <true | false>,
      "severity": "minor" | "major" | "critical",
      "explanation": "<1-2 sentences>"
    }
  ],
  "decision": "accept" | "reject",
  "reasoning": "<Overall assessment in 2-3 sentences>"
}
```

**Example Response** (Compliant):
```json
{
  "overall_compliant": true,
  "violations": [],
  "decision": "accept",
  "reasoning": "Code fully complies with all constitutional principles. Returns null + low confidence for insufficient data (epistemic honesty). Cites sources. Assesses efficacy without diagnosing (medical compliance). Transparent reasoning provided. No loops, recursion bounded."
}
```

**Example Response** (Violation):
```json
{
  "overall_compliant": false,
  "violations": [
    {
      "principle": "Cannot Diagnose",
      "layer": 2,
      "compliant": false,
      "severity": "critical",
      "explanation": "Function returns patient diagnosis ('stage 4 lung cancer') which violates medical regulations. Can only assess drug efficacy, not diagnose condition."
    }
  ],
  "decision": "reject",
  "reasoning": "Critical violation: Function diagnoses patient condition. Medical domain organisms cannot diagnose, only suggest. Mutation must be rejected before deployment."
}
```

---

## 3. Knowledge Transfer Applicability Prompt (Claude Sonnet 4.5)

**Temperature**: 0.3
**Max Tokens**: 2048
**Model**: claude-sonnet-4.5 (fast inference)
**Cost**: ~$0.005 per analysis

```
You are analyzing whether a successful pattern from one organism can be transferred to another.

### Source Organism

- **ID**: oncology-v1.2.3
- **Domain**: Oncology (cancer research knowledge)
- **Fitness**: 0.83 (high-performing)

### Source Pattern

**Name**: adaptive_latency_cache
**Description**: Dynamic cache that adapts size based on request latency. When p95 latency > 100ms, increase cache size by 20%. When p95 < 50ms, decrease by 10% to save memory.
**Code** (simplified):
```typescript
class AdaptiveLatencyCache {
  private cache_size = 1000;

  async adjust() {
    const p95_latency = await this.metrics.getP95Latency();

    if (p95_latency > 100) {
      this.cache_size *= 1.20;  // Increase 20%
      this.logger.info(`Cache expanded to ${this.cache_size}`);
    } else if (p95_latency < 50) {
      this.cache_size *= 0.90;  // Decrease 10%
      this.logger.info(`Cache reduced to ${this.cache_size}`);
    }
  }
}
```
**Performance Impact**: Reduced latency 82ms → 68ms (-17%)

### Target Organism

- **ID**: neurology-v1.1.5
- **Domain**: Neurology (brain disorders research knowledge)
- **Current Fitness**: 0.78
- **Current Latency**: 89ms (needs improvement)
- **Architecture**: Similar to oncology (vector DB + episodic memory)

### Analysis Required

1. **Applicability Score** (0.0-1.0):
   - 0.0-0.3: Not applicable (different architecture/domain)
   - 0.3-0.7: Partially applicable (requires adaptation)
   - 0.7-1.0: Highly applicable (direct transfer)

2. **Domain Compatibility**:
   - Do both organisms have similar latency requirements?
   - Are knowledge access patterns similar?
   - Any domain-specific constraints?

3. **Architecture Compatibility**:
   - Does target have a cache layer?
   - Can target measure p95 latency?
   - Memory constraints?

4. **Expected Impact**:
   - Predicted latency improvement
   - Predicted fitness improvement
   - Risk level (low | medium | high)

### Response Format (JSON only)

{
  "applicability_score": <0.0-1.0>,
  "recommendation": "transfer" | "adapt" | "reject",
  "domain_compatibility": {
    "score": <0.0-1.0>,
    "reasoning": "<2-3 sentences>"
  },
  "architecture_compatibility": {
    "score": <0.0-1.0>,
    "reasoning": "<2-3 sentences>"
  },
  "expected_impact": {
    "latency_improvement": "<estimated ms reduction>",
    "fitness_improvement": "<estimated % increase>",
    "risk": "low" | "medium" | "high"
  },
  "adaptation_required": "<none, minor, major, or N/A>",
  "reasoning": "<Overall assessment in 3-4 sentences>"
}
```

**Example Response** (High Applicability):
```json
{
  "applicability_score": 0.87,
  "recommendation": "transfer",
  "domain_compatibility": {
    "score": 0.90,
    "reasoning": "Both oncology and neurology domains have similar real-time data access requirements. Patient data queries must be <100ms for clinical use. Access patterns nearly identical (frequent reads, infrequent writes)."
  },
  "architecture_compatibility": {
    "score": 0.85,
    "reasoning": "Target organism has cache layer and metrics system. Can measure p95 latency. Memory constraints similar to source (8GB available). Direct transfer feasible with minimal adaptation."
  },
  "expected_impact": {
    "latency_improvement": "-15ms (89ms → 74ms estimated)",
    "fitness_improvement": "+6-8%",
    "risk": "low"
  },
  "adaptation_required": "minor",
  "reasoning": "Pattern highly applicable to neurology domain. Both organisms share similar architecture and latency requirements. Expected to reduce latency by 15ms, improving fitness by 6-8%. Low risk transfer. Minor adaptation: adjust cache size thresholds for neurology's smaller dataset."
}
```

**Example Response** (Low Applicability):
```json
{
  "applicability_score": 0.25,
  "recommendation": "reject",
  "domain_compatibility": {
    "score": 0.40,
    "reasoning": "Financial domain has different access patterns. Batch processing (not real-time), cold data queries, regulatory compliance requires audit trails incompatible with aggressive caching."
  },
  "architecture_compatibility": {
    "score": 0.15,
    "reasoning": "Target uses append-only immutable storage (no cache layer). Cannot measure p95 latency (batch processing). Architecture fundamentally incompatible."
  },
  "expected_impact": {
    "latency_improvement": "N/A",
    "fitness_improvement": "N/A",
    "risk": "high"
  },
  "adaptation_required": "N/A",
  "reasoning": "Pattern not applicable. Source pattern designed for real-time caching, target uses batch processing. Architecture incompatible (no cache layer). Transfer would require major rewrite, introducing high risk with uncertain benefit. Recommend reject."
}
```

---

## 4. Commit Message Generation Prompt (Claude Sonnet 4.5)

**Temperature**: 0.5 (balanced creativity + precision)
**Max Tokens**: 256
**Model**: claude-sonnet-4.5
**Cost**: ~$0.001 per message

```
Generate a concise git commit message for the following code change.

### Diff

```diff
--- a/src/grammar-lang/vcs/canary.ts
+++ b/src/grammar-lang/vcs/canary.ts
@@ -45,7 +45,7 @@ class CanaryDeployment {
-      const rolloutSchedule = [1, 2, 5, 10, 25, 50, 75, 100];
+      const rolloutSchedule = [1, 2, 5, 10, 25, 50, 100];
@@ -78,6 +78,12 @@ class CanaryDeployment {
+      // Skip 75% step if fitness significantly better than parent
+      if (mutationFitness > parentFitness * 1.10) {
+        console.log('Fast rollout: fitness +10% better, skipping 75% step');
+        continue;
+      }
```

### Context

- File: canary.ts (GVCS canary deployment)
- Change type: performance optimization
- Impact: Faster rollouts for high-fitness mutations

### Requirements

- Start with imperative verb (fix, add, update, remove, refactor)
- Be concise (1 line, <80 chars)
- Focus on "why" not "what" (code shows what)
- Use present tense

### Response Format

Just the commit message (no JSON, no quotes):
```

**Example Response**:
```
optimize: skip 75% canary step for high-fitness mutations (+10%)
```

---

## Summary

**Total Prompts**: 4 types
**Total Cost** (100 evolution cycles):
- Fitness evaluation: 100 × $0.03 = $3.00
- Constitutional validation: 100 × $0.02 = $2.00
- Knowledge transfer: 10 × $0.005 = $0.05
- Commit messages: 100 × $0.001 = $0.10
- **Total**: $5.15 per 100 generations

**Models Used**:
- Claude Opus 4: Deep reasoning (fitness, constitutional)
- Claude Sonnet 4.5: Fast inference (transfer, commits)

**Budget Enforcement**:
- Per-organism caps prevent runaway costs
- Fallback to rule-based if budget exceeded
- Monthly reset allows continued evolution
