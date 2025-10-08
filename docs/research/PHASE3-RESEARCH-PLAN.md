# Phase 3: Mitigation Strategies Research Plan

**Status:** 🚧 IN PROGRESS
**Start Date:** 2025-10-08
**Goal:** Design and validate architectural modifications to reduce hallucination risk

---

## Overview

Phase 2A identified **extreme FFN dominance (22,441×)** as the primary hallucination driver in layers 28-30. Phase 3 will design, simulate, and validate mitigation strategies using the weight profile data.

---

## Problem Statement

### Current Architecture Issues

1. **FFN Dominance:** Late layers show 20,000-27,000× FFN/Attention ratio
2. **Value Sparsity:** Bimodal pattern creates information bottlenecks
3. **Norm Amplification:** FFN norms 30-36% above average in late layers
4. **Layers 28-30:** Peak hallucination risk (30-34%)

### Target Goals

- Reduce FFN dominance to < 1,000× in late layers
- Decrease hallucination risk from 33% to < 15%
- Maintain model performance (perplexity, accuracy)
- Validate via weight-based simulation

---

## Proposed Mitigation Strategies

### Strategy 1: Attention Amplification 🎯 **[PRIORITY 1]**

**Concept:** Boost attention mechanism strength in late layers

**Implementation:**
```typescript
// Multiply attention weights by amplification factor
amplifiedAttention = baseAttention * amplificationFactor

// Adaptive factor based on layer depth
amplificationFactor = 1 + (layer / 32) * maxBoost
```

**Parameters:**
- `maxBoost`: 2.0 - 5.0 (2× to 5× amplification)
- `targetLayers`: 24-31 (late layers)

**Expected Impact:**
- Reduce FFN dominance from 22,441× to ~5,000×
- Increase attention contribution to final output
- Improve context grounding

**Validation:**
- Recalculate hallucination risk scores
- Measure attention/FFN balance
- Compare before/after dominance ratios

---

### Strategy 2: FFN Regularization 🎯 **[PRIORITY 2]**

**Concept:** Constrain FFN strength via weight scaling

**Implementation:**
```typescript
// Scale down FFN weights in late layers
regularizedFFN = {
  gate: gate.weights * gateScale,
  up: up.weights * upScale,
  down: down.weights * downScale
}

// Layer-dependent scaling
scale = 1 - (layer / 32) * maxReduction
```

**Parameters:**
- `maxReduction`: 0.3 - 0.7 (30% to 70% reduction)
- `targetLayers`: 28-31 (peak risk layers)

**Expected Impact:**
- Reduce FFN strength by 50-70%
- Lower FFN dominance to ~3,000×
- Decrease hallucination risk to ~15%

**Validation:**
- Recalculate FFN activation predictions
- Measure impact on global statistics
- Verify risk reduction

---

### Strategy 3: Norm Clipping ⚡ **[PRIORITY 3]**

**Concept:** Limit layer norm amplification in late layers

**Implementation:**
```typescript
// Clip FFN norms to maximum threshold
clippedNorm = Math.min(ffnNorm, maxNormValue)

// Adaptive threshold based on global average
maxNormValue = globalAvgNorm * clipRatio
```

**Parameters:**
- `clipRatio`: 1.0 - 1.2 (0% to 20% above average)
- `targetLayers`: 21-31 (all late layers)

**Expected Impact:**
- Reduce norm amplification from 36% to ~10%
- Prevent activation explosion
- Stabilize late-layer outputs

**Validation:**
- Recalculate norm amplification metrics
- Verify hypothesis 5 improvement
- Measure risk component reduction

---

### Strategy 4: Value Sparsity Equalization 📊 **[OPTIONAL]**

**Concept:** Normalize sparsity patterns across layers

**Implementation:**
```typescript
// Target uniform sparsity across all layers
targetSparsity = globalAvgSparsity

// Adjust weights to achieve target sparsity
adjustedWeights = adjustSparsity(weights, targetSparsity)
```

**Parameters:**
- `targetSparsity`: 1.5% (global average)
- `tolerance`: ±0.5%

**Expected Impact:**
- Eliminate bimodal sparsity pattern
- Reduce information loss
- Improve information flow

**Validation:**
- Recalculate sparsity distribution
- Verify hypothesis 1 mitigation
- Measure risk reduction

---

### Strategy 5: Hybrid Architecture 🏗️ **[ADVANCED]**

**Concept:** Replace peak risk layers (28-30) with attention-heavy variants

**Implementation:**
```typescript
// Replace FFN-dominant layers with multi-head attention
layer28 = MultiHeadAttention({ heads: 32, dim: 4096 })
layer29 = MultiHeadAttention({ heads: 32, dim: 4096 })
layer30 = MultiHeadAttention({ heads: 32, dim: 4096 })

// Keep FFN but reduce to auxiliary role
auxiliaryFFN = FFN({ reduction: 0.8 })
```

**Expected Impact:**
- Eliminate FFN dominance in peak layers
- Restore attention-based processing
- Radical risk reduction

**Validation:**
- Full architecture simulation
- Perplexity evaluation
- Downstream task benchmarks

---

## Implementation Roadmap

### Phase 3.1: Single Strategy Validation (Current Phase)

**Timeline:** 2-4 hours

**Tasks:**
1. ✅ Create research plan (this document)
2. 🔄 Implement Strategy 1 (Attention Amplification)
3. 🔄 Validate via weight-based simulation
4. 🔄 Document results
5. 🔄 Commit findings

**Deliverables:**
- `phase3-attention-amplification.ts` - Implementation
- `phase3-results-strategy1.json` - Validation results
- `PHASE3-STRATEGY1-REPORT.md` - Findings

### Phase 3.2: Multi-Strategy Comparison

**Timeline:** 4-6 hours

**Tasks:**
1. Implement Strategy 2 (FFN Regularization)
2. Implement Strategy 3 (Norm Clipping)
3. Run comparative analysis
4. Identify best single strategy
5. Test combined strategies

**Deliverables:**
- `phase3-comparison-analysis.ts`
- `phase3-strategy-comparison.json`
- `PHASE3-COMPARISON-REPORT.md`

### Phase 3.3: Hybrid Architecture (Optional)

**Timeline:** 8-12 hours

**Tasks:**
1. Design hybrid layer architecture
2. Implement layer replacement simulation
3. Validate architectural coherence
4. Document architectural changes

**Deliverables:**
- `phase3-hybrid-architecture.ts`
- `PHASE3-HYBRID-ARCHITECTURE-SPEC.md`

---

## Validation Methodology

### Weight-Based Simulation

For each strategy:

1. **Load baseline weights** from Phase 1 profile
2. **Apply mitigation** (scale, amplify, clip, etc.)
3. **Recalculate statistics**:
   - Attention strength
   - FFN strength
   - Dominance ratios
   - Global averages
4. **Recompute hallucination risks** for all 32 layers
5. **Compare before/after**:
   - Risk reduction per layer
   - Global risk average
   - Highest risk layer shift

### Success Criteria

A mitigation strategy is **successful** if:

- ✅ Reduces peak risk from 33.6% to < 15%
- ✅ Reduces FFN dominance from 22,441× to < 1,000×
- ✅ Maintains architectural coherence (no NaN/Inf values)
- ✅ Shows consistent improvement across layers 24-31

A strategy is **breakthrough** if:

- 🏆 Reduces peak risk to < 10%
- 🏆 Achieves FFN dominance < 100×
- 🏆 Improves all 5 risk components

---

## Technical Implementation Details

### Base Framework

```typescript
interface MitigationStrategy {
  name: string;
  description: string;
  apply(profile: LayerProfile[]): LayerProfile[];
  parameters: Record<string, number>;
}

class MitigationSimulator {
  baseline: LayerProfile[];

  applyStrategy(strategy: MitigationStrategy): SimulationResult {
    const modified = strategy.apply(this.baseline);
    const risks = this.calculateRisks(modified);
    const improvement = this.compareWithBaseline(risks);
    return { modified, risks, improvement };
  }

  compareStrategies(strategies: MitigationStrategy[]): ComparisonReport {
    const results = strategies.map(s => this.applyStrategy(s));
    return this.generateComparison(results);
  }
}
```

### Strategy Implementation Template

```typescript
const attentionAmplification: MitigationStrategy = {
  name: 'Attention Amplification',
  description: 'Boost attention in late layers',
  parameters: {
    maxBoost: 3.0,
    startLayer: 24,
    endLayer: 31,
  },

  apply(layers: LayerProfile[]): LayerProfile[] {
    return layers.map((layer, idx) => {
      if (idx < this.parameters.startLayer) return layer;

      const boost = 1 + ((idx - this.parameters.startLayer) /
                        (this.parameters.endLayer - this.parameters.startLayer)) *
                        this.parameters.maxBoost;

      return {
        ...layer,
        attention: this.amplifyAttention(layer.attention, boost),
      };
    });
  },
};
```

---

## Expected Outcomes

### Strategy 1: Attention Amplification

**Before:**
- Layer 30 risk: 33.6%
- FFN dominance: 22,441×
- Attention strength: 53.4

**After (predicted):**
- Layer 30 risk: ~18%
- FFN dominance: ~5,600×
- Attention strength: ~160

**Risk reduction:** ~45%

### Strategy 2: FFN Regularization

**Before:**
- Layer 30 FFN strength: 1,198,077
- FFN dominance: 22,441×

**After (predicted):**
- Layer 30 FFN strength: ~360,000
- FFN dominance: ~6,700×

**Risk reduction:** ~55%

### Combined Strategies (Best Case)

**Attention Amplification + FFN Regularization:**
- Amplify attention by 3×
- Reduce FFN by 60%
- **Predicted dominance:** ~900× (96% improvement)
- **Predicted risk:** ~8% (76% reduction)

---

## Next Steps

1. **Immediate:** Implement Strategy 1 (Attention Amplification)
2. **Short-term:** Validate and compare Strategies 1-3
3. **Medium-term:** Test combined strategies
4. **Long-term:** Design hybrid architecture spec

---

**Status:** 🚧 Phase 3.1 in progress
**Current Task:** Implementing Attention Amplification
**ETA:** 2-4 hours
