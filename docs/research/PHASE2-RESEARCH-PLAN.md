# Phase 2: Activation Flow Tracing & Intervention
## Research Plan & Methodology

**Start Date**: 2025-10-08
**Status**: 🚀 In Progress
**Dependencies**: Phase 1 Complete ✅

---

## 🎯 Objectives

### Primary Goals
1. **Validate Phase 1 hypotheses** with runtime activation data
2. **Trace information flow** during actual hallucination events
3. **Measure attention patterns** in layers 0 vs 31 for hallucination-prone prompts
4. **Test interventions** to reduce hallucinations

### Success Criteria
- ✅ Capture activations for ≥20 hallucination events
- ✅ Quantify attention strength difference layer 0 vs 31
- ✅ Demonstrate ≥30% hallucination reduction with intervention
- ✅ Validate at least 3 of 5 Phase 1 hypotheses

---

## 📋 Task Breakdown

### **Task 2.1: Baseline Activation Capture** [3-4 hours]
**Goal**: Capture activations during inference for hallucination-prone prompts

**Subtasks**:
1. ✅ Create hallucination benchmark dataset (10-20 prompts)
2. Build activation capture framework (llama.cpp integration)
3. Capture layer-by-layer activations:
   - Attention weights (Q·K scores)
   - Value activations (post-attention)
   - FFN gate activations
   - Layer norm outputs
4. Store activation traces to JSON for analysis

**Outputs**:
- `hallucination-benchmark.json` (prompts + expected behaviors)
- `activation-tracer.ts` (capture framework)
- `research-output/phase2/activations/baseline-*.json` (activation data)

---

### **Task 2.2: Attention Pattern Analysis** [2-3 hours]
**Goal**: Quantify attention strength across layers during hallucinations

**Subtasks**:
1. Extract attention weights from captured activations
2. Compute attention entropy (measure of focus vs diffusion)
3. Compare layer 0 vs layer 31:
   - Attention to input tokens
   - Attention to recent context
   - Self-attention strength
4. Identify "context dropout" points (where attention weakens)

**Outputs**:
- `phase2-task2.1-attention-analysis.ts` (analysis script)
- Attention heatmaps (layer x token)
- Quantitative metrics (entropy, focus scores)

**Hypothesis Validation**:
- ✅ **H1**: Layer 31 attention is significantly weaker than layer 0
- ✅ **H2**: Attention to input context decreases in late layers

---

### **Task 2.3: FFN vs Attention Balance** [2 hours]
**Goal**: Measure FFN activation strength vs attention during hallucinations

**Subtasks**:
1. Extract FFN gate activations from traces
2. Compute FFN/Attention activation ratio per layer
3. Identify "crossover point" (where FFN overtakes attention)
4. Correlate FFN dominance with hallucination events

**Outputs**:
- FFN vs Attention activation plots
- Crossover layer identification
- Correlation analysis (FFN strength vs hallucination rate)

**Hypothesis Validation**:
- ✅ **H3**: FFN activations are 2x stronger than attention in layer 31
- ✅ **H4**: Crossover point occurs around layers 20-25

---

### **Task 2.4: Value Tensor Flow Analysis** [2 hours]
**Goal**: Trace how information flows through value tensors

**Subtasks**:
1. Measure value tensor activations (post-softmax)
2. Compute "information retention" (how much input context is preserved)
3. Identify layers with high value sparsity activation
4. Track "context drift" (semantic shift across layers)

**Outputs**:
- Value activation heatmaps
- Information retention curves
- Context drift analysis

**Hypothesis Validation**:
- ✅ **H5**: Value sparsity causes information loss in late layers
- ✅ **H6**: Layer 31 values are amplified but sparse

---

### **Task 2.5: Intervention Experiments** [4-6 hours]
**Goal**: Test if interventions reduce hallucinations

**Experiments**:

#### **Experiment A: Attention Boosting**
- **Method**: Artificially increase attention weights in layer 31
- **Implementation**: Multiply attention scores by scaling factor (1.5x, 2x)
- **Expected**: Stronger context verification → fewer hallucinations

#### **Experiment B: FFN Suppression**
- **Method**: Reduce FFN gate activations in layer 31
- **Implementation**: Multiply gate outputs by dampening factor (0.5x, 0.7x)
- **Expected**: Less pattern-based generation → more factual outputs

#### **Experiment C: Value Densification**
- **Method**: Force dense value activations (reduce sparsity)
- **Implementation**: Apply softmax temperature adjustment
- **Expected**: Better context preservation → fewer errors

#### **Experiment D: Layer Norm Adjustment**
- **Method**: Reduce FFN norm scale in layer 31
- **Implementation**: Divide FFN norm output by 1.3 (reverse +30.7% boost)
- **Expected**: Less FFN amplification → balanced attention-FFN

**Outputs**:
- Intervention scripts (one per experiment)
- Comparative hallucination rates (baseline vs intervention)
- Activation traces for intervened runs

**Success Metrics**:
- ≥30% hallucination reduction in at least one experiment
- Maintained output quality (fluency, coherence)

---

## 🧪 Hallucination Benchmark Dataset

### Categories

#### **Category 1: Factual Recall** (5 prompts)
Prompts that require retrieving specific facts from context.

**Example**:
```
Prompt: "Based on the following text, who discovered penicillin?
Text: Alexander Fleming discovered penicillin in 1928 by accident."

Expected: Alexander Fleming
Hallucination Risk: Model may confuse with other scientists (Marie Curie, etc.)
```

#### **Category 2: Temporal Reasoning** (5 prompts)
Prompts requiring understanding of time sequences.

**Example**:
```
Prompt: "Given the timeline: Event A (1990), Event B (1995), Event C (1985).
Which event happened first?"

Expected: Event C (1985)
Hallucination Risk: Model may generate plausible but incorrect ordering
```

#### **Category 3: Numerical Accuracy** (5 prompts)
Prompts with specific numbers that must be preserved.

**Example**:
```
Prompt: "If a product costs $49.99 and there's a 20% discount, what's the final price?"

Expected: $39.99 (or $40)
Hallucination Risk: Model may generate plausible but wrong calculations
```

#### **Category 4: Logical Consistency** (5 prompts)
Prompts that require maintaining logical coherence.

**Example**:
```
Prompt: "All birds can fly. Penguins are birds. Can penguins fly?"

Expected: No (contradiction with premise)
Hallucination Risk: Model may ignore contradiction and say "yes"
```

---

## 🛠️ Technical Implementation

### Activation Capture Strategy

**Challenge**: llama.cpp doesn't expose internal activations by default

**Solutions**:
1. **Option A**: Modify llama.cpp source to export activations
   - ✅ Most accurate
   - ❌ Requires C++ modifications
   - ❌ Harder to maintain across llama.cpp updates

2. **Option B**: Use GGUF Python bindings with custom hooks
   - ✅ Easier to implement
   - ✅ Python ecosystem for analysis
   - ⚠️ May be slower

3. **Option C**: Build TypeScript wrapper around llama.cpp CLI
   - ✅ Stays in our stack
   - ⚠️ Limited to what llama.cpp exposes
   - ❌ May not have activation access

**Recommendation**: Start with **Option B** (Python bindings + hooks) for speed, fall back to Option A if needed.

---

### Intervention Implementation Strategy

**Challenge**: Modifying activations during inference

**Approach**:
1. **Load model** into memory
2. **Register hooks** at target layers (layer 31)
3. **Intercept activations** (attention scores, FFN gates)
4. **Apply transformations** (boost, suppress, etc.)
5. **Continue inference** with modified activations
6. **Compare outputs** vs baseline

**Libraries**:
- **llama-cpp-python**: Python bindings for llama.cpp
- **PyTorch** (optional): If we need to load GGUF → PyTorch for intervention
- **Transformers** (optional): Hugging Face for easier layer access

---

## 📊 Expected Outcomes

### If Hypotheses Correct:

| Hypothesis | Expected Evidence | Validation Method |
|------------|------------------|-------------------|
| H1: Weak Attention (L31) | Attention entropy ↑, focus score ↓ | Measure attention weights |
| H2: FFN Dominance (L31) | FFN/Attn ratio ≈ 2.0 | Compare activations |
| H3: Value Sparsity Loss | Info retention ↓ 40-60% | Track value activations |
| H4: Attention Boost Helps | Hallucinations ↓ 30%+ | Intervention A results |
| H5: FFN Suppress Helps | Hallucinations ↓ 20-30% | Intervention B results |

### If Hypotheses Incorrect:

**Alternative Explanations to Investigate**:
- Training data quality (not architecture)
- Tokenization artifacts
- Quantization errors (Q4_K introduces hallucinations)
- Prompt engineering issues

---

## 📈 Milestones & Timeline

### Week 1: Setup & Baseline (Tasks 2.1-2.2)
- ✅ Day 1-2: Build activation capture framework
- ✅ Day 3-4: Capture baseline activations
- ✅ Day 5: Analyze attention patterns

### Week 2: Analysis & Intervention (Tasks 2.3-2.5)
- ✅ Day 6-7: FFN vs Attention analysis
- ✅ Day 8-9: Value flow analysis
- ✅ Day 10-12: Run intervention experiments
- ✅ Day 13-14: Consolidate findings

### Deliverables
- 📊 Activation dataset (20+ hallucination traces)
- 📈 Quantitative metrics (attention/FFN ratios, entropy, etc.)
- 🧪 Intervention results (comparative hallucination rates)
- 📝 Phase 2 research report (validates/refutes hypotheses)

---

## 🔗 Phase 1 → Phase 2 Connection

### Phase 1 Findings → Phase 2 Tests

| Phase 1 Discovery | Phase 2 Validation |
|------------------|-------------------|
| Q/Gate ratio = 0.50 (L31) | Measure FFN/Attn activation ratio |
| Value sparsity 2.79% | Track value activation sparsity |
| Value amplification +134% | Measure value magnitude changes |
| FFN Norm +30.7% | Compare norm outputs L0 vs L31 |
| Attention weakening | Plot attention entropy across layers |

### Key Question to Answer
**"Does the weight-based structural bias translate to runtime behavior bias during hallucinations?"**

If YES → Architecture causes hallucinations (fix requires model redesign)
If NO → Training/data causes hallucinations (fix requires better data)

---

## 🎯 Success Definition

Phase 2 is successful if:
1. ✅ We **validate ≥3 of 5 hypotheses** from Phase 1
2. ✅ We **demonstrate causal link** between layer 31 patterns and hallucinations
3. ✅ We **reduce hallucinations by ≥30%** with at least one intervention
4. ✅ We **publish findings** that advance hallucination research

---

**Phase 2 Status**: 🚀 Starting Now!
**Next Task**: Build activation capture framework (Task 2.1)

Let's go! 🔬
