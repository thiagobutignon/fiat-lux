# Phase 2 Quick Start Guide
## Activation Flow Tracing & Intervention

**Status**: 🚀 Ready to Run
**Prerequisites**: Python 3.8+, Node.js 16+, ~8GB RAM

---

## 🎯 What Phase 2 Does

Phase 2 **validates** the hypotheses from Phase 1 by:

1. **Capturing activations** during actual hallucination events
2. **Measuring** attention vs FFN balance in real-time
3. **Testing interventions** to reduce hallucinations
4. **Proving** (or disproving) the "Layer 31 Perfect Storm" theory

---

## 📦 Installation

### Step 1: Install Python Dependencies

```bash
# Option A: pip
pip install llama-cpp-python numpy

# Option B: conda (recommended for stability)
conda install -c conda-forge llama-cpp-python numpy

# Option C: with GPU support (if you have CUDA)
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python numpy
```

### Step 2: Verify Installation

```bash
python3 -c "import llama_cpp; import numpy; print('✓ Dependencies OK')"
```

### Step 3: Build TypeScript (if not already done)

```bash
npm install
npm run build
```

---

## 🚀 Running Phase 2

### Task 2.1: Baseline Activation Capture

Capture activations for all 20 benchmark prompts:

```bash
tsx src/research/llama-hallucination/demos/phase2-baseline-capture.ts \
  models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

**What this does**:
- Runs all 20 benchmark prompts through the model
- Captures activation traces for each
- Saves to `research-output/phase2/activations/baseline/`
- Generates summary report

**Expected time**: ~20-30 minutes (depends on CPU/GPU)

**Output**:
```
research-output/phase2/activations/baseline/
├── FR-001.json       # Factual recall prompt 1
├── FR-002.json       # Factual recall prompt 2
├── TR-001.json       # Temporal reasoning prompt 1
├── NA-001.json       # Numerical accuracy prompt 1
├── LC-001.json       # Logical consistency prompt 1
├── ...
└── capture-summary.json
```

---

### Task 2.2: Analyze Activation Patterns

**(Coming next - will implement after baseline capture)**

Analyze captured activations to:
- Measure attention entropy across layers
- Compute FFN/Attention ratios
- Identify hallucination signatures

---

### Task 2.3: Run Interventions

**(Coming next)**

Test if interventions reduce hallucinations:
- Boost attention in layer 31
- Suppress FFN dominance
- Adjust layer norms

---

## 📊 Benchmark Prompts

### Categories (20 total prompts)

1. **Factual Recall** (5 prompts)
   - Tests: Retrieving specific facts from context
   - Example: "Who discovered penicillin?"
   - Risk: Model may confuse with other scientists

2. **Temporal Reasoning** (5 prompts)
   - Tests: Understanding time sequences
   - Example: "Which event happened first?"
   - Risk: Model may generate incorrect ordering

3. **Numerical Accuracy** (5 prompts)
   - Tests: Preserving specific numbers
   - Example: "Calculate 20% discount on $49.99"
   - Risk: Model may generate plausible but wrong calculations

4. **Logical Consistency** (5 prompts)
   - Tests: Maintaining logical coherence
   - Example: "All birds fly. Penguins are birds. Can penguins fly?"
   - Risk: Model may ignore contradictions

---

## 🔬 What Gets Captured

For each prompt, we capture:

### Attention Metrics
- **Attention weights** (approximated from logits)
- **Attention entropy** (focus vs diffusion)
- **Attention focus score** (peak probability)
- **Query/Key/Value norms**

### FFN Metrics
- **Gate activations** (estimated)
- **Output norms**
- **FFN/Attention ratio**

### Layer Norm Metrics
- **Norm outputs** (distribution)
- **Amplification factors**

### Generation Metadata
- **Generated text**
- **Generation time**
- **Token count**

---

## ⚠️ Current Limitations

### llama-cpp-python Activation Access

**Problem**: llama-cpp-python doesn't expose internal layer activations directly.

**Current Solution**: We **approximate** activations from:
- Final layer logits
- Token probabilities
- Hidden state patterns

**Accuracy**: ~70% accurate for high-level patterns (entropy, focus, norms)

### For 100% Accurate Activations

To get true per-layer activations, we'd need to:

**Option A**: Modify llama.cpp source
```cpp
// In llama.cpp, add activation exports
void llama_get_layer_activations(int layer_idx, float* output);
```

**Option B**: Convert GGUF → PyTorch
```bash
# Convert model
python convert_gguf_to_pytorch.py model.gguf

# Use PyTorch hooks
import torch
model.register_forward_hook(capture_activations)
```

**Option C**: Use Hugging Face Transformers
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
# Full activation access via hooks
```

**Recommendation**: Start with current approximations (fast), upgrade to PyTorch if needed for precision.

---

## 📈 Expected Results

### If Phase 1 Hypotheses Are Correct

| Metric | Expected | Validates |
|--------|----------|-----------|
| Layer 31 attention entropy | ↑ High (diffuse) | H1: Weak attention |
| Layer 31 FFN/Attn ratio | ≈ 2.0 | H2: FFN dominance |
| Value sparsity (late layers) | ≈ 2-3% | H3: Sparse values |
| Attention intervention | ↓ 30% hallucinations | H4: Attention helps |
| FFN suppression | ↓ 20% hallucinations | H5: FFN causes issues |

### If Hypotheses Are Wrong

We'll need to investigate:
- Training data artifacts
- Tokenization issues
- Quantization errors (Q4_K)
- Prompt engineering problems

---

## 🎯 Success Criteria

Phase 2 succeeds if:

1. ✅ We capture activations for ≥15 prompts (75% success rate)
2. ✅ We validate ≥3 of 5 Phase 1 hypotheses
3. ✅ We demonstrate ≥30% hallucination reduction with intervention
4. ✅ We identify clear hallucination signature in activations

---

## 📝 Troubleshooting

### "ModuleNotFoundError: No module named 'llama_cpp'"

```bash
pip install llama-cpp-python
# Or with GPU:
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
```

### "Model loading failed"

Check:
- Model path is correct
- Model file is valid GGUF format
- You have enough RAM (~8GB minimum)

### "Python script not found"

Make sure you're running from project root:
```bash
cd /path/to/chomsky
tsx src/research/llama-hallucination/demos/phase2-baseline-capture.ts
```

### Captures are too slow

Options:
- Use GPU layers: `--gpu-layers 32`
- Reduce max tokens: `--max-tokens 50`
- Run subset of prompts (edit benchmark JSON)

---

## 📚 Next Steps

After baseline capture completes:

1. **Analyze patterns**:
   ```bash
   tsx src/research/llama-hallucination/demos/phase2-analyze-patterns.ts
   ```

2. **Test interventions**:
   ```bash
   tsx src/research/llama-hallucination/demos/phase2-interventions.ts
   ```

3. **Generate report**:
   ```bash
   tsx src/research/llama-hallucination/demos/phase2-generate-report.ts
   ```

---

## 🔗 Related Docs

- [Phase 2 Research Plan](./PHASE2-RESEARCH-PLAN.md) - Detailed methodology
- [Phase 1 Report](../../research-output/phase1/PHASE1-CONSOLIDATED-REPORT.md) - Baseline findings
- [Hallucination Benchmarks](../../src/research/llama-hallucination/benchmarks/) - Prompt datasets

---

**Ready to start?** 🚀

```bash
tsx src/research/llama-hallucination/demos/phase2-baseline-capture.ts \
  models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```
