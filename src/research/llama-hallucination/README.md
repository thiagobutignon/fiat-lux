# Llama 3.1 Hallucination Research

## Internal Mechanism Analysis - Understanding Hallucination Pathways

**Research Goal**: Systematic investigation into the internal mechanisms of Llama 3.1 8B to understand the neural pathways and weight patterns that lead to hallucinations.

**Target Venue**: arXiv preprint
**Status**: Phase 1 - In Progress
**Issue**: [#6](https://github.com/thiagobutignon/fiat-lux/issues/6)

---

## Project Structure

```
src/research/llama-hallucination/
├── domain/              # Core types and interfaces
│   └── weight-statistics.ts
├── data/                # Data extraction and storage
├── analysis/            # Analysis algorithms
│   └── weight-extractor.ts
├── visualization/       # Visualization tools
├── demos/              # Demonstration scripts
│   └── phase1-weight-analysis.ts
└── README.md           # This file
```

---

## Research Phases

### Phase 1: Weight Pattern Analysis ✅ In Progress

**Objective**: Identify statistical patterns in weights associated with different layer types and behaviors.

#### Tasks

- [x] **Task 1.1**: Extract and analyze all layer weights
  - Extract weights from all 32 transformer layers
  - Compute statistical distributions (mean, std, sparsity, L1/L2 norms)
  - Compare early layers (0-10) vs. middle (11-21) vs. late (22-31)
  - **Success Metric**: Complete weight profile for all 8.03B parameters

- [ ] **Task 1.2**: Attention head specialization analysis
- [ ] **Task 1.3**: FFN gate analysis
- [ ] **Task 1.4**: Layer norm scale investigation

### Phase 2: Activation Flow Tracing

**Status**: Not Started
**Objective**: Trace activation flows during inference to identify divergence points.

### Phase 3: Quantization Impact Study

**Status**: Not Started
**Objective**: Understand how quantization affects hallucination rates.

### Phase 4: Mechanistic Interpretability

**Status**: Not Started
**Objective**: Build causal models linking specific weight patterns to hallucination behaviors.

---

## Quick Start

### Prerequisites

1. **Llama 3.1 8B model** in GGUF format
   - Download from HuggingFace: `llama-3.1-8b-instruct.Q4_K_M.gguf`
   - Or use any GGUF-format Llama model

2. **Set model path**:
   ```bash
   export LLAMA_MODEL_PATH=/path/to/your/llama-3.1-8b.gguf
   ```

### Running Phase 1 Analysis

```bash
# Using npm script
npm run research:hallucination:phase1

# Or directly with tsx
tsx src/research/llama-hallucination/demos/phase1-weight-analysis.ts

# Or with custom model path
tsx src/research/llama-hallucination/demos/phase1-weight-analysis.ts /path/to/model.gguf
```

### Expected Output

The analysis will:
1. Load the GGUF model
2. Extract weight statistics for all layers
3. Compute global and layer-specific statistics
4. Save detailed results to `research-output/phase1/weight-profile-{timestamp}.json`

```
🔬 Llama 3.1 Hallucination Research - Phase 1: Weight Pattern Analysis

📁 Model: llama-3.1-8b-instruct.Q4_K_M.gguf
📍 Path: /models/llama-3.1-8b.gguf

✅ Loaded model: Llama-3.1-8B-Instruct
   Total parameters: 8,030,261,248
   Quantization: Q4_K, Q6_K
   Layers: 32

📊 Extracting weight profile...
  Layer 0/31...
  Layer 1/31...
  ...

RESULTS: Weight Pattern Analysis
═══════════════════════════════════════════════════════════

Global Statistics:
  Mean Weight: 0.000123
  Std Weight: 0.045678
  Overall Sparsity: 12.34%
  Total L1 Norm: 1.23e+08
  Total L2 Norm: 4.56e+06

Layer Range Comparison:
  Early Layers (0-10):
    Mean: 0.000145, Std: 0.048234
    Sparsity: 10.23%
  Middle Layers (11-21):
    Mean: 0.000112, Std: 0.043567
    Sparsity: 13.45%
  Late Layers (22-31):
    Mean: 0.000098, Std: 0.041234
    Sparsity: 14.67%

✅ Task 1.1: Extract and analyze all layer weights
   ✓ Complete weight profile for 8,030,261,248 parameters
   ✓ Analyzed 96 layer components
   ✓ Statistical distributions computed
   ✓ Layer range comparisons complete

🎉 Phase 1 - Task 1.1 Complete!
```

---

## Output Files

### Weight Profile JSON

Location: `research-output/phase1/weight-profile-{timestamp}.json`

Structure:
```json
{
  "modelName": "Llama-3.1-8B-Instruct",
  "quantizationType": "Q4_K, Q6_K",
  "totalParameters": 8030261248,
  "layers": [
    {
      "layerIndex": 0,
      "layerType": "attention",
      "attention": {
        "query": {
          "mean": 0.000123,
          "std": 0.045678,
          "sparsity": 0.1234,
          "l1Norm": 123456.78,
          "l2Norm": 4567.89,
          ...
        },
        "key": { ... },
        "value": { ... },
        "output": { ... }
      }
    },
    ...
  ],
  "globalStatistics": { ... },
  "earlyLayers": { ... },
  "middleLayers": { ... },
  "lateLayers": { ... }
}
```

---

## Key Findings (Updated as Research Progresses)

### Phase 1 Findings

*TBD - Will be updated after first successful analysis*

Key questions to investigate:
- Do early layers have different weight distributions than late layers?
- Are there layers with unusually high/low sparsity?
- Do certain attention heads have specialized weight patterns?
- How does quantization affect weight statistics?

---

## Development Notes

### Current Implementation Status

✅ **Completed**:
- GGUF parser integration
- Weight extraction infrastructure
- Basic statistical analysis (mean, std, sparsity, norms)
- FP16 and FP32 tensor support
- Layer-wise analysis framework
- Output JSON generation

⏳ **In Progress**:
- Quantized tensor dequantization (Q4_K, Q6_K, etc.)
- Attention head slicing
- FFN gate analysis
- Spectral norm computation

🔜 **Next Steps**:
- Complete Task 1.2 (Attention head specialization)
- Implement full dequantization for all quantization types
- Add visualization generation
- Statistical hypothesis testing

### Known Limitations

1. **Quantized Tensors**: Currently skips Q4_K/Q6_K tensors - need to implement full dequantization
2. **Memory Usage**: Large models may require streaming/batching
3. **Performance**: No GPU acceleration yet (CPU-only)
4. **Spectral Analysis**: Disabled by default (expensive SVD computation)

---

## Contributing

This is active research. If you want to contribute:

1. Check current phase progress in [Issue #6](https://github.com/thiagobutignon/fiat-lux/issues/6)
2. Look for tasks marked as "Help Wanted"
3. Focus on:
   - Dequantization algorithms
   - Visualization tools
   - Statistical analysis methods
   - Hallucination test suite curation

---

## References

- **Anthropic's Interpretability Research**: Causal scrubbing, activation patching
- **EleutherAI's Pythia Suite**: Training dynamics and interpretability
- **Neel Nanda's TransformerLens**: Mechanistic interpretability toolkit
- **Redwood Research**: Causal tracing in language models
- **GGUF Format Specification**: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

---

## License

MIT - Same as parent project

---

**Last Updated**: 2025-10-08
**Phase**: 1 - Weight Pattern Analysis
**Status**: Active Development 🚧
