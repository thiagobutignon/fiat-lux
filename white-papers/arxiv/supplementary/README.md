# GVCS Supplementary Materials

Supporting data, prompts, and benchmarks for the paper **"GVCS: Genetic Version Control System - LLM-Assisted Evolution for 250-Year Software"**

**Authors**: VERDE Development Team (A.S., L.T.) with ROXO Integration (J.D., M.K.)
**Date**: October 10, 2025
**Paper Version**: 1.0

---

## 📁 Files Overview

| File | Type | Size | Description |
|------|------|------|-------------|
| `gvcs-benchmarks.csv` | CSV | ~2KB | 100 generations evolution data |
| `multi-organism-competition.json` | JSON | ~8KB | 3 organisms, 5 generations competition logs |
| `llm-prompts-examples.md` | Markdown | ~15KB | Complete LLM prompts with examples |
| `ablation-study.csv` | CSV | ~3KB | Component removal impact analysis |

---

## 📊 1. GVCS Benchmarks (`gvcs-benchmarks.csv`)

**Experiment**: Single organism (Oncology domain), 100 generations

### Columns

- `generation` (int): Generation number (0-100)
- `fitness` (float): Overall fitness score (0.0-1.0)
- `latency_ms` (int): p50 latency in milliseconds
- `throughput_rps` (int): Requests per second
- `error_rate` (float): Percentage of 4xx+5xx errors
- `crash_rate` (float): Percentage of unhandled exceptions
- `mutations_applied` (int): Number of mutations applied this generation
- `mutations_rejected` (int): Number of mutations rejected (constitutional violations)
- `knowledge_patterns` (int): Total knowledge patterns accumulated

### Key Results

- **Fitness improvement**: 0.42 → 0.87 (+107%)
- **Latency improvement**: 145ms → 48ms (-67%)
- **Throughput improvement**: 412 → 1021 RPS (+148%)
- **Error elimination**: 8.2% → 0.4% (-95%)
- **Crash elimination**: 2.1% → 0% (-100%)
- **Convergence**: Generation 75 (plateau at fitness ~0.87)
- **Total mutations**: 72 successful, 28 rejected
- **Knowledge growth**: 847 → 1770 patterns (+109%)

### Usage

```python
import pandas as pd
df = pd.read_csv('gvcs-benchmarks.csv', comment='#')

# Plot fitness over time
import matplotlib.pyplot as plt
plt.plot(df['generation'], df['fitness'])
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('GVCS Evolution: 100 Generations')
plt.show()
```

---

## 🏆 2. Multi-Organism Competition (`multi-organism-competition.json`)

**Experiment**: 3 organisms (Oncology, Neurology, Cardiology), 5 generations, knowledge transfer enabled

### Structure

```json
{
  "experiment": "Multi-Organism Competition",
  "organisms": 3,
  "generations": 5,
  "organisms_data": [...],
  "generation_logs": [...],
  "summary": {...}
}
```

### Key Events

**Generation 2**: Knowledge transfer
- Oncology organism (fitness 0.83) creates pattern `adaptive_latency_cache`
- LLM analyzes applicability to Neurology organism (0.87 applicability score)
- Neurology adopts pattern, fitness jumps 0.78 → 0.82 (+4.9% in single generation)

**Generation 5**: Natural selection
- Oncology promoted (0.78 → 0.867, +8.7%)
- Neurology promoted (0.75 → 0.864, +11.4%, benefited from transfer)
- Cardiology retired (0.82 → 0.796, -2.4%, declining trajectory)

### Key Results

- **Knowledge transfer impact**: +4.9% fitness gain in 1 generation
- **Natural selection effectiveness**: 100% (worst performer correctly retired despite high initial fitness)
- **Convergence**: Both survivors converged to ~0.86 fitness ceiling
- **Validation**: Natural selection works—selects based on trajectory, not just current fitness

### Usage

```python
import json
with open('multi-organism-competition.json') as f:
    data = json.load(f)

# Extract knowledge transfer event
gen2 = [g for g in data['generation_logs'] if g['generation'] == 2][0]
transfer = [e for e in gen2['events'] if e['type'] == 'knowledge_transfer'][0]
print(f"Pattern: {transfer['pattern']}")
print(f"Applicability: {transfer['llm_analysis']['applicability_score']}")
```

---

## 🤖 3. LLM Prompts Examples (`llm-prompts-examples.md`)

Complete prompts used in GVCS with Claude Opus 4 and Sonnet 4.5

### Prompts Included

1. **Fitness Evaluation** (Claude Opus 4, $0.03/eval)
   - Temperature: 0.3
   - Analyzes 4 metrics, compares to parent, predicts trends
   - Returns JSON with fitness score + rollout recommendation

2. **Constitutional Validation** (Claude Opus 4, $0.02/validation)
   - Temperature: 0.1 (maximum precision)
   - Validates against 6 universal + 2 domain-specific principles
   - Returns accept/reject decision with reasoning

3. **Knowledge Transfer Applicability** (Claude Sonnet 4.5, $0.005/analysis)
   - Temperature: 0.3
   - Analyzes domain + architecture compatibility
   - Returns applicability score (0.0-1.0) + expected impact

4. **Commit Message Generation** (Claude Sonnet 4.5, $0.001/message)
   - Temperature: 0.5
   - Generates concise imperative commit messages
   - Returns single line (<80 chars)

### Cost Summary

- **100 evolution cycles**: $5.15 total
  - Fitness: $3.00
  - Constitutional: $2.00
  - Transfer: $0.05
  - Commits: $0.10

### Usage

Copy prompts directly for:
- Reproducing experiments
- Adapting to other domains
- Testing with different LLMs
- Understanding LLM integration architecture

---

## 🔬 4. Ablation Study (`ablation-study.csv`)

**Methodology**: Remove each component, run 100 generations × 10 runs, measure impact

### Columns

- `configuration` (string): Which component was removed
- `final_fitness_mean` (float): Mean fitness at generation 100
- `final_fitness_std` (float): Standard deviation
- `convergence_generation` (int): Generation when fitness plateaued
- `convergence_speed` (enum): normal | slow | fast | stagnated | failed
- `safety_violations` (int): Number of constitutional violations in 10 runs
- `production_failures` (int): Number of production crashes in 10 runs
- `notes` (string): Observations

### Key Results

**ESSENTIAL Components** (cannot remove):
- Natural Selection: -26% fitness, no convergence
- Constitutional AI: 30% failure rate (3/10 runs)
- Canary Deployment: 20% production failures (2/10 runs)
- Auto-Commit: System cannot function
- Fitness Calculation: System cannot function
- Version Increment: System cannot function

**ENHANCEMENT Components** (improve performance):
- LLM Integration: +15% fitness ($5/100 gen)
- Old-But-Gold: +2% fitness, knowledge retention

**Validation**:
- No unnecessary components (all contribute)
- 6 ESSENTIAL, 2 ENHANCEMENT
- Architecture well-designed

### Usage

```python
import pandas as pd
df = pd.read_csv('ablation-study.csv', comment='#')

# Identify essential components
essential = df[df['final_fitness_mean'] < 0.70]
print(f"Essential components: {len(essential)}")

# Plot impact
import matplotlib.pyplot as plt
df_sorted = df.sort_values('final_fitness_mean')
plt.barh(df_sorted['configuration'], df_sorted['final_fitness_mean'])
plt.xlabel('Final Fitness')
plt.title('Component Ablation Impact')
plt.axvline(x=0.87, color='r', linestyle='--', label='Baseline')
plt.legend()
plt.tight_layout()
plt.show()
```

---

## 📈 Reproducing Results

### 1. Single Organism Evolution (100 Generations)

```bash
# Setup
cd /path/to/gvcs
npm install

# Run experiment
npm run gvcs:evolve -- \
  --organism oncology \
  --generations 100 \
  --output benchmarks.csv

# Expected time: ~18 minutes
# Expected cost: $3.00 (LLM) + $0 (compute)
```

### 2. Multi-Organism Competition

```bash
# Run 3 organisms with knowledge transfer
npm run gvcs:compete -- \
  --organisms oncology,neurology,cardiology \
  --generations 5 \
  --knowledge-transfer \
  --output competition.json

# Expected time: ~1 minute
# Expected cost: $0.25
```

### 3. Ablation Study

```bash
# Run ablation study (8 configurations × 10 runs × 100 generations)
npm run gvcs:ablation -- \
  --runs 10 \
  --generations 100 \
  --output ablation.csv

# Expected time: ~24 hours
# Expected cost: $240 (8 × 10 × $3)
# Recommendation: Run in parallel on multiple machines
```

---

## 📝 Citation

If you use these materials, please cite:

```bibtex
@article{gvcs2025,
  title={GVCS: Genetic Version Control System - LLM-Assisted Evolution for 250-Year Software},
  author={VERDE Team (A.S., L.T.) and ROXO Team (J.D., M.K.)},
  journal={arXiv preprint},
  year={2025},
  note={Supplementary materials available at [URL]}
}
```

---

## 📞 Contact

For questions about these materials:
- **Technical issues**: Open issue on GitHub
- **Data requests**: Contact corresponding author
- **Collaboration**: Email team lead

---

## 📜 License

These supplementary materials are released under the same license as the main paper.

**Copyright © 2025 Fiat Lux AGI Research Initiative**

---

**Last Updated**: October 10, 2025
**Version**: 1.0
**Paper DOI**: [To be assigned by arXiv]
