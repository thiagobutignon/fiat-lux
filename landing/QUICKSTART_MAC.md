# Quick Start Guide for Mac M4 (24GB RAM)

## TL;DR - Get Running in 5 Minutes

```bash
# 1. Install dependencies
npm install
brew install llama.cpp
pip install -U "huggingface_hub[cli]"

# 2. Download model (~5GB, takes 2-5 min)
mkdir -p models
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --local-dir ./models --local-dir-use-symlinks False

# 3. Start llama.cpp server (in terminal 1)
./scripts/start-llamacpp-mac.sh

# 4. Configure and run (in terminal 2)
cp .env.example .env
# Edit .env: Set ENABLE_LLAMACPP=true
npm run benchmark:quick
```

## Detailed Setup

### Prerequisites

Your Mac M4 with 24GB RAM is perfect for this! You can run:
- ✅ Llama 3.1 8B Q4 (6GB RAM, 300-400 tok/s)
- ✅ Llama 3.1 8B Q8 (9GB RAM, 200-300 tok/s)
- ✅ Multiple browser tabs + IDE while benchmarking

### Step 1: Install Tools (2 minutes)

```bash
# Node.js dependencies
npm install

# llama.cpp for Metal acceleration
brew install llama.cpp

# Hugging Face CLI for model downloads
pip install -U "huggingface_hub[cli]"
```

### Step 2: Download Model (2-5 minutes)

We'll use **Q4_K_M quantization** - best balance of speed/quality for benchmarking.

```bash
# Create models directory
mkdir -p models

# Download Llama 3.1 8B Q4 (~4.7GB)
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --local-dir ./models \
  --local-dir-use-symlinks False
```

**Download progress**: You'll see a progress bar. Takes 2-5 minutes on fast internet.

### Step 3: Start llama.cpp Server

Open a **new terminal window** and run:

```bash
./scripts/start-llamacpp-mac.sh
```

**Expected output**:
```
Starting llama.cpp server for Mac (Metal acceleration)...
Model: Meta-Llama-3.1-8B-Instruct-Q4_K_M
Port: 8080

✓ llama-server installed
✓ Model found: ./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
  Size: 4.7G

System Information:
  Chip: Apple M4
  RAM: 24 GB
  Cores: 10

✓ Port 8080 available

Starting server with optimized settings...
  GPU Layers: 999 (all layers offloaded to Metal)
  Context Size: 2048
  CPU Threads: 8

...loading model...

llama server listening at http://0.0.0.0:8080  <-- READY!
```

**Keep this terminal open!** The server needs to run during benchmarking.

### Step 4: Configure Environment

In your **original terminal**:

```bash
# Copy example config
cp .env.example .env

# Edit .env file
nano .env  # or use your preferred editor
```

**Set these values in .env**:
```bash
# Disable Gemini (to avoid API costs)
ENABLE_GEMINI=false

# Enable llama.cpp (our Mac-optimized option)
ENABLE_LLAMACPP=true
LLAMACPP_BASE_URL=http://localhost:8080
LLAMACPP_MODEL=Meta-Llama-3.1-8B-Instruct-Q4_K_M

# Disable other options
ENABLE_VLLM=false
ENABLE_LOCAL_LLAMA=false
```

Save and exit (Ctrl+X, then Y in nano).

### Step 5: Run Benchmark

```bash
# Quick test (100 cases, ~30 seconds)
npm run benchmark:quick

# Full benchmark (1000 cases, ~5 minutes)
npm run benchmark:full
```

## Expected Results

### Performance on Mac M4

```
Grammar Engine:      100% accuracy, 0.02ms latency
llama.cpp (Q4):      80-85% accuracy, 40-60ms latency
Simulated LLMs:      82-89% accuracy, simulated
```

### Benchmark Output

```
═══════════════════════════════════════════════════════
  DETERMINISTIC INTELLIGENCE BENCHMARK
  Domain: Trading Signal Generation
═══════════════════════════════════════════════════════

Generating 100 test cases...
✓ Generated 100 test cases
✓ llama.cpp enabled (Meta-Llama-3.1-8B-Instruct-Q4_K_M) - Metal Acceleration

Running benchmarks for 6 systems...

───────────────────────────────────────────────────────
Running: Grammar Engine...
Progress: 100/100 [████████████████████████████] 100%
✓ Completed in 2ms
  Accuracy: 100.00% | Avg Latency: 0.02ms
───────────────────────────────────────────────────────

───────────────────────────────────────────────────────
Running: llama.cpp (Meta-Llama-3.1-8B-Instruct-Q4_K_M)...
Progress: 100/100 [████████████████████████████] 100%
✓ Completed in 4500ms
  Accuracy: 82.00% | Avg Latency: 45ms
───────────────────────────────────────────────────────

...

📊 RESULTS SUMMARY

| System                      | Accuracy | Latency  | Cost/1k  | Explainable |
|----------------------------|----------|----------|----------|-------------|
| Grammar Engine              | 100%     | 0.0200ms | $0.00    | ✅ 100%     |
| llama.cpp (Q4_K_M)         | 82%      | 45.0ms   | $0.00    | ❌ 0%       |
| GPT-4 (sim)                | 87%      | 350ms    | $0.50    | ❌ 0%       |
| Claude 3.5 (sim)           | 89%      | 280ms    | $0.45    | ❌ 0%       |
| Llama 3.1 70B (sim)        | 82%      | 120ms    | $0.05    | ❌ 0%       |
| Custom LSTM                 | 75%      | 45ms     | $0.01    | ❌ 0%       |

🏆 WINNER: Grammar Engine

📈 COMPARISONS:

1. Grammar Engine is 2250x faster than llama.cpp with 18% higher accuracy

Results saved to: benchmark-results/benchmark-2025-10-07T123456.json
```

## Troubleshooting

### Server won't start

```bash
# Check if port is in use
lsof -i :8080

# Kill existing process
kill -9 $(lsof -t -i :8080)

# Restart server
./scripts/start-llamacpp-mac.sh
```

### Benchmark fails with "connection refused"

Make sure llama.cpp server is running! Check terminal 1 for:
```
llama server listening at http://0.0.0.0:8080
```

### Slow performance (< 100 tok/s)

Close other heavy apps:
- Chrome tabs
- Docker containers
- Video editors
- Other ML tools

Your M4 should easily hit 300+ tok/s with Q4 model.

### Out of memory

Use smaller quantization (Q4 uses only 6GB):
```bash
# You're already using Q4_K_M which is optimal for 24GB RAM
# If issues persist, close background apps
```

## Alternative: Use Gemini API (Easier but Costs Money)

If you don't want to run local models:

```bash
# Get free API key: https://aistudio.google.com/apikey
# Edit .env:
GEMINI_API_KEY=your_key_here
ENABLE_GEMINI=true
ENABLE_LLAMACPP=false

# Run benchmark (no local server needed)
npm run benchmark:quick
```

**Cost**: ~$0.08 per 1000 requests with Gemini 2.5 Flash

## Next Steps

1. ✅ Run quick benchmark to verify setup
2. 📊 Run full benchmark: `npm run benchmark:full`
3. 📈 Check results in `benchmark-results/`
4. 🌐 Try web UI: `npm run dev` → http://localhost:3000/benchmark
5. ⚙️ Experiment with different models (Q8, FP16) - see `MAC_SETUP.md`

## Performance Comparison

| Option | Speed | Setup | Cost | Recommended For |
|--------|-------|-------|------|-----------------|
| llama.cpp | 300-400 tok/s | Medium | Free | **Mac M4** ⭐ |
| Gemini API | ~200ms latency | Easy | $0.08/1k | Quick tests |
| Ollama | 20-50 tok/s | Easy | Free | Not recommended |
| vLLM | N/A | N/A | N/A | NVIDIA only |

## Full Documentation

- **MAC_SETUP.md** - Complete Mac setup guide with advanced tuning
- **BENCHMARK_README.md** - Full benchmark documentation
- **VLLM_SETUP.md** - vLLM setup for NVIDIA GPUs (not applicable for Mac)

## Support

Issues? Check:
1. llama.cpp server is running (terminal 1)
2. .env has ENABLE_LLAMACPP=true
3. Model file exists: `ls -lh models/`
4. Port 8080 is available: `lsof -i :8080`

For detailed troubleshooting, see **MAC_SETUP.md**.
