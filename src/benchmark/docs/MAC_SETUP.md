# Mac M-Series Setup Guide (M1/M2/M3/M4)

## Overview

This guide shows how to run **high-performance local LLM inference** on Mac M-series chips using **llama.cpp** with Metal acceleration.

### Performance on Mac M4 (24GB RAM)

| Model | Speed | RAM Usage | Quality |
|-------|-------|-----------|---------|
| Llama 3.1 8B Q4_K_M | **300-400 tok/s** | ~6GB | Excellent |
| Llama 3.1 8B Q8_0 | 200-300 tok/s | ~9GB | Best |
| Llama 3.1 8B FP16 | 150-200 tok/s | ~16GB | Perfect |

**Recommendation**: Use **Q4_K_M** for best balance of speed/quality.

## Quick Start

### 1. Install llama.cpp via Homebrew

```bash
# Install llama.cpp with Metal support
brew install llama.cpp

# Verify installation
llama-server --version
```

### 2. Install Hugging Face CLI (for downloading models)

```bash
# Install via pip
pip install -U "huggingface_hub[cli]"

# Or via Homebrew
brew install huggingface-cli
```

### 3. Download Llama 3.1 8B Model (Q4 quantized)

```bash
# Create models directory
mkdir -p models

# Download optimized GGUF model (~4.7GB)
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --local-dir ./models \
  --local-dir-use-symlinks False
```

**Note**: Download takes 2-5 minutes depending on your internet speed.

### 4. Start llama.cpp Server

Use the helper script (recommended):

```bash
# Make script executable
chmod +x scripts/start-llamacpp-mac.sh

# Start server (uses Q4_K_M by default)
./scripts/start-llamacpp-mac.sh
```

Or start manually:

```bash
llama-server \
  -m ./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  -ngl 999 \
  -c 2048 \
  --port 8080 \
  --host 0.0.0.0 \
  -t 8
```

**Server will be ready when you see**: `llama server listening at http://0.0.0.0:8080`

### 5. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add:
ENABLE_LLAMACPP=true
LLAMACPP_BASE_URL=http://localhost:8080
LLAMACPP_MODEL=Meta-Llama-3.1-8B-Instruct-Q4_K_M

# Disable other LLM options
ENABLE_GEMINI=false
ENABLE_VLLM=false
ENABLE_LOCAL_LLAMA=false
```

### 6. Run Benchmark

```bash
# Quick test (100 cases, ~30 seconds)
npm run benchmark:quick

# Full benchmark (1000 cases, ~5 minutes)
npm run benchmark:full
```

## Performance Tuning

### Model Quantization Levels

Choose based on your needs:

```bash
# Q4_K_M - RECOMMENDED (best balance)
# - Speed: 300-400 tok/s
# - RAM: ~6GB
# - Quality: Excellent (minimal loss)
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --local-dir ./models --local-dir-use-symlinks False

# Q8_0 - Higher Quality
# - Speed: 200-300 tok/s
# - RAM: ~9GB
# - Quality: Best (near-lossless)
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
  --local-dir ./models --local-dir-use-symlinks False

# FP16 - Maximum Quality
# - Speed: 150-200 tok/s
# - RAM: ~16GB
# - Quality: Perfect (no loss)
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  Meta-Llama-3.1-8B-Instruct-F16.gguf \
  --local-dir ./models --local-dir-use-symlinks False
```

### llama-server Options

Optimize for Mac M4:

```bash
llama-server \
  -m ./models/MODEL.gguf \
  -ngl 999 \              # Offload all layers to GPU (Metal)
  -c 2048 \               # Context size (2048 is good for this task)
  -t 8 \                  # CPU threads (M4 has 10 cores, use 8)
  --port 8080 \
  --host 0.0.0.0 \
  --n-gpu-layers 999 \    # Alias for -ngl
  --numa distribute       # NUMA optimization for multi-core
```

**Key Parameters**:
- `-ngl 999`: Offload ALL layers to Metal GPU (fastest)
- `-c 2048`: Context window size (lower = faster, but enough for patterns)
- `-t 8`: Use 8 CPU threads (good for M4's 10 cores)
- `--numa distribute`: Better multi-core utilization

### Memory Management

Mac M4 with 24GB RAM can handle:

| Concurrent Tasks | Model Size | Safe? |
|-----------------|------------|-------|
| Benchmark only | Q4 (6GB) | ✅ Yes |
| Benchmark + Browser | Q4 (6GB) | ✅ Yes |
| Benchmark + Heavy apps | Q8 (9GB) | ⚠️ Maybe |
| Multiple models | FP16 (16GB) | ❌ No |

**Recommendation**: Close other heavy apps during benchmark for best performance.

## Troubleshooting

### Server won't start

```bash
# Check if port is in use
lsof -i :8080

# Kill existing process
kill -9 $(lsof -t -i :8080)

# Try different port
llama-server -m ./models/MODEL.gguf -ngl 999 --port 8081
# Update .env: LLAMACPP_BASE_URL=http://localhost:8081
```

### Slow inference (< 100 tok/s)

1. **Check Metal is enabled**:
   ```bash
   # Should see: "ggml_metal_init: GPU name: Apple M4"
   # If not, Metal isn't working
   ```

2. **Verify GPU layers**:
   ```bash
   # Make sure you're using -ngl 999
   # Check logs for: "llm_load_tensors: offloaded 32/33 layers to GPU"
   ```

3. **Close other apps**:
   - Chrome/Safari with many tabs
   - Video/photo editing apps
   - Docker containers

4. **Try smaller model**:
   ```bash
   # Use Q4 instead of Q8/FP16
   ```

### Out of memory errors

```bash
# Use smaller quantization
# Q4 (~6GB) instead of Q8 (~9GB) or FP16 (~16GB)

# Or reduce context size
llama-server -m MODEL.gguf -ngl 999 -c 1024  # Half context
```

### Model not found

```bash
# Check model file exists
ls -lh models/

# Re-download if needed
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --local-dir ./models --local-dir-use-symlinks False
```

### Benchmark fails with connection error

```bash
# Make sure server is running
curl http://localhost:8080/health

# Should return: {"status":"ok"}

# If not, start server and wait for "listening" message
```

## Alternative: Use Ollama (Easier but Slower)

If llama.cpp is too complex, use Ollama (10x slower but easier):

```bash
# Install Ollama
brew install ollama

# Pull model
ollama pull llama3.1:8b

# Start Ollama (runs as service)
ollama serve

# Configure .env
ENABLE_LOCAL_LLAMA=true
OLLAMA_MODEL=llama3.1:8b
```

**Performance**: ~20-50 tok/s (vs 300-400 for llama.cpp)

## Comparison: llama.cpp vs vLLM vs Ollama

| Feature | llama.cpp (Mac) | vLLM (NVIDIA) | Ollama |
|---------|----------------|---------------|---------|
| Speed | 300-400 tok/s | 400-600 tok/s | 20-50 tok/s |
| Setup | Easy | Complex | Easiest |
| Mac Support | ✅ Native | ❌ No | ✅ Native |
| GPU | Metal | CUDA only | Metal |
| Memory | 6-16GB | 8-24GB | 6-16GB |

**Winner for Mac M4**: llama.cpp 🏆

## Next Steps

1. Start llama.cpp server (see step 4)
2. Configure .env (see step 5)
3. Run benchmark: `npm run benchmark:quick`
4. Check results in `benchmark-results/`

## Resources

- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [Hugging Face GGUF Models](https://huggingface.co/bartowski)
- [Metal Performance Guide](https://github.com/ggerganov/llama.cpp/blob/master/docs/backend/Metal.md)

## License

MIT
