# vLLM Setup Guide

## Why vLLM?

**Problem**: Ollama pode ser lento para high-throughput benchmarks
- Token generation: ~20-50 tokens/s
- Alto custo de memória
- Não otimizado para batch processing

**Solution**: vLLM
- **10-25x faster** throughput vs vanilla inference
- PagedAttention para otimizar memória GPU
- Continuous batching
- OpenAI-compatible API

## Performance Comparison

| Method | Tokens/s | Latency | Memory |
|--------|----------|---------|--------|
| Ollama | 20-50 | Alto | Alto |
| vLLM | 200-500+ | Baixo | Otimizado |
| vLLM (batch) | 1000+ | Médio | Muito Otimizado |

## Installation

### Requirements
- NVIDIA GPU (CUDA)
- Python 3.8+
- 16GB+ VRAM (for Llama 8B)
- 40GB+ VRAM (for Llama 70B)

### Install vLLM

```bash
# Via pip (recomendado)
pip install vllm

# Via conda
conda install -c pytorch -c nvidia vllm
```

### Download Model

```bash
# vLLM usa Hugging Face models
# Exemplo: Llama 3.1 8B
# Não precisa baixar manualmente, vLLM baixa automaticamente
```

## Running vLLM Server

### Start OpenAI-Compatible API

```bash
# Llama 3.1 8B (requer ~16GB VRAM)
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8000 \
  --tensor-parallel-size 1

# Llama 3.1 70B (requer ~40GB VRAM, multi-GPU)
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct \
  --port 8000 \
  --tensor-parallel-size 4

# Gemma 2 9B (alternativa menor)
python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-2-9b-it \
  --port 8000
```

### Performance Tuning

```bash
# Max throughput configuration
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8000 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 256 \
  --dtype half
```

**Flags Explanation**:
- `--max-model-len`: Reduce para aumentar throughput
- `--gpu-memory-utilization`: Use mais GPU (0.90-0.95)
- `--max-num-seqs`: Batch size (maior = mais throughput)
- `--dtype half`: FP16 (2x faster, menos preciso)

## Test vLLM API

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "prompt": "What is 2+2?",
    "max_tokens": 100
  }'
```

## Integration with Benchmark

vLLM exposes OpenAI-compatible API, so we can use it as a drop-in replacement for Ollama.

### Option 1: Direct vLLM Integration (Fastest)

Create new detector that uses vLLM directly via OpenAI-compatible API.

### Option 2: Use Existing Ollama Detector

Point `OLLAMA_BASE_URL` to vLLM server:

```bash
# In .env
ENABLE_LOCAL_LLAMA=true
OLLAMA_BASE_URL=http://localhost:8000/v1
OLLAMA_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
```

**Note**: vLLM API é ligeiramente diferente de Ollama, então pode precisar adapter.

## Recommended Setup

### For Benchmark (Best Performance)

```bash
# 1. Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8000 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 128

# 2. Run benchmark
cd landing
npm run benchmark:quick
```

### For Development (Balanced)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8000
```

## Performance Expectations

### Llama 3.1 8B on vLLM

**Ollama**:
- Latency: ~1-3s per request
- Throughput: 20-50 tokens/s
- 100 requests: ~150-300s

**vLLM**:
- Latency: ~200-500ms per request
- Throughput: 200-500 tokens/s
- 100 requests: ~20-50s

**Speed improvement**: 3-6x faster!

### Batch Performance

vLLM excels at batch processing:

```python
# 100 sequential requests: ~50s
# 100 batched requests: ~5s (10x faster!)
```

## Troubleshooting

### Out of Memory

```bash
# Reduce max-model-len
--max-model-len 1024  # ou 512

# Reduce batch size
--max-num-seqs 64

# Use quantization
--quantization awq  # ou gptq
```

### Model Not Found

```bash
# Manually download from Hugging Face
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct
```

### Slow First Request

vLLM compila kernels na primeira request. Isso é normal e acontece apenas uma vez.

## Alternative Models

### Smaller Models (< 16GB VRAM)

```bash
# Phi-3 Mini (requer ~8GB)
--model microsoft/Phi-3-mini-4k-instruct

# Gemma 2B (requer ~4GB)
--model google/gemma-2b-it

# Llama 3.2 3B (requer ~6GB)
--model meta-llama/Llama-3.2-3B-Instruct
```

### Quantized Models (Even Faster)

```bash
# AWQ quantized (2-4x faster, similar accuracy)
--model TheBloke/Llama-2-7B-Chat-AWQ \
--quantization awq

# GPTQ quantized
--model TheBloke/Llama-2-7B-Chat-GPTQ \
--quantization gptq
```

## Monitoring

vLLM exposes metrics:

```bash
curl http://localhost:8000/metrics
```

**Key metrics**:
- `vllm:num_requests_running`: Active requests
- `vllm:num_requests_waiting`: Queue depth
- `vllm:gpu_cache_usage_perc`: GPU memory usage
- `vllm:avg_generation_throughput_toks_per_s`: Throughput

## Next Steps

1. Install vLLM: `pip install vllm`
2. Start server: See "Running vLLM Server" above
3. Create vLLM detector: See integration guide below
4. Run benchmark: `npm run benchmark:quick`

## Integration Options

We can integrate vLLM in 3 ways:

1. **VllmPatternDetector** (New) - Direct vLLM integration (fastest)
2. **Modify LocalLlamaDetector** - Point to vLLM API
3. **Hybrid** - Support both Ollama and vLLM

Which approach do you prefer?
