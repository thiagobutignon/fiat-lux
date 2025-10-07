#!/bin/bash
# Start vLLM server for benchmarking
# Usage: ./scripts/start-vllm.sh [model] [port]

MODEL=${1:-"meta-llama/Meta-Llama-3.1-8B-Instruct"}
PORT=${2:-8000}

echo "Starting vLLM server..."
echo "Model: $MODEL"
echo "Port: $PORT"
echo ""

# Check if vLLM is installed
if ! python -c "import vllm" 2>/dev/null; then
    echo "❌ vLLM not installed!"
    echo "Install with: pip install vllm"
    exit 1
fi

echo "✓ vLLM installed"
echo ""

# Check GPU availability
if ! nvidia-smi &>/dev/null; then
    echo "⚠️  Warning: nvidia-smi not found. Make sure you have NVIDIA GPU with CUDA."
fi

echo "Starting server (this may take 1-2 minutes)..."
echo ""

# Start vLLM with optimized settings for benchmarking
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$PORT" \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 128 \
  --dtype half

echo ""
echo "vLLM server stopped"
