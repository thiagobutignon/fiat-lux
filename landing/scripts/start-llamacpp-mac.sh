#!/bin/bash
# Start llama.cpp server on Mac M-series (M1/M2/M3/M4)
# Usage: ./scripts/start-llamacpp-mac.sh [model_name] [port]

MODEL=${1:-"Meta-Llama-3.1-8B-Instruct-Q4_K_M"}
PORT=${2:-8080}
MODEL_PATH="./models/${MODEL}.gguf"

echo "Starting llama.cpp server for Mac (Metal acceleration)..."
echo "Model: $MODEL"
echo "Port: $PORT"
echo ""

# Check if llama-server is installed
if ! command -v llama-server &> /dev/null; then
    echo "❌ llama-server not found!"
    echo ""
    echo "Install with Homebrew:"
    echo "  brew install llama.cpp"
    echo ""
    echo "Or build from source:"
    echo "  git clone https://github.com/ggerganov/llama.cpp"
    echo "  cd llama.cpp"
    echo "  make LLAMA_METAL=1"
    exit 1
fi

echo "✓ llama-server installed"
echo ""

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model not found: $MODEL_PATH"
    echo ""
    echo "Download with:"
    echo "  huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \\"
    echo "    ${MODEL}.gguf \\"
    echo "    --local-dir ./models \\"
    echo "    --local-dir-use-symlinks False"
    echo ""
    echo "Available quantization levels:"
    echo "  - Q4_K_M (recommended): Fast, 6GB RAM, excellent quality"
    echo "  - Q8_0: Slower, 9GB RAM, best quality"
    echo "  - F16: Slowest, 16GB RAM, perfect quality"
    exit 1
fi

echo "✓ Model found: $MODEL_PATH"
MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
echo "  Size: $MODEL_SIZE"
echo ""

# Check system info
echo "System Information:"
echo "  Chip: $(sysctl -n machdep.cpu.brand_string)"
echo "  RAM: $(sysctl -n hw.memsize | awk '{print $1/1024/1024/1024 " GB"}')"
echo "  Cores: $(sysctl -n hw.ncpu)"
echo ""

# Check if port is available
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Port $PORT is already in use!"
    echo "   Kill existing process with: kill -9 \$(lsof -t -i :$PORT)"
    echo "   Or use a different port: ./scripts/start-llamacpp-mac.sh $MODEL 8081"
    exit 1
fi

echo "✓ Port $PORT available"
echo ""

# Determine optimal thread count (80% of cores)
CORES=$(sysctl -n hw.ncpu)
THREADS=$((CORES * 80 / 100))
if [ $THREADS -lt 4 ]; then
    THREADS=4
fi

echo "Starting server with optimized settings..."
echo "  GPU Layers: 999 (all layers offloaded to Metal)"
echo "  Context Size: 2048"
echo "  CPU Threads: $THREADS"
echo ""
echo "Server will be ready when you see: 'llama server listening at...'"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start llama.cpp server with Metal acceleration
llama-server \
  -m "$MODEL_PATH" \
  -ngl 999 \
  -c 2048 \
  -t $THREADS \
  --port $PORT \
  --host 0.0.0.0 \
  --numa distribute

echo ""
echo "llama.cpp server stopped"
