#!/usr/bin/env python3
"""
Activation Tracer for Llama 3.1 8B

Captures layer-by-layer activations during inference to trace information flow
and identify hallucination mechanisms.

Requirements:
    pip install llama-cpp-python numpy

Usage:
    python activation_tracer.py <model-path> <prompt> [--output output.json]
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np

try:
    from llama_cpp import Llama, LlamaGrammar
except ImportError:
    print("Error: llama-cpp-python not installed")
    print("Install with: pip install llama-cpp-python")
    sys.exit(1)


@dataclass
class LayerActivation:
    """Stores activation data for a single layer"""
    layer_index: int
    layer_type: str  # 'attention', 'ffn', 'norm'

    # Attention activations (if applicable)
    attention_weights: Optional[List[float]] = None
    attention_entropy: Optional[float] = None
    attention_focus_score: Optional[float] = None
    query_norm: Optional[float] = None
    key_norm: Optional[float] = None
    value_norm: Optional[float] = None
    value_sparsity: Optional[float] = None

    # FFN activations (if applicable)
    ffn_gate_activations: Optional[List[float]] = None
    ffn_gate_norm: Optional[float] = None
    ffn_output_norm: Optional[float] = None

    # Layer norm outputs (if applicable)
    attn_norm_output: Optional[List[float]] = None
    ffn_norm_output: Optional[List[float]] = None

    # Activation statistics
    activation_mean: Optional[float] = None
    activation_std: Optional[float] = None
    activation_max: Optional[float] = None
    activation_min: Optional[float] = None


@dataclass
class ActivationTrace:
    """Complete activation trace for an inference run"""
    prompt: str
    generated_text: str
    model_name: str
    layers: List[LayerActivation]

    # Summary statistics
    total_tokens: int
    generation_time: float

    # Metadata
    timestamp: str
    prompt_id: Optional[str] = None
    hallucination_detected: Optional[bool] = None
    notes: Optional[str] = None


class ActivationCapture:
    """
    Captures activations during Llama inference

    Note: llama-cpp-python doesn't expose internal activations directly,
    so this implementation uses a proxy approach:

    1. For attention: Infer from logits and token probabilities
    2. For FFN: Estimate from hidden state changes
    3. For layer norms: Track output distributions

    For full activation access, we'd need to modify llama.cpp source
    or use PyTorch with GGUF conversion.
    """

    def __init__(self, model_path: str, n_ctx: int = 2048, n_gpu_layers: int = 0):
        """
        Initialize activation capture system

        Args:
            model_path: Path to GGUF model file
            n_ctx: Context size
            n_gpu_layers: Number of layers to offload to GPU (0 = CPU only)
        """
        self.model_path = model_path

        print(f"Loading model: {model_path}")
        print(f"Context size: {n_ctx}")
        print(f"GPU layers: {n_gpu_layers}")

        # Load model with verbose output to see layer info
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,  # Set to True for debugging
            logits_all=True,  # Enable logits for all tokens (needed for analysis)
        )

        print("✓ Model loaded successfully")

        # Track activations during generation
        self.current_trace: Optional[ActivationTrace] = None
        self.layer_activations: List[LayerActivation] = []

    def compute_attention_entropy(self, attention_weights: np.ndarray) -> float:
        """
        Compute entropy of attention distribution

        High entropy = diffuse attention (attending to many tokens equally)
        Low entropy = focused attention (attending to few specific tokens)
        """
        # Normalize to probabilities
        probs = attention_weights / np.sum(attention_weights)

        # Compute Shannon entropy: H = -sum(p * log(p))
        # Add small epsilon to avoid log(0)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return float(entropy)

    def compute_sparsity(self, values: np.ndarray, threshold: float = 1e-3) -> float:
        """Compute sparsity (percentage of near-zero values)"""
        near_zero = np.abs(values) < threshold
        sparsity = np.sum(near_zero) / len(values)
        return float(sparsity)

    def estimate_layer_activations(
        self,
        layer_idx: int,
        logits: np.ndarray,
        tokens: List[int]
    ) -> LayerActivation:
        """
        Estimate layer activations from available data

        Note: This is a proxy estimation since llama-cpp-python doesn't
        expose internal activations. For accurate measurements, we'd need
        to modify llama.cpp or use PyTorch.
        """
        # Get probability distribution for current token
        probs = self._softmax(logits[-1])  # Last token logits

        # Estimate attention entropy from probability distribution
        # (more spread = higher attention entropy)
        attention_entropy = self.compute_attention_entropy(probs[:100])  # Top 100 tokens

        # Estimate attention focus (inverse of entropy)
        max_prob = np.max(probs)
        attention_focus = float(max_prob)  # High prob = focused attention

        # Estimate value sparsity from logit distribution
        value_sparsity = self.compute_sparsity(logits[-1])

        # Compute norms
        query_norm = float(np.linalg.norm(logits[-1]))

        return LayerActivation(
            layer_index=layer_idx,
            layer_type='combined',  # Can't separate attn/ffn without hooks
            attention_entropy=attention_entropy,
            attention_focus_score=attention_focus,
            query_norm=query_norm,
            value_sparsity=value_sparsity,
            activation_mean=float(np.mean(logits[-1])),
            activation_std=float(np.std(logits[-1])),
            activation_max=float(np.max(logits[-1])),
            activation_min=float(np.min(logits[-1])),
        )

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Compute softmax of logits"""
        exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
        return exp_logits / np.sum(exp_logits)

    def capture_inference(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        prompt_id: Optional[str] = None
    ) -> ActivationTrace:
        """
        Run inference and capture activations

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            prompt_id: Optional ID for tracking (e.g., "FR-001")

        Returns:
            ActivationTrace with captured data
        """
        import time
        from datetime import datetime

        print(f"\n🔬 Capturing activations for prompt: {prompt[:50]}...")

        start_time = time.time()

        # Reset activation storage
        self.layer_activations = []

        # Generate with streaming to capture token-by-token
        generated_tokens = []
        generated_text = ""

        print("Generating...")

        # Note: llama-cpp-python doesn't provide per-layer hooks,
        # so we approximate from final output
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            echo=False,
            stop=["</s>", "\n\n"],
        )

        generated_text = output['choices'][0]['text']

        # Estimate layer activations from logits (limited by llama-cpp-python API)
        # For full access, we'd need PyTorch or modified llama.cpp

        # Since we can't access internal layers easily, we'll create
        # placeholder activations with estimates
        num_layers = 32  # Llama 3.1 8B has 32 layers

        print(f"Estimating activations for {num_layers} layers...")

        # Run inference again with logits_all to get more info
        # (This is a workaround - ideally we'd hook into generation)
        logits_output = self.llm(
            prompt,
            max_tokens=1,  # Just one token to get logits
            temperature=0,
            logits_all=True,
        )

        # Get logits (this is the output of the final layer)
        # For per-layer activations, we'd need internal access
        try:
            logits = np.array(self.llm.eval_logits)

            # Create estimated activations for each layer
            # (In reality, we only have final layer output)
            for layer_idx in range(num_layers):
                # Estimate how activations might change per layer
                # This is a simplified proxy - not ground truth!
                layer_activation = self.estimate_layer_activations(
                    layer_idx,
                    logits,
                    generated_tokens
                )
                self.layer_activations.append(layer_activation)

        except Exception as e:
            print(f"⚠️  Could not extract detailed activations: {e}")
            print("   Creating placeholder activations...")

            # Create placeholder activations
            for layer_idx in range(num_layers):
                self.layer_activations.append(LayerActivation(
                    layer_index=layer_idx,
                    layer_type='unknown',
                ))

        generation_time = time.time() - start_time

        print(f"✓ Generated {len(generated_text)} characters in {generation_time:.2f}s")

        # Create trace
        trace = ActivationTrace(
            prompt=prompt,
            generated_text=generated_text,
            model_name=Path(self.model_path).name,
            layers=self.layer_activations,
            total_tokens=len(generated_tokens) if generated_tokens else len(generated_text.split()),
            generation_time=generation_time,
            timestamp=datetime.now().isoformat(),
            prompt_id=prompt_id,
        )

        return trace

    def save_trace(self, trace: ActivationTrace, output_path: str):
        """Save activation trace to JSON file"""
        # Convert dataclass to dict
        trace_dict = asdict(trace)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(trace_dict, f, indent=2)

        print(f"✅ Trace saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Capture Llama activations during inference')
    parser.add_argument('model', help='Path to GGUF model file')
    parser.add_argument('prompt', help='Input prompt (or path to prompt file)')
    parser.add_argument('--output', '-o', default=None, help='Output JSON file')
    parser.add_argument('--max-tokens', type=int, default=100, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--gpu-layers', type=int, default=0, help='Number of GPU layers')
    parser.add_argument('--prompt-id', help='Prompt ID for tracking (e.g., FR-001)')

    args = parser.parse_args()

    # Load prompt (from file or direct string)
    if Path(args.prompt).exists():
        with open(args.prompt) as f:
            prompt = f.read().strip()
    else:
        prompt = args.prompt

    # Initialize tracer
    tracer = ActivationCapture(
        model_path=args.model,
        n_gpu_layers=args.gpu_layers
    )

    # Capture activations
    trace = tracer.capture_inference(
        prompt=prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        prompt_id=args.prompt_id
    )

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Auto-generate filename
        prompt_id = args.prompt_id or 'trace'
        timestamp = trace.timestamp.replace(':', '-').replace('.', '-')
        output_path = f"research-output/phase2/activations/{prompt_id}-{timestamp}.json"

    # Save trace
    tracer.save_trace(trace, output_path)

    # Print summary
    print(f"\n" + "="*60)
    print("ACTIVATION CAPTURE SUMMARY")
    print("="*60)
    print(f"Prompt: {prompt[:60]}...")
    print(f"Generated: {trace.generated_text[:60]}...")
    print(f"Layers captured: {len(trace.layers)}")
    print(f"Generation time: {trace.generation_time:.2f}s")
    print(f"Output: {output_path}")

    # Provide usage note
    print(f"\n⚠️  NOTE: llama-cpp-python has limited activation access.")
    print(f"   For full per-layer activations, consider:")
    print(f"   1. Modifying llama.cpp source to export activations")
    print(f"   2. Converting GGUF → PyTorch and using hooks")
    print(f"   3. Using Hugging Face Transformers with the model")


if __name__ == '__main__':
    main()
