#!/usr/bin/env python3
"""
Phase 1 Findings Visualization

Generates plots for key findings from hallucination research Phase 1.
"""

import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_weight_profile(path):
    """Load weight profile JSON"""
    with open(path, 'r') as f:
        return json.load(f)

def extract_layer_data(profile):
    """Extract data from layers"""
    data = {
        'layers': [],
        'v_sparsity': [],
        'v_l2': [],
        'k_l2': [],
        'q_l2': [],
        'o_l2': [],
        'gate_l2': [],
        'attn_norm_mean': [],
        'ffn_norm_mean': [],
    }

    for layer in profile['layers']:
        idx = layer['layerIndex']

        # Get attention data
        if layer['layerType'] == 'attention' and layer.get('attention'):
            attn = layer['attention']
            data['layers'].append(idx)
            data['v_sparsity'].append(attn['value']['sparsity'] * 100)  # Convert to %
            data['v_l2'].append(attn['value']['l2Norm'])
            data['k_l2'].append(attn['key']['l2Norm'])
            data['q_l2'].append(attn['query']['l2Norm'])
            data['o_l2'].append(attn['output']['l2Norm'])

        # Get FFN gate data
        if layer['layerType'] == 'ffn' and layer.get('ffn'):
            ffn = layer['ffn']
            data['gate_l2'].append(ffn['gate']['l2Norm'])

        # Get norm data
        if layer['layerType'] == 'norm' and layer.get('norm'):
            norm = layer['norm']
            data['attn_norm_mean'].append(norm['attnNorm']['mean'])
            data['ffn_norm_mean'].append(norm['ffnNorm']['mean'])

    return data

def plot_findings(data, output_dir):
    """Generate all visualization plots"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = np.array(data['layers'])

    # ==========================================
    # Plot 1: Value Tensor Bimodal Sparsity
    # ==========================================
    plt.figure(figsize=(12, 6))
    plt.plot(layers, data['v_sparsity'], 'o-', linewidth=2, markersize=6, color='#e74c3c')
    plt.axhline(y=np.mean(data['v_sparsity']), color='gray', linestyle='--', alpha=0.5, label=f'Mean: {np.mean(data['v_sparsity']):.2f}%')
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Value Sparsity (%)', fontsize=12)
    plt.title('Discovery #1: Value Tensor Bimodal Sparsity Pattern\n(Alternating High/Low Creates Information Bottlenecks)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'finding1_value_sparsity.png', dpi=300)
    print(f"✓ Saved: {output_dir / 'finding1_value_sparsity.png'}")
    plt.close()

    # ==========================================
    # Plot 2: Progressive Attention Weakening
    # ==========================================
    q_gate_ratio = np.array(data['q_l2']) / np.array(data['gate_l2'])

    plt.figure(figsize=(12, 6))
    plt.plot(layers, q_gate_ratio, 'o-', linewidth=2, markersize=6, color='#3498db')
    plt.axhline(y=q_gate_ratio[0], color='green', linestyle='--', alpha=0.5, label=f'Layer 0: {q_gate_ratio[0]:.2f}')
    plt.axhline(y=q_gate_ratio[-1], color='red', linestyle='--', alpha=0.5, label=f'Layer 31: {q_gate_ratio[-1]:.2f}')
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Q/Gate Ratio (Attention Strength / FFN Strength)', fontsize=12)
    plt.title('Discovery #2: Progressive Attention Weakening\n(37% Decline from 0.79 → 0.50)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'finding2_attention_weakening.png', dpi=300)
    print(f"✓ Saved: {output_dir / 'finding2_attention_weakening.png'}")
    plt.close()

    # ==========================================
    # Plot 3: Value Amplification + Key Deterioration
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Value amplification
    ax1.plot(layers, data['v_l2'], 'o-', linewidth=2, markersize=6, color='#e74c3c', label='V L2 Norm')
    ax1.axhline(y=data['v_l2'][0], color='green', linestyle='--', alpha=0.5, label=f'Layer 0: {data["v_l2"][0]:.1f}')
    ax1.axhline(y=data['v_l2'][-1], color='red', linestyle='--', alpha=0.5, label=f'Layer 31: {data["v_l2"][-1]:.1f} (+134%!)')
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('Value L2 Norm', fontsize=12)
    ax1.set_title('Discovery #3a: Value Amplification (2.3x increase)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Key matching deterioration
    ax2.plot(layers, data['k_l2'], 'o-', linewidth=2, markersize=6, color='#9b59b6', label='K L2 Norm')
    ax2.axhline(y=data['k_l2'][0], color='green', linestyle='--', alpha=0.5, label=f'Layer 0: {data["k_l2"][0]:.1f}')
    ax2.axhline(y=data['k_l2'][-1], color='red', linestyle='--', alpha=0.5, label=f'Layer 31: {data["k_l2"][-1]:.1f} (-14%)')
    ax2.set_xlabel('Layer Index', fontsize=12)
    ax2.set_ylabel('Key L2 Norm', fontsize=12)
    ax2.set_title('Discovery #3b: Key Matching Deterioration', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'finding3_value_key_dynamics.png', dpi=300)
    print(f"✓ Saved: {output_dir / 'finding3_value_key_dynamics.png'}")
    plt.close()

    # ==========================================
    # Plot 4: Layer Norm Amplification Trends
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Attention norm
    ax1.plot(layers, data['attn_norm_mean'], 'o-', linewidth=2, markersize=6, color='#2ecc71', label='Attn Norm Mean')
    ax1.axhline(y=data['attn_norm_mean'][0], color='green', linestyle='--', alpha=0.5, label=f'Layer 0: {data["attn_norm_mean"][0]:.3f}')
    ax1.axhline(y=data['attn_norm_mean'][-1], color='red', linestyle='--', alpha=0.5, label=f'Layer 31: {data["attn_norm_mean"][-1]:.3f}')
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('Attention Norm Scale', fontsize=12)
    ax1.set_title('Discovery #4a: Attention Norm Amplification in Late Layers', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # FFN norm
    ax2.plot(layers, data['ffn_norm_mean'], 'o-', linewidth=2, markersize=6, color='#f39c12', label='FFN Norm Mean')
    ax2.axhline(y=data['ffn_norm_mean'][0], color='green', linestyle='--', alpha=0.5, label=f'Layer 0: {data["ffn_norm_mean"][0]:.3f}')
    ax2.axhline(y=data['ffn_norm_mean'][-1], color='red', linestyle='--', alpha=0.5, label=f'Layer 31: {data["ffn_norm_mean"][-1]:.3f} (+30.7%!)')
    ax2.axhline(y=np.mean(data['ffn_norm_mean']), color='gray', linestyle=':', alpha=0.5, label=f'Mean: {np.mean(data["ffn_norm_mean"]):.3f}')
    ax2.set_xlabel('Layer Index', fontsize=12)
    ax2.set_ylabel('FFN Norm Scale', fontsize=12)
    ax2.set_title('Discovery #4b: FFN Norm Amplification (Layer 31 = 30.7% above avg!)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'finding4_layernorm_amplification.png', dpi=300)
    print(f"✓ Saved: {output_dir / 'finding4_layernorm_amplification.png'}")
    plt.close()

    # ==========================================
    # Plot 5: Layer 31 Perfect Storm Summary
    # ==========================================
    fig, ax = plt.subplots(figsize=(14, 8))

    metrics = [
        'Weak K\nMatching',
        'Sparse V\n(2.79%)',
        'Amplified V\n(+134%)',
        'Strong O\nProjection',
        'Dominant\nFFN Gate',
        'High FFN\nNorm Scale\n(+30.7%)',
    ]

    values = [
        data['k_l2'][-1] / max(data['k_l2']),  # K norm relative to max
        (3 - data['v_sparsity'][-1]) / 3,  # Inverted sparsity (higher = worse)
        data['v_l2'][-1] / data['v_l2'][0],  # V amplification ratio
        data['o_l2'][-1] / max(data['o_l2']),  # O norm relative to max
        data['gate_l2'][-1] / max(data['gate_l2']),  # Gate norm relative to max
        data['ffn_norm_mean'][-1] / np.mean(data['ffn_norm_mean']),  # FFN norm vs avg
    ]

    colors = ['#e74c3c', '#e67e22', '#f39c12', '#3498db', '#9b59b6', '#e74c3c']

    bars = ax.barh(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_xlabel('Relative Magnitude (Normalized)', fontsize=12)
    ax.set_title('Discovery #5: Layer 31 "Perfect Storm" for Hallucinations\n(All Risk Factors Converge)', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Baseline (1.0)')
    ax.legend()

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax.text(value + 0.02, bar.get_y() + bar.get_height()/2,
                f'{value:.2f}',
                va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'finding5_layer31_perfect_storm.png', dpi=300)
    print(f"✓ Saved: {output_dir / 'finding5_layer31_perfect_storm.png'}")
    plt.close()

    print(f"\n✅ All visualizations generated in: {output_dir}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize-phase1-findings.py <weight-profile.json>")
        sys.exit(1)

    profile_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "research-output/phase1/visualizations"

    print(f"📊 Loading weight profile: {profile_path}\n")

    profile = load_weight_profile(profile_path)
    print(f"Model: {profile['modelName']}")
    print(f"Total Layers: {len(profile['layers'])}\n")

    print("Extracting layer data...")
    data = extract_layer_data(profile)
    print(f"✓ Extracted data for {len(data['layers'])} layers\n")

    print("Generating visualizations...\n")
    plot_findings(data, output_dir)

    print("\n🎉 Visualization complete!")

if __name__ == '__main__':
    main()
