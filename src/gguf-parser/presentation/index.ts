/**
 * GGUF Parser Public API
 * Entry point for GGUF model analysis
 */

import { GGUFParser } from '../domain/use-cases/gguf-parser';
import { TransformerAnalyzer, TransformerAnalysis } from '../domain/use-cases/transformer-analyzer';
import { NodeFileReader } from '../data/use-cases/node-file-reader';
import { GGUFModel } from '../domain/entities/gguf-metadata';

/**
 * Analyze a GGUF model file
 */
export async function analyzeGGUF(filePath: string): Promise<{
  model: GGUFModel;
  analysis: TransformerAnalysis;
}> {
  const fileReader = new NodeFileReader();
  const parser = new GGUFParser(fileReader);
  const analyzer = new TransformerAnalyzer();

  // Parse GGUF file
  const model = await parser.parse(filePath);

  // Get file size for memory calculations
  const fileSize = await fileReader.getFileSize(filePath);

  // Analyze transformer architecture
  const analysis = analyzer.analyze(model, fileSize);

  return {
    model,
    analysis,
  };
}

/**
 * Format analysis results for display
 */
export function formatAnalysis(analysis: TransformerAnalysis): string {
  let output = '';

  output += '═'.repeat(80) + '\n';
  output += `🤖 ${analysis.modelName}\n`;
  output += '═'.repeat(80) + '\n\n';

  // Model Overview
  output += '📊 MODEL OVERVIEW\n';
  output += '─'.repeat(80) + '\n';
  output += `Architecture:        ${analysis.architecture}\n`;
  output += `Total Parameters:    ${analysis.totalParameters}\n`;
  output += `Quantization:        ${analysis.quantization}\n`;
  output += `File Size:           ${analysis.fileSizeGB.toFixed(2)} GB\n`;
  output += `Memory Usage (Est):  ${analysis.memoryUsageGB.toFixed(2)} GB\n`;
  output += `KV Cache (Max):      ${analysis.kvCacheGB.toFixed(2)} GB\n\n`;

  // Transformer Architecture
  output += '🧠 TRANSFORMER ARCHITECTURE\n';
  output += '─'.repeat(80) + '\n';
  output += `Layers:              ${analysis.layers}\n`;
  output += `Attention Heads:     ${analysis.attentionHeads}\n`;
  if (analysis.attentionHeadsKV !== null) {
    output += `Attention Heads (KV):${analysis.attentionHeadsKV} ${analysis.hasGroupedQueryAttention ? '(GQA)' : ''}\n`;
  }
  output += `Embedding Dimension: ${analysis.embeddingDimension}\n`;
  output += `Vocab Size:          ${analysis.vocabSize.toLocaleString()}\n`;
  output += `Context Length:      ${analysis.contextLength.toLocaleString()}\n`;
  if (analysis.ffnDimension) {
    output += `FFN Dimension:       ${analysis.ffnDimension.toLocaleString()}\n`;
  }
  output += '\n';

  // Advanced Features
  output += '⚙️  ADVANCED FEATURES\n';
  output += '─'.repeat(80) + '\n';
  output += `Grouped-Query Attention: ${analysis.hasGroupedQueryAttention ? 'Yes' : 'No'}\n`;
  output += `RoPE:                    ${analysis.hasRoPE ? 'Yes' : 'No'}\n`;
  if (analysis.ropeFreqBase) {
    output += `RoPE Freq Base:          ${analysis.ropeFreqBase}\n`;
  }
  if (analysis.ropeScaling) {
    output += `RoPE Scaling:            ${analysis.ropeScaling}\n`;
  }
  output += '\n';

  // Special Tokens
  if (Object.keys(analysis.specialTokens).length > 0) {
    output += '🔤 SPECIAL TOKENS\n';
    output += '─'.repeat(80) + '\n';
    if (analysis.specialTokens.bos !== undefined) {
      output += `BOS (Beginning):     ${analysis.specialTokens.bos}\n`;
    }
    if (analysis.specialTokens.eos !== undefined) {
      output += `EOS (End):           ${analysis.specialTokens.eos}\n`;
    }
    if (analysis.specialTokens.pad !== undefined) {
      output += `PAD (Padding):       ${analysis.specialTokens.pad}\n`;
    }
    if (analysis.specialTokens.unk !== undefined) {
      output += `UNK (Unknown):       ${analysis.specialTokens.unk}\n`;
    }
    output += '\n';
  }

  // Tensor Breakdown by Type
  output += '🔢 TENSOR BREAKDOWN BY TYPE\n';
  output += '─'.repeat(80) + '\n';
  output += `${'Type'.padEnd(30)} ${'Count'.padStart(8)} ${'Parameters'.padStart(15)}\n`;
  output += '─'.repeat(80) + '\n';
  for (const group of analysis.tensorsByType) {
    output += `${group.type.padEnd(30)} ${group.count.toString().padStart(8)} ${group.totalParams.padStart(15)}\n`;
  }
  output += '\n';

  // Layer Analysis (first 3 and last 3 layers)
  if (analysis.tensorsByLayer.length > 0) {
    output += '📚 LAYER ANALYSIS\n';
    output += '─'.repeat(80) + '\n';

    const layersToShow = analysis.tensorsByLayer.length > 6
      ? [...analysis.tensorsByLayer.slice(0, 3), ...analysis.tensorsByLayer.slice(-3)]
      : analysis.tensorsByLayer;

    const showEllipsis = analysis.tensorsByLayer.length > 6;

    for (let i = 0; i < layersToShow.length; i++) {
      const layer = layersToShow[i];
      output += `Layer ${layer.layer}: ${layer.tensors.length} tensors, ${layer.totalParams} parameters\n`;

      if (showEllipsis && i === 2) {
        output += `... (${analysis.tensorsByLayer.length - 6} layers omitted) ...\n`;
      }
    }
    output += '\n';
  }

  output += '═'.repeat(80) + '\n';
  output += `✅ Analysis complete! Analyzed ${analysis.tensorsByType.reduce((sum, t) => sum + t.count, 0)} tensors.\n`;
  output += '═'.repeat(80) + '\n';

  return output;
}

// Re-export types
export * from '../domain/entities/gguf-metadata';
export * from '../domain/entities/tensor-shape';
export { TransformerAnalysis } from '../domain/use-cases/transformer-analyzer';
