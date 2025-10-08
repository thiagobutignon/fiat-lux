/**
 * List all tensors in a GGUF model
 */

import { GGUFParser } from '../src/gguf-parser/domain/use-cases/gguf-parser';
import { NodeFileReader } from '../src/gguf-parser/data/use-cases/node-file-reader';

async function listTensors() {
  const modelPath = process.argv[2];

  if (!modelPath) {
    console.error('Usage: tsx scripts/list-model-tensors.ts <model-path>');
    process.exit(1);
  }

  const fileReader = new NodeFileReader();
  const parser = new GGUFParser(fileReader);

  console.log('Parsing model...\n');
  const model = await parser.parse(modelPath);

  console.log(`Model: ${model.architecture.name}`);
  console.log(`Total tensors: ${model.tensors.length}\n`);

  // Group tensors by pattern
  const normTensors = model.tensors.filter((t) => t.name.includes('norm'));
  const attnTensors = model.tensors.filter((t) => t.name.includes('attn'));
  const ffnTensors = model.tensors.filter((t) => t.name.includes('ffn'));

  console.log(`Norm tensors (${normTensors.length}):`);
  normTensors.slice(0, 10).forEach((t) => console.log(`  ${t.name}`));
  if (normTensors.length > 10) console.log(`  ... and ${normTensors.length - 10} more`);

  console.log(`\nAttention tensors (${attnTensors.length}):`);
  attnTensors.slice(0, 10).forEach((t) => console.log(`  ${t.name}`));
  if (attnTensors.length > 10) console.log(`  ... and ${attnTensors.length - 10} more`);

  console.log(`\nFFN tensors (${ffnTensors.length}):`);
  ffnTensors.slice(0, 10).forEach((t) => console.log(`  ${t.name}`));
  if (ffnTensors.length > 10) console.log(`  ... and ${ffnTensors.length - 10} more`);
}

listTensors().catch(console.error);
