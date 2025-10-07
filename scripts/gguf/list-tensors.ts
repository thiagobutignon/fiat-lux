#!/usr/bin/env tsx

import { analyzeGGUF } from '../../src/gguf-parser/presentation';
import { GGMLType } from '../../src/gguf-parser/domain/entities/gguf-metadata';

async function main() {
  const filePath = process.argv[2];

  if (!filePath) {
    console.error('Usage: tsx scripts/gguf/list-tensors.ts <model.gguf>');
    process.exit(1);
  }

  const { model } = await analyzeGGUF(filePath);

  console.log('📋 Tensor List\n');
  console.log(`Total: ${model.tensors.length} tensors\n`);

  // Group by type
  const byType = new Map<number, typeof model.tensors>();

  for (const tensor of model.tensors) {
    if (!byType.has(tensor.type)) {
      byType.set(tensor.type, []);
    }
    byType.get(tensor.type)!.push(tensor);
  }

  for (const [type, tensors] of byType.entries()) {
    console.log(`\n${'='.repeat(80)}`);
    console.log(`Type: ${GGMLType[type]} (${type})`);
    console.log(`Count: ${tensors.length} tensors`);
    console.log('='.repeat(80));

    for (const tensor of tensors.slice(0, 5)) {
      console.log(`  ${tensor.name}`);
      console.log(`    Shape: ${tensor.dimensions.join(' × ')}`);
      console.log(`    Elements: ${tensor.dimensions.reduce((a, b) => a * b, 1).toLocaleString()}`);
    }

    if (tensors.length > 5) {
      console.log(`  ... and ${tensors.length - 5} more`);
    }
  }
}

main();
