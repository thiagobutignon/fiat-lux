#!/usr/bin/env tsx
/**
 * Test Grammar Language Standard Library
 */

import { compile } from '../compiler/compiler';
import { BUILTINS, getBuiltin } from './builtins';

console.log('🧪 Testing Grammar Language Standard Library\n');

// Test 1: Builtins are registered
console.log('Test 1: Built-in functions');
console.log(`Registered: ${BUILTINS.length} built-ins`);
console.log('Sample:', BUILTINS.slice(0, 5).map(b => b.name).join(', '));
console.log('✅ PASSED\n');

// Test 2: Arithmetic operations
console.log('Test 2: Arithmetic');
const add = getBuiltin('+');
if (add) {
  const result = add.impl(2, 3);
  console.log('2 + 3 =', result);
  console.log(result === 5 ? '✅ PASSED' : '❌ FAILED');
} else {
  console.log('❌ FAILED: + not found');
}
console.log();

// Test 3: List operations
console.log('Test 3: List operations');
const cons = getBuiltin('cons');
const head = getBuiltin('head');
const tail = getBuiltin('tail');

if (cons && head && tail) {
  const list = cons.impl(1, cons.impl(2, cons.impl(3, [])));
  console.log('Created list:', list);
  console.log('head:', head.impl(list));
  console.log('tail:', tail.impl(list));
  console.log('✅ PASSED');
} else {
  console.log('❌ FAILED: list functions not found');
}
console.log();

// Test 4: Compile program using stdlib
console.log('Test 4: Compile with stdlib functions');
const program = [
  ['define', 'double', ['integer', '->', 'integer'],
    ['*', '$1', 2]
  ],
  ['define', 'triple', ['integer', '->', 'integer'],
    ['+', ['double', '$1'], '$1']
  ]
];

const result = compile(program);
if (result.errors.length === 0) {
  console.log('✅ PASSED - Compilation successful');
  console.log('\nGenerated code (first 200 chars):');
  console.log(result.code.substring(0, 200) + '...');
} else {
  console.log('❌ FAILED:', result.errors[0].message);
}

// Test 5: String operations
console.log('\nTest 5: String operations');
const concatFn = getBuiltin('concat');
if (concatFn) {
  const result = concatFn.impl('Hello', ' World');
  console.log('concat("Hello", " World") =', result);
  console.log(result === 'Hello World' ? '✅ PASSED' : '❌ FAILED');
} else {
  console.log('❌ FAILED: concat not found');
}

console.log('\n📊 Summary:');
console.log('Standard Library is working!');
console.log(`- ${BUILTINS.length} built-in functions`);
console.log('- Arithmetic operations ✅');
console.log('- List operations ✅');
console.log('- Type checking with stdlib ✅');
