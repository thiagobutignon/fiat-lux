#!/usr/bin/env tsx

/**
 * Quick test to verify Gemini API key works
 */

import 'dotenv/config';
import { GoogleGenerativeAI } from '@google/generative-ai';

async function testGemini() {
  const apiKey = process.env.GEMINI_API_KEY;

  if (!apiKey) {
    console.error('❌ GEMINI_API_KEY not found in .env file');
    process.exit(1);
  }

  console.log('Testing Gemini API key...\n');
  console.log(`API Key: ${apiKey.substring(0, 10)}...${apiKey.substring(apiKey.length - 4)}\n`);

  try {
    const client = new GoogleGenerativeAI(apiKey);
    const model = client.getGenerativeModel({ model: 'gemini-2.0-flash-exp' });

    console.log('Sending test request...');

    const result = await Promise.race([
      model.generateContent('Say "Hello" in JSON format: {"message": "..."}'),
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error('Timeout after 10s')), 10000)
      )
    ]);

    const response = result.response;
    const text = response.text();

    console.log('✅ Success! Response:\n');
    console.log(text);
    console.log('\n✅ Gemini API is working correctly!');

    process.exit(0);
  } catch (error) {
    console.error('\n❌ Error testing Gemini API:');
    console.error(error);
    console.log('\n💡 Possible issues:');
    console.log('   1. API key is invalid or revoked');
    console.log('   2. API key doesn\'t have access to gemini-2.0-flash-exp');
    console.log('   3. Network connectivity issues');
    console.log('   4. Rate limiting\n');
    console.log('Get a new API key from: https://aistudio.google.com/apikey\n');
    process.exit(1);
  }
}

testGemini();
