# LLM Adapter Tests

## Overview

This directory will contain unit and integration tests for the Anthropic Adapter once a testing framework is set up.

## Test Coverage Needed

### Unit Tests for AnthropicAdapter

**Priority 1: Core Functionality**
```typescript
describe('AnthropicAdapter', () => {
  describe('constructor', () => {
    it('should initialize with API key');
    it('should set total cost to 0');
    it('should set total requests to 0');
  });

  describe('invoke', () => {
    it('should validate empty system prompt');
    it('should validate empty query');
    it('should reject prompts exceeding MAX_PROMPT_CHARS');
    it('should make API call with correct parameters');
    it('should track cumulative costs');
    it('should increment request counter');
    it('should handle API errors gracefully');
  });

  describe('calculateCost', () => {
    it('should correctly calculate cost for Opus 4', () => {
      const adapter = new AnthropicAdapter('test-key');
      const cost = adapter['calculateCost'](
        { input_tokens: 1000, output_tokens: 2000 },
        'claude-opus-4'
      );
      // $15/1M input + $75/1M output
      // (1000/1M * 15) + (2000/1M * 75) = 0.015 + 0.15 = 0.165
      expect(cost).toBe(0.165);
    });

    it('should correctly calculate cost for Sonnet 4.5', () => {
      const adapter = new AnthropicAdapter('test-key');
      const cost = adapter['calculateCost'](
        { input_tokens: 1000, output_tokens: 2000 },
        'claude-sonnet-4-5'
      );
      // $3/1M input + $15/1M output
      // (1000/1M * 3) + (2000/1M * 15) = 0.003 + 0.03 = 0.033
      expect(cost).toBe(0.033);
    });
  });

  describe('estimateCost', () => {
    it('should estimate cost using AVG_CHARS_PER_TOKEN');
    it('should allow custom output token estimate');
    it('should return note explaining estimation');
  });

  describe('compareCosts', () => {
    it('should compare costs for all models');
    it('should show Sonnet is cheaper than Opus');
  });

  describe('getTotalCost', () => {
    it('should accumulate costs across multiple requests');
  });

  describe('getTotalRequests', () => {
    it('should count all requests');
  });

  describe('resetStats', () => {
    it('should reset cost and request counters to 0');
  });
});
```

**Priority 2: Streaming**
```typescript
describe('invokeStream', () => {
  it('should yield text chunks');
  it('should track tokens in usage return value');
  it('should handle stream errors');
});
```

**Priority 3: Error Handling**
```typescript
describe('error handling', () => {
  it('should transform Anthropic API errors');
  it('should include status code in error message');
  it('should preserve original error for debugging');
});
```

### Integration Tests for MetaAgent

```typescript
describe('MetaAgent with AnthropicAdapter', () => {
  describe('cost tracking integration', () => {
    it('should update RecursionState.cost_so_far');
    it('should stop when maxCostUSD limit reached');
    it('should report accurate total costs');
  });

  describe('JSON parsing fallback', () => {
    it('should log warning when parsing fails');
    it('should use fallback with low confidence');
    it('should include error message in reasoning');
  });

  describe('domain decomposition fallback', () => {
    it('should use first 2 domains on parse failure');
    it('should log fallback strategy reason');
  });
});
```

## Mocking Strategy

### Mock Anthropic SDK

```typescript
// __mocks__/@anthropic-ai/sdk.ts
export class Anthropic {
  messages = {
    create: jest.fn().mockResolvedValue({
      content: [{ type: 'text', text: 'Mock response' }],
      usage: { input_tokens: 10, output_tokens: 20 },
      stop_reason: 'end_turn',
    }),
    stream: jest.fn().mockImplementation(async function* () {
      yield { type: 'content_block_delta', delta: { type: 'text_delta', text: 'Mock' } };
      yield { type: 'message_start', message: { usage: { input_tokens: 10 } } };
      yield { type: 'message_delta', usage: { output_tokens: 20 } };
    }),
  };
}

export namespace Anthropic {
  export class APIError extends Error {
    status: number;
    constructor(message: string, status: number) {
      super(message);
      this.status = status;
    }
  }
}
```

### Usage in Tests

```typescript
import { AnthropicAdapter } from '../anthropic-adapter';

jest.mock('@anthropic-ai/sdk');

describe('AnthropicAdapter', () => {
  let adapter: AnthropicAdapter;

  beforeEach(() => {
    adapter = new AnthropicAdapter('test-key');
  });

  it('should make API call with correct parameters', async () => {
    await adapter.invoke('System prompt', 'Query', {
      model: 'claude-sonnet-4-5',
      max_tokens: 1000,
    });

    expect(Anthropic.prototype.messages.create).toHaveBeenCalledWith({
      model: 'claude-sonnet-4-5-20250929',
      max_tokens: 1000,
      temperature: 0.5,
      system: 'System prompt',
      messages: [{ role: 'user', content: 'Query' }],
    });
  });
});
```

## Test Framework Options

### Option 1: Jest (Recommended)

**Pros**:
- Most popular Node.js testing framework
- Built-in mocking
- Snapshot testing
- Good TypeScript support

**Setup**:
```bash
npm install --save-dev jest @types/jest ts-jest

# jest.config.js
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  testMatch: ['**/__tests__/**/*.test.ts'],
};
```

### Option 2: Vitest (Modern Alternative)

**Pros**:
- Faster than Jest
- Better ESM support
- Compatible with Jest API

**Setup**:
```bash
npm install --save-dev vitest

# vitest.config.ts
export default {
  test: {
    globals: true,
  },
};
```

### Option 3: Custom Lightweight (Current Project Style)

The project already has a custom test framework (see `scripts/test.ts`).

**Pros**:
- No dependencies
- Ultra-fast (<5ms)
- Consistent with existing tests

**Extension for LLM Tests**:
```typescript
// src/agi-recursive/llm/__tests__/anthropic-adapter.test.ts
import { describe, it, expect } from '../../../shared/test-framework';
import { AnthropicAdapter } from '../anthropic-adapter';

describe('AnthropicAdapter', () => {
  describe('calculateCost', () => {
    it('should calculate Opus 4 cost correctly', () => {
      const adapter = new AnthropicAdapter('test-key');
      const cost = adapter['calculateCost'](
        { input_tokens: 1000, output_tokens: 2000 },
        'claude-opus-4'
      );
      expect(cost).toBe(0.165);
    });
  });
});
```

## Running Tests

```bash
# Option 1: Jest
npm test

# Option 2: Vitest
npm run test:watch

# Option 3: Custom framework
tsx src/agi-recursive/llm/__tests__/anthropic-adapter.test.ts
```

## Coverage Goals

| Component | Target Coverage | Priority |
|-----------|----------------|----------|
| AnthropicAdapter | 90% | High |
| MetaAgent (adapter integration) | 80% | High |
| Cost calculation | 100% | Critical |
| Error handling | 90% | High |
| Streaming | 80% | Medium |

## Test Data

### Mock Responses

```typescript
// __tests__/fixtures/mock-responses.ts
export const MOCK_OPUS_RESPONSE = {
  content: [{ type: 'text' as const, text: 'Opus response' }],
  usage: { input_tokens: 100, output_tokens: 200 },
  stop_reason: 'end_turn',
};

export const MOCK_SONNET_RESPONSE = {
  content: [{ type: 'text' as const, text: 'Sonnet response' }],
  usage: { input_tokens: 50, output_tokens: 100 },
  stop_reason: 'end_turn',
};

export const MOCK_JSON_RESPONSE = {
  content: [{
    type: 'text' as const,
    text: JSON.stringify({
      answer: 'Test answer',
      concepts: ['test'],
      confidence: 0.9,
      reasoning: 'Test reasoning',
    }),
  }],
  usage: { input_tokens: 20, output_tokens: 50 },
  stop_reason: 'end_turn',
};
```

## Next Steps

1. **Choose Test Framework** - Recommend Jest for AGI components
2. **Set Up Mock Infrastructure** - Mock @anthropic-ai/sdk
3. **Write Core Tests** - Cost calculation, validation
4. **Add Integration Tests** - MetaAgent + Adapter
5. **Set Up CI** - Run tests on every PR
6. **Add Coverage Reporting** - Track test coverage

## Contributing

When adding tests:

1. Follow existing naming conventions (`*.test.ts`)
2. Group related tests with `describe` blocks
3. Use clear, descriptive test names
4. Mock external dependencies
5. Test edge cases and error conditions
6. Aim for >80% coverage on new code

## References

- Jest: https://jestjs.io/docs/getting-started
- Testing Library: https://testing-library.com/
- Vitest: https://vitest.dev/
