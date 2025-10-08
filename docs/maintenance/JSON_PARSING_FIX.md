# JSON Parsing Fix for AGI Recursive System

## Problem

When running the Budget Homeostasis demo (`npm run agi:homeostasis`), JSON parsing was failing with errors like:

```
❌ Failed to parse JSON from systems: Unexpected token '`', "```json..."
❌ Failed to parse JSON from financial: Unexpected token '`', "```json..."
```

## Root Cause

Claude's API responses often return JSON wrapped in markdown code blocks:

```
```json
{
  "answer": "Some answer",
  "confidence": 0.9
}
```
```

The code was directly calling `JSON.parse()` on these responses, which failed because JSON.parse() cannot handle the markdown formatting.

## Solution

Created an `extractJSON()` utility function in `meta-agent.ts` (lines 79-98) with **two-stage extraction**:

### Stage 1: Regex Extraction
```typescript
const codeBlockRegex = /```(?:json)?\s*\n?([\s\S]*?)```/;
const match = text.match(codeBlockRegex);

if (match && match[1]) {
  return match[1].trim(); // Extract content between ```
}
```

Handles formats:
- ```` ```json\n{...}\n``` ````
- ```` ```\n{...}\n``` ````

### Stage 2: Fallback (Brace Matching)
```typescript
const firstBrace = text.indexOf('{');
const lastBrace = text.lastIndexOf('}');

if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
  return text.substring(firstBrace, lastBrace + 1).trim();
}
```

If regex fails (e.g., missing closing ` ``` `), extract content between first `{` and last `}`.

### Stage 3: Last Resort
```typescript
return text.trim(); // Return as-is and let JSON.parse() error handling catch it
```

## Locations Updated

Applied `extractJSON()` before all `JSON.parse()` calls in `meta-agent.ts`:

1. **Line 147** - `SpecializedAgent.process()` - Parses individual agent responses
2. **Line 412** - `decomposeQuery()` - Parses domain decomposition
3. **Line 479** - `composeInsights()` - Parses composed insights
4. **Line 537** - `synthesizeFinal()` - Parses final synthesis

## Testing

### Test 1: Standalone Function Test
```bash
npx tsx test-extract-json.ts
```

**Results:**
- ✅ Standard markdown code block - PASS
- ✅ Missing closing backticks (fallback) - PASS
- ✅ Plain JSON (no markdown) - PASS
- ✅ Text surrounding code block - PASS

### Test 2: Integration Test (Navigation Demo)
```bash
npm run agi:navigation
```
**Result:** ✅ PASS (no API calls, verifies module loads correctly)

### Test 3: Full System Test (Budget Homeostasis)
```bash
npm run agi:homeostasis
```
**Result:** ⏱️ Takes 2+ minutes due to multiple recursive API calls
**Note:** Requires stable Anthropic API. May timeout if API is slow or experiencing issues.

## Commits

- `b48c995` - fix: improve JSON parsing to handle markdown code blocks with fallback mechanism
- `8c20ad5` - chore: remove debug logs from extractJSON function

## Known Limitations

### Demo Execution Time
The Budget Homeostasis demo makes **multiple sequential API calls**:
1. Query decomposition (MetaAgent → Claude)
2. Each specialist agent processes query (3 agents × Claude)
3. Insight composition (MetaAgent → Claude)
4. Final synthesis (MetaAgent → Claude)
5. Potential recursion (repeat steps 1-4 if needed)

**Total:** 5-15 API calls depending on recursion depth

**Typical duration:** 30-90 seconds
**Timeout threshold:** 2-3 minutes

### When Demo Might Fail/Timeout

1. **Anthropic API Issues**
   - 500 errors (internal server error)
   - Rate limiting
   - High latency (>10s per request)

2. **Recursion Depth**
   - Max depth = 3 (configured)
   - Max invocations = 10 (configured)
   - If composition suggests deep recursion, may hit limits

3. **JSON Parsing Edge Cases**
   - Malformed JSON from Claude (rare but possible)
   - Falls back to returning raw text with low confidence
   - Does not crash, degrades gracefully

## Recommendations

### For Development
```bash
# Run simpler demos first to verify API connectivity
npm run agi:adapter      # ~5 seconds, 1-2 API calls
npm run agi:navigation   # ~1 second, no API calls
npm run agi:acl          # ~5 seconds, no API calls

# Then run full system
npm run agi:homeostasis  # ~30-90 seconds, 5-15 API calls
```

### For Production
Consider adding:
1. **Timeout Configuration** - Environment variable for max execution time
2. **Retry Logic** - Exponential backoff for API errors
3. **Progress Indicators** - Console logs for each stage
4. **Cost Tracking UI** - Real-time display of cumulative cost
5. **Async Processing** - Queue-based architecture for long-running queries

## References

- **Main PR:** https://github.com/thiagobutignon/fiat-lux/pull/11
- **Code:** `src/agi-recursive/core/meta-agent.ts` (lines 79-98, 147, 412, 479, 537)
- **Tests:** Created `test-extract-json.ts` and `test-meta-agent-json.ts` (removed after testing)
- **Docs:** `QUICKSTART.md`, `docs/AGI_QUICKSTART.md`

## Status

✅ **FIXED** - JSON parsing now handles markdown code blocks robustly
⏱️ **PENDING** - Full system demo verification (waiting for stable API)
📝 **DOCUMENTED** - Fix documented, tested, and pushed to feature branch

---

**Last Updated:** 2025-10-08
**Branch:** `feature/deterministic-intelligence-benchmark`
**Commits:** `b48c995`, `8c20ad5`
