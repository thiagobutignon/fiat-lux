# Model Versioning Strategy

## Overview

The Anthropic Adapter uses **pinned model versions** to ensure reproducibility and stability. This document explains the versioning strategy and how to handle model updates.

## Current Model IDs

```typescript
const MODEL_IDS: Record<ClaudeModel, string> = {
  'claude-opus-4': 'claude-opus-4-20250514',
  'claude-sonnet-4-5': 'claude-sonnet-4-5-20250929',
};
```

**Rationale**: Model IDs include date-based version suffixes to guarantee:
- **Reproducibility**: Same input always produces same output (within temperature=0)
- **Stability**: Model behavior doesn't change unexpectedly
- **Testing**: Test results remain consistent
- **Cost Tracking**: Pricing is predictable

## Versioning Philosophy

### 🎯 Pin vs. Latest

| Strategy | Pros | Cons | Use Case |
|----------|------|------|----------|
| **Pinned Versions** (Current) | Reproducible, stable, testable | Manual updates required | Production systems, research |
| **Latest Aliases** | Auto-updates, newest features | Breaking changes, cost surprises | Experimental, rapid prototyping |

**Decision**: We use **pinned versions** because:
1. AGI system requires consistent behavior
2. Cost budgets need predictable pricing
3. Constitutional validation depends on stable outputs
4. Research results need reproducibility

### 🔄 When to Update

Update model versions when:

1. **New Capabilities**: Anthropic releases models with significantly improved reasoning
2. **Cost Reduction**: Newer versions offer better price/performance ratio
3. **Bug Fixes**: Anthropic patches issues in older versions
4. **Deprecation**: Anthropic announces end-of-life for current versions

### ⚠️ Update Process

When updating model IDs:

```typescript
// 1. Update MODEL_IDS constant
const MODEL_IDS: Record<ClaudeModel, string> = {
  'claude-opus-4': 'claude-opus-4-20250610',  // ← NEW VERSION
  'claude-sonnet-4-5': 'claude-sonnet-4-5-20250929',
};

// 2. Update pricing if changed
const MODEL_PRICING: Record<ClaudeModel, ModelPricing> = {
  'claude-opus-4': {
    input_per_million: 15.0,  // ← CHECK PRICING
    output_per_million: 75.0,
  },
  // ...
};

// 3. Run tests
npm test

// 4. Run demos and verify behavior
npx tsx src/agi-recursive/examples/budget-homeostasis.ts

// 5. Update CHANGELOG.md
## [Version] - Date
### Changed
- Updated Claude Opus 4 to version 20250610
- Verified cost tracking accuracy
- Tested all demos successfully

// 6. Commit with version note
git commit -m "chore: update Claude Opus 4 to 20250610

- Model ID: claude-opus-4-20250610
- Pricing: $15/$75 per 1M tokens (unchanged)
- Tested: All demos passing
- Backward compatible: Yes"
```

## Model Selection Guide

### Claude Opus 4 (`claude-opus-4-20250514`)

**Best For**:
- Complex reasoning tasks
- Creative problem solving
- Multi-step logical inference
- Cross-domain synthesis

**Pricing**: $15 input / $75 output per 1M tokens

**Use When**:
```typescript
const config = { model: 'claude-opus-4' };
// Use for: Final synthesis, complex analysis
```

### Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`)

**Best For**:
- Fast responses
- Cost-effective processing
- High-volume tasks
- Standard queries

**Pricing**: $3 input / $15 output per 1M tokens (80% cheaper!)

**Use When**:
```typescript
const config = { model: 'claude-sonnet-4-5' };
// Use for: Agent processing, decomposition, standard tasks
```

## Override Mechanism

To allow users to override model IDs without code changes:

```typescript
// Future enhancement (not yet implemented)
export interface AdapterConfig {
  modelOverrides?: Partial<Record<ClaudeModel, string>>;
}

const adapter = new AnthropicAdapter(apiKey, {
  modelOverrides: {
    'claude-opus-4': 'claude-opus-4-latest', // Use latest
  }
});
```

## Monitoring Model Changes

### Check for New Versions

Periodically check Anthropic's documentation:
- **API Docs**: https://docs.anthropic.com/claude/docs/models-overview
- **Changelog**: https://docs.anthropic.com/claude/changelog
- **Pricing**: https://www.anthropic.com/pricing

### Deprecation Warnings

Anthropic typically:
1. Announces deprecation 3-6 months in advance
2. Provides migration guides
3. Maintains older versions during transition

**Action**: Subscribe to Anthropic's developer newsletter

## Testing Strategy

### Before Updating

```bash
# 1. Run adapter demo
npx tsx src/agi-recursive/examples/anthropic-adapter-demo.ts

# 2. Run full AGI demo
npx tsx src/agi-recursive/examples/budget-homeostasis.ts

# 3. Check cost tracking
# Verify total_cost matches expected pricing
```

### After Updating

```bash
# 1. Regression test (compare outputs)
# Run same query with old and new model
# Document behavior differences

# 2. Cost validation
# Ensure pricing constants match actual API charges

# 3. Performance test
# Measure latency changes
```

## Backward Compatibility

### Guaranteed

- ✅ Public API remains unchanged
- ✅ Existing agent code works without modification
- ✅ Cost tracking continues to function
- ✅ Configuration options stay compatible

### May Change

- ⚠️ Actual LLM outputs (even with same prompts)
- ⚠️ Token counts (tokenization may differ)
- ⚠️ Latency (newer models may be faster/slower)
- ⚠️ Costs (if Anthropic changes pricing)

## Emergency Rollback

If a model update causes issues:

```bash
# 1. Revert MODEL_IDS to previous version
git log --oneline | grep "update.*model"
git revert <commit-hash>

# 2. Or manually edit
# anthropic-adapter.ts line 69-72
const MODEL_IDS = {
  'claude-opus-4': 'claude-opus-4-20250514', // ← ROLLBACK
  // ...
};

# 3. Redeploy immediately
npm run build
# Deploy to production

# 4. Document incident
# Add to CHANGELOG.md
```

## Future Enhancements

### 1. Configuration File

```yaml
# config/models.yml
models:
  opus-4:
    id: claude-opus-4-20250514
    pricing:
      input: 15.0
      output: 75.0
  sonnet-4-5:
    id: claude-sonnet-4-5-20250929
    pricing:
      input: 3.0
      output: 15.0
```

### 2. Environment Variable Override

```bash
# .env
CLAUDE_OPUS_4_VERSION=claude-opus-4-20250610
CLAUDE_SONNET_45_VERSION=claude-sonnet-4-5-latest
```

### 3. Automatic Version Detection

```typescript
// Check Anthropic API for latest versions
const latestVersions = await anthropic.getAvailableModels();
```

## FAQ

### Q: Why not always use latest?

**A**: AGI systems need reproducibility. Model updates can:
- Change reasoning patterns
- Affect Constitutional validation
- Alter cost predictions
- Break existing integrations

### Q: How often should we update?

**A**: Recommended schedule:
- **Quarterly**: Review for new versions
- **As Needed**: When Anthropic announces major improvements
- **Immediately**: If security issues or deprecations announced

### Q: What if Anthropic changes pricing?

**A**: Update process:
1. Update `MODEL_PRICING` constants
2. Run cost estimation tests
3. Update documentation
4. Notify users of cost changes

### Q: Can we mix versions?

**A**: Yes! You can use:
- Opus 4 version A for synthesis
- Sonnet 4.5 version B for agents

This is safe as long as pricing is correct.

## References

- **Anthropic Models**: https://docs.anthropic.com/claude/docs/models-overview
- **Model Versions**: https://docs.anthropic.com/claude/docs/model-versions
- **Pricing**: https://www.anthropic.com/pricing
- **Changelog**: https://docs.anthropic.com/claude/changelog

---

**Last Updated**: 2025-01-07
**Reviewed By**: AGI Team
**Next Review**: 2025-04-07 (Quarterly)
