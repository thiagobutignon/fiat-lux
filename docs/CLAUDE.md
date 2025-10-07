# Claude Code Configuration

This file defines coding standards and guidelines for Claude Code when working on the fiat-lux project.

## Project Overview

Fiat-Lux is a Universal Grammar Engine - a generic, configurable grammar validation and auto-repair system that can be applied to any structured data domain (code architecture, natural language, configuration files, etc.).

## Core Principles

1. **Grammar as Data**: Rules are declarative and configurable
2. **Multiple Algorithms**: Pluggable similarity and repair strategies
3. **Explainability**: Every decision is traceable and reportable
4. **Performance**: Caching and optimization for large-scale processing
5. **Type Safety**: Full TypeScript support with generics

## Coding Standards

### TypeScript

- Use **TypeScript strict mode** with full type safety
- Prefer **interfaces** over types for object definitions
- Use **readonly** for immutable data structures
- Export all public APIs for reusability
- Document all public functions with JSDoc comments

### Code Organization

- Group related code into sections with clear separator comments
- Use section headers like: `// ============================================================================`
- Keep functions focused and single-purpose
- Prefer pure functions without side effects

### Naming Conventions

- **Classes**: PascalCase (e.g., `GrammarEngine`, `SimilarityCache`)
- **Functions**: camelCase (e.g., `calculateSimilarity`, `findBestMatch`)
- **Constants**: SCREAMING_SNAKE_CASE (e.g., `CLEAN_ARCHITECTURE_GRAMMAR`)
- **Interfaces**: PascalCase with descriptive names (e.g., `ProcessingResult`, `RepairOperation`)
- **Enums**: PascalCase for enum name, SCREAMING_SNAKE_CASE for values

### Documentation

- Every exported function must have JSDoc documentation
- Include:
  - Description of what the function does
  - `@param` for each parameter with type and description
  - `@returns` for return values
  - `@example` for complex functions
- Use clear, concise comments for complex logic

### Performance

- Cache expensive calculations when possible
- Use early returns to avoid unnecessary processing
- Consider algorithmic complexity (document Big-O when relevant)
- Implement performance tracking (timing, cache hits, etc.)

### Error Handling

- Validate inputs at public API boundaries
- Return rich error objects with helpful messages
- Include suggestions for fixing errors when possible
- Never silently fail - always provide feedback

### Testing

- Include demonstration functions (`runDemo()`)
- Provide multiple test scenarios
- Show both valid and invalid cases
- Include edge cases and performance tests

## Grammar Engine Specific Guidelines

### Grammar Definitions

- Keep grammars declarative and data-driven
- Include descriptions for each role
- Specify required vs optional fields clearly
- Document structural rules with clear names and messages

### Similarity Algorithms

- Implement multiple algorithms for different use cases
- Normalize scores to 0-1 range
- Document algorithm strengths and weaknesses
- Allow algorithm selection via configuration

### Auto-Repair

- Provide multiple repair suggestions (not just one)
- Include confidence scores with repairs
- Show alternatives ranked by similarity
- Make repair thresholds configurable

### Result Formatting

- Provide formatted output for CLI usage
- Include metadata (timing, cache stats, algorithms used)
- Use clear visual indicators (✅ ❌ 🔧 📊)
- Structure results for easy parsing

## Code Review Focus Areas

When reviewing code, prioritize:

1. **Type Safety**: Ensure full TypeScript coverage
2. **Documentation**: Verify JSDoc is complete and accurate
3. **Performance**: Check for unnecessary computations
4. **Generics**: Ensure code is generic and reusable
5. **Testing**: Validate demo/test coverage
6. **Error Messages**: Ensure errors are helpful and actionable

## Architecture Patterns

- Use **composition** over inheritance
- Implement **strategy pattern** for algorithms
- Use **builder pattern** for complex configurations
- Apply **facade pattern** for simplified APIs

## Version Control

- Use conventional commits (feat:, fix:, docs:, refactor:, test:, perf:)
- Write descriptive commit messages
- Include Co-Authored-By for Claude contributions
- Keep commits atomic and focused

## Dependencies

- Minimize external dependencies
- Prefer built-in Node.js/TypeScript features
- Document any required dependencies
- Consider bundle size impact

## Future Enhancements

Areas for potential improvement:
- Additional similarity algorithms (Soundex, Metaphone, etc.)
- Fuzzy matching with configurable fuzziness
- Machine learning integration for repair suggestions
- Grammar learning from examples
- Visual grammar editor
- Performance profiling tools
- Grammar composition and inheritance
