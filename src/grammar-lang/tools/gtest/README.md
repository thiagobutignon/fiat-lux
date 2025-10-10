# GTest - O(1) Testing Framework for Grammar Language

**O(1) testing framework** with hash-based discovery, incremental testing, and seamless integration with Grammar Language tools.

## 🚀 Features

- **O(1) Test Discovery**: Hash-based indexing for constant-time test lookup
- **Incremental Testing**: Only run tests for changed files (O(1) per file comparison)
- **Parallel Execution**: Run tests concurrently for maximum performance
- **Code Coverage**: Track line, branch, and function coverage with O(1) updates
- **Native .gtest Format**: Grammar Language-native test specification
- **Tool Integration**: Seamless integration with GLM, GSX, and GLC
- **Watch Mode**: Continuous testing on file changes
- **Zero Dependencies**: Standalone framework with no external dependencies

## 📦 Installation

```bash
npm install @grammar-lang/gtest
```

## 🧪 Quick Start

### 1. Create a Test File

Create a `.gtest` file (e.g., `math.gtest`):

```grammar
# Math - Test Suite

setup:
  # Initialize math library
  const add = (a, b) => a + b
  const multiply = (a, b) => a * b

test "should add two numbers":
  given: a = 2, b = 3
  when: result = add(a, b)
  then: expect result equals 5

test "should multiply two numbers":
  given: a = 4, b = 5
  when: result = multiply(a, b)
  then: expect result equals 20

test "should handle zero":
  given: a = 0, b = 10
  when: result = add(a, b)
  then: expect result equals 10
```

### 2. Run Tests

```bash
# Run all tests
gtest

# Run tests in specific directory
gtest ./tests

# Run with coverage
gtest -c

# Run in parallel
gtest -p

# Watch mode
gtest -w

# Incremental (only changed tests)
gtest -i
```

### 3. Programmatic Usage

```typescript
import { runTests, expect } from '@grammar-lang/gtest';

// Run all tests
const summary = await runTests('./tests');

console.log(`Total: ${summary.total}`);
console.log(`Passed: ${summary.passed}`);
console.log(`Failed: ${summary.failed}`);

// Use assertions in code
expect(2 + 2).toEqual(4);
expect('hello').toContainString('ell');
expect([1, 2, 3]).toHaveLength(3);
```

## 📋 Test File Format (.gtest)

### Basic Structure

```grammar
# Suite Name

setup:
  # Run once before all tests

teardown:
  # Run once after all tests

beforeEach:
  # Run before each test

afterEach:
  # Run after each test

test "test name":
  given: # Setup (optional)
  when: # Action (required)
  then: # Assertion (required)
```

### Given-When-Then Pattern

```grammar
test "should calculate fibonacci":
  given: n = 5
  when: result = fibonacci(n)
  then: expect result equals 5

test "should handle edge cases":
  given: n = 0
  when: result = fibonacci(n)
  then: expect result equals 0
```

### Assertions

GTest supports multiple assertion types:

```grammar
# Equality
then: expect result equals 5
then: expect value strictEquals expectedValue

# Comparison
then: expect score greaterThan 80
then: expect count lessThan 100

# String matching
then: expect message contains "error"
then: expect output matches /pattern/

# Exceptions
then: expect fn throws "Invalid input"

# Truthiness
then: expect value toBeTruthy
then: expect empty toBeFalsy
```

## 🔧 API Reference

### Test Execution

#### `runTests(rootDir: string): Promise<GTestSummary>`

Run all tests in a directory.

```typescript
const summary = await runTests('./tests');
```

#### `runIncrementalTests(rootDir: string): Promise<GTestSummary>`

Run only tests for changed files (O(1) per file).

```typescript
const summary = await runIncrementalTests('./tests');
```

#### `runParallelTests(rootDir: string, maxWorkers?: number): Promise<GTestSummary>`

Run tests in parallel.

```typescript
const summary = await runParallelTests('./tests', 4);
```

### Assertions

#### `expect(actual: any): Assertion`

Create an assertion.

```typescript
expect(value).toEqual(expected);
expect(value).not.toEqual(unexpected);
```

#### Available Matchers

**Equality:**
- `toEqual(expected)` - Loose equality (==)
- `toStrictEqual(expected)` - Strict equality (===)
- `toDeepEqual(expected)` - Deep object equality

**Truthiness:**
- `toBeTruthy()` - Value is truthy
- `toBeFalsy()` - Value is falsy
- `toBeNull()` - Value is null
- `toBeUndefined()` - Value is undefined
- `toBeDefined()` - Value is not undefined

**Comparison:**
- `toBeGreaterThan(n)` - Value > n
- `toBeLessThan(n)` - Value < n
- `toBeGreaterThanOrEqual(n)` - Value >= n
- `toBeLessThanOrEqual(n)` - Value <= n
- `toBeCloseTo(n, precision)` - Floating point comparison

**String:**
- `toMatch(pattern)` - Regex match
- `toContainString(substring)` - Contains substring
- `toStartWith(prefix)` - Starts with string
- `toEndWith(suffix)` - Ends with string

**Array/Object:**
- `toContain(item)` - Array contains item
- `toHaveLength(n)` - Array/string length
- `toHaveProperty(key, value?)` - Object has property
- `toBeEmpty()` - Array/object/string is empty

**Function:**
- `toThrow(message?)` - Function throws
- `toThrowError(type, message?)` - Function throws specific error
- `toHaveBeenCalled()` - Spy/mock was called
- `toHaveBeenCalledWith(...args)` - Spy/mock called with args
- `toHaveBeenCalledTimes(n)` - Spy/mock called n times

**Async:**
- `toResolve()` - Promise resolves
- `toResolveWith(value)` - Promise resolves with value
- `toReject()` - Promise rejects
- `toRejectWith(error)` - Promise rejects with error

**Type:**
- `toBeInstanceOf(type)` - Value is instance of type
- `toBeTypeOf(type)` - typeof value === type
- `toBeArray()` - Value is array
- `toBeObject()` - Value is object
- `toBeFunction()` - Value is function

### Coverage

#### `startCoverage(): void`

Start coverage tracking.

```typescript
import { startCoverage, stopCoverage, getCoverageReport } from '@grammar-lang/gtest';

startCoverage();
// Run tests...
stopCoverage();

const report = getCoverageReport();
console.log(`Line coverage: ${report.summary.percentage.lines.toFixed(2)}%`);
```

#### `trackLine(filePath: string, lineNumber: number): void`

Track line execution (O(1)).

#### `trackBranch(filePath: string, branchId: string, taken: boolean): void`

Track branch execution (O(1)).

#### `trackFunction(filePath: string, functionName: string): void`

Track function execution (O(1)).

### Integration

#### GLM Package Testing

```typescript
import { runGLMPackageTests } from '@grammar-lang/gtest';

const result = await runGLMPackageTests({
  packageDir: './my-package',
  coverage: true,
  incremental: true,
  parallel: true,
  outputDir: './reports'
});
```

#### GSX Integration

```typescript
import { createGSXExecutor, runTestsWithGSX } from '@grammar-lang/gtest';

const executor = createGSXExecutor();
executor.setGlobal('PI', 3.14159);

const summary = await runTestsWithGSX('./tests');
```

#### GLC Compilation Testing

```typescript
import { runTestsWithGLC } from '@grammar-lang/gtest';

const result = await runTestsWithGLC({
  sourceDir: './src',
  outputDir: './dist',
  compileBeforeTest: true,
  validateOutput: true
});
```

## 🎯 CLI Reference

```bash
# Basic usage
gtest [directory] [options]

# Options
-c, --coverage        # Enable code coverage
-i, --incremental     # Run only changed tests
-p, --parallel        # Run tests in parallel
-w, --watch           # Watch mode (re-run on changes)
-o, --output <dir>    # Output directory for reports
-v, --verbose         # Verbose output
--glm                 # Enable GLM package testing
--gsx                 # Enable GSX integration
--glc                 # Enable GLC compilation testing
--version             # Show version
--help                # Show help

# Examples
gtest                           # Run all tests in current directory
gtest ./tests                   # Run tests in ./tests directory
gtest -c -p                     # Run with coverage in parallel
gtest -w -i                     # Watch mode with incremental testing
gtest --glm -c                  # GLM package testing with coverage
gtest --glc ./src ./dist        # GLC compilation testing
```

## 🏗️ Architecture

### O(1) Operations

GTest achieves O(1) operations through:

1. **Hash-based Test Discovery**: Each test has a SHA256 hash ID for O(1) lookup
2. **Map-based Storage**: All indexes use JavaScript Map for O(1) insert/lookup
3. **Incremental Testing**: File changes detected via O(1) hash comparison
4. **Coverage Tracking**: Line/branch/function tracking with O(1) map updates

### File Structure

```
gtest/
├── spec.ts          # Test specification format
├── runner.ts        # O(1) test runner
├── assertions.ts    # Assertion library (25+ matchers)
├── coverage.ts      # O(1) coverage tracking
├── integration.ts   # GLM/GSX/GLC integration
├── index.ts         # Main exports
├── cli.ts           # Command-line interface
└── README.md        # Documentation
```

### Test Index Structure

```typescript
interface GTestIndex {
  version: string;
  timestamp: number;
  suites: Map<string, GTestSuite>;    // O(1) lookup
  tests: Map<string, GTestCase>;      // O(1) lookup
  files: Map<string, string>;         // O(1) lookup
  hashes: Map<string, string>;        // O(1) lookup
}
```

## 📊 Performance

GTest is designed for maximum performance:

- **Test Discovery**: O(1) per test via hash-based indexing
- **Test Selection**: O(1) lookup via Map
- **Incremental Run**: O(1) per changed file (hash comparison)
- **Coverage Update**: O(1) per line/branch/function
- **Result Aggregation**: O(n) streaming (unavoidable)

## 🔗 Integration Examples

### With GLM (Grammar Language Manager)

```typescript
import { validateGLMPackage } from '@grammar-lang/gtest';

// Validate package (runs tests + checks coverage)
const isValid = await validateGLMPackage('./my-package');

if (isValid) {
  console.log('✅ Package is valid!');
} else {
  console.log('❌ Package validation failed!');
}
```

### With GSX (Grammar Script eXecutor)

```typescript
import { createGSXExecutor } from '@grammar-lang/gtest';

const executor = createGSXExecutor();

// Set globals
executor.setGlobal('config', { debug: true });

// Load modules
executor.loadModule('math', mathLibrary);

// Execute code
const result = await executor.executeCode('add(2, 3)');
```

### With GLC (Grammar Language Compiler)

```typescript
import { validateGLCCompilation } from '@grammar-lang/gtest';

// Validate compilation + run tests
const isValid = await validateGLCCompilation('./src', './dist');

if (isValid) {
  console.log('✅ Compilation successful!');
} else {
  console.log('❌ Compilation failed!');
}
```

## 🧩 Advanced Usage

### Custom Matchers

```typescript
import { Assertion } from '@grammar-lang/gtest';

// Register custom matcher
Assertion.prototype.registerMatcher('toBeEven', (actual) => ({
  pass: actual % 2 === 0,
  message: `Expected ${actual} to be even`
}));

// Use custom matcher
expect(4).toBeEven();
```

### Mocking and Spying

```typescript
import { createMock, createSpy } from '@grammar-lang/gtest';

// Create mock
const mockFn = createMock<(x: number) => number>();
mockFn.mockReturnValue(42);

expect(mockFn(10)).toEqual(42);
expect(mockFn).toHaveBeenCalledWith(10);

// Create spy
const spy = createSpy(console, 'log');
console.log('test');

expect(spy).toHaveBeenCalled();
expect(spy).toHaveBeenCalledWith('test');
```

### Watch Mode

```typescript
import { watchTests } from '@grammar-lang/gtest';

// Start watch mode
await watchTests('./tests', {
  coverage: true,
  incremental: true,
  parallel: true
});
```

## 📝 Best Practices

1. **Use Incremental Testing**: Run only changed tests in development
   ```bash
   gtest -i -w  # Watch + incremental
   ```

2. **Enable Coverage in CI**: Track code coverage in CI/CD
   ```bash
   gtest -c -p -o ./reports
   ```

3. **Organize Tests by Feature**: Group related tests in same `.gtest` file
   ```
   tests/
   ├── math.gtest
   ├── string.gtest
   └── array.gtest
   ```

4. **Use Setup/Teardown**: Initialize shared resources in setup hooks
   ```grammar
   setup:
     const db = await initDatabase()

   teardown:
     await db.close()
   ```

5. **Write Descriptive Test Names**: Use clear, specific test descriptions
   ```grammar
   test "should throw error when input is negative":
     given: n = -1
     when: result = fibonacci(n)
     then: expect result throws "Input must be non-negative"
   ```

## 🐛 Troubleshooting

### Tests Not Found

Ensure test files use `.gtest` extension:
```bash
# Correct
tests/math.gtest

# Incorrect
tests/math.test.ts
```

### Hash Collisions

GTest uses SHA256 (16-char substring). Hash collisions are extremely rare (1 in 2^64). If you encounter one, increase hash length in `spec.ts`:
```typescript
return crypto.createHash('sha256').update(content).digest('hex').substring(0, 32); // Use 32 chars
```

### Coverage Inaccurate

Make sure to call `startCoverage()` before running tests:
```typescript
import { startCoverage, stopCoverage, getCoverageReport } from '@grammar-lang/gtest';

startCoverage();
// ... run tests ...
stopCoverage();
const report = getCoverageReport();
```

## 📚 Resources

- [Grammar Language Docs](https://github.com/chomsky/grammar-lang)
- [GTest Examples](https://github.com/chomsky/grammar-lang/tree/main/examples/gtest)
- [API Reference](https://github.com/chomsky/grammar-lang/tree/main/docs/gtest)

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

**GTest** - O(1) Testing Framework for Grammar Language
Built with ❤️ by the Grammar Language Team
