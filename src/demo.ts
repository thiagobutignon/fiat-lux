/**
 * Fiat Lux Demo
 *
 * Demonstrates the Grammar Engine capabilities
 */

import {
  makeGrammarEngine,
  CLEAN_ARCHITECTURE_GRAMMAR,
  HTTP_API_GRAMMAR,
  formatResult
} from './grammar-engine/presentation'

import {
  levenshteinSimilarity,
  jaroWinklerSimilarity,
  hybridSimilarity
} from './similarity-algorithms/presentation'

/**
 * Run comprehensive demonstration
 */
export function runDemo(): void {
  console.log("\n🌟 FIAT LUX - Universal Grammar Engine\n")
  console.log("A generic, configurable system for validating and repairing structured data\n")

  // Demo 1: Clean Architecture validation
  console.log("═".repeat(80))
  console.log("Demo 1: Clean Architecture - Invalid Object Token")
  console.log("═".repeat(80))

  const engine1 = makeGrammarEngine(CLEAN_ARCHITECTURE_GRAMMAR)
  const result1 = engine1.process({
    Subject: "DbAddAccount",
    Verb: "add",
    Object: "BLABAKABA", // noise
    Adverbs: ["Hasher", "Repository"],
    Context: "Controller"
  })
  console.log(formatResult(result1))

  // Demo 2: Multiple errors with different algorithms
  console.log("\n" + "═".repeat(80))
  console.log("Demo 2: Multiple Invalid Tokens with Hybrid Algorithm")
  console.log("═".repeat(80))

  const engine2 = makeGrammarEngine(CLEAN_ARCHITECTURE_GRAMMAR)
  const result2 = engine2.process({
    Subject: "RemoteLoadSurvey",
    Verb: "lod", // typo: should be "load"
    Object: "UserParams", // close to "User.Params"
    Adverbs: ["Hash", "Validatr"], // typos
    Context: "Facto" // typo: should be "MainFactory"
  })
  console.log(formatResult(result2))

  // Demo 3: Valid sentence
  console.log("\n" + "═".repeat(80))
  console.log("Demo 3: Valid Sentence - No Repairs Needed")
  console.log("═".repeat(80))

  const engine3 = makeGrammarEngine(CLEAN_ARCHITECTURE_GRAMMAR)
  const result3 = engine3.process({
    Subject: "DbLoadSurvey",
    Verb: "load",
    Object: "Survey.Params",
    Adverbs: ["Repository", "Validator"],
    Context: "UseCase"
  })
  console.log(formatResult(result3))

  // Demo 4: HTTP API Grammar
  console.log("\n" + "═".repeat(80))
  console.log("Demo 4: HTTP API Grammar - Different Domain")
  console.log("═".repeat(80))

  const engine4 = makeGrammarEngine(HTTP_API_GRAMMAR)
  const result4 = engine4.process({
    Method: "PSOT", // typo: should be "POST"
    Resource: "/user", // typo: should be "/users"
    Status: "201",
    Handler: ["Controller", "Middleware"]
  })
  console.log(formatResult(result4))

  // Demo 5: Algorithm comparison
  console.log("\n" + "═".repeat(80))
  console.log("Demo 5: Algorithm Comparison")
  console.log("═".repeat(80))

  const testPairs = [
    ["add", "ad"],
    ["Repository", "Repo"],
    ["Controller", "Control"],
    ["Account.Params", "AccountParams"],
  ]

  console.log("\nComparing Levenshtein, Jaro-Winkler, and Hybrid algorithms:\n")
  testPairs.forEach(([correct, typo]) => {
    const lev = levenshteinSimilarity(correct, typo, false)
    const jaro = jaroWinklerSimilarity(correct.toLowerCase(), typo.toLowerCase())
    const hybrid = hybridSimilarity(correct, typo, false)

    console.log(`"${typo}" → "${correct}"`)
    console.log(`  Levenshtein:  ${(lev * 100).toFixed(1)}%`)
    console.log(`  Jaro-Winkler: ${(jaro * 100).toFixed(1)}%`)
    console.log(`  Hybrid:       ${(hybrid * 100).toFixed(1)}%`)
    console.log("")
  })

  // Demo 6: Cache performance
  console.log("\n" + "═".repeat(80))
  console.log("Demo 6: Cache Performance")
  console.log("═".repeat(80))

  const engine6 = makeGrammarEngine(CLEAN_ARCHITECTURE_GRAMMAR)

  // Process same sentence multiple times
  const testSentence = {
    Subject: "DbAddAcount", // typo
    Verb: "ad", // typo
    Object: "Acount.Params" // typo
  }

  console.log("\nProcessing same sentence 100 times to test caching...\n")
  const iterations = 100
  const startTime = performance.now()

  for (let i = 0; i < iterations; i++) {
    engine6.process(testSentence)
  }

  const endTime = performance.now()
  const stats = engine6.getCacheStats()

  console.log(`Total time: ${(endTime - startTime).toFixed(2)}ms`)
  console.log(`Average per iteration: ${((endTime - startTime) / iterations).toFixed(2)}ms`)
  console.log(`Cache stats:`)
  console.log(`  - Hits: ${stats.hits}`)
  console.log(`  - Misses: ${stats.misses}`)
  console.log(`  - Hit rate: ${(stats.hitRate * 100).toFixed(1)}%`)
  console.log(`  - Cache size: ${stats.size} entries`)

  console.log("\n✨ Demo complete!\n")
}

// Run if executed directly
if (require.main === module) {
  runDemo()
}
