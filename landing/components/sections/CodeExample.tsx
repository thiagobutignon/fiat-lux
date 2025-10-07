'use client'

import { useState } from 'react'

export default function CodeExample() {
  const [activeTab, setActiveTab] = useState<'basic' | 'custom' | 'pattern'>('basic')

  const examples = {
    basic: `import { makeGrammarEngine, CLEAN_ARCHITECTURE_GRAMMAR } from 'fiat-lux'

// Create engine with predefined grammar
const engine = makeGrammarEngine(CLEAN_ARCHITECTURE_GRAMMAR)

// Process data
const result = engine.process({
  Subject: "DbAddAccount",
  Verb: "ad", // typo - will be auto-repaired to "add"
  Object: "Account.Params",
  Context: "Controller"
})

console.log(result)
// {
//   status: "INVALID",
//   repaired: { ..., Verb: "add" },
//   repairs: [{ original: "ad", repaired: "add", confidence: 0.823 }]
// }`,

    custom: `import { makeGrammarEngine, SimilarityAlgorithm } from 'fiat-lux'

const myGrammar: GrammarConfig = {
  roles: {
    Action: {
      values: ["create", "read", "update", "delete"],
      required: true
    },
    Resource: {
      values: ["user", "post", "comment"],
      required: true
    }
  },
  options: {
    similarityThreshold: 0.7,
    similarityAlgorithm: SimilarityAlgorithm.HYBRID
  }
}

const engine = makeGrammarEngine(myGrammar)

const result = engine.process({
  Action: "crate", // typo
  Resource: "usr"  // typo
})
// Auto-repairs: crate → create, usr → user`,

    pattern: `import { PatternLoader } from 'fiat-lux'

const loader = new PatternLoader(yamlContent)

// Validate naming conventions
const result = loader.validateNaming(
  'AddAccountUseCase',
  'domain',
  'usecases'
)
console.log(result.valid) // true

// Check dependency rules
const depResult = loader.validateDependency('domain', 'data')
console.log(depResult.valid) // false (forbidden)

// Get patterns by layer
const domainPatterns = loader.getPatternsByLayer('domain')
console.log(domainPatterns)
// [{ id: 'usecase', name: 'Use Case', layer: 'domain', ... }]`
  }

  return (
    <section className="py-24 bg-slate-50 dark:bg-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-4 gradient-text">
            Quick Start
          </h2>
          <p className="text-xl text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
            Get started in minutes with simple, intuitive APIs
          </p>
        </div>

        <div className="mb-8">
          <div className="flex flex-wrap gap-4 justify-center">
            <button
              onClick={() => setActiveTab('basic')}
              className={`px-6 py-3 rounded-lg font-semibold transition-all ${
                activeTab === 'basic'
                  ? 'bg-primary text-white shadow-lg'
                  : 'bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700'
              }`}
            >
              Basic Usage
            </button>
            <button
              onClick={() => setActiveTab('custom')}
              className={`px-6 py-3 rounded-lg font-semibold transition-all ${
                activeTab === 'custom'
                  ? 'bg-primary text-white shadow-lg'
                  : 'bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700'
              }`}
            >
              Custom Grammar
            </button>
            <button
              onClick={() => setActiveTab('pattern')}
              className={`px-6 py-3 rounded-lg font-semibold transition-all ${
                activeTab === 'pattern'
                  ? 'bg-primary text-white shadow-lg'
                  : 'bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700'
              }`}
            >
              Pattern Loader
            </button>
          </div>
        </div>

        <div className="bg-slate-900 dark:bg-slate-950 rounded-2xl p-6 overflow-x-auto border border-slate-700 shadow-2xl">
          <pre className="text-slate-100 font-mono text-sm leading-relaxed">
            {examples[activeTab]}
          </pre>
        </div>

        <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="p-6 bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-700">
            <h3 className="text-lg font-bold mb-2 text-slate-900 dark:text-white">
              📦 Installation
            </h3>
            <code className="text-sm text-primary">npm install fiat-lux</code>
          </div>

          <div className="p-6 bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-700">
            <h3 className="text-lg font-bold mb-2 text-slate-900 dark:text-white">
              📚 Documentation
            </h3>
            <a
              href="https://github.com/thiagobutignon/fiat-lux#readme"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-primary hover:underline"
            >
              View full docs →
            </a>
          </div>

          <div className="p-6 bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-700">
            <h3 className="text-lg font-bold mb-2 text-slate-900 dark:text-white">
              🧪 Test Suite
            </h3>
            <p className="text-sm text-slate-600 dark:text-slate-400">
              77 tests in 5ms
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
