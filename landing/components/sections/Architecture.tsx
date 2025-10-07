export default function Architecture() {
  return (
    <section id="architecture" className="py-24 bg-white dark:bg-slate-900">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-4 gradient-text">
            Clean Architecture
          </h2>
          <p className="text-xl text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
            Feature-based organization following the <code className="px-2 py-1 bg-slate-100 dark:bg-slate-800 rounded">src/[feature]/[use-cases]</code> pattern
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-start">
          <div>
            <h3 className="text-2xl font-bold mb-6 text-slate-900 dark:text-white">
              Project Structure
            </h3>
            <div className="bg-slate-900 dark:bg-slate-950 rounded-2xl p-6 font-mono text-sm overflow-x-auto border border-slate-700">
              <pre className="text-green-400">
{`src/
├── grammar-engine/
│   ├── domain/
│   │   ├── entities/
│   │   │   ├── types.ts
│   │   │   └── predefined-grammars.ts
│   │   └── use-cases/
│   │       └── grammar-engine.ts
│   ├── data/
│   │   ├── protocols/
│   │   │   └── similarity-cache.ts
│   │   └── use-cases/
│   │       └── similarity-cache-impl.ts
│   ├── presentation/
│   │   ├── factories/
│   │   ├── utils/
│   │   └── index.ts
│   └── __tests__/
│
├── similarity-algorithms/
│   ├── domain/
│   │   └── use-cases/
│   │       ├── levenshtein.ts
│   │       ├── jaro-winkler.ts
│   │       └── hybrid.ts
│   ├── presentation/
│   │   └── index.ts
│   └── __tests__/
│
├── pattern-loader/
│   ├── domain/
│   │   ├── entities/
│   │   └── use-cases/
│   ├── presentation/
│   └── __tests__/
│
└── shared/
    └── utils/
        └── test-runner.ts`}
              </pre>
            </div>
          </div>

          <div className="space-y-8">
            <div className="border-l-4 border-primary pl-6">
              <h4 className="text-xl font-bold mb-3 text-slate-900 dark:text-white">
                Domain Layer
              </h4>
              <p className="text-slate-600 dark:text-slate-400 mb-3">
                Pure business logic with no external dependencies
              </p>
              <ul className="space-y-2 text-slate-600 dark:text-slate-400">
                <li className="flex items-start">
                  <span className="text-primary mr-2">✓</span>
                  <span>Types and entities in <code className="px-1 bg-slate-100 dark:bg-slate-800 rounded text-sm">domain/entities</code></span>
                </li>
                <li className="flex items-start">
                  <span className="text-primary mr-2">✓</span>
                  <span>Use cases in <code className="px-1 bg-slate-100 dark:bg-slate-800 rounded text-sm">domain/use-cases</code></span>
                </li>
                <li className="flex items-start">
                  <span className="text-primary mr-2">✓</span>
                  <span>GrammarEngine core logic isolated</span>
                </li>
              </ul>
            </div>

            <div className="border-l-4 border-secondary pl-6">
              <h4 className="text-xl font-bold mb-3 text-slate-900 dark:text-white">
                Data Layer
              </h4>
              <p className="text-slate-600 dark:text-slate-400 mb-3">
                Interface adapters for external dependencies
              </p>
              <ul className="space-y-2 text-slate-600 dark:text-slate-400">
                <li className="flex items-start">
                  <span className="text-secondary mr-2">✓</span>
                  <span>Protocols (interfaces) in <code className="px-1 bg-slate-100 dark:bg-slate-800 rounded text-sm">data/protocols</code></span>
                </li>
                <li className="flex items-start">
                  <span className="text-secondary mr-2">✓</span>
                  <span>Implementations in <code className="px-1 bg-slate-100 dark:bg-slate-800 rounded text-sm">data/use-cases</code></span>
                </li>
                <li className="flex items-start">
                  <span className="text-secondary mr-2">✓</span>
                  <span>SimilarityCache with ISimilarityCache protocol</span>
                </li>
              </ul>
            </div>

            <div className="border-l-4 border-emerald-500 pl-6">
              <h4 className="text-xl font-bold mb-3 text-slate-900 dark:text-white">
                Presentation Layer
              </h4>
              <p className="text-slate-600 dark:text-slate-400 mb-3">
                Public API and utilities for library consumers
              </p>
              <ul className="space-y-2 text-slate-600 dark:text-slate-400">
                <li className="flex items-start">
                  <span className="text-emerald-500 mr-2">✓</span>
                  <span>Public API exports</span>
                </li>
                <li className="flex items-start">
                  <span className="text-emerald-500 mr-2">✓</span>
                  <span>Factory functions (<code className="px-1 bg-slate-100 dark:bg-slate-800 rounded text-sm">makeGrammarEngine</code>)</span>
                </li>
                <li className="flex items-start">
                  <span className="text-emerald-500 mr-2">✓</span>
                  <span>Utilities (<code className="px-1 bg-slate-100 dark:bg-slate-800 rounded text-sm">formatResult</code>)</span>
                </li>
              </ul>
            </div>
          </div>
        </div>

        <div className="mt-16 p-8 bg-primary/5 dark:bg-primary/10 rounded-2xl border border-primary/20">
          <h3 className="text-2xl font-bold mb-4 text-slate-900 dark:text-white text-center">
            Benefits of Clean Architecture
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-4xl mb-2">🧪</div>
              <div className="font-semibold text-slate-900 dark:text-white mb-1">Testability</div>
              <div className="text-sm text-slate-600 dark:text-slate-400">Easy to mock dependencies</div>
            </div>
            <div className="text-center">
              <div className="text-4xl mb-2">🔧</div>
              <div className="font-semibold text-slate-900 dark:text-white mb-1">Maintainability</div>
              <div className="text-sm text-slate-600 dark:text-slate-400">Clear separation of concerns</div>
            </div>
            <div className="text-center">
              <div className="text-4xl mb-2">📈</div>
              <div className="font-semibold text-slate-900 dark:text-white mb-1">Scalability</div>
              <div className="text-sm text-slate-600 dark:text-slate-400">Easy to add new features</div>
            </div>
            <div className="text-center">
              <div className="text-4xl mb-2">🔄</div>
              <div className="font-semibold text-slate-900 dark:text-white mb-1">Flexibility</div>
              <div className="text-sm text-slate-600 dark:text-slate-400">Swap implementations easily</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
