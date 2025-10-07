'use client'

import { useState } from 'react'

export default function Demo() {
  const [input, setInput] = useState(`{
  "Subject": "DbAddAccount",
  "Verb": "ad",
  "Object": "Account.Params",
  "Context": "Controller"
}`)

  const [output] = useState(`{
  "status": "INVALID",
  "errors": [
    {
      "role": "Verb",
      "value": "ad",
      "message": "Invalid Verb: 'ad' is not in allowed vocabulary",
      "suggestions": ["add", "delete", "update"]
    }
  ],
  "repaired": {
    "Subject": "DbAddAccount",
    "Verb": "add",
    "Object": "Account.Params",
    "Context": "Controller"
  },
  "repairs": [
    {
      "role": "Verb",
      "original": "ad",
      "repaired": "add",
      "confidence": 0.823,
      "algorithm": "hybrid"
    }
  ],
  "metadata": {
    "processingTime": "0.32ms",
    "cacheHits": 5,
    "algorithm": "hybrid"
  }
}`)

  return (
    <section id="demo" className="py-24 bg-slate-50 dark:bg-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-4 gradient-text">
            Live Demo
          </h2>
          <p className="text-xl text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
            See how Fiat Lux validates and auto-repairs structured data in real-time
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
                Input
              </h3>
              <span className="px-3 py-1 text-xs font-medium bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 rounded-full">
                Invalid
              </span>
            </div>
            <div className="relative">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                className="w-full h-96 p-4 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded-xl font-mono text-sm text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-primary resize-none"
                spellCheck={false}
              />
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
                Output
              </h3>
              <span className="px-3 py-1 text-xs font-medium bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400 rounded-full">
                Auto-Repaired
              </span>
            </div>
            <div className="relative">
              <pre className="w-full h-96 p-4 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded-xl font-mono text-sm text-slate-900 dark:text-slate-100 overflow-auto">
                {output}
              </pre>
            </div>
          </div>
        </div>

        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="p-6 bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-700">
            <div className="text-primary font-semibold mb-2">Algorithm</div>
            <div className="text-2xl font-bold text-slate-900 dark:text-white">Hybrid</div>
            <div className="text-sm text-slate-600 dark:text-slate-400 mt-1">
              60% Levenshtein + 40% Jaro-Winkler
            </div>
          </div>

          <div className="p-6 bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-700">
            <div className="text-primary font-semibold mb-2">Confidence</div>
            <div className="text-2xl font-bold text-slate-900 dark:text-white">82.3%</div>
            <div className="text-sm text-slate-600 dark:text-slate-400 mt-1">
              High confidence repair suggestion
            </div>
          </div>

          <div className="p-6 bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-700">
            <div className="text-primary font-semibold mb-2">Processing Time</div>
            <div className="text-2xl font-bold text-slate-900 dark:text-white">0.32ms</div>
            <div className="text-sm text-slate-600 dark:text-slate-400 mt-1">
              Ultra-fast validation with caching
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
