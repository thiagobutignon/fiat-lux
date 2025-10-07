export default function Footer() {
  return (
    <footer className="bg-white dark:bg-slate-900 border-t border-slate-200 dark:border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-12">
          <div className="col-span-1 md:col-span-2">
            <h3 className="text-3xl font-bold gradient-text mb-4">
              Fiat Lux
            </h3>
            <p className="text-slate-600 dark:text-slate-400 mb-6 max-w-md">
              A Universal Grammar Engine for Structured Data. Built with Clean Architecture principles and TypeScript.
            </p>
            <div className="flex gap-4">
              <a
                href="https://github.com/thiagobutignon/fiat-lux"
                target="_blank"
                rel="noopener noreferrer"
                className="p-3 bg-slate-100 dark:bg-slate-800 hover:bg-primary hover:text-white dark:hover:bg-primary rounded-lg transition-colors"
                aria-label="GitHub"
              >
                <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                </svg>
              </a>
              <a
                href="https://www.npmjs.com/package/fiat-lux"
                target="_blank"
                rel="noopener noreferrer"
                className="p-3 bg-slate-100 dark:bg-slate-800 hover:bg-primary hover:text-white dark:hover:bg-primary rounded-lg transition-colors"
                aria-label="NPM"
              >
                <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M0 7.334v8h6.666v1.332H12v-1.332h12v-8H0zm6.666 6.664H5.334v-4H3.999v4H1.335V8.667h5.331v5.331zm4 0v1.336H8.001V8.667h5.334v5.332h-2.669v-.001zm12.001 0h-1.33v-4h-1.336v4h-1.335v-4h-1.33v4h-2.671V8.667h8.002v5.331zM10.665 10H12v2.667h-1.335V10z"/>
                </svg>
              </a>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-slate-900 dark:text-white mb-4">Resources</h4>
            <ul className="space-y-3">
              <li>
                <a href="https://github.com/thiagobutignon/fiat-lux#readme" target="_blank" rel="noopener noreferrer" className="text-slate-600 dark:text-slate-400 hover:text-primary transition-colors">
                  Documentation
                </a>
              </li>
              <li>
                <a href="https://github.com/thiagobutignon/fiat-lux/tree/main/docs" target="_blank" rel="noopener noreferrer" className="text-slate-600 dark:text-slate-400 hover:text-primary transition-colors">
                  Grammar Analysis
                </a>
              </li>
              <li>
                <a href="https://github.com/thiagobutignon/fiat-lux/issues" target="_blank" rel="noopener noreferrer" className="text-slate-600 dark:text-slate-400 hover:text-primary transition-colors">
                  Issue Tracker
                </a>
              </li>
              <li>
                <a href="https://github.com/thiagobutignon/fiat-lux/discussions" target="_blank" rel="noopener noreferrer" className="text-slate-600 dark:text-slate-400 hover:text-primary transition-colors">
                  Discussions
                </a>
              </li>
            </ul>
          </div>

          <div>
            <h4 className="font-semibold text-slate-900 dark:text-white mb-4">Acknowledgements</h4>
            <ul className="space-y-3 text-sm text-slate-600 dark:text-slate-400">
              <li>@rmanguinho</li>
              <li>@barreirabruno</li>
              <li>@lnmunhoz</li>
              <li>@kidchenko</li>
              <li>Hernane Gomes</li>
              <li>Rebecca Barbosa</li>
              <li>Miller Cesar Oliveira</li>
            </ul>
          </div>
        </div>

        <div className="mt-12 pt-8 border-t border-slate-200 dark:border-slate-800">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <p className="text-slate-600 dark:text-slate-400 text-sm text-center md:text-left">
              © 2025 Fiat Lux. Built with <a href="https://claude.com/claude-code" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">Claude Code</a> by Anthropic.
            </p>
            <p className="text-slate-600 dark:text-slate-400 text-sm">
              MIT License
            </p>
          </div>
        </div>
      </div>
    </footer>
  )
}
