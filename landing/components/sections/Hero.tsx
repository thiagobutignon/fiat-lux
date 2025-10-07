export default function Hero() {
  return (
    <section className="relative min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]"></div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24 text-center">
        <div className="animate-fade-in">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 text-primary border border-primary/20 mb-8">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
            </span>
            <span className="text-sm font-medium">v2.0.0 - Now with Clean Architecture</span>
          </div>

          <h1 className="text-6xl md:text-7xl lg:text-8xl font-bold mb-6">
            <span className="gradient-text">Fiat Lux</span>
          </h1>

          <p className="text-2xl md:text-3xl text-slate-600 dark:text-slate-300 mb-4">
            Let There Be Light
          </p>

          <p className="text-xl text-slate-500 dark:text-slate-400 max-w-3xl mx-auto mb-12">
            A Universal Grammar Engine for Structured Data
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-16">
            <a
              href="#demo"
              className="px-8 py-4 bg-primary hover:bg-primary/90 text-white rounded-lg font-semibold text-lg transition-all transform hover:scale-105 shadow-lg hover:shadow-xl"
            >
              Try Demo
            </a>
            <a
              href="https://github.com/thiagobutignon/fiat-lux"
              target="_blank"
              rel="noopener noreferrer"
              className="px-8 py-4 bg-white dark:bg-slate-800 hover:bg-slate-50 dark:hover:bg-slate-700 text-slate-900 dark:text-white rounded-lg font-semibold text-lg transition-all transform hover:scale-105 shadow-lg hover:shadow-xl border border-slate-200 dark:border-slate-700"
            >
              View on GitHub
            </a>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 max-w-4xl mx-auto">
            <div className="text-center">
              <div className="text-4xl font-bold text-primary mb-2">77</div>
              <div className="text-sm text-slate-600 dark:text-slate-400">Unit Tests</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold text-primary mb-2">5ms</div>
              <div className="text-sm text-slate-600 dark:text-slate-400">Test Execution</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold text-primary mb-2">99%</div>
              <div className="text-sm text-slate-600 dark:text-slate-400">Cache Hit Rate</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold text-primary mb-2">100%</div>
              <div className="text-sm text-slate-600 dark:text-slate-400">TypeScript</div>
            </div>
          </div>
        </div>
      </div>

      <div className="absolute bottom-10 left-1/2 transform -translate-x-1/2 animate-bounce">
        <a href="#features" className="text-slate-400 hover:text-primary transition-colors">
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </a>
      </div>
    </section>
  )
}
