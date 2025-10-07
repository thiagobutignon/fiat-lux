export default function Features() {
  const features = [
    {
      icon: '🎯',
      title: 'Generic & Configurable',
      description: 'Instantiate grammar engines with custom configurations. Define roles, allowed values, and validation rules for any domain.',
    },
    {
      icon: '🔍',
      title: 'Multiple Similarity Algorithms',
      description: 'Choose from Levenshtein Distance, Jaro-Winkler, or Hybrid algorithms for optimal typo detection and correction.',
    },
    {
      icon: '⚡',
      title: 'Performance Optimized',
      description: 'Built-in caching with 99% hit rate after warm-up. Process thousands of validations in milliseconds.',
    },
    {
      icon: '🔧',
      title: 'Advanced Auto-Repair',
      description: 'Configurable similarity thresholds with multiple repair suggestions ranked by confidence scores.',
    },
    {
      icon: '🏗️',
      title: 'Clean Architecture',
      description: 'Feature-based organization following Clean Architecture principles with clear separation of concerns.',
    },
    {
      icon: '📊',
      title: 'Full Type Safety',
      description: 'Complete TypeScript support with generics, comprehensive interfaces, and detailed metadata tracking.',
    },
  ]

  return (
    <section id="features" className="py-24 bg-white dark:bg-slate-900">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-4 gradient-text">
            Features
          </h2>
          <p className="text-xl text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
            Everything you need for grammar validation and auto-repair
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <div
              key={index}
              className="p-8 rounded-2xl bg-slate-50 dark:bg-slate-800 hover:shadow-xl transition-all duration-300 transform hover:-translate-y-2 border border-slate-200 dark:border-slate-700"
            >
              <div className="text-5xl mb-4">{feature.icon}</div>
              <h3 className="text-2xl font-bold mb-3 text-slate-900 dark:text-white">
                {feature.title}
              </h3>
              <p className="text-slate-600 dark:text-slate-400 leading-relaxed">
                {feature.description}
              </p>
            </div>
          ))}
        </div>

        <div className="mt-16 text-center">
          <div className="inline-flex flex-col md:flex-row gap-4 items-center p-6 bg-primary/5 dark:bg-primary/10 rounded-2xl border border-primary/20">
            <div className="flex-1">
              <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-2">
                Built with Claude Code
              </h3>
              <p className="text-slate-600 dark:text-slate-400">
                Developed using Anthropic's official CLI for Claude
              </p>
            </div>
            <a
              href="https://claude.com/claude-code"
              target="_blank"
              rel="noopener noreferrer"
              className="px-6 py-3 bg-primary hover:bg-primary/90 text-white rounded-lg font-semibold transition-all transform hover:scale-105"
            >
              Learn More
            </a>
          </div>
        </div>
      </div>
    </section>
  )
}
