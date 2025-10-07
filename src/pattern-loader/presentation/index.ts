/**
 * Pattern Loader Public API
 */

import { readFileSync } from 'fs'
import { PatternLoader } from '../domain/use-cases/pattern-loader'

export * from '../domain/entities/types'
export { PatternLoader } from '../domain/use-cases/pattern-loader'

/**
 * Load patterns from YAML file
 */
export function loadPatternsFromFile(filePath: string): PatternLoader {
  const content = readFileSync(filePath, 'utf-8')
  return new PatternLoader(content)
}
