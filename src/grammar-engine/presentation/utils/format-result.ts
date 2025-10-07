/**
 * Result Formatting Utility
 *
 * Formats processing results for display
 */

import { ProcessingResult, GenericRecord } from '../../domain/entities/types'

/**
 * Format processing result for display
 */
export function formatResult<T extends GenericRecord>(result: ProcessingResult<T>): string {
  const lines: string[] = []

  lines.push("═".repeat(80))
  lines.push("FIAT LUX - Grammar Engine Processing Result")
  lines.push("═".repeat(80))
  lines.push("")
  lines.push("📝 Original Input:")
  lines.push(JSON.stringify(result.original, null, 2))
  lines.push("")

  if (result.isValid) {
    lines.push("✅ Status: VALID")
    lines.push("   All grammatical rules satisfied")
  } else {
    lines.push("❌ Status: INVALID")
    lines.push("")

    if (result.errors.length > 0) {
      lines.push(`   Validation Errors (${result.errors.length}):`)
      result.errors.forEach((error, i) => {
        lines.push(`   ${i + 1}. [${error.severity.toUpperCase()}] ${error.message}`)
        if (error.suggestions && error.suggestions.length > 0) {
          lines.push(`      Suggestions: ${error.suggestions.join(", ")}`)
        }
      })
      lines.push("")
    }

    if (result.structuralErrors.length > 0) {
      lines.push(`   Structural Errors (${result.structuralErrors.length}):`)
      result.structuralErrors.forEach((error, i) => {
        lines.push(`   ${i + 1}. ${error}`)
      })
      lines.push("")
    }

    if (result.repairs && result.repairs.length > 0) {
      lines.push("🔧 Auto-Repair Applied:")
      result.repairs.forEach((repair, i) => {
        lines.push(`   ${i + 1}. ${repair.message}`)
        lines.push(`      Algorithm: ${repair.algorithm}, Confidence: ${(repair.confidence * 100).toFixed(1)}%`)
        if (repair.alternatives && repair.alternatives.length > 0) {
          lines.push(`      Alternatives: ${repair.alternatives.map(a => `${a.value} (${(a.similarity * 100).toFixed(1)}%)`).join(", ")}`)
        }
      })
      lines.push("")
      lines.push("✅ Repaired Output:")
      lines.push(JSON.stringify(result.repaired, null, 2))
      lines.push("")
    }
  }

  lines.push("📊 Metadata:")
  lines.push(`   Processing time: ${result.metadata.processingTimeMs.toFixed(2)}ms`)
  lines.push(`   Cache hits: ${result.metadata.cacheHits}`)
  lines.push(`   Algorithms: ${result.metadata.algorithmsUsed.join(", ")}`)
  lines.push("")
  lines.push("═".repeat(80))

  return lines.join("\n")
}
