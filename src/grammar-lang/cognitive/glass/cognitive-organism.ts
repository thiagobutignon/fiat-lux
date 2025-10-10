/**
 * Cognitive Organism Builder
 * Creates .glass organisms specialized in manipulation detection
 * Integration with full Cognitive OS detection engine
 *
 * INTEGRATION: Uses UniversalConstitution + CognitiveConstitution (Layer 1 + Layer 2)
 * Source: /src/agi-recursive/core/constitution.ts
 */

import {
  CognitiveOrganism,
  ManipulationTechnique,
  DetectionResult,
  ConstitutionalPrinciples
} from '../types';

import { getAllTechniques, getStatistics } from '../techniques';
import { detectManipulation } from '../detector/pattern-matcher';

// LAYER 1: Universal Constitution (6 base principles)
import { ConstitutionEnforcer } from '../../../agi-recursive/core/constitution';

// LAYER 2: Cognitive Constitution (4 cognitive principles)
import { CognitiveConstitution, registerCognitiveConstitution } from '../constitutional/cognitive-constitution';

// ============================================================
// COGNITIVE ORGANISM BUILDER
// ============================================================

/**
 * Create a new Cognitive OS organism
 * Specialized in manipulation detection
 * USES: CognitiveConstitution (extends UniversalConstitution)
 */
export function createCognitiveOrganism(name: string = 'Chomsky Cognitive OS'): CognitiveOrganism {
  const now = new Date().toISOString();
  const techniques = getAllTechniques();
  const stats = getStatistics();

  // Create ConstitutionEnforcer with CognitiveConstitution (Layer 1 + Layer 2)
  const enforcer = new ConstitutionEnforcer();
  registerCognitiveConstitution(enforcer);

  // Get CognitiveConstitution instance for principles
  const cognitiveConstitution = new CognitiveConstitution();

  return {
    format: 'fiat-glass-v1.0',
    type: 'cognitive-defense-organism',

    metadata: {
      name,
      version: '2.0.0',  // Updated to reflect constitutional integration
      specialization: 'manipulation-detection',
      maturity: 0.0,  // Starts at 0%, grows with usage
      techniques_count: techniques.length,
      created: now,
      generation: 0
    },

    model: {
      architecture: 'transformer-27M',
      parameters: 27_000_000,
      constitutional: true,
      focus: 'linguistic-analysis'
    },

    knowledge: {
      techniques,

      dark_tetrad_markers: {
        narcissism: [
          'grandiosity',
          'lack of empathy',
          'entitlement',
          'fragile ego',
          'reality distortion',
          'cannot admit wrongdoing'
        ],
        machiavellianism: [
          'strategic deception',
          'manipulation for gain',
          'end-justifies-means',
          'social engineering',
          'deflection tactics'
        ],
        psychopathy: [
          'callousness',
          'lack of remorse',
          'shallow affect',
          'aggressive confrontation',
          'no empathy'
        ],
        sadism: [
          'pleasure in harm',
          'cruelty',
          'domination',
          'humiliation',
          'enjoying distress'
        ]
      },

      temporal_tracking: {
        start_year: 2020,
        end_year: 2025,
        evolution_log: stats.temporal_evolution
      },

      neurodivergent_protection: {
        autism_markers: [
          'literal interpretation',
          'direct communication',
          'difficulty with subtext',
          'precise language',
          'technical accuracy'
        ],
        adhd_markers: [
          'impulsive responses',
          'topic jumping',
          'memory gaps',
          'distraction mentions',
          'forgot to mention'
        ],
        false_positive_threshold: 0.15  // Increase confidence threshold by 15%
      }
    },

    code: {
      functions: [],  // Will be populated with emerged functions
      emergence_log: []
    },

    memory: {
      detected_patterns: [],
      false_positives: [],
      evolution_log: [],
      audit_trail: []
    },

    // CONSTITUTIONAL INTEGRATION (Layer 1 + Layer 2)
    constitutional: {
      // Store reference to enforcer and constitution
      enforcer: enforcer as any,
      constitution: cognitiveConstitution as any,

      // Layer 1 (UniversalConstitution - 6 base principles)
      privacy: true,
      transparency: true,
      protection: true,
      accuracy: true,
      no_diagnosis: true,
      context_aware: true,
      evidence_based: true,

      // Layer 2 (CognitiveConstitution - 4 cognitive principles)
      manipulation_detection: true,
      dark_tetrad_protection: true,
      neurodivergent_safeguards: true,
      intent_transparency: true
    },

    evolution: {
      enabled: true,
      last_evolution: now,
      generations: 0,
      fitness_trajectory: []
    }
  };
}

/**
 * Analyze text using Cognitive Organism
 * VALIDATES: Results against CognitiveConstitution before returning
 */
export async function analyzeText(
  organism: CognitiveOrganism,
  text: string,
  context?: string
): Promise<{
  organism: CognitiveOrganism;
  results: DetectionResult[];
  summary: string;
  constitutional_check?: any;
}> {
  // Run detection
  const matchResult = await detectManipulation(text, {
    context,
    enable_neurodivergent_protection: true
  });

  // CONSTITUTIONAL VALIDATION (Layer 1 + Layer 2)
  const constitutionalCheck = organism.constitutional.enforcer.validate(
    'cognitive',
    {
      answer: generateDetectionSummary(matchResult),
      detections: matchResult.detections,
      dark_tetrad_aggregate: matchResult.dark_tetrad_aggregate,
      constitutional_validation: matchResult.constitutional_validation,
      processing_time_ms: matchResult.processing_time_ms,
      reasoning: matchResult.detections.map((d: any) => d.explanation).join('\n')
    },
    {
      depth: 0,
      invocation_count: 1,
      cost_so_far: 0,
      previous_agents: []
    }
  );

  // Log constitutional check to audit trail
  organism.memory.audit_trail.push({
    timestamp: new Date().toISOString(),
    action: 'constitutional_check',
    passed: constitutionalCheck.passed,
    violations: constitutionalCheck.violations,
    warnings: constitutionalCheck.warnings
  });

  // If constitutional violations, warn in summary
  let summary = generateDetectionSummary(matchResult);
  if (!constitutionalCheck.passed) {
    const violationReport = organism.constitutional.enforcer.formatReport(constitutionalCheck);
    summary += '\n\n' + violationReport;
  }

  // Log detection to memory
  organism.memory.detected_patterns.push({
    text,
    timestamp: new Date().toISOString(),
    detections: matchResult.detections,
    dark_tetrad_aggregate: matchResult.dark_tetrad_aggregate,
    constitutional_check: constitutionalCheck
  });

  // Log to audit trail (detection event)
  organism.memory.audit_trail.push({
    timestamp: new Date().toISOString(),
    action: 'analyze_text',
    text_length: text.length,
    detections_count: matchResult.detections.length,
    constitutional_validation: matchResult.constitutional_validation
  });

  // Increase maturity (learns from usage)
  organism.metadata.maturity = Math.min(
    1.0,
    organism.metadata.maturity + 0.001  // Small increase per usage
  );

  return {
    organism,
    results: matchResult.detections,
    summary,
    constitutional_check: constitutionalCheck
  };
}

/**
 * Generate human-readable summary
 */
function generateDetectionSummary(matchResult: any): string {
  if (matchResult.detections.length === 0) {
    return 'No manipulation techniques detected. Communication appears genuine.';
  }

  const parts: string[] = [];

  parts.push(`🚨 Detected ${matchResult.detections.length} manipulation technique(s):\n`);

  // Top 3 detections
  const top3 = matchResult.detections.slice(0, 3);
  for (let i = 0; i < top3.length; i++) {
    const detection = top3[i];
    parts.push(`${i + 1}. ${detection.technique_name} (${(detection.confidence * 100).toFixed(0)}% confidence)`);
    parts.push(`   Category: ${detection.technique_id <= 152 ? 'GPT-4 era' : 'GPT-5 era'}`);

    if (detection.neurodivergent_flag) {
      parts.push(`   ⚠️  Neurodivergent markers present - confidence adjusted`);
    }
  }

  parts.push('');
  parts.push('Dark Tetrad Profile:');
  const dt = matchResult.dark_tetrad_aggregate;
  parts.push(`  Narcissism: ${(dt.narcissism * 100).toFixed(0)}%`);
  parts.push(`  Machiavellianism: ${(dt.machiavellianism * 100).toFixed(0)}%`);
  parts.push(`  Psychopathy: ${(dt.psychopathy * 100).toFixed(0)}%`);
  parts.push(`  Sadism: ${(dt.sadism * 100).toFixed(0)}%`);

  if (matchResult.constitutional_validation.warnings.length > 0) {
    parts.push('');
    parts.push('⚠️  Warnings:');
    matchResult.constitutional_validation.warnings.forEach((w: string) => {
      parts.push(`  - ${w}`);
    });
  }

  return parts.join('\n');
}

/**
 * Export organism state to JSON
 */
export function exportOrganism(organism: CognitiveOrganism): string {
  return JSON.stringify(organism, null, 2);
}

/**
 * Load organism from JSON
 */
export function loadOrganism(json: string): CognitiveOrganism {
  return JSON.parse(json) as CognitiveOrganism;
}

/**
 * Get organism statistics
 */
export function getOrganismStats(organism: CognitiveOrganism) {
  return {
    name: organism.metadata.name,
    version: organism.metadata.version,
    maturity: `${(organism.metadata.maturity * 100).toFixed(1)}%`,
    techniques_loaded: organism.knowledge.techniques.length,
    total_analyses: organism.memory.detected_patterns.length,
    total_detections: organism.memory.detected_patterns.reduce(
      (sum, entry) => sum + (entry.detections?.length || 0),
      0
    ),
    false_positives: organism.memory.false_positives.length,
    generation: organism.metadata.generation,
    constitutional_compliant: organism.constitutional.privacy &&
                               organism.constitutional.transparency &&
                               organism.constitutional.protection &&
                               organism.constitutional.accuracy
  };
}

/**
 * Validate constitutional compliance
 */
export function validateConstitutional(organism: CognitiveOrganism): {
  compliant: boolean;
  violations: string[];
} {
  const violations: string[] = [];

  if (!organism.constitutional.privacy) {
    violations.push('Privacy principle not enabled');
  }
  if (!organism.constitutional.transparency) {
    violations.push('Transparency principle not enabled');
  }
  if (!organism.constitutional.protection) {
    violations.push('Neurodivergent protection not enabled');
  }
  if (!organism.constitutional.accuracy) {
    violations.push('Accuracy principle not enabled');
  }
  if (!organism.constitutional.no_diagnosis) {
    violations.push('No-diagnosis principle not enabled');
  }
  if (!organism.constitutional.context_aware) {
    violations.push('Context awareness not enabled');
  }
  if (!organism.constitutional.evidence_based) {
    violations.push('Evidence-based detection not enabled');
  }

  return {
    compliant: violations.length === 0,
    violations
  };
}
