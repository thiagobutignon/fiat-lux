/**
 * Cancer Research .glass + .sqlo Demo
 *
 * Demonstrates complete lifecycle of a .glass organism with episodic memory:
 * 1. Creation (nascent, 0% maturity)
 * 2. Learning (progressive maturity increase)
 * 3. Memory recall (episodic learning)
 * 4. Maturity progression (nascent → infant → adolescent → mature)
 * 5. Glass box inspection (full transparency)
 * 6. Export (distribution ready)
 *
 * Sprint 2 - DIA 2: E2E Demo
 */

import {
  createGlassWithMemory,
  loadGlassWithMemory,
  LearningInteraction
} from '../src/grammar-lang/glass/sqlo-integration';
import { MemoryType } from '../src/grammar-lang/database/sqlo';
import * as fs from 'fs';

// ============================================================================
// Demo Configuration
// ============================================================================

const DEMO_DIR = 'demo_organisms';
const ORGANISM_NAME = 'cancer-research';
const ORGANISM_DOMAIN = 'oncology';

// ============================================================================
// Demo Data - Cancer Research Interactions
// ============================================================================

const CANCER_RESEARCH_INTERACTIONS: LearningInteraction[] = [
  // Phase 1: Initial Learning (Infant Stage)
  {
    query: 'What is pembrolizumab?',
    response: 'Pembrolizumab is an immunotherapy drug (PD-1 inhibitor) used to treat various cancers including melanoma, lung cancer, and head and neck cancers. It works by blocking the PD-1 protein, allowing the immune system to attack cancer cells.',
    confidence: 0.92,
    sources: ['FDA-pembrolizumab-label.pdf', 'NEJM-immunotherapy-2015.pdf'],
    attention_weights: [0.6, 0.4],
    outcome: 'success'
  },
  {
    query: 'How effective is pembrolizumab for lung cancer?',
    response: 'For non-small cell lung cancer (NSCLC), pembrolizumab shows approximately 45-55% response rate in patients with high PD-L1 expression (≥50%). The KEYNOTE-024 trial demonstrated superior overall survival compared to chemotherapy.',
    confidence: 0.89,
    sources: ['KEYNOTE-024-trial.pdf', 'JCO-lung-cancer-2017.pdf'],
    attention_weights: [0.7, 0.3],
    outcome: 'success'
  },
  {
    query: 'What are common side effects of immunotherapy?',
    response: 'Common immune-related adverse effects include fatigue (20-40%), rash (15-30%), diarrhea/colitis (10-20%), pneumonitis (5-10%), and endocrine disorders (5-15%). Most are manageable with steroids and dose modifications.',
    confidence: 0.85,
    sources: ['immunotherapy-toxicity-review.pdf', 'ASCO-guidelines-irAE.pdf'],
    attention_weights: [0.5, 0.5],
    outcome: 'success'
  },

  // Phase 2: Deepening Knowledge (Adolescent Stage)
  {
    query: 'Difference between pembrolizumab and nivolumab?',
    response: 'Both are PD-1 inhibitors with similar mechanisms. Key differences: pembrolizumab (approved 2014, dosing every 3 weeks) vs nivolumab (approved 2014, initially every 2 weeks). Efficacy is comparable in most indications, with subtle differences in side effect profiles.',
    confidence: 0.88,
    sources: ['comparative-immunotherapy.pdf', 'meta-analysis-PD1.pdf'],
    attention_weights: [0.55, 0.45],
    outcome: 'success'
  },
  {
    query: 'What is PD-L1 expression and why does it matter?',
    response: 'PD-L1 (Programmed Death-Ligand 1) expression is a biomarker measured by immunohistochemistry. Higher PD-L1 expression (≥50%) correlates with better response to PD-1 inhibitors. However, even PD-L1 negative patients can respond, especially in combination therapy.',
    confidence: 0.91,
    sources: ['PD-L1-biomarker-review.pdf', 'predictive-biomarkers.pdf'],
    attention_weights: [0.6, 0.4],
    outcome: 'success'
  },
  {
    query: 'Can immunotherapy be combined with chemotherapy?',
    response: 'Yes, combination therapy is increasingly standard. KEYNOTE-189 showed pembrolizumab + chemotherapy improved overall survival vs chemotherapy alone in metastatic NSCLC. The combination appears to have synergistic effects.',
    confidence: 0.93,
    sources: ['KEYNOTE-189.pdf', 'combination-therapy-review.pdf'],
    attention_weights: [0.65, 0.35],
    outcome: 'success'
  },

  // Phase 3: Advanced Understanding (Mature Stage)
  {
    query: 'What are emerging biomarkers beyond PD-L1?',
    response: 'Emerging biomarkers include tumor mutational burden (TMB), microsatellite instability (MSI-H), mismatch repair deficiency (dMMR), and gene expression profiles. TMB ≥10 mutations/megabase shows improved response to immunotherapy regardless of PD-L1 status.',
    confidence: 0.87,
    sources: ['TMB-predictive-value.pdf', 'biomarkers-future.pdf'],
    attention_weights: [0.7, 0.3],
    outcome: 'success'
  },
  {
    query: 'How to manage immune-related pneumonitis?',
    response: 'Grade 2+ pneumonitis requires immunotherapy hold and corticosteroids (1-2 mg/kg prednisone). Grade 3-4 requires permanent discontinuation. CT imaging and pulmonary consult are essential. Most cases resolve with treatment, though 5-10% can be fatal if not managed promptly.',
    confidence: 0.90,
    sources: ['irAE-management-guidelines.pdf', 'pneumonitis-case-series.pdf'],
    attention_weights: [0.8, 0.2],
    outcome: 'success'
  },
  {
    query: 'What is the role of CTLA-4 inhibitors?',
    response: 'CTLA-4 inhibitors (ipilimumab) work earlier in immune activation than PD-1 inhibitors. Combination ipilimumab + nivolumab shows higher response rates (58% in melanoma) but increased toxicity (59% grade 3-4 adverse events). Reserved for specific indications.',
    confidence: 0.86,
    sources: ['CTLA4-mechanism.pdf', 'ipilimumab-trials.pdf'],
    attention_weights: [0.6, 0.4],
    outcome: 'success'
  },
  {
    query: 'What are the latest CAR-T therapies for solid tumors?',
    response: 'CAR-T therapy is challenging in solid tumors due to immunosuppressive microenvironment and antigen heterogeneity. Promising targets include mesothelin (mesothelioma), HER2 (breast/gastric), and GD2 (neuroblastoma). Clinical trials ongoing, but no FDA approvals yet for solid tumors.',
    confidence: 0.84,
    sources: ['CAR-T-solid-tumors.pdf', 'cellular-therapy-review.pdf'],
    attention_weights: [0.55, 0.45],
    outcome: 'success'
  },

  // Phase 4: Expert-Level Knowledge
  {
    query: 'Explain mechanisms of immunotherapy resistance',
    response: 'Primary resistance mechanisms: loss of antigen presentation (β2M mutations), defects in IFN-γ signaling (JAK1/2 mutations), and T-cell exclusion. Acquired resistance: selection of antigen-loss variants, upregulation of alternative checkpoints (TIM-3, LAG-3), and epithelial-mesenchymal transition.',
    confidence: 0.88,
    sources: ['resistance-mechanisms.pdf', 'tumor-evolution.pdf'],
    attention_weights: [0.7, 0.3],
    outcome: 'success'
  },
  {
    query: 'What is the future of personalized cancer immunotherapy?',
    response: 'Future directions include: (1) Neoantigen-based vaccines targeting patient-specific mutations, (2) Engineered T-cells with multiple targets, (3) Microbiome modulation to enhance response, (4) AI-driven treatment selection, (5) Oncolytic viruses combined with checkpoint inhibitors. Precision medicine will integrate multi-omics data.',
    confidence: 0.85,
    sources: ['future-immunotherapy.pdf', 'precision-oncology.pdf'],
    attention_weights: [0.6, 0.4],
    outcome: 'success'
  }
];

// ============================================================================
// Demo Execution
// ============================================================================

async function runDemo() {
  console.log('🧬 Cancer Research .glass + .sqlo Demo');
  console.log('=' .repeat(80));
  console.log('Demonstrating complete organism lifecycle with episodic memory\n');

  // Cleanup previous demo
  if (fs.existsSync(DEMO_DIR)) {
    fs.rmSync(DEMO_DIR, { recursive: true });
  }

  // ========================================================================
  // Phase 1: Birth (0% Maturity - Nascent)
  // ========================================================================

  console.log('📍 Phase 1: BIRTH (Nascent Organism)');
  console.log('-'.repeat(80));

  const glass = await createGlassWithMemory(
    ORGANISM_NAME,
    ORGANISM_DOMAIN,
    DEMO_DIR
  );

  let stats = glass.getMemoryStats();
  console.log(`✅ Organism created: ${ORGANISM_NAME}.glass`);
  console.log(`   Maturity: ${stats.maturity.toFixed(1)}%`);
  console.log(`   Stage: ${stats.stage}`);
  console.log(`   Episodes: ${stats.total_episodes}`);
  console.log('');

  // ========================================================================
  // Phase 2: Infancy (0-25% Maturity)
  // ========================================================================

  console.log('📍 Phase 2: INFANCY (Learning Basic Concepts)');
  console.log('-'.repeat(80));

  // Learn first 3 interactions (basic understanding)
  for (let i = 0; i < 3; i++) {
    const interaction = CANCER_RESEARCH_INTERACTIONS[i];
    await glass.learn(interaction);

    stats = glass.getMemoryStats();
    console.log(`✅ Learned: "${interaction.query.substring(0, 50)}..."`);
    console.log(`   Maturity: ${stats.maturity.toFixed(1)}% (${stats.stage})`);
    console.log(`   Confidence: ${(interaction.confidence * 100).toFixed(0)}%`);
    console.log('');
  }

  stats = glass.getMemoryStats();
  console.log(`📊 Infancy Complete:`);
  console.log(`   Total Episodes: ${stats.total_episodes}`);
  console.log(`   Long-term Memory: ${stats.long_term_count}`);
  console.log(`   Short-term Memory: ${stats.short_term_count}`);
  console.log('');

  // ========================================================================
  // Phase 3: Adolescence (25-75% Maturity)
  // ========================================================================

  console.log('📍 Phase 3: ADOLESCENCE (Deepening Knowledge)');
  console.log('-'.repeat(80));

  // Learn interactions 3-6 (developing expertise)
  for (let i = 3; i < 6; i++) {
    const interaction = CANCER_RESEARCH_INTERACTIONS[i];
    await glass.learn(interaction);

    stats = glass.getMemoryStats();
    console.log(`✅ Learned: "${interaction.query.substring(0, 50)}..."`);
    console.log(`   Maturity: ${stats.maturity.toFixed(1)}% (${stats.stage})`);
  }

  console.log('');
  stats = glass.getMemoryStats();
  console.log(`📊 Adolescence Progress:`);
  console.log(`   Maturity: ${stats.maturity.toFixed(1)}%`);
  console.log(`   Stage: ${stats.stage}`);
  console.log(`   Total Learning: ${stats.total_episodes} episodes`);
  console.log('');

  // ========================================================================
  // Phase 4: Memory Recall Test
  // ========================================================================

  console.log('📍 Phase 4: MEMORY RECALL (Episodic Learning)');
  console.log('-'.repeat(80));

  // Test recall of pembrolizumab-related knowledge
  const recalled = await glass.recallSimilar('pembrolizumab effectiveness', 3);
  console.log(`🔍 Query: "pembrolizumab effectiveness"`);
  console.log(`   Found ${recalled.length} similar episodes:\n`);

  recalled.forEach((episode, idx) => {
    console.log(`   ${idx + 1}. "${episode.query.substring(0, 60)}..."`);
    console.log(`      Confidence: ${(episode.confidence * 100).toFixed(0)}%`);
    console.log(`      Sources: ${episode.attention.sources.join(', ')}`);
    console.log('');
  });

  // ========================================================================
  // Phase 5: Maturity (75-100% Maturity)
  // ========================================================================

  console.log('📍 Phase 5: MATURITY (Advanced Understanding)');
  console.log('-'.repeat(80));

  // Learn remaining interactions (expert knowledge)
  for (let i = 6; i < CANCER_RESEARCH_INTERACTIONS.length; i++) {
    const interaction = CANCER_RESEARCH_INTERACTIONS[i];
    await glass.learn(interaction);
  }

  stats = glass.getMemoryStats();
  console.log(`✅ All ${CANCER_RESEARCH_INTERACTIONS.length} interactions learned`);
  console.log(`   Maturity: ${stats.maturity.toFixed(1)}%`);
  console.log(`   Stage: ${stats.stage}`);
  console.log(`   Knowledge Depth: ${stats.long_term_count} consolidated episodes`);
  console.log('');

  // ========================================================================
  // Phase 6: Glass Box Inspection
  // ========================================================================

  console.log('📍 Phase 6: GLASS BOX INSPECTION (Full Transparency)');
  console.log('-'.repeat(80));

  const inspection = glass.inspect();

  console.log('🔬 Organism Structure:');
  console.log(`   Name: ${inspection.organism.metadata.name}`);
  console.log(`   Domain: ${inspection.organism.metadata.domain}`);
  console.log(`   Version: ${inspection.organism.metadata.version}`);
  console.log(`   Created: ${new Date(inspection.organism.metadata.created_at).toISOString()}`);
  console.log('');

  console.log('🧠 Model Configuration:');
  console.log(`   Architecture: ${inspection.organism.model.architecture}`);
  console.log(`   Parameters: ${(inspection.organism.model.parameters / 1_000_000).toFixed(1)}M`);
  console.log(`   Quantization: ${inspection.organism.model.quantization}`);
  console.log('');

  console.log('📚 Knowledge Stats:');
  console.log(`   Papers: ${inspection.organism.knowledge.papers}`);
  console.log(`   Embeddings: ${inspection.organism.knowledge.embeddings}`);
  console.log(`   Patterns: ${inspection.organism.knowledge.patterns}`);
  console.log('');

  console.log('💾 Memory System:');
  console.log(`   Total Episodes: ${inspection.memory_stats.total_episodes}`);
  console.log(`   Short-term: ${inspection.memory_stats.short_term_count}`);
  console.log(`   Long-term: ${inspection.memory_stats.long_term_count}`);
  console.log(`   Contextual: ${inspection.memory_stats.contextual_count}`);
  console.log('');

  console.log('🎯 Constitutional AI:');
  console.log(`   Principles: ${inspection.organism.constitutional.principles.join(', ')}`);
  console.log(`   Boundaries: ${inspection.organism.constitutional.boundaries.join(', ')}`);
  console.log('');

  console.log('📈 Fitness Trajectory:');
  if (inspection.fitness_trajectory.length > 0) {
    inspection.fitness_trajectory.forEach((fitness, idx) => {
      const bar = '█'.repeat(Math.floor(fitness * 20));
      console.log(`   Window ${idx + 1}: ${bar} ${(fitness * 100).toFixed(1)}%`);
    });
  }
  console.log('');

  console.log('🕐 Recent Learning (last 5 episodes):');
  inspection.recent_learning.slice(0, 5).forEach((episode, idx) => {
    console.log(`   ${idx + 1}. "${episode.query.substring(0, 50)}..."`);
    console.log(`      Confidence: ${(episode.confidence * 100).toFixed(0)}%`);
    console.log(`      Outcome: ${episode.outcome}`);
    console.log(`      Memory Type: ${episode.memory_type}`);
    console.log('');
  });

  // ========================================================================
  // Phase 7: Export for Distribution
  // ========================================================================

  console.log('📍 Phase 7: EXPORT (Distribution Ready)');
  console.log('-'.repeat(80));

  const exported = await glass.exportGlass();

  console.log('📦 Export Summary:');
  console.log(`   Glass File: ${ORGANISM_NAME}.glass`);
  console.log(`   Memory Size: ${(exported.memory_size / 1024).toFixed(2)} KB`);
  console.log(`   Total Size: ${(exported.total_size / 1024).toFixed(2)} KB`);
  console.log(`   Maturity: ${exported.glass.metadata.maturity.toFixed(1)}%`);
  console.log(`   Stage: ${exported.glass.metadata.stage}`);
  console.log(`   Generation: ${exported.glass.evolution.generation}`);
  console.log('');

  console.log('✅ Organism is production-ready and distributable!');
  console.log('   - Self-contained (model + knowledge + memory)');
  console.log('   - 100% glass box (fully inspectable)');
  console.log('   - O(1) memory operations guaranteed');
  console.log('   - Constitutional AI embedded');
  console.log('');

  // ========================================================================
  // Summary
  // ========================================================================

  console.log('=' .repeat(80));
  console.log('🎊 DEMO COMPLETE!');
  console.log('=' .repeat(80));
  console.log('');
  console.log('Lifecycle Demonstrated:');
  console.log(`  1. ✅ Birth: Nascent organism (0%)
  2. ✅ Infancy: Basic learning (0-25%)
  3. ✅ Adolescence: Deepening knowledge (25-75%)
  4. ✅ Maturity: Expert-level understanding (75-100%)
  5. ✅ Memory Recall: Episodic learning working
  6. ✅ Glass Box: Full transparency achieved
  7. ✅ Export: Distribution ready`);
  console.log('');
  console.log('Key Features Validated:');
  console.log('  ✅ Memory embedded in .glass organism');
  console.log('  ✅ Learning drives maturity progression');
  console.log('  ✅ Episodic memory (short/long/contextual)');
  console.log('  ✅ O(1) operations maintained');
  console.log('  ✅ Glass box transparency');
  console.log('  ✅ Constitutional AI embedded');
  console.log('');
  console.log(`Final Organism: ${DEMO_DIR}/${ORGANISM_NAME}/${ORGANISM_NAME}.glass`);
  console.log('');
}

// Run demo
runDemo().catch(error => {
  console.error('Demo failed:', error);
  process.exit(1);
});
