/**
 * Visual Debugger Demo
 *
 * Demonstrates how the Visual Debugger transforms the AGI from "black box" to "glass box"
 * by providing complete visibility into the reasoning flow.
 *
 * Features demonstrated:
 * 1. Explanation Layer: decision paths, confidence flows, concept activation
 * 2. Concept Attribution: track which concepts contributed to decisions
 * 3. Counterfactual Reasoning: "what if we didn't have X?"
 * 4. Interactive Visualizations: agent graphs, timelines, heatmaps
 *
 * This is innovation #24: "Deixar de Ser Caixa Preta" (Stop Being Black Box)
 */

import { createVisualDebugger } from '../core/visual-debugger';

function runVisualDebuggerDemo() {
  console.log('\n' + '='.repeat(80));
  console.log('🔍 VISUAL DEBUGGER DEMO');
  console.log('Innovation #24: "Deixar de Ser Caixa Preta" (Stop Being Black Box)');
  console.log('='.repeat(80) + '\n');

  // Create debugger instance
  const vDebugger = createVisualDebugger();

  // Simulate a query session
  const query = 'How should I optimize my monthly budget to save for retirement?';
  console.log(`📝 Query: "${query}"\n`);

  vDebugger.startSession(query);

  // Simulate meta-agent decomposition
  vDebugger.trackDecision({
    id: 'decision-1',
    agent: 'meta-agent',
    timestamp: new Date(),
    input: query,
    output: 'Decomposing into sub-queries: financial planning, biological aging, systems optimization',
    confidence: 0.85,
    reasoning: 'Query requires cross-domain analysis',
    concepts_used: ['decomposition', 'cross-domain-reasoning']
  });

  vDebugger.trackConceptActivation({
    concept: 'query-decomposition',
    slice: 'meta/orchestration',
    activation_strength: 0.9,
    timestamp: new Date(),
    context: 'Meta-agent planning',
    contribution_to_output: 0.8
  });

  // Simulate financial agent
  vDebugger.trackDecision({
    id: 'decision-2',
    agent: 'financial-agent',
    timestamp: new Date(),
    input: 'Analyze budget optimization for retirement savings',
    output: 'Recommend 50/30/20 rule: 50% essentials, 30% discretionary, 20% savings',
    confidence: 0.90,
    reasoning: 'Based on personal finance best practices',
    concepts_used: ['budgeting', 'compound-interest', 'retirement-planning'],
    parent_id: 'decision-1'
  });

  vDebugger.trackConceptActivation({
    concept: 'compound-interest',
    slice: 'financial/savings',
    activation_strength: 0.95,
    timestamp: new Date(),
    context: 'Retirement savings calculation',
    contribution_to_output: 0.9
  });

  vDebugger.trackConceptActivation({
    concept: '50-30-20-rule',
    slice: 'financial/budgeting',
    activation_strength: 0.85,
    timestamp: new Date(),
    context: 'Budget allocation strategy',
    contribution_to_output: 0.7
  });

  vDebugger.trackConfidenceFlow({
    node_id: 'decision-2',
    confidence_in: 0.6,
    confidence_out: 0.9,
    confidence_delta: 0.3,
    factors: [
      { concept: 'compound-interest', weight: 0.6, contribution: 0.2 },
      { concept: '50-30-20-rule', weight: 0.4, contribution: 0.1 }
    ]
  });

  // Simulate biology agent (homeostasis analogy)
  vDebugger.trackDecision({
    id: 'decision-3',
    agent: 'biology-agent',
    timestamp: new Date(),
    input: 'Analyze budget as biological system',
    output: 'Budget needs homeostatic control: set point (target savings), sensor (tracking), corrector (automatic adjustments)',
    confidence: 0.82,
    reasoning: 'Homeostasis principle applies to financial regulation',
    concepts_used: ['homeostasis', 'negative-feedback', 'set-point-regulation'],
    parent_id: 'decision-1'
  });

  vDebugger.trackConceptActivation({
    concept: 'homeostasis',
    slice: 'biology/regulation',
    activation_strength: 0.88,
    timestamp: new Date(),
    context: 'Budget control system',
    contribution_to_output: 0.75
  });

  vDebugger.trackConfidenceFlow({
    node_id: 'decision-3',
    confidence_in: 0.5,
    confidence_out: 0.82,
    confidence_delta: 0.32,
    factors: [
      { concept: 'homeostasis', weight: 0.7, contribution: 0.25 },
      { concept: 'negative-feedback', weight: 0.3, contribution: 0.07 }
    ]
  });

  // Simulate systems agent
  vDebugger.trackDecision({
    id: 'decision-4',
    agent: 'systems-agent',
    timestamp: new Date(),
    input: 'Identify leverage points for budget optimization',
    output: 'Key leverage: automate savings before discretionary spending (system structure change)',
    confidence: 0.88,
    reasoning: 'Structural changes > parameter adjustments',
    concepts_used: ['leverage-points', 'system-structure', 'feedback-loops'],
    parent_id: 'decision-1'
  });

  vDebugger.trackConceptActivation({
    concept: 'leverage-points',
    slice: 'systems/thinking',
    activation_strength: 0.92,
    timestamp: new Date(),
    context: 'System optimization strategy',
    contribution_to_output: 0.85
  });

  // Track alternative path that was considered
  vDebugger.trackAlternativePath({
    id: 'alt-1',
    description: 'Use psychology agent to analyze spending habits and emotional triggers',
    confidence: 0.65,
    reason_not_taken: 'Query focuses on optimization, not behavioral change',
    would_require: ['psychology/behavior-change', 'psychology/motivation']
  });

  vDebugger.trackAlternativePath({
    id: 'alt-2',
    description: 'Use economics agent for macroeconomic retirement planning',
    confidence: 0.60,
    reason_not_taken: 'Personal budgeting is microeconomic, not macro',
    would_require: ['economics/macro-planning']
  });

  // Final synthesis
  vDebugger.trackDecision({
    id: 'decision-5',
    agent: 'meta-agent',
    timestamp: new Date(),
    input: 'Synthesize insights from all agents',
    output: 'Implement homeostatic budget system: 1) Set savings target (20% = set point), 2) Automate tracking (sensor), 3) Auto-transfer to savings before discretionary spending (structural leverage), 4) Review monthly and adjust (negative feedback)',
    confidence: 0.93,
    reasoning: 'Emergent synthesis of financial, biological, and systems insights',
    concepts_used: ['synthesis', 'cross-domain-integration'],
    parent_id: 'decision-1'
  });

  vDebugger.trackConfidenceFlow({
    node_id: 'decision-5',
    confidence_in: 0.85,
    confidence_out: 0.93,
    confidence_delta: 0.08,
    factors: [
      { concept: 'homeostasis', weight: 0.35, contribution: 0.03 },
      { concept: 'leverage-points', weight: 0.35, contribution: 0.03 },
      { concept: 'compound-interest', weight: 0.30, contribution: 0.02 }
    ]
  });

  // Capture snapshot
  const snapshot = vDebugger.captureSnapshot(query);

  // Display visualizations
  console.log(vDebugger.visualizeAgentGraph(snapshot.agent_graph));
  console.log(vDebugger.visualizeTimeline(snapshot.timeline));
  console.log(vDebugger.visualizeConceptHeatmap(snapshot.explanation.concept_activation));

  // Show top influencers
  console.log('\n🌟 Top 5 Most Influential Concepts');
  console.log('═'.repeat(60) + '\n');
  const top = vDebugger.getTopInfluencers(snapshot, 5);
  for (let i = 0; i < top.length; i++) {
    const concept = top[i];
    console.log(`${i + 1}. ${concept.concept}`);
    console.log(`   Contribution: ${(concept.contribution_to_output * 100).toFixed(1)}%`);
    console.log(`   From: ${concept.slice}`);
    console.log(`   Context: ${concept.context}\n`);
  }

  // Show critical path
  console.log('\n🎯 Critical Decision Path (High Confidence ≥70%)');
  console.log('═'.repeat(60) + '\n');
  const critical = vDebugger.getCriticalPath(snapshot);
  for (const decision of critical) {
    console.log(`• ${decision.agent}: ${decision.output.substring(0, 60)}...`);
    console.log(`  Confidence: ${(decision.confidence * 100).toFixed(1)}%\n`);
  }

  // Counterfactual analysis
  console.log('\n🔬 Counterfactual Analysis: "What if we didn\'t have X?"');
  console.log('═'.repeat(60) + '\n');

  // What if no biology agent?
  const cf1 = vDebugger.analyzeCounterfactual(snapshot, 'biology-agent', 'agent');
  console.log(`Removing: ${cf1.removed_type} "${cf1.removed_element}"`);
  console.log(`Impact Score: ${(cf1.impact_score * 100).toFixed(1)}%`);
  console.log(`Confidence Change: ${(cf1.confidence_change * 100).toFixed(1)}%`);
  console.log(`Explanation: ${cf1.explanation}\n`);

  // What if no homeostasis concept?
  const cf2 = vDebugger.analyzeCounterfactual(snapshot, 'homeostasis', 'concept');
  console.log(`Removing: ${cf2.removed_type} "${cf2.removed_element}"`);
  console.log(`Impact Score: ${(cf2.impact_score * 100).toFixed(1)}%`);
  console.log(`Confidence Change: ${(cf2.confidence_change * 100).toFixed(1)}%`);
  console.log(`Explanation: ${cf2.explanation}\n`);

  // What if no financial/savings slice?
  const cf3 = vDebugger.analyzeCounterfactual(snapshot, 'financial/savings', 'slice');
  console.log(`Removing: ${cf3.removed_type} "${cf3.removed_element}"`);
  console.log(`Impact Score: ${(cf3.impact_score * 100).toFixed(1)}%`);
  console.log(`Confidence Change: ${(cf3.confidence_change * 100).toFixed(1)}%`);
  console.log(`Explanation: ${cf3.explanation}\n`);

  // Export full report
  console.log('\n📄 Exporting Full Report...\n');
  const report = vDebugger.exportReport(snapshot);
  console.log(report);

  // Summary
  console.log('\n' + '='.repeat(80));
  console.log('📊 SUMMARY');
  console.log('='.repeat(80) + '\n');
  console.log(`Total Agents Involved: ${snapshot.agent_graph.nodes.size}`);
  console.log(`Total Decisions: ${snapshot.explanation.decision_path.length}`);
  console.log(`Total Concepts Activated: ${snapshot.explanation.concept_activation.length}`);
  console.log(`Alternative Paths Considered: ${snapshot.explanation.alternative_paths.length}`);
  console.log(`Confidence Flows Tracked: ${snapshot.explanation.confidence_flow.length}`);
  console.log(`Timeline Events: ${snapshot.timeline.length}`);
  console.log('\n💡 Key Insights:');
  console.log('  • Complete transparency: every decision is traceable');
  console.log('  • Concept attribution: know which knowledge influenced what');
  console.log('  • Counterfactual reasoning: understand causal relationships');
  console.log('  • No more black box: AGI reasoning is now fully explainable\n');

  console.log('✅ Visual Debugger Demo Complete!');
  console.log('='.repeat(80) + '\n');
}

// Run demo if executed directly
if (require.main === module) {
  runVisualDebuggerDemo();
}

export { runVisualDebuggerDemo };
