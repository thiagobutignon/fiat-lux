/**
 * Visual Debugger Tests
 *
 * Test suite for the Visual Debugger system that transforms AGI from black box to glass box.
 * Tests cover:
 * 1. Explanation Layer (decision paths, confidence flows, concept activation)
 * 2. Concept Attribution tracking
 * 3. Counterfactual Reasoning
 * 4. Interactive Visualizations
 */

import { describe, it, expect } from 'vitest';
import {
  VisualDebugger,
  createVisualDebugger,
  DecisionNode,
  ConfidenceFlow,
  ConceptActivation,
  AlternativePath,
  type VisualDebuggerSnapshot
} from '../core/visual-debugger';

describe('VisualDebugger', () => {
  describe('Session Management', () => {
    it('should start a new session with empty state', () => {
      const vDebugger = createVisualDebugger();
      vDebugger.startSession('test query');

      const snapshot = vDebugger.captureSnapshot('test query');

      expect(snapshot.query).toBe('test query');
      expect(snapshot.explanation.decision_path).toHaveLength(0);
      expect(snapshot.explanation.confidence_flow).toHaveLength(0);
      expect(snapshot.explanation.concept_activation).toHaveLength(0);
      expect(snapshot.agent_graph.nodes.size).toBe(0);
      expect(snapshot.timeline).toHaveLength(0);
    });

    it('should capture and store snapshots', () => {
      const vDebugger = createVisualDebugger();
      vDebugger.startSession('query 1');
      vDebugger.captureSnapshot('query 1');

      vDebugger.startSession('query 2');
      vDebugger.captureSnapshot('query 2');

      const snapshots = vDebugger.getSnapshots();
      expect(snapshots).toHaveLength(2);
      expect(snapshots[0].query).toBe('query 1');
      expect(snapshots[1].query).toBe('query 2');
    });

    it('should clear all snapshots', () => {
      const vDebugger = createVisualDebugger();
      vDebugger.startSession('query');
      vDebugger.captureSnapshot('query');

      vDebugger.clearSnapshots();

      expect(vDebugger.getSnapshots()).toHaveLength(0);
    });
  });

  describe('Decision Tracking', () => {
    it('should track decision nodes', () => {
      const vDebugger = createVisualDebugger();
      vDebugger.startSession('test query');

      const decision: DecisionNode = {
        id: 'decision-1',
        agent: 'financial-agent',
        timestamp: new Date(),
        input: 'How to save money?',
        output: 'Create a budget',
        confidence: 0.85,
        reasoning: 'Based on financial principles',
        concepts_used: ['budgeting', 'savings']
      };

      vDebugger.trackDecision(decision);

      const snapshot = vDebugger.captureSnapshot('test query');
      expect(snapshot.explanation.decision_path).toHaveLength(1);
      expect(snapshot.explanation.decision_path[0].id).toBe(decision.id);
      expect(snapshot.explanation.decision_path[0].agent).toBe(decision.agent);
      expect(snapshot.explanation.decision_path[0].input).toBe(decision.input);
      expect(snapshot.explanation.decision_path[0].output).toBe(decision.output);
      expect(snapshot.explanation.decision_path[0].confidence).toBe(decision.confidence);
    });

    it('should update agent graph when tracking decisions', () => {
      const vDebugger = createVisualDebugger();
      vDebugger.startSession('test query');

      vDebugger.trackDecision({
        id: 'decision-1',
        agent: 'financial-agent',
        timestamp: new Date(),
        input: 'test',
        output: 'test',
        confidence: 0.8,
        reasoning: 'test',
        concepts_used: ['concept-1', 'concept-2']
      });

      const snapshot = vDebugger.captureSnapshot('test query');
      const node = snapshot.agent_graph.nodes.get('financial-agent');

      expect(node).toBeDefined();
      expect(node!.invocations).toBe(1);
      expect(node!.avg_confidence).toBe(0.8);
      expect(node!.concepts_used.size).toBe(2);
    });

    it('should track edges between agents', () => {
      const vDebugger = createVisualDebugger();
      vDebugger.startSession('test query');

      vDebugger.trackDecision({
        id: 'decision-1',
        agent: 'meta-agent',
        timestamp: new Date(),
        input: 'test',
        output: 'test',
        confidence: 0.8,
        reasoning: 'test',
        concepts_used: []
      });

      vDebugger.trackDecision({
        id: 'decision-2',
        agent: 'financial-agent',
        timestamp: new Date(),
        input: 'test',
        output: 'test',
        confidence: 0.75,
        reasoning: 'test',
        concepts_used: [],
        parent_id: 'decision-1'
      });

      const snapshot = vDebugger.captureSnapshot('test query');
      expect(snapshot.agent_graph.edges).toHaveLength(1);
      expect(snapshot.agent_graph.edges[0].from).toBe('meta-agent');
      expect(snapshot.agent_graph.edges[0].to).toBe('financial-agent');
      expect(snapshot.agent_graph.edges[0].weight).toBe(1);
    });

    it('should aggregate multiple invocations of the same agent', () => {
      const vDebugger = createVisualDebugger();
      vDebugger.startSession('test query');

      vDebugger.trackDecision({
        id: 'decision-1',
        agent: 'financial-agent',
        timestamp: new Date(),
        input: 'test',
        output: 'test',
        confidence: 0.8,
        reasoning: 'test',
        concepts_used: ['concept-1']
      });

      vDebugger.trackDecision({
        id: 'decision-2',
        agent: 'financial-agent',
        timestamp: new Date(),
        input: 'test',
        output: 'test',
        confidence: 0.6,
        reasoning: 'test',
        concepts_used: ['concept-2']
      });

      const snapshot = vDebugger.captureSnapshot('test query');
      const node = snapshot.agent_graph.nodes.get('financial-agent');

      expect(node!.invocations).toBe(2);
      expect(node!.avg_confidence).toBe(0.7); // (0.8 + 0.6) / 2
      expect(node!.concepts_used.size).toBe(2);
    });
  });

  describe('Confidence Flow Tracking', () => {
    it('should track confidence flow', () => {
      const vDebugger = createVisualDebugger();
      vDebugger.startSession('test query');

      const flow: ConfidenceFlow = {
        node_id: 'decision-1',
        confidence_in: 0.5,
        confidence_out: 0.8,
        confidence_delta: 0.3,
        factors: [
          { concept: 'budgeting', weight: 0.7, contribution: 0.2 },
          { concept: 'savings', weight: 0.3, contribution: 0.1 }
        ]
      };

      vDebugger.trackConfidenceFlow(flow);

      const snapshot = vDebugger.captureSnapshot('test query');
      expect(snapshot.explanation.confidence_flow).toHaveLength(1);
      expect(snapshot.explanation.confidence_flow[0]).toEqual(flow);
    });

    it('should add timeline events for confidence changes', () => {
      const vDebugger = createVisualDebugger();
      vDebugger.startSession('test query');

      vDebugger.trackConfidenceFlow({
        node_id: 'decision-1',
        confidence_in: 0.5,
        confidence_out: 0.8,
        confidence_delta: 0.3,
        factors: []
      });

      const snapshot = vDebugger.captureSnapshot('test query');
      const confidenceEvents = snapshot.timeline.filter(e => e.type === 'confidence_change');

      expect(confidenceEvents).toHaveLength(1);
      expect(confidenceEvents[0].confidence).toBe(0.8);
    });
  });

  describe('Concept Activation Tracking', () => {
    it('should track concept activation', () => {
      const vDebugger = createVisualDebugger();
      vDebugger.startSession('test query');

      const activation: ConceptActivation = {
        concept: 'compound-interest',
        slice: 'financial/savings',
        activation_strength: 0.9,
        timestamp: new Date(),
        context: 'Calculating savings growth',
        contribution_to_output: 0.7
      };

      vDebugger.trackConceptActivation(activation);

      const snapshot = vDebugger.captureSnapshot('test query');
      expect(snapshot.explanation.concept_activation).toHaveLength(1);
      expect(snapshot.explanation.concept_activation[0].concept).toBe(activation.concept);
      expect(snapshot.explanation.concept_activation[0].slice).toBe(activation.slice);
      expect(snapshot.explanation.concept_activation[0].activation_strength).toBe(activation.activation_strength);
      expect(snapshot.explanation.concept_activation[0].contribution_to_output).toBe(activation.contribution_to_output);
    });

    it('should add timeline events for concept activation', () => {
      const vDebugger = createVisualDebugger();
      vDebugger.startSession('test query');

      vDebugger.trackConceptActivation({
        concept: 'compound-interest',
        slice: 'financial/savings',
        activation_strength: 0.9,
        timestamp: new Date(),
        context: 'test',
        contribution_to_output: 0.7
      });

      const snapshot = vDebugger.captureSnapshot('test query');
      const activationEvents = snapshot.timeline.filter(e => e.type === 'concept_activation');

      expect(activationEvents).toHaveLength(1);
      expect(activationEvents[0].concept).toBe('compound-interest');
    });
  });

  describe('Alternative Paths Tracking', () => {
    it('should track alternative paths', () => {
      const vDebugger = createVisualDebugger();
      vDebugger.startSession('test query');

      const alternative: AlternativePath = {
        id: 'alt-1',
        description: 'Use biology agent for homeostasis analogy',
        confidence: 0.65,
        reason_not_taken: 'Financial agent had higher confidence',
        would_require: ['biology/homeostasis']
      };

      vDebugger.trackAlternativePath(alternative);

      const snapshot = vDebugger.captureSnapshot('test query');
      expect(snapshot.explanation.alternative_paths).toHaveLength(1);
      expect(snapshot.explanation.alternative_paths[0]).toEqual(alternative);
    });
  });

  describe('Top Influencers', () => {
    it('should return top N most influential concepts', () => {
      const vDebugger = createVisualDebugger();
      vDebugger.startSession('test query');

      vDebugger.trackConceptActivation({
        concept: 'concept-1',
        slice: 'slice-1',
        activation_strength: 0.9,
        timestamp: new Date(),
        context: 'test',
        contribution_to_output: 0.8
      });

      vDebugger.trackConceptActivation({
        concept: 'concept-2',
        slice: 'slice-2',
        activation_strength: 0.7,
        timestamp: new Date(),
        context: 'test',
        contribution_to_output: 0.5
      });

      vDebugger.trackConceptActivation({
        concept: 'concept-3',
        slice: 'slice-3',
        activation_strength: 0.6,
        timestamp: new Date(),
        context: 'test',
        contribution_to_output: 0.3
      });

      const snapshot = vDebugger.captureSnapshot('test query');
      const top = vDebugger.getTopInfluencers(snapshot, 2);

      expect(top).toHaveLength(2);
      expect(top[0].concept).toBe('concept-1');
      expect(top[1].concept).toBe('concept-2');
    });
  });

  describe('Critical Path', () => {
    it('should return only high-confidence decisions', () => {
      const vDebugger = createVisualDebugger();
      vDebugger.startSession('test query');

      vDebugger.trackDecision({
        id: 'decision-1',
        agent: 'agent-1',
        timestamp: new Date(),
        input: 'test',
        output: 'test',
        confidence: 0.9,
        reasoning: 'high confidence',
        concepts_used: []
      });

      vDebugger.trackDecision({
        id: 'decision-2',
        agent: 'agent-2',
        timestamp: new Date(),
        input: 'test',
        output: 'test',
        confidence: 0.5,
        reasoning: 'low confidence',
        concepts_used: []
      });

      vDebugger.trackDecision({
        id: 'decision-3',
        agent: 'agent-3',
        timestamp: new Date(),
        input: 'test',
        output: 'test',
        confidence: 0.75,
        reasoning: 'medium-high confidence',
        concepts_used: []
      });

      const snapshot = vDebugger.captureSnapshot('test query');
      const critical = vDebugger.getCriticalPath(snapshot);

      expect(critical).toHaveLength(2);
      expect(critical.every(d => d.confidence >= 0.7)).toBe(true);
    });
  });

  describe('Counterfactual Analysis', () => {
    it('should analyze impact of removing an agent', () => {
      const vDebugger = createVisualDebugger();
      vDebugger.startSession('test query');

      vDebugger.trackDecision({
        id: 'decision-1',
        agent: 'financial-agent',
        timestamp: new Date(),
        input: 'test',
        output: 'save money',
        confidence: 0.8,
        reasoning: 'test',
        concepts_used: []
      });

      vDebugger.trackDecision({
        id: 'decision-2',
        agent: 'biology-agent',
        timestamp: new Date(),
        input: 'test',
        output: 'homeostasis',
        confidence: 0.7,
        reasoning: 'test',
        concepts_used: []
      });

      const snapshot = vDebugger.captureSnapshot('test query');
      const counterfactual = vDebugger.analyzeCounterfactual(snapshot, 'biology-agent', 'agent');

      expect(counterfactual.removed_element).toBe('biology-agent');
      expect(counterfactual.removed_type).toBe('agent');
      expect(counterfactual.impact_score).toBeGreaterThan(0);
      expect(counterfactual.confidence_change).not.toBe(0); // Can be positive or negative depending on removed agent
    });

    it('should analyze impact of removing a concept', () => {
      const vDebugger = createVisualDebugger();
      vDebugger.startSession('test query');

      vDebugger.trackConceptActivation({
        concept: 'compound-interest',
        slice: 'financial/savings',
        activation_strength: 0.9,
        timestamp: new Date(),
        context: 'test',
        contribution_to_output: 0.8
      });

      vDebugger.trackConceptActivation({
        concept: 'budgeting',
        slice: 'financial/planning',
        activation_strength: 0.7,
        timestamp: new Date(),
        context: 'test',
        contribution_to_output: 0.5
      });

      const snapshot = vDebugger.captureSnapshot('test query');
      const counterfactual = vDebugger.analyzeCounterfactual(snapshot, 'compound-interest', 'concept');

      expect(counterfactual.removed_element).toBe('compound-interest');
      expect(counterfactual.removed_type).toBe('concept');
      expect(counterfactual.impact_score).toBeGreaterThan(0);
    });

    it('should analyze impact of removing a slice', () => {
      const vDebugger = createVisualDebugger();
      vDebugger.startSession('test query');

      vDebugger.trackConceptActivation({
        concept: 'concept-1',
        slice: 'financial/savings',
        activation_strength: 0.9,
        timestamp: new Date(),
        context: 'test',
        contribution_to_output: 0.8
      });

      vDebugger.trackConceptActivation({
        concept: 'concept-2',
        slice: 'financial/savings',
        activation_strength: 0.8,
        timestamp: new Date(),
        context: 'test',
        contribution_to_output: 0.7
      });

      vDebugger.trackConceptActivation({
        concept: 'concept-3',
        slice: 'biology/homeostasis',
        activation_strength: 0.6,
        timestamp: new Date(),
        context: 'test',
        contribution_to_output: 0.4
      });

      const snapshot = vDebugger.captureSnapshot('test query');
      const counterfactual = vDebugger.analyzeCounterfactual(snapshot, 'financial/savings', 'slice');

      expect(counterfactual.removed_element).toBe('financial/savings');
      expect(counterfactual.removed_type).toBe('slice');
      expect(counterfactual.impact_score).toBeGreaterThan(0.5); // Should be significant
    });

    it('should provide explanation for counterfactual', () => {
      const vDebugger = createVisualDebugger();
      vDebugger.startSession('test query');

      vDebugger.trackDecision({
        id: 'decision-1',
        agent: 'financial-agent',
        timestamp: new Date(),
        input: 'test',
        output: 'test',
        confidence: 0.8,
        reasoning: 'test',
        concepts_used: []
      });

      const snapshot = vDebugger.captureSnapshot('test query');
      const counterfactual = vDebugger.analyzeCounterfactual(snapshot, 'financial-agent', 'agent');

      expect(counterfactual.explanation).toBeTruthy();
      expect(counterfactual.explanation).toContain('financial-agent');
      expect(counterfactual.explanation.toLowerCase()).toContain('impact');
    });
  });

  describe('Visualization', () => {
    it('should visualize agent graph as ASCII', () => {
      const vDebugger = createVisualDebugger();
      vDebugger.startSession('test query');

      vDebugger.trackDecision({
        id: 'decision-1',
        agent: 'meta-agent',
        timestamp: new Date(),
        input: 'test',
        output: 'test',
        confidence: 0.8,
        reasoning: 'test',
        concepts_used: ['concept-1']
      });

      vDebugger.trackDecision({
        id: 'decision-2',
        agent: 'financial-agent',
        timestamp: new Date(),
        input: 'test',
        output: 'test',
        confidence: 0.75,
        reasoning: 'test',
        concepts_used: ['concept-2'],
        parent_id: 'decision-1'
      });

      const snapshot = vDebugger.captureSnapshot('test query');
      const visualization = vDebugger.visualizeAgentGraph(snapshot.agent_graph);

      expect(visualization).toContain('Agent Collaboration Graph');
      expect(visualization).toContain('meta-agent');
      expect(visualization).toContain('financial-agent');
      expect(visualization).toContain('█'); // confidence bar
      expect(visualization).toContain('─'); // edge
    });

    it('should visualize timeline as ASCII', () => {
      const vDebugger = createVisualDebugger();
      vDebugger.startSession('test query');

      vDebugger.trackDecision({
        id: 'decision-1',
        agent: 'financial-agent',
        timestamp: new Date(),
        input: 'How to save?',
        output: 'Create budget',
        confidence: 0.8,
        reasoning: 'test',
        concepts_used: []
      });

      const snapshot = vDebugger.captureSnapshot('test query');
      const visualization = vDebugger.visualizeTimeline(snapshot.timeline);

      expect(visualization).toContain('Decision Timeline');
      expect(visualization).toContain('financial-agent');
      expect(visualization).toContain('🤖'); // agent icon
    });

    it('should visualize concept heatmap as ASCII', () => {
      const vDebugger = createVisualDebugger();
      vDebugger.startSession('test query');

      vDebugger.trackConceptActivation({
        concept: 'compound-interest',
        slice: 'financial/savings',
        activation_strength: 0.9,
        timestamp: new Date(),
        context: 'test',
        contribution_to_output: 0.8
      });

      const snapshot = vDebugger.captureSnapshot('test query');
      const visualization = vDebugger.visualizeConceptHeatmap(snapshot.explanation.concept_activation);

      expect(visualization).toContain('Concept Activation Heatmap');
      expect(visualization).toContain('compound-interest');
      expect(visualization).toContain('█'); // heat bar
      expect(visualization).toContain('Strength:');
      expect(visualization).toContain('Contribution:');
    });
  });

  describe('Report Export', () => {
    it('should export a complete debugging report', () => {
      const vDebugger = createVisualDebugger();
      vDebugger.startSession('How to save money?');

      vDebugger.trackDecision({
        id: 'decision-1',
        agent: 'financial-agent',
        timestamp: new Date(),
        input: 'How to save money?',
        output: 'Create a budget',
        confidence: 0.85,
        reasoning: 'test',
        concepts_used: ['budgeting']
      });

      vDebugger.trackConceptActivation({
        concept: 'budgeting',
        slice: 'financial/planning',
        activation_strength: 0.9,
        timestamp: new Date(),
        context: 'test',
        contribution_to_output: 0.8
      });

      const snapshot = vDebugger.captureSnapshot('How to save money?');
      const report = vDebugger.exportReport(snapshot);

      expect(report).toContain('VISUAL DEBUGGER REPORT');
      expect(report).toContain('How to save money?');
      expect(report).toContain('Agent Collaboration Graph');
      expect(report).toContain('Decision Timeline');
      expect(report).toContain('Concept Activation Heatmap');
      expect(report).toContain('Top 5 Most Influential Concepts');
    });

    it('should include alternative paths in report if available', () => {
      const vDebugger = createVisualDebugger();
      vDebugger.startSession('test query');

      vDebugger.trackAlternativePath({
        id: 'alt-1',
        description: 'Use biology agent',
        confidence: 0.6,
        reason_not_taken: 'Lower confidence',
        would_require: ['biology/homeostasis']
      });

      const snapshot = vDebugger.captureSnapshot('test query');
      const report = vDebugger.exportReport(snapshot);

      expect(report).toContain('Alternative Paths Considered');
      expect(report).toContain('Use biology agent');
      expect(report).toContain('Lower confidence');
    });
  });
});
