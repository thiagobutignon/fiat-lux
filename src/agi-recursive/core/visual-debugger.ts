/**
 * Visual Debugger - Real-time reasoning flow visualization
 *
 * Transforms AGI from "black box" to "glass box" through:
 * 1. Explanation Layer: decision paths, confidence flows, concept activation
 * 2. Concept Attribution: track which concepts contributed to each decision
 * 3. Counterfactual Reasoning: "what if we didn't have slice X?"
 * 4. Interactive Visualizations: agent graphs, timelines, concept highlighting
 *
 * This is innovation #24: "Deixar de Ser Caixa Preta" (Stop Being Black Box)
 */

export interface DecisionNode {
  id: string;
  agent: string;
  timestamp: Date;
  input: string;
  output: string;
  confidence: number;
  reasoning: string;
  concepts_used: string[];
  parent_id?: string;
}

export interface ConfidenceFlow {
  node_id: string;
  confidence_in: number;
  confidence_out: number;
  confidence_delta: number;
  factors: {
    concept: string;
    weight: number;
    contribution: number; // how much this concept changed confidence
  }[];
}

export interface ConceptActivation {
  concept: string;
  slice: string;
  activation_strength: number;
  timestamp: Date;
  context: string;
  contribution_to_output: number; // 0-1 scale
}

export interface AlternativePath {
  id: string;
  description: string;
  confidence: number;
  reason_not_taken: string;
  would_require: string[]; // slices/concepts needed
}

export interface ExplanationLayer {
  decision_path: DecisionNode[];
  confidence_flow: ConfidenceFlow[];
  concept_activation: ConceptActivation[];
  alternative_paths: AlternativePath[];
}

export interface CounterfactualAnalysis {
  removed_element: string;
  removed_type: 'slice' | 'concept' | 'agent';
  original_output: string;
  counterfactual_output: string;
  confidence_change: number;
  impact_score: number; // 0-1 scale
  explanation: string;
}

export interface AgentGraphNode {
  agent: string;
  invocations: number;
  avg_confidence: number;
  concepts_used: Set<string>;
}

export interface AgentGraphEdge {
  from: string;
  to: string;
  weight: number; // how many times this path was used
  avg_confidence: number;
}

export interface AgentGraph {
  nodes: Map<string, AgentGraphNode>;
  edges: AgentGraphEdge[];
}

export interface TimelineEvent {
  timestamp: Date;
  type: 'agent_invocation' | 'concept_activation' | 'decision' | 'confidence_change';
  agent?: string;
  concept?: string;
  confidence?: number;
  description: string;
}

export interface VisualDebuggerSnapshot {
  query: string;
  timestamp: Date;
  explanation: ExplanationLayer;
  agent_graph: AgentGraph;
  timeline: TimelineEvent[];
  counterfactuals?: CounterfactualAnalysis[];
}

/**
 * VisualDebugger - Main class for real-time reasoning flow visualization
 */
export class VisualDebugger {
  private snapshots: VisualDebuggerSnapshot[] = [];
  private current_explanation: ExplanationLayer;
  private current_agent_graph: AgentGraph;
  private current_timeline: TimelineEvent[];

  constructor() {
    this.current_explanation = {
      decision_path: [],
      confidence_flow: [],
      concept_activation: [],
      alternative_paths: []
    };
    this.current_agent_graph = {
      nodes: new Map(),
      edges: []
    };
    this.current_timeline = [];
  }

  /**
   * Start tracking a new query session
   */
  startSession(query: string): void {
    this.current_explanation = {
      decision_path: [],
      confidence_flow: [],
      concept_activation: [],
      alternative_paths: []
    };
    this.current_agent_graph = {
      nodes: new Map(),
      edges: []
    };
    this.current_timeline = [];
  }

  /**
   * Track a decision node (agent invocation)
   */
  trackDecision(decision: DecisionNode): void {
    this.current_explanation.decision_path.push(decision);

    // Update agent graph
    if (!this.current_agent_graph.nodes.has(decision.agent)) {
      this.current_agent_graph.nodes.set(decision.agent, {
        agent: decision.agent,
        invocations: 0,
        avg_confidence: 0,
        concepts_used: new Set()
      });
    }

    const node = this.current_agent_graph.nodes.get(decision.agent)!;
    node.invocations++;
    node.avg_confidence = (node.avg_confidence * (node.invocations - 1) + decision.confidence) / node.invocations;
    decision.concepts_used.forEach(c => node.concepts_used.add(c));

    // Add to timeline
    this.current_timeline.push({
      timestamp: decision.timestamp,
      type: 'agent_invocation',
      agent: decision.agent,
      description: `${decision.agent} processed: "${decision.input.substring(0, 50)}..."`
    });

    // Track edges if this decision has a parent
    if (decision.parent_id) {
      const parent = this.current_explanation.decision_path.find(d => d.id === decision.parent_id);
      if (parent) {
        this.addOrUpdateEdge(parent.agent, decision.agent, decision.confidence);
      }
    }
  }

  /**
   * Track confidence flow through the system
   */
  trackConfidenceFlow(flow: ConfidenceFlow): void {
    this.current_explanation.confidence_flow.push(flow);

    if (flow.confidence_delta !== 0) {
      this.current_timeline.push({
        timestamp: new Date(),
        type: 'confidence_change',
        confidence: flow.confidence_out,
        description: `Confidence ${flow.confidence_delta > 0 ? 'increased' : 'decreased'} to ${(flow.confidence_out * 100).toFixed(1)}%`
      });
    }
  }

  /**
   * Track concept activation
   */
  trackConceptActivation(activation: ConceptActivation): void {
    this.current_explanation.concept_activation.push(activation);

    this.current_timeline.push({
      timestamp: activation.timestamp,
      type: 'concept_activation',
      concept: activation.concept,
      description: `Concept "${activation.concept}" activated (strength: ${(activation.activation_strength * 100).toFixed(1)}%)`
    });
  }

  /**
   * Track an alternative path that was considered but not taken
   */
  trackAlternativePath(path: AlternativePath): void {
    this.current_explanation.alternative_paths.push(path);
  }

  /**
   * Capture a complete snapshot of the current session
   */
  captureSnapshot(query: string): VisualDebuggerSnapshot {
    const snapshot: VisualDebuggerSnapshot = {
      query,
      timestamp: new Date(),
      explanation: JSON.parse(JSON.stringify(this.current_explanation)),
      agent_graph: {
        nodes: new Map(this.current_agent_graph.nodes),
        edges: [...this.current_agent_graph.edges]
      },
      timeline: [...this.current_timeline]
    };

    this.snapshots.push(snapshot);
    return snapshot;
  }

  /**
   * Perform counterfactual analysis: "What if we didn't have X?"
   */
  analyzeCounterfactual(
    snapshot: VisualDebuggerSnapshot,
    removed_element: string,
    removed_type: 'slice' | 'concept' | 'agent'
  ): CounterfactualAnalysis {
    // Filter out the removed element from the snapshot
    let filtered_path = snapshot.explanation.decision_path;
    let filtered_activations = snapshot.explanation.concept_activation;

    if (removed_type === 'agent') {
      filtered_path = filtered_path.filter(d => d.agent !== removed_element);
    } else if (removed_type === 'concept') {
      filtered_activations = filtered_activations.filter(a => a.concept !== removed_element);
    } else if (removed_type === 'slice') {
      filtered_activations = filtered_activations.filter(a => a.slice !== removed_element);
    }

    // Calculate impact
    const original_confidence = this.calculateAverageConfidence(snapshot.explanation.decision_path);
    const counterfactual_confidence = this.calculateAverageConfidence(filtered_path);
    const confidence_change = counterfactual_confidence - original_confidence;

    // Calculate impact score (0-1)
    const removed_activations = snapshot.explanation.concept_activation.length - filtered_activations.length;
    const removed_decisions = snapshot.explanation.decision_path.length - filtered_path.length;
    const impact_score = Math.min(1, (removed_activations * 0.3 + removed_decisions * 0.7) / snapshot.explanation.decision_path.length);

    const analysis: CounterfactualAnalysis = {
      removed_element,
      removed_type,
      original_output: this.reconstructOutput(snapshot.explanation.decision_path),
      counterfactual_output: this.reconstructOutput(filtered_path),
      confidence_change,
      impact_score,
      explanation: this.generateCounterfactualExplanation(removed_element, removed_type, impact_score, confidence_change)
    };

    return analysis;
  }

  /**
   * Get top N most influential concepts
   */
  getTopInfluencers(snapshot: VisualDebuggerSnapshot, n: number = 5): ConceptActivation[] {
    return snapshot.explanation.concept_activation
      .sort((a, b) => b.contribution_to_output - a.contribution_to_output)
      .slice(0, n);
  }

  /**
   * Get the critical decision path (highest impact decisions)
   */
  getCriticalPath(snapshot: VisualDebuggerSnapshot): DecisionNode[] {
    return snapshot.explanation.decision_path
      .sort((a, b) => b.confidence - a.confidence)
      .filter(d => d.confidence >= 0.7);
  }

  /**
   * Visualize agent graph as ASCII art
   */
  visualizeAgentGraph(graph: AgentGraph): string {
    let output = '\n📊 Agent Collaboration Graph\n';
    output += '═'.repeat(60) + '\n\n';

    // Show nodes
    output += '🔵 Agents:\n';
    for (const [name, node] of graph.nodes) {
      const bar = '█'.repeat(Math.floor(node.avg_confidence * 20));
      output += `  ${name.padEnd(20)} ${bar} ${(node.avg_confidence * 100).toFixed(1)}%\n`;
      output += `    Invocations: ${node.invocations}, Concepts: ${node.concepts_used.size}\n`;
    }

    // Show edges
    output += '\n🔗 Interactions:\n';
    for (const edge of graph.edges) {
      const weight_bar = '─'.repeat(Math.min(edge.weight, 30));
      output += `  ${edge.from} ${weight_bar}> ${edge.to} (${edge.weight}x, ${(edge.avg_confidence * 100).toFixed(1)}%)\n`;
    }

    return output;
  }

  /**
   * Visualize timeline as ASCII art
   */
  visualizeTimeline(timeline: TimelineEvent[]): string {
    let output = '\n📅 Decision Timeline\n';
    output += '═'.repeat(60) + '\n\n';

    const sorted = [...timeline].sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());

    for (let i = 0; i < sorted.length; i++) {
      const event = sorted[i];
      const time = event.timestamp.toISOString().split('T')[1].split('.')[0];

      let icon = '•';
      if (event.type === 'agent_invocation') icon = '🤖';
      else if (event.type === 'concept_activation') icon = '💡';
      else if (event.type === 'decision') icon = '✓';
      else if (event.type === 'confidence_change') icon = '📊';

      output += `${time} ${icon} ${event.description}\n`;
    }

    return output;
  }

  /**
   * Visualize concept activation heatmap
   */
  visualizeConceptHeatmap(activations: ConceptActivation[]): string {
    let output = '\n🔥 Concept Activation Heatmap\n';
    output += '═'.repeat(60) + '\n\n';

    const sorted = [...activations].sort((a, b) => b.activation_strength - a.activation_strength);

    for (const activation of sorted.slice(0, 15)) {
      const bar = '█'.repeat(Math.floor(activation.activation_strength * 30));
      const contribution = (activation.contribution_to_output * 100).toFixed(1);
      output += `${activation.concept.padEnd(25)} ${bar}\n`;
      output += `  Strength: ${(activation.activation_strength * 100).toFixed(1)}%, Contribution: ${contribution}%\n`;
      output += `  From: ${activation.slice}\n\n`;
    }

    return output;
  }

  /**
   * Export full debugging report
   */
  exportReport(snapshot: VisualDebuggerSnapshot): string {
    let report = '\n';
    report += '═'.repeat(80) + '\n';
    report += '🔍 VISUAL DEBUGGER REPORT\n';
    report += '═'.repeat(80) + '\n\n';

    report += `Query: ${snapshot.query}\n`;
    report += `Timestamp: ${snapshot.timestamp.toISOString()}\n`;
    report += `Agents Involved: ${snapshot.agent_graph.nodes.size}\n`;
    report += `Decisions Made: ${snapshot.explanation.decision_path.length}\n`;
    report += `Concepts Activated: ${snapshot.explanation.concept_activation.length}\n`;
    report += `Alternative Paths: ${snapshot.explanation.alternative_paths.length}\n\n`;

    report += this.visualizeAgentGraph(snapshot.agent_graph);
    report += '\n' + this.visualizeTimeline(snapshot.timeline);
    report += '\n' + this.visualizeConceptHeatmap(snapshot.explanation.concept_activation);

    // Show top influencers
    report += '\n🌟 Top 5 Most Influential Concepts\n';
    report += '═'.repeat(60) + '\n';
    const top = this.getTopInfluencers(snapshot);
    for (let i = 0; i < top.length; i++) {
      const concept = top[i];
      report += `\n${i + 1}. ${concept.concept}\n`;
      report += `   Contribution: ${(concept.contribution_to_output * 100).toFixed(1)}%\n`;
      report += `   From: ${concept.slice}\n`;
      report += `   Context: ${concept.context}\n`;
    }

    // Show alternative paths
    if (snapshot.explanation.alternative_paths.length > 0) {
      report += '\n\n🔀 Alternative Paths Considered\n';
      report += '═'.repeat(60) + '\n';
      for (const path of snapshot.explanation.alternative_paths) {
        report += `\n• ${path.description}\n`;
        report += `  Confidence: ${(path.confidence * 100).toFixed(1)}%\n`;
        report += `  Not taken because: ${path.reason_not_taken}\n`;
      }
    }

    report += '\n' + '═'.repeat(80) + '\n';

    return report;
  }

  /**
   * Get all snapshots
   */
  getSnapshots(): VisualDebuggerSnapshot[] {
    return this.snapshots;
  }

  /**
   * Clear all snapshots
   */
  clearSnapshots(): void {
    this.snapshots = [];
  }

  // Private helper methods

  private addOrUpdateEdge(from: string, to: string, confidence: number): void {
    const existing = this.current_agent_graph.edges.find(e => e.from === from && e.to === to);

    if (existing) {
      existing.weight++;
      existing.avg_confidence = (existing.avg_confidence * (existing.weight - 1) + confidence) / existing.weight;
    } else {
      this.current_agent_graph.edges.push({
        from,
        to,
        weight: 1,
        avg_confidence: confidence
      });
    }
  }

  private calculateAverageConfidence(decisions: DecisionNode[]): number {
    if (decisions.length === 0) return 0;
    return decisions.reduce((sum, d) => sum + d.confidence, 0) / decisions.length;
  }

  private reconstructOutput(decisions: DecisionNode[]): string {
    if (decisions.length === 0) return '';
    return decisions[decisions.length - 1].output;
  }

  private generateCounterfactualExplanation(
    element: string,
    type: string,
    impact: number,
    confidence_change: number
  ): string {
    const impact_level = impact > 0.7 ? 'critical' : impact > 0.4 ? 'significant' : 'minor';
    const direction = confidence_change < 0 ? 'decreased' : 'increased';

    return `Removing ${type} "${element}" would have a ${impact_level} impact on the reasoning chain. ` +
           `Average confidence would have ${direction} by ${Math.abs(confidence_change * 100).toFixed(1)}%.`;
  }
}

/**
 * Factory function to create a VisualDebugger instance
 */
export function createVisualDebugger(): VisualDebugger {
  return new VisualDebugger();
}
