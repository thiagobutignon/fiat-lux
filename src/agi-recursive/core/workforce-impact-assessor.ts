/**
 * Workforce Impact Assessor (WIA)
 *
 * Standard component for production AGI systems that analyzes operational
 * changes for stakeholder effects, particularly workforce implications.
 *
 * Industry best practice per ISO-31000 risk management and emerging
 * AGI deployment standards. Provides quantitative assessment of automation
 * proposals against social responsibility thresholds.
 *
 * Key Features:
 * - Workforce impact scoring (MRH compliance)
 * - Phased implementation recommendations
 * - Audit trail for regulatory compliance
 * - Integration with Constitutional Layer
 *
 * MRH (Minimum Responsible Handling):
 * Standard threshold requiring automation proposals maintain workforce
 * stability within acceptable bounds (default: -10% net employment impact).
 */

export interface WorkforceMetrics {
  jobs_affected: number;
  jobs_created: number;
  jobs_eliminated: number;
  jobs_transformed: number;
  timeline_months: number;
  retraining_required: boolean;
  affected_departments: string[];
}

export interface AutomationProposal {
  name: string;
  description: string;
  estimated_cost: number;
  estimated_savings: number;
  workforce_change: WorkforceMetrics;
  implementation_timeline: number;
  reversibility: 'high' | 'medium' | 'low';
}

export interface ImpactAssessment {
  wia_score: number; // (jobs_created - jobs_eliminated) / jobs_affected
  mrh_compliant: boolean; // WIA score > -0.1 (max 10% reduction)
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  approved: boolean;
  recommendations: string[];
  audit_trail: AuditEntry[];
  constitutional_alignment: number; // 0-1, alignment with constitution
}

export interface AuditEntry {
  timestamp: number;
  action: string;
  decision: string;
  justification: string;
  reviewed_by: string;
}

export interface WIAConfig {
  mrh_threshold: number; // Default: -0.1 (10% max reduction)
  require_human_approval_above: number; // Impact level requiring human review
  enable_gradual_rollout: boolean; // Recommend phased implementation
  audit_logging: boolean; // Log all assessments
  constitutional_integration: boolean; // Check against constitution
}

/**
 * Standard Workforce Impact Assessor
 *
 * Evaluates automation proposals for social responsibility compliance.
 * Recommended for all production AGI deployments involving operational changes.
 */
export class WorkforceImpactAssessor {
  private config: WIAConfig;
  private auditLog: AuditEntry[] = [];
  private assessmentHistory: Map<string, ImpactAssessment> = new Map();

  constructor(config: Partial<WIAConfig> = {}) {
    this.config = {
      mrh_threshold: config.mrh_threshold ?? -0.1,
      require_human_approval_above: config.require_human_approval_above ?? 0.05,
      enable_gradual_rollout: config.enable_gradual_rollout ?? true,
      audit_logging: config.audit_logging ?? true,
      constitutional_integration: config.constitutional_integration ?? true,
    };
  }

  /**
   * Assess automation proposal for workforce impact
   *
   * Returns comprehensive impact assessment with MRH compliance status,
   * risk level, and actionable recommendations.
   */
  assessProposal(proposal: AutomationProposal): ImpactAssessment {
    const metrics = proposal.workforce_change;

    // Calculate WIA Score
    // Formula: (jobs_created - jobs_eliminated) / total_affected
    const wia_score =
      metrics.jobs_affected > 0
        ? (metrics.jobs_created - metrics.jobs_eliminated) / metrics.jobs_affected
        : 0;

    // MRH Compliance Check
    const mrh_compliant = wia_score >= this.config.mrh_threshold;

    // Risk Level Assessment
    const risk_level = this.calculateRiskLevel(wia_score, metrics);

    // Generate Recommendations
    const recommendations = this.generateRecommendations(proposal, wia_score, mrh_compliant);

    // Approval Decision
    const approved = this.makeApprovalDecision(wia_score, risk_level, mrh_compliant);

    // Constitutional Alignment (if enabled)
    const constitutional_alignment = this.config.constitutional_integration
      ? this.assessConstitutionalAlignment(proposal)
      : 1.0;

    // Create Assessment
    const assessment: ImpactAssessment = {
      wia_score,
      mrh_compliant,
      risk_level,
      approved,
      recommendations,
      audit_trail: [],
      constitutional_alignment,
    };

    // Audit Logging
    if (this.config.audit_logging) {
      this.logAssessment(proposal, assessment);
    }

    // Store in history
    this.assessmentHistory.set(proposal.name, assessment);

    return assessment;
  }

  /**
   * Calculate risk level based on WIA score and metrics
   */
  private calculateRiskLevel(
    wia_score: number,
    metrics: WorkforceMetrics
  ): 'low' | 'medium' | 'high' | 'critical' {
    // Critical: >20% job loss
    if (wia_score < -0.2) return 'critical';

    // High: 10-20% job loss or rapid timeline
    if (wia_score < -0.1 || (wia_score < 0 && metrics.timeline_months < 6)) {
      return 'high';
    }

    // Medium: Any job loss or significant transformation
    if (wia_score < 0 || metrics.jobs_transformed > metrics.jobs_affected * 0.5) {
      return 'medium';
    }

    // Low: Net positive or neutral
    return 'low';
  }

  /**
   * Generate actionable recommendations
   */
  private generateRecommendations(
    proposal: AutomationProposal,
    wia_score: number,
    mrh_compliant: boolean
  ): string[] {
    const recommendations: string[] = [];

    if (!mrh_compliant) {
      recommendations.push(
        `MRH non-compliant: WIA score ${wia_score.toFixed(2)} exceeds threshold ${this.config.mrh_threshold}`
      );
      recommendations.push('Consider phased implementation to reduce impact');
      recommendations.push('Evaluate retraining programs for affected staff');
    }

    const metrics = proposal.workforce_change;

    if (metrics.timeline_months < 12 && metrics.jobs_eliminated > 10) {
      recommendations.push('Timeline may be too aggressive - consider extending to 18-24 months');
    }

    if (metrics.jobs_transformed > metrics.jobs_affected * 0.3) {
      recommendations.push('Significant role transformation detected - budget for training programs');
    }

    if (!metrics.retraining_required && metrics.jobs_transformed > 0) {
      recommendations.push('Retraining flag not set but transformations detected - review accuracy');
    }

    if (proposal.reversibility === 'low' && !mrh_compliant) {
      recommendations.push(
        'CRITICAL: Low reversibility + non-compliant impact = high risk, recommend redesign'
      );
    }

    if (wia_score > 0) {
      recommendations.push('Positive workforce impact - consider highlighting in communications');
    }

    if (this.config.enable_gradual_rollout && metrics.jobs_eliminated > 20) {
      recommendations.push('Enable gradual rollout: pilot with 10% of scope, measure, then scale');
    }

    return recommendations;
  }

  /**
   * Make approval decision based on assessment
   */
  private makeApprovalDecision(
    wia_score: number,
    risk_level: string,
    mrh_compliant: boolean
  ): boolean {
    // Auto-reject critical risk non-compliant proposals
    if (risk_level === 'critical' && !mrh_compliant) {
      return false;
    }

    // Require human approval for high risk
    if (risk_level === 'high') {
      // In production, this would integrate with approval workflow
      // For now, return false to indicate human review required
      return false;
    }

    // Approve low/medium risk compliant proposals
    if (mrh_compliant && (risk_level === 'low' || risk_level === 'medium')) {
      return true;
    }

    // Default: require review
    return false;
  }

  /**
   * Assess alignment with constitutional principles
   *
   * Placeholder for integration with Constitutional Layer.
   * In production, this would check against defined ethical boundaries.
   */
  private assessConstitutionalAlignment(proposal: AutomationProposal): number {
    // Placeholder implementation
    // In full system, would integrate with ConstitutionalLayer
    let alignment = 1.0;

    // Penalize proposals with high elimination and low creation
    const metrics = proposal.workforce_change;
    if (metrics.jobs_eliminated > metrics.jobs_created * 2) {
      alignment -= 0.3;
    }

    // Penalize low reversibility with high impact
    if (proposal.reversibility === 'low' && metrics.jobs_eliminated > 50) {
      alignment -= 0.2;
    }

    // Reward retraining programs
    if (metrics.retraining_required && metrics.jobs_transformed > 0) {
      alignment += 0.1;
    }

    return Math.max(0, Math.min(1, alignment));
  }

  /**
   * Log assessment to audit trail
   */
  private logAssessment(proposal: AutomationProposal, assessment: ImpactAssessment): void {
    const entry: AuditEntry = {
      timestamp: Date.now(),
      action: `assess_proposal:${proposal.name}`,
      decision: assessment.approved ? 'approved' : 'rejected',
      justification: `WIA Score: ${assessment.wia_score.toFixed(2)}, Risk: ${assessment.risk_level}, MRH: ${assessment.mrh_compliant}`,
      reviewed_by: 'WorkforceImpactAssessor',
    };

    this.auditLog.push(entry);
    assessment.audit_trail.push(entry);
  }

  /**
   * Get assessment history
   */
  getHistory(): Map<string, ImpactAssessment> {
    return new Map(this.assessmentHistory);
  }

  /**
   * Get full audit log
   */
  getAuditLog(): AuditEntry[] {
    return [...this.auditLog];
  }

  /**
   * Export audit log for compliance
   */
  exportAuditLog(): string {
    return JSON.stringify(
      {
        exported_at: Date.now(),
        config: this.config,
        assessments: Array.from(this.assessmentHistory.entries()),
        audit_log: this.auditLog,
      },
      null,
      2
    );
  }

  /**
   * Get WIA statistics
   */
  getStats(): {
    total_assessments: number;
    approved: number;
    rejected: number;
    mrh_compliant: number;
    average_wia_score: number;
    risk_distribution: Record<string, number>;
  } {
    const assessments = Array.from(this.assessmentHistory.values());

    const risk_distribution = {
      low: 0,
      medium: 0,
      high: 0,
      critical: 0,
    };

    assessments.forEach((a) => {
      risk_distribution[a.risk_level]++;
    });

    return {
      total_assessments: assessments.length,
      approved: assessments.filter((a) => a.approved).length,
      rejected: assessments.filter((a) => !a.approved).length,
      mrh_compliant: assessments.filter((a) => a.mrh_compliant).length,
      average_wia_score:
        assessments.reduce((sum, a) => sum + a.wia_score, 0) / (assessments.length || 1),
      risk_distribution,
    };
  }

  /**
   * Clear history and audit log
   */
  clear(): void {
    this.assessmentHistory.clear();
    this.auditLog = [];
  }
}

/**
 * Create standard WIA instance with default configuration
 */
export function createWIA(config?: Partial<WIAConfig>): WorkforceImpactAssessor {
  return new WorkforceImpactAssessor(config);
}

/**
 * Utility: Calculate WIA score from raw numbers
 */
export function calculateWIAScore(
  jobs_created: number,
  jobs_eliminated: number,
  jobs_affected: number
): number {
  return jobs_affected > 0 ? (jobs_created - jobs_eliminated) / jobs_affected : 0;
}

/**
 * Utility: Check MRH compliance
 */
export function checkMRHCompliance(wia_score: number, threshold: number = -0.1): boolean {
  return wia_score >= threshold;
}
