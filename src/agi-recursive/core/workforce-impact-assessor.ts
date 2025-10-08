/**
 * Workforce Impact Assessor (WIA)
 *
 * Evaluates automation proposals for their impact on workforce,
 * enforcing Minimum Responsible Handling (MRH) standards.
 *
 * MRH Standard: Max 10% net workforce displacement per year (-0.1 threshold)
 */

export interface WorkforceChange {
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
  workforce_change: WorkforceChange;
  implementation_timeline: number;
  reversibility: 'low' | 'medium' | 'high';
}

export interface WorkforceMetrics {
  total_jobs_affected: number;
  displacement_rate: number;
  retraining_required: boolean;
}

export interface ImpactAssessment {
  wia_score: number;
  mrh_compliant: boolean;
  approved: boolean;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  recommendations: string[];
  constitutional_alignment?: number;
  metrics?: WorkforceMetrics;
}

export interface WorkforceImpactConfig {
  mrh_threshold?: number; // Default: -0.1 (max 10% net loss)
  require_human_approval_above?: number; // Default: 0.05 (5% impact)
  enable_gradual_rollout?: boolean;
  audit_logging?: boolean;
  constitutional_integration?: boolean;
  max_displacement_rate?: number; // Legacy compatibility
  require_retraining_plan?: boolean; // Legacy compatibility
}

export class WorkforceImpactAssessor {
  private config: Required<WorkforceImpactConfig>;

  constructor(config: WorkforceImpactConfig = {}) {
    this.config = {
      mrh_threshold: config.mrh_threshold ?? -0.1,
      require_human_approval_above: config.require_human_approval_above ?? 0.05,
      enable_gradual_rollout: config.enable_gradual_rollout ?? true,
      audit_logging: config.audit_logging ?? false,
      constitutional_integration: config.constitutional_integration ?? false,
      max_displacement_rate: config.max_displacement_rate ?? 0.1,
      require_retraining_plan: config.require_retraining_plan ?? true,
    };
  }

  assessProposal(proposal: AutomationProposal): ImpactAssessment {
    const wfc = proposal.workforce_change;

    // Calculate WIA Score: (jobs_created - jobs_eliminated) / jobs_affected
    // Range: -1.0 (all jobs eliminated) to +∞ (pure job creation)
    // MRH standard: >= -0.1 (max 10% net loss)
    let wia_score: number;

    if (wfc.jobs_affected === 0) {
      // Edge case: No jobs affected = neutral score
      wia_score = 0;
    } else {
      wia_score = (wfc.jobs_created - wfc.jobs_eliminated) / wfc.jobs_affected;
    }

    // MRH Compliance Check
    const mrh_compliant = wia_score >= this.config.mrh_threshold;

    // Risk Level Assessment
    let risk_level: ImpactAssessment['risk_level'];

    if (wia_score <= -0.2) {
      risk_level = 'critical'; // >=20% job loss
    } else if (wia_score <= -0.1 || wfc.timeline_months < 6 ||
               (proposal.reversibility === 'low' && wia_score < 0)) {
      risk_level = 'high'; // MRH threshold/violation, rapid timeline, or low reversibility with job loss
    } else if (wia_score < 0 || wfc.timeline_months < 12) {
      risk_level = 'medium'; // Some job loss or aggressive timeline
    } else {
      risk_level = 'low'; // Workforce neutral or positive
    }

    // Approval Decision
    const approved = mrh_compliant;

    // Generate Recommendations
    const recommendations: string[] = [];

    if (!mrh_compliant) {
      recommendations.push(
        `MRH non-compliant: WIA score ${wia_score.toFixed(2)} is below threshold ${this.config.mrh_threshold}. ` +
        `Consider reducing job elimination or creating new roles.`
      );
    }

    if (wia_score < -0.05 && wia_score >= this.config.mrh_threshold) {
      recommendations.push(
        `Close to MRH threshold. Monitor implementation carefully.`
      );
    }

    if (wfc.timeline_months < 6) {
      recommendations.push(
        `Timeline may be too aggressive (${wfc.timeline_months} months). ` +
        `Consider extending to allow for proper retraining and transition.`
      );
    }

    if (wfc.jobs_transformed > wfc.jobs_affected * 0.5 && !wfc.retraining_required) {
      recommendations.push(
        `Retraining flag not set despite ${wfc.jobs_transformed} jobs being transformed. ` +
        `Ensure comprehensive retraining programs are in place.`
      );
    }

    if (proposal.reversibility === 'low' && wia_score < 0) {
      recommendations.push(
        `Low reversibility combined with job loss creates significant risk. ` +
        `Consider pilot program before full rollout.`
      );
    }

    if (this.config.enable_gradual_rollout && wfc.jobs_affected > 100) {
      recommendations.push(
        `Large-scale impact detected (${wfc.jobs_affected} jobs). ` +
        `Implement gradual rollout with pilot departments first.`
      );
    }

    // Constitutional Alignment (if enabled)
    let constitutional_alignment: number | undefined;

    if (this.config.constitutional_integration) {
      // Calculate alignment with constitutional principles
      // Higher score = better alignment with human dignity, workforce protection
      constitutional_alignment = this.calculateConstitutionalAlignment(proposal);
    } else {
      constitutional_alignment = 1.0; // Default: full alignment if not checking
    }

    // Audit Logging (if enabled)
    if (this.config.audit_logging) {
      this.logAudit(proposal, {
        wia_score,
        mrh_compliant,
        approved,
        risk_level,
        recommendations,
        constitutional_alignment,
      });
    }

    return {
      wia_score,
      mrh_compliant,
      approved,
      risk_level,
      recommendations,
      constitutional_alignment,
      metrics: {
        total_jobs_affected: wfc.jobs_affected,
        displacement_rate: wfc.jobs_affected > 0 ?
          wfc.jobs_eliminated / wfc.jobs_affected : 0,
        retraining_required: wfc.retraining_required,
      },
    };
  }

  async assessQuery(query: string): Promise<ImpactAssessment> {
    // Simplified query assessment - defaults to low risk
    // In production, this would use LLM to analyze the query
    return {
      wia_score: 0,
      mrh_compliant: true,
      approved: true,
      risk_level: 'low',
      recommendations: [],
      constitutional_alignment: 1.0,
      metrics: {
        total_jobs_affected: 0,
        displacement_rate: 0,
        retraining_required: false,
      },
    };
  }

  private calculateConstitutionalAlignment(proposal: AutomationProposal): number {
    // Calculate alignment score based on multiple factors
    // Range: 0.0 (poor alignment) to 1.0 (perfect alignment)

    let score = 0.5; // Base score

    const wfc = proposal.workforce_change;

    // Factor 1: Job creation vs elimination
    if (wfc.jobs_created >= wfc.jobs_eliminated) {
      score += 0.2; // Workforce positive
    } else {
      const loss_ratio = wfc.jobs_eliminated / wfc.jobs_affected;
      score -= loss_ratio * 0.2; // Penalty for job loss
    }

    // Factor 2: Retraining commitment
    if (wfc.retraining_required && wfc.jobs_transformed > 0) {
      score += 0.15; // Good: investing in people
    }

    // Factor 3: Timeline reasonableness
    if (wfc.timeline_months >= 18) {
      score += 0.1; // Adequate time for transition
    } else if (wfc.timeline_months < 6) {
      score -= 0.1; // Too rushed
    }

    // Factor 4: Reversibility
    if (proposal.reversibility === 'high') {
      score += 0.1;
    } else if (proposal.reversibility === 'low') {
      score -= 0.05;
    }

    // Factor 5: Economic balance
    const roi = proposal.estimated_savings / proposal.estimated_cost;
    if (roi > 5 && wfc.jobs_eliminated > wfc.jobs_created) {
      // High profit with job loss = concerning
      score -= 0.1;
    }

    // Clamp to [0, 1]
    return Math.max(0, Math.min(1, score));
  }

  private logAudit(
    proposal: AutomationProposal,
    assessment: Partial<ImpactAssessment>
  ): void {
    // In production, this would write to a persistent audit log
    const auditEntry = {
      timestamp: new Date().toISOString(),
      proposal_name: proposal.name,
      assessment,
    };

    // For now, just console log (could be extended to file/database)
    if (process.env.WIA_DEBUG === 'true') {
      console.log('[WIA AUDIT]', JSON.stringify(auditEntry, null, 2));
    }
  }
}
