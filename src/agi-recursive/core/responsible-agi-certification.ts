/**
 * Responsible AGI Certification System
 *
 * Validates commercial AGI deployments against social responsibility standards.
 * Provides certification tiers (Bronze/Silver/Gold) and generates public badges.
 */

import { WorkforceImpactAssessor, ImpactAssessment } from './workforce-impact-assessor';

// ============================================================================
// Types
// ============================================================================

export type CertificationTier = 'none' | 'bronze' | 'silver' | 'gold';

export interface CompanyProfile {
  name: string;
  size_employees: number;
  ceo_to_worker_ratio: number;
  employee_ownership_percent: number;
  layoffs_last_24_months: number;
  layoffs_last_24_months_percent: number;
  has_labor_violations: boolean;
  has_democratic_governance: boolean;
  has_retraining_programs: boolean;
  quarterly_reports_submitted: number; // out of last 4 quarters
}

export interface DeploymentMetrics {
  total_queries_processed: number;
  queries_with_automation_intent: number;
  jobs_affected: number;
  jobs_created: number;
  jobs_eliminated: number;
  net_workforce_change: number;
  wia_average_score: number;
  mrh_violations: number;
  constitutional_violations: number;
  retraining_budget_usd: number;
}

export interface CertificationResult {
  tier: CertificationTier;
  approved: boolean;
  badge_url: string;
  score: number; // 0-100
  expiration_date: Date;
  violations: string[];
  recommendations: string[];
  license_fee_annual: number;
  license_discount_percent: number;
  audit_required: boolean;
}

// ============================================================================
// Certification System
// ============================================================================

export class ResponsibleAGICertification {
  private wia: WorkforceImpactAssessor;

  constructor() {
    this.wia = new WorkforceImpactAssessor();
  }

  /**
   * Evaluate company for Responsible AGI certification
   *
   * Returns certification tier and badge based on:
   * 1. MRH compliance (minimum requirement)
   * 2. Company structure (employee ownership, governance)
   * 3. Track record (layoffs, labor violations)
   * 4. Deployment impact (net job creation/elimination)
   * 5. Reporting compliance
   */
  evaluate(company: CompanyProfile, metrics: DeploymentMetrics): CertificationResult {
    const violations: string[] = [];
    const recommendations: string[] = [];
    let score = 100;

    // ========================================================================
    // Automatic Disqualifications
    // ========================================================================

    // Check: Blocked use violations
    if (company.layoffs_last_24_months_percent > 0.05) {
      violations.push(
        `Company laid off ${company.layoffs_last_24_months_percent.toFixed(1)}% of workforce in last 24 months (threshold: 5%)`
      );
      score = 0;
    }

    if (company.ceo_to_worker_ratio > 100) {
      violations.push(
        `CEO-to-worker pay ratio ${company.ceo_to_worker_ratio}:1 exceeds maximum 100:1`
      );
      score = 0;
    }

    if (company.has_labor_violations) {
      violations.push('Company has active or recent labor law violations');
      score = 0;
    }

    // Check: MRH violations
    if (metrics.mrh_violations > 0) {
      violations.push(`${metrics.mrh_violations} MRH violations detected in deployment`);
      score -= 50;
    }

    // Check: Constitutional violations
    if (metrics.constitutional_violations > 5) {
      violations.push(`${metrics.constitutional_violations} constitutional violations (threshold: 5)`);
      score -= 30;
    }

    // Check: Reporting compliance
    if (company.quarterly_reports_submitted < 3) {
      violations.push(`Only ${company.quarterly_reports_submitted}/4 quarterly reports submitted`);
      score -= 20;
    }

    // If disqualified, return immediately
    if (score <= 0) {
      return {
        tier: 'none',
        approved: false,
        badge_url: '',
        score: 0,
        expiration_date: new Date(),
        violations,
        recommendations: [
          'Address all violations before reapplying for certification',
          'Submit detailed remediation plan',
        ],
        license_fee_annual: this.calculateBaseFee(company.size_employees),
        license_discount_percent: 0,
        audit_required: true,
      };
    }

    // ========================================================================
    // Tier Evaluation
    // ========================================================================

    // Calculate WIA score
    const wia_score =
      metrics.jobs_affected > 0
        ? (metrics.jobs_created - metrics.jobs_eliminated) / metrics.jobs_affected
        : 0;

    let tier: CertificationTier = 'none';

    // GOLD: Cooperative Model
    if (
      company.employee_ownership_percent >= 50 &&
      company.has_democratic_governance &&
      wia_score >= 0 &&
      metrics.mrh_violations === 0 &&
      company.has_retraining_programs
    ) {
      tier = 'gold';
      score += 10; // Bonus for gold tier
      recommendations.push('Exemplary model - consider publishing case study');
    }
    // SILVER: Workforce Positive
    else if (
      wia_score >= 0.05 &&
      company.has_retraining_programs &&
      company.employee_ownership_percent > 0 &&
      metrics.mrh_violations === 0
    ) {
      tier = 'silver';
      recommendations.push('Consider increasing employee ownership to qualify for Gold');
    }
    // BRONZE: MRH Compliant
    else if (wia_score >= -0.1 && metrics.mrh_violations === 0) {
      tier = 'bronze';
      recommendations.push('Implement retraining programs to qualify for Silver');
      recommendations.push('Consider employee stock ownership plan (ESOP)');
    } else {
      tier = 'none';
      violations.push('Does not meet minimum MRH standards for Bronze certification');
    }

    // ========================================================================
    // Additional Scoring Factors
    // ========================================================================

    // Positive factors
    if (company.has_retraining_programs && metrics.retraining_budget_usd > 0) {
      const retraining_per_affected = metrics.retraining_budget_usd / Math.max(metrics.jobs_affected, 1);
      if (retraining_per_affected > 10000) {
        score += 5; // $10K+ per affected worker
      }
    }

    if (company.quarterly_reports_submitted === 4) {
      score += 5; // Full reporting compliance
    }

    if (wia_score > 0.1) {
      score += 10; // 10%+ job growth
    }

    // Negative factors
    if (metrics.jobs_eliminated > metrics.jobs_created) {
      score -= Math.min(10, (metrics.jobs_eliminated - metrics.jobs_created) / 10);
    }

    if (company.ceo_to_worker_ratio > 50) {
      score -= 5; // Still high even if under 100
    }

    // Cap score at 100
    score = Math.min(100, Math.max(0, score));

    // ========================================================================
    // License Fee Calculation
    // ========================================================================

    const baseFee = this.calculateBaseFee(company.size_employees);
    let discount = 0;

    if (tier === 'gold') {
      discount = 50; // 50% off for cooperatives
    } else if (company.employee_ownership_percent > 0) {
      discount += 10; // 10% for any employee ownership
    }

    if (company.employee_ownership_percent >= 25) {
      discount += 15; // Additional 15% for >25% ownership
    }

    if (wia_score > 0) {
      discount += 10; // 10% for net positive workforce impact
    }

    discount = Math.min(discount, tier === 'gold' ? 50 : 40); // Cap at 40% unless Gold

    // Impact surcharge
    let surcharge = 0;
    if (wia_score < 0 && wia_score >= -0.05) {
      surcharge = 0.05; // 5% of savings
    } else if (wia_score < -0.05 && wia_score >= -0.1) {
      surcharge = 0.15; // 15% of savings
    }

    // ========================================================================
    // Generate Result
    // ========================================================================

    const expirationDate = new Date();
    expirationDate.setFullYear(expirationDate.getFullYear() + 1); // 1 year validity

    return {
      tier,
      approved: tier !== 'none',
      badge_url: this.generateBadgeUrl(tier, company.name),
      score,
      expiration_date: expirationDate,
      violations,
      recommendations,
      license_fee_annual: baseFee * (1 + surcharge) * (1 - discount / 100),
      license_discount_percent: discount,
      audit_required: score < 70 || violations.length > 0,
    };
  }

  /**
   * Calculate base license fee based on company size
   */
  private calculateBaseFee(employees: number): number {
    if (employees < 100) return 10000;
    if (employees < 1000) return 50000;
    if (employees < 10000) return 200000;
    return 500000;
  }

  /**
   * Generate public badge URL for certified companies
   */
  private generateBadgeUrl(tier: CertificationTier, companyName: string): string {
    if (tier === 'none') return '';

    const colors = {
      bronze: 'cd7f32',
      silver: 'c0c0c0',
      gold: 'ffd700',
    };

    const color = colors[tier];
    const label = encodeURIComponent('Responsible AGI');
    const message = encodeURIComponent(`${tier.toUpperCase()} Certified`);

    // Using shields.io badge format
    return `https://img.shields.io/badge/${label}-${message}-${color}?style=for-the-badge&logo=robot`;
  }

  /**
   * Generate markdown badge for README
   */
  generateBadgeMarkdown(result: CertificationResult, companyName: string): string {
    if (!result.approved) {
      return '<!-- Not certified -->';
    }

    const emoji = {
      bronze: '🥉',
      silver: '🥈',
      gold: '🥇',
    }[result.tier as 'bronze' | 'silver' | 'gold'];

    return `
## ${emoji} Responsible AGI Certification

[![${result.tier.toUpperCase()} Certified](${result.badge_url})](LICENSE-COMMERCIAL.md)

**Company**: ${companyName}
**Tier**: ${result.tier.toUpperCase()}
**Score**: ${result.score}/100
**Valid Until**: ${result.expiration_date.toISOString().split('T')[0]}

This deployment has been certified as compliant with responsible AGI standards, including:
- ✅ Workforce Impact Assessment (WIA) monitoring
- ✅ MRH compliance (max 10% displacement)
- ✅ Constitutional governance
- ✅ Quarterly impact reporting

[View Certification Details](LICENSE-COMMERCIAL.md) | [Audit Trail](#)
`;
  }

  /**
   * Validate quarterly report for compliance
   */
  validateQuarterlyReport(report: {
    period: string;
    employees_start: number;
    employees_end: number;
    roles_automated: string[];
    roles_created: string[];
    retraining_participants: number;
    wia_assessments: ImpactAssessment[];
  }): { valid: boolean; issues: string[] } {
    const issues: string[] = [];

    // Check: Basic data completeness
    if (report.employees_start === 0 || report.employees_end === 0) {
      issues.push('Employee counts missing');
    }

    // Check: Net change vs WIA predictions
    const netChange = report.employees_end - report.employees_start;
    const netChangePercent = netChange / report.employees_start;

    if (netChangePercent < -0.1) {
      issues.push(`Net workforce decline ${(netChangePercent * 100).toFixed(1)}% exceeds MRH threshold (-10%)`);
    }

    // Check: WIA assessments exist
    if (report.wia_assessments.length === 0) {
      issues.push('No WIA assessments reported - system may not be monitoring properly');
    }

    // Check: Retraining participation
    const automationCount = report.roles_automated.length;
    if (automationCount > 0 && report.retraining_participants === 0) {
      issues.push('Roles automated but no retraining reported');
    }

    // Check: MRH violations in assessments
    const mrhViolations = report.wia_assessments.filter((a) => !a.mrh_compliant);
    if (mrhViolations.length > 0) {
      issues.push(`${mrhViolations.length} MRH-violating automation proposals detected`);
    }

    return {
      valid: issues.length === 0,
      issues,
    };
  }
}

// ============================================================================
// Factory Function
// ============================================================================

export function createCertification(): ResponsibleAGICertification {
  return new ResponsibleAGICertification();
}
