/**
 * Tests for Responsible AGI Certification System
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  ResponsibleAGICertification,
  CompanyProfile,
  DeploymentMetrics,
  CertificationTier,
} from '../core/responsible-agi-certification';

describe('ResponsibleAGICertification', () => {
  let cert: ResponsibleAGICertification;

  beforeEach(() => {
    cert = new ResponsibleAGICertification();
  });

  describe('Gold Tier - Cooperative Model', () => {
    it('should award Gold certification for exemplary cooperative', () => {
      const company: CompanyProfile = {
        name: 'Worker Cooperative Inc',
        size_employees: 500,
        ceo_to_worker_ratio: 3, // Nearly flat structure
        employee_ownership_percent: 100, // 100% worker-owned
        layoffs_last_24_months: 0,
        layoffs_last_24_months_percent: 0,
        has_labor_violations: false,
        has_democratic_governance: true,
        has_retraining_programs: true,
        quarterly_reports_submitted: 4,
      };

      const metrics: DeploymentMetrics = {
        total_queries_processed: 10000,
        queries_with_automation_intent: 100,
        jobs_affected: 50,
        jobs_created: 10,
        jobs_eliminated: 5,
        net_workforce_change: 5,
        wia_average_score: 0.1, // 10% net positive
        mrh_violations: 0,
        constitutional_violations: 0,
        retraining_budget_usd: 500000,
      };

      const result = cert.evaluate(company, metrics);

      expect(result.tier).toBe('gold');
      expect(result.approved).toBe(true);
      expect(result.score).toBeGreaterThan(90);
      expect(result.license_discount_percent).toBe(50); // Max discount
      expect(result.badge_url.toUpperCase()).toContain('GOLD');
    });

    it('should require 50%+ ownership for Gold', () => {
      const company: CompanyProfile = {
        name: 'Almost Cooperative',
        size_employees: 200,
        ceo_to_worker_ratio: 10,
        employee_ownership_percent: 49, // Just under threshold
        layoffs_last_24_months: 0,
        layoffs_last_24_months_percent: 0,
        has_labor_violations: false,
        has_democratic_governance: true,
        has_retraining_programs: true,
        quarterly_reports_submitted: 4,
      };

      const metrics: DeploymentMetrics = {
        total_queries_processed: 5000,
        queries_with_automation_intent: 50,
        jobs_affected: 20,
        jobs_created: 5,
        jobs_eliminated: 2,
        net_workforce_change: 3,
        wia_average_score: 0.15,
        mrh_violations: 0,
        constitutional_violations: 0,
        retraining_budget_usd: 200000,
      };

      const result = cert.evaluate(company, metrics);

      expect(result.tier).not.toBe('gold'); // Should be Silver
      expect(result.tier).toBe('silver');
    });
  });

  describe('Silver Tier - Workforce Positive', () => {
    it('should award Silver for workforce-positive deployment', () => {
      const company: CompanyProfile = {
        name: 'Progressive Tech Corp',
        size_employees: 1000,
        ceo_to_worker_ratio: 20,
        employee_ownership_percent: 15, // Some ownership
        layoffs_last_24_months: 0,
        layoffs_last_24_months_percent: 0,
        has_labor_violations: false,
        has_democratic_governance: false,
        has_retraining_programs: true,
        quarterly_reports_submitted: 4,
      };

      const metrics: DeploymentMetrics = {
        total_queries_processed: 50000,
        queries_with_automation_intent: 500,
        jobs_affected: 100,
        jobs_created: 15,
        jobs_eliminated: 10,
        net_workforce_change: 5,
        wia_average_score: 0.05, // Exactly at Silver threshold
        mrh_violations: 0,
        constitutional_violations: 2,
        retraining_budget_usd: 1000000,
      };

      const result = cert.evaluate(company, metrics);

      expect(result.tier).toBe('silver');
      expect(result.approved).toBe(true);
      expect(result.score).toBeGreaterThan(70);
      expect(result.license_discount_percent).toBeGreaterThan(0);
    });

    it('should require WIA score >= 0.05 for Silver', () => {
      const company: CompanyProfile = {
        name: 'Tech Corp',
        size_employees: 500,
        ceo_to_worker_ratio: 25,
        employee_ownership_percent: 10,
        layoffs_last_24_months: 0,
        layoffs_last_24_months_percent: 0,
        has_labor_violations: false,
        has_democratic_governance: false,
        has_retraining_programs: true,
        quarterly_reports_submitted: 4,
      };

      const metrics: DeploymentMetrics = {
        total_queries_processed: 10000,
        queries_with_automation_intent: 100,
        jobs_affected: 100,
        jobs_created: 5,
        jobs_eliminated: 5,
        net_workforce_change: 0,
        wia_average_score: 0.0, // Below Silver threshold
        mrh_violations: 0,
        constitutional_violations: 0,
        retraining_budget_usd: 300000,
      };

      const result = cert.evaluate(company, metrics);

      expect(result.tier).toBe('bronze'); // Not Silver
    });
  });

  describe('Bronze Tier - MRH Compliant', () => {
    it('should award Bronze for basic MRH compliance', () => {
      const company: CompanyProfile = {
        name: 'Standard Corp',
        size_employees: 2000,
        ceo_to_worker_ratio: 50,
        employee_ownership_percent: 0,
        layoffs_last_24_months: 0,
        layoffs_last_24_months_percent: 0,
        has_labor_violations: false,
        has_democratic_governance: false,
        has_retraining_programs: false,
        quarterly_reports_submitted: 3,
      };

      const metrics: DeploymentMetrics = {
        total_queries_processed: 100000,
        queries_with_automation_intent: 1000,
        jobs_affected: 500,
        jobs_created: 10,
        jobs_eliminated: 50, // 10% loss - at threshold
        net_workforce_change: -40,
        wia_average_score: -0.08, // Above -0.1
        mrh_violations: 0,
        constitutional_violations: 3,
        retraining_budget_usd: 100000,
      };

      const result = cert.evaluate(company, metrics);

      expect(result.tier).toBe('bronze');
      expect(result.approved).toBe(true);
      expect(result.score).toBeGreaterThan(50);
      expect(result.recommendations.some((r) => r.toLowerCase().includes('retraining'))).toBe(true);
    });
  });

  describe('Disqualifications', () => {
    it('should disqualify companies with recent layoffs >5%', () => {
      const company: CompanyProfile = {
        name: 'Layoff Corp',
        size_employees: 1000,
        ceo_to_worker_ratio: 30,
        employee_ownership_percent: 0,
        layoffs_last_24_months: 100,
        layoffs_last_24_months_percent: 0.1, // 10% layoffs
        has_labor_violations: false,
        has_democratic_governance: false,
        has_retraining_programs: true,
        quarterly_reports_submitted: 4,
      };

      const metrics: DeploymentMetrics = {
        total_queries_processed: 1000,
        queries_with_automation_intent: 10,
        jobs_affected: 10,
        jobs_created: 2,
        jobs_eliminated: 1,
        net_workforce_change: 1,
        wia_average_score: 0.1,
        mrh_violations: 0,
        constitutional_violations: 0,
        retraining_budget_usd: 50000,
      };

      const result = cert.evaluate(company, metrics);

      expect(result.tier).toBe('none');
      expect(result.approved).toBe(false);
      expect(result.score).toBe(0);
      expect(result.violations.some((v) => v.includes('laid off'))).toBe(true);
    });

    it('should disqualify companies with CEO ratio >100:1', () => {
      const company: CompanyProfile = {
        name: 'Inequality Inc',
        size_employees: 5000,
        ceo_to_worker_ratio: 150, // Excessive
        employee_ownership_percent: 0,
        layoffs_last_24_months: 0,
        layoffs_last_24_months_percent: 0,
        has_labor_violations: false,
        has_democratic_governance: false,
        has_retraining_programs: true,
        quarterly_reports_submitted: 4,
      };

      const metrics: DeploymentMetrics = {
        total_queries_processed: 50000,
        queries_with_automation_intent: 100,
        jobs_affected: 50,
        jobs_created: 5,
        jobs_eliminated: 5,
        net_workforce_change: 0,
        wia_average_score: 0.0,
        mrh_violations: 0,
        constitutional_violations: 0,
        retraining_budget_usd: 200000,
      };

      const result = cert.evaluate(company, metrics);

      expect(result.tier).toBe('none');
      expect(result.approved).toBe(false);
      expect(result.violations.some((v) => v.includes('CEO-to-worker pay ratio'))).toBe(true);
    });

    it('should disqualify companies with labor violations', () => {
      const company: CompanyProfile = {
        name: 'Violator Corp',
        size_employees: 1000,
        ceo_to_worker_ratio: 40,
        employee_ownership_percent: 5,
        layoffs_last_24_months: 0,
        layoffs_last_24_months_percent: 0,
        has_labor_violations: true, // Active violations
        has_democratic_governance: false,
        has_retraining_programs: true,
        quarterly_reports_submitted: 4,
      };

      const metrics: DeploymentMetrics = {
        total_queries_processed: 10000,
        queries_with_automation_intent: 50,
        jobs_affected: 20,
        jobs_created: 3,
        jobs_eliminated: 2,
        net_workforce_change: 1,
        wia_average_score: 0.05,
        mrh_violations: 0,
        constitutional_violations: 0,
        retraining_budget_usd: 100000,
      };

      const result = cert.evaluate(company, metrics);

      expect(result.tier).toBe('none');
      expect(result.approved).toBe(false);
      expect(result.violations.some((v) => v.includes('labor law violations'))).toBe(true);
    });

    it('should disqualify deployments with MRH violations', () => {
      const company: CompanyProfile = {
        name: 'Good Company',
        size_employees: 500,
        ceo_to_worker_ratio: 20,
        employee_ownership_percent: 10,
        layoffs_last_24_months: 0,
        layoffs_last_24_months_percent: 0,
        has_labor_violations: false,
        has_democratic_governance: false,
        has_retraining_programs: true,
        quarterly_reports_submitted: 4,
      };

      const metrics: DeploymentMetrics = {
        total_queries_processed: 5000,
        queries_with_automation_intent: 50,
        jobs_affected: 100,
        jobs_created: 5,
        jobs_eliminated: 5,
        net_workforce_change: 0,
        wia_average_score: 0.0,
        mrh_violations: 5, // Violations detected
        constitutional_violations: 0,
        retraining_budget_usd: 200000,
      };

      const result = cert.evaluate(company, metrics);

      expect(result.score).toBeLessThan(50);
      expect(result.violations.some((v) => v.includes('MRH violations'))).toBe(true);
    });
  });

  describe('License Fee Calculation', () => {
    it('should calculate base fee for small company (<100 employees)', () => {
      const company: CompanyProfile = {
        name: 'Small Startup',
        size_employees: 50,
        ceo_to_worker_ratio: 5,
        employee_ownership_percent: 20,
        layoffs_last_24_months: 0,
        layoffs_last_24_months_percent: 0,
        has_labor_violations: false,
        has_democratic_governance: true,
        has_retraining_programs: true,
        quarterly_reports_submitted: 4,
      };

      const metrics: DeploymentMetrics = {
        total_queries_processed: 1000,
        queries_with_automation_intent: 10,
        jobs_affected: 5,
        jobs_created: 1,
        jobs_eliminated: 0,
        net_workforce_change: 1,
        wia_average_score: 0.2,
        mrh_violations: 0,
        constitutional_violations: 0,
        retraining_budget_usd: 50000,
      };

      const result = cert.evaluate(company, metrics);

      expect(result.license_fee_annual).toBeLessThan(10000); // With discounts
      expect(result.license_discount_percent).toBeGreaterThan(0);
    });

    it('should calculate base fee for medium company (100-1000)', () => {
      const company: CompanyProfile = {
        name: 'Medium Corp',
        size_employees: 500,
        ceo_to_worker_ratio: 30,
        employee_ownership_percent: 0,
        layoffs_last_24_months: 0,
        layoffs_last_24_months_percent: 0,
        has_labor_violations: false,
        has_democratic_governance: false,
        has_retraining_programs: true,
        quarterly_reports_submitted: 4,
      };

      const metrics: DeploymentMetrics = {
        total_queries_processed: 10000,
        queries_with_automation_intent: 100,
        jobs_affected: 50,
        jobs_created: 5,
        jobs_eliminated: 5,
        net_workforce_change: 0,
        wia_average_score: 0.0,
        mrh_violations: 0,
        constitutional_violations: 0,
        retraining_budget_usd: 200000,
      };

      const result = cert.evaluate(company, metrics);

      expect(result.license_fee_annual).toBeGreaterThanOrEqual(40000);
      expect(result.license_fee_annual).toBeLessThanOrEqual(50000);
    });

    it('should apply 50% discount for Gold cooperatives', () => {
      const company: CompanyProfile = {
        name: 'Pure Cooperative',
        size_employees: 1000,
        ceo_to_worker_ratio: 2,
        employee_ownership_percent: 100,
        layoffs_last_24_months: 0,
        layoffs_last_24_months_percent: 0,
        has_labor_violations: false,
        has_democratic_governance: true,
        has_retraining_programs: true,
        quarterly_reports_submitted: 4,
      };

      const metrics: DeploymentMetrics = {
        total_queries_processed: 50000,
        queries_with_automation_intent: 500,
        jobs_affected: 100,
        jobs_created: 20,
        jobs_eliminated: 10,
        net_workforce_change: 10,
        wia_average_score: 0.1,
        mrh_violations: 0,
        constitutional_violations: 0,
        retraining_budget_usd: 1000000,
      };

      const result = cert.evaluate(company, metrics);

      expect(result.tier).toBe('gold');
      expect(result.license_discount_percent).toBe(50);
      expect(result.license_fee_annual).toBe(200000 * 0.5); // 50% off base
    });
  });

  describe('Badge Generation', () => {
    it('should generate badge URL for certified companies', () => {
      const company: CompanyProfile = {
        name: 'Badge Test Corp',
        size_employees: 200,
        ceo_to_worker_ratio: 20,
        employee_ownership_percent: 10,
        layoffs_last_24_months: 0,
        layoffs_last_24_months_percent: 0,
        has_labor_violations: false,
        has_democratic_governance: false,
        has_retraining_programs: true,
        quarterly_reports_submitted: 4,
      };

      const metrics: DeploymentMetrics = {
        total_queries_processed: 5000,
        queries_with_automation_intent: 50,
        jobs_affected: 30,
        jobs_created: 5,
        jobs_eliminated: 3,
        net_workforce_change: 2,
        wia_average_score: 0.067,
        mrh_violations: 0,
        constitutional_violations: 0,
        retraining_budget_usd: 150000,
      };

      const result = cert.evaluate(company, metrics);

      expect(result.badge_url).toBeDefined();
      expect(result.badge_url).toContain('img.shields.io');
      expect(result.badge_url.toUpperCase()).toContain(result.tier.toUpperCase());
    });

    it('should generate markdown badge', () => {
      const company: CompanyProfile = {
        name: 'Markdown Corp',
        size_employees: 300,
        ceo_to_worker_ratio: 15,
        employee_ownership_percent: 25,
        layoffs_last_24_months: 0,
        layoffs_last_24_months_percent: 0,
        has_labor_violations: false,
        has_democratic_governance: false,
        has_retraining_programs: true,
        quarterly_reports_submitted: 4,
      };

      const metrics: DeploymentMetrics = {
        total_queries_processed: 10000,
        queries_with_automation_intent: 100,
        jobs_affected: 50,
        jobs_created: 10,
        jobs_eliminated: 5,
        net_workforce_change: 5,
        wia_average_score: 0.1,
        mrh_violations: 0,
        constitutional_violations: 0,
        retraining_budget_usd: 300000,
      };

      const result = cert.evaluate(company, metrics);
      const markdown = cert.generateBadgeMarkdown(result, company.name);

      expect(markdown).toContain('Responsible AGI Certification');
      expect(markdown).toContain(result.tier.toUpperCase());
      expect(markdown).toContain(company.name);
      expect(markdown).toContain(`${result.score}/100`);
    });
  });

  describe('Quarterly Report Validation', () => {
    it('should validate complete quarterly report', () => {
      const report = {
        period: 'Q1 2025',
        employees_start: 1000,
        employees_end: 1005,
        roles_automated: ['data-entry-clerk'],
        roles_created: ['ai-trainer', 'process-analyst'],
        retraining_participants: 50,
        wia_assessments: [
          {
            wia_score: 0.05,
            mrh_compliant: true,
            risk_level: 'low' as const,
            approved: true,
            recommendations: [],
            audit_trail: [],
            constitutional_alignment: 0.95,
          },
        ],
      };

      const validation = cert.validateQuarterlyReport(report);

      expect(validation.valid).toBe(true);
      expect(validation.issues).toHaveLength(0);
    });

    it('should flag excessive workforce decline', () => {
      const report = {
        period: 'Q1 2025',
        employees_start: 1000,
        employees_end: 850, // 15% decline
        roles_automated: ['customer-service'],
        roles_created: [],
        retraining_participants: 0,
        wia_assessments: [],
      };

      const validation = cert.validateQuarterlyReport(report);

      expect(validation.valid).toBe(false);
      expect(validation.issues.some((i) => i.includes('exceeds MRH threshold'))).toBe(true);
    });

    it('should flag missing retraining', () => {
      const report = {
        period: 'Q1 2025',
        employees_start: 500,
        employees_end: 495,
        roles_automated: ['data-entry', 'receptionist'],
        roles_created: ['analyst'],
        retraining_participants: 0, // No retraining!
        wia_assessments: [],
      };

      const validation = cert.validateQuarterlyReport(report);

      expect(validation.valid).toBe(false);
      expect(validation.issues.some((i) => i.toLowerCase().includes('no retraining'))).toBe(true);
    });

    it('should flag MRH violations in assessments', () => {
      const report = {
        period: 'Q1 2025',
        employees_start: 1000,
        employees_end: 980,
        roles_automated: ['support-agent'],
        roles_created: ['ai-trainer'],
        retraining_participants: 20,
        wia_assessments: [
          {
            wia_score: -0.15, // Violation
            mrh_compliant: false,
            risk_level: 'high' as const,
            approved: false,
            recommendations: [],
            audit_trail: [],
            constitutional_alignment: 0.5,
          },
          {
            wia_score: -0.12, // Another violation
            mrh_compliant: false,
            risk_level: 'high' as const,
            approved: false,
            recommendations: [],
            audit_trail: [],
            constitutional_alignment: 0.6,
          },
        ],
      };

      const validation = cert.validateQuarterlyReport(report);

      expect(validation.valid).toBe(false);
      expect(validation.issues.some((i) => i.includes('MRH-violating'))).toBe(true);
    });
  });
});
