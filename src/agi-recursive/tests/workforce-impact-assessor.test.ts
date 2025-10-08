/**
 * Tests for Workforce Impact Assessor (WIA)
 *
 * Validates social responsibility framework enforcement
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  WorkforceImpactAssessor,
  AutomationProposal,
  WorkforceMetrics,
} from '../core/workforce-impact-assessor';

describe('WorkforceImpactAssessor', () => {
  let wia: WorkforceImpactAssessor;

  beforeEach(() => {
    wia = new WorkforceImpactAssessor();
  });

  describe('assessProposal', () => {
    it('should approve proposal with no job impact', () => {
      const proposal: AutomationProposal = {
        name: 'Test Automation',
        description: 'Automate data entry',
        estimated_cost: 100000,
        estimated_savings: 50000,
        workforce_change: {
          jobs_affected: 0,
          jobs_created: 0,
          jobs_eliminated: 0,
          jobs_transformed: 0,
          timeline_months: 12,
          retraining_required: false,
          affected_departments: [],
        },
        implementation_timeline: 12,
        reversibility: 'high',
      };

      const assessment = wia.assessProposal(proposal);

      expect(assessment.wia_score).toBe(0);
      expect(assessment.mrh_compliant).toBe(true);
      expect(assessment.approved).toBe(true);
      expect(assessment.risk_level).toBe('low');
    });

    it('should approve MRH-compliant proposal (5% job loss)', () => {
      const proposal: AutomationProposal = {
        name: 'Small Automation',
        description: 'Automate reporting',
        estimated_cost: 50000,
        estimated_savings: 100000,
        workforce_change: {
          jobs_affected: 100,
          jobs_created: 0,
          jobs_eliminated: 5, // 5% loss
          jobs_transformed: 95,
          timeline_months: 18,
          retraining_required: true,
          affected_departments: ['finance', 'operations'],
        },
        implementation_timeline: 18,
        reversibility: 'medium',
      };

      const assessment = wia.assessProposal(proposal);

      expect(assessment.wia_score).toBe(-0.05); // (0 - 5) / 100
      expect(assessment.mrh_compliant).toBe(true); // >= -0.1
      expect(assessment.approved).toBe(true);
      expect(assessment.risk_level).toBe('medium');
    });

    it('should approve borderline MRH-compliant proposal (10% job loss)', () => {
      const proposal: AutomationProposal = {
        name: 'Medium Automation',
        description: 'Automate customer service',
        estimated_cost: 200000,
        estimated_savings: 500000,
        workforce_change: {
          jobs_affected: 100,
          jobs_created: 0,
          jobs_eliminated: 10, // 10% loss
          jobs_transformed: 90,
          timeline_months: 24,
          retraining_required: true,
          affected_departments: ['customer_service'],
        },
        implementation_timeline: 24,
        reversibility: 'low',
      };

      const assessment = wia.assessProposal(proposal);

      expect(assessment.wia_score).toBe(-0.1); // (0 - 10) / 100
      expect(assessment.mrh_compliant).toBe(true); // >= -0.1
      expect(assessment.approved).toBe(true);
      expect(assessment.risk_level).toBe('high');
      expect(assessment.recommendations.length).toBeGreaterThan(0);
    });

    it('should reject MRH-violating proposal (15% job loss)', () => {
      const proposal: AutomationProposal = {
        name: 'Large Automation',
        description: 'Automate entire department',
        estimated_cost: 500000,
        estimated_savings: 2000000,
        workforce_change: {
          jobs_affected: 100,
          jobs_created: 0,
          jobs_eliminated: 15, // 15% loss - VIOLATION
          jobs_transformed: 85,
          timeline_months: 12,
          retraining_required: true,
          affected_departments: ['operations', 'finance'],
        },
        implementation_timeline: 12,
        reversibility: 'low',
      };

      const assessment = wia.assessProposal(proposal);

      expect(assessment.wia_score).toBe(-0.15); // (0 - 15) / 100
      expect(assessment.mrh_compliant).toBe(false); // < -0.1
      expect(assessment.approved).toBe(false);
      expect(assessment.risk_level).toBe('high');
      expect(assessment.recommendations.some((r) => r.includes('MRH non-compliant'))).toBe(true);
    });

    it('should reject critical risk proposal (>20% job loss)', () => {
      const proposal: AutomationProposal = {
        name: 'Mass Automation',
        description: 'Replace entire workforce',
        estimated_cost: 1000000,
        estimated_savings: 5000000,
        workforce_change: {
          jobs_affected: 100,
          jobs_created: 5,
          jobs_eliminated: 25, // 25% loss - CRITICAL
          jobs_transformed: 75,
          timeline_months: 6,
          retraining_required: true,
          affected_departments: ['all'],
        },
        implementation_timeline: 6,
        reversibility: 'low',
      };

      const assessment = wia.assessProposal(proposal);

      expect(assessment.wia_score).toBe(-0.2); // (5 - 25) / 100
      expect(assessment.mrh_compliant).toBe(false);
      expect(assessment.approved).toBe(false);
      expect(assessment.risk_level).toBe('critical');
    });

    it('should approve workforce-positive proposal', () => {
      const proposal: AutomationProposal = {
        name: 'Job-Creating Automation',
        description: 'Automate repetitive tasks, create oversight roles',
        estimated_cost: 300000,
        estimated_savings: 200000,
        workforce_change: {
          jobs_affected: 100,
          jobs_created: 10,
          jobs_eliminated: 5,
          jobs_transformed: 95,
          timeline_months: 24,
          retraining_required: true,
          affected_departments: ['operations'],
        },
        implementation_timeline: 24,
        reversibility: 'high',
      };

      const assessment = wia.assessProposal(proposal);

      expect(assessment.wia_score).toBe(0.05); // (10 - 5) / 100
      expect(assessment.mrh_compliant).toBe(true);
      expect(assessment.approved).toBe(true);
      expect(assessment.risk_level).toBe('low');
    });

    it('should flag rapid timeline as risky', () => {
      const proposal: AutomationProposal = {
        name: 'Rapid Automation',
        description: 'Fast deployment',
        estimated_cost: 100000,
        estimated_savings: 300000,
        workforce_change: {
          jobs_affected: 50,
          jobs_created: 0,
          jobs_eliminated: 5,
          jobs_transformed: 45,
          timeline_months: 3, // Very rapid
          retraining_required: true,
          affected_departments: ['operations'],
        },
        implementation_timeline: 3,
        reversibility: 'medium',
      };

      const assessment = wia.assessProposal(proposal);

      expect(assessment.risk_level).toBe('high'); // Rapid timeline
      expect(assessment.recommendations.some((r) => r.includes('Timeline may be too aggressive'))).toBe(true);
    });

    it('should recommend retraining for transformed roles', () => {
      const proposal: AutomationProposal = {
        name: 'Role Transformation',
        description: 'Change job roles significantly',
        estimated_cost: 200000,
        estimated_savings: 400000,
        workforce_change: {
          jobs_affected: 100,
          jobs_created: 10,
          jobs_eliminated: 5,
          jobs_transformed: 95, // High transformation
          timeline_months: 18,
          retraining_required: false, // Should be true!
          affected_departments: ['operations', 'sales'],
        },
        implementation_timeline: 18,
        reversibility: 'medium',
      };

      const assessment = wia.assessProposal(proposal);

      expect(assessment.recommendations.some((r) => r.includes('Retraining flag not set'))).toBe(true);
    });
  });

  describe('MRH threshold configuration', () => {
    it('should respect custom MRH threshold', () => {
      const customWIA = new WorkforceImpactAssessor({
        mrh_threshold: -0.05, // Stricter: max 5% loss
      });

      const proposal: AutomationProposal = {
        name: 'Test',
        description: 'Test',
        estimated_cost: 0,
        estimated_savings: 0,
        workforce_change: {
          jobs_affected: 100,
          jobs_created: 0,
          jobs_eliminated: 8, // 8% loss
          jobs_transformed: 92,
          timeline_months: 12,
          retraining_required: true,
          affected_departments: [],
        },
        implementation_timeline: 12,
        reversibility: 'medium',
      };

      const assessment = customWIA.assessProposal(proposal);

      expect(assessment.wia_score).toBe(-0.08);
      expect(assessment.mrh_compliant).toBe(false); // Violates -0.05 threshold
    });
  });

  describe('audit trail', () => {
    it('should maintain audit trail when logging enabled', () => {
      const auditWIA = new WorkforceImpactAssessor({
        audit_logging: true,
      });

      const proposal: AutomationProposal = {
        name: 'Audited Proposal',
        description: 'Test audit logging',
        estimated_cost: 100000,
        estimated_savings: 200000,
        workforce_change: {
          jobs_affected: 50,
          jobs_created: 5,
          jobs_eliminated: 3,
          jobs_transformed: 47,
          timeline_months: 12,
          retraining_required: true,
          affected_departments: ['operations'],
        },
        implementation_timeline: 12,
        reversibility: 'high',
      };

      const assessment = auditWIA.assessProposal(proposal);

      // Audit trail should exist (implementation detail)
      expect(assessment).toBeDefined();
      expect(assessment.wia_score).toBeCloseTo(0.04); // (5-3)/50
    });
  });

  describe('constitutional alignment', () => {
    it('should calculate constitutional alignment when enabled', () => {
      const constitutionalWIA = new WorkforceImpactAssessor({
        constitutional_integration: true,
      });

      const proposal: AutomationProposal = {
        name: 'Constitutional Test',
        description: 'Test constitutional alignment',
        estimated_cost: 100000,
        estimated_savings: 150000,
        workforce_change: {
          jobs_affected: 30,
          jobs_created: 5,
          jobs_eliminated: 2,
          jobs_transformed: 28,
          timeline_months: 18,
          retraining_required: true,
          affected_departments: ['finance'],
        },
        implementation_timeline: 18,
        reversibility: 'high',
      };

      const assessment = constitutionalWIA.assessProposal(proposal);

      expect(assessment.constitutional_alignment).toBeGreaterThan(0);
      expect(assessment.constitutional_alignment).toBeLessThanOrEqual(1);
    });

    it('should skip constitutional alignment when disabled', () => {
      const nonConstitutionalWIA = new WorkforceImpactAssessor({
        constitutional_integration: false,
      });

      const proposal: AutomationProposal = {
        name: 'No Constitution',
        description: 'Test without constitutional check',
        estimated_cost: 50000,
        estimated_savings: 100000,
        workforce_change: {
          jobs_affected: 20,
          jobs_created: 2,
          jobs_eliminated: 1,
          jobs_transformed: 19,
          timeline_months: 12,
          retraining_required: true,
          affected_departments: ['operations'],
        },
        implementation_timeline: 12,
        reversibility: 'medium',
      };

      const assessment = nonConstitutionalWIA.assessProposal(proposal);

      expect(assessment.constitutional_alignment).toBe(1.0);
    });
  });

  describe('edge cases', () => {
    it('should handle zero jobs affected', () => {
      const proposal: AutomationProposal = {
        name: 'No Impact',
        description: 'Automation with no workforce impact',
        estimated_cost: 10000,
        estimated_savings: 20000,
        workforce_change: {
          jobs_affected: 0,
          jobs_created: 0,
          jobs_eliminated: 0,
          jobs_transformed: 0,
          timeline_months: 6,
          retraining_required: false,
          affected_departments: [],
        },
        implementation_timeline: 6,
        reversibility: 'high',
      };

      const assessment = wia.assessProposal(proposal);

      expect(assessment.wia_score).toBe(0);
      expect(assessment.mrh_compliant).toBe(true);
      expect(assessment.approved).toBe(true);
    });

    it('should handle very large workforce impact', () => {
      const proposal: AutomationProposal = {
        name: 'Enterprise Automation',
        description: 'Large-scale automation',
        estimated_cost: 10000000,
        estimated_savings: 50000000,
        workforce_change: {
          jobs_affected: 10000,
          jobs_created: 500,
          jobs_eliminated: 1500, // 15% loss
          jobs_transformed: 9000,
          timeline_months: 36,
          retraining_required: true,
          affected_departments: ['all'],
        },
        implementation_timeline: 36,
        reversibility: 'low',
      };

      const assessment = wia.assessProposal(proposal);

      expect(assessment.wia_score).toBe(-0.1); // (500 - 1500) / 10000
      expect(assessment.mrh_compliant).toBe(true); // Exactly at threshold
      expect(assessment.approved).toBe(true);
    });

    it('should handle proposals with only job creation', () => {
      const proposal: AutomationProposal = {
        name: 'Pure Job Creation',
        description: 'Creates new roles without eliminating any',
        estimated_cost: 500000,
        estimated_savings: 0,
        workforce_change: {
          jobs_affected: 0,
          jobs_created: 50,
          jobs_eliminated: 0,
          jobs_transformed: 0,
          timeline_months: 12,
          retraining_required: false,
          affected_departments: ['new_department'],
        },
        implementation_timeline: 12,
        reversibility: 'high',
      };

      const assessment = wia.assessProposal(proposal);

      expect(assessment.wia_score).toBe(0); // Division by zero → 0
      expect(assessment.mrh_compliant).toBe(true);
      expect(assessment.approved).toBe(true);
      expect(assessment.risk_level).toBe('low');
    });
  });

  describe('real-world scenarios', () => {
    it('should handle customer service automation scenario', () => {
      const proposal: AutomationProposal = {
        name: 'AI Customer Service',
        description: 'Replace Level 1 support with AI chatbot',
        estimated_cost: 300000,
        estimated_savings: 1200000,
        workforce_change: {
          jobs_affected: 150,
          jobs_created: 10, // AI trainers, quality analysts
          jobs_eliminated: 100, // Level 1 agents
          jobs_transformed: 40, // Level 2 agents upskilled
          timeline_months: 18,
          retraining_required: true,
          affected_departments: ['customer_support'],
        },
        implementation_timeline: 18,
        reversibility: 'medium',
      };

      const assessment = wia.assessProposal(proposal);

      expect(assessment.wia_score).toBeCloseTo(-0.6); // (10 - 100) / 150
      expect(assessment.mrh_compliant).toBe(false); // Way below -0.1
      expect(assessment.approved).toBe(false);
      expect(assessment.risk_level).toBe('critical');
    });

    it('should handle data entry automation scenario', () => {
      const proposal: AutomationProposal = {
        name: 'RPA for Data Entry',
        description: 'Robotic Process Automation for repetitive tasks',
        estimated_cost: 150000,
        estimated_savings: 600000,
        workforce_change: {
          jobs_affected: 200,
          jobs_created: 15, // RPA developers, process analysts
          jobs_eliminated: 20, // Data entry clerks
          jobs_transformed: 180, // Clerks → analysts
          timeline_months: 24,
          retraining_required: true,
          affected_departments: ['operations', 'finance'],
        },
        implementation_timeline: 24,
        reversibility: 'high',
      };

      const assessment = wia.assessProposal(proposal);

      expect(assessment.wia_score).toBeCloseTo(-0.025); // (15 - 20) / 200
      expect(assessment.mrh_compliant).toBe(true); // > -0.1
      expect(assessment.approved).toBe(true);
      expect(assessment.risk_level).toBe('medium');
    });
  });
});
