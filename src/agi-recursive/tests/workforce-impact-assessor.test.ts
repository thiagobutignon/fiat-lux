/**
 * @file workforce-impact-assessor.test.ts
 * Tests for Workforce Impact Assessor (WIA)
 *
 * Key capabilities tested:
 * - WIA score calculation
 * - MRH compliance checking
 * - Risk level assessment
 * - Recommendation generation
 * - Approval decision logic
 * - Audit logging
 * - Statistics tracking
 * - Constitutional alignment
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  WorkforceImpactAssessor,
  AutomationProposal,
  WorkforceMetrics,
  createWIA,
  calculateWIAScore,
  checkMRHCompliance,
} from '../core/workforce-impact-assessor';

describe('WorkforceImpactAssessor', () => {
  let wia: WorkforceImpactAssessor;
  let testProposal: AutomationProposal;

  beforeEach(() => {
    wia = new WorkforceImpactAssessor();

    testProposal = {
      name: 'Test Automation',
      description: 'Test proposal',
      estimated_cost: 100000,
      estimated_savings: 200000,
      workforce_change: {
        jobs_affected: 100,
        jobs_created: 10,
        jobs_eliminated: 5,
        jobs_transformed: 20,
        timeline_months: 12,
        retraining_required: true,
        affected_departments: ['IT', 'Operations'],
      },
      implementation_timeline: 12,
      reversibility: 'medium',
    };
  });

  describe('Constructor', () => {
    it('should create instance with default config', () => {
      expect(wia).toBeInstanceOf(WorkforceImpactAssessor);
    });

    it('should accept custom config', () => {
      const customWIA = new WorkforceImpactAssessor({
        mrh_threshold: -0.2,
        enable_gradual_rollout: false,
      });

      expect(customWIA).toBeInstanceOf(WorkforceImpactAssessor);
    });

    it('should use default MRH threshold of -0.1', () => {
      const proposal = { ...testProposal };
      proposal.workforce_change.jobs_eliminated = 11; // 10 created - 11 eliminated = -0.01 score (compliant)

      const assessment = wia.assessProposal(proposal);
      expect(assessment.mrh_compliant).toBe(true);
    });
  });

  describe('WIA Score Calculation', () => {
    it('should calculate positive WIA score', () => {
      // 10 created - 5 eliminated = 5 net / 100 affected = 0.05
      const assessment = wia.assessProposal(testProposal);

      expect(assessment.wia_score).toBeCloseTo(0.05, 2);
    });

    it('should calculate negative WIA score', () => {
      testProposal.workforce_change.jobs_created = 5;
      testProposal.workforce_change.jobs_eliminated = 20;
      // 5 created - 20 eliminated = -15 net / 100 affected = -0.15

      const assessment = wia.assessProposal(testProposal);

      expect(assessment.wia_score).toBeCloseTo(-0.15, 2);
    });

    it('should calculate zero WIA score', () => {
      testProposal.workforce_change.jobs_created = 10;
      testProposal.workforce_change.jobs_eliminated = 10;

      const assessment = wia.assessProposal(testProposal);

      expect(assessment.wia_score).toBe(0);
    });

    it('should handle zero affected jobs', () => {
      testProposal.workforce_change.jobs_affected = 0;

      const assessment = wia.assessProposal(testProposal);

      expect(assessment.wia_score).toBe(0);
    });
  });

  describe('MRH Compliance', () => {
    it('should be compliant for positive impact', () => {
      // Default: 10 created - 5 eliminated = +0.05 (compliant)
      const assessment = wia.assessProposal(testProposal);

      expect(assessment.mrh_compliant).toBe(true);
    });

    it('should be compliant at threshold boundary', () => {
      testProposal.workforce_change.jobs_created = 0;
      testProposal.workforce_change.jobs_eliminated = 10;
      // 0 created - 10 eliminated = -0.1 (exactly at threshold)

      const assessment = wia.assessProposal(testProposal);

      expect(assessment.mrh_compliant).toBe(true);
    });

    it('should be non-compliant beyond threshold', () => {
      testProposal.workforce_change.jobs_created = 0;
      testProposal.workforce_change.jobs_eliminated = 11;
      // 0 created - 11 eliminated = -0.11 (beyond threshold)

      const assessment = wia.assessProposal(testProposal);

      expect(assessment.mrh_compliant).toBe(false);
    });

    it('should support custom threshold', () => {
      const customWIA = new WorkforceImpactAssessor({ mrh_threshold: -0.2 });

      testProposal.workforce_change.jobs_eliminated = 15;
      // -0.15 score (compliant with -0.2 threshold)

      const assessment = customWIA.assessProposal(testProposal);

      expect(assessment.mrh_compliant).toBe(true);
    });
  });

  describe('Risk Level Assessment', () => {
    it('should assess low risk for positive impact', () => {
      const assessment = wia.assessProposal(testProposal);

      expect(assessment.risk_level).toBe('low');
    });

    it('should assess medium risk for small negative impact', () => {
      testProposal.workforce_change.jobs_created = 5;
      testProposal.workforce_change.jobs_eliminated = 10;
      // -0.05 score (negative but < 10%)

      const assessment = wia.assessProposal(testProposal);

      expect(assessment.risk_level).toBe('medium');
    });

    it('should assess high risk for 10-20% job loss', () => {
      testProposal.workforce_change.jobs_created = 0;
      testProposal.workforce_change.jobs_eliminated = 15;
      // -0.15 score (10-20% loss)

      const assessment = wia.assessProposal(testProposal);

      expect(assessment.risk_level).toBe('high');
    });

    it('should assess critical risk for >20% job loss', () => {
      testProposal.workforce_change.jobs_created = 0;
      testProposal.workforce_change.jobs_eliminated = 25;
      // -0.25 score (>20% loss)

      const assessment = wia.assessProposal(testProposal);

      expect(assessment.risk_level).toBe('critical');
    });

    it('should assess high risk for rapid timeline with job loss', () => {
      testProposal.workforce_change.jobs_created = 0;
      testProposal.workforce_change.jobs_eliminated = 8;
      testProposal.workforce_change.timeline_months = 3;
      // Negative score + short timeline

      const assessment = wia.assessProposal(testProposal);

      expect(assessment.risk_level).toBe('high');
    });

    it('should assess medium risk for high transformation rate', () => {
      testProposal.workforce_change.jobs_eliminated = 0;
      testProposal.workforce_change.jobs_transformed = 60;
      // 60/100 = 60% transformation

      const assessment = wia.assessProposal(testProposal);

      expect(assessment.risk_level).toBe('medium');
    });
  });

  describe('Approval Decisions', () => {
    it('should approve low risk compliant proposals', () => {
      const assessment = wia.assessProposal(testProposal);

      expect(assessment.approved).toBe(true);
    });

    it('should reject critical risk non-compliant proposals', () => {
      testProposal.workforce_change.jobs_eliminated = 30;
      // Critical risk + non-compliant

      const assessment = wia.assessProposal(testProposal);

      expect(assessment.approved).toBe(false);
    });

    it('should require review for high risk proposals', () => {
      testProposal.workforce_change.jobs_created = 0;
      testProposal.workforce_change.jobs_eliminated = 15;
      // High risk (-0.15 score)

      const assessment = wia.assessProposal(testProposal);

      expect(assessment.approved).toBe(false);
    });

    it('should approve medium risk compliant proposals', () => {
      testProposal.workforce_change.jobs_eliminated = 8;
      // Medium risk but compliant

      const assessment = wia.assessProposal(testProposal);

      expect(assessment.approved).toBe(true);
    });
  });

  describe('Recommendations', () => {
    it('should recommend phased implementation for non-compliant', () => {
      testProposal.workforce_change.jobs_created = 0;
      testProposal.workforce_change.jobs_eliminated = 15;
      // -0.15 score (non-compliant)

      const assessment = wia.assessProposal(testProposal);

      expect(assessment.recommendations.some((r) => r.includes('phased'))).toBe(true);
    });

    it('should recommend timeline extension for aggressive rollout', () => {
      testProposal.workforce_change.jobs_eliminated = 15;
      testProposal.workforce_change.timeline_months = 6;

      const assessment = wia.assessProposal(testProposal);

      expect(assessment.recommendations.some((r) => r.includes('Timeline'))).toBe(true);
    });

    it('should recommend training for high transformation', () => {
      testProposal.workforce_change.jobs_transformed = 40;

      const assessment = wia.assessProposal(testProposal);

      expect(assessment.recommendations.some((r) => r.includes('training'))).toBe(true);
    });

    it('should flag retraining inconsistency', () => {
      testProposal.workforce_change.retraining_required = false;
      testProposal.workforce_change.jobs_transformed = 30;

      const assessment = wia.assessProposal(testProposal);

      expect(assessment.recommendations.some((r) => r.includes('Retraining flag'))).toBe(true);
    });

    it('should warn about low reversibility', () => {
      testProposal.reversibility = 'low';
      testProposal.workforce_change.jobs_created = 0;
      testProposal.workforce_change.jobs_eliminated = 15;
      // Non-compliant + low reversibility

      const assessment = wia.assessProposal(testProposal);

      expect(assessment.recommendations.some((r) => r.includes('reversibility'))).toBe(true);
    });

    it('should suggest highlighting positive impact', () => {
      testProposal.workforce_change.jobs_created = 50;

      const assessment = wia.assessProposal(testProposal);

      expect(assessment.recommendations.some((r) => r.includes('Positive'))).toBe(true);
    });

    it('should recommend gradual rollout for large eliminations', () => {
      testProposal.workforce_change.jobs_eliminated = 25;

      const assessment = wia.assessProposal(testProposal);

      expect(assessment.recommendations.some((r) => r.includes('gradual rollout'))).toBe(true);
    });
  });

  describe('Constitutional Alignment', () => {
    it('should assess constitutional alignment by default', () => {
      const assessment = wia.assessProposal(testProposal);

      expect(assessment.constitutional_alignment).toBeGreaterThan(0);
      expect(assessment.constitutional_alignment).toBeLessThanOrEqual(1);
    });

    it('should penalize high elimination low creation ratio', () => {
      testProposal.workforce_change.jobs_created = 5;
      testProposal.workforce_change.jobs_eliminated = 20;

      const assessment = wia.assessProposal(testProposal);

      expect(assessment.constitutional_alignment).toBeLessThan(1.0);
    });

    it('should penalize low reversibility high impact', () => {
      testProposal.reversibility = 'low';
      testProposal.workforce_change.jobs_eliminated = 60;

      const assessment = wia.assessProposal(testProposal);

      expect(assessment.constitutional_alignment).toBeLessThan(1.0);
    });

    it('should reward retraining programs', () => {
      testProposal.workforce_change.retraining_required = true;
      testProposal.workforce_change.jobs_transformed = 30;

      const assessment = wia.assessProposal(testProposal);

      // Should get bonus for retraining
      expect(assessment.constitutional_alignment).toBeGreaterThan(0.5);
    });

    it('should support disabling constitutional integration', () => {
      const customWIA = new WorkforceImpactAssessor({ constitutional_integration: false });

      const assessment = customWIA.assessProposal(testProposal);

      expect(assessment.constitutional_alignment).toBe(1.0);
    });
  });

  describe('Audit Logging', () => {
    it('should log assessments by default', () => {
      wia.assessProposal(testProposal);

      const auditLog = wia.getAuditLog();
      expect(auditLog.length).toBe(1);
    });

    it('should include assessment details in audit entry', () => {
      wia.assessProposal(testProposal);

      const auditLog = wia.getAuditLog();
      expect(auditLog[0].action).toContain('assess_proposal');
      expect(auditLog[0].decision).toBeDefined();
      expect(auditLog[0].justification).toContain('WIA Score');
    });

    it('should support disabling audit logging', () => {
      const customWIA = new WorkforceImpactAssessor({ audit_logging: false });

      customWIA.assessProposal(testProposal);

      const auditLog = customWIA.getAuditLog();
      expect(auditLog.length).toBe(0);
    });

    it('should accumulate audit entries', () => {
      wia.assessProposal(testProposal);
      wia.assessProposal({ ...testProposal, name: 'Proposal 2' });
      wia.assessProposal({ ...testProposal, name: 'Proposal 3' });

      const auditLog = wia.getAuditLog();
      expect(auditLog.length).toBe(3);
    });

    it('should include timestamp in audit entries', () => {
      const before = Date.now();
      wia.assessProposal(testProposal);
      const after = Date.now();

      const auditLog = wia.getAuditLog();
      expect(auditLog[0].timestamp).toBeGreaterThanOrEqual(before);
      expect(auditLog[0].timestamp).toBeLessThanOrEqual(after);
    });
  });

  describe('Assessment History', () => {
    it('should store assessment history', () => {
      wia.assessProposal(testProposal);

      const history = wia.getHistory();
      expect(history.size).toBe(1);
      expect(history.has('Test Automation')).toBe(true);
    });

    it('should retrieve assessments by proposal name', () => {
      wia.assessProposal(testProposal);

      const history = wia.getHistory();
      const assessment = history.get('Test Automation');

      expect(assessment).toBeDefined();
      expect(assessment?.wia_score).toBeCloseTo(0.05, 2);
    });

    it('should overwrite duplicate proposal names', () => {
      wia.assessProposal(testProposal);

      testProposal.workforce_change.jobs_eliminated = 20;
      wia.assessProposal(testProposal);

      const history = wia.getHistory();
      expect(history.size).toBe(1);
      expect(history.get('Test Automation')?.wia_score).toBeCloseTo(-0.1, 2);
    });
  });

  describe('Statistics', () => {
    beforeEach(() => {
      // Add multiple assessments
      wia.assessProposal(testProposal);

      wia.assessProposal({
        ...testProposal,
        name: 'Proposal 2',
        workforce_change: { ...testProposal.workforce_change, jobs_eliminated: 15 },
      });

      wia.assessProposal({
        ...testProposal,
        name: 'Proposal 3',
        workforce_change: { ...testProposal.workforce_change, jobs_eliminated: 25 },
      });
    });

    it('should track total assessments', () => {
      const stats = wia.getStats();
      expect(stats.total_assessments).toBe(3);
    });

    it('should track approved count', () => {
      const stats = wia.getStats();
      expect(stats.approved).toBeGreaterThan(0);
    });

    it('should track rejected count', () => {
      const stats = wia.getStats();
      expect(stats.rejected).toBeGreaterThan(0);
    });

    it('should track MRH compliant count', () => {
      const stats = wia.getStats();
      expect(stats.mrh_compliant).toBeGreaterThan(0);
    });

    it('should calculate average WIA score', () => {
      const stats = wia.getStats();
      expect(stats.average_wia_score).toBeDefined();
      expect(typeof stats.average_wia_score).toBe('number');
    });

    it('should track risk distribution', () => {
      const stats = wia.getStats();
      expect(stats.risk_distribution).toBeDefined();
      expect(stats.risk_distribution.low).toBeGreaterThanOrEqual(0);
      expect(stats.risk_distribution.medium).toBeGreaterThanOrEqual(0);
      expect(stats.risk_distribution.high).toBeGreaterThanOrEqual(0);
      expect(stats.risk_distribution.critical).toBeGreaterThanOrEqual(0);
    });

    it('should sum risk distribution to total', () => {
      const stats = wia.getStats();
      const sum =
        stats.risk_distribution.low +
        stats.risk_distribution.medium +
        stats.risk_distribution.high +
        stats.risk_distribution.critical;

      expect(sum).toBe(stats.total_assessments);
    });
  });

  describe('Export', () => {
    it('should export audit log as JSON', () => {
      wia.assessProposal(testProposal);

      const exported = wia.exportAuditLog();

      expect(typeof exported).toBe('string');
      expect(() => JSON.parse(exported)).not.toThrow();
    });

    it('should include config in export', () => {
      wia.assessProposal(testProposal);

      const exported = JSON.parse(wia.exportAuditLog());

      expect(exported.config).toBeDefined();
      expect(exported.config.mrh_threshold).toBe(-0.1);
    });

    it('should include assessments in export', () => {
      wia.assessProposal(testProposal);

      const exported = JSON.parse(wia.exportAuditLog());

      expect(exported.assessments).toBeDefined();
      expect(Array.isArray(exported.assessments)).toBe(true);
    });

    it('should include audit log in export', () => {
      wia.assessProposal(testProposal);

      const exported = JSON.parse(wia.exportAuditLog());

      expect(exported.audit_log).toBeDefined();
      expect(Array.isArray(exported.audit_log)).toBe(true);
    });

    it('should include export timestamp', () => {
      const exported = JSON.parse(wia.exportAuditLog());

      expect(exported.exported_at).toBeDefined();
      expect(typeof exported.exported_at).toBe('number');
    });
  });

  describe('Clear', () => {
    beforeEach(() => {
      wia.assessProposal(testProposal);
      wia.assessProposal({ ...testProposal, name: 'Proposal 2' });
    });

    it('should clear assessment history', () => {
      wia.clear();

      const history = wia.getHistory();
      expect(history.size).toBe(0);
    });

    it('should clear audit log', () => {
      wia.clear();

      const auditLog = wia.getAuditLog();
      expect(auditLog.length).toBe(0);
    });

    it('should reset statistics', () => {
      wia.clear();

      const stats = wia.getStats();
      expect(stats.total_assessments).toBe(0);
    });
  });

  describe('Factory Function', () => {
    it('should create WIA instance', () => {
      const instance = createWIA();

      expect(instance).toBeInstanceOf(WorkforceImpactAssessor);
    });

    it('should accept custom config', () => {
      const instance = createWIA({ mrh_threshold: -0.2 });

      expect(instance).toBeInstanceOf(WorkforceImpactAssessor);
    });
  });

  describe('Utility Functions', () => {
    describe('calculateWIAScore', () => {
      it('should calculate positive score', () => {
        const score = calculateWIAScore(10, 5, 100);
        expect(score).toBeCloseTo(0.05, 2);
      });

      it('should calculate negative score', () => {
        const score = calculateWIAScore(5, 20, 100);
        expect(score).toBeCloseTo(-0.15, 2);
      });

      it('should handle zero affected', () => {
        const score = calculateWIAScore(10, 5, 0);
        expect(score).toBe(0);
      });
    });

    describe('checkMRHCompliance', () => {
      it('should check compliance with default threshold', () => {
        expect(checkMRHCompliance(0.05)).toBe(true);
        expect(checkMRHCompliance(-0.09)).toBe(true);
        expect(checkMRHCompliance(-0.1)).toBe(true);
        expect(checkMRHCompliance(-0.11)).toBe(false);
      });

      it('should check compliance with custom threshold', () => {
        expect(checkMRHCompliance(-0.15, -0.2)).toBe(true);
        expect(checkMRHCompliance(-0.25, -0.2)).toBe(false);
      });
    });
  });

  describe('Integration', () => {
    it('should handle complete workflow', () => {
      // Assess proposal
      const assessment = wia.assessProposal(testProposal);

      // Verify assessment
      expect(assessment.wia_score).toBeDefined();
      expect(assessment.mrh_compliant).toBe(true);
      expect(assessment.approved).toBe(true);

      // Check history
      const history = wia.getHistory();
      expect(history.size).toBe(1);

      // Check audit log
      const auditLog = wia.getAuditLog();
      expect(auditLog.length).toBe(1);

      // Get stats
      const stats = wia.getStats();
      expect(stats.total_assessments).toBe(1);

      // Export
      const exported = wia.exportAuditLog();
      expect(exported).toBeDefined();

      // Clear
      wia.clear();
      expect(wia.getHistory().size).toBe(0);
    });

    it('should handle multiple proposals', () => {
      const proposals = [
        testProposal,
        { ...testProposal, name: 'P2', workforce_change: { ...testProposal.workforce_change, jobs_eliminated: 15 } },
        { ...testProposal, name: 'P3', workforce_change: { ...testProposal.workforce_change, jobs_eliminated: 25 } },
      ];

      proposals.forEach((p) => wia.assessProposal(p));

      const stats = wia.getStats();
      expect(stats.total_assessments).toBe(3);
      expect(stats.average_wia_score).toBeDefined();
    });
  });
});
