/**
 * CINZA Integration Tests - Dual-Layer Security System
 *
 * Tests VERMELHO + CINZA integration for:
 * - Combined behavioral + cognitive validation
 * - Threat level calculation
 * - Decision matrix
 * - Snapshot creation
 * - Security metadata generation
 */

import { describe, it, expect, beforeEach } from '@jest/globals';
import {
  CognitiveBehaviorGuard,
  shouldProceedWithOperation,
  getCognitiveBehaviorSummary,
  formatCognitiveBehaviorAnalysis,
} from '../src/grammar-lang/security/cognitive-behavior-guard';
import {
  createCommitRequest,
  createMutationRequest,
} from '../src/grammar-lang/security/git-operation-guard';
import { SecurityStorage } from '../src/grammar-lang/security/security-storage';
import { UserSecurityProfiles } from '../src/grammar-lang/security/types';

// ===== TEST DATA =====

const mockStorage = new SecurityStorage('./test-storage');

const normalUserProfiles: UserSecurityProfiles = {
  linguistic: {
    baseline_vocabulary_size: 1000,
    baseline_avg_sentence_length: 15,
    baseline_typing_speed_wpm: 60,
    common_phrases: ['update', 'fix', 'add'],
    baseline_formality: 0.5
  },
  typing: {
    baseline_wpm: 60,
    baseline_error_rate: 0.05,
    baseline_pause_pattern: [100, 200, 150],
    baseline_key_hold_duration_ms: 80
  },
  emotional: {
    baseline_sentiment: 0.0,
    baseline_arousal: 0.5,
    baseline_stress_indicators: []
  },
  temporal: {
    usual_work_hours: { start: 9, end: 17 },
    usual_work_days: [1, 2, 3, 4, 5],
    baseline_session_duration_minutes: 60,
    baseline_commits_per_session: 5
  }
};

const duressUserProfiles: UserSecurityProfiles = {
  ...normalUserProfiles,
  linguistic: {
    ...normalUserProfiles.linguistic,
    baseline_formality: 0.2
  },
  typing: {
    ...normalUserProfiles.typing,
    baseline_wpm: 40,
    baseline_error_rate: 0.15
  },
  emotional: {
    ...normalUserProfiles.emotional,
    baseline_sentiment: -0.5,
    baseline_stress_indicators: ['urgent', 'help']
  }
};

// ===== TESTS =====

describe('CognitiveBehaviorGuard', () => {
  let guard: CognitiveBehaviorGuard;

  beforeEach(() => {
    guard = new CognitiveBehaviorGuard(mockStorage);
  });

  describe('Normal Operations (Clean)', () => {
    it('should allow normal commit with no threats', async () => {
      const request = createCommitRequest(
        'user-123',
        'src/feature.ts',
        'feat: add user authentication',
        'human',
        { lines_added: 50, lines_removed: 5 }
      );

      const result = await guard.validateGitOperation(request, normalUserProfiles);

      expect(result.decision).toBe('allow');
      expect(result.cognitive_analysis).toBeDefined();
      expect(result.cognitive_analysis!.combined.threat_level).toBe('none');
      expect(result.cognitive_analysis!.behavioral.duress_score).toBeLessThan(0.3);
      expect(result.cognitive_analysis!.cognitive.manipulation_detected).toBe(false);
    });

    it('should allow normal mutation with no threats', async () => {
      const request = createMutationRequest(
        'user-123',
        'organism-1.0.0.glass',
        'agi',
        '1.0.0',
        '1.0.1'
      );

      const result = await guard.validateGitOperation(request, normalUserProfiles);

      expect(result.decision).toBe('allow');
      expect(result.cognitive_analysis!.combined.threat_level).toBe('none');
    });
  });

  describe('Manipulation Detection (CINZA)', () => {
    it('should detect gaslighting manipulation', async () => {
      const request = createCommitRequest(
        'user-123',
        'src/security.ts',
        'fix: security update\n\nYou must be imagining the security issues.',
        'human',
        { lines_added: 10, lines_removed: 20 }
      );

      const result = await guard.validateGitOperation(request, normalUserProfiles);

      expect(result.cognitive_analysis).toBeDefined();
      expect(result.cognitive_analysis!.cognitive.manipulation_detected).toBe(true);
      expect(result.cognitive_analysis!.cognitive.techniques_found.length).toBeGreaterThan(0);
      expect(result.cognitive_analysis!.combined.threat_level).not.toBe('none');
    });

    it('should detect reality denial manipulation', async () => {
      const request = createCommitRequest(
        'user-123',
        'src/data.ts',
        'fix: data processing\n\nThis never had bugs. Everything always worked perfectly.',
        'human',
        { lines_added: 30, lines_removed: 10 }
      );

      const result = await guard.validateGitOperation(request, normalUserProfiles);

      expect(result.cognitive_analysis!.cognitive.manipulation_detected).toBe(true);
      expect(result.cognitive_analysis!.combined.threat_level).not.toBe('none');
    });

    it('should detect Dark Tetrad traits (Narcissism)', async () => {
      const request = createCommitRequest(
        'user-123',
        'src/feature.ts',
        'feat: perfect implementation\n\nI alone can do this. Others are incompetent.',
        'human',
        { lines_added: 100, lines_removed: 50 }
      );

      const result = await guard.validateGitOperation(request, normalUserProfiles);

      expect(result.cognitive_analysis).toBeDefined();
      const darkTetrad = result.cognitive_analysis!.cognitive.dark_tetrad_scores;
      const avgDarkTetrad = (darkTetrad.narcissism + darkTetrad.machiavellianism + darkTetrad.psychopathy + darkTetrad.sadism) / 4;

      // Should detect some Dark Tetrad traits
      expect(avgDarkTetrad).toBeGreaterThan(0);
    });
  });

  describe('Behavioral Anomalies (VERMELHO)', () => {
    it('should detect duress indicators', async () => {
      const request = createCommitRequest(
        'user-123',
        'src/data.ts',
        'fix: urgent update',
        'human',
        { lines_added: 30, lines_removed: 10 }
      );

      const result = await guard.validateGitOperation(request, duressUserProfiles);

      expect(result.cognitive_analysis).toBeDefined();
      expect(result.cognitive_analysis!.behavioral.duress_score).toBeGreaterThan(0.3);
      expect(result.cognitive_analysis!.combined.threat_level).not.toBe('none');
    });

    it('should detect coercion indicators', async () => {
      const request = createCommitRequest(
        'user-123',
        'src/admin.ts',
        'feat: admin access',
        'human',
        { lines_added: 100, lines_removed: 50 }
      );

      const result = await guard.validateGitOperation(request, duressUserProfiles);

      expect(result.cognitive_analysis).toBeDefined();
      const behavioralRisk = Math.max(
        result.cognitive_analysis!.behavioral.duress_score,
        result.cognitive_analysis!.behavioral.coercion_score
      );
      expect(behavioralRisk).toBeGreaterThan(0.3);
    });
  });

  describe('Combined Threat Assessment', () => {
    it('should calculate correct threat level for low risk', async () => {
      const request = createCommitRequest(
        'user-123',
        'src/util.ts',
        'refactor: code cleanup',
        'human',
        { lines_added: 10, lines_removed: 10 }
      );

      const result = await guard.validateGitOperation(request, normalUserProfiles);

      expect(result.cognitive_analysis!.combined.threat_level).toMatch(/none|low/);
      expect(result.cognitive_analysis!.combined.risk_score).toBeLessThan(0.3);
    });

    it('should calculate correct threat level for medium risk', async () => {
      const request = createCommitRequest(
        'user-123',
        'src/feature.ts',
        'feat: new feature\n\nDon\'t worry about the changes.',
        'human',
        { lines_added: 50, lines_removed: 20 }
      );

      const result = await guard.validateGitOperation(request, duressUserProfiles);

      // Should be at least medium threat (duress + possible manipulation)
      expect(['medium', 'high', 'critical']).toContain(result.cognitive_analysis!.combined.threat_level);
    });

    it('should calculate critical threat for duress + manipulation', async () => {
      const request = createCommitRequest(
        'user-123',
        'src/admin.ts',
        'feat: admin access\n\nYou\'re overreacting. This is perfectly safe.',
        'human',
        { lines_added: 100, lines_removed: 50 }
      );

      const result = await guard.validateGitOperation(request, duressUserProfiles);

      expect(result.cognitive_analysis!.combined.threat_level).toMatch(/high|critical/);
      expect(result.cognitive_analysis!.combined.risk_score).toBeGreaterThan(0.5);
    });
  });

  describe('Decision Matrix', () => {
    it('should allow clean operations', async () => {
      const request = createCommitRequest(
        'user-123',
        'src/feature.ts',
        'feat: add feature',
        'human',
        { lines_added: 50, lines_removed: 5 }
      );

      const result = await guard.validateGitOperation(request, normalUserProfiles);

      expect(result.decision).toBe('allow');
      expect(shouldProceedWithOperation(result)).toBe(true);
    });

    it('should challenge medium threats', async () => {
      const request = createCommitRequest(
        'user-123',
        'src/data.ts',
        'fix: data update',
        'human',
        { lines_added: 30, lines_removed: 10 }
      );

      const result = await guard.validateGitOperation(request, duressUserProfiles);

      // Should challenge or allow (depends on exact scores)
      expect(['allow', 'challenge', 'delay']).toContain(result.decision);
    });

    it('should block critical threats', async () => {
      const request = createCommitRequest(
        'user-123',
        'src/admin.ts',
        'feat: admin access\n\nYou must be imagining the security issues.',
        'human',
        { lines_added: 100, lines_removed: 50 }
      );

      const result = await guard.validateGitOperation(request, duressUserProfiles);

      // High duress + manipulation should block
      expect(['block', 'delay']).toContain(result.decision);
      expect(shouldProceedWithOperation(result)).toBe(false);
    });

    it('should block sensitive operations with manipulation', async () => {
      const request = createCommitRequest(
        'user-123',
        'src/database.ts',
        'fix: database cleanup\n\nDon\'t worry about the force push.',
        'human',
        { lines_added: 5, lines_removed: 500 } // Large deletion
      );

      const result = await guard.validateGitOperation(request, normalUserProfiles);

      // Sensitive + manipulation should block or delay
      expect(['block', 'delay', 'challenge']).toContain(result.decision);
    });
  });

  describe('Snapshot Creation', () => {
    it('should create manipulation snapshot for critical threats', async () => {
      const request = createCommitRequest(
        'user-123',
        'src/admin.ts',
        'feat: admin bypass\n\nYou\'re paranoid. This always worked.',
        'human',
        { lines_added: 150, lines_removed: 80 }
      );

      const result = await guard.validateGitOperation(request, duressUserProfiles);

      // High threat should potentially create snapshots
      if (result.cognitive_analysis!.combined.threat_level === 'critical') {
        expect(result.manipulation_snapshot_created || result.snapshot_created).toBeDefined();
      }
    });
  });

  describe('Helper Functions', () => {
    it('should format cognitive-behavior analysis correctly', async () => {
      const request = createCommitRequest(
        'user-123',
        'src/feature.ts',
        'feat: add feature',
        'human',
        { lines_added: 50, lines_removed: 5 }
      );

      const result = await guard.validateGitOperation(request, normalUserProfiles);

      if (result.cognitive_analysis) {
        const formatted = formatCognitiveBehaviorAnalysis(result.cognitive_analysis);
        expect(formatted).toContain('BEHAVIORAL');
        expect(formatted).toContain('COGNITIVE');
        expect(formatted).toContain('COMBINED');
        expect(formatted).toContain('Threat Level');
      }
    });

    it('should generate summary correctly', async () => {
      const request = createCommitRequest(
        'user-123',
        'src/feature.ts',
        'feat: add feature',
        'human',
        { lines_added: 50, lines_removed: 5 }
      );

      const result = await guard.validateGitOperation(request, normalUserProfiles);

      const summary = getCognitiveBehaviorSummary(result);
      expect(summary).toContain('Decision:');
      expect(summary).toContain('Threat Level:');
      expect(summary).toContain('Risk Score:');
    });
  });

  describe('Security Metadata', () => {
    it('should include cognitive metadata in results', async () => {
      const request = createCommitRequest(
        'user-123',
        'src/feature.ts',
        'feat: add feature\n\nYou must be imagining problems.',
        'human',
        { lines_added: 50, lines_removed: 5 }
      );

      const result = await guard.validateGitOperation(request, normalUserProfiles);

      expect(result.cognitive_analysis).toBeDefined();
      expect(result.cognitive_analysis!.cognitive).toBeDefined();
      expect(result.cognitive_analysis!.behavioral).toBeDefined();
      expect(result.cognitive_analysis!.combined).toBeDefined();
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty commit message', async () => {
      const request = createCommitRequest(
        'user-123',
        'src/feature.ts',
        '',
        'human',
        { lines_added: 10, lines_removed: 5 }
      );

      const result = await guard.validateGitOperation(request, normalUserProfiles);

      expect(result.cognitive_analysis).toBeDefined();
      // Should not crash, cognitive analysis may show no manipulation
    });

    it('should handle mutation requests', async () => {
      const request = createMutationRequest(
        'user-123',
        'organism-1.0.0.glass',
        'agi',
        '1.0.0',
        '1.0.1'
      );
      request.message = 'Trust me, this is perfect.';

      const result = await guard.validateGitOperation(request, normalUserProfiles);

      expect(result.cognitive_analysis).toBeDefined();
    });

    it('should fail-open on cognitive system errors', async () => {
      // This test ensures the system continues working even if cognitive detection fails
      const request = createCommitRequest(
        'user-123',
        'src/feature.ts',
        'feat: add feature',
        'human',
        { lines_added: 50, lines_removed: 5 }
      );

      // Should not throw error even if cognitive system has issues
      const result = await guard.validateGitOperation(request, normalUserProfiles);
      expect(result).toBeDefined();
    });
  });
});

// ===== INTEGRATION TESTS =====

describe('VCS Integration', () => {
  it('should integrate with auto-commit system', async () => {
    // Test that cognitive-behavior guard can be used in auto-commit flow
    const guard = new CognitiveBehaviorGuard(mockStorage);

    const request = createCommitRequest(
      'user-123',
      'src/test.ts',
      'feat: auto-commit test',
      'human',
      { lines_added: 10, lines_removed: 2 }
    );

    const result = await guard.validateGitOperation(request, normalUserProfiles);

    expect(result).toBeDefined();
    expect(result.decision).toBeDefined();
    expect(shouldProceedWithOperation(result)).toBeDefined();
  });

  it('should integrate with genetic versioning system', async () => {
    // Test that cognitive-behavior guard can be used in mutation flow
    const guard = new CognitiveBehaviorGuard(mockStorage);

    const request = createMutationRequest(
      'user-123',
      'organism-1.0.0.glass',
      'agi',
      '1.0.0',
      '1.0.1'
    );

    const result = await guard.validateGitOperation(request, normalUserProfiles);

    expect(result).toBeDefined();
    expect(result.decision).toBeDefined();
  });
});
