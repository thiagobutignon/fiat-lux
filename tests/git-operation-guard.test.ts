/**
 * Git Operation Guard Integration Tests
 *
 * Tests the integration between VERMELHO (Security) and VERDE (VCS)
 * Covers commit validation, mutation validation, duress snapshots, and audit trail
 */

import {
  describe,
  it,
  expect,
  beforeEach,
  afterEach,
  runTests,
} from '../src/shared/utils/test-runner';
import * as fs from 'fs';
import * as path from 'path';
import {
  GitOperationGuard,
  createCommitRequest,
  createMutationRequest,
  shouldProceedWithGitOperation,
  getGitValidationSummary,
  generateSecurityMetadata,
} from '../src/grammar-lang/security/git-operation-guard';
import { SecurityStorage } from '../src/grammar-lang/security/security-storage';
import { LinguisticCollector } from '../src/grammar-lang/security/linguistic-collector';
import { TypingCollector } from '../src/grammar-lang/security/typing-collector';
import { EmotionalCollector } from '../src/grammar-lang/security/emotional-collector';
import { TemporalCollector } from '../src/grammar-lang/security/temporal-collector';
import { UserSecurityProfiles, Interaction } from '../src/grammar-lang/security/types';

// Test directories
const TEST_STORAGE_DIR = 'test_sqlo_git_security';
const TEST_SNAPSHOT_DIR = '.git/test-duress-snapshots';

// Global instances
let storage: SecurityStorage;
let gitGuard: GitOperationGuard;
let normalProfiles: UserSecurityProfiles;
let coercionProfiles: UserSecurityProfiles;

// Helper function
function createTestProfiles(userId: string): UserSecurityProfiles {
  return {
    user_id: userId,
    linguistic: LinguisticCollector.createProfile(userId),
    typing: TypingCollector.createProfile(userId),
    emotional: EmotionalCollector.createProfile(userId),
    temporal: TemporalCollector.createProfile(userId, 'UTC'),
    overall_confidence: 0.5,
    last_interaction: Date.now(),
  };
}

// Helper to build normal baseline
function buildNormalBaseline(userId: string): UserSecurityProfiles {
  let linguistic = LinguisticCollector.createProfile(userId);
  let typing = TypingCollector.createProfile(userId);
  let emotional = EmotionalCollector.createProfile(userId);
  let temporal = TemporalCollector.createProfile(userId, 'UTC');

  for (let i = 0; i < 30; i++) {
    const text = 'Committing changes to the project. Making good progress.';
    const interaction: Interaction = {
      interaction_id: `baseline_${i}`,
      user_id: userId,
      timestamp: Date.now() - 1000 * i,
      text,
      text_length: text.length,
      word_count: text.split(/\s+/).length,
      session_id: 'session_baseline',
      typing_data: {
        keystroke_intervals: Array(text.length)
          .fill(0)
          .map(() => 100 + Math.random() * 20),
        total_typing_time: text.length * 110,
        pauses: [300, 250],
        backspaces: 0,
        corrections: 0,
      },
    };

    linguistic = LinguisticCollector.analyzeAndUpdate(linguistic, interaction);
    typing = TypingCollector.analyzeAndUpdate(typing, interaction);
    emotional = EmotionalCollector.analyzeAndUpdate(emotional, interaction);
    temporal = TemporalCollector.analyzeAndUpdate(temporal, interaction, 30);
  }

  return {
    user_id: userId,
    linguistic,
    typing,
    emotional,
    temporal,
    overall_confidence: 0.5,
    last_interaction: Date.now(),
  };
}

// Helper to build coercion profiles
function buildCoercionProfiles(baseProfiles: UserSecurityProfiles): UserSecurityProfiles {
  const coercionInteraction: Interaction = {
    interaction_id: 'coercion_test',
    user_id: baseProfiles.user_id,
    timestamp: Date.now(),
    text: 'I must commit this now. They are forcing me to do it.',
    text_length: 50,
    word_count: 10,
    session_id: 'session_coercion',
    typing_data: {
      keystroke_intervals: Array(50)
        .fill(0)
        .map(() => 45 + Math.random() * 10),
      total_typing_time: 50 * 50,
      pauses: [120, 140],
      backspaces: 8,
      corrections: 6,
    },
  };

  return {
    user_id: baseProfiles.user_id,
    linguistic: LinguisticCollector.analyzeAndUpdate(
      baseProfiles.linguistic,
      coercionInteraction
    ),
    typing: TypingCollector.analyzeAndUpdate(baseProfiles.typing, coercionInteraction),
    emotional: EmotionalCollector.analyzeAndUpdate(
      baseProfiles.emotional,
      coercionInteraction
    ),
    temporal: TemporalCollector.analyzeAndUpdate(
      baseProfiles.temporal,
      coercionInteraction,
      30
    ),
    overall_confidence: 0.3,
    last_interaction: Date.now(),
  };
}

// ===========================================================================
// COMMIT VALIDATION TESTS
// ===========================================================================

describe('Commit Validation', () => {
  beforeEach(() => {
    storage = new SecurityStorage(TEST_STORAGE_DIR);
    gitGuard = new GitOperationGuard(storage, TEST_SNAPSHOT_DIR);
    normalProfiles = buildNormalBaseline('alice');
    coercionProfiles = buildCoercionProfiles(normalProfiles);
  });

  afterEach(() => {
    if (fs.existsSync(TEST_STORAGE_DIR)) {
      fs.rmSync(TEST_STORAGE_DIR, { recursive: true, force: true });
    }
    if (fs.existsSync(TEST_SNAPSHOT_DIR)) {
      fs.rmSync(TEST_SNAPSHOT_DIR, { recursive: true, force: true });
    }
  });

  it('should allow normal commits', () => {
    const request = createCommitRequest(
      'alice',
      'test.glass',
      'feat: add new feature',
      'human',
      { lines_added: 10, lines_removed: 0 }
    );

    const result = gitGuard.validateCommitRequest(request, normalProfiles);

    expect.toEqual(result.decision, 'allow');
    expect.toBeTruthy(result.allowed);
    expect.toEqual(result.security_context.is_sensitive_git_operation, false);
  });

  it('should challenge sensitive commits with normal behavior', () => {
    const request = createCommitRequest(
      'alice',
      'test.glass',
      'refactor: force delete implementation',
      'human',
      { lines_added: 0, lines_removed: 150 }
    );

    const result = gitGuard.validateCommitRequest(request, normalProfiles);

    expect.toEqual(result.decision, 'challenge');
    expect.toEqual(result.security_context.is_sensitive_git_operation, true);
    expect.toBeTruthy(result.requires_cognitive_challenge);
  });

  it('should block commits under coercion', () => {
    const request = createCommitRequest(
      'alice',
      'test.glass',
      'feat: add feature',
      'human',
      { lines_added: 10, lines_removed: 0 }
    );

    const result = gitGuard.validateCommitRequest(request, coercionProfiles);

    expect.toEqual(result.decision, 'block');
    expect.toEqual(result.allowed, false);
  });

  it('should block sensitive commits under coercion', () => {
    const request = createCommitRequest(
      'alice',
      'test.glass',
      'refactor: force-push delete everything',
      'human',
      { lines_added: 0, lines_removed: 200 }
    );

    const result = gitGuard.validateCommitRequest(request, coercionProfiles);

    expect.toEqual(result.decision, 'block');
    expect.toEqual(result.allowed, false);
    expect.toEqual(result.security_context.is_sensitive_git_operation, true);
  });
});

// ===========================================================================
// MUTATION VALIDATION TESTS
// ===========================================================================

describe('Mutation Validation', () => {
  beforeEach(() => {
    storage = new SecurityStorage(TEST_STORAGE_DIR);
    gitGuard = new GitOperationGuard(storage, TEST_SNAPSHOT_DIR);
    normalProfiles = buildNormalBaseline('alice');
    coercionProfiles = buildCoercionProfiles(normalProfiles);
  });

  afterEach(() => {
    if (fs.existsSync(TEST_STORAGE_DIR)) {
      fs.rmSync(TEST_STORAGE_DIR, { recursive: true, force: true });
    }
    if (fs.existsSync(TEST_SNAPSHOT_DIR)) {
      fs.rmSync(TEST_SNAPSHOT_DIR, { recursive: true, force: true });
    }
  });

  it('should allow normal mutations', () => {
    const request = createMutationRequest('alice', 'test-1.0.0.glass', 'agi', '1.0.0', '1.0.1');

    const result = gitGuard.validateMutationRequest(request, normalProfiles);

    expect.toEqual(result.decision, 'allow');
    expect.toBeTruthy(result.allowed);
  });

  it('should block mutations under coercion', () => {
    const request = createMutationRequest('alice', 'test-1.0.0.glass', 'human', '1.0.0', '2.0.0');

    const result = gitGuard.validateMutationRequest(request, coercionProfiles);

    expect.toEqual(result.decision, 'block');
    expect.toEqual(result.allowed, false);
  });
});

// ===========================================================================
// SENSITIVE OPERATION DETECTION TESTS
// ===========================================================================

describe('Sensitive Operation Detection', () => {
  beforeEach(() => {
    storage = new SecurityStorage(TEST_STORAGE_DIR);
    gitGuard = new GitOperationGuard(storage, TEST_SNAPSHOT_DIR);
    normalProfiles = buildNormalBaseline('alice');
  });

  afterEach(() => {
    if (fs.existsSync(TEST_STORAGE_DIR)) {
      fs.rmSync(TEST_STORAGE_DIR, { recursive: true, force: true });
    }
    if (fs.existsSync(TEST_SNAPSHOT_DIR)) {
      fs.rmSync(TEST_SNAPSHOT_DIR, { recursive: true, force: true });
    }
  });

  it('should detect force-push as sensitive', () => {
    const request = createCommitRequest(
      'alice',
      'test.glass',
      'feat: force-push changes',
      'human'
    );

    const result = gitGuard.validateCommitRequest(request, normalProfiles);

    expect.toEqual(result.security_context.is_sensitive_git_operation, true);
    expect.toBeTruthy(
      result.security_context.sensitive_keywords.some((k) => k.includes('force'))
    );
  });

  it('should detect delete operations as sensitive', () => {
    const request = createCommitRequest('alice', 'test.glass', 'refactor: delete old code', 'human');

    const result = gitGuard.validateCommitRequest(request, normalProfiles);

    expect.toEqual(result.security_context.is_sensitive_git_operation, true);
    expect.toBeTruthy(result.security_context.sensitive_keywords.includes('delete'));
  });

  it('should detect reset operations as sensitive', () => {
    const request = createCommitRequest(
      'alice',
      'test.glass',
      'refactor: git reset --hard',
      'human'
    );

    const result = gitGuard.validateCommitRequest(request, normalProfiles);

    expect.toEqual(result.security_context.is_sensitive_git_operation, true);
    expect.toBeTruthy(result.security_context.sensitive_keywords.includes('reset'));
  });

  it('should detect large deletions as sensitive', () => {
    const request = createCommitRequest(
      'alice',
      'test.glass',
      'refactor: cleanup',
      'human',
      { lines_added: 5, lines_removed: 150 }
    );

    const result = gitGuard.validateCommitRequest(request, normalProfiles);

    expect.toEqual(result.security_context.is_sensitive_git_operation, true);
    expect.toBeTruthy(result.security_context.sensitive_keywords.includes('large-deletion'));
  });

  it('should not flag normal commits as sensitive', () => {
    const request = createCommitRequest(
      'alice',
      'test.glass',
      'feat: add new function',
      'human',
      { lines_added: 20, lines_removed: 2 }
    );

    const result = gitGuard.validateCommitRequest(request, normalProfiles);

    expect.toEqual(result.security_context.is_sensitive_git_operation, false);
    expect.toEqual(result.security_context.sensitive_keywords.length, 0);
  });
});

// ===========================================================================
// DURESS SNAPSHOT TESTS
// ===========================================================================

describe('Duress Snapshot System', () => {
  beforeEach(() => {
    storage = new SecurityStorage(TEST_STORAGE_DIR);
    gitGuard = new GitOperationGuard(storage, TEST_SNAPSHOT_DIR);
    normalProfiles = buildNormalBaseline('alice');
    coercionProfiles = buildCoercionProfiles(normalProfiles);

    // Create test file
    const testDir = 'test_git_files';
    if (!fs.existsSync(testDir)) {
      fs.mkdirSync(testDir, { recursive: true });
    }
    fs.writeFileSync(
      path.join(testDir, 'test.glass'),
      '(define test-function () "test content")'
    );
  });

  afterEach(() => {
    if (fs.existsSync(TEST_STORAGE_DIR)) {
      fs.rmSync(TEST_STORAGE_DIR, { recursive: true, force: true });
    }
    if (fs.existsSync(TEST_SNAPSHOT_DIR)) {
      fs.rmSync(TEST_SNAPSHOT_DIR, { recursive: true, force: true });
    }
    if (fs.existsSync('test_git_files')) {
      fs.rmSync('test_git_files', { recursive: true, force: true });
    }
  });

  it('should create snapshot for sensitive operations under duress', () => {
    const request = createCommitRequest(
      'alice',
      'test_git_files/test.glass',
      'refactor: force delete everything',
      'human',
      { lines_added: 0, lines_removed: 200 }
    );

    const result = gitGuard.validateCommitRequest(request, coercionProfiles);

    expect.toBeTruthy(result.snapshot_created);
    expect.toBeTruthy(result.snapshot_path);
  });

  it('should not create snapshot for normal operations', () => {
    const request = createCommitRequest(
      'alice',
      'test_git_files/test.glass',
      'feat: add feature',
      'human',
      { lines_added: 10, lines_removed: 0 }
    );

    const result = gitGuard.validateCommitRequest(request, normalProfiles);

    expect.toEqual(result.snapshot_created, false);
  });

  it('should list all duress snapshots', () => {
    // Create a snapshot by making sensitive operation under coercion
    const request = createCommitRequest(
      'alice',
      'test_git_files/test.glass',
      'refactor: force delete',
      'human',
      { lines_added: 0, lines_removed: 200 }
    );

    gitGuard.validateCommitRequest(request, coercionProfiles);

    const snapshots = gitGuard.listDuressSnapshots();

    expect.toBeGreaterThan(snapshots.length, 0);
    expect.toEqual(snapshots[0].user_id, 'alice');
    expect.toBeTruthy(snapshots[0].snapshot_id);
  });

  it('should restore from duress snapshot', () => {
    // Create a snapshot
    const request = createCommitRequest(
      'alice',
      'test_git_files/test.glass',
      'refactor: force delete',
      'human',
      { lines_added: 0, lines_removed: 200 }
    );

    const result = gitGuard.validateCommitRequest(request, coercionProfiles);

    if (result.snapshot_created && result.snapshot_path) {
      const snapshotId = path.basename(result.snapshot_path);

      // Modify the original file
      fs.writeFileSync('test_git_files/test.glass', '(define modified () "changed")');

      // Restore from snapshot
      const restoreResult = gitGuard.restoreFromSnapshot(snapshotId);

      expect.toBeTruthy(restoreResult.success);
      expect.toEqual(restoreResult.file_path, 'test_git_files/test.glass');

      // Verify content restored
      const restoredContent = fs.readFileSync('test_git_files/test.glass', 'utf-8');
      expect.toEqual(restoredContent, '(define test-function () "test content")');
    }
  });
});

// ===========================================================================
// SECURITY METADATA TESTS
// ===========================================================================

describe('Security Metadata', () => {
  beforeEach(() => {
    storage = new SecurityStorage(TEST_STORAGE_DIR);
    gitGuard = new GitOperationGuard(storage, TEST_SNAPSHOT_DIR);
    normalProfiles = buildNormalBaseline('alice');
  });

  afterEach(() => {
    if (fs.existsSync(TEST_STORAGE_DIR)) {
      fs.rmSync(TEST_STORAGE_DIR, { recursive: true, force: true });
    }
    if (fs.existsSync(TEST_SNAPSHOT_DIR)) {
      fs.rmSync(TEST_SNAPSHOT_DIR, { recursive: true, force: true });
    }
  });

  it('should generate security metadata', () => {
    const request = createCommitRequest(
      'alice',
      'test.glass',
      'feat: add feature',
      'human',
      { lines_added: 10, lines_removed: 0 }
    );

    const result = gitGuard.validateCommitRequest(request, normalProfiles);
    const metadata = generateSecurityMetadata(result.security_context);

    expect.toBeTruthy(metadata.includes('X-Security-Validated: true'));
    expect.toBeTruthy(metadata.includes('X-Duress-Score:'));
    expect.toBeTruthy(metadata.includes('X-Coercion-Score:'));
    expect.toBeTruthy(metadata.includes('X-Confidence:'));
    expect.toBeTruthy(metadata.includes('X-Author:'));
  });
});

// ===========================================================================
// AUDIT TRAIL TESTS
// ===========================================================================

describe('Audit Trail', () => {
  beforeEach(() => {
    storage = new SecurityStorage(TEST_STORAGE_DIR);
    gitGuard = new GitOperationGuard(storage, TEST_SNAPSHOT_DIR);
    normalProfiles = buildNormalBaseline('alice');
    coercionProfiles = buildCoercionProfiles(normalProfiles);
  });

  afterEach(() => {
    if (fs.existsSync(TEST_STORAGE_DIR)) {
      fs.rmSync(TEST_STORAGE_DIR, { recursive: true, force: true });
    }
    if (fs.existsSync(TEST_SNAPSHOT_DIR)) {
      fs.rmSync(TEST_SNAPSHOT_DIR, { recursive: true, force: true });
    }
  });

  it('should log Git operations to audit trail', () => {
    const request = createCommitRequest('alice', 'test.glass', 'feat: add feature', 'human');

    gitGuard.validateCommitRequest(request, normalProfiles);

    const events = storage.getUserEvents('alice', 10);
    const gitEvents = events.filter((e) => e.operation_type === 'git_operation');

    expect.toBeGreaterThan(gitEvents.length, 0);
  });

  it('should track Git operation statistics', () => {
    // Normal commit
    const normalRequest = createCommitRequest('alice', 'test.glass', 'feat: add', 'human');
    gitGuard.validateCommitRequest(normalRequest, normalProfiles);

    // Blocked commit
    const blockedRequest = createCommitRequest('alice', 'test.glass', 'feat: add', 'human');
    gitGuard.validateCommitRequest(blockedRequest, coercionProfiles);

    const stats = gitGuard.getGitStatistics('alice', 24);

    expect.toEqual(stats.total_git_operations, 2);
    expect.toBeGreaterThan(stats.blocked_operations, 0);
  });
});

// Run tests
runTests();
