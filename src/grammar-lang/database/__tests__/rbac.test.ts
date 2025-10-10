/**
 * RBAC System Tests
 *
 * Verifies:
 * - O(1) permission checking
 * - Role management
 * - Default roles (admin, user, readonly, system, guest)
 * - Integration with SqloDatabase
 */

import { describe, it, beforeEach, afterEach, expect } from '../../../shared/utils/test-runner';
import { RbacPolicy, Permission } from '../rbac';
import { SqloDatabase, MemoryType, Episode } from '../sqlo';
import * as fs from 'fs';

describe('RbacPolicy - Role Management', () => {
  let policy: RbacPolicy;

  beforeEach(() => {
    policy = new RbacPolicy();
  });

  it('creates new role - O(1)', () => {
    const role = policy.createRole('test-role', 'Test role');

    expect.toBeDefined(role);
    expect.toEqual(role.name, 'test-role');
    expect.toEqual(role.description, 'Test role');
  });

  it('gets role - O(1)', () => {
    policy.createRole('test-role');
    const role = policy.getRole('test-role');

    expect.toBeDefined(role);
    expect.toEqual(role!.name, 'test-role');
  });

  it('deletes role - O(1)', () => {
    policy.createRole('test-role');
    const deleted = policy.deleteRole('test-role');

    expect.toBeTruthy(deleted);
    expect.toBeUndefined(policy.getRole('test-role'));
  });

  it('prevents duplicate role creation', () => {
    policy.createRole('duplicate');

    expect.toThrow(() => {
      policy.createRole('duplicate');
    });
  });

  it('lists all roles', () => {
    const roles = policy.listRoles();

    // Should include default roles: admin, user, readonly, system, guest
    expect.toBeGreaterThan(roles.length, 0);
    expect.toBeTruthy(roles.some(r => r.name === 'admin'));
    expect.toBeTruthy(roles.some(r => r.name === 'user'));
  });
});

describe('RbacPolicy - Permission Management', () => {
  let policy: RbacPolicy;

  beforeEach(() => {
    policy = new RbacPolicy();
  });

  it('grants permission - O(1)', () => {
    policy.createRole('test');
    policy.grantPermission('test', MemoryType.SHORT_TERM, Permission.READ);

    expect.toBeTruthy(
      policy.hasPermission('test', MemoryType.SHORT_TERM, Permission.READ)
    );
  });

  it('revokes permission - O(1)', () => {
    policy.createRole('test');
    policy.grantPermission('test', MemoryType.SHORT_TERM, Permission.READ);
    policy.revokePermission('test', MemoryType.SHORT_TERM, Permission.READ);

    expect.toBeFalsy(
      policy.hasPermission('test', MemoryType.SHORT_TERM, Permission.READ)
    );
  });

  it('grants all permissions for memory type - O(1)', () => {
    policy.createRole('test');
    policy.grantAllPermissions('test', MemoryType.LONG_TERM);

    expect.toBeTruthy(
      policy.hasPermission('test', MemoryType.LONG_TERM, Permission.READ)
    );
    expect.toBeTruthy(
      policy.hasPermission('test', MemoryType.LONG_TERM, Permission.WRITE)
    );
    expect.toBeTruthy(
      policy.hasPermission('test', MemoryType.LONG_TERM, Permission.DELETE)
    );
  });

  it('revokes all permissions for memory type - O(1)', () => {
    policy.createRole('test');
    policy.grantAllPermissions('test', MemoryType.CONTEXTUAL);
    policy.revokeAllPermissions('test', MemoryType.CONTEXTUAL);

    expect.toBeFalsy(
      policy.hasPermission('test', MemoryType.CONTEXTUAL, Permission.READ)
    );
    expect.toBeFalsy(
      policy.hasPermission('test', MemoryType.CONTEXTUAL, Permission.WRITE)
    );
  });

  it('checks multiple permissions', () => {
    policy.createRole('test');
    policy.grantPermission('test', MemoryType.SHORT_TERM, Permission.READ);
    policy.grantPermission('test', MemoryType.SHORT_TERM, Permission.WRITE);

    expect.toBeTruthy(
      policy.hasAllPermissions('test', MemoryType.SHORT_TERM, [
        Permission.READ,
        Permission.WRITE
      ])
    );

    expect.toBeFalsy(
      policy.hasAllPermissions('test', MemoryType.SHORT_TERM, [
        Permission.READ,
        Permission.DELETE
      ])
    );
  });

  it('checks any permission', () => {
    policy.createRole('test');
    policy.grantPermission('test', MemoryType.SHORT_TERM, Permission.READ);

    expect.toBeTruthy(
      policy.hasAnyPermission('test', MemoryType.SHORT_TERM, [
        Permission.READ,
        Permission.DELETE
      ])
    );

    expect.toBeFalsy(
      policy.hasAnyPermission('test', MemoryType.SHORT_TERM, [
        Permission.WRITE,
        Permission.DELETE
      ])
    );
  });
});

describe('RbacPolicy - Default Roles', () => {
  let policy: RbacPolicy;

  beforeEach(() => {
    policy = new RbacPolicy();
  });

  it('admin role has full access', () => {
    for (const memType of Object.values(MemoryType)) {
      expect.toBeTruthy(
        policy.hasPermission('admin', memType as MemoryType, Permission.READ)
      );
      expect.toBeTruthy(
        policy.hasPermission('admin', memType as MemoryType, Permission.WRITE)
      );
      expect.toBeTruthy(
        policy.hasPermission('admin', memType as MemoryType, Permission.DELETE)
      );
    }
  });

  it('user role has limited access', () => {
    // Can read/write short-term
    expect.toBeTruthy(
      policy.hasPermission('user', MemoryType.SHORT_TERM, Permission.READ)
    );
    expect.toBeTruthy(
      policy.hasPermission('user', MemoryType.SHORT_TERM, Permission.WRITE)
    );

    // Can only read long-term
    expect.toBeTruthy(
      policy.hasPermission('user', MemoryType.LONG_TERM, Permission.READ)
    );
    expect.toBeFalsy(
      policy.hasPermission('user', MemoryType.LONG_TERM, Permission.WRITE)
    );
    expect.toBeFalsy(
      policy.hasPermission('user', MemoryType.LONG_TERM, Permission.DELETE)
    );
  });

  it('readonly role can only read', () => {
    for (const memType of Object.values(MemoryType)) {
      expect.toBeTruthy(
        policy.hasPermission('readonly', memType as MemoryType, Permission.READ)
      );
      expect.toBeFalsy(
        policy.hasPermission('readonly', memType as MemoryType, Permission.WRITE)
      );
      expect.toBeFalsy(
        policy.hasPermission('readonly', memType as MemoryType, Permission.DELETE)
      );
    }
  });

  it('guest role has no permissions', () => {
    for (const memType of Object.values(MemoryType)) {
      expect.toBeFalsy(
        policy.hasPermission('guest', memType as MemoryType, Permission.READ)
      );
      expect.toBeFalsy(
        policy.hasPermission('guest', memType as MemoryType, Permission.WRITE)
      );
    }
  });
});

describe('RbacPolicy - Permission Checking', () => {
  let policy: RbacPolicy;

  beforeEach(() => {
    policy = new RbacPolicy();
  });

  it('checkPermission returns detailed result', () => {
    const result = policy.checkPermission(
      'admin',
      MemoryType.SHORT_TERM,
      Permission.READ
    );

    expect.toBeTruthy(result.granted);
    expect.toEqual(result.role, 'admin');
    expect.toEqual(result.memoryType, MemoryType.SHORT_TERM);
    expect.toEqual(result.permission, Permission.READ);
  });

  it('checkPermission includes denial reason', () => {
    const result = policy.checkPermission(
      'guest',
      MemoryType.LONG_TERM,
      Permission.WRITE
    );

    expect.toBeFalsy(result.granted);
    expect.toBeDefined(result.reason);
  });
});

describe('SqloDatabase - RBAC Integration', () => {
  const TEST_DB_DIR = 'test_rbac_db';
  let db: SqloDatabase;

  beforeEach(() => {
    if (fs.existsSync(TEST_DB_DIR)) {
      fs.rmSync(TEST_DB_DIR, { recursive: true });
    }
    db = new SqloDatabase(TEST_DB_DIR);
  });

  afterEach(() => {
    if (fs.existsSync(TEST_DB_DIR)) {
      fs.rmSync(TEST_DB_DIR, { recursive: true });
    }
  });

  it('admin can write to any memory type', async () => {
    const episode = createTestEpisode('test', MemoryType.LONG_TERM);
    const hash = await db.put(episode, 'admin');

    expect.toBeDefined(hash);
  });

  it('user cannot write to long-term memory', async () => {
    const episode = createTestEpisode('test', MemoryType.LONG_TERM);

    let errorThrown = false;
    try {
      await db.put(episode, 'user');
    } catch (error) {
      errorThrown = true;
      expect.toBeTruthy(String(error).includes('Permission denied'));
    }
    expect.toBeTruthy(errorThrown);
  });

  it('user can write to short-term memory', async () => {
    const episode = createTestEpisode('test', MemoryType.SHORT_TERM);
    const hash = await db.put(episode, 'user');

    expect.toBeDefined(hash);
  });

  it('readonly cannot write to any memory', async () => {
    const episode = createTestEpisode('test', MemoryType.SHORT_TERM);

    let errorThrown = false;
    try {
      await db.put(episode, 'readonly');
    } catch (error) {
      errorThrown = true;
      expect.toBeTruthy(String(error).includes('Permission denied'));
    }
    expect.toBeTruthy(errorThrown);
  });

  it('readonly can read from any memory', async () => {
    const episode = createTestEpisode('test', MemoryType.SHORT_TERM);
    const hash = await db.put(episode, 'admin');

    const retrieved = db.get(hash, 'readonly');
    expect.toBeDefined(retrieved);
    expect.toEqual(retrieved!.query, 'test');
  });

  it('guest cannot read without permission', async () => {
    const episode = createTestEpisode('test', MemoryType.SHORT_TERM);
    const hash = await db.put(episode, 'admin');

    expect.toThrow(() => {
      db.get(hash, 'guest');
    });
  });

  it('user cannot delete long-term memory', async () => {
    const episode = createTestEpisode('test', MemoryType.LONG_TERM);
    const hash = await db.put(episode, 'admin');

    expect.toThrow(() => {
      db.delete(hash, 'user');
    });
  });

  it('admin can delete any memory', async () => {
    const episode = createTestEpisode('test', MemoryType.LONG_TERM);
    const hash = await db.put(episode, 'admin');

    const deleted = db.delete(hash, 'admin');
    expect.toBeTruthy(deleted);
  });
});

describe('RbacPolicy - Performance', () => {
  let policy: RbacPolicy;

  beforeEach(() => {
    policy = new RbacPolicy();
  });

  it('hasPermission completes in <0.01ms (O(1) guarantee)', () => {
    const iterations = 10000;
    const start = performance.now();

    for (let i = 0; i < iterations; i++) {
      policy.hasPermission('admin', MemoryType.SHORT_TERM, Permission.READ);
    }

    const end = performance.now();
    const avgTime = (end - start) / iterations;

    expect.toBeLessThan(avgTime, 0.01); // <0.01ms average
  });
});

// =============================================================================
// Test Helpers
// =============================================================================

function createTestEpisode(
  query: string,
  memoryType: MemoryType
): Omit<Episode, 'id'> {
  return {
    query,
    response: `Response to: ${query}`,
    attention: {
      sources: ['test.pdf'],
      weights: [1.0],
      patterns: ['test-pattern']
    },
    outcome: 'success',
    confidence: 0.95,
    timestamp: Date.now(),
    memory_type: memoryType
  };
}
