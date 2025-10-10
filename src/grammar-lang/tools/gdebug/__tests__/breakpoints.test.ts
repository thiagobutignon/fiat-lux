/**
 * GDebug Breakpoints - Unit Tests
 */

import { BreakpointManager, BreakpointGroup, createBreakpointManager, createGroup } from '../breakpoints';

describe('BreakpointManager', () => {
  let manager: BreakpointManager;

  beforeEach(() => {
    manager = new BreakpointManager();
  });

  describe('add', () => {
    it('should add breakpoint', () => {
      const bp = manager.add('test.ts', 10);

      expect(bp.file).toBe('test.ts');
      expect(bp.line).toBe(10);
      expect(bp.enabled).toBe(true);
      expect(bp.hit_count).toBe(0);
    });

    it('should add breakpoint with column', () => {
      const bp = manager.add('test.ts', 10, { column: 5 });

      expect(bp.column).toBe(5);
      expect(bp.id).toContain(':5');
    });

    it('should add conditional breakpoint', () => {
      const bp = manager.add('test.ts', 10, {
        condition: (locals) => locals.x > 5
      });

      expect(bp.id).toBeDefined();
    });

    it('should add breakpoint with max hits', () => {
      const bp = manager.add('test.ts', 10, { max_hits: 3 });

      expect(bp.max_hits).toBe(3);
    });

    it('should add breakpoint with log message', () => {
      const bp = manager.add('test.ts', 10, {
        log_message: 'Debug: x = {x}'
      });

      expect(bp.log_message).toBe('Debug: x = {x}');
    });

    it('should throw on duplicate breakpoint', () => {
      manager.add('test.ts', 10);

      expect(() => manager.add('test.ts', 10)).toThrow();
    });
  });

  describe('remove', () => {
    it('should remove breakpoint', () => {
      const bp = manager.add('test.ts', 10);
      const removed = manager.remove(bp.id);

      expect(removed).toBe(true);
      expect(manager.get(bp.id)).toBeUndefined();
    });

    it('should return false for non-existent breakpoint', () => {
      expect(manager.remove('nonexistent')).toBe(false);
    });

    it('should remove from location index', () => {
      const bp = manager.add('test.ts', 10);
      manager.remove(bp.id);

      expect(manager.getAtLocation('test.ts', 10)).toHaveLength(0);
    });
  });

  describe('shouldBreak', () => {
    it('should return breakpoint when hit', () => {
      const bp = manager.add('test.ts', 10);
      const result = manager.shouldBreak('test.ts', 10);

      expect(result).not.toBeNull();
      expect(result?.id).toBe(bp.id);
    });

    it('should return null when no breakpoint', () => {
      const result = manager.shouldBreak('test.ts', 10);
      expect(result).toBeNull();
    });

    it('should skip disabled breakpoints', () => {
      const bp = manager.add('test.ts', 10);
      manager.disable(bp.id);

      const result = manager.shouldBreak('test.ts', 10);
      expect(result).toBeNull();
    });

    it('should respect max hits', () => {
      const bp = manager.add('test.ts', 10, { max_hits: 2 });

      // First hit
      manager.shouldBreak('test.ts', 10);
      manager.recordHit(bp.id);

      // Second hit
      manager.shouldBreak('test.ts', 10);
      manager.recordHit(bp.id);

      // Third attempt - should not break
      const result = manager.shouldBreak('test.ts', 10);
      expect(result).toBeNull();
    });

    it('should evaluate conditions', () => {
      manager.add('test.ts', 10, {
        condition: (locals) => locals.x > 10
      });

      // Should not break (x = 5)
      let result = manager.shouldBreak('test.ts', 10, { x: 5 });
      expect(result).toBeNull();

      // Should break (x = 15)
      result = manager.shouldBreak('test.ts', 10, { x: 15 });
      expect(result).not.toBeNull();
    });
  });

  describe('recordHit', () => {
    it('should record hit', () => {
      const bp = manager.add('test.ts', 10);
      const hit = manager.recordHit(bp.id, { x: 5 }, 1);

      expect(hit.breakpoint_id).toBe(bp.id);
      expect(hit.file).toBe('test.ts');
      expect(hit.line).toBe(10);
      expect(hit.locals).toEqual({ x: 5 });
      expect(hit.stack_depth).toBe(1);
    });

    it('should increment hit count', () => {
      const bp = manager.add('test.ts', 10);

      manager.recordHit(bp.id);
      manager.recordHit(bp.id);

      const updated = manager.get(bp.id);
      expect(updated?.hit_count).toBe(2);
    });

    it('should throw for non-existent breakpoint', () => {
      expect(() => manager.recordHit('nonexistent')).toThrow();
    });

    it('should maintain hit history', () => {
      const bp = manager.add('test.ts', 10);

      manager.recordHit(bp.id);
      manager.recordHit(bp.id);

      const history = manager.getHitHistory(bp.id);
      expect(history).toHaveLength(2);
    });
  });

  describe('enable/disable', () => {
    it('should enable breakpoint', () => {
      const bp = manager.add('test.ts', 10);
      manager.disable(bp.id);
      manager.enable(bp.id);

      const updated = manager.get(bp.id);
      expect(updated?.enabled).toBe(true);
    });

    it('should disable breakpoint', () => {
      const bp = manager.add('test.ts', 10);
      manager.disable(bp.id);

      const updated = manager.get(bp.id);
      expect(updated?.enabled).toBe(false);
    });

    it('should return false for non-existent breakpoint', () => {
      expect(manager.enable('nonexistent')).toBe(false);
      expect(manager.disable('nonexistent')).toBe(false);
    });
  });

  describe('getAtLocation', () => {
    it('should return all breakpoints at location', () => {
      manager.add('test.ts', 10);
      manager.add('test.ts', 10, { column: 5 });

      const breakpoints = manager.getAtLocation('test.ts', 10);
      expect(breakpoints).toHaveLength(2);
    });

    it('should return empty array for no breakpoints', () => {
      expect(manager.getAtLocation('test.ts', 10)).toHaveLength(0);
    });
  });

  describe('getAll', () => {
    it('should return all breakpoints', () => {
      manager.add('test.ts', 10);
      manager.add('test.ts', 20);
      manager.add('other.ts', 5);

      expect(manager.getAll()).toHaveLength(3);
    });
  });

  describe('clearAll', () => {
    it('should clear all breakpoints', () => {
      manager.add('test.ts', 10);
      manager.add('test.ts', 20);

      manager.clearAll();

      expect(manager.getAll()).toHaveLength(0);
    });
  });

  describe('clearFile', () => {
    it('should clear breakpoints for file', () => {
      manager.add('test.ts', 10);
      manager.add('test.ts', 20);
      manager.add('other.ts', 5);

      manager.clearFile('test.ts');

      expect(manager.getAll()).toHaveLength(1);
      expect(manager.getAll()[0].file).toBe('other.ts');
    });
  });

  describe('getStats', () => {
    it('should return statistics', () => {
      const bp1 = manager.add('test.ts', 10);
      manager.add('test.ts', 20, { condition: (l) => l.x > 5 });
      const bp3 = manager.add('other.ts', 5);

      manager.disable(bp3.id);
      manager.recordHit(bp1.id);
      manager.recordHit(bp1.id);

      const stats = manager.getStats();

      expect(stats.total).toBe(3);
      expect(stats.enabled).toBe(2);
      expect(stats.disabled).toBe(1);
      expect(stats.with_conditions).toBe(1);
      expect(stats.total_hits).toBe(2);
    });
  });
});

describe('BreakpointGroup', () => {
  let group: BreakpointGroup;

  beforeEach(() => {
    group = new BreakpointGroup('test-group');
  });

  describe('add', () => {
    it('should add breakpoint to group', () => {
      group.add('bp-1');
      expect(group.has('bp-1')).toBe(true);
    });
  });

  describe('remove', () => {
    it('should remove breakpoint from group', () => {
      group.add('bp-1');
      group.remove('bp-1');

      expect(group.has('bp-1')).toBe(false);
    });
  });

  describe('getAll', () => {
    it('should return all breakpoint IDs', () => {
      group.add('bp-1');
      group.add('bp-2');

      const ids = group.getAll();
      expect(ids).toHaveLength(2);
      expect(ids).toContain('bp-1');
      expect(ids).toContain('bp-2');
    });
  });

  describe('clear', () => {
    it('should clear group', () => {
      group.add('bp-1');
      group.add('bp-2');
      group.clear();

      expect(group.getAll()).toHaveLength(0);
    });
  });

  describe('getName', () => {
    it('should return group name', () => {
      expect(group.getName()).toBe('test-group');
    });
  });
});

describe('Factory functions', () => {
  it('should create breakpoint manager', () => {
    const manager = createBreakpointManager();
    expect(manager).toBeInstanceOf(BreakpointManager);
  });

  it('should create breakpoint group', () => {
    const group = createGroup('test');
    expect(group).toBeInstanceOf(BreakpointGroup);
    expect(group.getName()).toBe('test');
  });
});
