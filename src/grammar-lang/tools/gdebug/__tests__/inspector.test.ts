/**
 * GDebug Inspector - Unit Tests
 */

import { VariableInspector, ScopeInspector, createInspector, createScopeInspector } from '../inspector';

describe('VariableInspector', () => {
  let inspector: VariableInspector;

  beforeEach(() => {
    inspector = new VariableInspector();
  });

  describe('setLocals', () => {
    it('should set local variables', () => {
      inspector.setLocals({ x: 10, y: 20 });

      expect(inspector.getVariable('x')?.value).toBe(10);
      expect(inspector.getVariable('y')?.value).toBe(20);
    });

    it('should replace existing locals', () => {
      inspector.setLocals({ x: 10 });
      inspector.setLocals({ y: 20 });

      expect(inspector.getVariable('x')).toBeNull();
      expect(inspector.getVariable('y')?.value).toBe(20);
    });
  });

  describe('getVariable', () => {
    beforeEach(() => {
      inspector.setLocals({
        num: 42,
        str: 'hello',
        obj: { a: 1, b: 2 },
        arr: [1, 2, 3],
        nil: null,
        undef: undefined
      });
    });

    it('should get number variable', () => {
      const v = inspector.getVariable('num');

      expect(v?.name).toBe('num');
      expect(v?.value).toBe(42);
      expect(v?.type).toBe('number');
      expect(v?.scope).toBe('local');
      expect(v?.writable).toBe(true);
      expect(v?.size_bytes).toBe(8);
    });

    it('should get string variable', () => {
      const v = inspector.getVariable('str');

      expect(v?.type).toBe('string');
      expect(v?.size_bytes).toBe(10); // 'hello' = 5 chars * 2 bytes
    });

    it('should get object variable', () => {
      const v = inspector.getVariable('obj');

      expect(v?.type).toBe('Object');
    });

    it('should get array variable', () => {
      const v = inspector.getVariable('arr');

      expect(v?.type).toBe('array');
    });

    it('should get null variable', () => {
      const v = inspector.getVariable('nil');

      expect(v?.type).toBe('null');
      expect(v?.size_bytes).toBe(0);
    });

    it('should return null for non-existent variable', () => {
      expect(inspector.getVariable('nonexistent')).toBeNull();
    });
  });

  describe('inspect', () => {
    it('should deeply inspect object', () => {
      inspector.setLocals({
        obj: { x: 10, y: 20, nested: { z: 30 } }
      });

      const result = inspector.inspect('obj');

      expect(result?.variable.name).toBe('obj');
      expect(result?.properties).toEqual({
        x: 10,
        y: 20,
        nested: { z: 30 }
      });
      expect(result?.constructor_name).toBe('Object');
    });

    it('should inspect array', () => {
      inspector.setLocals({ arr: [1, 2, 3] });

      const result = inspector.inspect('arr');

      expect(result?.variable.type).toBe('array');
      expect(result?.properties).toEqual({ '0': 1, '1': 2, '2': 3 });
    });

    it('should return null for primitive', () => {
      inspector.setLocals({ num: 42 });

      const result = inspector.inspect('num');

      expect(result?.properties).toBeUndefined();
    });

    it('should return null for non-existent variable', () => {
      expect(inspector.inspect('nonexistent')).toBeNull();
    });
  });

  describe('getAllVariables', () => {
    it('should return all variables', () => {
      inspector.setLocals({ x: 10, y: 20, z: 30 });

      const vars = inspector.getAllVariables();

      expect(vars).toHaveLength(3);
      expect(vars.map(v => v.name)).toContain('x');
      expect(vars.map(v => v.name)).toContain('y');
      expect(vars.map(v => v.name)).toContain('z');
    });

    it('should return empty array for no variables', () => {
      expect(inspector.getAllVariables()).toHaveLength(0);
    });
  });

  describe('addWatch', () => {
    it('should add watch expression', () => {
      inspector.setLocals({ x: 10 });

      const id = inspector.addWatch('x * 2', () => {
        const x = inspector.getVariable('x');
        return x ? x.value * 2 : null;
      });

      const watch = inspector.getWatch(id);

      expect(watch?.expression).toBe('x * 2');
      expect(watch?.value).toBe(20);
      expect(watch?.type).toBe('number');
      expect(watch?.change_count).toBe(0);
    });

    it('should handle errors in evaluator', () => {
      const id = inspector.addWatch('error', () => {
        throw new Error('Test error');
      });

      const watch = inspector.getWatch(id);
      expect(watch?.value).toHaveProperty('error');
    });
  });

  describe('updateWatch', () => {
    it('should update watch value', () => {
      inspector.setLocals({ x: 10 });

      const id = inspector.addWatch('x', () => {
        const x = inspector.getVariable('x');
        return x?.value;
      });

      // Update locals
      inspector.setLocals({ x: 20 });

      inspector.updateWatch(id, () => {
        const x = inspector.getVariable('x');
        return x?.value;
      });

      const watch = inspector.getWatch(id);
      expect(watch?.value).toBe(20);
      expect(watch?.change_count).toBe(1);
    });

    it('should not increment change count if value same', () => {
      inspector.setLocals({ x: 10 });

      const id = inspector.addWatch('x', () => {
        const x = inspector.getVariable('x');
        return x?.value;
      });

      inspector.updateWatch(id, () => {
        const x = inspector.getVariable('x');
        return x?.value;
      });

      const watch = inspector.getWatch(id);
      expect(watch?.change_count).toBe(0);
    });

    it('should return false for non-existent watch', () => {
      expect(inspector.updateWatch('nonexistent', () => 42)).toBe(false);
    });
  });

  describe('removeWatch', () => {
    it('should remove watch', () => {
      const id = inspector.addWatch('test', () => 42);
      const removed = inspector.removeWatch(id);

      expect(removed).toBe(true);
      expect(inspector.getWatch(id)).toBeUndefined();
    });

    it('should return false for non-existent watch', () => {
      expect(inspector.removeWatch('nonexistent')).toBe(false);
    });
  });

  describe('getAllWatches', () => {
    it('should return all watches', () => {
      inspector.addWatch('w1', () => 1);
      inspector.addWatch('w2', () => 2);

      expect(inspector.getAllWatches()).toHaveLength(2);
    });
  });

  describe('trackValue', () => {
    it('should track value changes', () => {
      inspector.trackValue('x', 10);
      inspector.trackValue('x', 20);
      inspector.trackValue('x', 30);

      const history = inspector.getValueHistory('x');
      expect(history).toHaveLength(3);
    });

    it('should limit history to 100 entries', () => {
      for (let i = 0; i < 150; i++) {
        inspector.trackValue('x', i);
      }

      expect(inspector.getValueHistory('x')).toHaveLength(100);
    });
  });

  describe('evaluate', () => {
    it('should evaluate simple variable', () => {
      inspector.setLocals({ x: 42 });

      const result = inspector.evaluate('x');

      expect(result.success).toBe(true);
      expect(result.value).toBe(42);
    });

    it('should return error for non-existent variable', () => {
      const result = inspector.evaluate('nonexistent');

      expect(result.success).toBe(false);
      expect(result.error).toContain('not found');
    });
  });

  describe('clear', () => {
    it('should clear all data', () => {
      inspector.setLocals({ x: 10 });
      inspector.addWatch('w', () => 1);
      inspector.trackValue('x', 10);

      inspector.clear();

      expect(inspector.getAllVariables()).toHaveLength(0);
      expect(inspector.getAllWatches()).toHaveLength(0);
      expect(inspector.getValueHistory('x')).toHaveLength(0);
    });
  });
});

describe('ScopeInspector', () => {
  let inspector: ScopeInspector;

  beforeEach(() => {
    inspector = new ScopeInspector();
  });

  describe('setScope', () => {
    it('should set variables in scope', () => {
      inspector.setScope('local', { x: 10, y: 20 });

      expect(inspector.getVariable('x')?.scope).toBe('local');
    });

    it('should replace existing scope variables', () => {
      inspector.setScope('local', { x: 10 });
      inspector.setScope('local', { y: 20 });

      expect(inspector.getVariable('x')).toBeNull();
      expect(inspector.getVariable('y')?.value).toBe(20);
    });
  });

  describe('getVariable', () => {
    beforeEach(() => {
      inspector.setScope('local', { local_var: 'local' });
      inspector.setScope('closure', { closure_var: 'closure' });
      inspector.setScope('module', { module_var: 'module' });
      inspector.setScope('global', { global_var: 'global' });
    });

    it('should find variable in local scope', () => {
      const v = inspector.getVariable('local_var');

      expect(v?.scope).toBe('local');
      expect(v?.value).toBe('local');
    });

    it('should find variable in closure scope', () => {
      const v = inspector.getVariable('closure_var');

      expect(v?.scope).toBe('closure');
      expect(v?.value).toBe('closure');
    });

    it('should find variable in module scope', () => {
      const v = inspector.getVariable('module_var');

      expect(v?.scope).toBe('module');
      expect(v?.value).toBe('module');
    });

    it('should find variable in global scope', () => {
      const v = inspector.getVariable('global_var');

      expect(v?.scope).toBe('global');
      expect(v?.value).toBe('global');
      expect(v?.writable).toBe(false); // globals are readonly
    });

    it('should prioritize local over closure', () => {
      inspector.setScope('local', { x: 'local' });
      inspector.setScope('closure', { x: 'closure' });

      const v = inspector.getVariable('x');
      expect(v?.value).toBe('local');
    });

    it('should return null for non-existent variable', () => {
      expect(inspector.getVariable('nonexistent')).toBeNull();
    });
  });

  describe('getScopeVariables', () => {
    it('should return variables in specific scope', () => {
      inspector.setScope('local', { x: 10, y: 20 });
      inspector.setScope('global', { z: 30 });

      const localVars = inspector.getScopeVariables('local');

      expect(localVars).toHaveLength(2);
      expect(localVars.every(v => v.scope === 'local')).toBe(true);
    });
  });

  describe('getSummary', () => {
    it('should return scope summary', () => {
      inspector.setScope('local', { x: 10, y: 20 });
      inspector.setScope('closure', { z: 30 });
      inspector.setScope('global', { w: 40 });

      const summary = inspector.getSummary();

      expect(summary.local).toBe(2);
      expect(summary.closure).toBe(1);
      expect(summary.module).toBe(0);
      expect(summary.global).toBe(1);
    });
  });
});

describe('Factory functions', () => {
  it('should create variable inspector', () => {
    const inspector = createInspector();
    expect(inspector).toBeInstanceOf(VariableInspector);
  });

  it('should create scope inspector', () => {
    const inspector = createScopeInspector();
    expect(inspector).toBeInstanceOf(ScopeInspector);
  });
});
