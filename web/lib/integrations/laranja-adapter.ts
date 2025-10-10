/**
 * LARANJA Adapter - Bridge between AMARELO and LARANJA Core
 *
 * This adapter provides a bridge between AMARELO's web dashboard
 * and LARANJA's core .sqlo O(1) database system.
 *
 * Architecture:
 * AMARELO → laranja-adapter.ts → LARANJA Core (.sqlo database)
 *
 * Performance Targets:
 * - Queries: <1ms (67μs-1.23ms)
 * - Inserts: <500μs
 * - Permission checks: <100μs
 *
 * NOTE: This is currently a mock in-memory implementation.
 * Will be replaced with real .sqlo database calls when LARANJA is ready.
 */

import type {
  GlassOrganism,
  QueryResult,
  ConstitutionalLog,
  LLMCall,
} from '../types';

import type {
  EpisodicMemory,
  RBACRole,
  RBACUser,
} from './sqlo';

// ============================================================================
// In-Memory Storage (Mock until real .sqlo is ready)
// ============================================================================

class LaranjaAdapter {
  // In-memory stores (will be replaced with .sqlo queries)
  private organisms: Map<string, GlassOrganism>;
  private episodicMemory: Map<string, EpisodicMemory>;
  private constitutionalLogs: Map<string, ConstitutionalLog>;
  private llmCalls: Map<string, LLMCall>;
  private rbacUsers: Map<string, RBACUser>;
  private rbacRoles: Map<string, RBACRole>;

  // Performance metrics
  private queryCount: number;
  private totalQueryTime: number; // microseconds

  constructor() {
    this.organisms = new Map();
    this.episodicMemory = new Map();
    this.constitutionalLogs = new Map();
    this.llmCalls = new Map();
    this.rbacUsers = new Map();
    this.rbacRoles = new Map();

    this.queryCount = 0;
    this.totalQueryTime = 0;

    // Initialize default admin role
    this.rbacRoles.set('admin', {
      role_id: 'admin',
      name: 'Administrator',
      permissions: ['read', 'write', 'delete', 'query', 'debug', 'admin'],
      created_at: new Date().toISOString(),
    });

    // Initialize default developer role
    this.rbacRoles.set('developer', {
      role_id: 'developer',
      name: 'Developer',
      permissions: ['read', 'write', 'query', 'debug'],
      created_at: new Date().toISOString(),
    });
  }

  // ==========================================================================
  // Performance Tracking
  // ==========================================================================

  private trackQuery(startTime: number): void {
    const duration = (Date.now() - startTime) * 1000; // Convert ms to μs
    this.queryCount++;
    this.totalQueryTime += duration;
  }

  // ==========================================================================
  // Organism Storage (O(1) operations)
  // ==========================================================================

  /**
   * Get organism by ID
   * Target: <1ms
   */
  async getOrganism(organismId: string): Promise<GlassOrganism | null> {
    const start = Date.now();

    const organism = this.organisms.get(organismId) || null;

    this.trackQuery(start);
    return organism;
  }

  /**
   * Get all organisms
   * Target: <1ms (returns Map values)
   */
  async getAllOrganisms(): Promise<GlassOrganism[]> {
    const start = Date.now();

    const organisms = Array.from(this.organisms.values());

    this.trackQuery(start);
    return organisms;
  }

  /**
   * Store organism
   * Target: <1ms
   */
  async storeOrganism(organism: GlassOrganism): Promise<void> {
    const start = Date.now();

    this.organisms.set(organism.id, organism);

    this.trackQuery(start);
  }

  /**
   * Update organism
   * Target: <1ms
   */
  async updateOrganism(
    organismId: string,
    updates: Partial<GlassOrganism>
  ): Promise<void> {
    const start = Date.now();

    const existing = this.organisms.get(organismId);
    if (existing) {
      this.organisms.set(organismId, { ...existing, ...updates });
    }

    this.trackQuery(start);
  }

  /**
   * Delete organism
   * Target: <1ms
   */
  async deleteOrganism(organismId: string): Promise<void> {
    const start = Date.now();

    this.organisms.delete(organismId);

    this.trackQuery(start);
  }

  // ==========================================================================
  // Episodic Memory (O(1) operations)
  // ==========================================================================

  /**
   * Store episodic memory
   * Target: <500μs
   */
  async storeEpisodicMemory(memory: Omit<EpisodicMemory, 'id'>): Promise<void> {
    const start = Date.now();

    const id = `mem_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    this.episodicMemory.set(id, { ...memory, id });

    this.trackQuery(start);
  }

  /**
   * Get episodic memory for organism
   * Target: <1ms
   */
  async getEpisodicMemory(organismId: string, limit: number = 100): Promise<EpisodicMemory[]> {
    const start = Date.now();

    const memories = Array.from(this.episodicMemory.values())
      .filter((m) => m.organism_id === organismId)
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      .slice(0, limit);

    this.trackQuery(start);
    return memories;
  }

  /**
   * Get user query history
   * Target: <1ms
   */
  async getUserQueryHistory(userId: string, limit: number = 50): Promise<EpisodicMemory[]> {
    const start = Date.now();

    const memories = Array.from(this.episodicMemory.values())
      .filter((m) => m.user_id === userId)
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      .slice(0, limit);

    this.trackQuery(start);
    return memories;
  }

  // ==========================================================================
  // Constitutional Logs (O(1) operations)
  // ==========================================================================

  /**
   * Store constitutional log
   * Target: <500μs
   */
  async storeConstitutionalLog(log: Omit<ConstitutionalLog, 'id'>): Promise<void> {
    const start = Date.now();

    const id = `log_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    this.constitutionalLogs.set(id, { ...log, id });

    this.trackQuery(start);
  }

  /**
   * Get constitutional logs
   * Target: <1ms
   */
  async getConstitutionalLogs(
    organismId: string,
    filters?: { principle?: string; status?: 'pass' | 'fail' | 'warning' },
    limit: number = 100
  ): Promise<ConstitutionalLog[]> {
    const start = Date.now();

    let logs = Array.from(this.constitutionalLogs.values()).filter(
      (log) => log.organism_id === organismId
    );

    if (filters?.principle) {
      logs = logs.filter((log) => log.principle === filters.principle);
    }

    if (filters?.status) {
      logs = logs.filter((log) => log.status === filters.status);
    }

    logs = logs
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      .slice(0, limit);

    this.trackQuery(start);
    return logs;
  }

  // ==========================================================================
  // LLM Calls (O(1) operations)
  // ==========================================================================

  /**
   * Store LLM call
   * Target: <500μs
   */
  async storeLLMCall(call: Omit<LLMCall, 'id'>): Promise<void> {
    const start = Date.now();

    const id = `llm_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    this.llmCalls.set(id, { ...call, id });

    this.trackQuery(start);
  }

  /**
   * Get LLM calls
   * Target: <1ms
   */
  async getLLMCalls(
    organismId: string,
    filters?: { task_type?: string; model?: string },
    limit: number = 100
  ): Promise<LLMCall[]> {
    const start = Date.now();

    let calls = Array.from(this.llmCalls.values()).filter(
      (call) => call.organism_id === organismId
    );

    if (filters?.task_type) {
      calls = calls.filter((call) => call.task_type === filters.task_type);
    }

    if (filters?.model) {
      calls = calls.filter((call) => call.model === filters.model);
    }

    calls = calls
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      .slice(0, limit);

    this.trackQuery(start);
    return calls;
  }

  // ==========================================================================
  // RBAC (O(1) operations)
  // ==========================================================================

  /**
   * Get user roles
   * Target: <500μs
   */
  async getUserRoles(userId: string): Promise<RBACUser> {
    const start = Date.now();

    let user = this.rbacUsers.get(userId);

    if (!user) {
      // Create default user with developer role
      user = {
        user_id: userId,
        roles: ['developer'],
        permissions: ['read', 'write', 'query', 'debug'],
        created_at: new Date().toISOString(),
      };
      this.rbacUsers.set(userId, user);
    }

    this.trackQuery(start);
    return user;
  }

  /**
   * Check permission
   * Target: <100μs
   */
  async checkPermission(userId: string, permission: string): Promise<boolean> {
    const start = Date.now();

    const user = await this.getUserRoles(userId);
    const hasPermission = user.permissions.includes(permission);

    this.trackQuery(start);
    return hasPermission;
  }

  /**
   * Create role
   * Target: <500μs
   */
  async createRole(role: Omit<RBACRole, 'created_at'>): Promise<void> {
    const start = Date.now();

    this.rbacRoles.set(role.role_id, {
      ...role,
      created_at: new Date().toISOString(),
    });

    this.trackQuery(start);
  }

  /**
   * Assign role to user
   * Target: <500μs
   */
  async assignRole(userId: string, roleId: string): Promise<void> {
    const start = Date.now();

    const user = await this.getUserRoles(userId);
    const role = this.rbacRoles.get(roleId);

    if (role && !user.roles.includes(roleId)) {
      user.roles.push(roleId);

      // Add role permissions to user permissions
      role.permissions.forEach((perm) => {
        if (!user.permissions.includes(perm)) {
          user.permissions.push(perm);
        }
      });

      this.rbacUsers.set(userId, user);
    }

    this.trackQuery(start);
  }

  // ==========================================================================
  // Consolidation Optimizer
  // ==========================================================================

  /**
   * Run consolidation optimizer
   * Target: Background process
   */
  async runConsolidation(): Promise<{ optimized: number; duration_ms: number }> {
    const start = Date.now();

    // Mock consolidation (would optimize storage in real .sqlo)
    const duration_ms = Math.random() * 100; // Simulate work

    return {
      optimized: 0, // No consolidation needed in mock
      duration_ms,
    };
  }

  /**
   * Get consolidation status
   * Target: <100μs
   */
  async getConsolidationStatus(): Promise<{
    last_run: string;
    next_run: string;
    status: string;
  }> {
    const start = Date.now();

    const status = {
      last_run: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
      next_run: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(),
      status: 'idle',
    };

    this.trackQuery(start);
    return status;
  }

  // ==========================================================================
  // Performance Metrics
  // ==========================================================================

  /**
   * Get .sqlo performance metrics
   * Target: <100μs
   */
  async getSQLOMetrics(): Promise<{
    avg_query_time_us: number;
    total_queries: number;
    cache_hit_rate: number;
  }> {
    const avg_query_time_us = this.queryCount > 0 ? this.totalQueryTime / this.queryCount : 0;

    return {
      avg_query_time_us: Math.round(avg_query_time_us),
      total_queries: this.queryCount,
      cache_hit_rate: 0.95, // Mock cache hit rate
    };
  }

  // ==========================================================================
  // Health & Status
  // ==========================================================================

  /**
   * Check if LARANJA is available
   */
  isAvailable(): boolean {
    return true; // Mock is always available
  }

  /**
   * Get LARANJA health status
   */
  async getHealth(): Promise<{
    status: string;
    version: string;
    performance_us: number;
    total_queries: number;
  }> {
    const metrics = await this.getSQLOMetrics();

    return {
      status: 'healthy',
      version: '1.0.0-mock',
      performance_us: metrics.avg_query_time_us,
      total_queries: metrics.total_queries,
    };
  }

  /**
   * Get storage statistics
   */
  getStorageStats(): {
    organisms: number;
    episodic_memory: number;
    constitutional_logs: number;
    llm_calls: number;
    rbac_users: number;
    rbac_roles: number;
  } {
    return {
      organisms: this.organisms.size,
      episodic_memory: this.episodicMemory.size,
      constitutional_logs: this.constitutionalLogs.size,
      llm_calls: this.llmCalls.size,
      rbac_users: this.rbacUsers.size,
      rbac_roles: this.rbacRoles.size,
    };
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let adapterInstance: LaranjaAdapter | null = null;

export function getLaranjaAdapter(): LaranjaAdapter {
  if (!adapterInstance) {
    adapterInstance = new LaranjaAdapter();
  }
  return adapterInstance;
}
