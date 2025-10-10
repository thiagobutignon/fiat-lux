/**
 * RBAC (Role-Based Access Control) for SQLO
 *
 * O(1) permission checking using hash-based lookups
 * Controls access to memory types (SHORT_TERM, LONG_TERM, CONTEXTUAL)
 *
 * Why RBAC?
 * - Constitutional AI requires access control
 * - Different users/contexts have different permissions
 * - Must maintain O(1) performance guarantee
 *
 * Design:
 * - Role → MemoryType → Permission[] mapping
 * - Map-based storage for O(1) lookups
 * - Immutable permission sets (Set operations)
 */

import { MemoryType } from './sqlo';

// ============================================================================
// Types
// ============================================================================

/**
 * Permission types for RBAC
 */
export enum Permission {
  READ = 'read',       // Can read episodes
  WRITE = 'write',     // Can create/update episodes
  DELETE = 'delete'    // Can delete episodes
}

/**
 * Role with permissions for each memory type
 */
export interface Role {
  name: string;
  description?: string;
  permissions: Map<MemoryType, Set<Permission>>;
}

/**
 * Permission check result
 */
export interface PermissionCheckResult {
  granted: boolean;
  role: string;
  memoryType: MemoryType;
  permission: Permission;
  reason?: string;
}

// ============================================================================
// RBAC Policy Engine (all O(1) operations)
// ============================================================================

export class RbacPolicy {
  private roles: Map<string, Role>;

  constructor() {
    this.roles = new Map();
    this.initializeDefaultRoles();
  }

  // ==========================================================================
  // Permission Checking (O(1))
  // ==========================================================================

  /**
   * Check if role has permission - O(1)
   * Uses Map lookup + Set membership check
   */
  hasPermission(
    roleName: string,
    memoryType: MemoryType,
    permission: Permission
  ): boolean {
    const role = this.roles.get(roleName); // O(1)
    if (!role) {
      return false;
    }

    const permissions = role.permissions.get(memoryType); // O(1)
    if (!permissions) {
      return false;
    }

    return permissions.has(permission); // O(1)
  }

  /**
   * Check permission with detailed result - O(1)
   */
  checkPermission(
    roleName: string,
    memoryType: MemoryType,
    permission: Permission
  ): PermissionCheckResult {
    const granted = this.hasPermission(roleName, memoryType, permission);

    return {
      granted,
      role: roleName,
      memoryType,
      permission,
      reason: granted ? undefined : this.getPermissionDenialReason(roleName, memoryType, permission)
    };
  }

  /**
   * Check multiple permissions at once - O(k) where k = number of permissions
   */
  hasAllPermissions(
    roleName: string,
    memoryType: MemoryType,
    permissions: Permission[]
  ): boolean {
    return permissions.every(p => this.hasPermission(roleName, memoryType, p));
  }

  /**
   * Check if role has ANY of the permissions - O(k) where k = number of permissions
   */
  hasAnyPermission(
    roleName: string,
    memoryType: MemoryType,
    permissions: Permission[]
  ): boolean {
    return permissions.some(p => this.hasPermission(roleName, memoryType, p));
  }

  // ==========================================================================
  // Role Management (O(1))
  // ==========================================================================

  /**
   * Create new role - O(1)
   */
  createRole(name: string, description?: string): Role {
    if (this.roles.has(name)) {
      throw new Error(`Role '${name}' already exists`);
    }

    const role: Role = {
      name,
      description,
      permissions: new Map()
    };

    this.roles.set(name, role); // O(1)
    return role;
  }

  /**
   * Get role - O(1)
   */
  getRole(name: string): Role | undefined {
    return this.roles.get(name); // O(1)
  }

  /**
   * Delete role - O(1)
   */
  deleteRole(name: string): boolean {
    return this.roles.delete(name); // O(1)
  }

  /**
   * List all roles - O(n) where n = number of roles
   */
  listRoles(): Role[] {
    return Array.from(this.roles.values());
  }

  // ==========================================================================
  // Permission Management (O(1))
  // ==========================================================================

  /**
   * Grant permission to role - O(1)
   */
  grantPermission(
    roleName: string,
    memoryType: MemoryType,
    permission: Permission
  ): void {
    const role = this.roles.get(roleName); // O(1)
    if (!role) {
      throw new Error(`Role '${roleName}' does not exist`);
    }

    // Get or create permission set for this memory type
    let permissions = role.permissions.get(memoryType); // O(1)
    if (!permissions) {
      permissions = new Set();
      role.permissions.set(memoryType, permissions); // O(1)
    }

    permissions.add(permission); // O(1)
  }

  /**
   * Revoke permission from role - O(1)
   */
  revokePermission(
    roleName: string,
    memoryType: MemoryType,
    permission: Permission
  ): void {
    const role = this.roles.get(roleName); // O(1)
    if (!role) {
      throw new Error(`Role '${roleName}' does not exist`);
    }

    const permissions = role.permissions.get(memoryType); // O(1)
    if (permissions) {
      permissions.delete(permission); // O(1)
    }
  }

  /**
   * Grant all permissions for a memory type - O(1)
   */
  grantAllPermissions(roleName: string, memoryType: MemoryType): void {
    for (const permission of Object.values(Permission)) {
      this.grantPermission(roleName, memoryType, permission as Permission);
    }
  }

  /**
   * Revoke all permissions for a memory type - O(1)
   */
  revokeAllPermissions(roleName: string, memoryType: MemoryType): void {
    const role = this.roles.get(roleName);
    if (!role) {
      throw new Error(`Role '${roleName}' does not exist`);
    }

    role.permissions.delete(memoryType); // O(1)
  }

  // ==========================================================================
  // Default Roles (Constitutional AI defaults)
  // ==========================================================================

  /**
   * Initialize default roles for constitutional AI
   */
  private initializeDefaultRoles(): void {
    // ADMIN: Full access to everything
    const admin = this.createRole('admin', 'Full access to all memory types');
    this.grantAllPermissions('admin', MemoryType.SHORT_TERM);
    this.grantAllPermissions('admin', MemoryType.LONG_TERM);
    this.grantAllPermissions('admin', MemoryType.CONTEXTUAL);

    // USER: Read/write to short-term and contextual, read-only long-term
    const user = this.createRole('user', 'Standard user with limited long-term access');
    this.grantPermission('user', MemoryType.SHORT_TERM, Permission.READ);
    this.grantPermission('user', MemoryType.SHORT_TERM, Permission.WRITE);
    this.grantPermission('user', MemoryType.CONTEXTUAL, Permission.READ);
    this.grantPermission('user', MemoryType.CONTEXTUAL, Permission.WRITE);
    this.grantPermission('user', MemoryType.LONG_TERM, Permission.READ);

    // READONLY: Read-only access to all memory types
    const readonly = this.createRole('readonly', 'Read-only access for auditing');
    this.grantPermission('readonly', MemoryType.SHORT_TERM, Permission.READ);
    this.grantPermission('readonly', MemoryType.LONG_TERM, Permission.READ);
    this.grantPermission('readonly', MemoryType.CONTEXTUAL, Permission.READ);

    // SYSTEM: System-level access (like consolidation service)
    const system = this.createRole('system', 'System services with write access to long-term');
    this.grantAllPermissions('system', MemoryType.SHORT_TERM);
    this.grantAllPermissions('system', MemoryType.LONG_TERM);
    this.grantAllPermissions('system', MemoryType.CONTEXTUAL);

    // GUEST: No access (must be explicitly granted)
    this.createRole('guest', 'No default permissions');
  }

  // ==========================================================================
  // Helpers
  // ==========================================================================

  /**
   * Get reason for permission denial
   */
  private getPermissionDenialReason(
    roleName: string,
    memoryType: MemoryType,
    permission: Permission
  ): string {
    const role = this.roles.get(roleName);
    if (!role) {
      return `Role '${roleName}' does not exist`;
    }

    const permissions = role.permissions.get(memoryType);
    if (!permissions || permissions.size === 0) {
      return `Role '${roleName}' has no permissions for ${memoryType} memory`;
    }

    return `Role '${roleName}' does not have '${permission}' permission for ${memoryType} memory`;
  }

  /**
   * Get all permissions for a role and memory type - O(1)
   */
  getPermissions(roleName: string, memoryType: MemoryType): Set<Permission> | undefined {
    const role = this.roles.get(roleName);
    if (!role) {
      return undefined;
    }

    return role.permissions.get(memoryType);
  }

  /**
   * Export policy as JSON (for persistence)
   */
  toJSON(): any {
    const rolesObj: any = {};

    for (const [name, role] of this.roles) {
      const perms: any = {};
      for (const [memType, permSet] of role.permissions) {
        perms[memType] = Array.from(permSet);
      }

      rolesObj[name] = {
        name: role.name,
        description: role.description,
        permissions: perms
      };
    }

    return { roles: rolesObj };
  }

  /**
   * Import policy from JSON (for persistence)
   */
  static fromJSON(data: any): RbacPolicy {
    const policy = new RbacPolicy();
    policy.roles.clear(); // Remove default roles

    if (!data.roles) {
      return policy;
    }

    for (const [name, roleDataRaw] of Object.entries(data.roles as any)) {
      const roleData = roleDataRaw as any;
      const role = policy.createRole(name, roleData.description);

      for (const [memType, permissionsRaw] of Object.entries(roleData.permissions as any)) {
        const permissions = permissionsRaw as Permission[];
        for (const permission of permissions) {
          policy.grantPermission(name, memType as MemoryType, permission);
        }
      }
    }

    return policy;
  }
}

// ============================================================================
// Singleton Instance (Global Policy)
// ============================================================================

let _globalRbacPolicy: RbacPolicy | null = null;

/**
 * Get global RBAC policy instance (lazy initialization)
 * Can be customized per deployment
 */
export function getGlobalRbacPolicy(): RbacPolicy {
  if (!_globalRbacPolicy) {
    _globalRbacPolicy = new RbacPolicy();
  }
  return _globalRbacPolicy;
}
