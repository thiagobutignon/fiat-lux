/**
 * LARANJA RBAC API
 *
 * GET /api/sqlo/rbac?userId=... - Get user roles
 * POST /api/sqlo/rbac - Check permission, create role, assign role
 *
 * Manages Role-Based Access Control with O(1) operations
 */

import { NextRequest, NextResponse } from 'next/server';
import {
  getUserRoles,
  checkPermission,
  createRole,
  assignRole,
} from '@/lib/integrations/sqlo';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const userId = searchParams.get('userId');

    if (!userId) {
      return NextResponse.json(
        { error: 'Missing required parameter: userId' },
        { status: 400 }
      );
    }

    const roles = await getUserRoles(userId);

    return NextResponse.json({
      success: true,
      data: roles,
      timestamp: Date.now(),
    });
  } catch (error) {
    console.error('[API] /api/sqlo/rbac GET error:', error);

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Internal server error',
      },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, userId, permission, role } = body;

    if (!action) {
      return NextResponse.json(
        { error: 'Missing required parameter: action' },
        { status: 400 }
      );
    }

    let result;

    switch (action) {
      case 'check_permission':
        if (!userId || !permission) {
          return NextResponse.json(
            { error: 'userId and permission required' },
            { status: 400 }
          );
        }
        result = await checkPermission(userId, permission);
        break;

      case 'create_role':
        if (!role || !role.role_id || !role.name || !role.permissions) {
          return NextResponse.json(
            { error: 'role with role_id, name, and permissions required' },
            { status: 400 }
          );
        }
        await createRole(role);
        result = { message: 'Role created successfully' };
        break;

      case 'assign_role':
        if (!userId || !role?.role_id) {
          return NextResponse.json(
            { error: 'userId and role.role_id required' },
            { status: 400 }
          );
        }
        await assignRole(userId, role.role_id);
        result = { message: 'Role assigned successfully' };
        break;

      default:
        return NextResponse.json(
          { error: `Unknown action: ${action}` },
          { status: 400 }
        );
    }

    return NextResponse.json({
      success: true,
      data: result,
      timestamp: Date.now(),
    });
  } catch (error) {
    console.error('[API] /api/sqlo/rbac POST error:', error);

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Internal server error',
      },
      { status: 500 }
    );
  }
}
