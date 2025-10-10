/**
 * VERDE Canary Deployment API
 *
 * GET /api/gvcs/canary?organismId=... - Get canary status
 * POST /api/gvcs/canary - Deploy/promote/rollback canary
 *
 * Manages canary deployments with genetic traffic control
 */

import { NextRequest, NextResponse } from 'next/server';
import {
  getCanaryStatus,
  deployCanary,
  promoteCanary,
  rollbackCanary,
  rollbackVersion,
} from '@/lib/integrations/gvcs';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const organismId = searchParams.get('organismId');

    if (!organismId) {
      return NextResponse.json(
        { error: 'Missing required parameter: organismId' },
        { status: 400 }
      );
    }

    const status = await getCanaryStatus(organismId);

    return NextResponse.json({
      success: true,
      data: status,
      timestamp: Date.now(),
    });
  } catch (error) {
    console.error('[API] /api/gvcs/canary GET error:', error);

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
    const { organismId, action, version, filePath, trafficPercent } = body;

    if (!organismId) {
      return NextResponse.json(
        { error: 'Missing required parameter: organismId' },
        { status: 400 }
      );
    }

    if (!action) {
      return NextResponse.json(
        { error: 'Missing required parameter: action' },
        { status: 400 }
      );
    }

    let result;

    switch (action) {
      case 'deploy':
        if (!filePath) {
          return NextResponse.json(
            { error: 'filePath required for deploy action' },
            { status: 400 }
          );
        }
        result = await deployCanary(organismId, filePath, trafficPercent || 1);
        break;

      case 'promote':
        if (!version) {
          return NextResponse.json(
            { error: 'version required for promote action' },
            { status: 400 }
          );
        }
        result = await promoteCanary(organismId, version);
        break;

      case 'rollback':
        if (!version) {
          return NextResponse.json(
            { error: 'version required for rollback action' },
            { status: 400 }
          );
        }
        result = await rollbackCanary(organismId, version);
        break;

      case 'rollback_version':
        if (!version) {
          return NextResponse.json(
            { error: 'version required for rollback_version action' },
            { status: 400 }
          );
        }
        result = await rollbackVersion(organismId, version);
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
    console.error('[API] /api/gvcs/canary POST error:', error);

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Internal server error',
      },
      { status: 500 }
    );
  }
}
