/**
 * VERDE Version Management API
 *
 * GET /api/gvcs/versions?organismId=...
 * POST /api/gvcs/versions (get evolution data)
 *
 * Provides version history and evolution data for organisms
 */

import { NextRequest, NextResponse } from 'next/server';
import {
  getVersionHistory,
  getCurrentVersion,
  getEvolutionData,
  getFitnessTrajectory,
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

    const versions = await getVersionHistory(organismId);
    const current = await getCurrentVersion(organismId);

    return NextResponse.json({
      success: true,
      data: {
        versions,
        current,
      },
      timestamp: Date.now(),
    });
  } catch (error) {
    console.error('[API] /api/gvcs/versions GET error:', error);

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
    const { organismId, includeTrajectory = false } = body;

    if (!organismId) {
      return NextResponse.json(
        { error: 'Missing required parameter: organismId' },
        { status: 400 }
      );
    }

    const evolutionData = await getEvolutionData(organismId);

    // Optionally include fitness trajectory
    let trajectory = null;
    if (includeTrajectory) {
      trajectory = await getFitnessTrajectory(organismId);
    }

    return NextResponse.json({
      success: true,
      data: {
        ...evolutionData,
        fitness_trajectory: trajectory,
      },
      timestamp: Date.now(),
    });
  } catch (error) {
    console.error('[API] /api/gvcs/versions POST error:', error);

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Internal server error',
      },
      { status: 500 }
    );
  }
}
