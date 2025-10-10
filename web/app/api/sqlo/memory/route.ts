/**
 * LARANJA Episodic Memory API
 *
 * GET /api/sqlo/memory?organismId=... - Get episodic memory
 * POST /api/sqlo/memory - Store episodic memory
 *
 * Manages O(1) episodic memory storage and retrieval
 */

import { NextRequest, NextResponse } from 'next/server';
import {
  storeEpisodicMemory,
  getEpisodicMemory,
  getUserQueryHistory,
} from '@/lib/integrations/sqlo';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const organismId = searchParams.get('organismId');
    const userId = searchParams.get('userId');
    const limit = parseInt(searchParams.get('limit') || '100');

    if (organismId) {
      // Get organism memory
      const memory = await getEpisodicMemory(organismId, limit);

      return NextResponse.json({
        success: true,
        data: memory,
        timestamp: Date.now(),
      });
    } else if (userId) {
      // Get user history
      const history = await getUserQueryHistory(userId, limit);

      return NextResponse.json({
        success: true,
        data: history,
        timestamp: Date.now(),
      });
    } else {
      return NextResponse.json(
        { error: 'Missing required parameter: organismId or userId' },
        { status: 400 }
      );
    }
  } catch (error) {
    console.error('[API] /api/sqlo/memory GET error:', error);

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
    const { organism_id, query, result, user_id, session_id } = body;

    if (!organism_id || !query || !result || !user_id) {
      return NextResponse.json(
        { error: 'Missing required parameters' },
        { status: 400 }
      );
    }

    await storeEpisodicMemory({
      organism_id,
      query,
      result,
      user_id,
      session_id: session_id || `session_${Date.now()}`,
      timestamp: new Date().toISOString(),
    });

    return NextResponse.json({
      success: true,
      message: 'Episodic memory stored',
      timestamp: Date.now(),
    });
  } catch (error) {
    console.error('[API] /api/sqlo/memory POST error:', error);

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Internal server error',
      },
      { status: 500 }
    );
  }
}
