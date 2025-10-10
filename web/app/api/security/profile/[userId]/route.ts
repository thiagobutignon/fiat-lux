/**
 * VERMELHO Behavioral Profile API
 *
 * GET /api/security/profile/[userId]
 *
 * Gets behavioral profile for a user
 */

import { NextRequest, NextResponse } from 'next/server';
import { getBehavioralProfile } from '@/lib/integrations/security';

export async function GET(
  request: NextRequest,
  { params }: { params: { userId: string } }
) {
  try {
    const { userId } = params;

    if (!userId) {
      return NextResponse.json(
        { error: 'Missing userId parameter' },
        { status: 400 }
      );
    }

    const profile = await getBehavioralProfile(userId);

    return NextResponse.json({
      success: true,
      data: profile,
      timestamp: Date.now(),
    });
  } catch (error) {
    console.error(`[API] /api/security/profile/${params.userId} error:`, error);

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Internal server error',
      },
      { status: 500 }
    );
  }
}
