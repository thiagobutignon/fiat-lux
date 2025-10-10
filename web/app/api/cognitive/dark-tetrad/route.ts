/**
 * CINZA Dark Tetrad Analysis API
 *
 * POST /api/cognitive/dark-tetrad
 *
 * Analyzes Dark Tetrad personality traits using CINZA
 */

import { NextRequest, NextResponse } from 'next/server';
import {
  getDarkTetradProfile,
  getUserDarkTetradProfile,
} from '@/lib/integrations/cognitive';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const { text, userId } = body;

    let result;

    if (text) {
      // Analyze specific text
      result = await getDarkTetradProfile(text);
    } else if (userId) {
      // Get user's historical profile
      result = await getUserDarkTetradProfile(userId);
    } else {
      return NextResponse.json(
        { error: 'Either text or userId required' },
        { status: 400 }
      );
    }

    return NextResponse.json({
      success: true,
      data: result,
      timestamp: Date.now(),
    });
  } catch (error) {
    console.error('[API] /api/cognitive/dark-tetrad error:', error);

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Internal server error',
      },
      { status: 500 }
    );
  }
}

// GET endpoint for testing
export async function GET() {
  return NextResponse.json({
    endpoint: '/api/cognitive/dark-tetrad',
    method: 'POST',
    description: 'Analyze Dark Tetrad personality traits using CINZA',
    parameters: {
      text: 'string (optional) - Text to analyze',
      userId: 'string (optional) - User ID for historical analysis',
    },
    dark_tetrad_dimensions: {
      narcissism: 'Grandiosity, entitlement, need for admiration',
      machiavellianism: 'Manipulation, exploitation, strategic thinking',
      psychopathy: 'Lack of empathy, impulsivity, antisocial behavior',
      sadism: 'Enjoyment of causing pain or suffering',
    },
    example: {
      text: 'I alone can fix this. Others are incompetent.',
    },
  });
}
