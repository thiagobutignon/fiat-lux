/**
 * CINZA Manipulation Detection API
 *
 * POST /api/cognitive/analyze
 *
 * Analyzes text for cognitive manipulation using CINZA detection engine
 */

import { NextRequest, NextResponse } from 'next/server';
import {
  detectManipulation,
  detectQueryManipulation,
  comprehensiveCognitiveAnalysis,
} from '@/lib/integrations/cognitive';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const { text, userId, organismId, analysisType = 'manipulation' } = body;

    if (!text) {
      return NextResponse.json(
        { error: 'Missing required parameter: text' },
        { status: 400 }
      );
    }

    let result;

    switch (analysisType) {
      case 'manipulation':
        result = await detectManipulation(text);
        break;

      case 'query':
        if (!userId || !organismId) {
          return NextResponse.json(
            { error: 'userId and organismId required for query analysis' },
            { status: 400 }
          );
        }
        result = await detectQueryManipulation(text, userId, organismId);
        break;

      case 'comprehensive':
        result = await comprehensiveCognitiveAnalysis({
          text,
          userId,
          organismId,
        });
        break;

      default:
        return NextResponse.json(
          { error: `Unknown analysisType: ${analysisType}` },
          { status: 400 }
        );
    }

    return NextResponse.json({
      success: true,
      data: result,
      timestamp: Date.now(),
    });
  } catch (error) {
    console.error('[API] /api/cognitive/analyze error:', error);

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
    endpoint: '/api/cognitive/analyze',
    method: 'POST',
    description: 'Analyze text for cognitive manipulation using CINZA',
    parameters: {
      text: 'string (required) - Text to analyze',
      userId: 'string (optional) - User ID',
      organismId: 'string (optional) - Organism ID for query analysis',
      analysisType: 'string (optional) - Type: manipulation, query, comprehensive',
    },
    example: {
      text: 'You must be imagining the security issues',
      analysisType: 'manipulation',
    },
  });
}
