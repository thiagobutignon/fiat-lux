/**
 * VERMELHO Security Analysis API
 *
 * POST /api/security/analyze
 *
 * Analyzes text for duress/coercion using VERMELHO behavioral biometrics
 */

import { NextRequest, NextResponse } from 'next/server';
import {
  analyzeDuress,
  analyzeQueryDuress,
  comprehensiveSecurityAnalysis,
} from '@/lib/integrations/security';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const { text, userId, organismId, analysisType = 'duress' } = body;

    if (!text || !userId) {
      return NextResponse.json(
        { error: 'Missing required parameters: text, userId' },
        { status: 400 }
      );
    }

    let result;

    switch (analysisType) {
      case 'duress':
        result = await analyzeDuress(text, userId);
        break;

      case 'query':
        if (!organismId) {
          return NextResponse.json(
            { error: 'organismId required for query analysis' },
            { status: 400 }
          );
        }
        result = await analyzeQueryDuress(text, userId, organismId);
        break;

      case 'comprehensive':
        result = await comprehensiveSecurityAnalysis({
          text,
          userId,
          organismId,
          timestamp: Date.now(),
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
    console.error('[API] /api/security/analyze error:', error);

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
    endpoint: '/api/security/analyze',
    method: 'POST',
    description: 'Analyze text for duress/coercion using VERMELHO',
    parameters: {
      text: 'string (required) - Text to analyze',
      userId: 'string (required) - User ID',
      organismId: 'string (optional) - Organism ID for query analysis',
      analysisType: 'string (optional) - Type: duress, query, comprehensive',
    },
    example: {
      text: 'I need to delete all my data urgently',
      userId: 'user-123',
      analysisType: 'duress',
    },
  });
}
