"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { LLMCall } from "@/lib/types";
import { formatCurrency, formatDuration } from "@/lib/utils";
import {
  MessageSquareIcon,
  CopyIcon,
  CheckIcon,
  ChevronDownIcon,
  ChevronUpIcon,
} from "lucide-react";

interface LLMInspectorProps {
  calls: LLMCall[];
}

export function LLMInspector({ calls }: LLMInspectorProps) {
  const [expandedCall, setExpandedCall] = useState<string | null>(null);
  const [copiedPrompt, setCopiedPrompt] = useState<string | null>(null);
  const [copiedResponse, setCopiedResponse] = useState<string | null>(null);

  const handleCopy = async (text: string, type: 'prompt' | 'response', callId: string) => {
    try {
      await navigator.clipboard.writeText(text);
      if (type === 'prompt') {
        setCopiedPrompt(callId);
        setTimeout(() => setCopiedPrompt(null), 2000);
      } else {
        setCopiedResponse(callId);
        setTimeout(() => setCopiedResponse(null), 2000);
      }
    } catch (error) {
      console.error("Failed to copy:", error);
    }
  };

  const toggleExpand = (callId: string) => {
    setExpandedCall(expandedCall === callId ? null : callId);
  };

  // Stats
  const totalCost = calls.reduce((sum, call) => sum + call.cost, 0);
  const totalTokensIn = calls.reduce((sum, call) => sum + call.tokens_in, 0);
  const totalTokensOut = calls.reduce((sum, call) => sum + call.tokens_out, 0);
  const avgLatency = calls.length > 0
    ? calls.reduce((sum, call) => sum + call.latency_ms, 0) / calls.length
    : 0;

  if (calls.length === 0) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-12">
          <div className="text-center">
            <MessageSquareIcon className="h-12 w-12 text-muted-foreground/50 mx-auto mb-4" />
            <p className="text-muted-foreground">No LLM calls yet</p>
            <p className="text-sm text-muted-foreground mt-2">
              LLM calls will appear after queries are executed
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="p-4">
            <CardDescription className="text-xs">Total Calls</CardDescription>
            <CardTitle className="text-2xl">{calls.length}</CardTitle>
          </CardHeader>
        </Card>

        <Card>
          <CardHeader className="p-4">
            <CardDescription className="text-xs">Total Cost</CardDescription>
            <CardTitle className="text-2xl">{formatCurrency(totalCost)}</CardTitle>
          </CardHeader>
        </Card>

        <Card>
          <CardHeader className="p-4">
            <CardDescription className="text-xs">Tokens (In/Out)</CardDescription>
            <CardTitle className="text-xl">
              {totalTokensIn.toLocaleString()} / {totalTokensOut.toLocaleString()}
            </CardTitle>
          </CardHeader>
        </Card>

        <Card>
          <CardHeader className="p-4">
            <CardDescription className="text-xs">Avg Latency</CardDescription>
            <CardTitle className="text-2xl">{Math.round(avgLatency)}ms</CardTitle>
          </CardHeader>
        </Card>
      </div>

      {/* LLM Calls */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <MessageSquareIcon className="h-5 w-5 text-primary" />
            <CardTitle>LLM Calls ({calls.length})</CardTitle>
          </div>
          <CardDescription>
            All LLM API calls made by this organism
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {calls.map((call, index) => {
              const isExpanded = expandedCall === call.id;
              const isPromptCopied = copiedPrompt === call.id;
              const isResponseCopied = copiedResponse === call.id;

              return (
                <Card key={call.id}>
                  <CardHeader className="p-4">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="font-semibold">
                            Call #{calls.length - index}
                          </span>
                          <Badge variant="outline">
                            {call.task_type.replace(/-/g, ' ')}
                          </Badge>
                          <Badge
                            variant={call.constitutional_status === 'pass' ? 'default' : 'destructive'}
                          >
                            {call.constitutional_status.toUpperCase()}
                          </Badge>
                        </div>

                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                          <div>
                            <div className="text-muted-foreground">Model</div>
                            <div className="font-semibold">{call.model}</div>
                          </div>
                          <div>
                            <div className="text-muted-foreground">Tokens</div>
                            <div className="font-semibold">
                              {call.tokens_in} → {call.tokens_out}
                            </div>
                          </div>
                          <div>
                            <div className="text-muted-foreground">Cost</div>
                            <div className="font-semibold">{formatCurrency(call.cost)}</div>
                          </div>
                          <div>
                            <div className="text-muted-foreground">Latency</div>
                            <div className="font-semibold">{call.latency_ms}ms</div>
                          </div>
                        </div>

                        {isExpanded && (
                          <div className="mt-4 space-y-4">
                            {/* Prompt */}
                            <div>
                              <div className="flex items-center justify-between mb-2">
                                <span className="text-sm font-semibold">Prompt</span>
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => handleCopy(call.prompt, 'prompt', call.id)}
                                >
                                  {isPromptCopied ? (
                                    <>
                                      <CheckIcon className="h-3 w-3 mr-1" />
                                      Copied
                                    </>
                                  ) : (
                                    <>
                                      <CopyIcon className="h-3 w-3 mr-1" />
                                      Copy
                                    </>
                                  )}
                                </Button>
                              </div>
                              <pre className="bg-muted/50 rounded-lg p-4 overflow-x-auto text-xs font-mono">
                                {call.prompt}
                              </pre>
                            </div>

                            {/* Response */}
                            <div>
                              <div className="flex items-center justify-between mb-2">
                                <span className="text-sm font-semibold">Response</span>
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => handleCopy(call.response, 'response', call.id)}
                                >
                                  {isResponseCopied ? (
                                    <>
                                      <CheckIcon className="h-3 w-3 mr-1" />
                                      Copied
                                    </>
                                  ) : (
                                    <>
                                      <CopyIcon className="h-3 w-3 mr-1" />
                                      Copy
                                    </>
                                  )}
                                </Button>
                              </div>
                              <pre className="bg-muted/50 rounded-lg p-4 overflow-x-auto text-xs font-mono">
                                {call.response}
                              </pre>
                            </div>
                          </div>
                        )}
                      </div>

                      <div className="flex flex-col items-end gap-2">
                        <div className="text-xs text-muted-foreground">
                          {new Date(call.timestamp).toLocaleString()}
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => toggleExpand(call.id)}
                        >
                          {isExpanded ? (
                            <>
                              <ChevronUpIcon className="h-4 w-4 mr-1" />
                              Hide
                            </>
                          ) : (
                            <>
                              <ChevronDownIcon className="h-4 w-4 mr-1" />
                              Show
                            </>
                          )}
                        </Button>
                      </div>
                    </div>
                  </CardHeader>
                </Card>
              );
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
