import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Pattern } from "@/lib/types";
import { formatPercentage } from "@/lib/utils";
import { SparklesIcon, CheckCircle2Icon, ZapIcon } from "lucide-react";

interface PatternListProps {
  patterns: Pattern[];
}

export function PatternList({ patterns }: PatternListProps) {
  // Sort by emergence score descending
  const sortedPatterns = [...patterns].sort((a, b) => b.emergence_score - a.emergence_score);

  if (patterns.length === 0) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-12">
          <div className="text-center">
            <SparklesIcon className="h-12 w-12 text-muted-foreground/50 mx-auto mb-4" />
            <p className="text-muted-foreground">No patterns detected yet</p>
            <p className="text-sm text-muted-foreground mt-2">
              Patterns will emerge as knowledge is ingested
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold">Detected Patterns ({patterns.length})</h3>
        <div className="text-sm text-muted-foreground">
          Sorted by emergence score
        </div>
      </div>

      <div className="grid gap-4">
        {sortedPatterns.map((pattern, index) => {
          const isEmergenceReady = pattern.emergence_score >= 0.75;
          const hasEmergedFunction = !!pattern.emerged_function;

          return (
            <Card key={`${pattern.keyword}-${index}`}>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <SparklesIcon className="h-5 w-5 text-primary" />
                    <CardTitle className="text-lg">{pattern.keyword}</CardTitle>
                    {hasEmergedFunction && (
                      <Badge variant="default">
                        <CheckCircle2Icon className="h-3 w-3 mr-1" />
                        Emerged
                      </Badge>
                    )}
                    {isEmergenceReady && !hasEmergedFunction && (
                      <Badge variant="secondary">
                        <ZapIcon className="h-3 w-3 mr-1" />
                        Ready to Emerge
                      </Badge>
                    )}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {pattern.frequency} occurrences
                  </div>
                </div>
                {hasEmergedFunction && (
                  <CardDescription>
                    Emerged as: <span className="font-mono">{pattern.emerged_function}</span>
                  </CardDescription>
                )}
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <div className="text-sm text-muted-foreground mb-1">Frequency</div>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 bg-muted rounded-full h-2">
                        <div
                          className="bg-blue-500 rounded-full h-2 transition-all"
                          style={{ width: `${Math.min(pattern.frequency / 3, 100)}%` }}
                        />
                      </div>
                      <span className="text-sm font-semibold">{pattern.frequency}</span>
                    </div>
                  </div>

                  <div>
                    <div className="text-sm text-muted-foreground mb-1">Confidence</div>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 bg-muted rounded-full h-2">
                        <div
                          className="bg-green-500 rounded-full h-2 transition-all"
                          style={{ width: `${pattern.confidence * 100}%` }}
                        />
                      </div>
                      <span className="text-sm font-semibold">
                        {formatPercentage(pattern.confidence)}
                      </span>
                    </div>
                  </div>

                  <div>
                    <div className="text-sm text-muted-foreground mb-1">Emergence Score</div>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 bg-muted rounded-full h-2">
                        <div
                          className={`rounded-full h-2 transition-all ${
                            isEmergenceReady ? 'bg-yellow-500' : 'bg-orange-500'
                          }`}
                          style={{ width: `${pattern.emergence_score * 100}%` }}
                        />
                      </div>
                      <span className="text-sm font-semibold">
                        {formatPercentage(pattern.emergence_score)}
                      </span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
