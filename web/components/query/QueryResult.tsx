import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { QueryResult as QueryResultType } from "@/lib/types";
import { formatCost, formatDuration, formatPercentage } from "@/lib/utils";
import { AttentionViz } from "./AttentionViz";
import { ReasoningChain } from "./ReasoningChain";
import {
  CheckCircle2Icon,
  XCircleIcon,
  DollarSignIcon,
  ClockIcon,
  BrainIcon,
  FileTextIcon,
} from "lucide-react";

interface QueryResultProps {
  result: QueryResultType;
  query: string;
  compact?: boolean;
}

export function QueryResult({ result, query, compact = false }: QueryResultProps) {
  const confidenceColor = result.confidence >= 0.8
    ? "text-green-600"
    : result.confidence >= 0.6
    ? "text-yellow-600"
    : "text-red-600";

  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <CardTitle className="text-base font-medium text-muted-foreground">
              Query
            </CardTitle>
            <p className="text-lg mt-1">{query}</p>
          </div>
          <Badge variant={result.constitutional === "pass" ? "default" : "destructive"}>
            {result.constitutional === "pass" ? (
              <>
                <CheckCircle2Icon className="h-3 w-3 mr-1" />
                Constitutional Pass
              </>
            ) : (
              <>
                <XCircleIcon className="h-3 w-3 mr-1" />
                Constitutional Fail
              </>
            )}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Answer */}
        <div>
          <h3 className="font-semibold mb-2">Answer</h3>
          <div className="prose dark:prose-invert max-w-none">
            <p className="whitespace-pre-wrap">{result.answer}</p>
          </div>
        </div>

        {/* Metadata */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t">
          <div className="flex items-center gap-2">
            <BrainIcon className="h-4 w-4 text-muted-foreground" />
            <div>
              <div className="text-xs text-muted-foreground">Confidence</div>
              <div className={`font-semibold ${confidenceColor}`}>
                {formatPercentage(result.confidence)}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <DollarSignIcon className="h-4 w-4 text-muted-foreground" />
            <div>
              <div className="text-xs text-muted-foreground">Cost</div>
              <div className="font-semibold font-mono">
                {formatCost(result.cost)}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <ClockIcon className="h-4 w-4 text-muted-foreground" />
            <div>
              <div className="text-xs text-muted-foreground">Time</div>
              <div className="font-semibold">
                {formatDuration(result.time_ms)}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <FileTextIcon className="h-4 w-4 text-muted-foreground" />
            <div>
              <div className="text-xs text-muted-foreground">Functions</div>
              <div className="font-semibold">
                {result.functions_used.length}
              </div>
            </div>
          </div>
        </div>

        {!compact && (
          <>
            {/* Functions Used */}
            {result.functions_used.length > 0 && (
              <div className="pt-4 border-t">
                <h3 className="font-semibold mb-2">Functions Used</h3>
                <div className="flex flex-wrap gap-2">
                  {result.functions_used.map((fn, i) => (
                    <Badge key={i} variant="secondary">
                      {fn}
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            {/* Sources */}
            {result.sources.length > 0 && (
              <div className="pt-4 border-t">
                <h3 className="font-semibold mb-2">
                  Sources ({result.sources.length})
                </h3>
                <ul className="space-y-2">
                  {result.sources.map((source, i) => (
                    <li key={source.id} className="flex items-start gap-2">
                      <span className="text-muted-foreground">{i + 1}.</span>
                      <div className="flex-1">
                        <div className="font-medium">{source.title}</div>
                        <div className="text-sm text-muted-foreground">
                          {source.type} • Relevance: {formatPercentage(source.relevance)}
                        </div>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Attention Visualization */}
            {result.attention.length > 0 && (
              <div className="pt-4 border-t">
                <h3 className="font-semibold mb-4">Attention Weights</h3>
                <AttentionViz attention={result.attention} />
              </div>
            )}

            {/* Reasoning Chain */}
            {result.reasoning.length > 0 && (
              <div className="pt-4 border-t">
                <h3 className="font-semibold mb-4">Reasoning Chain</h3>
                <ReasoningChain steps={result.reasoning} />
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
