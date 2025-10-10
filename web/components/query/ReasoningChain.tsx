import { ReasoningStep } from "@/lib/types";
import { formatDuration, formatPercentage } from "@/lib/utils";
import { CheckCircle2Icon, ChevronRightIcon } from "lucide-react";

interface ReasoningChainProps {
  steps: ReasoningStep[];
}

export function ReasoningChain({ steps }: ReasoningChainProps) {
  return (
    <div className="space-y-2">
      {steps.map((step, index) => (
        <div
          key={step.step}
          className="flex items-start gap-3 p-3 rounded-lg bg-muted/50 hover:bg-muted transition-colors"
        >
          <div className="flex items-center justify-center w-6 h-6 rounded-full bg-primary text-primary-foreground text-xs font-semibold flex-shrink-0">
            {step.step}
          </div>

          <div className="flex-1">
            <div className="font-medium">{step.description}</div>
            <div className="flex items-center gap-4 mt-1 text-sm text-muted-foreground">
              <span>Confidence: {formatPercentage(step.confidence)}</span>
              <span>•</span>
              <span>Time: {formatDuration(step.time_ms)}</span>
            </div>
          </div>

          <div className="flex items-center">
            {step.confidence >= 0.8 ? (
              <CheckCircle2Icon className="h-5 w-5 text-green-600" />
            ) : (
              <CheckCircle2Icon className="h-5 w-5 text-yellow-600" />
            )}
          </div>

          {index < steps.length - 1 && (
            <ChevronRightIcon className="h-4 w-4 text-muted-foreground absolute right-0" />
          )}
        </div>
      ))}
    </div>
  );
}
