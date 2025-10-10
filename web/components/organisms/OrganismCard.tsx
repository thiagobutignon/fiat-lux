import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { GlassOrganism } from "@/lib/types";
import { formatPercentage, formatCost } from "@/lib/utils";
import {
  FlaskConicalIcon,
  BrainIcon,
  CodeIcon,
  TrendingUpIcon,
  ClockIcon,
} from "lucide-react";

interface OrganismCardProps {
  organism: GlassOrganism;
}

export function OrganismCard({ organism }: OrganismCardProps) {
  const { metadata, knowledge, code, evolution, stats } = organism;

  const getMaturityColor = (maturity: number) => {
    if (maturity >= 0.75) return "bg-green-500";
    if (maturity >= 0.5) return "bg-yellow-500";
    if (maturity >= 0.25) return "bg-orange-500";
    return "bg-red-500";
  };

  const getStageLabel = (stage: string) => {
    const labels: Record<string, string> = {
      nascent: "Nascent",
      infancy: "Infancy",
      adolescence: "Adolescence",
      maturity: "Maturity",
      evolution: "Evolution",
    };
    return labels[stage] || stage;
  };

  return (
    <Card className="hover:shadow-lg transition-shadow">
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-2">
            <FlaskConicalIcon className="h-5 w-5 text-primary" />
            <CardTitle>{metadata.name}</CardTitle>
          </div>
          <Badge variant={metadata.maturity >= 0.75 ? "default" : "secondary"}>
            {getStageLabel(metadata.stage)}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Maturity Progress */}
        <div>
          <div className="flex items-center justify-between mb-2 text-sm">
            <span className="text-muted-foreground">Maturity</span>
            <span className="font-semibold">
              {formatPercentage(metadata.maturity)}
            </span>
          </div>
          <div className="w-full bg-muted rounded-full h-2">
            <div
              className={`h-2 rounded-full transition-all ${getMaturityColor(
                metadata.maturity
              )}`}
              style={{ width: `${metadata.maturity * 100}%` }}
            />
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div className="flex items-center gap-2">
            <CodeIcon className="h-4 w-4 text-muted-foreground" />
            <div>
              <div className="text-muted-foreground">Functions</div>
              <div className="font-semibold">
                {code.functions.length}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <BrainIcon className="h-4 w-4 text-muted-foreground" />
            <div>
              <div className="text-muted-foreground">Knowledge</div>
              <div className="font-semibold">
                {knowledge.papers} papers
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <TrendingUpIcon className="h-4 w-4 text-muted-foreground" />
            <div>
              <div className="text-muted-foreground">Fitness</div>
              <div className="font-semibold">
                {formatPercentage(evolution.fitness)}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <ClockIcon className="h-4 w-4 text-muted-foreground" />
            <div>
              <div className="text-muted-foreground">Generation</div>
              <div className="font-semibold">Gen {evolution.generation}</div>
            </div>
          </div>
        </div>

        {/* Specialization & Cost */}
        <div className="pt-2 border-t space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Specialization</span>
            <Badge variant="outline">{metadata.specialization}</Badge>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Total Cost</span>
            <span className="font-mono font-semibold">
              {formatCost(stats.total_cost)}
            </span>
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-2 pt-2">
          <Button asChild variant="default" className="flex-1">
            <Link href={`/organisms/${organism.id}/query`}>Query</Link>
          </Button>
          <Button asChild variant="outline" className="flex-1">
            <Link href={`/organisms/${organism.id}/inspect`}>Inspect</Link>
          </Button>
          <Button asChild variant="ghost">
            <Link href={`/organisms/${organism.id}/debug`}>Debug</Link>
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
