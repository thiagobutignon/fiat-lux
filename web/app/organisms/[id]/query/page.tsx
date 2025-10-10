import { ApiClient } from "@/lib/api-client";
import { QueryConsole } from "@/components/query/QueryConsole";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { formatPercentage } from "@/lib/utils";
import { notFound } from "next/navigation";
import {
  FlaskConicalIcon,
  BrainIcon,
  CodeIcon,
  TrendingUpIcon,
} from "lucide-react";

export const dynamic = 'force-dynamic';

interface PageProps {
  params: { id: string };
}

export default async function QueryPage({ params }: PageProps) {
  let organism;

  try {
    organism = await ApiClient.getOrganism(params.id);
  } catch (error) {
    notFound();
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 mb-2">
          <FlaskConicalIcon className="h-6 w-6 text-primary" />
          <h1 className="text-3xl font-bold tracking-tight">
            {organism.metadata.name}
          </h1>
          <Badge>{organism.metadata.stage}</Badge>
        </div>
        <p className="text-muted-foreground">
          Query Console - Ask questions and get answers
        </p>
      </div>

      {/* Organism Info */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Organism Information</CardTitle>
          <CardDescription>
            Current capabilities and knowledge
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="flex items-center gap-2">
              <BrainIcon className="h-4 w-4 text-muted-foreground" />
              <div>
                <div className="text-xs text-muted-foreground">Maturity</div>
                <div className="font-semibold">
                  {formatPercentage(organism.metadata.maturity)}
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <CodeIcon className="h-4 w-4 text-muted-foreground" />
              <div>
                <div className="text-xs text-muted-foreground">Functions</div>
                <div className="font-semibold">
                  {organism.code.functions.length}
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <FlaskConicalIcon className="h-4 w-4 text-muted-foreground" />
              <div>
                <div className="text-xs text-muted-foreground">Knowledge</div>
                <div className="font-semibold">
                  {organism.knowledge.papers} papers
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <TrendingUpIcon className="h-4 w-4 text-muted-foreground" />
              <div>
                <div className="text-xs text-muted-foreground">Fitness</div>
                <div className="font-semibold">
                  {formatPercentage(organism.evolution.fitness)}
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Query Console */}
      <QueryConsole
        organismId={organism.id}
        organismName={organism.metadata.name}
      />
    </div>
  );
}
