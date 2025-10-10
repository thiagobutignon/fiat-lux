import { ApiClient } from "@/lib/api-client";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { formatPercentage, formatCost } from "@/lib/utils";
import { notFound, redirect } from "next/navigation";
import Link from "next/link";
import {
  FlaskConicalIcon,
  BrainIcon,
  CodeIcon,
  TrendingUpIcon,
  MessageSquareIcon,
  EyeIcon,
  BugIcon,
} from "lucide-react";

export const dynamic = 'force-dynamic';

interface PageProps {
  params: { id: string };
}

export default async function OrganismDetailPage({ params }: PageProps) {
  let organism;

  try {
    organism = await ApiClient.getOrganism(params.id);
  } catch (error) {
    notFound();
  }

  // Auto-redirect to query page (main use case)
  redirect(`/organisms/${params.id}/query`);

  // This code won't be reached due to redirect, but kept for reference
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
          {organism.metadata.specialization}
        </p>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="hover:shadow-lg transition-shadow cursor-pointer" asChild>
          <Link href={`/organisms/${organism.id}/query`}>
            <CardHeader>
              <MessageSquareIcon className="h-8 w-8 text-primary mb-2" />
              <CardTitle>Query Console</CardTitle>
              <CardDescription>Ask questions and get answers</CardDescription>
            </CardHeader>
          </Link>
        </Card>

        <Card className="hover:shadow-lg transition-shadow cursor-pointer" asChild>
          <Link href={`/organisms/${organism.id}/inspect`}>
            <CardHeader>
              <EyeIcon className="h-8 w-8 text-primary mb-2" />
              <CardTitle>Inspect</CardTitle>
              <CardDescription>View internals and emerged code</CardDescription>
            </CardHeader>
          </Link>
        </Card>

        <Card className="hover:shadow-lg transition-shadow cursor-pointer" asChild>
          <Link href={`/organisms/${organism.id}/debug`}>
            <CardHeader>
              <BugIcon className="h-8 w-8 text-primary mb-2" />
              <CardTitle>Debug</CardTitle>
              <CardDescription>Monitor and debug performance</CardDescription>
            </CardHeader>
          </Link>
        </Card>
      </div>

      {/* Stats */}
      <Card>
        <CardHeader>
          <CardTitle>Organism Stats</CardTitle>
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
                <div className="text-xs text-muted-foreground">Total Cost</div>
                <div className="font-semibold font-mono">
                  {formatCost(organism.stats.total_cost)}
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
