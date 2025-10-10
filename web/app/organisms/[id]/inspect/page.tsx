import { ApiClient } from "@/lib/api-client";
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { FunctionViewer } from "@/components/inspector/FunctionViewer";
import { KnowledgeGraph } from "@/components/inspector/KnowledgeGraph";
import { PatternList } from "@/components/inspector/PatternList";
import { formatPercentage } from "@/lib/utils";
import { notFound } from "next/navigation";
import {
  FlaskConicalIcon,
  CodeIcon,
  NetworkIcon,
  SparklesIcon,
} from "lucide-react";

export const dynamic = 'force-dynamic';

interface PageProps {
  params: { id: string };
}

export default async function InspectPage({ params }: PageProps) {
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
          <h1 className="text-3xl font-bold tracking-tight">Glass Box Inspector</h1>
          <Badge>{organism.metadata.stage}</Badge>
        </div>
        <p className="text-muted-foreground">
          Inspect {organism.metadata.name} internals - 100% transparent
        </p>
      </div>

      {/* Organism Summary */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>{organism.metadata.name}</CardTitle>
              <CardDescription className="mt-1">
                {organism.metadata.specialization} • {formatPercentage(organism.metadata.maturity)} mature • Generation {organism.evolution.generation}
              </CardDescription>
            </div>
            <div className="flex gap-4 text-sm">
              <div className="text-center">
                <div className="text-2xl font-bold text-primary">
                  {organism.code.functions.length}
                </div>
                <div className="text-muted-foreground">Functions</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-primary">
                  {organism.knowledge.papers}
                </div>
                <div className="text-muted-foreground">Papers</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-primary">
                  {organism.knowledge.patterns?.length || 0}
                </div>
                <div className="text-muted-foreground">Patterns</div>
              </div>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Tabs for Different Views */}
      <Tabs defaultValue="functions" className="space-y-4">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="functions" className="flex items-center gap-2">
            <CodeIcon className="h-4 w-4" />
            Functions ({organism.code.functions.length})
          </TabsTrigger>
          <TabsTrigger value="knowledge" className="flex items-center gap-2">
            <NetworkIcon className="h-4 w-4" />
            Knowledge Graph
          </TabsTrigger>
          <TabsTrigger value="patterns" className="flex items-center gap-2">
            <SparklesIcon className="h-4 w-4" />
            Patterns ({organism.knowledge.patterns?.length || 0})
          </TabsTrigger>
        </TabsList>

        {/* Functions Tab */}
        <TabsContent value="functions">
          <FunctionViewer functions={organism.code.functions} />
        </TabsContent>

        {/* Knowledge Graph Tab */}
        <TabsContent value="knowledge">
          <KnowledgeGraph
            papers={organism.knowledge.papers}
            connections={organism.knowledge.connections}
            clusters={organism.knowledge.clusters}
            embeddings_dim={organism.knowledge.embeddings_dim}
          />
        </TabsContent>

        {/* Patterns Tab */}
        <TabsContent value="patterns">
          <PatternList patterns={organism.knowledge.patterns || []} />
        </TabsContent>
      </Tabs>
    </div>
  );
}
