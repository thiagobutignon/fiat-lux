import { ApiClient } from "@/lib/api-client";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { ConstitutionalLogs } from "@/components/debug/ConstitutionalLogs";
import { LLMInspector } from "@/components/debug/LLMInspector";
import { CostTracker } from "@/components/debug/CostTracker";
import { PerformanceMetrics } from "@/components/debug/PerformanceMetrics";
import { EvolutionTracker } from "@/components/debug/EvolutionTracker";
import { notFound } from "next/navigation";
import {
  BugIcon,
  ScaleIcon,
  MessageSquareIcon,
  DollarSignIcon,
  ZapIcon,
  TrendingUpIcon,
} from "lucide-react";
import {
  ConstitutionalLog,
  LLMCall,
  PerformanceMetrics as Metrics,
  EvolutionData,
} from "@/lib/types";

export const dynamic = 'force-dynamic';

interface PageProps {
  params: { id: string };
}

// Simulated debug data (will be replaced with real data in DIA 5)
function generateMockDebugData(organismId: string) {
  const now = new Date();

  // Constitutional logs
  const constitutionalLogs: ConstitutionalLog[] = [
    {
      id: '1',
      timestamp: new Date(now.getTime() - 5000).toISOString(),
      organism_id: organismId,
      principle: 'epistemic_honesty',
      status: 'pass',
      details: 'Confidence 0.87 exceeds threshold of 0.7',
      context: { confidence: 0.87, threshold: 0.7 },
    },
    {
      id: '2',
      timestamp: new Date(now.getTime() - 4000).toISOString(),
      organism_id: organismId,
      principle: 'safety',
      status: 'pass',
      details: 'No harmful content detected',
    },
    {
      id: '3',
      timestamp: new Date(now.getTime() - 3000).toISOString(),
      organism_id: organismId,
      principle: 'cannot_diagnose',
      status: 'pass',
      details: 'Context-based validation passed (biology agent)',
      context: { agent_type: 'biology', query_type: 'information_seeking' },
    },
  ];

  // LLM calls
  const llmCalls: LLMCall[] = [
    {
      id: '1',
      timestamp: new Date(now.getTime() - 30000).toISOString(),
      organism_id: organismId,
      task_type: 'intent-analysis',
      model: 'claude-sonnet-4.5',
      tokens_in: 150,
      tokens_out: 50,
      cost: 0.02,
      latency_ms: 800,
      prompt: 'Analyze the intent of the following query:\n\n"What is the treatment efficacy?"',
      response: 'Intent: seek_clinical_information\nCategory: efficacy_query\nConfidence: 0.95',
      constitutional_status: 'pass',
    },
    {
      id: '2',
      timestamp: new Date(now.getTime() - 28000).toISOString(),
      organism_id: organismId,
      task_type: 'query-execution',
      model: 'claude-opus-4',
      tokens_in: 500,
      tokens_out: 200,
      cost: 0.05,
      latency_ms: 1200,
      prompt: 'Based on the knowledge base, answer: What is the treatment efficacy?',
      response: 'The treatment shows an efficacy rate of 75% in clinical trials...',
      constitutional_status: 'pass',
    },
  ];

  // Performance metrics
  const performanceMetrics: Metrics = {
    query_processing_ms: 26000,
    pattern_detection_ms: 0.3,
    knowledge_access_ms: 450,
    llm_latency_ms: 25000,
    total_ms: 26450,
  };

  return {
    constitutionalLogs,
    llmCalls,
    performanceMetrics,
  };
}

export default async function DebugPage({ params }: PageProps) {
  let organism;

  try {
    organism = await ApiClient.getOrganism(params.id);
  } catch (error) {
    notFound();
  }

  // Generate mock debug data
  const debugData = generateMockDebugData(params.id);

  // Generate evolution data from organism
  const evolutionData: EvolutionData = {
    organism_id: params.id,
    current_generation: organism.evolution.generation,
    current_fitness: organism.evolution.fitness,
    maturity: organism.metadata.maturity,
    versions: [
      {
        version: organism.metadata.version,
        generation: organism.evolution.generation,
        fitness: organism.evolution.fitness,
        traffic_percent: 99,
        deployed_at: organism.metadata.created_at,
        status: 'active',
      },
      {
        version: `${organism.metadata.version.split('.')[0]}.${parseInt(organism.metadata.version.split('.')[1]) - 1}.0`,
        generation: organism.evolution.generation - 1,
        fitness: organism.evolution.fitness - 0.05,
        traffic_percent: 1,
        deployed_at: new Date(new Date(organism.metadata.created_at).getTime() - 7 * 24 * 60 * 60 * 1000).toISOString(),
        status: 'canary',
      },
    ],
    canary_status: {
      current_version: organism.metadata.version,
      canary_version: `${organism.metadata.version.split('.')[0]}.${parseInt(organism.metadata.version.split('.')[1]) - 1}.0`,
      current_traffic: 99,
      canary_traffic: 1,
      status: 'monitoring',
    },
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 mb-2">
          <BugIcon className="h-6 w-6 text-primary" />
          <h1 className="text-3xl font-bold tracking-tight">Debug Tools</h1>
          <Badge>{organism.metadata.name}</Badge>
        </div>
        <p className="text-muted-foreground">
          Debug and monitor organism performance, costs, and constitutional compliance
        </p>
      </div>

      {/* Tabs */}
      <Tabs defaultValue="constitutional" className="space-y-4">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="constitutional" className="flex items-center gap-2">
            <ScaleIcon className="h-4 w-4" />
            Constitutional ({debugData.constitutionalLogs.length})
          </TabsTrigger>
          <TabsTrigger value="llm" className="flex items-center gap-2">
            <MessageSquareIcon className="h-4 w-4" />
            LLM Calls ({debugData.llmCalls.length})
          </TabsTrigger>
          <TabsTrigger value="cost" className="flex items-center gap-2">
            <DollarSignIcon className="h-4 w-4" />
            Costs
          </TabsTrigger>
          <TabsTrigger value="performance" className="flex items-center gap-2">
            <ZapIcon className="h-4 w-4" />
            Performance
          </TabsTrigger>
          <TabsTrigger value="evolution" className="flex items-center gap-2">
            <TrendingUpIcon className="h-4 w-4" />
            Evolution
          </TabsTrigger>
        </TabsList>

        {/* Constitutional Tab */}
        <TabsContent value="constitutional">
          <ConstitutionalLogs logs={debugData.constitutionalLogs} />
        </TabsContent>

        {/* LLM Calls Tab */}
        <TabsContent value="llm">
          <LLMInspector calls={debugData.llmCalls} />
        </TabsContent>

        {/* Cost Tab */}
        <TabsContent value="cost">
          <CostTracker organism={organism} budgetLimit={10.0} />
        </TabsContent>

        {/* Performance Tab */}
        <TabsContent value="performance">
          <PerformanceMetrics metrics={debugData.performanceMetrics} />
        </TabsContent>

        {/* Evolution Tab */}
        <TabsContent value="evolution">
          <EvolutionTracker data={evolutionData} />
        </TabsContent>
      </Tabs>
    </div>
  );
}
