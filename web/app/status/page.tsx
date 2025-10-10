import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { checkAllNodesHealth, getIntegrationStatus } from "@/lib/integrations";
import { Progress } from "@/components/ui/progress";
import {
  ActivityIcon,
  CheckCircle2Icon,
  XCircleIcon,
  AlertCircleIcon,
  ServerIcon,
  NetworkIcon,
} from "lucide-react";

export const dynamic = 'force-dynamic';

export default async function StatusPage() {
  // Get integration status
  const integrationStatus = getIntegrationStatus();

  // Get health of all nodes
  const nodesHealth = await checkAllNodesHealth();

  const getStatusIcon = (available: boolean) => {
    return available ? (
      <CheckCircle2Icon className="h-5 w-5 text-green-500" />
    ) : (
      <XCircleIcon className="h-5 w-5 text-red-500" />
    );
  };

  const getStatusBadge = (available: boolean, status: string) => {
    if (available && status === 'active') {
      return <Badge className="bg-green-500">Connected</Badge>;
    }
    if (available && status === 'disabled') {
      return <Badge variant="secondary">Stub Mode</Badge>;
    }
    return <Badge variant="destructive">Offline</Badge>;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 mb-2">
          <ActivityIcon className="h-6 w-6 text-primary" />
          <h1 className="text-3xl font-bold tracking-tight">System Status</h1>
        </div>
        <p className="text-muted-foreground">
          Monitor integration status and health of all 5 Chomsky nodes
        </p>
      </div>

      {/* Overall Status */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <ServerIcon className="h-5 w-5 text-primary" />
            <CardTitle>Integration Progress</CardTitle>
          </div>
          <CardDescription>
            Overall integration readiness across all nodes
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold">Nodes Connected</span>
            <span className="text-2xl font-bold">
              {integrationStatus.available_count}/{integrationStatus.total_count}
            </span>
          </div>

          <Progress value={integrationStatus.progress_percent} className="h-3" />

          <div className="flex items-center justify-between text-sm text-muted-foreground">
            <span>{integrationStatus.progress_percent.toFixed(0)}% Complete</span>
            {integrationStatus.ready ? (
              <span className="text-green-500 font-semibold">✓ All Nodes Ready</span>
            ) : (
              <span className="text-yellow-500">⚠ Integration Pending</span>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Node Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {/* ROXO */}
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {getStatusIcon(nodesHealth.roxo.available)}
                <CardTitle className="text-lg">🟣 ROXO</CardTitle>
              </div>
              {getStatusBadge(nodesHealth.roxo.available, nodesHealth.roxo.status)}
            </div>
            <CardDescription>Core .glass organisms</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="text-sm">
              <span className="text-muted-foreground">Status: </span>
              <span className="font-semibold capitalize">{nodesHealth.roxo.status}</span>
            </div>
            <div className="text-sm">
              <span className="text-muted-foreground">Version: </span>
              <span className="font-mono">{nodesHealth.roxo.version}</span>
            </div>
            <div className="text-xs text-muted-foreground mt-2">
              GlassRuntime, query execution, code emergence, pattern detection
            </div>
          </CardContent>
        </Card>

        {/* VERDE */}
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {getStatusIcon(nodesHealth.verde.available)}
                <CardTitle className="text-lg">🟢 VERDE</CardTitle>
              </div>
              {getStatusBadge(nodesHealth.verde.available, nodesHealth.verde.status)}
            </div>
            <CardDescription>Genetic Version Control</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="text-sm">
              <span className="text-muted-foreground">Status: </span>
              <span className="font-semibold capitalize">{nodesHealth.verde.status}</span>
            </div>
            <div className="text-sm">
              <span className="text-muted-foreground">Version: </span>
              <span className="font-mono">{nodesHealth.verde.version}</span>
            </div>
            <div className="text-xs text-muted-foreground mt-2">
              GVCS, canary deployment, fitness tracking, old-but-gold
            </div>
          </CardContent>
        </Card>

        {/* VERMELHO */}
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {getStatusIcon(nodesHealth.vermelho.available)}
                <CardTitle className="text-lg">🔴 VERMELHO</CardTitle>
              </div>
              {getStatusBadge(nodesHealth.vermelho.available, nodesHealth.vermelho.status)}
            </div>
            <CardDescription>Security & Behavioral</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="text-sm">
              <span className="text-muted-foreground">Status: </span>
              <span className="font-semibold capitalize">{nodesHealth.vermelho.status}</span>
            </div>
            <div className="text-sm">
              <span className="text-muted-foreground">Version: </span>
              <span className="font-mono">{nodesHealth.vermelho.version}</span>
            </div>
            <div className="text-xs text-muted-foreground mt-2">
              Duress detection, behavioral profiling, linguistic fingerprinting
            </div>
          </CardContent>
        </Card>

        {/* CINZA */}
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {getStatusIcon(nodesHealth.cinza.available)}
                <CardTitle className="text-lg">🩶 CINZA</CardTitle>
              </div>
              {getStatusBadge(nodesHealth.cinza.available, nodesHealth.cinza.status)}
            </div>
            <CardDescription>Cognitive OS</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="text-sm">
              <span className="text-muted-foreground">Status: </span>
              <span className="font-semibold capitalize">{nodesHealth.cinza.status}</span>
            </div>
            <div className="text-sm">
              <span className="text-muted-foreground">Version: </span>
              <span className="font-mono">{nodesHealth.cinza.version}</span>
            </div>
            <div className="text-xs text-muted-foreground mt-2">
              Manipulation detection, Dark Tetrad, cognitive biases, self-surgery
            </div>
          </CardContent>
        </Card>

        {/* LARANJA */}
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {getStatusIcon(nodesHealth.laranja.available)}
                <CardTitle className="text-lg">🟠 LARANJA</CardTitle>
              </div>
              {getStatusBadge(nodesHealth.laranja.available, nodesHealth.laranja.status)}
            </div>
            <CardDescription>O(1) Database</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="text-sm">
              <span className="text-muted-foreground">Status: </span>
              <span className="font-semibold capitalize">{nodesHealth.laranja.status}</span>
            </div>
            <div className="text-sm">
              <span className="text-muted-foreground">Version: </span>
              <span className="font-mono">{nodesHealth.laranja.version}</span>
            </div>
            {nodesHealth.laranja.performance_us !== undefined && (
              <div className="text-sm">
                <span className="text-muted-foreground">Performance: </span>
                <span className="font-mono text-green-500">{nodesHealth.laranja.performance_us}μs</span>
              </div>
            )}
            <div className="text-xs text-muted-foreground mt-2">
              .sqlo database, episodic memory, RBAC, consolidation optimizer
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Integration Instructions */}
      {!integrationStatus.ready && (
        <Card className="border-yellow-500/20 bg-yellow-500/5">
          <CardHeader>
            <div className="flex items-center gap-2">
              <AlertCircleIcon className="h-5 w-5 text-yellow-500" />
              <CardTitle>Integration Pending</CardTitle>
            </div>
            <CardDescription>
              Some nodes are not yet connected. Follow the instructions below to enable them.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="text-sm">
              <p className="font-semibold mb-2">To enable a node:</p>
              <ol className="list-decimal list-inside space-y-1 text-muted-foreground">
                <li>Read <code className="bg-muted px-1 rounded">/lib/integrations/README.md</code></li>
                <li>Open the integration file for your node in <code className="bg-muted px-1 rounded">/lib/integrations/</code></li>
                <li>Set <code className="bg-muted px-1 rounded">*_ENABLED = true</code></li>
                <li>Configure <code className="bg-muted px-1 rounded">*_API_URL</code> in <code className="bg-muted px-1 rounded">.env.local</code></li>
                <li>Replace <code className="bg-muted px-1 rounded">// TODO: Real implementation</code> with actual API calls</li>
                <li>Refresh this page to see the updated status</li>
              </ol>
            </div>
          </CardContent>
        </Card>
      )}

      {/* All Ready Celebration */}
      {integrationStatus.ready && (
        <Card className="border-green-500/20 bg-green-500/5">
          <CardHeader>
            <div className="flex items-center gap-2">
              <CheckCircle2Icon className="h-5 w-5 text-green-500" />
              <CardTitle className="text-green-500">All Systems Operational! 🎊</CardTitle>
            </div>
            <CardDescription>
              All 5 nodes are connected and ready. The Chomsky AGI system is fully operational.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-sm text-muted-foreground">
              You can now use all features of the DevTools Dashboard with real data from all integrated nodes.
            </div>
          </CardContent>
        </Card>
      )}

      {/* Integration Details */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <NetworkIcon className="h-5 w-5 text-primary" />
            <CardTitle>Integration Details</CardTitle>
          </div>
          <CardDescription>
            Available integration functions by node
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {integrationStatus.nodes.map((node) => (
              <div
                key={node.name}
                className="flex items-center justify-between p-3 bg-muted/50 rounded-lg"
              >
                <div className="flex items-center gap-3">
                  <span className="text-2xl">{node.color}</span>
                  <div>
                    <div className="font-semibold">{node.name}</div>
                    <div className="text-xs text-muted-foreground">
                      {node.name === 'ROXO' && '13 functions: query, patterns, emergence'}
                      {node.name === 'VERDE' && '15 functions: versions, canary, fitness'}
                      {node.name === 'VERMELHO' && '12 functions: duress, behavioral, emotional'}
                      {node.name === 'CINZA' && '15 functions: manipulation, Dark Tetrad, biases'}
                      {node.name === 'LARANJA' && '21 functions: database, RBAC, consolidation'}
                    </div>
                  </div>
                </div>
                <Badge variant={node.available ? "default" : "secondary"}>
                  {node.available ? "Ready" : "Stub"}
                </Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
