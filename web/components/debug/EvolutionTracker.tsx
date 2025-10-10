"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { EvolutionData } from "@/lib/types";
import { formatPercentage } from "@/lib/utils";
import {
  TrendingUpIcon,
  GitBranchIcon,
  ActivityIcon,
  CheckCircle2Icon,
  ClockIcon,
  AlertCircleIcon,
} from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

interface EvolutionTrackerProps {
  data: EvolutionData;
}

export function EvolutionTracker({ data }: EvolutionTrackerProps) {
  const activeVersion = data.versions.find(v => v.status === 'active');
  const canaryVersion = data.versions.find(v => v.status === 'canary');
  const oldButGold = data.versions.filter(v => v.status === 'old-but-gold');

  // Generate fitness trajectory data
  const fitnessData = data.versions
    .sort((a, b) => a.generation - b.generation)
    .map(v => ({
      generation: v.generation,
      fitness: v.fitness,
    }));

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-green-500';
      case 'canary':
        return 'bg-yellow-500';
      case 'old-but-gold':
        return 'bg-blue-500';
      case 'deprecated':
        return 'bg-gray-500';
      default:
        return 'bg-gray-500';
    }
  };

  const getCanaryStatusColor = (status: string) => {
    switch (status) {
      case 'monitoring':
        return 'text-blue-500';
      case 'promoting':
        return 'text-green-500';
      case 'rolling_back':
        return 'text-red-500';
      default:
        return 'text-gray-500';
    }
  };

  return (
    <div className="space-y-6">
      {/* Current Status */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="p-4">
            <CardDescription className="text-xs">Generation</CardDescription>
            <CardTitle className="text-2xl">{data.current_generation}</CardTitle>
          </CardHeader>
        </Card>

        <Card>
          <CardHeader className="p-4">
            <CardDescription className="text-xs">Fitness</CardDescription>
            <CardTitle className="text-2xl">{data.current_fitness.toFixed(2)}</CardTitle>
          </CardHeader>
        </Card>

        <Card>
          <CardHeader className="p-4">
            <CardDescription className="text-xs">Maturity</CardDescription>
            <CardTitle className="text-2xl">{formatPercentage(data.maturity)}</CardTitle>
          </CardHeader>
        </Card>

        <Card>
          <CardHeader className="p-4">
            <CardDescription className="text-xs">Total Versions</CardDescription>
            <CardTitle className="text-2xl">{data.versions.length}</CardTitle>
          </CardHeader>
        </Card>
      </div>

      {/* Fitness Trajectory */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <TrendingUpIcon className="h-5 w-5 text-primary" />
            <CardTitle>Fitness Trajectory</CardTitle>
          </div>
          <CardDescription>
            Evolution of fitness across generations
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={fitnessData}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis dataKey="generation" label={{ value: 'Generation', position: 'insideBottom', offset: -5 }} />
                <YAxis domain={[0, 1]} label={{ value: 'Fitness', angle: -90, position: 'insideLeft' }} />
                <Tooltip
                  content={({ payload }) => {
                    if (!payload || !payload[0]) return null;
                    const data = payload[0].payload;
                    return (
                      <div className="bg-card border rounded-lg p-2 shadow-lg">
                        <p className="text-xs font-semibold">Generation {data.generation}</p>
                        <p className="text-xs text-muted-foreground">
                          Fitness: {data.fitness.toFixed(2)}
                        </p>
                      </div>
                    );
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="fitness"
                  stroke="hsl(var(--primary))"
                  strokeWidth={2}
                  dot={{ fill: "hsl(var(--primary))", r: 4 }}
                  activeDot={{ r: 6 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Canary Status */}
      {data.canary_status && (
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <GitBranchIcon className="h-5 w-5 text-primary" />
              <CardTitle>Canary Deployment</CardTitle>
            </div>
            <CardDescription>
              Current canary deployment status
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold">Status</span>
              <Badge className={getCanaryStatusColor(data.canary_status.status)}>
                {data.canary_status.status.replace(/_/g, ' ').toUpperCase()}
              </Badge>
            </div>

            <div className="space-y-3">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm">
                    <span className="font-semibold">{data.canary_status.current_version}</span>
                    <span className="text-muted-foreground"> (Current)</span>
                  </span>
                  <span className="text-sm font-semibold">
                    {data.canary_status.current_traffic}%
                  </span>
                </div>
                <Progress value={data.canary_status.current_traffic} className="h-2" />
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm">
                    <span className="font-semibold">{data.canary_status.canary_version}</span>
                    <span className="text-muted-foreground"> (Canary)</span>
                  </span>
                  <span className="text-sm font-semibold">
                    {data.canary_status.canary_traffic}%
                  </span>
                </div>
                <Progress value={data.canary_status.canary_traffic} className="h-2 [&>div]:bg-yellow-500" />
              </div>
            </div>

            {data.canary_status.status === 'monitoring' && (
              <div className="flex items-start gap-2 p-3 bg-blue-500/10 rounded-lg">
                <ActivityIcon className="h-5 w-5 text-blue-500 flex-shrink-0 mt-0.5" />
                <div className="flex-1">
                  <p className="text-sm font-semibold text-blue-500">Monitoring Canary</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    Canary version {data.canary_status.canary_version} is receiving {data.canary_status.canary_traffic}% of traffic.
                    Will auto-promote if performance is stable.
                  </p>
                </div>
              </div>
            )}

            {data.canary_status.status === 'rolling_back' && (
              <div className="flex items-start gap-2 p-3 bg-red-500/10 rounded-lg">
                <AlertCircleIcon className="h-5 w-5 text-red-500 flex-shrink-0 mt-0.5" />
                <div className="flex-1">
                  <p className="text-sm font-semibold text-red-500">Rolling Back</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    Issues detected with canary version. Rolling back to {data.canary_status.current_version}.
                  </p>
                </div>
              </div>
            )}

            <div className="flex gap-2 pt-2 border-t">
              <Button variant="outline" size="sm" className="flex-1">
                View GVCS Status
              </Button>
              <Button variant="destructive" size="sm" className="flex-1">
                Rollback Canary
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Version History */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <ClockIcon className="h-5 w-5 text-primary" />
            <CardTitle>Version History</CardTitle>
          </div>
          <CardDescription>
            All versions of this organism
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {data.versions.map((version) => (
              <Card key={version.version}>
                <CardHeader className="p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className={`w-2 h-2 rounded-full ${getStatusColor(version.status)}`} />
                      <div>
                        <div className="flex items-center gap-2 mb-1">
                          <span className="font-semibold font-mono">{version.version}</span>
                          <Badge
                            variant={version.status === 'active' ? 'default' : 'secondary'}
                            className={version.status === 'active' ? '' : 'bg-muted'}
                          >
                            {version.status === 'old-but-gold' ? 'Old but Gold' : version.status}
                          </Badge>
                        </div>
                        <div className="flex items-center gap-4 text-xs text-muted-foreground">
                          <span>Gen {version.generation}</span>
                          <span>Fitness: {version.fitness.toFixed(2)}</span>
                          <span>{version.traffic_percent}% traffic</span>
                          <span>{new Date(version.deployed_at).toLocaleDateString()}</span>
                        </div>
                      </div>
                    </div>
                    {version.status === 'active' && (
                      <CheckCircle2Icon className="h-5 w-5 text-green-500" />
                    )}
                  </div>
                </CardHeader>
              </Card>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Old but Gold */}
      {oldButGold.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Old but Gold 🏆</CardTitle>
            <CardDescription>
              Previous versions maintained for specific use cases
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {oldButGold.map((version) => (
                <div
                  key={version.version}
                  className="flex items-center justify-between p-3 bg-blue-500/10 rounded-lg border border-blue-500/20"
                >
                  <div>
                    <div className="font-semibold font-mono">{version.version}</div>
                    <div className="text-xs text-muted-foreground">
                      Fitness: {version.fitness.toFixed(2)} • {version.traffic_percent}% traffic
                    </div>
                  </div>
                  <Badge className="bg-blue-500">
                    Old but Gold
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
