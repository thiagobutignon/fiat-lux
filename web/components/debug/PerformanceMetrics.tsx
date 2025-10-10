"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { PerformanceMetrics as Metrics } from "@/lib/types";
import { ZapIcon, CheckCircle2Icon, AlertCircleIcon } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

interface PerformanceMetricsProps {
  metrics: Metrics;
  target?: {
    query_processing_ms: number;
    pattern_detection_ms: number;
    knowledge_access_ms: number;
    llm_latency_ms: number;
  };
}

export function PerformanceMetrics({
  metrics,
  target = {
    query_processing_ms: 30000, // 30s
    pattern_detection_ms: 0.5, // 0.5ms
    knowledge_access_ms: 500, // 500ms
    llm_latency_ms: 25000, // 25s
  }
}: PerformanceMetricsProps) {

  const metricsData = [
    {
      name: 'Pattern Detection',
      actual: metrics.pattern_detection_ms,
      target: target.pattern_detection_ms,
      unit: 'ms'
    },
    {
      name: 'Knowledge Access',
      actual: metrics.knowledge_access_ms,
      target: target.knowledge_access_ms,
      unit: 'ms'
    },
    {
      name: 'LLM Latency',
      actual: metrics.llm_latency_ms,
      target: target.llm_latency_ms,
      unit: 'ms'
    },
    {
      name: 'Total Processing',
      actual: metrics.query_processing_ms,
      target: target.query_processing_ms,
      unit: 'ms'
    },
  ];

  const getPerformanceStatus = (actual: number, target: number) => {
    const ratio = actual / target;
    if (ratio <= 0.8) return { status: 'excellent', color: 'text-green-500', icon: CheckCircle2Icon };
    if (ratio <= 1.0) return { status: 'good', color: 'text-blue-500', icon: CheckCircle2Icon };
    if (ratio <= 1.5) return { status: 'acceptable', color: 'text-yellow-500', icon: AlertCircleIcon };
    return { status: 'slow', color: 'text-red-500', icon: AlertCircleIcon };
  };

  const formatMetricValue = (value: number, unit: string) => {
    if (value < 1) return `${(value * 1000).toFixed(2)}μs`;
    if (value < 1000) return `${value.toFixed(2)}${unit}`;
    if (value < 60000) return `${(value / 1000).toFixed(2)}s`;
    return `${(value / 60000).toFixed(2)}m`;
  };

  const totalStatus = getPerformanceStatus(metrics.total_ms, target.query_processing_ms);

  return (
    <div className="space-y-6">
      {/* Overall Status */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <ZapIcon className="h-5 w-5 text-primary" />
            <CardTitle>Performance Overview</CardTitle>
          </div>
          <CardDescription>
            System performance metrics vs targets
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between mb-4">
            <div>
              <div className="text-sm text-muted-foreground mb-1">Overall Status</div>
              <div className="flex items-center gap-2">
                <totalStatus.icon className={`h-5 w-5 ${totalStatus.color}`} />
                <span className={`text-2xl font-bold capitalize ${totalStatus.color}`}>
                  {totalStatus.status}
                </span>
              </div>
            </div>
            <div>
              <div className="text-sm text-muted-foreground mb-1">Total Time</div>
              <div className="text-2xl font-bold">
                {formatMetricValue(metrics.total_ms, 'ms')}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Detailed Metrics */}
      <Card>
        <CardHeader>
          <CardTitle>Metric Breakdown</CardTitle>
          <CardDescription>
            Detailed performance metrics for each component
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {metricsData.map((metric) => {
            const status = getPerformanceStatus(metric.actual, metric.target);
            const percentOfTarget = (metric.actual / metric.target) * 100;

            return (
              <div key={metric.name} className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold">{metric.name}</span>
                    <Badge variant={status.status === 'excellent' || status.status === 'good' ? 'default' : 'secondary'}>
                      {status.status}
                    </Badge>
                  </div>
                  <div className="text-sm">
                    <span className={`font-semibold ${status.color}`}>
                      {formatMetricValue(metric.actual, metric.unit)}
                    </span>
                    <span className="text-muted-foreground"> / {formatMetricValue(metric.target, metric.unit)}</span>
                  </div>
                </div>
                <Progress value={Math.min(percentOfTarget, 100)} className="h-2" />
                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <span>{percentOfTarget.toFixed(0)}% of target</span>
                  {metric.actual <= metric.target && (
                    <span className="text-green-500">✓ Under target</span>
                  )}
                  {metric.actual > metric.target && (
                    <span className="text-red-500">
                      {formatMetricValue(metric.actual - metric.target, metric.unit)} over
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </CardContent>
      </Card>

      {/* Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Performance Visualization</CardTitle>
          <CardDescription>
            Actual vs target performance metrics
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={metricsData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis type="number" />
                <YAxis dataKey="name" type="category" width={130} />
                <Tooltip
                  content={({ payload }) => {
                    if (!payload || !payload[0]) return null;
                    const data = payload[0].payload;
                    const status = getPerformanceStatus(data.actual, data.target);
                    return (
                      <div className="bg-card border rounded-lg p-3 shadow-lg">
                        <p className="text-xs font-semibold mb-2">{data.name}</p>
                        <div className="space-y-1 text-xs">
                          <p className={status.color}>
                            Actual: {formatMetricValue(data.actual, data.unit)}
                          </p>
                          <p className="text-muted-foreground">
                            Target: {formatMetricValue(data.target, data.unit)}
                          </p>
                          <p className={`font-semibold capitalize ${status.color}`}>
                            Status: {status.status}
                          </p>
                        </div>
                      </div>
                    );
                  }}
                />
                <Bar dataKey="actual" fill="hsl(var(--primary))" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Bottlenecks */}
      <Card>
        <CardHeader>
          <CardTitle>Bottleneck Analysis</CardTitle>
          <CardDescription>
            Identify performance bottlenecks
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
              <span className="font-semibold">LLM Latency</span>
              <span className="text-muted-foreground">
                {((metrics.llm_latency_ms / metrics.total_ms) * 100).toFixed(0)}% of total time
              </span>
            </div>
            <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
              <span className="font-semibold">Knowledge Access</span>
              <span className="text-muted-foreground">
                {((metrics.knowledge_access_ms / metrics.total_ms) * 100).toFixed(0)}% of total time
              </span>
            </div>
            <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
              <span className="font-semibold">Pattern Detection</span>
              <span className="text-muted-foreground">
                {((metrics.pattern_detection_ms / metrics.total_ms) * 100).toFixed(2)}% of total time
              </span>
            </div>
          </div>

          {metrics.llm_latency_ms / metrics.total_ms > 0.8 && (
            <div className="mt-4 p-3 bg-yellow-500/10 rounded-lg border border-yellow-500/20">
              <p className="text-sm font-semibold text-yellow-500 mb-1">
                LLM Latency Bottleneck Detected
              </p>
              <p className="text-xs text-muted-foreground">
                LLM calls are taking {((metrics.llm_latency_ms / metrics.total_ms) * 100).toFixed(0)}% of total time.
                This is expected as LLM calls are external. Consider caching or using faster models.
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
