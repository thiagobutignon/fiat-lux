"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { GlassOrganism } from "@/lib/types";
import { formatCurrency } from "@/lib/utils";
import { DollarSignIcon, TrendingUpIcon, AlertCircleIcon } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";

interface CostTrackerProps {
  organism: GlassOrganism;
  budgetLimit?: number;
}

export function CostTracker({ organism, budgetLimit = 10.0 }: CostTrackerProps) {
  const totalCost = organism.stats.total_cost;
  const queriesCount = organism.stats.queries_count;
  const avgCostPerQuery = queriesCount > 0 ? totalCost / queriesCount : 0;
  const budgetUsedPercent = (totalCost / budgetLimit) * 100;

  // Simulated cost breakdown by task type
  const costBreakdown = [
    { task: 'Query Execution', cost: totalCost * 0.5 },
    { task: 'Intent Analysis', cost: totalCost * 0.2 },
    { task: 'Code Synthesis', cost: totalCost * 0.15 },
    { task: 'Pattern Detection', cost: totalCost * 0.1 },
    { task: 'Knowledge Access', cost: totalCost * 0.05 },
  ];

  const getBudgetStatus = () => {
    if (budgetUsedPercent >= 90) return { status: 'critical', color: 'text-red-500', bg: 'bg-red-500' };
    if (budgetUsedPercent >= 75) return { status: 'warning', color: 'text-yellow-500', bg: 'bg-yellow-500' };
    return { status: 'healthy', color: 'text-green-500', bg: 'bg-green-500' };
  };

  const budgetStatus = getBudgetStatus();

  return (
    <div className="space-y-6">
      {/* Summary Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="p-4">
            <CardDescription className="text-xs">Total Spent</CardDescription>
            <CardTitle className="text-2xl">{formatCurrency(totalCost)}</CardTitle>
          </CardHeader>
        </Card>

        <Card>
          <CardHeader className="p-4">
            <CardDescription className="text-xs">Budget Remaining</CardDescription>
            <CardTitle className="text-2xl">
              {formatCurrency(budgetLimit - totalCost)}
            </CardTitle>
          </CardHeader>
        </Card>

        <Card>
          <CardHeader className="p-4">
            <CardDescription className="text-xs">Avg Per Query</CardDescription>
            <CardTitle className="text-2xl">{formatCurrency(avgCostPerQuery)}</CardTitle>
          </CardHeader>
        </Card>

        <Card>
          <CardHeader className="p-4">
            <CardDescription className="text-xs">Total Queries</CardDescription>
            <CardTitle className="text-2xl">{queriesCount}</CardTitle>
          </CardHeader>
        </Card>
      </div>

      {/* Budget Status */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <DollarSignIcon className="h-5 w-5 text-primary" />
            <CardTitle>Budget Status</CardTitle>
          </div>
          <CardDescription>
            Current budget usage and limits
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-semibold">
                Budget Usage
              </span>
              <span className={`text-sm font-semibold ${budgetStatus.color}`}>
                {budgetUsedPercent.toFixed(1)}%
              </span>
            </div>
            <Progress value={budgetUsedPercent} className="h-3" />
            <div className="flex items-center justify-between mt-2 text-xs text-muted-foreground">
              <span>{formatCurrency(totalCost)} spent</span>
              <span>{formatCurrency(budgetLimit)} limit</span>
            </div>
          </div>

          {budgetUsedPercent >= 75 && (
            <div className={`flex items-start gap-2 p-3 rounded-lg ${
              budgetUsedPercent >= 90 ? 'bg-red-500/10' : 'bg-yellow-500/10'
            }`}>
              <AlertCircleIcon className={`h-5 w-5 ${budgetStatus.color} flex-shrink-0 mt-0.5`} />
              <div className="flex-1">
                <p className={`text-sm font-semibold ${budgetStatus.color}`}>
                  {budgetUsedPercent >= 90 ? 'Budget Critical!' : 'Budget Warning'}
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  {budgetUsedPercent >= 90
                    ? `You've used ${budgetUsedPercent.toFixed(0)}% of your budget. Consider pausing queries or increasing the limit.`
                    : `You've used ${budgetUsedPercent.toFixed(0)}% of your budget. Monitor usage carefully.`
                  }
                </p>
              </div>
            </div>
          )}

          <div className="pt-4 border-t">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <div className="text-muted-foreground mb-1">Budget Limit</div>
                <div className="font-semibold">{formatCurrency(budgetLimit)}</div>
              </div>
              <div>
                <div className="text-muted-foreground mb-1">Remaining</div>
                <div className={`font-semibold ${budgetStatus.color}`}>
                  {formatCurrency(Math.max(0, budgetLimit - totalCost))}
                </div>
              </div>
              <div>
                <div className="text-muted-foreground mb-1">Projected Total</div>
                <div className="font-semibold">
                  {formatCurrency(totalCost + avgCostPerQuery * 10)} <span className="text-xs text-muted-foreground">(+10 queries)</span>
                </div>
              </div>
              <div>
                <div className="text-muted-foreground mb-1">Status</div>
                <div className={`font-semibold capitalize ${budgetStatus.color}`}>
                  {budgetStatus.status}
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Cost Breakdown */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <TrendingUpIcon className="h-5 w-5 text-primary" />
            <CardTitle>Cost Breakdown by Task Type</CardTitle>
          </div>
          <CardDescription>
            How costs are distributed across different tasks
          </CardDescription>
        </CardHeader>
        <CardContent>
          {totalCost > 0 ? (
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={costBreakdown} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis type="number" />
                  <YAxis dataKey="task" type="category" width={120} />
                  <Tooltip
                    content={({ payload }) => {
                      if (!payload || !payload[0]) return null;
                      const data = payload[0].payload;
                      return (
                        <div className="bg-card border rounded-lg p-2 shadow-lg">
                          <p className="text-xs font-semibold">{data.task}</p>
                          <p className="text-xs text-muted-foreground">
                            Cost: {formatCurrency(data.cost)}
                          </p>
                          <p className="text-xs text-muted-foreground">
                            {((data.cost / totalCost) * 100).toFixed(1)}% of total
                          </p>
                        </div>
                      );
                    }}
                  />
                  <Bar dataKey="cost" fill="hsl(var(--primary))" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="flex items-center justify-center h-80">
              <div className="text-center">
                <DollarSignIcon className="h-12 w-12 text-muted-foreground/50 mx-auto mb-4" />
                <p className="text-muted-foreground">No cost data yet</p>
                <p className="text-sm text-muted-foreground mt-2">
                  Execute queries to see cost breakdown
                </p>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
