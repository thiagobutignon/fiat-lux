import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ApiClient } from "@/lib/api-client";
import { formatCost, formatPercentage } from "@/lib/utils";
import {
  FlaskConicalIcon,
  DollarSignIcon,
  ActivityIcon,
  HeartPulseIcon,
  ArrowRightIcon,
} from "lucide-react";

export const dynamic = 'force-dynamic';

export default async function DashboardPage() {
  // Fetch data
  const [organisms, stats] = await Promise.all([
    ApiClient.listOrganisms().catch(() => []),
    ApiClient.getSystemStats().catch(() => ({
      total_organisms: 0,
      total_queries: 0,
      total_cost: 0,
      budget_limit: 100,
      health: 'healthy' as const,
      uptime: 0,
    })),
  ]);

  const recentOrganisms = organisms.slice(0, 5);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-muted-foreground">
          Monitor and manage your .glass organisms
        </p>
      </div>

      {/* Stats Overview */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Organisms</CardTitle>
            <FlaskConicalIcon className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.total_organisms}</div>
            <p className="text-xs text-muted-foreground">
              Active organisms running
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Cost</CardTitle>
            <DollarSignIcon className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatCost(stats.total_cost)}</div>
            <p className="text-xs text-muted-foreground">
              {formatPercentage(stats.total_cost / stats.budget_limit)} of ${stats.budget_limit} budget
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Queries</CardTitle>
            <ActivityIcon className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.total_queries}</div>
            <p className="text-xs text-muted-foreground">
              Total queries processed
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Health</CardTitle>
            <HeartPulseIcon className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold capitalize">{stats.health}</div>
            <p className="text-xs text-muted-foreground">
              System status
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Recent Organisms */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Recent Organisms</CardTitle>
            <Button asChild variant="ghost" size="sm">
              <Link href="/organisms">
                View All
                <ArrowRightIcon className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {recentOrganisms.length === 0 ? (
            <div className="text-center py-8">
              <FlaskConicalIcon className="mx-auto h-12 w-12 text-muted-foreground/50" />
              <h3 className="mt-4 text-lg font-semibold">No organisms yet</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Get started by uploading a .glass file
              </p>
              <Button asChild>
                <Link href="/organisms">Go to Organisms</Link>
              </Button>
            </div>
          ) : (
            <div className="space-y-4">
              {recentOrganisms.map((organism) => (
                <div
                  key={organism.id}
                  className="flex items-center justify-between p-4 border rounded-lg hover:bg-muted/50 transition-colors"
                >
                  <div className="flex items-center gap-4">
                    <FlaskConicalIcon className="h-8 w-8 text-primary" />
                    <div>
                      <h4 className="font-semibold">{organism.metadata.name}</h4>
                      <p className="text-sm text-muted-foreground">
                        {organism.metadata.specialization} • {organism.code.functions.length} functions • {formatPercentage(organism.metadata.maturity)} mature
                      </p>
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <Button asChild variant="outline" size="sm">
                      <Link href={`/organisms/${organism.id}/query`}>Query</Link>
                    </Button>
                    <Button asChild variant="ghost" size="sm">
                      <Link href={`/organisms/${organism.id}/inspect`}>Inspect</Link>
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Recent Activity */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Activity</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2 text-sm">
            <div className="flex items-center gap-2 text-muted-foreground">
              <div className="h-2 w-2 rounded-full bg-green-500" />
              <span>System is healthy and running</span>
            </div>
            <div className="flex items-center gap-2 text-muted-foreground">
              <div className="h-2 w-2 rounded-full bg-blue-500" />
              <span>{organisms.length} organisms active</span>
            </div>
            <div className="flex items-center gap-2 text-muted-foreground">
              <div className="h-2 w-2 rounded-full bg-yellow-500" />
              <span>Budget: {formatPercentage(stats.total_cost / stats.budget_limit)} used</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
