"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ConstitutionalLog } from "@/lib/types";
import { CheckCircle2Icon, XCircleIcon, AlertTriangleIcon, ScaleIcon, SearchIcon } from "lucide-react";

interface ConstitutionalLogsProps {
  logs: ConstitutionalLog[];
}

export function ConstitutionalLogs({ logs }: ConstitutionalLogsProps) {
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [principleFilter, setPrincipleFilter] = useState<string>("all");

  // Get unique principles
  const uniquePrinciples = Array.from(new Set(logs.map(log => log.principle)));

  // Filter logs
  const filteredLogs = logs.filter(log => {
    const matchesSearch = log.details.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         log.principle.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = statusFilter === "all" || log.status === statusFilter;
    const matchesPrinciple = principleFilter === "all" || log.principle === principleFilter;
    return matchesSearch && matchesStatus && matchesPrinciple;
  });

  // Stats
  const passCount = logs.filter(l => l.status === 'pass').length;
  const failCount = logs.filter(l => l.status === 'fail').length;
  const warningCount = logs.filter(l => l.status === 'warning').length;

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pass':
        return <CheckCircle2Icon className="h-5 w-5 text-green-500" />;
      case 'fail':
        return <XCircleIcon className="h-5 w-5 text-red-500" />;
      case 'warning':
        return <AlertTriangleIcon className="h-5 w-5 text-yellow-500" />;
      default:
        return null;
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'pass':
        return <Badge variant="default" className="bg-green-500">PASS</Badge>;
      case 'fail':
        return <Badge variant="destructive">FAIL</Badge>;
      case 'warning':
        return <Badge variant="secondary" className="bg-yellow-500">WARNING</Badge>;
      default:
        return null;
    }
  };

  if (logs.length === 0) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-12">
          <div className="text-center">
            <ScaleIcon className="h-12 w-12 text-muted-foreground/50 mx-auto mb-4" />
            <p className="text-muted-foreground">No constitutional logs yet</p>
            <p className="text-sm text-muted-foreground mt-2">
              Logs will appear after queries are executed
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Stats */}
      <div className="grid grid-cols-3 gap-4">
        <Card>
          <CardHeader className="p-4">
            <CardDescription className="text-xs">Passed</CardDescription>
            <CardTitle className="text-2xl text-green-500">{passCount}</CardTitle>
          </CardHeader>
        </Card>

        <Card>
          <CardHeader className="p-4">
            <CardDescription className="text-xs">Warnings</CardDescription>
            <CardTitle className="text-2xl text-yellow-500">{warningCount}</CardTitle>
          </CardHeader>
        </Card>

        <Card>
          <CardHeader className="p-4">
            <CardDescription className="text-xs">Failed</CardDescription>
            <CardTitle className="text-2xl text-red-500">{failCount}</CardTitle>
          </CardHeader>
        </Card>
      </div>

      {/* Filters */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <ScaleIcon className="h-5 w-5 text-primary" />
            <CardTitle>Constitutional Logs ({filteredLogs.length})</CardTitle>
          </div>
          <CardDescription>
            All constitutional checks performed on this organism
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-4">
            <div className="flex-1 relative">
              <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search logs..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-9"
              />
            </div>
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Filter by status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Statuses</SelectItem>
                <SelectItem value="pass">Pass</SelectItem>
                <SelectItem value="warning">Warning</SelectItem>
                <SelectItem value="fail">Fail</SelectItem>
              </SelectContent>
            </Select>
            <Select value={principleFilter} onValueChange={setPrincipleFilter}>
              <SelectTrigger className="w-[200px]">
                <SelectValue placeholder="Filter by principle" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Principles</SelectItem>
                {uniquePrinciples.map(principle => (
                  <SelectItem key={principle} value={principle}>
                    {principle.replace(/_/g, ' ')}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Logs */}
          <div className="space-y-3 max-h-[600px] overflow-y-auto">
            {filteredLogs.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                No logs match your filters
              </div>
            ) : (
              filteredLogs.map((log) => (
                <Card key={log.id}>
                  <CardHeader className="p-4">
                    <div className="flex items-start justify-between">
                      <div className="flex items-start gap-3">
                        {getStatusIcon(log.status)}
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="font-semibold">
                              {log.principle.replace(/_/g, ' ')}
                            </span>
                            {getStatusBadge(log.status)}
                          </div>
                          <p className="text-sm text-muted-foreground">
                            {log.details}
                          </p>
                          {log.context && (
                            <details className="mt-2">
                              <summary className="text-xs text-muted-foreground cursor-pointer hover:text-foreground">
                                View context
                              </summary>
                              <pre className="text-xs bg-muted p-2 rounded mt-2 overflow-x-auto">
                                {JSON.stringify(log.context, null, 2)}
                              </pre>
                            </details>
                          )}
                        </div>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {new Date(log.timestamp).toLocaleString()}
                      </div>
                    </div>
                  </CardHeader>
                </Card>
              ))
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
