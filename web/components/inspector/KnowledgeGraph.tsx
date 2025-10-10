"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { NetworkIcon } from "lucide-react";

interface KnowledgeGraphProps {
  papers: number;
  connections: number;
  clusters: number;
  embeddings_dim: number;
}

export function KnowledgeGraph({ papers, connections, clusters, embeddings_dim }: KnowledgeGraphProps) {
  // Generate simulated graph data for visualization
  const generateGraphData = () => {
    const data = [];
    const clusterColors = [
      "#3b82f6", // blue
      "#10b981", // green
      "#f59e0b", // yellow
      "#ef4444", // red
      "#8b5cf6", // purple
      "#ec4899", // pink
      "#14b8a6", // teal
      "#f97316", // orange
      "#6366f1", // indigo
      "#84cc16", // lime
    ];

    for (let i = 0; i < papers; i++) {
      const cluster = Math.floor(Math.random() * clusters);
      const baseX = (cluster % 3) * 300 + 150;
      const baseY = Math.floor(cluster / 3) * 300 + 150;

      data.push({
        x: baseX + (Math.random() - 0.5) * 200,
        y: baseY + (Math.random() - 0.5) * 200,
        cluster,
        color: clusterColors[cluster % clusterColors.length],
        size: 50 + Math.random() * 100,
      });
    }

    return data;
  };

  const graphData = generateGraphData();

  return (
    <div className="space-y-6">
      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="p-4">
            <CardDescription className="text-xs">Papers</CardDescription>
            <CardTitle className="text-2xl">{papers}</CardTitle>
          </CardHeader>
        </Card>

        <Card>
          <CardHeader className="p-4">
            <CardDescription className="text-xs">Connections</CardDescription>
            <CardTitle className="text-2xl">{connections}</CardTitle>
          </CardHeader>
        </Card>

        <Card>
          <CardHeader className="p-4">
            <CardDescription className="text-xs">Clusters</CardDescription>
            <CardTitle className="text-2xl">{clusters}</CardTitle>
          </CardHeader>
        </Card>

        <Card>
          <CardHeader className="p-4">
            <CardDescription className="text-xs">Embedding Dim</CardDescription>
            <CardTitle className="text-2xl">{embeddings_dim}</CardTitle>
          </CardHeader>
        </Card>
      </div>

      {/* Graph Visualization */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <NetworkIcon className="h-5 w-5 text-primary" />
            <CardTitle>Knowledge Graph</CardTitle>
          </div>
          <CardDescription>
            Visual representation of {papers} papers organized into {clusters} clusters
          </CardDescription>
        </CardHeader>
        <CardContent>
          {papers > 0 ? (
            <div className="w-full h-96">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis type="number" dataKey="x" hide />
                  <YAxis type="number" dataKey="y" hide />
                  <Tooltip
                    cursor={{ strokeDasharray: '3 3' }}
                    content={({ payload }) => {
                      if (!payload || !payload[0]) return null;
                      const data = payload[0].payload;
                      return (
                        <div className="bg-card border rounded-lg p-2 shadow-lg">
                          <p className="text-xs font-semibold">Cluster {data.cluster}</p>
                          <p className="text-xs text-muted-foreground">
                            Position: ({Math.round(data.x)}, {Math.round(data.y)})
                          </p>
                        </div>
                      );
                    }}
                  />
                  <Scatter data={graphData} fill="#8884d8">
                    {graphData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="flex items-center justify-center h-96">
              <div className="text-center">
                <NetworkIcon className="h-12 w-12 text-muted-foreground/50 mx-auto mb-4" />
                <p className="text-muted-foreground">No knowledge graph yet</p>
                <p className="text-sm text-muted-foreground mt-2">
                  Graph will appear after knowledge ingestion
                </p>
              </div>
            </div>
          )}

          {/* Legend */}
          {papers > 0 && (
            <div className="mt-4 pt-4 border-t">
              <div className="flex flex-wrap gap-2">
                {Array.from({ length: Math.min(clusters, 10) }).map((_, i) => {
                  const colors = [
                    "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
                    "#ec4899", "#14b8a6", "#f97316", "#6366f1", "#84cc16",
                  ];
                  return (
                    <div key={i} className="flex items-center gap-2 text-xs">
                      <div
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: colors[i] }}
                      />
                      <span className="text-muted-foreground">Cluster {i}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
