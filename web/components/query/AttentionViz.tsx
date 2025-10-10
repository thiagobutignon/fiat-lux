"use client";

import { AttentionWeight } from "@/lib/types";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { formatPercentage } from "@/lib/utils";

interface AttentionVizProps {
  attention: AttentionWeight[];
}

export function AttentionViz({ attention }: AttentionVizProps) {
  // Sort by weight descending and take top 10
  const topAttention = [...attention]
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 10);

  const data = topAttention.map((att) => ({
    name: att.source_id,
    weight: att.weight * 100, // Convert to percentage
  }));

  return (
    <div className="w-full h-64">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="horizontal">
          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
          <XAxis
            type="number"
            domain={[0, 100]}
            tickFormatter={(value) => `${value}%`}
            className="text-xs"
          />
          <YAxis
            type="category"
            dataKey="name"
            width={120}
            className="text-xs"
          />
          <Tooltip
            formatter={(value: number) => [`${value.toFixed(2)}%`, "Attention"]}
            contentStyle={{
              backgroundColor: "hsl(var(--card))",
              border: "1px solid hsl(var(--border))",
              borderRadius: "6px",
            }}
          />
          <Bar
            dataKey="weight"
            fill="hsl(var(--primary))"
            radius={[0, 4, 4, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
