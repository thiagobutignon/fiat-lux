"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { ApiClient } from "@/lib/api-client";
import { QueryResult as QueryResultType } from "@/lib/types";
import { QueryResult } from "./QueryResult";
import { SendIcon, Loader2Icon } from "lucide-react";

interface QueryConsoleProps {
  organismId: string;
  organismName: string;
}

interface QueryHistoryItem {
  query: string;
  result: QueryResultType;
  timestamp: string;
}

export function QueryConsole({ organismId, organismName }: QueryConsoleProps) {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState<QueryHistoryItem[]>([]);
  const [currentResult, setCurrentResult] = useState<QueryResultType | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!query.trim() || loading) return;

    setLoading(true);
    setCurrentResult(null);

    try {
      const result = await ApiClient.executeQuery(organismId, query);

      const historyItem: QueryHistoryItem = {
        query,
        result,
        timestamp: new Date().toISOString(),
      };

      setHistory([historyItem, ...history]);
      setCurrentResult(result);
      setQuery("");
    } catch (error) {
      console.error("Query failed:", error);
      alert("Query failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as any);
    }
  };

  return (
    <div className="space-y-6">
      {/* Query Input */}
      <Card className="p-4">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <Input
            placeholder={`Ask ${organismName} a question...`}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
            className="flex-1"
          />
          <Button type="submit" disabled={loading || !query.trim()}>
            {loading ? (
              <>
                <Loader2Icon className="h-4 w-4 mr-2 animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <SendIcon className="h-4 w-4 mr-2" />
                Send
              </>
            )}
          </Button>
        </form>
        <p className="text-xs text-muted-foreground mt-2">
          Press Enter to send, Shift+Enter for new line
        </p>
      </Card>

      {/* Current Result */}
      {currentResult && (
        <QueryResult result={currentResult} query={history[0]?.query || query} />
      )}

      {/* Query History */}
      {history.length > 1 && (
        <div className="space-y-4">
          <h2 className="text-lg font-semibold">Query History</h2>
          {history.slice(1).map((item, index) => (
            <div key={index} className="opacity-60 hover:opacity-100 transition-opacity">
              <QueryResult
                result={item.result}
                query={item.query}
                compact
              />
            </div>
          ))}
        </div>
      )}

      {/* Empty State */}
      {history.length === 0 && !currentResult && (
        <Card className="p-12 text-center">
          <p className="text-muted-foreground">
            Ask a question to get started
          </p>
          <p className="text-sm text-muted-foreground mt-2">
            The organism will process your query and return an answer with sources, reasoning, and attention tracking
          </p>
        </Card>
      )}
    </div>
  );
}
