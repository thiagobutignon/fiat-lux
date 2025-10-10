"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { EmergedFunction } from "@/lib/types";
import {
  CopyIcon,
  CheckIcon,
  DownloadIcon,
  CodeIcon,
  CheckCircle2Icon,
  XCircleIcon,
} from "lucide-react";

interface FunctionViewerProps {
  functions: EmergedFunction[];
}

export function FunctionViewer({ functions }: FunctionViewerProps) {
  const [selectedFunction, setSelectedFunction] = useState<EmergedFunction | null>(
    functions[0] || null
  );
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    if (!selectedFunction) return;

    try {
      await navigator.clipboard.writeText(selectedFunction.code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      console.error("Failed to copy:", error);
    }
  };

  const handleDownload = () => {
    if (!selectedFunction) return;

    const blob = new Blob([selectedFunction.code], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${selectedFunction.name}.gl`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (functions.length === 0) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-12">
          <div className="text-center">
            <CodeIcon className="h-12 w-12 text-muted-foreground/50 mx-auto mb-4" />
            <p className="text-muted-foreground">No functions emerged yet</p>
            <p className="text-sm text-muted-foreground mt-2">
              Functions will appear here as patterns are detected
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Function List */}
      <div className="space-y-2">
        <h3 className="font-semibold mb-4">Functions ({functions.length})</h3>
        {functions.map((fn) => (
          <Card
            key={fn.name}
            className={`cursor-pointer transition-all ${
              selectedFunction?.name === fn.name
                ? "ring-2 ring-primary"
                : "hover:bg-muted/50"
            }`}
            onClick={() => setSelectedFunction(fn)}
          >
            <CardHeader className="p-4">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm font-mono">{fn.name}</CardTitle>
                <Badge
                  variant={fn.constitutional_status === "pass" ? "default" : "destructive"}
                  className="text-xs"
                >
                  {fn.constitutional_status === "pass" ? (
                    <CheckCircle2Icon className="h-3 w-3" />
                  ) : (
                    <XCircleIcon className="h-3 w-3" />
                  )}
                </Badge>
              </div>
              <CardDescription className="text-xs truncate">
                {fn.signature}
              </CardDescription>
              <div className="flex items-center gap-2 mt-2 text-xs text-muted-foreground">
                <span>{fn.lines} lines</span>
                <span>•</span>
                <span>{fn.occurrences} occur</span>
              </div>
            </CardHeader>
          </Card>
        ))}
      </div>

      {/* Function Code */}
      <div className="lg:col-span-2">
        {selectedFunction && (
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="font-mono">{selectedFunction.name}</CardTitle>
                  <CardDescription className="mt-1">
                    Emerged from: {selectedFunction.emerged_from} ({selectedFunction.occurrences} occurrences)
                  </CardDescription>
                </div>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleCopy}
                  >
                    {copied ? (
                      <>
                        <CheckIcon className="h-4 w-4 mr-2" />
                        Copied
                      </>
                    ) : (
                      <>
                        <CopyIcon className="h-4 w-4 mr-2" />
                        Copy
                      </>
                    )}
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleDownload}
                  >
                    <DownloadIcon className="h-4 w-4 mr-2" />
                    Download
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="relative">
                {/* Code Block with Line Numbers */}
                <pre className="bg-muted/50 rounded-lg p-4 overflow-x-auto">
                  <code className="text-sm font-mono">
                    {selectedFunction.code.split('\n').map((line, i) => (
                      <div key={i} className="flex">
                        <span className="text-muted-foreground select-none mr-4 text-right" style={{ minWidth: '2rem' }}>
                          {i + 1}
                        </span>
                        <span className="flex-1">{line}</span>
                      </div>
                    ))}
                  </code>
                </pre>
              </div>

              {/* Metadata */}
              <div className="mt-4 pt-4 border-t grid grid-cols-2 gap-4 text-sm">
                <div>
                  <div className="text-muted-foreground">Signature</div>
                  <div className="font-mono text-xs mt-1">{selectedFunction.signature}</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Constitutional Status</div>
                  <Badge
                    variant={selectedFunction.constitutional_status === "pass" ? "default" : "destructive"}
                    className="mt-1"
                  >
                    {selectedFunction.constitutional_status === "pass" ? "✅ PASS" : "❌ FAIL"}
                  </Badge>
                </div>
                <div>
                  <div className="text-muted-foreground">Lines of Code</div>
                  <div className="font-semibold mt-1">{selectedFunction.lines}</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Created At</div>
                  <div className="text-xs mt-1">
                    {new Date(selectedFunction.created_at).toLocaleString()}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
