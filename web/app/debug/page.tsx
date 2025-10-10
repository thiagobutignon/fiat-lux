import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { BugIcon } from "lucide-react";

export default function DebugPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Debug Tools</h1>
        <p className="text-muted-foreground">
          System-wide debugging and monitoring
        </p>
      </div>

      <Card>
        <CardHeader>
          <BugIcon className="h-12 w-12 text-muted-foreground/50 mx-auto" />
          <CardTitle className="text-center">Coming Soon</CardTitle>
          <CardDescription className="text-center">
            Debug tools will be available in DIA 4
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2 text-sm text-muted-foreground">
            <li>• Constitutional logs viewer</li>
            <li>• LLM call inspector</li>
            <li>• Cost tracking dashboard</li>
            <li>• Performance metrics</li>
            <li>• Evolution tracker</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
