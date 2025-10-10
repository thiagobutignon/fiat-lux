import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ActivityIcon } from "lucide-react";

export default function ActivityPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Activity</h1>
        <p className="text-muted-foreground">
          Recent system activity and logs
        </p>
      </div>

      <Card>
        <CardHeader>
          <ActivityIcon className="h-12 w-12 text-muted-foreground/50 mx-auto" />
          <CardTitle className="text-center">Coming Soon</CardTitle>
          <CardDescription className="text-center">
            Activity logs will be available soon
          </CardDescription>
        </CardHeader>
      </Card>
    </div>
  );
}
