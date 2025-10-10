import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { SettingsIcon } from "lucide-react";

export default function SettingsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Settings</h1>
        <p className="text-muted-foreground">
          Configure your DevTools preferences
        </p>
      </div>

      <Card>
        <CardHeader>
          <SettingsIcon className="h-12 w-12 text-muted-foreground/50 mx-auto" />
          <CardTitle className="text-center">Coming Soon</CardTitle>
          <CardDescription className="text-center">
            Settings will be available soon
          </CardDescription>
        </CardHeader>
      </Card>
    </div>
  );
}
