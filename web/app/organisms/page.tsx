"use client";

import { useEffect, useState } from "react";
import { OrganismList } from "@/components/organisms/OrganismList";
import { Button } from "@/components/ui/button";
import { GlassOrganism } from "@/lib/types";
import { ApiClient } from "@/lib/api-client";
import { UploadIcon, RefreshCwIcon } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export default function OrganismsPage() {
  const [organisms, setOrganisms] = useState<GlassOrganism[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);

  const loadOrganisms = async () => {
    setLoading(true);
    try {
      const data = await ApiClient.listOrganisms();
      setOrganisms(data);
    } catch (error) {
      console.error("Failed to load organisms:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadOrganisms();
  }, []);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.name.endsWith(".glass")) {
      alert("Please upload a .glass file");
      return;
    }

    setUploading(true);
    try {
      await ApiClient.uploadOrganism(file);
      await loadOrganisms(); // Refresh list
      alert("Organism uploaded successfully!");
    } catch (error) {
      console.error("Failed to upload organism:", error);
      alert("Failed to upload organism");
    } finally {
      setUploading(false);
      // Reset input
      event.target.value = "";
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Organisms</h1>
          <p className="text-muted-foreground">
            Manage your .glass organisms
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={() => loadOrganisms()}
            variant="outline"
            disabled={loading}
          >
            <RefreshCwIcon className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button asChild disabled={uploading}>
            <label className="cursor-pointer">
              <UploadIcon className="h-4 w-4 mr-2" />
              {uploading ? "Uploading..." : "Upload .glass"}
              <input
                type="file"
                accept=".glass"
                className="hidden"
                onChange={handleFileUpload}
                disabled={uploading}
              />
            </label>
          </Button>
        </div>
      </div>

      {/* Content */}
      {loading ? (
        <Card>
          <CardContent className="flex items-center justify-center py-12">
            <div className="text-center">
              <RefreshCwIcon className="h-8 w-8 animate-spin mx-auto text-muted-foreground" />
              <p className="mt-4 text-muted-foreground">Loading organisms...</p>
            </div>
          </CardContent>
        </Card>
      ) : organisms.length === 0 ? (
        <Card>
          <CardHeader>
            <CardTitle>No organisms yet</CardTitle>
            <CardDescription>
              Upload a .glass file to get started
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button asChild className="w-full" size="lg">
              <label className="cursor-pointer">
                <UploadIcon className="h-5 w-5 mr-2" />
                Upload your first .glass organism
                <input
                  type="file"
                  accept=".glass"
                  className="hidden"
                  onChange={handleFileUpload}
                />
              </label>
            </Button>
          </CardContent>
        </Card>
      ) : (
        <OrganismList organisms={organisms} />
      )}
    </div>
  );
}
