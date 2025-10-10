import { GlassOrganism, QueryResult, SystemStats } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "";

export class ApiClient {
  private static async request<T>(
    path: string,
    options?: RequestInit
  ): Promise<T> {
    const response = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }

    return response.json();
  }

  // Organisms
  static async listOrganisms(): Promise<GlassOrganism[]> {
    return this.request("/api/organisms");
  }

  static async getOrganism(id: string): Promise<GlassOrganism> {
    return this.request(`/api/organisms/${id}`);
  }

  static async uploadOrganism(file: File): Promise<GlassOrganism> {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_BASE}/api/organisms`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Upload failed: ${response.statusText}`);
    }

    return response.json();
  }

  static async deleteOrganism(id: string): Promise<void> {
    await this.request(`/api/organisms/${id}`, {
      method: "DELETE",
    });
  }

  // Queries
  static async executeQuery(
    organismId: string,
    query: string
  ): Promise<QueryResult> {
    return this.request(`/api/query`, {
      method: "POST",
      body: JSON.stringify({ organismId, query }),
    });
  }

  // System stats
  static async getSystemStats(): Promise<SystemStats> {
    return this.request("/api/stats");
  }
}
