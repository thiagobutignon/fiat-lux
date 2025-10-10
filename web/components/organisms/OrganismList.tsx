"use client";

import { useState } from "react";
import { GlassOrganism } from "@/lib/types";
import { OrganismCard } from "./OrganismCard";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { SearchIcon, FilterIcon } from "lucide-react";

interface OrganismListProps {
  organisms: GlassOrganism[];
}

export function OrganismList({ organisms }: OrganismListProps) {
  const [search, setSearch] = useState("");
  const [stageFilter, setStageFilter] = useState<string>("all");
  const [sortBy, setSortBy] = useState<string>("name");

  // Filter organisms
  const filtered = organisms.filter((org) => {
    const matchesSearch =
      org.metadata.name.toLowerCase().includes(search.toLowerCase()) ||
      org.metadata.specialization.toLowerCase().includes(search.toLowerCase());

    const matchesStage =
      stageFilter === "all" || org.metadata.stage === stageFilter;

    return matchesSearch && matchesStage;
  });

  // Sort organisms
  const sorted = [...filtered].sort((a, b) => {
    switch (sortBy) {
      case "name":
        return a.metadata.name.localeCompare(b.metadata.name);
      case "maturity":
        return b.metadata.maturity - a.metadata.maturity;
      case "cost":
        return b.stats.total_cost - a.stats.total_cost;
      case "fitness":
        return b.evolution.fitness - a.evolution.fitness;
      default:
        return 0;
    }
  });

  return (
    <div className="space-y-6">
      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="relative flex-1">
          <SearchIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search organisms..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-10"
          />
        </div>

        <Select value={stageFilter} onValueChange={setStageFilter}>
          <SelectTrigger className="w-full sm:w-[180px]">
            <FilterIcon className="h-4 w-4 mr-2" />
            <SelectValue placeholder="Filter by stage" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Stages</SelectItem>
            <SelectItem value="nascent">Nascent</SelectItem>
            <SelectItem value="infancy">Infancy</SelectItem>
            <SelectItem value="adolescence">Adolescence</SelectItem>
            <SelectItem value="maturity">Maturity</SelectItem>
            <SelectItem value="evolution">Evolution</SelectItem>
          </SelectContent>
        </Select>

        <Select value={sortBy} onValueChange={setSortBy}>
          <SelectTrigger className="w-full sm:w-[180px]">
            <SelectValue placeholder="Sort by" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="name">Name</SelectItem>
            <SelectItem value="maturity">Maturity</SelectItem>
            <SelectItem value="cost">Cost</SelectItem>
            <SelectItem value="fitness">Fitness</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Results */}
      {sorted.length === 0 ? (
        <div className="text-center py-12">
          <p className="text-muted-foreground">No organisms found</p>
        </div>
      ) : (
        <>
          <div className="text-sm text-muted-foreground">
            Showing {sorted.length} of {organisms.length} organisms
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {sorted.map((organism) => (
              <OrganismCard key={organism.id} organism={organism} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
