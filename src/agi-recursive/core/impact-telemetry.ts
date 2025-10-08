/**
 * Impact Telemetry System
 *
 * Transparent, opt-out telemetry for workforce impact monitoring.
 * All data is aggregated, anonymized, and open-source auditable.
 *
 * Purpose: Enable global monitoring of AGI workforce impact without
 * compromising privacy or competitive secrets.
 */

import * as fs from 'fs/promises';
import * as path from 'path';
import * as crypto from 'crypto';
import { ImpactAssessment } from './workforce-impact-assessor';

// ============================================================================
// Types
// ============================================================================

export interface TelemetryConfig {
  enabled: boolean; // Opt-out: true by default
  endpoint?: string; // Where to send data (defaults to public aggregator)
  local_only: boolean; // Only store locally, never send
  anonymize: boolean; // Hash identifying information
  sample_rate: number; // 0.0-1.0, percentage of events to record
}

export interface ImpactEvent {
  timestamp: number;
  event_type:
    | 'query_assessed'
    | 'automation_proposed'
    | 'mrh_violation'
    | 'workforce_change'
    | 'retraining_initiated';
  deployment_id: string; // Anonymous hash
  industry_sector?: string; // Optional categorization
  company_size_bracket?: 'small' | 'medium' | 'large' | 'enterprise'; // Anonymized
  wia_score?: number;
  mrh_compliant?: boolean;
  jobs_affected?: number;
  jobs_created?: number;
  jobs_eliminated?: number;
  risk_level?: 'low' | 'medium' | 'high' | 'critical';
  blocked?: boolean;
  metadata?: Record<string, any>; // Additional anonymized context
}

export interface AggregatedStatistics {
  period_start: Date;
  period_end: Date;
  total_deployments: number;
  total_queries_assessed: number;
  queries_with_automation_intent: number;
  total_jobs_affected: number;
  total_jobs_created: number;
  total_jobs_eliminated: number;
  net_workforce_change: number;
  mrh_violations: number;
  average_wia_score: number;
  deployments_by_sector: Record<string, number>;
  blocked_queries: number;
}

// ============================================================================
// Telemetry System
// ============================================================================

export class ImpactTelemetry {
  private config: TelemetryConfig;
  private deploymentId: string;
  private eventBuffer: ImpactEvent[] = [];
  private localStoragePath: string;
  private flushInterval: NodeJS.Timeout | null = null;

  constructor(config?: Partial<TelemetryConfig>) {
    this.config = {
      enabled: config?.enabled ?? true, // Opt-out by default
      endpoint: config?.endpoint ?? 'https://agi-impact-telemetry.org/api/v1/events',
      local_only: config?.local_only ?? false,
      anonymize: config?.anonymize ?? true,
      sample_rate: config?.sample_rate ?? 1.0,
    };

    // Generate anonymous deployment ID (persistent across runs)
    this.deploymentId = this.generateDeploymentId();

    // Local storage for telemetry
    this.localStoragePath = path.join(process.cwd(), '.agi-telemetry');

    // Auto-flush every 5 minutes
    if (this.config.enabled) {
      this.flushInterval = setInterval(() => this.flush(), 5 * 60 * 1000);
    }

    this.logTelemetryStatus();
  }

  /**
   * Record an impact event
   */
  record(event: Omit<ImpactEvent, 'timestamp' | 'deployment_id'>): void {
    // Check: Telemetry disabled
    if (!this.config.enabled) return;

    // Check: Sample rate
    if (Math.random() > this.config.sample_rate) return;

    // Create full event
    const fullEvent: ImpactEvent = {
      ...event,
      timestamp: Date.now(),
      deployment_id: this.deploymentId,
    };

    // Anonymize if configured
    if (this.config.anonymize) {
      this.anonymizeEvent(fullEvent);
    }

    // Add to buffer
    this.eventBuffer.push(fullEvent);

    // Flush if buffer is large
    if (this.eventBuffer.length >= 100) {
      this.flush();
    }
  }

  /**
   * Record WIA assessment
   */
  recordAssessment(assessment: ImpactAssessment, query?: string): void {
    const hasAutomationIntent = assessment.risk_level !== 'low';

    this.record({
      event_type: hasAutomationIntent ? 'automation_proposed' : 'query_assessed',
      wia_score: assessment.wia_score,
      mrh_compliant: assessment.mrh_compliant,
      risk_level: assessment.risk_level,
      blocked: !assessment.approved,
    });

    if (!assessment.mrh_compliant) {
      this.record({
        event_type: 'mrh_violation',
        wia_score: assessment.wia_score,
        risk_level: assessment.risk_level,
      });
    }
  }

  /**
   * Record workforce change
   */
  recordWorkforceChange(change: {
    jobs_affected: number;
    jobs_created: number;
    jobs_eliminated: number;
    industry_sector?: string;
  }): void {
    this.record({
      event_type: 'workforce_change',
      jobs_affected: change.jobs_affected,
      jobs_created: change.jobs_created,
      jobs_eliminated: change.jobs_eliminated,
      industry_sector: change.industry_sector,
    });
  }

  /**
   * Flush buffered events (send or save locally)
   */
  async flush(): Promise<void> {
    if (this.eventBuffer.length === 0) return;

    const events = [...this.eventBuffer];
    this.eventBuffer = [];

    try {
      // Always save locally for transparency
      await this.saveLocal(events);

      // Send to aggregator if not local_only
      if (!this.config.local_only && this.config.endpoint) {
        await this.sendToAggregator(events);
      }
    } catch (error) {
      console.warn('⚠️  Telemetry flush failed:', error);
      // Re-add events to buffer for retry
      this.eventBuffer.push(...events);
    }
  }

  /**
   * Get local statistics (for transparency)
   */
  async getLocalStats(days: number = 30): Promise<AggregatedStatistics> {
    const events = await this.loadLocal(days);

    const stats: AggregatedStatistics = {
      period_start: new Date(Date.now() - days * 24 * 60 * 60 * 1000),
      period_end: new Date(),
      total_deployments: new Set(events.map((e) => e.deployment_id)).size,
      total_queries_assessed: events.filter((e) => e.event_type === 'query_assessed').length,
      queries_with_automation_intent: events.filter((e) => e.event_type === 'automation_proposed').length,
      total_jobs_affected: events.reduce((sum, e) => sum + (e.jobs_affected || 0), 0),
      total_jobs_created: events.reduce((sum, e) => sum + (e.jobs_created || 0), 0),
      total_jobs_eliminated: events.reduce((sum, e) => sum + (e.jobs_eliminated || 0), 0),
      net_workforce_change: 0,
      mrh_violations: events.filter((e) => e.event_type === 'mrh_violation').length,
      average_wia_score: 0,
      deployments_by_sector: {},
      blocked_queries: events.filter((e) => e.blocked).length,
    };

    stats.net_workforce_change = stats.total_jobs_created - stats.total_jobs_eliminated;

    // Calculate average WIA score
    const wiaEvents = events.filter((e) => e.wia_score !== undefined);
    if (wiaEvents.length > 0) {
      stats.average_wia_score = wiaEvents.reduce((sum, e) => sum + (e.wia_score || 0), 0) / wiaEvents.length;
    }

    // Aggregate by sector
    events.forEach((e) => {
      if (e.industry_sector) {
        stats.deployments_by_sector[e.industry_sector] =
          (stats.deployments_by_sector[e.industry_sector] || 0) + 1;
      }
    });

    return stats;
  }

  /**
   * Disable telemetry (opt-out)
   */
  disable(): void {
    this.config.enabled = false;
    if (this.flushInterval) {
      clearInterval(this.flushInterval);
      this.flushInterval = null;
    }
    console.log('📊 Telemetry DISABLED. No data will be collected.');
  }

  /**
   * Enable telemetry (opt-in)
   */
  enable(): void {
    this.config.enabled = true;
    if (!this.flushInterval) {
      this.flushInterval = setInterval(() => this.flush(), 5 * 60 * 1000);
    }
    console.log('📊 Telemetry ENABLED. Anonymized impact data will be collected.');
  }

  /**
   * Clean up resources
   */
  async shutdown(): Promise<void> {
    if (this.flushInterval) {
      clearInterval(this.flushInterval);
    }
    await this.flush();
  }

  // ==========================================================================
  // Private Methods
  // ==========================================================================

  /**
   * Generate anonymous deployment ID
   */
  private generateDeploymentId(): string {
    const idFile = path.join(this.localStoragePath, 'deployment-id');

    try {
      // Try to load existing ID
      const existing = require('fs').readFileSync(idFile, 'utf-8');
      return existing.trim();
    } catch {
      // Generate new ID
      const id = crypto.randomBytes(16).toString('hex');

      // Save for persistence
      try {
        require('fs').mkdirSync(this.localStoragePath, { recursive: true });
        require('fs').writeFileSync(idFile, id);
      } catch (e) {
        console.warn('Could not persist deployment ID:', e);
      }

      return id;
    }
  }

  /**
   * Anonymize sensitive fields
   */
  private anonymizeEvent(event: ImpactEvent): void {
    // Remove any potentially identifying metadata
    if (event.metadata) {
      delete event.metadata.query;
      delete event.metadata.company_name;
      delete event.metadata.user_id;
      delete event.metadata.ip_address;
    }

    // Anonymize numeric values (round to brackets)
    if (event.jobs_affected !== undefined) {
      event.jobs_affected = this.roundToBracket(event.jobs_affected);
    }
    if (event.jobs_created !== undefined) {
      event.jobs_created = this.roundToBracket(event.jobs_created);
    }
    if (event.jobs_eliminated !== undefined) {
      event.jobs_eliminated = this.roundToBracket(event.jobs_eliminated);
    }
  }

  /**
   * Round numbers to brackets for anonymity
   */
  private roundToBracket(n: number): number {
    if (n < 10) return Math.ceil(n / 5) * 5; // Round to 5
    if (n < 100) return Math.ceil(n / 10) * 10; // Round to 10
    if (n < 1000) return Math.ceil(n / 50) * 50; // Round to 50
    return Math.ceil(n / 100) * 100; // Round to 100
  }

  /**
   * Save events to local storage (for transparency/audit)
   */
  private async saveLocal(events: ImpactEvent[]): Promise<void> {
    const date = new Date().toISOString().split('T')[0];
    const filename = `telemetry-${date}.jsonl`;
    const filepath = path.join(this.localStoragePath, filename);

    try {
      await fs.mkdir(this.localStoragePath, { recursive: true });

      const lines = events.map((e) => JSON.stringify(e)).join('\n') + '\n';
      await fs.appendFile(filepath, lines, 'utf-8');
    } catch (error) {
      console.warn('⚠️  Could not save telemetry locally:', error);
    }
  }

  /**
   * Load local events for analysis
   */
  private async loadLocal(days: number): Promise<ImpactEvent[]> {
    const events: ImpactEvent[] = [];
    const cutoff = Date.now() - days * 24 * 60 * 60 * 1000;

    try {
      const files = await fs.readdir(this.localStoragePath);
      const telemetryFiles = files.filter((f) => f.startsWith('telemetry-') && f.endsWith('.jsonl'));

      for (const file of telemetryFiles) {
        const filepath = path.join(this.localStoragePath, file);
        const content = await fs.readFile(filepath, 'utf-8');
        const lines = content.trim().split('\n');

        for (const line of lines) {
          if (!line) continue;
          const event: ImpactEvent = JSON.parse(line);
          if (event.timestamp >= cutoff) {
            events.push(event);
          }
        }
      }
    } catch (error) {
      console.warn('⚠️  Could not load local telemetry:', error);
    }

    return events;
  }

  /**
   * Send events to public aggregator
   */
  private async sendToAggregator(events: ImpactEvent[]): Promise<void> {
    if (!this.config.endpoint) return;

    // Note: This would be a real HTTP POST in production
    // For now, just log that we would send
    console.log(`📡 Would send ${events.length} telemetry events to ${this.config.endpoint}`);

    // In production:
    // await fetch(this.config.endpoint, {
    //   method: 'POST',
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify({ events }),
    // });
  }

  /**
   * Log telemetry status on initialization
   */
  private logTelemetryStatus(): void {
    if (!this.config.enabled) {
      console.log('\n📊 ========================================');
      console.log('📊 Telemetry: DISABLED (opt-out)');
      console.log('📊 ========================================\n');
      return;
    }

    console.log('\n📊 ========================================');
    console.log('📊 Telemetry: ENABLED (opt-out anytime)');
    console.log(`📊 Deployment ID: ${this.deploymentId.substring(0, 8)}...`);
    console.log(`📊 Anonymization: ${this.config.anonymize ? 'ON' : 'OFF'}`);
    console.log(`📊 Local only: ${this.config.local_only ? 'YES' : 'NO'}`);
    console.log(`📊 Sample rate: ${(this.config.sample_rate * 100).toFixed(0)}%`);
    console.log('📊 ');
    console.log('📊 Purpose: Monitor global AGI workforce impact');
    console.log('📊 Data: Fully anonymized, auditable locally');
    console.log('📊 Opt-out: Set TELEMETRY_ENABLED=false');
    console.log('📊 Details: LICENSE-COMMERCIAL.md');
    console.log('📊 ========================================\n');
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create telemetry instance from environment variables
 */
export function createTelemetryFromEnv(): ImpactTelemetry {
  const config: Partial<TelemetryConfig> = {
    enabled: process.env.TELEMETRY_ENABLED !== 'false',
    local_only: process.env.TELEMETRY_LOCAL_ONLY === 'true',
    anonymize: process.env.TELEMETRY_ANONYMIZE !== 'false',
    sample_rate: parseFloat(process.env.TELEMETRY_SAMPLE_RATE || '1.0'),
  };

  if (process.env.TELEMETRY_ENDPOINT) {
    config.endpoint = process.env.TELEMETRY_ENDPOINT;
  }

  return new ImpactTelemetry(config);
}

// ============================================================================
// Global Singleton (opt-out friendly)
// ============================================================================

let globalTelemetry: ImpactTelemetry | null = null;

export function getGlobalTelemetry(): ImpactTelemetry {
  if (!globalTelemetry) {
    globalTelemetry = createTelemetryFromEnv();
  }
  return globalTelemetry;
}

export async function shutdownGlobalTelemetry(): Promise<void> {
  if (globalTelemetry) {
    await globalTelemetry.shutdown();
    globalTelemetry = null;
  }
}
