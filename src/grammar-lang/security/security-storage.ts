/**
 * Security Profile Storage
 *
 * Persistence layer for behavioral security profiles using SQLO
 * O(1) lookups, content-addressable, RBAC-protected
 *
 * Storage Structure:
 * ```
 * sqlo_security/
 * ├── profiles/
 * │   ├── <user_hash>/
 * │   │   ├── linguistic.json
 * │   │   ├── typing.json
 * │   │   ├── emotional.json
 * │   │   ├── temporal.json
 * │   │   ├── challenges.json
 * │   │   └── metadata.json
 * │   └── .index (user_id → hash mapping)
 * └── events/
 *     ├── <event_hash>/
 *     │   ├── event.json
 *     │   └── metadata.json
 *     └── .index (timestamp index)
 * ```
 */

import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';
import {
  UserSecurityProfiles,
  LinguisticProfile,
  TypingProfile,
  EmotionalProfile,
  TemporalProfile,
  SecurityContext,
} from './types';
import { ChallengeSet } from './cognitive-challenge';
import { RbacPolicy, Permission, getGlobalRbacPolicy } from '../database/rbac';
import { MemoryType } from '../database/sqlo';
import { LinguisticCollector } from './linguistic-collector';
import { TypingCollector } from './typing-collector';
import { EmotionalCollector } from './emotional-collector';
import { TemporalCollector } from './temporal-collector';

// =============================================================================
// TYPES
// =============================================================================

/**
 * Security event for audit log
 */
export interface SecurityEvent {
  event_id: string; // Content hash
  user_id: string;
  timestamp: number;
  event_type:
    | 'duress_detected'
    | 'coercion_detected'
    | 'impersonation_detected'
    | 'panic_code_detected'
    | 'cognitive_challenge_passed'
    | 'cognitive_challenge_failed'
    | 'operation_blocked'
    | 'operation_delayed'
    | 'profile_updated';

  // Event details
  duress_score?: number;
  coercion_score?: number;
  confidence: number;
  decision: 'allow' | 'challenge' | 'delay' | 'block';
  reason: string;

  // Context
  operation_type?: string;
  operation_value?: number;
  context?: any; // Additional context
}

/**
 * Profile metadata
 */
export interface ProfileMetadata {
  user_hash: string;
  user_id: string;
  created_at: number;
  last_updated: number;
  samples_analyzed: number;
  overall_confidence: number;
  last_event?: string; // Last security event hash
}

/**
 * Storage index
 */
export interface SecurityStorageIndex {
  profiles: Record<string, ProfileMetadata>; // user_id → metadata
  user_id_to_hash: Record<string, string>; // user_id → hash (O(1) lookup)
  statistics: {
    total_profiles: number;
    total_events: number;
    alerts_last_24h: number;
  };
}

// =============================================================================
// CONSTANTS
// =============================================================================

const SECURITY_STORAGE_DIR = 'sqlo_security';
const PROFILES_DIR = 'profiles';
const EVENTS_DIR = 'events';
const INDEX_FILE = '.index';

// =============================================================================
// SECURITY STORAGE
// =============================================================================

export class SecurityStorage {
  private readonly baseDir: string;
  private index: SecurityStorageIndex;
  private readonly rbacPolicy: RbacPolicy;

  constructor(baseDir: string = SECURITY_STORAGE_DIR, rbacPolicy?: RbacPolicy) {
    this.baseDir = baseDir;
    this.rbacPolicy = rbacPolicy || getGlobalRbacPolicy();
    this.index = this.loadIndex();
    this.ensureDirectories();
  }

  // ===========================================================================
  // PROFILE OPERATIONS (O(1))
  // ===========================================================================

  /**
   * Save complete security profiles for a user - O(1)
   */
  saveProfile(
    profiles: UserSecurityProfiles,
    roleName: string = 'admin'
  ): string {
    // RBAC check
    if (!this.rbacPolicy.hasPermission(roleName, MemoryType.LONG_TERM, Permission.WRITE)) {
      throw new Error(
        `Permission denied: Role '${roleName}' cannot write security profiles`
      );
    }

    // Hash user_id for content-addressable storage
    const userHash = this.hash(profiles.user_id);

    // Create profile directory
    const profileDir = this.getProfileDir(userHash);
    if (!fs.existsSync(profileDir)) {
      fs.mkdirSync(profileDir, { recursive: true });
    }

    // Save individual profiles (using collectors' toJSON for proper serialization)
    this.saveJSON(
      path.join(profileDir, 'linguistic.json'),
      LinguisticCollector.toJSON(profiles.linguistic)
    );
    this.saveJSON(
      path.join(profileDir, 'typing.json'),
      TypingCollector.toJSON(profiles.typing)
    );
    this.saveJSON(
      path.join(profileDir, 'emotional.json'),
      EmotionalCollector.toJSON(profiles.emotional)
    );
    this.saveJSON(
      path.join(profileDir, 'temporal.json'),
      TemporalCollector.toJSON(profiles.temporal)
    );

    // Save metadata
    const metadata: ProfileMetadata = {
      user_hash: userHash,
      user_id: profiles.user_id,
      created_at: this.index.profiles[profiles.user_id]?.created_at || Date.now(),
      last_updated: Date.now(),
      samples_analyzed: profiles.linguistic.samples_analyzed, // Use linguistic as reference
      overall_confidence: profiles.overall_confidence,
    };

    this.saveJSON(path.join(profileDir, 'metadata.json'), metadata);

    // Update index
    this.index.profiles[profiles.user_id] = metadata;
    this.index.user_id_to_hash[profiles.user_id] = userHash;
    this.updateStatistics();
    this.saveIndex();

    return userHash;
  }

  /**
   * Load complete security profiles for a user - O(1)
   */
  loadProfile(userId: string, roleName: string = 'admin'): UserSecurityProfiles | null {
    // RBAC check
    if (!this.rbacPolicy.hasPermission(roleName, MemoryType.LONG_TERM, Permission.READ)) {
      throw new Error(
        `Permission denied: Role '${roleName}' cannot read security profiles`
      );
    }

    // Get user hash - O(1)
    const userHash = this.index.user_id_to_hash[userId];
    if (!userHash) {
      return null;
    }

    const profileDir = this.getProfileDir(userHash);
    if (!fs.existsSync(profileDir)) {
      return null;
    }

    // Load individual profiles (using collectors' fromJSON for proper deserialization)
    const linguisticData = this.loadJSON<any>(
      path.join(profileDir, 'linguistic.json')
    );
    const typingData = this.loadJSON<any>(path.join(profileDir, 'typing.json'));
    const emotionalData = this.loadJSON<any>(
      path.join(profileDir, 'emotional.json')
    );
    const temporalData = this.loadJSON<any>(
      path.join(profileDir, 'temporal.json')
    );
    const metadata = this.loadJSON<ProfileMetadata>(
      path.join(profileDir, 'metadata.json')
    );

    if (!linguisticData || !typingData || !emotionalData || !temporalData || !metadata) {
      return null;
    }

    // Deserialize using collectors' fromJSON (converts Maps/Sets properly)
    const linguistic = LinguisticCollector.fromJSON(linguisticData);
    const typing = TypingCollector.fromJSON(typingData);
    const emotional = EmotionalCollector.fromJSON(emotionalData);
    const temporal = TemporalCollector.fromJSON(temporalData);

    return {
      user_id: userId,
      linguistic,
      typing,
      emotional,
      temporal,
      overall_confidence: metadata.overall_confidence,
      last_interaction: metadata.last_updated,
    };
  }

  /**
   * Update profile incrementally - O(1)
   * More efficient than saving entire profile
   */
  updateProfile(
    userId: string,
    updates: {
      linguistic?: LinguisticProfile;
      typing?: TypingProfile;
      emotional?: EmotionalProfile;
      temporal?: TemporalProfile;
      overall_confidence?: number;
    },
    roleName: string = 'admin'
  ): boolean {
    // RBAC check
    if (!this.rbacPolicy.hasPermission(roleName, MemoryType.LONG_TERM, Permission.WRITE)) {
      throw new Error(
        `Permission denied: Role '${roleName}' cannot update security profiles`
      );
    }

    const userHash = this.index.user_id_to_hash[userId];
    if (!userHash) {
      return false;
    }

    const profileDir = this.getProfileDir(userHash);

    // Update individual files (only those provided, using proper serialization)
    if (updates.linguistic) {
      this.saveJSON(
        path.join(profileDir, 'linguistic.json'),
        LinguisticCollector.toJSON(updates.linguistic)
      );
    }
    if (updates.typing) {
      this.saveJSON(
        path.join(profileDir, 'typing.json'),
        TypingCollector.toJSON(updates.typing)
      );
    }
    if (updates.emotional) {
      this.saveJSON(
        path.join(profileDir, 'emotional.json'),
        EmotionalCollector.toJSON(updates.emotional)
      );
    }
    if (updates.temporal) {
      this.saveJSON(
        path.join(profileDir, 'temporal.json'),
        TemporalCollector.toJSON(updates.temporal)
      );
    }

    // Update metadata
    const metadata = this.index.profiles[userId];
    if (metadata) {
      metadata.last_updated = Date.now();
      if (updates.overall_confidence !== undefined) {
        metadata.overall_confidence = updates.overall_confidence;
      }
      // Update samples if linguistic profile provided
      if (updates.linguistic) {
        metadata.samples_analyzed = updates.linguistic.samples_analyzed;
      }

      this.saveJSON(path.join(profileDir, 'metadata.json'), metadata);
      this.saveIndex();
    }

    return true;
  }

  /**
   * Check if profile exists - O(1)
   */
  hasProfile(userId: string): boolean {
    return userId in this.index.user_id_to_hash;
  }

  /**
   * Delete profile - O(1)
   */
  deleteProfile(userId: string, roleName: string = 'admin'): boolean {
    // RBAC check
    if (!this.rbacPolicy.hasPermission(roleName, MemoryType.LONG_TERM, Permission.DELETE)) {
      throw new Error(
        `Permission denied: Role '${roleName}' cannot delete security profiles`
      );
    }

    const userHash = this.index.user_id_to_hash[userId];
    if (!userHash) {
      return false;
    }

    const profileDir = this.getProfileDir(userHash);
    if (fs.existsSync(profileDir)) {
      fs.rmSync(profileDir, { recursive: true });
    }

    // Remove from index
    delete this.index.profiles[userId];
    delete this.index.user_id_to_hash[userId];
    this.updateStatistics();
    this.saveIndex();

    return true;
  }

  // ===========================================================================
  // COGNITIVE CHALLENGE OPERATIONS
  // ===========================================================================

  /**
   * Save cognitive challenge set for a user - O(1)
   */
  saveChallenges(
    userId: string,
    challenges: ChallengeSet,
    roleName: string = 'admin'
  ): boolean {
    // RBAC check
    if (!this.rbacPolicy.hasPermission(roleName, MemoryType.LONG_TERM, Permission.WRITE)) {
      throw new Error(
        `Permission denied: Role '${roleName}' cannot write challenge sets`
      );
    }

    const userHash = this.index.user_id_to_hash[userId];
    if (!userHash) {
      return false;
    }

    const profileDir = this.getProfileDir(userHash);
    this.saveJSON(path.join(profileDir, 'challenges.json'), challenges);

    return true;
  }

  /**
   * Load cognitive challenge set for a user - O(1)
   */
  loadChallenges(userId: string, roleName: string = 'admin'): ChallengeSet | null {
    // RBAC check
    if (!this.rbacPolicy.hasPermission(roleName, MemoryType.LONG_TERM, Permission.READ)) {
      throw new Error(
        `Permission denied: Role '${roleName}' cannot read challenge sets`
      );
    }

    const userHash = this.index.user_id_to_hash[userId];
    if (!userHash) {
      return null;
    }

    const profileDir = this.getProfileDir(userHash);
    return this.loadJSON<ChallengeSet>(path.join(profileDir, 'challenges.json'));
  }

  // ===========================================================================
  // SECURITY EVENTS (AUDIT LOG)
  // ===========================================================================

  /**
   * Log security event - O(1)
   */
  logEvent(event: Omit<SecurityEvent, 'event_id'>, roleName: string = 'admin'): string {
    // RBAC check
    if (!this.rbacPolicy.hasPermission(roleName, MemoryType.LONG_TERM, Permission.WRITE)) {
      throw new Error(
        `Permission denied: Role '${roleName}' cannot write security events`
      );
    }

    // Hash event content for ID
    const content = JSON.stringify({
      user_id: event.user_id,
      timestamp: event.timestamp,
      event_type: event.event_type,
    });
    const eventHash = this.hash(content);

    // Create full event
    const fullEvent: SecurityEvent = {
      ...event,
      event_id: eventHash,
    };

    // Create event directory
    const eventDir = this.getEventDir(eventHash);
    if (!fs.existsSync(eventDir)) {
      fs.mkdirSync(eventDir, { recursive: true });
    }

    // Save event
    this.saveJSON(path.join(eventDir, 'event.json'), fullEvent);

    // Save metadata
    const metadata = {
      event_hash: eventHash,
      user_id: event.user_id,
      timestamp: event.timestamp,
      event_type: event.event_type,
      is_alert: ['duress_detected', 'coercion_detected', 'impersonation_detected', 'panic_code_detected', 'operation_blocked'].includes(event.event_type),
    };

    this.saveJSON(path.join(eventDir, 'metadata.json'), metadata);

    // Update user's last event
    if (this.index.profiles[event.user_id]) {
      this.index.profiles[event.user_id].last_event = eventHash;
    }

    this.updateStatistics();
    this.saveIndex();

    return eventHash;
  }

  /**
   * Get events for a user - O(n) where n = events for that user
   */
  getUserEvents(
    userId: string,
    limit: number = 100,
    roleName: string = 'admin'
  ): SecurityEvent[] {
    // RBAC check
    if (!this.rbacPolicy.hasPermission(roleName, MemoryType.LONG_TERM, Permission.READ)) {
      throw new Error(
        `Permission denied: Role '${roleName}' cannot read security events`
      );
    }

    const eventsDir = path.join(this.baseDir, EVENTS_DIR);
    if (!fs.existsSync(eventsDir)) {
      return [];
    }

    const events: SecurityEvent[] = [];
    const eventHashes = fs.readdirSync(eventsDir);

    for (const hash of eventHashes) {
      if (hash === INDEX_FILE) continue;

      const eventPath = path.join(eventsDir, hash, 'event.json');
      if (fs.existsSync(eventPath)) {
        const event = this.loadJSON<SecurityEvent>(eventPath);
        if (event && event.user_id === userId) {
          events.push(event);
        }
      }
    }

    // Sort by timestamp (newest first) and limit
    return events
      .sort((a, b) => b.timestamp - a.timestamp)
      .slice(0, limit);
  }

  /**
   * Get recent alerts (last 24h) - O(n) where n = total events
   */
  getRecentAlerts(
    hours: number = 24,
    roleName: string = 'admin'
  ): SecurityEvent[] {
    // RBAC check
    if (!this.rbacPolicy.hasPermission(roleName, MemoryType.LONG_TERM, Permission.READ)) {
      throw new Error(
        `Permission denied: Role '${roleName}' cannot read security events`
      );
    }

    const cutoff = Date.now() - hours * 60 * 60 * 1000;
    const eventsDir = path.join(this.baseDir, EVENTS_DIR);

    if (!fs.existsSync(eventsDir)) {
      return [];
    }

    const alerts: SecurityEvent[] = [];
    const eventHashes = fs.readdirSync(eventsDir);

    for (const hash of eventHashes) {
      if (hash === INDEX_FILE) continue;

      const eventPath = path.join(eventsDir, hash, 'event.json');
      if (fs.existsSync(eventPath)) {
        const event = this.loadJSON<SecurityEvent>(eventPath);
        if (
          event &&
          event.timestamp >= cutoff &&
          ['duress_detected', 'coercion_detected', 'impersonation_detected', 'panic_code_detected', 'operation_blocked'].includes(event.event_type)
        ) {
          alerts.push(event);
        }
      }
    }

    return alerts.sort((a, b) => b.timestamp - a.timestamp);
  }

  // ===========================================================================
  // STATISTICS
  // ===========================================================================

  /**
   * Get storage statistics - O(1)
   */
  getStatistics(): SecurityStorageIndex['statistics'] {
    return this.index.statistics;
  }

  /**
   * Get profile metadata - O(1)
   */
  getProfileMetadata(userId: string): ProfileMetadata | null {
    return this.index.profiles[userId] || null;
  }

  // ===========================================================================
  // INTERNAL HELPERS
  // ===========================================================================

  /**
   * Hash content using SHA256 - O(1)
   */
  private hash(content: string): string {
    return crypto.createHash('sha256').update(content).digest('hex');
  }

  /**
   * Get profile directory path
   */
  private getProfileDir(userHash: string): string {
    return path.join(this.baseDir, PROFILES_DIR, userHash);
  }

  /**
   * Get event directory path
   */
  private getEventDir(eventHash: string): string {
    return path.join(this.baseDir, EVENTS_DIR, eventHash);
  }

  /**
   * Save JSON to file
   */
  private saveJSON(filePath: string, data: any): void {
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
  }

  /**
   * Load JSON from file
   */
  private loadJSON<T>(filePath: string): T | null {
    if (!fs.existsSync(filePath)) {
      return null;
    }
    const content = fs.readFileSync(filePath, 'utf-8');
    return JSON.parse(content);
  }

  /**
   * Update statistics
   */
  private updateStatistics(): void {
    const profileCount = Object.keys(this.index.profiles).length;

    // Count events in last 24h
    const alerts = this.getRecentAlerts(24, 'admin');

    this.index.statistics = {
      total_profiles: profileCount,
      total_events: this.countTotalEvents(),
      alerts_last_24h: alerts.length,
    };
  }

  /**
   * Count total events
   */
  private countTotalEvents(): number {
    const eventsDir = path.join(this.baseDir, EVENTS_DIR);
    if (!fs.existsSync(eventsDir)) {
      return 0;
    }

    const eventHashes = fs.readdirSync(eventsDir);
    return eventHashes.filter((hash) => hash !== INDEX_FILE).length;
  }

  /**
   * Load index from disk - O(1)
   */
  private loadIndex(): SecurityStorageIndex {
    const indexPath = path.join(this.baseDir, INDEX_FILE);

    if (fs.existsSync(indexPath)) {
      const content = fs.readFileSync(indexPath, 'utf-8');
      return JSON.parse(content);
    }

    return {
      profiles: {},
      user_id_to_hash: {},
      statistics: {
        total_profiles: 0,
        total_events: 0,
        alerts_last_24h: 0,
      },
    };
  }

  /**
   * Save index to disk - O(1)
   */
  private saveIndex(): void {
    const indexPath = path.join(this.baseDir, INDEX_FILE);
    fs.writeFileSync(indexPath, JSON.stringify(this.index, null, 2));
  }

  /**
   * Ensure directories exist
   */
  private ensureDirectories(): void {
    const dirs = [
      this.baseDir,
      path.join(this.baseDir, PROFILES_DIR),
      path.join(this.baseDir, EVENTS_DIR),
    ];

    for (const dir of dirs) {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
    }
  }
}
