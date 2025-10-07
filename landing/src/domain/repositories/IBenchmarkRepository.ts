import { BenchmarkResult } from '../entities/BenchmarkResult';

/**
 * Domain Repository Interface: IBenchmarkRepository
 * Defines the contract for storing and retrieving benchmark results
 */
export interface IBenchmarkRepository {
  /**
   * Save a benchmark result
   */
  save(result: BenchmarkResult): Promise<void>;

  /**
   * Get all benchmark results
   */
  getAll(): Promise<BenchmarkResult[]>;

  /**
   * Get benchmark results for a specific system
   */
  getBySystem(systemName: string): Promise<BenchmarkResult[]>;

  /**
   * Clear all benchmark results
   */
  clear(): Promise<void>;
}
