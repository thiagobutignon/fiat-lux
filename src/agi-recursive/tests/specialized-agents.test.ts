/**
 * @file specialized-agents.test.ts
 * Tests for all Specialized Agents - Domain-specific expert agents
 *
 * Agents tested:
 * - FinancialAgent: Personal finance, budgeting, spending analysis
 * - BiologyAgent: Biological systems, homeostasis, abstraction mapping
 * - SystemsAgent: Systems thinking, feedback loops, leverage points
 * - ArchitectureAgent: Software architecture, design patterns, system design
 * - LinguisticsAgent: Language, communication, NLP analysis
 *
 * Key capabilities tested:
 * - Agent instantiation
 * - Domain identification
 * - Inheritance from SpecializedAgent
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { FinancialAgent } from '../agents/financial-agent';
import { BiologyAgent } from '../agents/biology-agent';
import { SystemsAgent } from '../agents/systems-agent';
import { ArchitectureAgent } from '../agents/architecture-agent';
import { LinguisticsAgent } from '../agents/linguistics-agent';

// Mock the SpecializedAgent base class to avoid API calls
vi.mock('../core/meta-agent', async () => {
  const actual = await vi.importActual('../core/meta-agent');
  return {
    ...actual,
    SpecializedAgent: class MockSpecializedAgent {
      private apiKey: string;
      private systemPrompt: string;
      private confidenceThreshold: number;

      constructor(apiKey: string, systemPrompt: string, confidenceThreshold: number = 0.5) {
        this.apiKey = apiKey;
        this.systemPrompt = systemPrompt;
        this.confidenceThreshold = confidenceThreshold;
      }

      async process(query: string) {
        return {
          answer: `Mock response for: ${query}`,
          concepts: ['mock_concept'],
          suggestions_to_invoke: [],
          confidence: 0.9,
          reasoning: 'Mock reasoning',
        };
      }

      getDomain(): string {
        return 'base';
      }
    },
  };
});

describe('FinancialAgent', () => {
  let agent: FinancialAgent;

  beforeEach(() => {
    agent = new FinancialAgent('test-api-key');
  });

  describe('Constructor', () => {
    it('should create instance', () => {
      expect(agent).toBeInstanceOf(FinancialAgent);
    });

    it('should accept API key', () => {
      const customAgent = new FinancialAgent('custom-key');
      expect(customAgent).toBeInstanceOf(FinancialAgent);
    });
  });

  describe('getDomain', () => {
    it('should return financial domain', () => {
      expect(agent.getDomain()).toBe('financial');
    });
  });

  describe('Domain Knowledge', () => {
    it('should specialize in personal finance', () => {
      const domain = agent.getDomain();
      expect(domain).toBe('financial');
    });
  });
});

describe('BiologyAgent', () => {
  let agent: BiologyAgent;

  beforeEach(() => {
    agent = new BiologyAgent('test-api-key');
  });

  describe('Constructor', () => {
    it('should create instance', () => {
      expect(agent).toBeInstanceOf(BiologyAgent);
    });

    it('should accept API key', () => {
      const customAgent = new BiologyAgent('custom-key');
      expect(customAgent).toBeInstanceOf(BiologyAgent);
    });
  });

  describe('getDomain', () => {
    it('should return biology domain', () => {
      expect(agent.getDomain()).toBe('biology');
    });
  });

  describe('Domain Knowledge', () => {
    it('should specialize in biological systems', () => {
      const domain = agent.getDomain();
      expect(domain).toBe('biology');
    });
  });
});

describe('SystemsAgent', () => {
  let agent: SystemsAgent;

  beforeEach(() => {
    agent = new SystemsAgent('test-api-key');
  });

  describe('Constructor', () => {
    it('should create instance', () => {
      expect(agent).toBeInstanceOf(SystemsAgent);
    });

    it('should accept API key', () => {
      const customAgent = new SystemsAgent('custom-key');
      expect(customAgent).toBeInstanceOf(SystemsAgent);
    });
  });

  describe('getDomain', () => {
    it('should return systems domain', () => {
      expect(agent.getDomain()).toBe('systems');
    });
  });

  describe('Domain Knowledge', () => {
    it('should specialize in systems thinking', () => {
      const domain = agent.getDomain();
      expect(domain).toBe('systems');
    });
  });
});

describe('ArchitectureAgent', () => {
  let agent: ArchitectureAgent;

  beforeEach(() => {
    agent = new ArchitectureAgent('test-api-key');
  });

  describe('Constructor', () => {
    it('should create instance', () => {
      expect(agent).toBeInstanceOf(ArchitectureAgent);
    });

    it('should accept API key', () => {
      const customAgent = new ArchitectureAgent('custom-key');
      expect(customAgent).toBeInstanceOf(ArchitectureAgent);
    });
  });

  describe('getDomain', () => {
    it('should return architecture domain', () => {
      expect(agent.getDomain()).toBe('architecture');
    });
  });

  describe('Domain Knowledge', () => {
    it('should specialize in software architecture', () => {
      const domain = agent.getDomain();
      expect(domain).toBe('architecture');
    });
  });
});

describe('LinguisticsAgent', () => {
  let agent: LinguisticsAgent;

  beforeEach(() => {
    agent = new LinguisticsAgent('test-api-key');
  });

  describe('Constructor', () => {
    it('should create instance', () => {
      expect(agent).toBeInstanceOf(LinguisticsAgent);
    });

    it('should accept API key', () => {
      const customAgent = new LinguisticsAgent('custom-key');
      expect(customAgent).toBeInstanceOf(LinguisticsAgent);
    });
  });

  describe('getDomain', () => {
    it('should return linguistics domain', () => {
      expect(agent.getDomain()).toBe('linguistics');
    });
  });

  describe('Domain Knowledge', () => {
    it('should specialize in language and communication', () => {
      const domain = agent.getDomain();
      expect(domain).toBe('linguistics');
    });
  });
});

describe('Agent Integration', () => {
  it('should create all agents with same API key', () => {
    const apiKey = 'shared-key';

    const financial = new FinancialAgent(apiKey);
    const biology = new BiologyAgent(apiKey);
    const systems = new SystemsAgent(apiKey);
    const architecture = new ArchitectureAgent(apiKey);
    const linguistics = new LinguisticsAgent(apiKey);

    expect(financial).toBeInstanceOf(FinancialAgent);
    expect(biology).toBeInstanceOf(BiologyAgent);
    expect(systems).toBeInstanceOf(SystemsAgent);
    expect(architecture).toBeInstanceOf(ArchitectureAgent);
    expect(linguistics).toBeInstanceOf(LinguisticsAgent);
  });

  it('should have unique domains for each agent', () => {
    const financial = new FinancialAgent('key');
    const biology = new BiologyAgent('key');
    const systems = new SystemsAgent('key');
    const architecture = new ArchitectureAgent('key');
    const linguistics = new LinguisticsAgent('key');

    const domains = [
      financial.getDomain(),
      biology.getDomain(),
      systems.getDomain(),
      architecture.getDomain(),
      linguistics.getDomain(),
    ];

    // All domains should be unique
    const uniqueDomains = new Set(domains);
    expect(uniqueDomains.size).toBe(5);
  });

  it('should have expected domain names', () => {
    const financial = new FinancialAgent('key');
    const biology = new BiologyAgent('key');
    const systems = new SystemsAgent('key');
    const architecture = new ArchitectureAgent('key');
    const linguistics = new LinguisticsAgent('key');

    expect(financial.getDomain()).toBe('financial');
    expect(biology.getDomain()).toBe('biology');
    expect(systems.getDomain()).toBe('systems');
    expect(architecture.getDomain()).toBe('architecture');
    expect(linguistics.getDomain()).toBe('linguistics');
  });

  it('should support agent swapping with same interface', () => {
    // All agents share the same base interface
    const agents = [
      new FinancialAgent('key'),
      new BiologyAgent('key'),
      new SystemsAgent('key'),
      new ArchitectureAgent('key'),
      new LinguisticsAgent('key'),
    ];

    // All should have getDomain() method
    agents.forEach((agent) => {
      expect(typeof agent.getDomain).toBe('function');
      expect(typeof agent.getDomain()).toBe('string');
    });
  });
});

describe('Agent Domains', () => {
  describe('Domain Uniqueness', () => {
    it('should not have overlapping domains', () => {
      const domains = {
        financial: new FinancialAgent('key').getDomain(),
        biology: new BiologyAgent('key').getDomain(),
        systems: new SystemsAgent('key').getDomain(),
        architecture: new ArchitectureAgent('key').getDomain(),
        linguistics: new LinguisticsAgent('key').getDomain(),
      };

      const allDomains = Object.values(domains);
      const uniqueDomains = new Set(allDomains);

      expect(uniqueDomains.size).toBe(allDomains.length);
    });
  });

  describe('Domain Naming', () => {
    it('should use lowercase domain names', () => {
      const agents = [
        new FinancialAgent('key'),
        new BiologyAgent('key'),
        new SystemsAgent('key'),
        new ArchitectureAgent('key'),
        new LinguisticsAgent('key'),
      ];

      agents.forEach((agent) => {
        const domain = agent.getDomain();
        expect(domain).toBe(domain.toLowerCase());
      });
    });

    it('should use single-word domain names', () => {
      const agents = [
        new FinancialAgent('key'),
        new BiologyAgent('key'),
        new SystemsAgent('key'),
        new ArchitectureAgent('key'),
        new LinguisticsAgent('key'),
      ];

      agents.forEach((agent) => {
        const domain = agent.getDomain();
        expect(domain).not.toContain(' ');
        expect(domain).not.toContain('-');
        expect(domain).not.toContain('_');
      });
    });
  });
});

describe('Agent Factory Pattern', () => {
  it('should support creating agents from domain name', () => {
    const createAgent = (domain: string, apiKey: string) => {
      switch (domain) {
        case 'financial':
          return new FinancialAgent(apiKey);
        case 'biology':
          return new BiologyAgent(apiKey);
        case 'systems':
          return new SystemsAgent(apiKey);
        case 'architecture':
          return new ArchitectureAgent(apiKey);
        case 'linguistics':
          return new LinguisticsAgent(apiKey);
        default:
          throw new Error(`Unknown domain: ${domain}`);
      }
    };

    const financial = createAgent('financial', 'key');
    const biology = createAgent('biology', 'key');

    expect(financial.getDomain()).toBe('financial');
    expect(biology.getDomain()).toBe('biology');
  });

  it('should map agent class name to domain', () => {
    const mappings = {
      FinancialAgent: 'financial',
      BiologyAgent: 'biology',
      SystemsAgent: 'systems',
      ArchitectureAgent: 'architecture',
      LinguisticsAgent: 'linguistics',
    };

    const financial = new FinancialAgent('key');
    const biology = new BiologyAgent('key');
    const systems = new SystemsAgent('key');
    const architecture = new ArchitectureAgent('key');
    const linguistics = new LinguisticsAgent('key');

    expect(financial.getDomain()).toBe(mappings.FinancialAgent);
    expect(biology.getDomain()).toBe(mappings.BiologyAgent);
    expect(systems.getDomain()).toBe(mappings.SystemsAgent);
    expect(architecture.getDomain()).toBe(mappings.ArchitectureAgent);
    expect(linguistics.getDomain()).toBe(mappings.LinguisticsAgent);
  });
});

describe('Agent API Key Handling', () => {
  it('should accept different API keys per agent', () => {
    const financial = new FinancialAgent('key1');
    const biology = new BiologyAgent('key2');

    // Both should be valid instances
    expect(financial).toBeDefined();
    expect(biology).toBeDefined();
  });

  it('should accept empty string as API key (for testing)', () => {
    const agent = new FinancialAgent('');
    expect(agent).toBeDefined();
  });

  it('should accept special characters in API key', () => {
    const agent = new FinancialAgent('sk-test_123-456_ABC');
    expect(agent).toBeDefined();
  });
});
