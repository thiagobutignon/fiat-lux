# Integrating WIA into Your AGI System

## Quick Start

### 1. Install

```bash
# WIA is included in the standard AGI package
npm install @agi-recursive/core

# Or add to existing project
npm install @agi-recursive/workforce-impact
```

### 2. Basic Integration

```typescript
import { MetaAgent } from '@agi-recursive/core';
import { WorkforceImpactAssessor } from '@agi-recursive/workforce-impact';

class ProductionAGI {
  private meta: MetaAgent;
  private wia: WorkforceImpactAssessor;

  constructor() {
    this.meta = new MetaAgent(process.env.API_KEY);
    this.wia = new WorkforceImpactAssessor({
      mrh_threshold: -0.1,
      audit_logging: true,
    });
  }

  async proposeAutomation(proposal: AutomationProposal) {
    // Step 1: WIA Assessment
    const assessment = this.wia.assessProposal(proposal);

    if (!assessment.approved) {
      return {
        status: 'requires_review',
        reasons: assessment.recommendations,
        wia_score: assessment.wia_score,
      };
    }

    // Step 2: Proceed with implementation
    return {
      status: 'approved',
      wia_score: assessment.wia_score,
      risk_level: assessment.risk_level,
    };
  }
}
```

### 3. With Constitutional Layer

```typescript
import { Constitution } from '@agi-recursive/constitution';
import { WorkforceImpactAssessor } from '@agi-recursive/workforce-impact';

class EthicalAGI {
  private constitution: Constitution;
  private wia: WorkforceImpactAssessor;

  constructor() {
    this.constitution = new Constitution();

    // Add workforce principles to constitution
    this.constitution.addPrinciple({
      id: 'workforce-protection',
      text: 'System must not eliminate more than 10% of workforce without retraining programs',
      priority: 1,
      category: 'social-responsibility',
    });

    this.wia = new WorkforceImpactAssessor({
      constitutional_integration: true,
    });
  }

  async validateProposal(proposal: AutomationProposal) {
    // Constitutional check
    const constitutionalViolations = this.constitution.check(
      `Proposing automation that affects ${proposal.workforce_change.jobs_affected} jobs`
    );

    // WIA check
    const wiaAssessment = this.wia.assessProposal(proposal);

    return {
      constitutional_compliant: constitutionalViolations.length === 0,
      wia_compliant: wiaAssessment.mrh_compliant,
      approved: constitutionalViolations.length === 0 && wiaAssessment.approved,
      details: {
        violations: constitutionalViolations,
        wia_score: wiaAssessment.wia_score,
        recommendations: wiaAssessment.recommendations,
      }
    };
  }
}
```

## Production Deployment Pattern

### Standard 4-Layer Architecture

```typescript
import {
  Constitution,
  AttentionTracker,
  WorkforceImpactAssessor,
  EpisodicMemory,
  MetaAgent,
} from '@agi-recursive/core';

export class ProductionAGISystem {
  // Layer 1: Governance
  private constitution: Constitution;

  // Layer 2: Interpretability
  private attention: AttentionTracker;

  // Layer 3: Social Responsibility
  private wia: WorkforceImpactAssessor;

  // Layer 4: Learning
  private memory: EpisodicMemory;

  // Orchestration
  private meta: MetaAgent;

  constructor(apiKey: string) {
    // Initialize all four core components
    this.constitution = new Constitution();
    this.attention = new AttentionTracker();
    this.wia = new WorkforceImpactAssessor();
    this.memory = new EpisodicMemory();

    this.meta = new MetaAgent(apiKey);
  }

  async processQuery(query: string) {
    // Track attention
    this.attention.startTracking(query);

    // Check constitution
    const violations = this.constitution.check(query);
    if (violations.length > 0) {
      throw new Error(`Constitutional violations: ${violations.map(v => v.reason).join(', ')}`);
    }

    // Process with meta-agent
    const result = await this.meta.process(query);

    // Store in memory
    this.memory.addEpisode(query, result.final_answer, /* ... */);

    // Finalize attention
    const attentionData = this.attention.finalizeAttention(result);

    return {
      answer: result.final_answer,
      attention: attentionData,
      constitutional_compliant: violations.length === 0,
    };
  }

  async evaluateAutomation(proposal: AutomationProposal) {
    // Use WIA for automation decisions
    const assessment = this.wia.assessProposal(proposal);

    // Log to memory for learning
    if (this.memory) {
      this.memory.addEpisode(
        `automation_proposal:${proposal.name}`,
        JSON.stringify(assessment),
        ['automation', 'workforce'],
        ['operations'],
        ['workforce_assessor'],
        0,
        assessment.approved,
        assessment.constitutional_alignment,
        [],
        []
      );
    }

    return assessment;
  }

  // Compliance reporting
  exportComplianceReport() {
    return {
      constitution: this.constitution.exportPrinciples(),
      attention: this.attention.getStatistics(),
      workforce: this.wia.exportAuditLog(),
      memory: this.memory.export(),
    };
  }
}
```

## Integration Patterns

### Pattern 1: Pre-Deployment Gate

Use WIA as a gate before deploying automation:

```typescript
class AutomationPipeline {
  async deploy(proposal: AutomationProposal) {
    // Gate 1: WIA Assessment
    const wiaCheck = await this.wia.assessProposal(proposal);
    if (!wiaCheck.approved) {
      throw new DeploymentError('WIA check failed', wiaCheck.recommendations);
    }

    // Gate 2: Security Review
    await this.securityReview(proposal);

    // Gate 3: Performance Testing
    await this.loadTest(proposal);

    // All gates passed - deploy
    return await this.execute(proposal);
  }
}
```

### Pattern 2: Continuous Monitoring

Monitor workforce impact during rollout:

```typescript
class GradualRollout {
  async rolloutWithMonitoring(proposal: AutomationProposal) {
    const phases = [0.1, 0.25, 0.5, 1.0]; // 10%, 25%, 50%, 100%

    for (const phase of phases) {
      // Scale proposal to phase
      const phasedProposal = this.scaleProposal(proposal, phase);

      // Reassess at each phase
      const assessment = this.wia.assessProposal(phasedProposal);

      if (!assessment.approved) {
        console.log(`Halting at ${phase * 100}% - reassessment required`);
        return { halted_at: phase, reason: assessment.recommendations };
      }

      // Deploy phase
      await this.deployPhase(phasedProposal);

      // Wait and measure
      await this.sleep(30 * 24 * 60 * 60 * 1000); // 30 days
    }

    return { status: 'complete', phases_deployed: phases.length };
  }
}
```

### Pattern 3: Decision Support Dashboard

```typescript
class AutomationDashboard {
  renderProposal(proposal: AutomationProposal) {
    const assessment = this.wia.assessProposal(proposal);

    return {
      proposal_name: proposal.name,
      impact: {
        wia_score: assessment.wia_score,
        mrh_compliant: assessment.mrh_compliant,
        risk_level: assessment.risk_level,
        jobs_affected: proposal.workforce_change.jobs_affected,
        jobs_created: proposal.workforce_change.jobs_created,
        jobs_eliminated: proposal.workforce_change.jobs_eliminated,
      },
      decision: {
        approved: assessment.approved,
        requires_review: assessment.risk_level === 'high' || assessment.risk_level === 'critical',
      },
      recommendations: assessment.recommendations,
      financial: {
        estimated_cost: proposal.estimated_cost,
        estimated_savings: proposal.estimated_savings,
        roi: (proposal.estimated_savings - proposal.estimated_cost) / proposal.estimated_cost,
      },
      compliance: {
        constitutional_alignment: assessment.constitutional_alignment,
        audit_trail: assessment.audit_trail,
      }
    };
  }
}
```

## Testing Your Integration

### Unit Tests

```typescript
import { describe, it, expect } from 'vitest';
import { WorkforceImpactAssessor } from '@agi-recursive/workforce-impact';

describe('WIA Integration', () => {
  it('should reject high-impact proposals', () => {
    const wia = new WorkforceImpactAssessor();

    const proposal = {
      name: 'Test',
      description: 'High impact automation',
      estimated_cost: 100000,
      estimated_savings: 200000,
      workforce_change: {
        jobs_affected: 100,
        jobs_created: 10,
        jobs_eliminated: 50, // 40% elimination
        jobs_transformed: 0,
        timeline_months: 6,
        retraining_required: false,
        affected_departments: ['Operations'],
      },
      implementation_timeline: 6,
      reversibility: 'low',
    };

    const assessment = wia.assessProposal(proposal);

    expect(assessment.mrh_compliant).toBe(false);
    expect(assessment.approved).toBe(false);
    expect(assessment.risk_level).toBe('critical');
  });

  it('should approve compliant proposals', () => {
    const wia = new WorkforceImpactAssessor();

    const proposal = {
      name: 'Test',
      description: 'Balanced automation',
      estimated_cost: 100000,
      estimated_savings: 200000,
      workforce_change: {
        jobs_affected: 100,
        jobs_created: 20,
        jobs_eliminated: 10,
        jobs_transformed: 30,
        timeline_months: 24,
        retraining_required: true,
        affected_departments: ['Operations'],
      },
      implementation_timeline: 24,
      reversibility: 'medium',
    };

    const assessment = wia.assessProposal(proposal);

    expect(assessment.mrh_compliant).toBe(true);
    expect(assessment.approved).toBe(true);
  });
});
```

## Compliance Checklist

- [ ] WIA assessor initialized with audit logging
- [ ] All automation proposals assessed before implementation
- [ ] High-risk proposals routed for human review
- [ ] Audit logs exported monthly for compliance
- [ ] Statistics tracked and reported to leadership
- [ ] Integration with Constitutional Layer verified
- [ ] Gradual rollout enabled for large impacts
- [ ] Retraining programs budgeted for transformations
- [ ] Reversibility considered in proposal design
- [ ] Documentation maintained for regulatory review

## Troubleshooting

### Issue: All proposals rejected

**Cause**: MRH threshold too strict

**Solution**: Review threshold setting, consider custom threshold with justification:

```typescript
const wia = new WorkforceImpactAssessor({
  mrh_threshold: -0.15, // Allow 15% reduction with justification
});
```

### Issue: No audit logs

**Cause**: Audit logging disabled

**Solution**: Enable in configuration:

```typescript
const wia = new WorkforceImpactAssessor({
  audit_logging: true,
});
```

### Issue: Constitutional alignment always 1.0

**Cause**: Constitutional integration disabled

**Solution**: Enable integration:

```typescript
const wia = new WorkforceImpactAssessor({
  constitutional_integration: true,
});
```

## Best Practices Summary

1. **Assess Early**: Evaluate proposals before significant development investment
2. **Phase Large Changes**: Use gradual rollout for high-impact automations
3. **Document Everything**: Maintain complete audit trails for compliance
4. **Human Review**: Route high-risk proposals to human decision makers
5. **Retraining Budget**: Allocate resources for employee transformation
6. **Monitor Continuously**: Track actual impact vs. projected impact
7. **Iterate Proposals**: Redesign rejected proposals based on recommendations
8. **Constitutional Alignment**: Integrate with governance frameworks
9. **Transparency**: Share WIA scores with stakeholders
10. **Regular Audits**: Review statistics and export compliance reports

## Next Steps

1. Read the [WIA Standard documentation](/docs/WIA_STANDARD.md)
2. Review the [API Reference](/docs/WIA_API.md)
3. Explore [example templates](/examples/wia-templates/)
4. Join the [community discussion](https://github.com/agi-standards/workforce-impact/discussions)
5. Consider [WIA certification](https://agi-ethics.org/wia-certification)

---

*Responsible AGI deployment starts with workforce impact assessment.*
