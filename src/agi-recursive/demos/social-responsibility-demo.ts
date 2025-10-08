/**
 * Social Responsibility Framework Demo
 *
 * Demonstrates the mandatory social responsibility framework:
 * 1. Workforce Impact Assessment (WIA) - evaluates automation proposals
 * 2. Responsible AGI Certification - validates companies against MRH standards
 * 3. Quarterly Report Validation - ensures ongoing compliance
 *
 * This shows how the system enforces social responsibility at every layer.
 */

import {
  WorkforceImpactAssessor,
  AutomationProposal,
} from '../core/workforce-impact-assessor';
import {
  ResponsibleAGICertification,
  CompanyProfile,
  DeploymentMetrics,
} from '../core/responsible-agi-certification';
import { getGlobalTelemetry } from '../core/impact-telemetry';

// ============================================================================
// Setup
// ============================================================================

console.log('⚖️  Social Responsibility Framework Demo\n');
console.log('Demonstrating how the AGI system enforces social responsibility');
console.log('through mandatory WIA, certification, and reporting.\n');
console.log('='.repeat(70));

const wia = new WorkforceImpactAssessor({
  mrh_threshold: -0.1, // Max 10% displacement
  audit_logging: true,
  constitutional_integration: true,
});

const cert = new ResponsibleAGICertification();
const telemetry = getGlobalTelemetry();

// ============================================================================
// Demo 1: Workforce Impact Assessment
// ============================================================================

console.log('\n📊 DEMO 1: Workforce Impact Assessment (WIA)\n');
console.log('Evaluating automation proposals for workforce impact...\n');

// Scenario A: MRH-Compliant Proposal (APPROVED)
console.log('Scenario A: Data Entry Automation (MRH-Compliant)');
console.log('-'.repeat(70));

const dataEntryProposal: AutomationProposal = {
  name: 'RPA for Data Entry',
  description: 'Automate repetitive data entry tasks',
  estimated_cost: 150000,
  estimated_savings: 600000,
  workforce_change: {
    jobs_affected: 200,
    jobs_created: 15, // RPA developers, process analysts
    jobs_eliminated: 20, // Data entry clerks
    jobs_transformed: 180, // Clerks become analysts
    timeline_months: 24,
    retraining_required: true,
    affected_departments: ['operations', 'finance'],
  },
  implementation_timeline: 24,
  reversibility: 'high',
};

const assessment1 = wia.assessProposal(dataEntryProposal);

console.log(`  Name: ${dataEntryProposal.name}`);
console.log(`  WIA Score: ${assessment1.wia_score.toFixed(3)} (${((assessment1.wia_score * 100).toFixed(1))}% net change)`);
console.log(`  MRH Compliant: ${assessment1.mrh_compliant ? '✅ YES' : '❌ NO'}`);
console.log(`  Risk Level: ${assessment1.risk_level.toUpperCase()}`);
console.log(`  Approved: ${assessment1.approved ? '✅ YES' : '❌ NO'}`);
if (assessment1.recommendations.length > 0) {
  console.log(`  Recommendations:`);
  assessment1.recommendations.forEach((r) => console.log(`    • ${r}`));
}
console.log();

// Scenario B: MRH-Violating Proposal (REJECTED)
console.log('Scenario B: Mass Customer Service Automation (MRH-VIOLATING)');
console.log('-'.repeat(70));

const customerServiceProposal: AutomationProposal = {
  name: 'AI Customer Service',
  description: 'Replace Level 1 support with AI chatbot',
  estimated_cost: 300000,
  estimated_savings: 1200000,
  workforce_change: {
    jobs_affected: 150,
    jobs_created: 10, // AI trainers, quality analysts
    jobs_eliminated: 100, // Level 1 agents
    jobs_transformed: 40, // Level 2 agents upskilled
    timeline_months: 18,
    retraining_required: true,
    affected_departments: ['customer_support'],
  },
  implementation_timeline: 18,
  reversibility: 'medium',
};

const assessment2 = wia.assessProposal(customerServiceProposal);

console.log(`  Name: ${customerServiceProposal.name}`);
console.log(`  WIA Score: ${assessment2.wia_score.toFixed(3)} (${((assessment2.wia_score * 100).toFixed(1))}% net change)`);
console.log(`  MRH Compliant: ${assessment2.mrh_compliant ? '✅ YES' : '❌ NO'}`);
console.log(`  Risk Level: ${assessment2.risk_level.toUpperCase()}`);
console.log(`  Approved: ${assessment2.approved ? '✅ YES' : '❌ NO'}`);
if (assessment2.recommendations.length > 0) {
  console.log(`  Recommendations:`);
  assessment2.recommendations.forEach((r) => console.log(`    • ${r}`));
}
console.log();

// Scenario C: Workforce-Positive Proposal (IDEAL)
console.log('Scenario C: Job-Creating Automation (IDEAL)');
console.log('-'.repeat(70));

const jobCreatingProposal: AutomationProposal = {
  name: 'Process Automation with Oversight',
  description: 'Automate tasks while creating oversight roles',
  estimated_cost: 300000,
  estimated_savings: 200000,
  workforce_change: {
    jobs_affected: 100,
    jobs_created: 10,
    jobs_eliminated: 5,
    jobs_transformed: 95,
    timeline_months: 24,
    retraining_required: true,
    affected_departments: ['operations'],
  },
  implementation_timeline: 24,
  reversibility: 'high',
};

const assessment3 = wia.assessProposal(jobCreatingProposal);

console.log(`  Name: ${jobCreatingProposal.name}`);
console.log(`  WIA Score: ${assessment3.wia_score.toFixed(3)} (${((assessment3.wia_score * 100).toFixed(1))}% net change)`);
console.log(`  MRH Compliant: ${assessment3.mrh_compliant ? '✅ YES' : '❌ NO'}`);
console.log(`  Risk Level: ${assessment3.risk_level.toUpperCase()}`);
console.log(`  Approved: ${assessment3.approved ? '✅ YES' : '❌ NO'}`);
if (assessment3.recommendations.length > 0) {
  console.log(`  Recommendations:`);
  assessment3.recommendations.forEach((r) => console.log(`    • ${r}`));
}
console.log();

// ============================================================================
// Demo 2: Responsible AGI Certification
// ============================================================================

console.log('\n🏅 DEMO 2: Responsible AGI Certification\n');
console.log('Evaluating companies for certification tiers...\n');

// Scenario D: Gold Tier - Worker Cooperative
console.log('Scenario D: Worker Cooperative (GOLD TIER)');
console.log('-'.repeat(70));

const cooperative: CompanyProfile = {
  name: 'Tech Workers Cooperative',
  size_employees: 150,
  ceo_to_worker_ratio: 3, // Flat structure
  employee_ownership_percent: 100,
  layoffs_last_24_months: 0,
  layoffs_last_24_months_percent: 0,
  has_labor_violations: false,
  has_democratic_governance: true,
  has_retraining_programs: true,
  quarterly_reports_submitted: 4,
};

const cooperativeMetrics: DeploymentMetrics = {
  total_queries_processed: 50000,
  queries_with_automation_intent: 500,
  jobs_affected: 30,
  jobs_created: 5,
  jobs_eliminated: 2,
  net_workforce_change: 3,
  wia_average_score: 0.1, // 10% positive
  mrh_violations: 0,
  constitutional_violations: 0,
  retraining_budget_usd: 50000,
};

const goldResult = cert.evaluate(cooperative, cooperativeMetrics);

console.log(`  Company: ${cooperative.name}`);
console.log(`  Tier: ${goldResult.tier.toUpperCase()}`);
console.log(`  Approved: ${goldResult.approved ? '✅ YES' : '❌ NO'}`);
console.log(`  Score: ${goldResult.score}/100`);
console.log(`  License Fee: $${goldResult.license_fee_annual.toLocaleString()}/year`);
console.log(`  Discount: ${goldResult.license_discount_percent}%`);
console.log(`  Badge URL: ${goldResult.badge_url}`);
if (goldResult.recommendations.length > 0) {
  console.log(`  Recommendations:`);
  goldResult.recommendations.forEach((r) => console.log(`    • ${r}`));
}
console.log();

// Scenario E: Silver Tier - Progressive Company
console.log('Scenario E: Progressive Tech Company (SILVER TIER)');
console.log('-'.repeat(70));

const progressiveCompany: CompanyProfile = {
  name: 'Progressive Tech Inc',
  size_employees: 500,
  ceo_to_worker_ratio: 25,
  employee_ownership_percent: 15,
  layoffs_last_24_months: 0,
  layoffs_last_24_months_percent: 0,
  has_labor_violations: false,
  has_democratic_governance: false,
  has_retraining_programs: true,
  quarterly_reports_submitted: 4,
};

const progressiveMetrics: DeploymentMetrics = {
  total_queries_processed: 200000,
  queries_with_automation_intent: 2000,
  jobs_affected: 100,
  jobs_created: 8,
  jobs_eliminated: 3,
  net_workforce_change: 5,
  wia_average_score: 0.05, // 5% positive
  mrh_violations: 0,
  constitutional_violations: 2,
  retraining_budget_usd: 150000,
};

const silverResult = cert.evaluate(progressiveCompany, progressiveMetrics);

console.log(`  Company: ${progressiveCompany.name}`);
console.log(`  Tier: ${silverResult.tier.toUpperCase()}`);
console.log(`  Approved: ${silverResult.approved ? '✅ YES' : '❌ NO'}`);
console.log(`  Score: ${silverResult.score}/100`);
console.log(`  License Fee: $${silverResult.license_fee_annual.toLocaleString()}/year`);
console.log(`  Discount: ${silverResult.license_discount_percent}%`);
if (silverResult.recommendations.length > 0) {
  console.log(`  Recommendations:`);
  silverResult.recommendations.forEach((r) => console.log(`    • ${r}`));
}
console.log();

// Scenario F: Disqualified - Recent Layoffs
console.log('Scenario F: Company with Recent Layoffs (DISQUALIFIED)');
console.log('-'.repeat(70));

const layoffCompany: CompanyProfile = {
  name: 'LayoffCorp Inc',
  size_employees: 800,
  ceo_to_worker_ratio: 45,
  employee_ownership_percent: 0,
  layoffs_last_24_months: 50,
  layoffs_last_24_months_percent: 0.06, // 6% - VIOLATION
  has_labor_violations: false,
  has_democratic_governance: false,
  has_retraining_programs: true,
  quarterly_reports_submitted: 3,
};

const layoffMetrics: DeploymentMetrics = {
  total_queries_processed: 100000,
  queries_with_automation_intent: 1000,
  jobs_affected: 50,
  jobs_created: 5,
  jobs_eliminated: 3,
  net_workforce_change: 2,
  wia_average_score: 0.04,
  mrh_violations: 0,
  constitutional_violations: 1,
  retraining_budget_usd: 80000,
};

const disqualifiedResult = cert.evaluate(layoffCompany, layoffMetrics);

console.log(`  Company: ${layoffCompany.name}`);
console.log(`  Tier: ${disqualifiedResult.tier.toUpperCase()}`);
console.log(`  Approved: ${disqualifiedResult.approved ? '✅ YES' : '❌ NO'}`);
console.log(`  Score: ${disqualifiedResult.score}/100`);
console.log(`  Violations:`);
disqualifiedResult.violations.forEach((v) => console.log(`    • ${v}`));
console.log();

// ============================================================================
// Demo 3: Quarterly Report Validation
// ============================================================================

console.log('\n📋 DEMO 3: Quarterly Report Validation\n');
console.log('Validating quarterly compliance reports...\n');

// Scenario G: Valid Report
console.log('Scenario G: Compliant Quarterly Report');
console.log('-'.repeat(70));

const validReport = {
  period: 'Q1 2025',
  employees_start: 100,
  employees_end: 102, // 2% growth
  roles_automated: ['data-entry-clerk'],
  roles_created: ['rpa-developer', 'process-analyst'],
  retraining_participants: 15,
  wia_assessments: [
    {
      wia_score: 0.05,
      mrh_compliant: true,
      approved: true,
      risk_level: 'low' as const,
      recommendations: [],
    },
  ],
};

const validation1 = cert.validateQuarterlyReport(validReport);

console.log(`  Period: ${validReport.period}`);
console.log(`  Valid: ${validation1.valid ? '✅ YES' : '❌ NO'}`);
console.log(`  Workforce Change: ${validReport.employees_start} → ${validReport.employees_end} (+${((validReport.employees_end - validReport.employees_start) / validReport.employees_start * 100).toFixed(1)}%)`);
console.log(`  Roles Automated: ${validReport.roles_automated.length}`);
console.log(`  Roles Created: ${validReport.roles_created.length}`);
console.log(`  Retraining Participants: ${validReport.retraining_participants}`);
if (validation1.issues.length > 0) {
  console.log(`  Issues:`);
  validation1.issues.forEach((i) => console.log(`    • ${i}`));
}
console.log();

// Scenario H: Invalid Report - Excessive Decline
console.log('Scenario H: Non-Compliant Report (Excessive Workforce Decline)');
console.log('-'.repeat(70));

const invalidReport = {
  period: 'Q2 2025',
  employees_start: 100,
  employees_end: 85, // 15% decline - VIOLATION
  roles_automated: ['customer-support-agent'],
  roles_created: ['ai-trainer'],
  retraining_participants: 0, // No retraining!
  wia_assessments: [],
};

const validation2 = cert.validateQuarterlyReport(invalidReport);

console.log(`  Period: ${invalidReport.period}`);
console.log(`  Valid: ${validation2.valid ? '✅ YES' : '❌ NO'}`);
console.log(`  Workforce Change: ${invalidReport.employees_start} → ${invalidReport.employees_end} (${((invalidReport.employees_end - invalidReport.employees_start) / invalidReport.employees_start * 100).toFixed(1)}%)`);
console.log(`  Issues:`);
validation2.issues.forEach((i) => console.log(`    • ${i}`));
console.log();

// ============================================================================
// Summary
// ============================================================================

console.log('\n' + '='.repeat(70));
console.log('📊 Summary of Social Responsibility Framework\n');
console.log('WIA Assessments:');
console.log(`  ✅ Approved: 2/3 proposals (MRH-compliant)`);
console.log(`  ❌ Rejected: 1/3 proposals (MRH-violating)`);
console.log();
console.log('Certifications:');
console.log(`  🥇 Gold: 1 (worker cooperative)`);
console.log(`  🥈 Silver: 1 (workforce-positive deployment)`);
console.log(`  ❌ Disqualified: 1 (recent layoffs)`);
console.log();
console.log('Quarterly Reports:');
console.log(`  ✅ Valid: 1/2`);
console.log(`  ❌ Invalid: 1/2 (MRH violations)`);
console.log();
console.log('Telemetry:');
console.log(`  Status: ${telemetry['config']?.enabled ? 'Enabled' : 'Disabled'} (opt-out via TELEMETRY_ENABLED=false)`);
console.log();
console.log('Key Principles:');
console.log('  • MRH Standard: Max 10% net workforce displacement per year');
console.log('  • Constitutional Governance: AI decisions aligned with human dignity');
console.log('  • Transparent Monitoring: All impact assessments logged and auditable');
console.log('  • Tiered Incentives: Cooperatives get 50% license discount');
console.log('  • Enforcement: Technical (hardcoded WIA) + Legal (license terms) + Social (public badges)');
console.log();
console.log('='.repeat(70));
console.log('\n✅ Demo complete!\n');
