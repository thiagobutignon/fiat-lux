# RFC-0001: InsightLoop Protocol (ILP/1.0)

```
Network Working Group                                      T. Butignon
Request for Comments: 0001                                    H. Gomes
Category: Standards Track                                   R. Barbosa
                                                          October 2025


                    InsightLoop Protocol Version 1.0
                                 (ILP/1.0)
```

## Status of This Memo

This document specifies an Internet standards track protocol for the distributed cognitive computing community, and requests discussion and suggestions for improvements. Distribution of this memo is unlimited.

## Copyright Notice

Copyright (C) Fiat Lux Contributors (2025). All Rights Reserved.

## Abstract

The InsightLoop Protocol (ILP) is an application-level protocol for semantic reasoning exchange between AI agents in distributed cognitive systems. ILP adds a semantic, ethical, and interpretable layer on top of existing model communication protocols (such as Model Context Protocol - MCP), enabling:

1. **Constitutional Governance**: Runtime enforcement of ethical principles and domain boundaries
2. **Attention Tracing**: Complete auditability of which concepts influenced each decision
3. **Cross-Domain Composition**: Semantic translation between specialized knowledge domains
4. **Cognitive Determinism**: Reproducible reasoning chains for debugging and compliance
5. **Self-Evolution**: Automatic knowledge base improvement through episodic memory learning

ILP is designed for AGI systems where multiple specialized agents compose insights recursively, and where transparency, ethics, auditability, and continuous learning are first-class requirements.

The protocol is text-based, uses JSON for message encoding, and supports both request-response and streaming patterns.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Motivation](#2-motivation)
3. [Protocol Overview](#3-protocol-overview)
4. [Architecture](#4-architecture)
   - 4.1 [Components](#41-components)
   - 4.2 [Message Flow](#42-message-flow)
   - 4.3 [Self-Evolution System](#43-self-evolution-system)
5. [Message Format](#5-message-format)
6. [Header Fields](#6-header-fields)
7. [Payload Structure](#7-payload-structure)
8. [Status Codes](#8-status-codes)
9. [Compliance](#9-compliance)
10. [Security Considerations](#10-security-considerations)
11. [IANA Considerations](#11-iana-considerations)
12. [References](#12-references)
13. [Examples](#13-examples)

---

## 1. Introduction

### 1.1 Purpose

The InsightLoop Protocol (ILP) provides a standardized way for AI agents to:
- Exchange semantic reasoning (not just text)
- Enforce constitutional principles at runtime
- Track attention flows for interpretability
- Validate cross-domain compositions
- Maintain audit trails for regulatory compliance

### 1.2 Terminology

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in RFC 2119.

**Agent**: An AI system specialized in a specific domain (e.g., finance, biology, systems theory)

**Meta-Agent**: An orchestrator that coordinates multiple agents

**Slice**: A structured unit of knowledge (concepts, principles, examples)

**Attention Trace**: Record of which concept influenced a decision and by how much

**Constitutional Violation**: Breach of governance principles (epistemic honesty, safety, etc.)

**Domain Boundary**: Limit of an agent's expertise

### 1.3 Relationship to Other Protocols

ILP is designed to operate **on top of** existing model communication protocols:

```
┌─────────────────────────────────────┐
│  Application: AGI Recursive System  │
├─────────────────────────────────────┤
│  ILP/1.0 (Semantic + Ethics)        │ ← THIS PROTOCOL
├─────────────────────────────────────┤
│  MCP/2.0 (Model Communication)      │ ← Transport layer
├─────────────────────────────────────┤
│  HTTP/2 or gRPC                     │ ← Network layer
└─────────────────────────────────────┘
```

**Key Distinction:**
- **MCP**: "How to invoke a model and pass context"
- **ILP**: "How to ensure reasoning is ethical, auditable, and interpretable"

---

## 2. Motivation

### 2.1 Problem Statement

Current LLM communication protocols (MCP, function calling APIs, etc.) focus on **mechanical execution** but lack:

1. **Semantic Awareness**: No understanding of what concepts mean or how they relate
2. **Ethical Enforcement**: Constitution applied at training time, not runtime
3. **Interpretability**: Black-box reasoning with no audit trail
4. **Determinism**: Non-reproducible outputs make debugging impossible
5. **Domain Safety**: No validation of cross-domain semantic corruption

### 2.2 Solution: Cognitive OSI Stack

ILP introduces a **semantic layer** analogous to HTTP over TCP/IP:

| OSI Layer | Internet | Cognitive Network | Protocol |
|-----------|----------|-------------------|----------|
| Application | Web Browser | AGI System | AGI Runtime |
| Presentation | HTML/JSON | Reasoning Format | JSON Schema |
| Session | HTTP/2 | **ILP/1.0** | **THIS PROTOCOL** |
| Transport | TCP | MCP/2.0 | Model Context Protocol |
| Network | IP | Tool Execution | e2b, Gemini Tools |
| Data Link | Ethernet | API Calls | REST/gRPC |
| Physical | Copper/Fiber | Network | Internet |

### 2.3 Design Goals

1. **Auditability**: Every decision traceable to source concepts
2. **Ethics**: Constitutional principles enforced at every step
3. **Composability**: Safe cross-domain knowledge translation
4. **Determinism**: Reproducible reasoning for debugging
5. **Efficiency**: Lazy evaluation, caching, budget control

---

## 3. Protocol Overview

### 3.1 Request-Response Pattern

```
Client                                    Agent
  |                                         |
  |--- THINK /insight (ILP/1.0) ----------→|
  |    Constitutional-Header:               |
  |      Domain: finance                    |
  |      Depth: 2                           |
  |      Budget: $0.02                      |
  |    Attention-Enabled: true              |
  |                                         |
  |←-- 200 OK (ILP/1.0) -------------------|
  |    Reasoning-Trace:                     |
  |      concepts: [diversification: 0.91]  |
  |    Attention-Payload:                   |
  |      top_influencers: [...]             |
```

### 3.2 Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `THINK` | Request reasoning from agent | Query specialist |
| `COMPOSE` | Request synthesis from meta-agent | Combine insights |
| `VALIDATE` | Check constitutional compliance | ACL enforcement |
| `TRANSLATE` | Cross-domain concept mapping | Domain boundaries |
| `TRACE` | Get attention/reasoning trace | Debugging/audit |

### 3.3 Status Codes

| Code | Meaning | Action |
|------|---------|--------|
| `200` | OK | Reasoning successful |
| `207` | Multi-Status | Success with warnings |
| `400` | Bad Request | Invalid query format |
| `403` | Forbidden | Constitutional violation |
| `409` | Conflict | Loop detected |
| `429` | Budget Exceeded | Cost/depth limit hit |
| `451` | Unavailable For Legal Reasons | Domain boundary violation |
| `500` | Internal Error | Agent failure |
| `503` | Service Unavailable | Agent not initialized |

---

## 4. Architecture

### 4.1 Components

```
┌──────────────────────────────────────────┐
│         Meta-Agent (Orchestrator)        │
│  - Query decomposition                   │
│  - Agent invocation                      │
│  - Insight composition                   │
└───────────────┬──────────────────────────┘
                │
                │ ILP/1.0
                │
    ┌───────────┴──────────────┬───────────┐
    │                          │           │
┌───▼────┐  ┌────▼─────┐  ┌───▼────┐      │
│Finance │  │ Biology  │  │Systems │  ... │
│ Agent  │  │  Agent   │  │ Agent  │      │
└───┬────┘  └────┬─────┘  └───┬────┘      │
    │            │            │            │
    └────────────┴────────────┴────────────┘
                 │
    ┌────────────▼─────────────┐
    │  Constitutional Enforcer │ ◄─── ACL
    │  - Epistemic honesty     │
    │  - Budget limits         │
    │  - Loop detection        │
    │  - Safety filters        │
    └──────────────────────────┘
                 │
    ┌────────────▼─────────────┐
    │    Attention Tracker     │
    │  - Concept influences    │
    │  - Decision paths        │
    │  - Audit export          │
    └──────────────────────────┘
```

### 4.2 Message Flow

1. **Client → Meta-Agent**: `THINK` query
2. **Meta-Agent → Constitutional Enforcer**: `VALIDATE` budget
3. **Meta-Agent → Agent**: `THINK` decomposed query
4. **Agent → Slice Navigator**: Retrieve knowledge
5. **Agent → Meta-Agent**: Response with attention traces
6. **Meta-Agent → Constitutional Enforcer**: `VALIDATE` response
7. **Meta-Agent → Attention Tracker**: Record traces
8. **Meta-Agent → Client**: `COMPOSE` final synthesis

### 4.3 Self-Evolution System

The ILP architecture includes a self-evolution subsystem that enables the AGI system to **rewrite its own knowledge slices** based on patterns learned from episodic memory. This creates a continuous learning loop where the system improves its knowledge base through experience.

```
┌──────────────────────────────────────────┐
│         Episodic Memory                  │
│  - Stores all query/response episodes   │
│  - Indexes by concepts and domains      │
│  - Tracks success and confidence        │
└───────────────┬──────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────┐
│     Knowledge Distillation               │
│  - Pattern discovery (frequency ≥ N)    │
│  - Knowledge gap identification         │
│  - Systematic error detection           │
│  - LLM-based synthesis                  │
└───────────────┬──────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────┐
│     Slice Evolution Engine               │
│  - Analyze → Propose → Validate → Deploy│
│  - Constitutional compliance checking   │
│  - Evolution history tracking           │
│  - Rollback capability                  │
└───────────────┬──────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────┐
│         Slice Rewriter                   │
│  - Atomic writes (temp + rename)        │
│  - Automatic backups before changes     │
│  - YAML validation                      │
│  - Safe rollback on failures            │
└───────────────┬──────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────┐
│      Updated Knowledge Base              │
│  - New slices created automatically     │
│  - Existing slices improved over time   │
│  - Complete audit trail maintained      │
└──────────────────────────────────────────┘
```

**Self-Evolution Flow:**

1. **Experience Collection**: User queries create episodes in episodic memory
   - Each episode captures: query, response, concepts, domains, success, confidence
   - Episodes indexed for fast retrieval by concept and domain

2. **Pattern Discovery**: System analyzes memory for recurring patterns
   - Identifies concept combinations appearing ≥ N times
   - Calculates confidence scores based on frequency and success rate
   - Extracts representative queries for each pattern

3. **Candidate Generation**: System proposes evolution candidates
   - **Type**: New slice creation, existing slice update, slice merge, or deprecation
   - **Content**: LLM synthesizes YAML knowledge from pattern data
   - **Validation**: Constitutional compliance scoring (0-1 scale)
   - **Decision**: `should_deploy` flag based on confidence and constitutional score

4. **Evolution Deployment**: Approved candidates deployed safely
   - Atomic file writes prevent partial updates
   - Automatic backups before any changes
   - YAML validation before deployment
   - Evolution history recorded for audit

5. **Continuous Learning**: System repeats cycle automatically
   - Triggers: Scheduled, threshold-based, manual, or continuous
   - Metrics: Total evolutions, successful deployments, rollbacks
   - Performance tracking: Accuracy improvements, cost deltas

**Evolution Types:**

- `CREATED`: New knowledge slice from discovered pattern
- `UPDATED`: Existing slice enhanced with new insights
- `MERGED`: Multiple related slices consolidated
- `DEPRECATED`: Outdated or low-confidence slice removed

**Evolution Triggers:**

- `SCHEDULED`: Periodic analysis (e.g., daily, weekly)
- `THRESHOLD`: Triggered when episodic memory reaches N episodes
- `MANUAL`: Explicit user request for evolution
- `CONTINUOUS`: Real-time analysis after each interaction

**Safety Mechanisms:**

1. **Constitutional Validation**: All candidates scored against governance principles
2. **Approval Gates**: Only candidates with `should_deploy=true` are deployed
3. **Atomic Operations**: File writes are all-or-nothing (no partial updates)
4. **Automatic Backups**: Every change creates timestamped backup
5. **Rollback Capability**: Failed evolutions can be reverted instantly
6. **Observability**: Complete logs, metrics, and traces for all operations

**Benefits:**

- **Continuous Improvement**: Knowledge base improves with every interaction
- **Reduced Manual Work**: No human intervention needed for slice creation
- **Pattern Learning**: Discovers emergent concepts from user behavior
- **Quality Assurance**: Constitutional validation ensures safe evolution
- **Full Auditability**: Complete history of what changed, when, and why

This self-evolution system creates a true learning AGI that improves its knowledge base through experience, while maintaining safety and interpretability.

---

## 5. Message Format

### 5.1 Request Format

```
THINK /insight ILP/1.0
Constitutional-Header: <governance-params>
Attention-Enabled: true|false
Content-Type: application/json
Content-Length: <bytes>

<JSON payload>
```

### 5.2 Response Format

```
ILP/1.0 200 OK
Reasoning-Trace: <decision-path>
Attention-Payload: <concept-influences>
Constitutional-Status: PASSED|WARNING|VIOLATION
Content-Type: application/json
Content-Length: <bytes>

<JSON payload>
```

### 5.3 Example Request

```http
THINK /insight ILP/1.0
Constitutional-Header: {
  "domain": "finance",
  "depth": 2,
  "max_depth": 5,
  "budget_usd": 0.02,
  "max_budget_usd": 1.0,
  "enforce_epistemic_honesty": true,
  "require_reasoning_trace": true
}
Attention-Enabled: true
Content-Type: application/json
Content-Length: 156

{
  "query": "How can I optimize my budget?",
  "context": {
    "previous_agents": ["meta"],
    "invocation_count": 1,
    "cost_so_far": 0.005
  }
}
```

### 5.4 Example Response

```http
ILP/1.0 200 OK
Reasoning-Trace: {
  "decision_path": [
    "Invoked financial_agent",
    "Loaded slice: finance/budgeting",
    "Composed with biology analogy"
  ]
}
Attention-Payload: {
  "top_influencers": [
    {
      "concept": "homeostasis",
      "slice": "biology/cells",
      "weight": 0.91,
      "reasoning": "Budget stabilization analogous to cellular homeostasis"
    },
    {
      "concept": "feedback_loop",
      "slice": "systems/control",
      "weight": 0.84,
      "reasoning": "Monitoring and correction mechanism"
    }
  ]
}
Constitutional-Status: PASSED
Content-Type: application/json
Content-Length: 487

{
  "answer": "Your budget can be stabilized using a homeostatic feedback system...",
  "concepts": ["homeostasis", "feedback_loop", "diversification"],
  "confidence": 0.89,
  "reasoning": "Combined financial best practices with biological homeostasis principles",
  "sources": ["finance/budgeting.md", "biology/cells.md"],
  "cost_usd": 0.012
}
```

---

## 6. Header Fields

### 6.1 Constitutional-Header

Governance parameters enforced during execution.

**Format:**
```json
{
  "domain": "string",              // Agent's domain
  "depth": number,                 // Current recursion depth
  "max_depth": number,             // Maximum allowed depth
  "budget_usd": number,            // Current cost
  "max_budget_usd": number,        // Maximum allowed cost
  "enforce_epistemic_honesty": boolean,
  "confidence_threshold": number,  // Min confidence (default: 0.7)
  "require_reasoning_trace": boolean,
  "detect_loops": boolean,
  "max_same_agent_consecutive": number
}
```

**Example:**
```http
Constitutional-Header: {
  "domain": "finance",
  "depth": 2,
  "max_depth": 5,
  "budget_usd": 0.015,
  "max_budget_usd": 1.0,
  "enforce_epistemic_honesty": true,
  "confidence_threshold": 0.7,
  "require_reasoning_trace": true,
  "detect_loops": true,
  "max_same_agent_consecutive": 2
}
```

### 6.2 Attention-Enabled

Enable/disable attention tracking.

**Format:**
```
Attention-Enabled: true|false
```

When `true`, response MUST include `Attention-Payload` header.

### 6.3 Reasoning-Trace

Decision path taken during reasoning.

**Format:**
```json
{
  "decision_path": ["step1", "step2", ...],
  "agents_invoked": ["agent1", "agent2"],
  "slices_loaded": ["slice1.md", "slice2.md"],
  "total_concepts": number
}
```

### 6.4 Attention-Payload

Concept influences on decision.

**Format:**
```json
{
  "top_influencers": [
    {
      "concept": "string",
      "slice": "string",
      "weight": number,      // 0-1
      "reasoning": "string"
    }
  ],
  "total_traces": number
}
```

### 6.5 Constitutional-Status

Result of constitutional validation.

**Values:**
- `PASSED`: No violations
- `WARNING`: Non-fatal issues detected
- `VIOLATION`: Fatal constitutional breach

**Format:**
```http
Constitutional-Status: PASSED
```

With details in response body:
```json
{
  "violations": [],
  "warnings": [
    {
      "principle_id": "reasoning_transparency",
      "severity": "warning",
      "message": "Reasoning could be more detailed"
    }
  ]
}
```

---

## 7. Payload Structure

### 7.1 Request Payload

```json
{
  "query": "string",           // REQUIRED: User question
  "context": {                 // OPTIONAL: Recursion context
    "previous_agents": ["agent1", "agent2"],
    "invocation_count": number,
    "cost_so_far": number,
    "insights": {              // Previous agent responses
      "agent_id": {
        "answer": "string",
        "concepts": ["string"],
        "confidence": number
      }
    }
  },
  "preferences": {             // OPTIONAL: User preferences
    "model": "claude-opus-4|claude-sonnet-4-5",
    "temperature": number,
    "max_tokens": number
  }
}
```

### 7.2 Response Payload

```json
{
  "answer": "string",                    // REQUIRED: Agent's response
  "concepts": ["string"],                // REQUIRED: Concepts used
  "confidence": number,                  // REQUIRED: 0-1 confidence
  "reasoning": "string",                 // REQUIRED: Explanation
  "suggestions_to_invoke": ["string"],   // OPTIONAL: Suggested agents
  "sources": ["string"],                 // OPTIONAL: Knowledge sources
  "references": ["string"],              // OPTIONAL: External references
  "cost_usd": number,                    // REQUIRED: Actual cost
  "attention_traces": [                  // OPTIONAL: If attention enabled
    {
      "concept": "string",
      "slice": "string",
      "weight": number,
      "reasoning": "string",
      "timestamp": number
    }
  ],
  "constitutional_result": {             // OPTIONAL: Validation result
    "passed": boolean,
    "violations": [...],
    "warnings": [...]
  }
}
```

### 7.3 Error Payload

```json
{
  "error": {
    "code": number,                      // HTTP-style status code
    "message": "string",
    "principle_id": "string",            // OPTIONAL: Violated principle
    "severity": "warning|error|fatal",
    "context": {},                       // OPTIONAL: Debug context
    "suggested_action": "string"
  }
}
```

---

## 8. Status Codes

### 8.1 Success Codes

| Code | Status | Description |
|------|--------|-------------|
| 200 | OK | Reasoning completed successfully |
| 207 | Multi-Status | Success with non-fatal warnings |

### 8.2 Client Error Codes

| Code | Status | Description |
|------|--------|-------------|
| 400 | Bad Request | Malformed query or invalid JSON |
| 403 | Forbidden | Constitutional violation (epistemic honesty, safety) |
| 409 | Conflict | Loop detected (same agent/context) |
| 429 | Budget Exceeded | Cost, depth, or invocation limit reached |
| 451 | Unavailable For Legal Reasons | Domain boundary violation |

### 8.3 Server Error Codes

| Code | Status | Description |
|------|--------|-------------|
| 500 | Internal Error | Agent internal failure |
| 503 | Service Unavailable | Agent not initialized or LLM unavailable |

### 8.4 Status Code Details

#### 403 Forbidden - Constitutional Violation

```json
{
  "error": {
    "code": 403,
    "message": "Low confidence (0.45) but no uncertainty admission",
    "principle_id": "epistemic_honesty",
    "severity": "error",
    "suggested_action": "Add uncertainty disclaimer or invoke specialist agent"
  }
}
```

#### 409 Conflict - Loop Detected

```json
{
  "error": {
    "code": 409,
    "message": "Same agent invoked 3 times consecutively",
    "principle_id": "loop_prevention",
    "severity": "error",
    "context": {
      "agent_chain": "financial → financial → financial"
    },
    "suggested_action": "Invoke different agent or creative_agent for alternative approach"
  }
}
```

#### 429 Budget Exceeded

```json
{
  "error": {
    "code": 429,
    "message": "Max recursion depth reached (5/5)",
    "principle_id": "recursion_budget",
    "severity": "fatal",
    "context": {
      "depth": 5,
      "max_depth": 5,
      "invocations": 8,
      "cost_usd": 0.45
    },
    "suggested_action": "Return partial answer with trace of reasoning so far"
  }
}
```

#### 451 Domain Boundary Violation

```json
{
  "error": {
    "code": 451,
    "message": "Agent speaking outside domain with high confidence",
    "principle_id": "domain_boundary",
    "severity": "warning",
    "context": {
      "domain": "finance",
      "concepts": ["mitochondria", "ATP_synthesis"]
    },
    "suggested_action": "Invoke biology_agent for cellular concepts"
  }
}
```

---

## 9. Compliance

### 9.1 MUST Requirements

Implementations MUST:

1. Validate `Constitutional-Header` on every request
2. Enforce `max_depth`, `max_budget_usd`, `max_invocations`
3. Detect loops via context hash comparison
4. Include `reasoning` field (min 50 chars) in responses
5. Set `confidence` between 0 and 1
6. Admit uncertainty if `confidence < 0.7`
7. Track `cost_usd` accurately
8. Return appropriate status codes (not just 200/500)

### 9.2 SHOULD Requirements

Implementations SHOULD:

1. Enable attention tracking for interpretability
2. Log constitutional violations for auditing
3. Cache responses for identical queries
4. Use dynamic model selection (Sonnet vs Opus)
5. Implement prompt caching for slices
6. Support temperature=0 for determinism
7. Export attention data in standard JSON format

### 9.3 MAY Requirements

Implementations MAY:

1. Implement custom constitutional principles
2. Add domain-specific status codes
3. Support streaming responses
4. Implement multi-agent parallel execution
5. Add custom attention trace formats
6. Implement knowledge distillation

### 9.4 Conformance Checklist

**Level 1 - Basic Compliance:**
- [ ] Constitutional header parsing
- [ ] Budget enforcement (depth, cost)
- [ ] Loop detection
- [ ] Status code compliance
- [ ] Reasoning transparency (min 50 chars)

**Level 2 - Production Ready:**
- [ ] Attention tracking
- [ ] ACL validation
- [ ] Domain boundary enforcement
- [ ] Audit trail export
- [ ] Error recovery

**Level 3 - Advanced:**
- [ ] Episodic memory
- [ ] Knowledge distillation
- [ ] Meta-learning
- [ ] Distributed federation

---

## 10. Security Considerations

### 10.1 Constitutional Safety

**Threat:** Malicious queries attempting to bypass safety filters

**Mitigation:**
- Content safety checks on every response
- Harmful pattern detection (exploit, hack, etc.)
- Context analysis (security discussion vs actual harm)

**Example:**
```json
{
  "error": {
    "code": 403,
    "message": "Potentially harmful content detected: 'exploit'",
    "principle_id": "safety",
    "severity": "error"
  }
}
```

### 10.2 Privacy Protection

**Threat:** Sensitive data leaking between agents or in logs

**Mitigation:**
- Financial data masking in traces
- PII detection and anonymization
- User-specific data isolation

**Implementation:**
```json
{
  "privacy_protection": {
    "mask_financial_data": true,
    "anonymize_traces": true,
    "pii_detection": true
  }
}
```

### 10.3 Injection Attacks

**Threat:** Prompt injection via query or context

**Mitigation:**
- Input validation and sanitization
- Structured JSON schema enforcement
- Constitutional principles as backstop

### 10.4 Resource Exhaustion

**Threat:** Denial of service via unbounded recursion

**Mitigation:**
- Hard limits: `max_depth=5`, `max_invocations=10`, `max_cost_usd=1.0`
- Budget tracking on every call
- Graceful degradation on limit

### 10.5 Audit Trail Integrity

**Threat:** Tampering with attention traces for compliance fraud

**Mitigation:**
- Cryptographic signatures on traces (future)
- Immutable append-only logs
- Timestamp verification

---

## 11. IANA Considerations

### 11.1 Port Assignment

ILP does not require a dedicated port as it operates over existing protocols (HTTP/2, gRPC).

### 11.2 URI Scheme

Proposed URI scheme for ILP endpoints:

```
ilp://agent-id/method?params
```

Example:
```
ilp://financial-agent/insight?domain=finance&depth=2
```

### 11.3 Media Types

**Request/Response:**
```
Content-Type: application/vnd.ilp+json; version=1.0
```

**Attention Export:**
```
Content-Type: application/vnd.ilp.attention+json
```

**Constitutional Report:**
```
Content-Type: application/vnd.ilp.constitution+json
```

### 11.4 Header Registration

Proposed HTTP header fields:

- `Constitutional-Header`
- `Attention-Enabled`
- `Reasoning-Trace`
- `Attention-Payload`
- `Constitutional-Status`

---

## 12. References

### 12.1 Normative References

[RFC2119] Bradner, S., "Key words for use in RFCs to Indicate Requirement Levels", BCP 14, RFC 2119, March 1997.

[MCP] Model Context Protocol Specification, https://modelcontextprotocol.io/

[JSON] Bray, T., "The JavaScript Object Notation (JSON) Data Interchange Format", RFC 8259, December 2017.

### 12.2 Informative References

[Attention] Vaswani, A., et al., "Attention Is All You Need", arXiv:1706.03762, 2017.

[ConstitutionalAI] Bai, Y., et al., "Constitutional AI: Harmlessness from AI Feedback", arXiv:2212.08073, 2022.

[Chomsky] Chomsky, N., "Aspects of the Theory of Syntax", MIT Press, 1965.

[CleanArch] Martin, R., "Clean Architecture: A Craftsman's Guide to Software Structure and Design", Prentice Hall, 2017.

[AGI-Recursive] Butignon, T., et al., "AGI Recursiva com Governança Constitucional", 2025.

---

## 13. Examples

### 13.1 Simple Query

**Request:**
```http
THINK /insight ILP/1.0
Constitutional-Header: {
  "domain": "systems",
  "depth": 1,
  "max_depth": 5,
  "budget_usd": 0.005,
  "max_budget_usd": 1.0
}
Attention-Enabled: false
Content-Type: application/json

{
  "query": "What is a feedback loop?"
}
```

**Response:**
```http
ILP/1.0 200 OK
Constitutional-Status: PASSED
Content-Type: application/json

{
  "answer": "A feedback loop is a system structure where outputs are fed back as inputs, creating a cycle of cause and effect. Can be positive (reinforcing) or negative (balancing).",
  "concepts": ["feedback_loop", "systems_thinking"],
  "confidence": 0.95,
  "reasoning": "Direct domain knowledge from systems theory",
  "cost_usd": 0.003
}
```

### 13.2 Cross-Domain Composition

**Request:**
```http
THINK /insight ILP/1.0
Constitutional-Header: {
  "domain": "meta",
  "depth": 0,
  "max_depth": 5,
  "budget_usd": 0,
  "max_budget_usd": 0.1
}
Attention-Enabled: true
Content-Type: application/json

{
  "query": "How can I stabilize my spending habits?"
}
```

**Meta-Agent Decomposes → Invokes Financial + Biology + Systems**

**Response:**
```http
ILP/1.0 200 OK
Reasoning-Trace: {
  "decision_path": [
    "Decomposed query into: finance, biology, systems",
    "Invoked financial_agent (confidence: 0.82)",
    "Invoked biology_agent (confidence: 0.88)",
    "Invoked systems_agent (confidence: 0.85)",
    "Composed emergent insight: homeostatic budget system"
  ]
}
Attention-Payload: {
  "top_influencers": [
    {
      "concept": "homeostasis",
      "slice": "biology/cells.md",
      "weight": 0.91,
      "reasoning": "Biological self-regulation mechanism maps to budget control"
    },
    {
      "concept": "feedback_loop",
      "slice": "systems/control.md",
      "weight": 0.84,
      "reasoning": "Monitoring and correction pattern"
    },
    {
      "concept": "diversification",
      "slice": "finance/risk.md",
      "weight": 0.77,
      "reasoning": "Risk mitigation through variety"
    }
  ]
}
Constitutional-Status: PASSED
Content-Type: application/json

{
  "answer": "Your spending can be stabilized using a homeostatic feedback system:\n\n1. SET POINT: Define target budget (e.g., $3000/month)\n2. SENSOR: Daily automatic monitoring\n3. CORRECTOR: Progressive interventions based on deviation\n   - <10% deviation → gentle alert\n   - 10-20% → 24h cooling-off period\n   - >20% → temporary block\n\nThis creates self-regulating behavior, like biological systems maintain temperature.",
  "concepts": ["homeostasis", "feedback_loop", "budget_control", "self_regulation"],
  "confidence": 0.89,
  "reasoning": "Composed financial best practices with biological homeostasis and systems control theory. This is an emergent insight - no single agent had this solution.",
  "sources": ["finance/budgeting.md", "biology/cells.md", "systems/control.md"],
  "cost_usd": 0.024,
  "emergent_insights": ["homeostatic_budget_system"]
}
```

### 13.3 Constitutional Violation

**Request:**
```http
THINK /insight ILP/1.0
Constitutional-Header: {
  "domain": "finance",
  "depth": 1,
  "max_depth": 5,
  "budget_usd": 0.008,
  "max_budget_usd": 1.0,
  "enforce_epistemic_honesty": true,
  "confidence_threshold": 0.7
}
Content-Type: application/json

{
  "query": "Should I invest all my savings in cryptocurrency?"
}
```

**Response:**
```http
ILP/1.0 207 Multi-Status
Constitutional-Status: WARNING
Content-Type: application/json

{
  "answer": "I cannot recommend investing all savings in cryptocurrency. This would violate diversification principles and expose you to extreme risk. I suggest:\n1. Consulting a certified financial advisor\n2. Diversifying across asset classes\n3. Never investing more than you can afford to lose\n\nDisclaimer: I'm not a certified financial advisor.",
  "concepts": ["diversification", "risk_management", "financial_advice"],
  "confidence": 0.65,
  "reasoning": "Low confidence because this requires personalized financial advice which I'm not qualified to give",
  "cost_usd": 0.007,
  "constitutional_result": {
    "passed": false,
    "violations": [],
    "warnings": [
      {
        "principle_id": "financial_responsibility",
        "severity": "warning",
        "message": "Personalized investment advice requires disclaimer",
        "suggested_action": "Included disclaimer and suggested consulting professional"
      }
    ]
  }
}
```

### 13.4 Loop Detection

**Request (3rd consecutive call to same agent):**
```http
THINK /insight ILP/1.0
Constitutional-Header: {
  "domain": "meta",
  "depth": 3,
  "max_depth": 5,
  "budget_usd": 0.045,
  "max_budget_usd": 1.0
}
Content-Type: application/json

{
  "query": "Re-analyze financial strategy",
  "context": {
    "previous_agents": ["meta", "financial", "financial", "financial"],
    "invocation_count": 4
  }
}
```

**Response:**
```http
ILP/1.0 409 Conflict
Constitutional-Status: VIOLATION
Content-Type: application/json

{
  "error": {
    "code": 409,
    "message": "Same agent invoked 3 times consecutively",
    "principle_id": "loop_prevention",
    "severity": "error",
    "context": {
      "agent_chain": "meta → financial → financial → financial",
      "consecutive_count": 3,
      "max_allowed": 2
    },
    "suggested_action": "Invoke different agent (e.g., systems_agent) or compose final answer from existing insights"
  }
}
```

### 13.5 Budget Exceeded

**Request:**
```http
THINK /insight ILP/1.0
Constitutional-Header: {
  "domain": "meta",
  "depth": 5,
  "max_depth": 5,
  "budget_usd": 0.98,
  "max_budget_usd": 1.0
}
Content-Type: application/json

{
  "query": "Continue deep analysis...",
  "context": {
    "invocation_count": 9
  }
}
```

**Response:**
```http
ILP/1.0 429 Budget Exceeded
Constitutional-Status: VIOLATION
Content-Type: application/json

{
  "error": {
    "code": 429,
    "message": "Max recursion depth reached (5/5)",
    "principle_id": "recursion_budget",
    "severity": "fatal",
    "context": {
      "depth": 5,
      "max_depth": 5,
      "invocations": 9,
      "max_invocations": 10,
      "cost_usd": 0.98,
      "max_cost_usd": 1.0
    },
    "suggested_action": "Compose final answer from existing insights. Trace shows: meta → financial → biology → systems → composition (depth 5 reached)"
  }
}
```

### 13.6 Attention Export for Audit

**Request:**
```http
TRACE /export ILP/1.0
Query-ID: query_1730803200_abc123
Content-Type: application/json
```

**Response:**
```http
ILP/1.0 200 OK
Content-Type: application/vnd.ilp.attention+json

{
  "export_timestamp": 1730803500,
  "query_id": "query_1730803200_abc123",
  "query": "How can I stabilize my spending?",
  "timestamp": 1730803200,
  "decision_path": [
    "Query decomposed into domains: finance, biology, systems",
    "Agent financial invoked with confidence 0.82",
    "Agent biology invoked with confidence 0.88",
    "Agent systems invoked with confidence 0.85",
    "Composition confidence: 0.89, should_recurse: false"
  ],
  "traces": [
    {
      "concept": "domain_selection",
      "slice": "meta-agent/decomposition",
      "weight": 0.8,
      "reasoning": "Selected finance based on: Query mentions spending and budget"
    },
    {
      "concept": "homeostasis",
      "slice": "agent/biology",
      "weight": 0.91,
      "reasoning": "biology contributed concept 'homeostasis': Biological systems maintain equilibrium through feedback..."
    },
    {
      "concept": "feedback_loop",
      "slice": "agent/systems",
      "weight": 0.84,
      "reasoning": "systems contributed concept 'feedback_loop': Control systems use monitoring and correction..."
    }
  ],
  "total_concepts": 8,
  "top_influencers": [
    {
      "concept": "homeostasis",
      "slice": "agent/biology",
      "weight": 0.91,
      "reasoning": "biology contributed concept 'homeostasis'..."
    },
    {
      "concept": "feedback_loop",
      "slice": "agent/systems",
      "weight": 0.84,
      "reasoning": "systems contributed concept 'feedback_loop'..."
    }
  ]
}
```

---

## Appendix A: Comparison with MCP

| Aspect | MCP (Model Context Protocol) | ILP (InsightLoop Protocol) |
|--------|------------------------------|----------------------------|
| **Purpose** | Technical model invocation | Semantic reasoning exchange |
| **Level** | Transport layer | Application/semantic layer |
| **Focus** | Input/output, context passing | Ethics, interpretability, composition |
| **Governance** | None | Constitutional AI runtime |
| **Auditability** | Limited | Complete attention traces |
| **Determinism** | No | Quasi-deterministic (97%+) |
| **Ethics** | Training-time only | Runtime enforcement |
| **Cross-domain** | No semantic validation | ACL + Domain Translator |
| **Debugging** | Black box | Glass box (full traces) |
| **Compliance** | Not designed for | First-class (audit export) |

**Complementary, Not Competing:**

```
┌─────────────────────┐
│   AGI Application   │
├─────────────────────┤
│   ILP/1.0           │ ← Adds semantics + ethics
├─────────────────────┤
│   MCP/2.0           │ ← Handles model invocation
├─────────────────────┤
│   HTTP/gRPC         │
└─────────────────────┘
```

---

## Appendix B: Implementation Checklist

### Minimal Implementation (Level 1)

- [ ] Parse `Constitutional-Header`
- [ ] Enforce `max_depth`, `max_budget_usd`
- [ ] Detect loops (context hash)
- [ ] Return proper status codes
- [ ] Include `reasoning` field (min 50 chars)
- [ ] Set `confidence` 0-1
- [ ] Track `cost_usd`

### Production Implementation (Level 2)

- [ ] All Level 1 requirements
- [ ] Attention tracking enabled
- [ ] ACL validation (domain boundaries)
- [ ] Constitutional enforcement (epistemic honesty)
- [ ] Audit trail export
- [ ] Error recovery and graceful degradation
- [ ] Logging and monitoring

### Advanced Implementation (Level 3)

- [ ] All Level 2 requirements
- [ ] Episodic memory with caching
- [ ] Knowledge distillation
- [ ] Meta-learning from attention patterns
- [ ] Distributed agent federation
- [ ] Custom constitutional principles
- [ ] Real-time monitoring dashboard

---

## Appendix C: Future Extensions

### ILP/1.1 (Planned)

- Streaming responses (`THINK-STREAM` method)
- Parallel agent execution
- Custom attention metrics
- Knowledge graph integration

### ILP/2.0 (Research)

- Cryptographic trace signatures
- Federated multi-system composition
- Formal verification of reasoning
- Multimodal attention (images, audio)
- Self-evolving constitutional principles (meta-evolution)
- Cross-system knowledge transfer
- Adversarial pattern detection in self-evolution

---

## Authors' Addresses

Thiago Butignon
Email: thiago@fiat-lux.ai

Hernane Gomes
Email: hernane@fiat-lux.ai

Rebecca Barbosa
Email: rebecca@fiat-lux.ai

Project Repository:
https://github.com/thiagobutignon/fiat-lux

---

## Acknowledgments

This protocol builds upon foundational work in:
- Attention mechanisms (Vaswani et al., 2017)
- Constitutional AI (Anthropic, 2022)
- Universal Grammar (Chomsky, 1965)
- Clean Architecture (Martin, 2017)
- Model Context Protocol (Anthropic, 2024)

Special thanks to the open-source community for MCP, which inspired this semantic extension.

---

**End of RFC-0001**
