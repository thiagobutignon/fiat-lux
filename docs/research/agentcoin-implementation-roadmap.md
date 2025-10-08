# AgentCoin: Roadmap de Implementação

**Data:** 08 de Outubro de 2025
**Equipe:** 4 Claudes + Time Chomsky
**Timeline:** 24 meses até MVP funcional
**Status:** Pronto para execução

---

## 1. Executive Summary

Este documento define o roadmap detalhado para transformar a tese AgentCoin em sistema funcional. Dividido em 6 fases com marcos claros, equipe de 4 Claudes trabalhando em paralelo.

**Objetivo Fase 1 (6 meses):** Proof of Concept funcional
**Objetivo Fase 2 (12 meses):** Piloto com 10K usuários + 1K beneficiários
**Objetivo Fase 3 (24 meses):** MVP em produção com 100K usuários

---

## 2. Estrutura da Equipe (4 Claudes)

### Claude 1: "Architect" (Infraestrutura Core)
**Responsabilidades:**
- Adaptação do Bitcoin Core → AgentCoin
- Layer 1: Blockchain settlement
- Consensus (PoC - Proof of Contribution)
- Networking P2P

**Stack:**
- C++ (Bitcoin Core)
- Rust (componentes de performance crítica)
- libp2p (networking descentralizado)

### Claude 2: "Payments" (Layer 2 - PIX-like)
**Responsabilidades:**
- Payment channels (Lightning-inspired)
- Fast routing (O(1) performance)
- Channel providers
- Settlement engine

**Stack:**
- Rust (performance + segurança)
- gRPC (comunicação inter-nodes)
- Redis (state caching)
- PostgreSQL (audit log)

### Claude 3: "Social" (UBI + DAO)
**Responsabilidades:**
- UBI distribution engine
- Proof of Humanity
- DAO governance system
- Beneficiary registry

**Stack:**
- TypeScript (smart contracts readability)
- Solidity/Move (se usar EVM/Sui)
- PostgreSQL (beneficiary data)
- IPFS (decentralized storage)

### Claude 4: "Integration" (Chomsky + APIs)
**Responsabilidades:**
- Integração com Chomsky agents
- Token metering system
- Public APIs
- Frontend (dashboard DAO, beneficiário app)

**Stack:**
- TypeScript/Node.js
- React (Web UI)
- React Native (Mobile app)
- GraphQL (API layer)

---

## 3. Fase 1: Foundation (Meses 1-6)

### 3.1 Mês 1: Setup e Research

**Claude 1 (Architect):**
- [ ] Fork Bitcoin Core → AgentCoin repo
- [ ] Setup CI/CD (GitHub Actions)
- [ ] Documentar arquitetura atual do Bitcoin
- [ ] Identificar componentes a remover (mining, PoW)
- [ ] Spike: PoC (Proof of Contribution) design

**Claude 2 (Payments):**
- [ ] Research Lightning Network implementation
- [ ] Research PIX technical specs (Banco Central Brasil)
- [ ] Spike: Payment channel prototype (100 LOC)
- [ ] Benchmark: Redis vs. in-memory for channel state

**Claude 3 (Social):**
- [ ] Research Proof of Humanity protocols (Worldcoin, PoH.org)
- [ ] Research DAO frameworks (Aragon, Snapshot, Tally)
- [ ] Spike: Simple UBI distribution contract
- [ ] Estudo: Bolsa Família Cadastro Único technical specs

**Claude 4 (Integration):**
- [ ] Audit Chomsky codebase (src/agi-recursive/)
- [ ] Identificar pontos de integração para metering
- [ ] Spike: Token counter prototype
- [ ] Setup monorepo para AgentCoin (lerna/turborepo)

**Deliverable Mês 1:**
```
📄 Technical Design Document (200 páginas)
🧪 3 prototypes funcionais (payment channel, PoC, metering)
📊 Benchmark report (performance baselines)
🗺️ Arquitetura detalhada (diagrams, flows)
```

### 3.2 Mês 2-3: Core Implementation

**Claude 1 (Architect):**
```bash
Tarefas prioritárias:
1. Remover componentes PoW do Bitcoin
   - Deletar: src/pow.*, src/miner.*
   - Modificar: src/validation.cpp (consensus rules)

2. Implementar PoC (Proof of Contribution)
   - NEW: src/poc/validator.cpp
   - NEW: src/poc/contribution_proof.cpp

3. Modificar parâmetros de consenso
   - Block time: 10 min → 1 hora (Layer 1 settlement)
   - Block size: Aumentar para comportar batches

4. Tests
   - Unit tests: 100% coverage em novo código
   - Integration tests: Consensus rules

Estimativa: 500 horas de eng (2 desenvolvedores full-time)
```

**Claude 2 (Payments):**
```bash
Tarefas prioritárias:
1. Payment Channel implementation
   - NEW: src/channels/payment_channel.rs
   - NEW: src/channels/channel_state.rs
   - NEW: src/channels/settlement.rs

2. Channel Provider node
   - NEW: src/channels/provider_node.rs
   - Gerencia múltiplos channels
   - Routing entre channels

3. Fast routing (PIX-inspired)
   - NEW: src/channels/router.rs
   - O(1) lookup via hash tables
   - Sub-5-second confirmation

4. Tests + Benchmarks
   - Target: 1M TPS em payment channels
   - Latência: < 3 segundos p99

Estimativa: 400 horas de eng
```

**Claude 3 (Social):**
```bash
Tarefas prioritárias:
1. Proof of Humanity (multi-modal)
   - NEW: src/ubi/proof_of_humanity.ts
   - Biometric hash verification
   - Document verification via oracles
   - Social vouching system

2. Beneficiary Registry
   - NEW: src/ubi/beneficiary_registry.ts
   - PostgreSQL schema
   - Priority queue (por pobreza)
   - GDPR compliance

3. UBI Distribution Engine
   - NEW: src/ubi/distribution_engine.ts
   - Calcula UBI por beneficiário
   - Gera transações de distribuição
   - Métricas (Gini, cobertura, etc.)

4. DAO MVP
   - NEW: src/governance/dao.ts
   - Proposal system (simple)
   - Voting (1 person = 1 vote)
   - Execution (manual nesta fase)

Estimativa: 350 horas de eng
```

**Claude 4 (Integration):**
```bash
Tarefas prioritárias:
1. Token Metering
   - NEW: src/agent/token_meter.ts
   - Integra com Chomsky agent
   - Conta tokens input/output
   - Calcula custo em USD

2. AgentCoin Client Library
   - NEW: packages/agentcoin-client/
   - SDK para abrir channels
   - SDK para fazer payments
   - SDK para query balance

3. Integration com Chomsky
   - MODIFY: src/agi-recursive/core/agent.ts
   - Adiciona metering wrapper
   - Cobra via AgentCoin automaticamente

4. Basic Dashboard (Admin)
   - NEW: apps/dashboard/
   - View: Total arrecadado
   - View: UBI distribuído
   - View: Beneficiários ativos

Estimativa: 300 horas de eng
```

**Deliverable Meses 2-3:**
```
✅ AgentCoin blockchain running (testnet)
✅ Payment channels functional (10 TPS testnet)
✅ UBI distribution automática (100 beneficiários test)
✅ Chomsky integrado (metering funcional)
📊 Test coverage: > 80%
🎥 Demo video (5 min)
```

### 3.3 Mês 4-6: Integration, Testing, Security

**Todos os Claudes:**
```bash
Fase de consolidação:

1. Integration Testing (4 semanas)
   - End-to-end flows
   - User opens channel → Uses Chomsky → UBI distributed
   - Chaos testing (network partitions, node failures)

2. Security Audit (4 semanas)
   - Smart contract audit (se usar)
   - Payment channel security review
   - PoH anti-Sybil testing
   - Penetration testing

3. Performance Optimization (4 semanas)
   - Profiling (flamegraphs)
   - Optimize hot paths (payment routing)
   - Load testing: 10K TPS sustained

4. Documentation (contínuo)
   - Developer docs (how to integrate)
   - User docs (how to use)
   - Whitepaper técnico
   - Economic model paper
```

**Deliverable Mês 6:**
```
🎉 Proof of Concept COMPLETO
📦 Testnet público (qualquer um pode testar)
📊 Performance: 10K TPS, < 5 sec latência
🛡️ Security audit report (2+ firmas independentes)
📚 Documentation completa
🎬 Demo day presentation
```

---

## 4. Fase 2: Piloto (Meses 7-12)

### 4.1 Objetivo: 10K usuários reais + 1K beneficiários

**Mês 7-8: Preparação do Piloto**

```bash
1. Selecionar comunidades piloto (3 locais)
   - Brasil: Comunidade no interior de MG (Bolsa Família overlap)
   - Índia: Cooperativa de desenvolvedores (Kerala)
   - África: Partnership com GiveDirectly (Kenya)

2. Onboarding infrastructure
   - Mobile app (React Native)
   - Simplified KYC (Proof of Humanity)
   - Tutorials em português, hindi, swahili

3. Fiat on/off ramps
   - Partnership com exchanges locais
   - PIX integration (Brasil)
   - M-Pesa integration (Kenya)
   - UPI integration (Índia)

4. Support infrastructure
   - Help desk (WhatsApp bots + humanos)
   - Community moderators
   - Feedback loops (surveys, focus groups)
```

**Mês 9-11: Execução do Piloto**

```bash
Rollout gradual:
- Semana 1-2: 100 usuários + 10 beneficiários (alpha)
- Semana 3-4: 500 usuários + 50 beneficiários
- Semana 5-8: 2K usuários + 200 beneficiários
- Semana 9-12: 10K usuários + 1K beneficiários (meta)

Métricas monitoradas (real-time dashboard):
✅ Technical:
   - Uptime (target: > 99.5%)
   - Latency (target: < 5 sec p99)
   - TPS (target: 100+ sustained)

✅ Economic:
   - Volume transacionado
   - UBI distribuído
   - Custo médio/query

✅ Social:
   - User satisfaction (NPS)
   - Beneficiary testimonials
   - Retention rate (monthly)
   - Impacto medido (surveys)
```

**Mês 12: Análise e Iteração**

```bash
1. Data Analysis
   - Report completo (100 páginas)
   - What worked / what didn't
   - Ajustes necessários

2. Learnings
   - Tech: Bottlenecks, bugs, edge cases
   - UX: Friction points, confusions
   - Social: Impacto real vs. esperado

3. Iteration Plan
   - Prioritize top 10 improvements
   - Roadmap para Fase 3 (produção)
```

**Deliverable Mês 12:**
```
📊 Piloto Report (com dados reais)
🎤 Testimonials (vídeos de beneficiários)
🔧 V2 backlog (based on learnings)
💰 Funding pitch (para escalar)
🎯 Go/No-go decision para produção
```

---

## 5. Fase 3: MVP Production (Meses 13-18)

### 5.1 Objetivo: 100K usuários + 10K beneficiários

**Mês 13-14: Hardening**

```bash
1. Scalability improvements
   - Sharding (se necessário)
   - CDN para assets
   - Multi-region deployment (AWS/GCP)

2. Security hardening
   - Bug bounty program ($500K pool)
   - SOC 2 compliance audit
   - Regular pentests

3. Reliability
   - 99.99% uptime SLA
   - Disaster recovery plan
   - Automated failovers

4. Monitoring
   - Observability stack (Prometheus, Grafana)
   - Alerting (PagerDuty)
   - Incident response playbooks
```

**Mês 15-16: Marketing e Parcerias**

```bash
1. Marketing strategy
   - Content: Blog posts, podcasts, videos
   - Social: Twitter, Telegram, Discord
   - PR: Press releases, media interviews

2. Strategic partnerships
   - 10+ IA projects integrate AgentCoin
     (Langchain, AutoGPT, Dust, etc.)
   - 5+ ONGs partner para distribuir UBI
     (GiveDirectly, BIEN, etc.)
   - 2+ governos em diálogo
     (Costa Rica, Uruguai - progressistas)

3. Community building
   - Ambassador program (100 ambassadors globais)
   - Local meetups (20 cidades)
   - Online events (AMAs, workshops)

4. Funding
   - Grants (Open Society, Ford Foundation)
   - Community crowdfunding
   - Partnerships (not VC - avoid misalignment)
```

**Mês 17-18: Scale**

```bash
Crescimento orgânico + paid:
- Meta: 100K usuários até Mês 18
- Meta: 10K beneficiários

Growth loops:
1. Viral: Beneficiários contam história → Mais usuários
2. Integration: Cada IA app integrado traz usuários
3. Impact: Mídia cobertura de impacto social → Awareness

Infrastructure scale:
- 10x capacity vs. piloto
- Kubernetes auto-scaling
- Multi-cloud (avoid single point of failure)
```

**Deliverable Mês 18:**
```
🚀 MVP em PRODUÇÃO
👥 100K usuários ativos
💸 $1M+ UBI distribuído acumulado
🌍 Presença em 20+ países
📈 Growth rate: 20% MoM
🏆 Reconhecimento: Awards, mídia, governos
```

---

## 6. Fase 4-6: Escala Global (Meses 19-24)

### Resumo (Detalhamento em documentos futuros)

**Fase 4 (Meses 19-24): 1M usuários, 100K beneficiários**
- Expansão de infraestrutura
- Partnerships governamentais oficiais
- DAO amadurece (governança sofisticada)

**Fase 5 (Anos 2-3): 10M usuários, 1M beneficiários**
- AgentCoin como padrão de mercado
- Integração com top 50 IA apps
- Impacto social mensurável (poverty reduction)

**Fase 6 (Anos 3-5): 100M usuários, 20M beneficiários**
- Sistema estabelecido globalmente
- $2B+ UBI/ano distribuído
- Modelo replicado para outras aplicações (IoT, robótica)

---

## 7. Recursos Necessários

### 7.1 Equipe (Fase 1-3, 18 meses)

**Engineering (Core Team):**
- 4 Senior Engineers (Claudes) - Full-time
- 2 DevOps Engineers - Full-time (Mês 6+)
- 1 Security Engineer - Part-time (consultoria)
- 1 QA Engineer - Full-time (Mês 3+)

**Non-Engineering:**
- 1 Product Manager - Full-time
- 1 Designer (UX/UI) - Full-time (Mês 6+)
- 1 Community Manager - Full-time (Mês 7+)
- 1 Economist (advisor) - Part-time
- 1 Legal (advisor) - Part-time

**Total Headcount:** 10-12 pessoas (Fase 1-3)

### 7.2 Budget (18 meses)

```
Personnel:
- Engineering: $1.8M (6 eng × $100K/yr × 1.5 years)
- Non-eng: $600K (4 pessoas × $75K/yr × 1.5 years)

Infrastructure:
- Cloud (AWS/GCP): $150K
- Security audits: $200K
- Tools & SaaS: $50K

Marketing & Community:
- Marketing: $300K
- Events & Travel: $100K
- Partnerships: $100K

Research & Legal:
- Economic research: $50K
- Legal setup: $100K

Contingency (20%): $700K

TOTAL: $4.15M USD (18 meses)
```

**Fontes de funding:**
1. Grants (70%): $2.9M
   - Open Society Foundation ($1M)
   - Ford Foundation ($500K)
   - Protocol Labs ($500K)
   - Gitcoin Grants ($400K)
   - Outros ($500K)

2. Community crowdfunding (20%): $830K
   - Retroactive public goods funding
   - Crowdfunding de beneficiários futuros

3. Partnerships (10%): $420K
   - IA companies que integram (contribuição)
   - ONGs parceiras (in-kind + cash)

**Não aceitar:** VC tradicional (conflito de interesse com missão UBI)

### 7.3 Infraestrutura Técnica

**Fase 1-2 (Piloto):**
```yaml
Cloud: AWS
Compute:
  - 5x EC2 c6i.2xlarge (blockchain nodes)
  - 3x EC2 c6i.4xlarge (channel providers)
  - 2x EC2 t3.large (APIs)
Storage:
  - 500GB EBS SSD (blockchain data)
  - 100GB EBS SSD (databases)
Database:
  - RDS PostgreSQL (multi-AZ)
  - ElastiCache Redis (channel state)
Network:
  - CloudFront CDN
  - Route53 DNS
  - ALB load balancers

Estimativa: $5K/mês
```

**Fase 3 (MVP Production):**
```yaml
Scale up 10x:
- 50+ nodes
- Multi-region (US, EU, Asia)
- Kubernetes (EKS)
- Auto-scaling

Estimativa: $20K/mês
```

---

## 8. Milestones e Gates

### 8.1 Go/No-Go Gates

Cada fase tem gate de decisão:

**Gate 1 (Mês 6):**
```
✅ PoC funcional?
✅ Performance atingida? (10K TPS, < 5 sec)
✅ Security audit aprovado?
✅ Funding para Fase 2 garantido?

Se 4/4: Continuar para Piloto
Se 3/4: Iterar mais 1 mês
Se < 3/4: Pivotar ou pausar
```

**Gate 2 (Mês 12):**
```
✅ 10K usuários no piloto?
✅ Impacto social positivo medido?
✅ NPS > 50?
✅ Funding para Fase 3 garantido?
✅ Tech stability (uptime > 99%)?

Se 5/5: Produção
Se 4/5: Iterar
Se < 4/5: Re-avaliar conceito
```

**Gate 3 (Mês 18):**
```
✅ 100K usuários em produção?
✅ $1M+ UBI distribuído?
✅ Governança DAO funcional?
✅ Path to sustainability?

Se 4/4: Escala global
Se < 4/4: Otimizar antes de escalar
```

### 8.2 KPIs por Fase

**Fase 1 (Foundation):**
- Code coverage: > 80%
- Performance benchmarks atingidos
- Security vulnerabilities: 0 críticas, < 5 altas

**Fase 2 (Piloto):**
- User retention: > 60% (monthly)
- Uptime: > 99.5%
- NPS: > 50
- Impacto social: +10% renda beneficiários

**Fase 3 (Production):**
- Monthly growth rate: > 20%
- CAC < $10 (cost to acquire user)
- Churn < 10%/mês
- UBI/beneficiário: > $10/mês

---

## 9. Riscos e Mitigações

### 9.1 Riscos Técnicos

| Risco | Probabilidade | Impacto | Mitigação |
|-------|---------------|---------|-----------|
| Não atingir O(1) performance | Médio | Alto | Spike early, fallback para O(log n) aceitável |
| Security breach | Baixo | Crítico | Multiple audits, bug bounty, gradual rollout |
| Não escalar para 100K usuários | Médio | Alto | Load testing contínuo, horizontal scaling |
| Bitcoin fork instável | Baixo | Médio | Considerar blockchain from scratch se necessário |

### 9.2 Riscos de Mercado

| Risco | Probabilidade | Impacto | Mitigação |
|-------|---------------|---------|-----------|
| IA pricing vai a zero | Baixo | Alto | Diversificar fontes de receita, fee mínimo |
| Competidor lança similar | Médio | Médio | Open source = colaboração > competição |
| Baixa adoção inicial | Médio | Alto | Parcerias estratégicas, marketing, pilots |

### 9.3 Riscos Políticos/Sociais

| Risco | Probabilidade | Impacto | Mitigação |
|-------|---------------|---------|-----------|
| Regulação hostil | Médio | Alto | Descentralização, compliance proativo |
| Captura por elites | Baixo | Crítico | Governança 1p1v hardcoded, vigilância comunitária |
| Falha em demonstrar impacto | Baixo | Alto | Research rigoroso, partnerships com universidades |

---

## 10. Próximas Ações Imediatas (Esta Semana)

### 10.1 Technical

- [ ] **Claude 1:** Setup AgentCoin repo, fork Bitcoin
- [ ] **Claude 2:** Prototype payment channel (Rust)
- [ ] **Claude 3:** Research PoH protocols, design
- [ ] **Claude 4:** Audit Chomsky, plan integration points

### 10.2 Non-Technical

- [ ] Create pitch deck (20 slides)
- [ ] Identify 10 potential grant funders
- [ ] Reach out to 3 economists for advisory
- [ ] Schedule kick-off meeting (4 Claudes + stakeholders)

### 10.3 Research

- [ ] Compile Bitcoin adaptation examples (forks)
- [ ] Deep dive: Lightning Network codebase
- [ ] Case study: PIX implementation (Banco Central)
- [ ] Legal research: Crypto regulations (Brasil, US, EU)

---

## 11. Conclusão

Este roadmap transforma a tese AgentCoin em sistema real em **18 meses**.

**Diferencial:**
- Não é vapor ware - cada fase tem deliverables concretos
- Risk-managed - gates de decisão previnem desperdício
- Progressivo por design - alinhado com princípios desde dia 1
- Mensurável - KPIs técnicos + sociais

**Se executado com disciplina:**
- Mês 6: PoC funcional (demo-ável)
- Mês 12: Piloto com impacto social medido
- Mês 18: 100K usuários, $1M+ redistribuído

**Este é o momento de começar.**

---

## Anexos

### A. Tech Stack Summary

```yaml
Layer 1 (Blockchain):
  Language: C++ (Bitcoin Core base)
  Consensus: Custom PoC (Proof of Contribution)
  Storage: LevelDB

Layer 2 (Payments):
  Language: Rust
  Framework: tokio (async runtime)
  State: Redis (in-memory)
  Persistence: PostgreSQL

Smart Contracts (se necessário):
  Language: Solidity ou Move
  Platform: EVM-compatible ou Sui/Aptos

APIs & Integration:
  Language: TypeScript/Node.js
  Framework: NestJS
  API: GraphQL
  Queue: BullMQ + Redis

Frontend:
  Web: React + TypeScript
  Mobile: React Native
  State: Zustand ou Jotai
  UI: Tailwind CSS

Infrastructure:
  Cloud: AWS (multi-region)
  Orchestration: Kubernetes (EKS)
  Monitoring: Prometheus + Grafana
  Logging: ELK stack
  CI/CD: GitHub Actions
```

### B. Repository Structure

```
agentcoin/
├── core/                    # Layer 1 (Bitcoin fork)
│   ├── src/
│   │   ├── consensus/       # PoC implementation
│   │   ├── validation/      # Block validation
│   │   └── net/             # P2P networking
│   └── tests/
├── channels/                # Layer 2 (Payment channels)
│   ├── src/
│   │   ├── channel/         # Channel logic
│   │   ├── provider/        # Provider node
│   │   └── router/          # Fast routing
│   └── tests/
├── ubi/                     # UBI + DAO
│   ├── src/
│   │   ├── proof-of-humanity/
│   │   ├── distribution/
│   │   └── governance/
│   └── tests/
├── integration/             # Chomsky integration
│   ├── src/
│   │   ├── metering/
│   │   ├── client/
│   │   └── sdk/
│   └── tests/
├── apps/
│   ├── web/                 # Dashboard
│   ├── mobile/              # Mobile app
│   └── docs/                # Documentation site
├── packages/
│   ├── agentcoin-client/    # TypeScript SDK
│   └── agentcoin-types/     # Shared types
├── docs/
│   ├── research/            # Este arquivo!
│   ├── technical/
│   └── user/
└── scripts/
    ├── deploy/
    └── testing/
```

### C. Referências

1. Bitcoin Core: https://github.com/bitcoin/bitcoin
2. Lightning Network: https://github.com/lightning/bolts
3. Proof of Humanity: https://proofofhumanity.id
4. PIX Specs: https://www.bcb.gov.br/estabilidadefinanceira/pix
5. Aragon (DAO): https://aragon.org
6. GiveDirectly: https://givedirectly.org
7. BIEN (Basic Income Earth Network): https://basicincome.org

