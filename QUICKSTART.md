# Quick Start Guide 🚀

## Setup (30 segundos)

```bash
# 1. Clone o projeto (se ainda não tem)
git clone https://github.com/thiagobutignon/fiat-lux.git
cd fiat-lux

# 2. Instalar dependências
npm install

# 3. Configurar API key
cp .env.example .env
# Edite .env e adicione: ANTHROPIC_API_KEY=sk-ant-api03-sua-key-aqui
```

## Rodar Sistema Completo

### Opção 1: Script Automático (Recomendado) ⭐

```bash
npm run agi:all
```

Roda todos os 4 demos em sequência com pausa entre cada um:
1. 🤖 Anthropic Adapter (~$0.007)
2. 🧭 Slice Navigation ($0)
3. 🛡️ Anti-Corruption Layer ($0)
4. 🧠 Budget Homeostasis (~$0.02-0.05)

**Custo total**: ~$0.03-0.06 (~R$0.15-0.30)

### Opção 2: Demos Individuais

```bash
# Adapter - Cost tracking e model selection
npm run agi:adapter

# Navigation - Dynamic knowledge discovery
npm run agi:navigation

# ACL - Safety mechanisms
npm run agi:acl

# Homeostasis - FULL AGI (emergent intelligence)
npm run agi:homeostasis
```

### Opção 3: Manual (Tradicional)

```bash
npx tsx src/agi-recursive/examples/anthropic-adapter-demo.ts
npx tsx src/agi-recursive/examples/slice-navigation-demo.ts
npx tsx src/agi-recursive/examples/acl-protection-demo.ts
npx tsx src/agi-recursive/examples/budget-homeostasis.ts
```

## O Que Esperar

### Demo 1: Anthropic Adapter
```
✅ Model recommendations (Opus vs Sonnet)
✅ Cost estimation ($0.075 Opus, $0.015 Sonnet)
✅ Real API call (22 input, 290 output tokens)
✅ 80% savings using Sonnet!
✅ Cumulative cost tracking
```

### Demo 2: Slice Navigation
```
✅ 3 slices indexed, 17 concepts
✅ Search "homeostasis" → 2 results
✅ Cross-domain connections found
✅ 2.6x cache speedup
```

### Demo 3: Anti-Corruption Layer
```
✅ Domain boundary violations blocked
✅ Loop detection working
✅ Content safety (SQL injection blocked)
✅ Budget enforcement
```

### Demo 4: Budget Homeostasis (⭐ PRINCIPAL)
```
🎯 Query: "Gasto com delivery tá fora de controle"

Individual agents:
- Financial: "Set limits, track spending"
- Biology: "Homeostasis, regulation"
- Systems: "Feedback loops"

🧠 EMERGENT SYNTHESIS:
"Your budget needs a HOMEOSTATIC SYSTEM:
 - SET POINT: R$1500/month
 - SENSOR: Real-time tracking
 - CORRECTOR: Auto freeze at 90%
 - HANDLER: Pre-order groceries Thursday"

💡 "Budget as Biological System" = EMERGENTE!
Nenhum agente sozinho sugeriria isso!
```

## Troubleshooting

### "ANTHROPIC_API_KEY not found"
```bash
# Verifique se .env existe
ls -la .env

# Deve conter:
cat .env
# ANTHROPIC_API_KEY=sk-ant-...
```

### "Module not found"
```bash
npm install
```

### "Permission denied" (no script)
```bash
chmod +x scripts/run-agi-demos.sh
```

## Próximos Passos

📚 **Documentação Completa**:
- `docs/AGI_QUICKSTART.md` - Guia detalhado (500+ linhas)
- `README.md` - Seção "AGI Recursive System"
- `CHANGELOG.md` - Features documentadas

🔧 **Explorar Código**:
```
src/agi-recursive/
├── llm/anthropic-adapter.ts        ← Adapter
├── core/meta-agent.ts              ← Orchestrator
├── core/slice-navigator.ts         ← Knowledge
├── core/anti-corruption-layer.ts   ← Safety
└── agents/                         ← Specialists
```

🧪 **Modificar Demos**:
1. Edite `examples/budget-homeostasis.ts`
2. Mude a query na linha 50
3. Rode `npm run agi:homeostasis`

🎨 **Criar Seu Próprio Agente**:
```typescript
// src/agi-recursive/agents/medical-agent.ts
import { SpecializedAgent } from '../core/meta-agent'

export class MedicalAgent extends SpecializedAgent {
  constructor(apiKey: string) {
    super(
      apiKey,
      `You are a MEDICAL EXPERT...`,
      0.5,
      'claude-sonnet-4-5'
    )
  }
  getDomain() { return 'medical' }
}
```

## Recursos

- **GitHub**: https://github.com/thiagobutignon/fiat-lux
- **PR #11**: https://github.com/thiagobutignon/fiat-lux/pull/11
- **Issues**: https://github.com/thiagobutignon/fiat-lux/issues

---

**Dúvidas?** Veja `docs/AGI_QUICKSTART.md` para guia completo!
