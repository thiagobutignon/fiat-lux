# AgentCoin: Modelo de Economia de Tokens para UBI

**Data:** 08 de Outubro de 2025
**Versão:** 1.0
**Status:** Draft para validação econômica

---

## 1. Executive Summary

Este documento detalha o modelo econômico do AgentCoin, demonstrando como micropagamentos de uso de IA podem financiar Renda Básica Universal (UBI) sustentável e escalável.

**Premissa Econômica:**
```
Valor gerado por IA → Redistribuição automática → Dignidade econômica universal
```

**Métricas Chave (Projeção Ano 5):**
- Volume: $22B USD/ano em transações IA
- UBI Pool: $15.4B USD/ano (70%)
- Beneficiários: 100M pessoas
- UBI/pessoa: $154 USD/ano (~$12.80/mês)

---

## 2. Fontes de Receita

### 2.1 Precificação de Tokens IA

**Tabela de Preços Base (USD por 1M tokens):**

| Modelo | Input | Output | Custo Médio/Query |
|--------|-------|--------|-------------------|
| GPT-4 level | $10 | $30 | $0.02 - $0.05 |
| Claude Sonnet level | $3 | $15 | $0.01 - $0.03 |
| Llama 3 (70B) local | $0.5 | $1.5 | $0.001 - $0.005 |
| Chomsky-v1 (custom) | $2 | $10 | $0.01 - $0.02 |

**Custo Médio Ponderado:** $0.02 USD/query (assumindo mix de modelos)

### 2.2 Distribuição de Valor

```
┌────────────────────────────────────────┐
│  $0.02 USD por query (100%)            │
└────────────────┬───────────────────────┘
                 │
     ┌───────────┴───────────┐
     │                       │
     ▼                       ▼
┌─────────┐            ┌──────────┐
│ 70% UBI │            │ 30% Infra│
│ $0.014  │            │  $0.006  │
└─────────┘            └──────────┘
     │                       │
     │                       ├─→ 50% Compute ($0.003)
     │                       ├─→ 25% Development ($0.0015)
     │                       └─→ 25% Operations ($0.0015)
     │
     └─→ Pool UBI → Distribuição mensal
```

**Justificativa do 70/30 Split:**
- 70% UBI: Maximiza redistribuição (inspirado em Alaska Permanent Fund: 50%+)
- 30% Infra: Garante sustentabilidade operacional
  - Compute: Custos de GPU/CPU para rodar agentes
  - Development: Melhorias contínuas, novos recursos
  - Operations: Segurança, suporte, governança

**Mecanismo de Ajuste:**
- DAO pode votar para mudar split (ex: 75/25, 80/20)
- Limite mínimo para UBI: 60% (hardcoded no protocolo)
- Limite máximo para Infra: 40%

---

## 3. Projeções de Volume

### 3.1 Cenário Conservador

**Ano 1 (Piloto):**
```
Usuários:           1M
Queries/usuário/dia: 5
Dias ativos/mês:    20
Custo médio/query:  $0.02

Cálculo:
Queries/mês = 1M × 5 × 20 = 100M queries
Receita/mês = 100M × $0.02 = $2M USD
Receita/ano = $2M × 12 = $24M USD

UBI Pool/ano = $24M × 0.70 = $16.8M
Beneficiários = 100K
UBI/pessoa/ano = $168 (~$14/mês)
```

**Ano 2 (Expansão):**
```
Usuários:           10M (+900%)
Queries/usuário/dia: 8
Custo médio/query:  $0.015 (economia de escala)

Receita/ano = 10M × 8 × 20 × 12 × $0.015 = $288M
UBI Pool = $288M × 0.70 = $201.6M
Beneficiários = 1M
UBI/pessoa/ano = $201.6 (~$16.80/mês)
```

**Ano 5 (Escala Global):**
```
Usuários:           100M
Queries/usuário/dia: 15
Custo médio/query:  $0.01 (commodity pricing)

Receita/ano = 100M × 15 × 20 × 12 × $0.01 = $3.6B
UBI Pool = $3.6B × 0.70 = $2.52B
Beneficiários = 20M
UBI/pessoa/ano = $126 (~$10.50/mês)
```

### 3.2 Cenário Otimista

**Ano 5 (Adoção massiva):**
```
Usuários:           1B (China + Índia + América Latina)
Queries/usuário/dia: 20 (IA integrada no dia-a-dia)
Custo médio/query:  $0.01

Receita/ano = 1B × 20 × 20 × 12 × $0.01 = $48B
UBI Pool = $48B × 0.70 = $33.6B
Beneficiários = 250M (países em desenvolvimento)
UBI/pessoa/ano = $134.4 (~$11.20/mês)
```

**Nota:** Ano 5 otimista gera UBI comparável a Bolsa Família em escala!

### 3.3 Cenário Pessimista

**Ano 5 (Adoção lenta):**
```
Usuários:           20M (nicho técnico)
Queries/usuário/dia: 10
Custo médio/query:  $0.02 (concorrência baixa)

Receita/ano = 20M × 10 × 20 × 12 × $0.02 = $960M
UBI Pool = $960M × 0.70 = $672M
Beneficiários = 5M
UBI/pessoa/ano = $134.4 (~$11.20/mês)
```

**Conclusão:** Mesmo em cenário pessimista, sistema é viável para milhões.

---

## 4. Análise de Paridade de Poder de Compra (PPP)

### 4.1 Impacto Regional do UBI

**$10-15 USD/mês significa coisas MUITO diferentes globalmente:**

| Região | UBI/mês | % Salário Mín. | Impacto |
|--------|---------|----------------|---------|
| **Brasil (Interior)** | $12 | ~10% | 🟢 Significativo |
| Salário mín: $240/mês | | | Paga alimentação básica |
| **Índia (Rural)** | $12 | ~40% | 🟢🟢 Transformador |
| Renda média: $30/mês | | | Dobra renda familiar |
| **África Subsaariana** | $12 | ~60% | 🟢🟢🟢 Revolucionário |
| Renda média: $20/mês | | | Sai da pobreza extrema |
| **Brasil (São Paulo)** | $12 | ~1% | 🟡 Complementar |
| Salário mín: $1,300/mês | | | Ajuda, mas não transforma |
| **EUA** | $12 | ~0.8% | 🟡 Simbólico |
| Salário mín: $1,500/mês | | | Café por semana |

**Estratégia de Distribuição:**
- **Prioridade 1:** Países com renda < $5/dia (pobreza extrema)
- **Prioridade 2:** Países com renda < $15/dia (pobreza)
- **Prioridade 3:** Regiões pobres em países ricos
- **Longo prazo:** Universal (todos humanos)

### 4.2 Ajuste por Custo de Vida (Opcional)

**Proposta para votação DAO:**

```cpp
// Ajuste opcional baseado em PPP
struct PPPAdjustedUBI {
    CAmount baseUBI;           // Ex: $12 USD
    double pppMultiplier;      // Ex: Índia = 3.5x

    CAmount GetAdjustedUBI(const std::string& country) {
        double multiplier = GetPPPMultiplier(country);
        return baseUBI * multiplier;
    }
};

// Exemplo:
// Índia: $12 × 3.5 = $42 USD (poder de compra equivalente)
// Brasil: $12 × 2.0 = $24 USD
// EUA: $12 × 1.0 = $12 USD
```

**Desafio:** Requer pool UBI maior. Alternativa:
- Fase inicial: UBI fixo global ($12)
- Fase madura: Transição gradual para PPP-adjusted

---

## 5. Comparação com Programas Existentes

### 5.1 Bolsa Família (Brasil)

**Bolsa Família 2024:**
```
Beneficiários:      21 milhões de famílias (~60M pessoas)
Valor médio:        R$600/mês ($120 USD/mês)
Custo total:        R$150 bilhões/ano ($30B USD/ano)
Financiamento:      Impostos (orçamento federal)
```

**AgentCoin UBI (Projeção Ano 10):**
```
Beneficiários:      250M pessoas (4x Bolsa Família)
Valor médio:        $15 USD/mês (12.5% do Bolsa Família)
Custo total:        $45B USD/ano (1.5x Bolsa Família)
Financiamento:      Auto-sustentável (uso de IA)
```

**Vantagens sobre Bolsa Família:**
- ✅ Não depende de orçamento governamental (resistente a mudanças políticas)
- ✅ Escalável globalmente (não limitado a fronteiras)
- ✅ Transparente (blockchain auditável)
- ✅ Resistente à corrupção (distribuição automática via smart contracts)

**Desvantagens:**
- ❌ Valor menor (mas complementar, não substituto)
- ❌ Requer acesso à internet/smartphone
- ❌ Dependente de adoção de IA

**Modelo Ideal:** AgentCoin + Bolsa Família (complementares)

### 5.2 Alaska Permanent Fund

**Alaska PFD (2023):**
```
Beneficiários:      ~650K residentes do Alaska
Valor médio:        $1,312 USD/ano (~$109/mês)
Fonte:              Royalties de petróleo
Modelo:             50% dos lucros → fundo → dividendos
```

**Similaridades com AgentCoin:**
- ✅ Redistribuição de receita de recurso (petróleo vs. computação IA)
- ✅ Distribuição universal (todos residentes/humanos)
- ✅ Sustentável (enquanto houver recurso)

**Diferenças:**
- AgentCoin é renovável (IA não acaba como petróleo)
- AgentCoin é global (não limitado geograficamente)
- AgentCoin tem governança democrática (Alaska PFD é top-down)

---

## 6. Análise de Sustentabilidade

### 6.1 Modelo de Crescimento

**Premissa:** Uso de IA cresce exponencialmente (similar a internet nos anos 90-2000)

```
Crescimento histórico de tecnologias transformadoras:

Internet (1995-2005):
- 1995: 16M usuários → 2005: 1B usuários (62.5x em 10 anos)

Smartphones (2007-2017):
- 2007: 122M usuários → 2017: 5B usuários (40x em 10 anos)

IA/LLMs (2022-2032 projeção):
- 2022: 100M usuários → 2032: 4B usuários? (40x em 10 anos)
```

**AgentCoin projections baseadas em curva S:**

```
Fase 1 (Anos 1-2): Adoção inicial (early adopters)
- Crescimento: 50-100% ao ano
- Usuários: 1M → 10M

Fase 2 (Anos 3-5): Crescimento acelerado (early majority)
- Crescimento: 100-200% ao ano
- Usuários: 10M → 100M

Fase 3 (Anos 6-10): Maturação (late majority)
- Crescimento: 20-50% ao ano
- Usuários: 100M → 1B

Fase 4 (Anos 10+): Saturação (laggards)
- Crescimento: 5-10% ao ano
- Usuários: 1B → 2B+
```

### 6.2 Riscos de Sustentabilidade

**Risco 1: Race to zero pricing**
- **Ameaça:** Concorrência leva preço de tokens IA → $0
- **Mitigação:** Custo computacional tem piso (eletricidade, hardware)
- **Análise:** Mesmo com 90% queda de preços, sistema permanece viável

**Risco 2: Concentração de uso**
- **Ameaça:** Poucos power users geram maior parte da receita
- **Mitigação:** Cap de contribuição individual ($100/mês)
- **Análise:** Incentiva distribuição ampla

**Risco 3: Substituição por IA gratuita**
- **Ameaça:** Modelos open-source gratuitos competem
- **Mitigação:**
  - AgentCoin suporta modelos locais (Llama, etc.)
  - Taxa mínima (~$0.001) mesmo para local
  - Valor agregado: UBI + Governança + Comunidade

**Risco 4: Regulação hostil**
- **Ameaça:** Governos proíbem sistema (visto como ameaça fiscal)
- **Mitigação:**
  - Descentralização (impossível de "desligar")
  - Lobby com evidências de impacto social
  - Parcerias com governos progressistas

---

## 7. Token Supply e Inflação

### 7.1 Modelo de Emissão

**Diferença crítica vs. Bitcoin:**
- Bitcoin: Supply fixo (21M BTC) → deflacionário
- AgentCoin: Supply elástico → estável

**Mecanismo:**
```
Supply AgentCoin = Demanda por IA queries

Se uso de IA ↑ → Mais tokens emitidos → Mais UBI distribuído
Se uso de IA ↓ → Menos tokens → Menos UBI (mas sistema se ajusta)
```

**Stablecoin mechanism:**
```cpp
class AgentCoinSupply {
public:
    // Token é sempre atrelado a $1 USD (stablecoin)
    CAmount GetTokenPrice() const {
        return COIN;  // 1 AgentCoin = $1 USD sempre
    }

    // Emissão elástica baseada em demanda
    void MintTokens(CAmount usdValue) {
        // Quando usuário deposita $10 USD
        // → Sistema minta 10 AgentCoins
        // → Usuário usa para queries
        // → 70% vai para UBI pool
        // → UBI pool distribui tokens
        // → Beneficiários podem redimir por USD
    }

    void BurnTokens(CAmount tokens) {
        // Quando beneficiário resgata UBI
        // → Tokens são queimados
        // → Mantém paridade 1:1 com USD
    }
};
```

**Vantagens:**
- ✅ Sem inflação (sempre 1 token = $1 USD)
- ✅ Sem especulação (token é utility, não investimento)
- ✅ UBI previsível (beneficiários sabem quanto vale)

### 7.2 Reservas de Estabilidade

**Reserve Pool (inspirado em Dai/MakerDAO):**
```
Reservas mantidas:
- 110% do supply em circulation (over-collateralized)
- Composição: 50% USD, 30% BTC, 20% ETH
- Gerenciado por DAO (não por entidade única)

Exemplo:
Se 100M AgentCoins em circulação
→ Reservas de $110M em ativos
→ Garante resgate mesmo se 10% inadimplência
```

---

## 8. Análise de Casos de Uso

### 8.1 Persona 1: Maria (Beneficiária, Brasil Interior)

**Contexto:**
- Renda: R$400/mês ($80 USD)
- Família: 4 pessoas
- Localização: Interior de Minas Gerais

**Impacto do UBI ($12/mês):**
```
Antes:
R$400/mês = R$100/pessoa/semana
Gasta: R$300 alimentação, R$100 contas

Depois (com UBI):
R$400 + R$60 (UBI) = R$460/mês
Gasta: R$300 alimentação, R$100 contas, R$60 educação filhos

Impacto: +15% renda → Pode pagar curso técnico filho mais velho
```

### 8.2 Persona 2: Raj (Usuário + Beneficiário, Índia)

**Contexto:**
- Desenvolvedor freelancer
- Usa IA para código (~20 queries/dia)
- Renda: $200/mês

**Fluxo econômico:**
```
Raj paga: 20 queries/dia × $0.01 × 30 dias = $6/mês
Raj recebe: $12/mês (UBI)
Net: +$6/mês (recebe mais do que paga!)

Impacto: Sistema é progressivo - redistribui de usuários ricos
para beneficiários + usuários de baixa renda
```

### 8.3 Persona 3: TechCorp (Empresa, EUA)

**Contexto:**
- Usa AgentCoin para customer support (1M queries/mês)
- Custo: $10K/mês

**Visão da empresa:**
```
Custo AgentCoin: $10K/mês
Custo anterior (humanos): $50K/mês
Economia: $40K/mês

Percepção: "Economizamos E contribuímos para UBI - win-win"

Impacto: $7K/mês (70% de $10K) vai para UBI
→ Sustenta 583 beneficiários ($12/pessoa)
```

---

## 9. Métricas de Sucesso Econômico

### 9.1 KPIs Primários

**Ano 1:**
- [ ] Volume transacionado: > $20M USD
- [ ] UBI distribuído: > $14M USD
- [ ] Beneficiários ativos: > 100K pessoas
- [ ] Custo médio/query: < $0.02 USD
- [ ] Taxa de retenção usuários: > 60%

**Ano 5:**
- [ ] Volume transacionado: > $3B USD
- [ ] UBI distribuído: > $2B USD
- [ ] Beneficiários ativos: > 20M pessoas
- [ ] Cobertura geográfica: > 50 países
- [ ] Gini coefficient interno: < 0.3

### 9.2 Métricas de Impacto Social

**Baseadas em estudos de UBI (GiveDirectly, Stockton):**

```
Expected outcomes (3 anos):

- 📈 Aumento de renda familiar: +10-15%
- 📚 Aumento de matrícula escolar: +5-8%
- 🏥 Melhoria de acesso à saúde: +12%
- 💼 Aumento de empreendedorismo: +15%
- 😊 Melhoria de saúde mental: +20%
- 🍞 Redução de insegurança alimentar: -25%
```

**Medição:**
- Surveys trimestrais com beneficiários
- Parceria com universidades para estudos longitudinais
- Dados on-chain (transparentes e auditáveis)

---

## 10. Comparação: AgentCoin vs. Outros Modelos

### 10.1 vs. Criptomoedas Especulativas

| Aspecto | Bitcoin/Altcoins | AgentCoin |
|---------|------------------|-----------|
| **Propósito** | Reserva de valor, especulação | Utility (pagamento IA + UBI) |
| **Volatilidade** | Alta (50-80% ao ano) | Baixa (pegged a USD) |
| **Holders** | Investidores, whales | Usuários, beneficiários |
| **Distribuição** | Concentrada (top 1% tem 90%) | Redistributiva (Gini < 0.3) |
| **Valor social** | Limitado | Alto (UBI, redução pobreza) |

### 10.2 vs. Plataformas IA Comerciais

| Aspecto | OpenAI/Anthropic | AgentCoin |
|---------|------------------|-----------|
| **Receita** | Capturada por acionistas | 70% redistribuída |
| **Governança** | CEO + Board | DAO democrática |
| **Transparência** | Parcial | Total (blockchain) |
| **Acesso** | Paywall ($20-100/mês) | Pay-per-use ($0.01-0.05) |
| **Impacto social** | Indireto | Direto (UBI) |

### 10.3 vs. Programas Governamentais UBI

| Aspecto | Gov. UBI (ex: Bolsa Família) | AgentCoin |
|---------|------------------------------|-----------|
| **Financiamento** | Impostos (político) | Auto-sustentável (tecnológico) |
| **Escala** | Nacional | Global |
| **Burocracia** | Alta | Baixa (automática) |
| **Corrupção** | Risco médio | Risco baixo (blockchain) |
| **Sustentabilidade** | Depende de gov. | Depende de adoção IA |
| **Valor** | Alto ($50-200/mês) | Baixo ($10-20/mês) |

**Conclusão:** Modelos são complementares, não competitivos!

---

## 11. Roadmap Econômico

### Fase 1: Bootstrap (Anos 1-2)

**Objetivo:** Provar viabilidade econômica

```
Meta financeira:
- Ano 1: $24M transacionado, $16.8M UBI, 100K beneficiários
- Ano 2: $288M transacionado, $201.6M UBI, 1M beneficiários

Ações:
- Grant inicial: $2M (18 meses runway)
- Partnerships: 5 projetos IA existentes integram AgentCoin
- Marketing: Foco em early adopters técnicos + ONGs
```

### Fase 2: Escala (Anos 3-5)

**Objetivo:** Crescimento exponencial

```
Meta financeira:
- Ano 5: $3.6B transacionado, $2.52B UBI, 20M beneficiários

Ações:
- Parcerias governamentais: 3+ países piloto oficial
- Integração massiva: Top 50 apps IA integram AgentCoin
- Educação: Campanha global sobre UBI via IA
```

### Fase 3: Maturação (Anos 6-10)

**Objetivo:** Sistema estabelecido globalmente

```
Meta financeira:
- Ano 10: $48B transacionado, $33.6B UBI, 250M beneficiários

Ações:
- Padrão de facto: AgentCoin como principal payment rail para IA
- Regulação: Reconhecimento legal em 50+ países
- Evolução: DAO decide próximas fronteiras (IoT? Robótica?)
```

---

## 12. Análise de Sensibilidade

### 12.1 Variáveis Críticas

**Simulação de impacto em UBI/pessoa (Ano 5):**

| Variável | Base | -50% | +50% | Impacto UBI |
|----------|------|------|------|-------------|
| Usuários | 100M | 50M | 150M | $252 → $126 → $378 |
| Queries/dia | 15 | 7.5 | 22.5 | $126 → $63 → $189 |
| Custo/query | $0.01 | $0.005 | $0.015 | $126 → $63 → $189 |
| UBI % | 70% | 35% | 85% | $126 → $63 → $153 |
| Beneficiários | 20M | 10M | 30M | $126 → $252 → $84 |

**Insights:**
- **Mais sensível a:** Número de beneficiários (inverso)
- **Menos sensível a:** UBI % (DAO pode ajustar)
- **Sweet spot:** 100M usuários, 20M beneficiários iniciais

### 12.2 Break-even Analysis

**Pergunta:** Quantos usuários necessários para sustentar X beneficiários?

```
Fórmula:
Usuários necessários = (Beneficiários × UBI desejado) / (Queries/usuário/mês × Custo/query × 70%)

Exemplos:
1M beneficiários, $12/mês UBI:
= (1M × $12) / (300 queries × $0.01 × 0.70)
= $12M / $2.10
= 5.7M usuários necessários

Ratio: ~6 usuários para cada 1 beneficiário
```

**Viabilidade:** Ratio de 6:1 é muito alcançável (similar a impostos progressivos onde minoria paga maioria dos impostos sociais).

---

## 13. Próximos Passos - Validação Econômica

### 13.1 Peer Review

- [ ] Submeter modelo para economistas especializados em UBI
- [ ] Consulta com Banco Mundial / UNDP para feedback
- [ ] Apresentar em conferência de Basic Income Earth Network (BIEN)

### 13.2 Simulação Detalhada

- [ ] Modelagem agent-based (NetLogo ou similar)
- [ ] Simulação Monte Carlo (1000 cenários)
- [ ] Stress testing (choques econômicos, crises)

### 13.3 Piloto Econômico

- [ ] Testar com 1000 usuários + 100 beneficiários reais
- [ ] Duração: 6 meses
- [ ] Métricas: Elasticidade de demanda, impacto social real
- [ ] Ajustar modelo baseado em dados empíricos

---

## 14. Conclusão Econômica

**O modelo é viável?** ✅ **SIM**, sob estas condições:

1. **Adoção mínima:** 5-10M usuários ativos (atingível em 2-3 anos)
2. **Pricing sustentável:** $0.005-0.02/query (alinhado com mercado)
3. **Governança responsável:** DAO mantém 60%+ para UBI
4. **Diversificação:** Não dependência de single model/provider

**Impacto projetado (Ano 5):**
- $2.5B+ redistribuídos para UBI
- 20M+ pessoas beneficiadas
- Redução mensurável de pobreza em comunidades piloto
- Modelo replicável para outras aplicações (IoT, robótica)

**Diferencial transformador:**
Este não é apenas um sistema de pagamentos. É uma **infraestrutura de redistribuição de riqueza automatizada** que responde à pergunta:

*"Como garantimos que os benefícios da IA sejam compartilhados com toda a humanidade, não apenas uma elite tecnocrática?"*

AgentCoin oferece uma resposta técnica, econômica e eticamente fundamentada.

---

## Referências Econômicas

1. "The Economics of Basic Income" - Philippe Van Parijs (2017)
2. Alaska Permanent Fund financial reports (2010-2024)
3. Bolsa Família impact studies - IPEA Brasil (2015-2023)
4. GiveDirectly UBI Kenya study (2023)
5. "Radical Markets" - Eric Posner & Glen Weyl (2018)
6. OpenAI pricing analysis (API costs 2023-2025)
7. World Bank poverty data (2024)
8. IMF reports on UBI economic viability (2020-2024)

