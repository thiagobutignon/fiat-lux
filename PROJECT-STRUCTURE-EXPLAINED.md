# 🗺️ Estrutura de Projetos - Explicação

## 📂 Hierarquia Atual

```
/Users/thiagobutignon/dev/chomsky/
├── 📁 the-regent/                    ← Gemini CLI ORIGINAL (Google)
│   └── Código base do Google
│
├── 📁 gemini-cli-fork/               ← Fork documentado
│   ├── GEMINI.md
│   └── 📁 the-regent/                ← 👈 AQUI ESTAMOS! The Regent + ILP
│       ├── packages/
│       │   ├── core/                 ← ILP integrado aqui
│       │   └── cli/                  ← CLI modificada
│       ├── ILP-INTEGRATION-GUIDE.md
│       ├── IMPLEMENTATION-STATUS.md
│       └── QUICK-START.md
│
├── src/                              ← AGI Recursive (outro projeto)
├── white-paper/                      ← Papers de pesquisa
├── README.md                         ← "Fiat Lux" monorepo
└── ... (outros projetos)
```

## 🎯 Qual é o Projeto Correto?

### ✅ **TRABALHE AQUI** (Recomendado):
```
/Users/thiagobutignon/dev/chomsky/gemini-cli-fork/the-regent/
```

**Por quê?**
- ✅ ILP já integrado (Claude, Constitution, Attention)
- ✅ Build passando 100%
- ✅ Documentação completa criada
- ✅ CLI funcionando
- ✅ Todas as modificações estão aqui

### ⚠️ Outros Diretórios:

**`/Users/thiagobutignon/dev/chomsky/the-regent/`**
- ❌ Gemini CLI **ORIGINAL** do Google (sem ILP)
- ❌ Não modificado
- ℹ️ Pode ser usado como referência

**`/Users/thiagobutignon/dev/chomsky/gemini-cli-fork/`**
- ℹ️ Raiz do fork
- ℹ️ Tem documentação sobre o fork

**`/Users/thiagobutignon/dev/chomsky/`** (root)
- ℹ️ Monorepo "Fiat Lux"
- ℹ️ Contém múltiplos projetos AGI

## 🚀 Como Usar (Passo a Passo Definitivo)

### 1. Ir para o diretório correto

```bash
cd /Users/thiagobutignon/dev/chomsky/gemini-cli-fork/the-regent
```

### 2. Verificar que estamos no lugar certo

```bash
pwd
# Deve mostrar: /Users/thiagobutignon/dev/chomsky/gemini-cli-fork/the-regent

ls -la ILP-INTEGRATION-GUIDE.md
# Se existir, estamos no lugar certo! ✅
```

### 3. Setup e Run

```bash
# Install (se necessário)
npm install

# Build
npm run build

# Configure API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Run CLI
npm start
```

## 📝 Como Programar

### Modificar a CLI

```bash
# Navegar para o diretório correto
cd /Users/thiagobutignon/dev/chomsky/gemini-cli-fork/the-regent

# Editar arquivos
vim packages/cli/src/gemini.tsx

# Rebuild
npm run build

# Testar
npm start
```

### Adicionar Features ILP

**Localização dos arquivos principais:**

```
gemini-cli-fork/the-regent/
├── packages/core/src/
│   ├── ilp/                          ← Todos os módulos ILP
│   │   ├── llm/
│   │   │   ├── anthropic-adapter.ts  ← Claude adapter
│   │   │   └── claude-content-generator.ts ← Novo!
│   │   ├── constitution/
│   │   │   └── constitution.ts       ← Governance
│   │   ├── attention/
│   │   │   └── attention-tracker.ts  ← Tracking
│   │   └── ...
│   ├── core/
│   │   ├── contentGenerator.ts       ← AuthType.USE_CLAUDE
│   │   ├── turn.ts                   ← Constitution enabled
│   │   └── geminiChat.ts             ← Attention tracking
│   └── index.ts                      ← Exports
│
└── packages/cli/src/
    ├── gemini.tsx                    ← Main UI
    ├── commands/                     ← Slash commands
    └── ui/                           ← React components
```

## 🧹 Limpeza Opcional

Se quiser limpar a confusão, você pode:

### Opção 1: Renomear para clarear

```bash
cd /Users/thiagobutignon/dev/chomsky

# Renomear o original
mv the-regent gemini-cli-original

# Agora ficou claro:
# - gemini-cli-original/     ← Google original
# - gemini-cli-fork/the-regent/  ← Nosso projeto com ILP
```

### Opção 2: Mover The Regent para raiz

```bash
cd /Users/thiagobutignon/dev/chomsky

# Backup do original
mv the-regent gemini-cli-google-backup

# Mover nosso projeto para raiz
mv gemini-cli-fork/the-regent ./the-regent-ilp

# Agora:
# - gemini-cli-google-backup/   ← Original
# - the-regent-ilp/             ← Nosso com ILP (limpo!)
# - gemini-cli-fork/            ← Pode deletar se quiser
```

## 🎯 TL;DR - Resumo Executivo

### Onde Trabalhar?
```bash
cd /Users/thiagobutignon/dev/chomsky/gemini-cli-fork/the-regent
```

### Como Rodar?
```bash
npm start
```

### Onde Está o Código?
- **CLI**: `packages/cli/src/`
- **ILP**: `packages/core/src/ilp/`
- **Integration**: `packages/core/src/core/`

### Documentação?
- `ILP-INTEGRATION-GUIDE.md` - Como usar ILP
- `IMPLEMENTATION-STATUS.md` - Status atual
- `QUICK-START.md` - Como rodar
- `PROJECT-STRUCTURE-EXPLAINED.md` - Este arquivo

---

**Tudo claro agora? 🎯**

Você tem 3 projetos mas só precisa trabalhar em 1:
**`gemini-cli-fork/the-regent/`** ← Este é O projeto! 👑
