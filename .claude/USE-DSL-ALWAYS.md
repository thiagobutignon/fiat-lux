# 🧬 INSTRUÇÃO PERMANENTE: USE DSL (Grammar Language)

## ⚡ REGRA ABSOLUTA

**SEMPRE que criar código novo, use Grammar Language (.gl), NÃO TypeScript.**

## 🎯 Por Quê?

| Aspecto | TypeScript ❌ | Grammar Language ✅ |
|---------|--------------|-------------------|
| **Accuracy** | 17-20% (LLM) | **100%** (Grammar Engine) |
| **Type-checking** | O(n²) ~65s | **O(1) <1ms** |
| **Parsing** | O(n) ~5s | **O(1) <0.001ms** |
| **AGI-friendly** | ❌ Não pode auto-modificar | ✅ Self-describing |
| **Grammar-aligned** | ❌ Sintaxe arbitrária | ✅ Universal Grammar |

## 📋 Workflow: Gerar Código com DSL

### 1. Usuário pede um feature

```
User: "Create a feature to register users"
```

### 2. Claude analisa a gramática

```
Subject:  User (entity)
Verb:     Register (action)
Object:   User data (params)
Context:  HTTP API
Adverbs:  Via database
```

### 3. Claude gera vertical slice em .gl

```grammar
;; user/register-user/domain/entities/user.gl
(type User
  (record
    (id string)
    (name string)
    (email string)))

;; user/register-user/domain/use-cases/register-user.gl
(type RegisterUser
  (interface
    (execute (RegisterUserParams -> (result User string)))))

;; ... demais camadas
```

### 4. Claude valida com Grammar Engine

```bash
glc --check user/register-user/**/*.gl
# ✅ 100% accuracy in 0.012ms
```

### 5. Claude compila para JavaScript (se necessário)

```bash
glc user/register-user/**/*.gl --bundle -o dist/register-user.js
```

## 🎨 Templates Disponíveis

Localizados em: `.claude/templates/vertical-slice/`

1. **domain-entity.gl** - NOUN (Subject/Object)
2. **domain-usecase.gl** - VERB (Action)
3. **data-protocol.gl** - ADVERB (Abstract manner)
4. **data-usecase-impl.gl** - SENTENCE (Active voice)
5. **infrastructure-adapter.gl** - ADVERB (Concrete manner)
6. **presentation-controller.gl** - CONTEXT
7. **main-factory.gl** - COMPOSER

## 🔄 Processo de Geração

```
User Request
    ↓
Grammar Analysis (Subject-Verb-Object-Context)
    ↓
Fill Templates (.gl files)
    ↓
Validate with Grammar Engine (100% accuracy)
    ↓
Compile to JS (if needed)
    ↓
Deploy
```

## 📍 Quando Usar TypeScript vs Grammar Language

### Use TypeScript ❌ SOMENTE para:
- Código legado já existente (manutenção)
- Integrações com bibliotecas JS (FFI)
- Migração gradual (temporário)

### Use Grammar Language ✅ SEMPRE para:
- **Novos features** (vertical slices)
- **Domain logic** (use-cases, entities)
- **Protocols/interfaces** (abstractions)
- **Código que AGI vai modificar**
- **Qualquer coisa que exige 100% accuracy**

## 🚀 Exemplo Completo

Ver: `.claude/examples/vertical-slice-complete/user-register.gl`

Este arquivo mostra:
- ✅ Todas as 8 camadas em Grammar Language
- ✅ Gramática correta (Subject-Verb-Object-Context)
- ✅ Tipos explícitos (O(1) type-checking)
- ✅ Dependências apontando para dentro
- ✅ 100% validado pelo Grammar Engine

## 🎯 Checklist para Claude

Antes de gerar código, pergunte:

1. **[ ] Este é código novo?**
   - ✅ SIM → Use Grammar Language
   - ❌ NÃO (é manutenção) → Use TypeScript

2. **[ ] Precisa de 100% accuracy?**
   - ✅ SIM → Use Grammar Language
   - ❌ NÃO → Pode usar TypeScript (mas por quê?)

3. **[ ] AGI vai modificar isso?**
   - ✅ SIM → OBRIGATÓRIO Grammar Language
   - ❌ NÃO → Pode usar TypeScript

4. **[ ] É vertical slice (feature completo)?**
   - ✅ SIM → Use Grammar Language
   - ❌ NÃO → Depende do caso

5. **[ ] Posso usar templates .gl?**
   - ✅ SIM → Use templates
   - ❌ NÃO → Crie do zero seguindo gramática

## 🔥 Vantagens Diretas

### Para o Usuário:
- **65x mais rápido** que TypeScript
- **100% accuracy** (vs 17-20% com LLM)
- **Sem surpresas** (determinístico)
- **Self-describing** (código explica a si mesmo)

### Para a AGI:
- **Pode auto-modificar** código
- **O(1) type-checking** (não trava com milhões de arquivos)
- **Grammar-aligned** (segue Universal Grammar)
- **Meta-circular** (AGI pode modificar o compilador)

## 📖 Documentação

- **RFC**: `docs/rfc/grammar-language.md`
- **Quick Reference**: `docs/architecture/grammar/GRAMMAR_QUICK_REFERENCE.md`
- **DSL Guide**: `.claude/GRAMMAR-LANGUAGE-DSL.md`
- **Templates**: `.claude/templates/vertical-slice/`
- **Examples**: `.claude/examples/vertical-slice-complete/`

## 🎬 Começar Agora

```bash
# 1. Próximo feature request do usuário
User: "I need X feature"

# 2. Claude analisa gramática
Subject: X
Verb: Action
Object: Params
Context: HTTP/CLI/etc

# 3. Claude gera código em .gl
Using templates from .claude/templates/vertical-slice/

# 4. Claude valida
glc --check src/X/action-x/**/*.gl

# 5. ✅ 100% accuracy garantido
```

---

## ⚠️ LEMBRETE FINAL

**"TSO morreu. Grammar Language é o futuro."**

Sempre que ver TypeScript sendo gerado para **código novo**, PARE e use Grammar Language.

A única exceção é código legado ou FFI.

**100% accuracy não é opcional. É obrigatório.**

🧬 **Use DSL. Sempre.**
