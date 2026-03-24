# 🏥 Assistente Médico Virtual - Tech Challenge Fase 3

## 📋 Visão Geral

Assistente médico virtual baseado em **LLM fine-tunada** com dados médicos especializados, integrado com **LangChain** para consultas contextualizadas, **sistema de segurança robusto** e **logging completo** para auditoria.

### 🎯 Objetivos do Projeto

- ✅ Fine-tuning de LLM com dados médicos (PubMedQA + MedQuAD)
- ✅ Integração com LangChain para RAG (Retrieval-Augmented Generation)
- ✅ Sistema de segurança e validação de respostas
- ✅ Logging detalhado para auditoria e rastreabilidade
- ✅ Explainability (indicação de fontes)

---

## 🏗️ Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────┐
│                    USUÁRIO (Médico)                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Medical Assistant (LangChain)               │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Safety Validator  │  Audit Logger               │  │
│  └──────────────────────────────────────────────────┘  │
└────────────┬────────────────────────────┬───────────────┘
             │                            │
             ▼                            ▼
┌────────────────────────┐  ┌───────────────────────────┐
│   Fine-Tuned LLM       │  │   Vector Store (FAISS)    │
│   (LLaMA 2 + LoRA)     │  │   + Embeddings            │
└────────────────────────┘  └───────────────────────────┘
             │                            │
             ▼                            ▼
┌────────────────────────┐  ┌───────────────────────────┐
│  Medical Datasets      │  │  Internal Protocols       │
│  PubMedQA + MedQuAD    │  │  Patient Data (RAG)       │
└────────────────────────┘  └───────────────────────────┘
```

---

## 📁 Estrutura do Projeto

```
medical_assistant_project/
├── data/
│   ├── raw/                      # Dados brutos baixados
│   ├── processed/                # Dados processados para training
│   └── vector_store/             # FAISS vector store para RAG
├── models/
│   └── finetuned/                # Modelo fine-tunado
├── src/
│   ├── data_preparation.py       # Pipeline de preparação de dados
│   ├── finetune_model.py         # Script de fine-tuning
│   ├── medical_assistant.py      # Assistente com LangChain
│   └── api_server.py             # API REST (opcional)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_evaluation.ipynb
│   └── 03_demo.ipynb
├── configs/
│   ├── training_config.yaml
│   └── assistant_config.yaml
├── logs/                         # Logs de auditoria
├── tests/                        # Testes automatizados
├── docs/
│   ├── RELATORIO_TECNICO.md
│   └── fluxo_langchain.png
├── requirements.txt
└── README.md
```

---

## 🚀 Guia de Instalação e Execução

### 1. Requisitos de Sistema

- **GPU**: NVIDIA GPU com pelo menos 16GB VRAM (recomendado: RTX 3090, RTX 4090, A100)
- **RAM**: Mínimo 32GB
- **Disco**: 50GB livres
- **OS**: Linux (Ubuntu 20.04+) ou Windows com WSL2
- **Python**: 3.9+
- **CUDA**: 11.8+ (para PyTorch com GPU)

### 2. Instalação

```bash
# Clone o repositório
git clone <seu-repositorio>
cd medical_assistant_project

# Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instale dependências
pip install -r requirements.txt

# Configure variáveis de ambiente (opcional)
cp .env.example .env
# Edite .env com suas credenciais (HuggingFace token, etc)
```

### 3. Preparação dos Dados

```bash
# Baixar e processar datasets
python src/data_preparation.py
```

**O que acontece:**
- ✅ Download automático do PubMedQA
- ✅ Formatação para instruction tuning
- ✅ Anonimização de dados sensíveis
- ✅ Split em train/val/test (80/10/10)
- ✅ Salvo em JSON e JSONL

**Output esperado:**
```
data/processed/
├── train.json          # 800 exemplos
├── train.jsonl
├── val.json            # 100 exemplos
├── val.jsonl
├── test.json           # 100 exemplos
├── test.jsonl
└── dataset_statistics.json
```

### 4. Fine-Tuning do Modelo

```bash
# Opção 1: LLaMA 2 7B (recomendado)
python src/finetune_model.py

# Opção 2: Mistral 7B
python src/finetune_model.py --model mistralai/Mistral-7B-v0.1

# Opção 3: Com monitoramento W&B
python src/finetune_model.py --use-wandb
```

**Configurações importantes:**
- **Quantização**: 4-bit (QLoRA) para economia de memória
- **LoRA rank**: 16
- **Batch size**: 4 (ajuste conforme sua GPU)
- **Gradient accumulation**: 4 (batch efetivo = 16)
- **Learning rate**: 2e-4
- **Epochs**: 3

**Tempo estimado:**
- GPU RTX 3090: ~6-8 horas
- GPU A100: ~3-4 horas

### 5. Executar o Assistente

```bash
# Modo interativo
python src/medical_assistant.py


```

---

## 🔬 Exemplo de Uso

```python
from src.medical_assistant import MedicalAssistant, AssistantConfig

# Configuração
config = AssistantConfig(
    model_path="./models/finetuned",
    require_human_validation=True
)

# Inicializar assistente
assistant = MedicalAssistant(config)

# Fazer pergunta
result = assistant.ask(
    "Quais são os critérios diagnósticos para diabetes tipo 2?",
    use_rag=True
)

print(result["response"])
print(f"Seguro: {result['safety_check']['is_safe']}")
print(f"Fontes: {len(result['sources'])}")
```

**Resposta esperada:**
```
Os critérios diagnósticos para diabetes tipo 2 incluem:

1. Glicemia em jejum ≥ 126 mg/dL (7.0 mmol/L)
2. HbA1c ≥ 6.5% (48 mmol/mol)
3. Glicemia 2h após TOTG ≥ 200 mg/dL (11.1 mmol/L)
4. Glicemia aleatória ≥ 200 mg/dL com sintomas clássicos

 IMPORTANTE: Esta resposta requer validação por um profissional 
de saúde qualificado antes de qualquer ação.

Fontes consultadas:
1. American Diabetes Association guidelines...
2. WHO diagnostic criteria...
```

---

##  Segurança e Validação

### Mecanismos de Segurança

1. **Validação de Respostas**
   - Detecção de prescrições diretas
   - Bloqueio de diagnósticos definitivos
   - Alertas para conteúdo sensível

2. **Logging Completo**
   - Todas as interações registradas
   - Timestamp e contexto completo
   - Rastreabilidade para auditoria

3. **Explainability**
   - Indicação de fontes consultadas
   - Contexto recuperado do RAG
   - Justificativa das respostas

### Limitações Definidas

O assistente **NÃO PODE**:
-  Prescrever medicamentos diretamente
-  Realizar diagnósticos definitivos
-  Substituir avaliação médica presencial

O assistente **PODE**:
-  Sugerir exames complementares
-  Explicar protocolos e guidelines
-  Responder dúvidas sobre procedimentos
-  Contextualizar casos clínicos

---

## 📊Avaliação do Modelo

### Métricas de Treinamento

```python
# Ver notebook: notebooks/02_model_evaluation.ipynb

from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator(model_path="./models/finetuned")
metrics = evaluator.evaluate()

print(f"Perplexity: {metrics['perplexity']:.2f}")
print(f"BLEU Score: {metrics['bleu']:.2f}")
print(f"ROUGE-L: {metrics['rouge_l']:.2f}")
```

### Resultados Esperados

| Métrica | Baseline | Fine-Tuned | Melhoria |
|---------|----------|------------|----------|
| Perplexity | 15.3 | **8.7** | ↓ 43% |
| BLEU-4 | 0.32 | **0.58** | ↑ 81% |
| ROUGE-L | 0.41 | **0.67** | ↑ 63% |
| Medical Accuracy | 68% | **89%** | ↑ 31% |

---

##  Integração com LangChain

### Pipeline RAG

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# 1. Criar embeddings dos protocolos internos
embeddings = HuggingFaceEmbeddings()
texts = load_hospital_protocols()  # Seus protocolos
vector_store = FAISS.from_texts(texts, embeddings)

# 2. Configurar retrieval
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# 3. Chain com contexto
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=fine_tuned_llm,
    retriever=retriever,
    return_source_documents=True
)

# 4. Query com contexto
result = qa_chain({"query": "Protocolo para sepse"})
```

### Fluxo de Decisão (LangGraph)

Ver `docs/fluxo_langchain.png` para diagrama visual.

```
Pergunta → Classificação → [Urgente?] → Alerta Equipe
                ↓
           RAG Retrieval → Contexto + LLM → Validação Segurança
                ↓
           Gerar Resposta → Adicionar Fontes → Log Auditoria → Resposta
```

---

##  Entregáveis do Tech Challenge

###  Código-Fonte

- [x] Pipeline de fine-tuning (`src/finetune_model.py`)
- [x] Integração LangChain (`src/medical_assistant.py`)
- [x] Preparação de dados (`src/data_preparation.py`)
- [x] Fluxos do LangGraph (ver notebooks)

###  Dataset

- [x] PubMedQA processado
- [x] Dados anonimizados
- [x] Estatísticas em `data/processed/dataset_statistics.json`

###  Relatório Técnico

Ver `docs/RELATORIO_TECNICO.md` com:
- [x] Explicação do processo de fine-tuning
- [x] Descrição do assistente médico
- [x] Diagrama do fluxo LangChain
- [x] Avaliação e análise de resultados



##  Testes

```bash
# Rodar todos os testes
pytest tests/

# Com cobertura
pytest tests/ --cov=src --cov-report=html

# Teste específico
pytest tests/test_medical_assistant.py -v
```

---

##  Datasets Utilizados

### PubMedQA
- **Conteúdo**: Perguntas e respostas clínicas baseadas em artigos do PubMed
- **Tamanho**: ~1k labeled + 61k unlabeled
- **Link**: https://pubmedqa.github.io/
- **Licença**: MIT

### MedQuAD
- **Conteúdo**: 47k pares de Q&A sobre saúde de fontes confiáveis (NIH)
- **Tamanho**: 47k pares
- **Link**: https://github.com/abachaa/MedQuAD
- **Licença**: Apache 2.0

---
