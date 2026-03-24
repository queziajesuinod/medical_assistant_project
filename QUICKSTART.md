# 🚀 Guia de Início Rápido - Medical Assistant

Este guia mostra como começar rapidamente com o projeto.

## ⚡ Setup Rápido (5 minutos)

### 1. Pré-requisitos
```bash
# Verifique suas versões
python --version  # Precisa 3.9+
nvidia-smi       # Verifique sua GPU

# Recomendado: GPU com 16GB+ VRAM
```

### 2. Instalação
```bash
# Clone o repositório
git clone <seu-repo-url>
cd medical_assistant_project

# Crie ambiente virtual
python -m venv venv

# Ative o ambiente
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instale dependências
pip install -r requirements.txt
```

### 3. Configuração
```bash
# Copie o arquivo de configuração
cp .env.example .env

# Edite .env e adicione seu token do HuggingFace
# HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# 
# Obtenha seu token em: https://huggingface.co/settings/tokens
```

## 🎯 Workflow Completo (Primeira Execução)

### Passo 1: Preparar os Dados (15-30 min)
```bash
python src/data_preparation.py
```

**O que acontece:**
- ✅ Download automático do PubMedQA
- ✅ Processamento e formatação
- ✅ Anonimização de dados
- ✅ Criação de splits (train/val/test)

**Output esperado:**
```
data/processed/
├── train.json (800 exemplos)
├── val.json (100 exemplos)
├── test.json (100 exemplos)
└── dataset_statistics.json
```

### Passo 2: Fine-Tuning do Modelo (6-8 horas em RTX 3090)
```bash
python src/finetune_model.py
```

**O que acontece:**
- ✅ Download do LLaMA 2 7B (se necessário)
- ✅ Setup de quantização 4-bit (QLoRA)
- ✅ Treinamento por 3 épocas
- ✅ Salvamento do modelo fine-tunado

**Output esperado:**
```
models/finetuned/
├── adapter_config.json
├── adapter_model.bin
├── tokenizer_config.json
└── training_config.json
```

**💡 Dica:** Use `tmux` ou `screen` para deixar rodando (Linux/Mac):
```bash
tmux new -s training
python src/finetune_model.py
# Ctrl+B, depois D para detach
# tmux attach -t training para voltar
```

**💡 Para Windows:** Use PowerShell:
```powershell
Start-Job -ScriptBlock { python src/finetune_model.py }
# Ver jobs: Get-Job
# Ver output: Receive-Job -Id 1 -Keep
```

Ou abra uma **nova janela PowerShell**:
```powershell
Start-Process powershell -ArgumentList "python src/finetune_model.py"
```

### Passo 3: Avaliar o Modelo (30-60 min)
```bash
python src/evaluate_model.py
```

**Métricas calculadas:**
- Perplexity
- BLEU Score
- ROUGE-1, ROUGE-2, ROUGE-L
- Exemplos qualitativos

### Passo 4: Executar o Assistente
```bash
# Demo interativo
python quick_demo.py

# Ou diretamente
python src/medical_assistant.py

# Ou via notebook
jupyter notebook notebooks/03_demo.ipynb
```

## 📝 Exemplo de Uso Rápido

```python
from src.medical_assistant import MedicalAssistant, AssistantConfig

# Configurar
config = AssistantConfig(
    model_path="./models/finetuned",
    require_human_validation=True
)

# Inicializar
assistant = MedicalAssistant(config)

# Fazer pergunta
result = assistant.ask(
    "Quais são os sintomas de diabetes tipo 2?",
    use_rag=True
)

# Ver resposta
print(result["response"])
```

## 🎬 Para o Vídeo do Tech Challenge

### Estrutura Sugerida (15 minutos):

**[0-2 min] Introdução**
- Apresentação do projeto
- Objetivos e motivação
- Arquitetura em alto nível

**[2-6 min] Demonstração dos Dados**
- Mostrar datasets (PubMedQA, MedQuAD)
- Processo de preparação
- Estatísticas dos dados

**[6-10 min] Fine-Tuning**
- Explicar QLoRA
- Mostrar logs de treinamento
- Métricas de avaliação

**[10-13 min] Assistente em Ação**
- Demo interativo
- Perguntas variadas
- Mostrar RAG funcionando
- Destacar safety features

**[13-14 min] Logs e Auditoria**
- Mostrar sistema de logging
- Explicar rastreabilidade
- Validação de segurança

**[14-15 min] Conclusão**
- Resultados alcançados
- Próximos passos
- Agradecimentos

### Comandos para Gravar:

```bash
# Terminal 1: Logs em tempo real
tail -f logs/audit_*.log

# Terminal 2: Demo do assistente
python quick_demo.py

# Terminal 3: Jupyter para visualizações
jupyter notebook notebooks/03_demo.ipynb
```

## 🐛 Troubleshooting Rápido

### Erro: CUDA out of memory
```bash
# Reduza o batch size em src/finetune_model.py:
per_device_train_batch_size = 2  # em vez de 4
gradient_accumulation_steps = 8  # em vez de 4
```

### Erro: Model not found
```bash
# Certifique-se de ter o token do HuggingFace
export HUGGINGFACE_TOKEN=hf_your_token_here

# Ou adicione no .env
```

### Erro: Import error
```bash
# Reinstale dependências
pip install -r requirements.txt --force-reinstall
```

### Download lento
```bash
# Use mirror brasileiro do HuggingFace (se disponível)
# ou faça download manual e coloque em ./models/base/
```

## 📚 Recursos Adicionais

- **Documentação Completa:** Ver `README.md`
- **Relatório Técnico:** Ver `docs/RELATORIO_TECNICO.md`
- **Notebooks de Exemplo:** Ver `notebooks/`
- **Testes:** `pytest tests/`

## ⏱️ Estimativa de Tempo Total

| Etapa | RTX 3090 | RTX 4090 | A100 |
|-------|----------|----------|------|
| Setup | 15 min | 15 min | 15 min |
| Preparação de Dados | 30 min | 30 min | 30 min |
| Fine-Tuning | 7h | 5h | 3h |
| Avaliação | 1h | 45 min | 30 min |
| **TOTAL** | **~9h** | **~6.5h** | **~4.5h** |

## 💡 Dicas Finais

1. **Use tmux/screen** para processos longos
2. **Monitore GPU** com `watch -n 1 nvidia-smi`
3. **Faça backups** do modelo treinado
4. **Teste gradualmente** antes de gravar o vídeo
5. **Prepare exemplos** de perguntas interessantes

## 🆘 Precisa de Ajuda?

- 📧 Email: seu.email@example.com
- 💬 GitHub Issues: <repo-url>/issues
- 📖 Documentação: Ver README.md completo

---

**Boa sorte com o Tech Challenge! 🚀**
