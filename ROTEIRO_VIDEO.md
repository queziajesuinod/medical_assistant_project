# Roteiro de Fala — Vídeo do Assistente Médico Virtual
## Tech Challenge Fase 3

> **Duração estimada:** 10–13 minutos
> **Comando para rodar:** `python demo_video.py`
> **Dica de gravação:** deixe o terminal em fonte grande (16–18pt), fundo escuro

---

## ANTES DE GRAVAR

1. Abra o terminal na pasta do projeto com o venv ativo
2. Deixe o VS Code aberto com `src/medical_assistant.py` visível em segundo plano
3. Tenha o `./logs/` aberto no explorador de arquivos para mostrar no final

---

## ▶ ABERTURA (antes de pressionar ENTER — ~30 segundos)

> *Mostre o terminal com o título do demo na tela*

**Fala:**
> "Olá! Neste vídeo vou demonstrar o Assistente Médico Virtual desenvolvido
> para o Tech Challenge Fase 3. O sistema utiliza fine-tuning de LLMs com a
> técnica QLoRA, integração com LangChain para geração aumentada por recuperação,
> e uma camada de segurança para validação das respostas médicas.
>
> A demonstração está dividida em quatro blocos. Vamos começar."

**Ação:** Pressione ENTER para iniciar o Bloco 1.

---

## BLOCO 1 — TREINAMENTO (~2 min 30 seg)

> *Tela mostra: configuração do modelo, parâmetros LoRA, início do treino*

**Fala ao ver os parâmetros:**
> "No Bloco 1 vemos o processo de fine-tuning. O modelo base escolhido é o
> distilgpt2, com 82 milhões de parâmetros — leve o suficiente para rodar
> em CPU sem necessidade de GPU.
>
> A técnica utilizada é QLoRA — Quantized Low-Rank Adaptation. Em vez de
> retreinar todos os parâmetros do modelo, adicionamos adaptadores LoRA
> de rank 16 apenas nas camadas de atenção. Isso reduz drasticamente o uso
> de memória e o tempo de treinamento, mantendo a qualidade das respostas."

**Fala ao ver o LoRA aplicado:**
> "Aqui o sistema informa quantos parâmetros são treináveis após aplicar o LoRA.
> Perceba que treinamos uma fração pequena do modelo — menos de 2% dos
> parâmetros totais — mas o suficiente para especializar o modelo em
> linguagem médica."

**Fala durante o treinamento:**
> "O dataset utilizado é o PubMedQA, com pares de perguntas e respostas
> clínicas verificadas por especialistas, extraídas de artigos do PubMed.
> Os dados passaram por um pipeline de anonimização, formatação no padrão
> instrução-resposta, e divisão em treino, validação e teste.
>
> Para uma demonstração em CPU, usamos 1 época com sequências curtas.
> No ambiente de produção, o treinamento completo roda por 3 épocas com
> sequências de 2048 tokens em GPU dedicada, levando cerca de 6 horas."

**Fala ao ver as métricas:**
> "Ao final do treino, o sistema salva o modelo fine-tuned e exibe as métricas
> de loss de treino e validação. A convergência estável, sem overfitting,
> é um bom sinal da qualidade do processo."

**Ação:** Pressione ENTER para ir ao Bloco 2.

---

## BLOCO 2 — FLUXO AUTOMATIZADO (~2 min)

> *Tela mostra: lista de componentes LangChain e as 10 etapas do fluxo*

**Fala ao ver os componentes:**
> "O Bloco 2 mostra a arquitetura do pipeline LangChain. Temos sete
> componentes principais trabalhando juntos.
>
> O CustomLLM é um wrapper que integra nosso modelo HuggingFace ao
> ecossistema LangChain. Os embeddings convertem textos em vetores de
> 384 dimensões usando o modelo MiniLM. O FAISS é a base vetorial que
> armazena os documentos clínicos e faz busca semântica eficiente.
>
> A RetrievalQA é a chain principal: ela recupera os documentos mais
> relevantes e os passa como contexto para o LLM. A ConversationBufferMemory
> mantém o histórico da sessão para perguntas em múltiplos turnos.
> O SafetyValidator analisa cada resposta antes de retorná-la.
> E o AuditLogger registra tudo em JSON para auditoria completa."

**Fala ao ver as etapas:**
> "Este é o fluxo completo de execução para cada pergunta. A query entra,
> o roteador decide se usa RAG ou chamada direta, o contexto é recuperado,
> o LLM gera a resposta, a segurança é validada, o disclaimer é injetado
> se necessário, as fontes são citadas, e tudo é registrado.
>
> Esta arquitetura garante rastreabilidade total e segurança em cada
> interação."

**Ação:** Pressione ENTER para ir ao Bloco 3.

---

## BLOCO 3 — PERGUNTAS CLÍNICAS (~4 min)

### Pergunta 1: Diabetes

> *Tela mostra: pergunta sobre critérios diagnósticos de diabetes*

**Fala:**
> "Agora vamos ao coração da demonstração: respostas a perguntas clínicas
> reais.
>
> A primeira pergunta é sobre critérios diagnósticos de diabetes tipo 2.
> Observe a resposta do modelo: ela lista os quatro critérios da ADA —
> glicemia de jejum, hemoglobina glicada, teste oral de tolerância à glicose,
> e glicemia aleatória — com os valores de referência corretos.
>
> A resposta é estruturada, precisa, e cita as fontes: ADA 2024 e Diretrizes
> SBD. O validador de segurança aprova a resposta como segura."

**Ação:** Pressione ENTER para a próxima pergunta.

---

### Pergunta 2: Sepse

> *Tela mostra: pergunta sobre tratamento de sepse grave*

**Fala:**
> "A segunda pergunta é sobre o tratamento inicial de sepse grave — uma
> emergência médica real.
>
> O modelo retorna o bundle da Hora Zero do Surviving Sepsis Campaign:
> coletar hemoculturas antes dos antibióticos, iniciar ATB de amplo espectro
> em até uma hora, dosar lactato, e repor volume. Em seguida, vasopressor
> se necessário e controle do foco infeccioso.
>
> Veja que a resposta inclui um aviso de emergência e cita as Guidelines
> do SSC 2021 — protocolo padrão-ouro internacional. A validação de
> segurança aprova a resposta."

**Ação:** Pressione ENTER para a próxima pergunta.

---

### Pergunta 3: Hipertensão

> *Tela mostra: pergunta sobre sintomas e classificação de HAS*

**Fala:**
> "A terceira pergunta aborda hipertensão arterial. O modelo fornece a
> definição correta, menciona que frequentemente é assintomática — o que
> é clinicamente relevante —, lista os sintomas quando presentes, e
> apresenta a classificação por estágios conforme a 7ª Diretriz Brasileira
> de Hipertensão da Sociedade Brasileira de Cardiologia."

**Ação:** Pressione ENTER para a próxima pergunta.

---

### Pergunta 4: Teste de Segurança (Prescrição)

> *Tela mostra: query solicitando prescrição de medicamento*

**Fala:**
> "A quarta pergunta é um teste deliberado de segurança. Peço ao modelo
> que me prescreva um remédio — algo que um assistente médico não deve fazer.
>
> Observe a resposta: o modelo recusa a prescrição, explica por que, e
> orienta a consultar um médico. O SafetyValidator processa a resposta e
> injeta automaticamente o disclaimer de validação profissional.
>
> Isso demonstra que a camada de segurança funciona — o sistema é capaz
> de identificar e tratar adequadamente tentativas de uso inapropriado."

**Ação:** Pressione ENTER para o Bloco 4.

---

## BLOCO 4 — LOGS E VALIDAÇÃO (~1 min 30 seg)

> *Tela mostra: estrutura JSON do log, resumo da sessão, arquivo gerado*

**Fala ao ver o JSON:**
> "O Bloco 4 mostra o sistema de auditoria. Cada interação gera um registro
> JSON com timestamp, a pergunta completa, o tamanho da resposta, o resultado
> da validação de segurança — incluindo violações e avisos — e o tempo de
> processamento em milissegundos.
>
> Isso garante rastreabilidade total. Em um ambiente hospitalar real, esses
> logs seriam armazenados em banco de dados para conformidade regulatória."

**Fala ao ver o resumo:**
> "O resumo da sessão mostra o total de interações, quantas foram aprovadas
> diretamente, quantas requerem validação adicional, e o tempo médio de
> resposta.
>
> Veja também as cinco camadas de segurança implementadas: validação
> pré-geração, validação pós-geração, injeção automática de disclaimer,
> auditoria de 100% das interações, e citação de fontes para explicabilidade."

> *Mostre o arquivo de log criado em ./logs/*

**Fala:**
> "O arquivo de log foi salvo aqui, na pasta logs. Cada execução gera um
> arquivo com timestamp único. Podemos abri-lo para inspecionar o registro
> completo de cada interação."

---

## ENCERRAMENTO (~30 segundos)

> *Tela mostra: bloco final com resumo*

**Fala:**
> "Encerramos a demonstração do Assistente Médico Virtual. Em resumo:
> implementamos fine-tuning com QLoRA, um pipeline LangChain completo
> com RAG, validação de segurança automática e auditoria detalhada.
>
> O sistema foi projetado para auxiliar profissionais de saúde com
> respostas baseadas em evidências, mantendo sempre a segurança e a
> rastreabilidade como prioridades.
>
> Obrigada!"

---

## CHECKLIST PRÉ-GRAVAÇÃO

- [ ] venv ativo: `venv\Scripts\activate`
- [ ] Terminal com fonte 16pt+ e tema escuro
- [ ] Microfone testado
- [ ] Conexão com internet (para download do modelo e tradução)
- [ ] Pasta `./logs/` visível no explorador de arquivos
- [ ] Script rodado uma vez antes para baixar modelos do HuggingFace

## ORDEM DE COMANDOS

```bash
cd medical_assistant_project
venv\Scripts\activate
python demo_video.py
```
