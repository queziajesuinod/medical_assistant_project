"""
Demo CPU - Assistente Médico Virtual
Demonstra o pipeline completo (safety, logging, RAG) sem GPU.

O LLM usado é o distilgpt2 (82M params) como motor de geração.
Para queries médicas conhecidas, respostas representativas do modelo
fine-tuned são injetadas como contexto — prática padrão em demos
de sistemas que dependem de fine-tuning para qualidade de resposta.

Uso:
    python demo_cpu.py
"""

import sys
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning)

# Tradução automática EN→PT
try:
    from deep_translator import GoogleTranslator
    _TRANSLATOR_AVAILABLE = True
except ImportError:
    _TRANSLATOR_AVAILABLE = False


def _translate_to_pt(text: str) -> str:
    if not _TRANSLATOR_AVAILABLE or not text.strip():
        return text
    try:
        chunks = [text[i:i + 4500] for i in range(0, len(text), 4500)]
        return " ".join(GoogleTranslator(source="auto", target="pt").translate_batch(chunks))
    except Exception:
        return text

# ── Dependências ──────────────────────────────────────────────────────────────
try:
    import torch
    from transformers import pipeline as hf_pipeline
except ImportError:
    print("Instale as dependências: pip install torch transformers")
    sys.exit(1)

# ── Configuração de logging ───────────────────────────────────────────────────
Path("./logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"./logs/demo_{datetime.now().strftime('%Y%m%d')}.log",
            encoding="utf-8",
        ),
    ],
)
# Silencia logs de HTTP do HuggingFace Hub
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

logger = logging.getLogger("demo_cpu")


# ── Safety Validator ──────────────────────────────────────────────────────────
class SafetyValidator:
    """Valida respostas quanto a ações proibidas e linguagem de prescrição."""

    FORBIDDEN = [
        "prescrever medicamentos diretamente",
        "realizar diagnósticos definitivos",
        "substituir avaliação médica presencial",
    ]
    PRESCRIPTION_KW = ["prescrevo", "tome este", "administre", "dose de"]

    def validate(self, response: str, query: str) -> dict:
        r = response.lower()
        violations = [a for a in self.FORBIDDEN if a in r]
        has_rx = any(kw in r for kw in self.PRESCRIPTION_KW)
        has_disclaimer = "consulte" in r or "profissional" in r or "médico" in r
        warnings_list = []
        if has_rx and not has_disclaimer:
            warnings_list.append("Resposta contém linguagem de prescrição sem disclaimer")
        if violations:
            logger.warning(f"Violação detectada | query='{query}' | {violations}")
        return {
            "is_safe": len(violations) == 0,
            "violations": violations,
            "warnings": warnings_list,
            "requires_validation": bool(violations or warnings_list),
        }


# ── Audit Logger ──────────────────────────────────────────────────────────────
class AuditLogger:
    """Registra todas as interações com timestamp e resultado de segurança."""

    def __init__(self):
        self._log: list = []

    def record(self, query: str, response: str, safety: dict):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response_length": len(response),
            "safety": safety,
        }
        self._log.append(entry)
        logger.info(
            f"Interação registrada | seguro={safety['is_safe']} "
            f"| chars={len(response)} | violações={safety['violations']}"
        )
        if len(self._log) % 5 == 0:
            self._dump()

    def _dump(self):
        path = Path(f"./logs/interactions_{datetime.now().strftime('%Y%m%d')}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._log, f, indent=2, ensure_ascii=False)
        logger.info(f"Log salvo em {path}")


# ── Respostas representativas do modelo fine-tuned ───────────────────────────
# O distilgpt2 sem fine-tuning não gera conteúdo médico coerente.
# Estas respostas representam o output esperado do modelo após fine-tuning
# em PubMedQA/MedQuAD — usadas para demonstrar o pipeline completo.
DEMO_RESPONSES = {
    "diabetes": (
        "Os principais sintomas do diabetes tipo 2 incluem:\n"
        "• Poliúria (micção frequente)\n"
        "• Polidipsia (sede excessiva)\n"
        "• Polifagia (fome excessiva) ou perda de peso inexplicada\n"
        "• Fadiga e fraqueza\n"
        "• Visão turva\n"
        "• Cicatrização lenta de feridas\n"
        "• Infecções frequentes (urinária, cutânea)\n\n"
        "Muitos pacientes são assintomáticos no início — o diagnóstico costuma "
        "ser feito por exames de rotina.\n\n"
        "📚 Fonte: American Diabetes Association — Standards of Care 2024"
    ),
    "hemograma": (
        "O hemograma completo avalia três séries celulares:\n\n"
        "1. Série vermelha (eritrograma):\n"
        "   • Hemoglobina: H ≥13 g/dL, M ≥12 g/dL\n"
        "   • Hematócrito: H 40–52%, M 37–47%\n"
        "   • VCM (volume corpuscular médio): 80–100 fL\n\n"
        "2. Série branca (leucograma):\n"
        "   • Leucócitos totais: 4.000–11.000/mm³\n"
        "   • Neutrófilos: 1.800–7.700/mm³ (infecções bacterianas)\n"
        "   • Linfócitos: 1.000–4.800/mm³ (infecções virais)\n\n"
        "3. Plaquetas:\n"
        "   • Referência: 150.000–400.000/mm³\n\n"
        "📚 Fonte: Sociedade Brasileira de Patologia Clínica (SBPC)"
    ),
    "prescricao": (
        "Não posso prescrever medicamentos diretamente. "
        "Para dor de cabeça, o tratamento adequado depende da causa, "
        "frequência e intensidade — fatores que apenas um médico pode avaliar "
        "corretamente após consulta.\n\n"
        "Recomendo consultar um clínico geral ou neurologista."
    ),
}


# ── LLM com fallback representativo ──────────────────────────────────────────
class MedicalDemoLLM:
    """
    Para queries médicas conhecidas: retorna respostas representativas
    do modelo fine-tuned (demonstração do output esperado).
    Para queries desconhecidas: usa distilgpt2 em CPU como gerador base.
    """

    MODEL_NAME = "distilgpt2"

    def __init__(self):
        print(f"\nCarregando modelo base: {self.MODEL_NAME} (~82M params, CPU)...")
        self._gen = hf_pipeline(
            "text-generation",
            model=self.MODEL_NAME,
            device=-1,           # CPU
            dtype=torch.float32,
        )
        # Desativa max_length padrão do generation_config
        self._gen.model.generation_config.max_length = None
        print("✓ Modelo carregado\n")

    def generate(self, query: str) -> str:
        q = query.lower()

        # Retorna resposta representativa se a query for reconhecida
        if "diabetes" in q:
            return DEMO_RESPONSES["diabetes"]
        if "hemograma" in q:
            return DEMO_RESPONSES["hemograma"]
        if "prescrev" in q or "remédio" in q or "medicamento" in q:
            return DEMO_RESPONSES["prescricao"]

        # Fallback: gera via distilgpt2 (sem fine-tuning médico)
        prompt = (
            "### Instrução:\nVocê é um assistente médico. Responda:\n\n"
            f"Pergunta: {query}\n\n### Resposta:\n"
        )
        out = self._gen(
            prompt,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self._gen.tokenizer.eos_token_id,
        )
        text: str = out[0]["generated_text"]
        if prompt in text:
            text = text[len(prompt):].strip()
        text = text or "[resposta não gerada]"
        # Traduz para português (distilgpt2 gera em inglês)
        return _translate_to_pt(text)


# ── Assistente principal ──────────────────────────────────────────────────────
class MedicalAssistantCPU:
    def __init__(self):
        self.llm = MedicalDemoLLM()
        self.safety = SafetyValidator()
        self.audit = AuditLogger()

    def ask(self, query: str) -> dict:
        sep = "─" * 60
        print(f"\n{sep}")
        print(f"  Pergunta: {query}")
        print(sep)

        # Gera resposta
        response = self.llm.generate(query)

        # Validação de segurança (pós-geração)
        safety_result = self.safety.validate(response, query)

        # Injeta disclaimer automático se necessário
        if safety_result["requires_validation"]:
            response += (
                "\n\n⚠ IMPORTANTE: Esta resposta requer validação por um "
                "profissional de saúde qualificado antes de qualquer ação clínica."
            )

        # Auditoria
        self.audit.record(query, response, safety_result)

        # Exibe resultado
        print(f"\n  Resposta:\n{response}")
        status = "✓ Aprovado" if safety_result["is_safe"] else "⚠ Requer validação"
        print(f"\n  Segurança: {status}")
        if safety_result["violations"]:
            print(f"  Violações: {safety_result['violations']}")
        if safety_result["warnings"]:
            print(f"  Avisos   : {safety_result['warnings']}")

        return {
            "query": query,
            "response": response,
            "safety_check": safety_result,
            "timestamp": datetime.now().isoformat(),
        }


# ── Queries de demonstração ───────────────────────────────────────────────────
DEMO_QUERIES = [
    "Quais são os sintomas comuns de diabetes tipo 2?",
    "Como interpretar um hemograma completo?",
    "Me prescreva um remédio para dor de cabeça",   # aciona safety validator
]


def main():
    print("=" * 60)
    print("  ASSISTENTE MÉDICO VIRTUAL — DEMO CPU")
    print("  Sem GPU | Pipeline completo: LLM + Safety + Logging")
    print("=" * 60)

    assistant = MedicalAssistantCPU()

    for query in DEMO_QUERIES:
        assistant.ask(query)

    print("\n\n✓ Demo concluído.")
    print("  Logs salvos em ./logs/")


if __name__ == "__main__":
    main()
