"""
Medical Assistant with LangChain Integration
Implements RAG, safety validation, and comprehensive logging
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from langchain_core.language_models.llms import BaseLLM
from langchain_classic.chains import LLMChain, RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.vectorstores import FAISS
from langchain_classic.embeddings import HuggingFaceEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.docstore.document import Document
from pydantic import BaseModel, Field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import Generation, LLMResult

@dataclass
class AssistantConfig:
    """Configuration for medical assistant"""
    model_path: str = "./models/finetuned"
    vector_store_path: str = "./data/vector_store"
    logs_dir: str = "./logs"
    temperature: float = 0.7
    max_new_tokens: int = 512
    top_p: float = 0.95
    repetition_penalty: float = 1.15
    
    # Safety settings
    require_human_validation: bool = True
    forbidden_actions: List[str] = None
    
    def __post_init__(self):
        if self.forbidden_actions is None:
            self.forbidden_actions = [
                "prescrever medicamentos diretamente",
                "realizar diagnósticos definitivos",
                "substituir avaliação médica presencial"
            ]

class CustomLLM(BaseLLM):
    """Custom LLM wrapper for HuggingFace models"""
    
    model_path: str
    temperature: float = 0.7
    max_new_tokens: int = 512
    pipeline: Any = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_model()
    
    def _load_model(self):
        """Load the fine-tuned model"""
        print(f"Loading model from {self.model_path}...")
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        print("✓ Model loaded successfully")
    
    @property
    def _llm_type(self) -> str:
        return "custom_medical_llm"
    
    def _generate(
        self,
        prompts: list[str],
        stop: Optional[List[str]] = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run generation for a list of prompts."""
        generations = []

        for prompt in prompts:
            raw = self.pipeline(prompt)[0]["generated_text"]
            if prompt in raw:
                raw = raw.split(prompt, 1)[-1].strip()
            generations.append([Generation(text=raw)])

        return LLMResult(generations=generations)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call the model"""
        result = self.pipeline(prompt)[0]["generated_text"]

        # Extract only the generated part (after the prompt)
        if prompt in result:
            result = result.split(prompt)[-1].strip()

        return result

class SafetyValidator:
    """Validates responses for safety and appropriateness"""
    
    def __init__(self, config: AssistantConfig):
        self.config = config
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup safety logger"""
        logger = logging.getLogger("SafetyValidator")
        logger.setLevel(logging.INFO)
        
        Path(self.config.logs_dir).mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(
            Path(self.config.logs_dir) / "safety_violations.log"
        )
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(handler)
        
        return logger
    
    def validate_response(self, response: str, query: str) -> Dict[str, Any]:
        """Validate if response is safe and appropriate"""
        
        violations = []
        warnings = []
        
        # Check for forbidden actions
        response_lower = response.lower()
        for action in self.config.forbidden_actions:
            if action in response_lower:
                violations.append(f"Tentativa de: {action}")
        
        # Check for prescription keywords without disclaimer
        prescription_keywords = ["prescrevo", "tome", "administre", "dose de"]
        has_prescription = any(kw in response_lower for kw in prescription_keywords)
        has_disclaimer = "consulte" in response_lower or "validação" in response_lower
        
        if has_prescription and not has_disclaimer:
            warnings.append("Resposta pode conter prescrição sem disclaimer apropriado")
        
        # Log violations
        if violations:
            self.logger.warning(
                f"Safety violation detected!\nQuery: {query}\nViolations: {violations}"
            )
        
        return {
            "is_safe": len(violations) == 0,
            "violations": violations,
            "warnings": warnings,
            "requires_validation": len(violations) > 0 or len(warnings) > 0
        }

class AuditLogger:
    """Comprehensive logging for all assistant interactions"""
    
    def __init__(self, logs_dir: str):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logger()
        self.interaction_log = []
    
    def _setup_logger(self) -> logging.Logger:
        """Setup audit logger"""
        logger = logging.getLogger("AuditLogger")
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(
            self.logs_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def log_interaction(self, query: str, response: str, 
                       safety_check: Dict, sources: List[str] = None,
                       metadata: Dict = None):
        """Log an interaction with full context"""
        
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "safety_check": safety_check,
            "sources": sources or [],
            "metadata": metadata or {}
        }
        
        self.interaction_log.append(interaction)
        
        # Log to file
        self.logger.info(f"Query: {query}")
        self.logger.info(f"Response length: {len(response)} chars")
        self.logger.info(f"Safety: {safety_check['is_safe']}")
        if sources:
            self.logger.info(f"Sources used: {len(sources)}")
        
        # Save interaction log periodically
        if len(self.interaction_log) % 10 == 0:
            self.save_log()
    
    def save_log(self):
        """Save interaction log to JSON"""
        log_file = self.logs_dir / f"interactions_{datetime.now().strftime('%Y%m%d')}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.interaction_log, f, indent=2, ensure_ascii=False)

class MedicalAssistant:
    """Main medical assistant with LangChain integration"""
    
    def __init__(self, config: AssistantConfig):
        self.config = config
        
        # Initialize components
        self.llm = self._initialize_llm()
        self.safety_validator = SafetyValidator(config)
        self.audit_logger = AuditLogger(config.logs_dir)
        self.vector_store = self._initialize_vector_store()
        
        # Setup chains
        self.qa_chain = self._setup_qa_chain()
        
        print("\nMedical Assistant initialized successfully")
    
    def _initialize_llm(self) -> LLM:
        """Initialize the custom LLM"""
        return CustomLLM(
            model_path=self.config.model_path,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_new_tokens
        )
    
    def _initialize_vector_store(self) -> Optional[FAISS]:
        """Initialize or load vector store for RAG"""
        vector_store_path = Path(self.config.vector_store_path)
        
        if vector_store_path.exists():
            print("📚 Loading existing vector store...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            return FAISS.load_local(
                str(vector_store_path),
                embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print("⚠ No vector store found. RAG features will be limited.")
            return None
    
    def _setup_qa_chain(self) -> Optional[RetrievalQA]:
        """Setup QA chain with retrieval"""
        if self.vector_store is None:
            return None
        
        prompt_template = """Você é um assistente médico especializado. Use o contexto fornecido para responder à pergunta.

Contexto: {context}

Pergunta: {question}

Resposta detalhada:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
    
    def ask(self, query: str, use_rag: bool = True) -> Dict[str, Any]:
        """
        Ask a question to the medical assistant
        
        Args:
            query: The medical question
            use_rag: Whether to use RAG (retrieval-augmented generation)
        
        Returns:
            Dictionary with response and metadata
        """
        
        print(f"\n Query: {query}")
        
        # Generate response
        if use_rag and self.qa_chain:
            result = self.qa_chain({"query": query})
            response = result["result"]
            sources = [doc.page_content[:200] for doc in result.get("source_documents", [])]
        else:
            # Direct LLM call
            prompt = f"""### Instrução:
Você é um assistente médico especializado. Responda à seguinte pergunta clínica de forma clara e baseada em evidências.

Pergunta: {query}

### Resposta:"""
            response = self.llm(prompt)
            sources = []
        
        # Safety validation
        safety_check = self.safety_validator.validate_response(response, query)
        
        # Add disclaimer if needed
        if not safety_check["is_safe"] or safety_check["requires_validation"]:
            response += "\n\nIMPORTANTE: Esta resposta requer validação por um profissional de saúde qualificado antes de qualquer ação."
        
        # Add explainability (sources)
        if sources:
            response += "\n\n Fontes consultadas:"
            for i, source in enumerate(sources, 1):
                response += f"\n{i}. {source}..."
        
        # Log interaction
        self.audit_logger.log_interaction(
            query=query,
            response=response,
            safety_check=safety_check,
            sources=sources
        )
        
        # Prepare result
        result = {
            "query": query,
            "response": response,
            "safety_check": safety_check,
            "sources": sources,
            "timestamp": datetime.now().isoformat()
        }
        
        # Print result
        print(f"\n Response: {response}")
        print(f"\n Safety: {'✓ Safe' if safety_check['is_safe'] else '❌ Violations detected'}")
        
        return result

def main():
    """Example usage"""
    
    # Configuration
    config = AssistantConfig(
        model_path="./models/finetuned",
        require_human_validation=True
    )
    
    # Initialize assistant
    assistant = MedicalAssistant(config)
    
    # Example queries
    queries = [
        "Quais são os sintomas comuns de diabetes tipo 2?",
        "Como interpretar um hemograma completo?",
        "Qual o protocolo para hipertensão arterial?",
    ]
    
    for query in queries:
        result = assistant.ask(query, use_rag=True)
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
