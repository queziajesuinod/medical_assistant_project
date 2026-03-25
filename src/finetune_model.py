"""
Fine-Tuning Pipeline for Medical LLM
Uses LoRA/QLoRA for efficient training on consumer GPUs
Supports LLaMA 2, Mistral, and other models
"""

import os
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset
import wandb

@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning"""
    
    # Model settings
    model_name: str = "distilgpt2"  # Changed to smaller model for CPU
    use_4bit: bool = False
    use_8bit: bool = False  # Changed to False
    
    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["c_attn"])  # Changed for GPT2
    
    # Training settings
    output_dir: str = "./models/finetuned"
    num_train_epochs: int = 1  # Reduced for demo
    per_device_train_batch_size: int = 1  # Reduced
    per_device_eval_batch_size: int = 1  # Reduced
    gradient_accumulation_steps: int = 1  # Reduced
    learning_rate: float = 2e-4
    max_grad_norm: float = 0.3
    warmup_steps: int = 10  # Reduced
    weight_decay: float = 0.001
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 50  # Reduced
    save_steps: int = 50  # Reduced
    use_wandb: bool = False
    
    # Data
    data_dir: str = "./data/processed"
    max_seq_length: int = 512  # Reduced
    
    # Hardware — fp16 requer GPU; desabilitado para rodar em CPU
    fp16: bool = False
    bf16: bool = False

class MedicalLLMFineTuner:
    """Fine-tune LLM for medical assistant"""
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging and experiment tracking"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"medical_llm_{timestamp}"
        
        if self.config.use_wandb:
            wandb.init(
                project="medical-assistant",
                name=self.run_name,
                config=self.config.__dict__
            )
        
        print(f"🏃 Run name: {self.run_name}")
    
    def load_model_and_tokenizer(self):
        """Load base model with quantization"""
        print("\n" + "="*60)
        print(f"📦 LOADING MODEL: {self.config.model_name}")
        print("="*60)
        
        # Quantization config
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            print("✓ Using 4-bit quantization (QLoRA)")
        elif self.config.use_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True  # Allow CPU offload for 8-bit
            )
            print("✓ Using 8-bit quantization")
        else:
            bnb_config = None
            print("✓ No quantization (full precision)")
        
        # Load model
        token = os.environ.get('HUGGINGFACE_TOKEN')
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map="cpu",  # Changed to CPU
            trust_remote_code=True,
            token=token
        )
        
        # Prepare for k-bit training
        if self.config.use_4bit or self.config.use_8bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            token=token
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        print(f"✓ Model loaded: {self.model.num_parameters():,} parameters")
        
    def setup_lora(self):
        """Setup LoRA adapters"""
        print("\n" + "="*60)
        print("SETTING UP LORA")
        print("="*60)
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        trainable_percent = 100 * trainable_params / all_params
        
        print(f"✓ LoRA configured")
        print(f"  Trainable params: {trainable_params:,} ({trainable_percent:.2f}%)")
        print(f"  All params: {all_params:,}")
    
    def load_datasets(self):
        """Load and prepare datasets"""
        print("\n" + "="*60)
        print("LOADING DATASETS")
        print("="*60)
        
        data_path = Path(self.config.data_dir)
        
        # Load datasets
        dataset = load_dataset(
            'json',
            data_files={
                'train': str(data_path / 'train.jsonl'),
                'validation': str(data_path / 'val.jsonl'),
                'test': str(data_path / 'test.jsonl')
            }
        )
        
        print(f"✓ Loaded datasets:")
        print(f"  Train: {len(dataset['train'])} samples")
        print(f"  Validation: {len(dataset['validation'])} samples")
        print(f"  Test: {len(dataset['test'])} samples")
        
        # Tokenize datasets
        def format_prompt(example):
            """Format example as instruction-following prompt"""
            prompt = f"""### Instrução:
{example['instruction']}

### Resposta:
{example['output']}"""
            return {"text": prompt}
        
        def tokenize_function(examples):
            """Tokenize examples"""
            outputs = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
            )
            outputs["labels"] = outputs["input_ids"].copy()
            return outputs
        
        # Apply formatting and tokenization
        print("\nFormatting and tokenizing...")
        dataset = dataset.map(format_prompt, remove_columns=dataset['train'].column_names)
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        print("✓ Datasets tokenized")
        
        return tokenized_dataset
    
    def train(self):
        """Run training"""
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        # Load model
        self.load_model_and_tokenizer()
        
        # Setup LoRA
        self.setup_lora()
        
        # Load datasets
        tokenized_dataset = self.load_datasets()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            max_grad_norm=self.config.max_grad_norm,
            warmup_steps=self.config.warmup_steps,  # Changed from warmup_ratio
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            optim="adamw_torch",  # paged_adamw_32bit requer bitsandbytes com GPU
            report_to="wandb" if self.config.use_wandb else "none",
            run_name=self.run_name,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator,
        )
        
        # Train
        print("\nTraining...")
        trainer.train()
        
        # Save final model
        print("\nSaving final model...")
        trainer.save_model(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Save training config
        config_path = Path(self.config.output_dir) / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Model saved to: {self.config.output_dir}")
        
        if self.config.use_wandb:
            wandb.finish()

def main():
    # Configuração leve para rodar em CPU sem GPU
    # distilgpt2 tem apenas 82M parâmetros (vs 7B do Mistral/LLaMA)
    config = FineTuningConfig(
        model_name="distilgpt2",
        use_4bit=False,
        use_8bit=False,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        max_seq_length=128,
        fp16=False,
        use_wandb=False,
    )
    
    # Initialize fine-tuner
    finetuner = MedicalLLMFineTuner(config)
    
    # Run training
    finetuner.train()

if __name__ == "__main__":
    main()
