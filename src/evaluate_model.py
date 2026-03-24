"""
Model Evaluation Pipeline
Comprehensive evaluation of fine-tuned medical LLM
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ModelEvaluator:
    """Evaluate fine-tuned medical LLM"""
    
    def __init__(self, model_path: str, test_data_path: str = "./data/processed/test.jsonl"):
        self.model_path = Path(model_path)
        self.test_data_path = Path(test_data_path)
        
        print(f"📊 Initializing evaluator...")
        print(f"   Model: {self.model_path}")
        print(f"   Test data: {self.test_data_path}")
        
        self.model = None
        self.tokenizer = None
        self.test_data = None
        
        self._load_model()
        self._load_test_data()
    
    def _load_model(self):
        """Load model and tokenizer"""
        print("\n📦 Loading model...")
        
        model_path = str(self.model_path)
        
        # Check if local fine-tuned model exists
        try:
            from pathlib import Path
            if Path(model_path).exists() and any(Path(model_path).iterdir()):
                print(f"  Loading fine-tuned model from: {model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, token=os.environ.get('HUGGINGFACE_TOKEN'))
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
            else:
                raise FileNotFoundError(f"Model not found in {model_path}, using base model instead")
        except (FileNotFoundError, ValueError, Exception) as e:
            print(f"  ⚠️  Could not load fine-tuned model: {e}")
            print("  Loading base Mistral-7B model instead...")
            model_name = os.environ.get('MODEL_NAME', 'mistralai/Mistral-7B-v0.1')
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ.get('HUGGINGFACE_TOKEN'))
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        
        self.model.eval()
        print("✓ Model loaded")
    
    def _load_test_data(self):
        """Load test dataset"""
        print("\n📚 Loading test data...")
        
        self.test_data = load_dataset(
            'json',
            data_files={'test': str(self.test_data_path)}
        )['test']
        
        print(f"✓ Loaded {len(self.test_data)} test examples")
    
    def calculate_perplexity(self, num_samples: int = 100) -> float:
        """Calculate perplexity on test set"""
        print("\n🔢 Calculating perplexity...")
        
        total_loss = 0
        num_tokens = 0
        
        samples = self.test_data.select(range(min(num_samples, len(self.test_data))))
        
        for example in tqdm(samples, desc="Computing perplexity"):
            text = f"{example['instruction']}\n\n{example['output']}"
            
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
            
            total_loss += loss.item() * inputs["input_ids"].size(1)
            num_tokens += inputs["input_ids"].size(1)
        
        avg_loss = total_loss / num_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        print(f"✓ Perplexity: {perplexity:.2f}")
        return perplexity
    
    def calculate_bleu_rouge(self, num_samples: int = 100) -> Dict[str, float]:
        """Calculate BLEU and ROUGE scores"""
        print("\n📝 Calculating BLEU and ROUGE scores...")
        
        rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        smoothing = SmoothingFunction().method1
        
        bleu_scores = []
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        samples = self.test_data.select(range(min(num_samples, len(self.test_data))))
        
        for example in tqdm(samples, desc="Computing BLEU/ROUGE"):
            # Generate prediction
            prompt = f"### Instrução:\n{example['instruction']}\n\n### Resposta:\n"
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = generated_text.split("### Resposta:")[-1].strip()
            
            reference = example['output']
            
            # BLEU
            reference_tokens = nltk.word_tokenize(reference.lower())
            generated_tokens = nltk.word_tokenize(generated_text.lower())
            bleu = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=smoothing)
            bleu_scores.append(bleu)
            
            # ROUGE
            rouge_scores = rouge_scorer_obj.score(reference, generated_text)
            rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
            rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
            rougeL_scores.append(rouge_scores['rougeL'].fmeasure)
        
        results = {
            'bleu': np.mean(bleu_scores),
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores)
        }
        
        print(f"✓ BLEU: {results['bleu']:.4f}")
        print(f"✓ ROUGE-1: {results['rouge1']:.4f}")
        print(f"✓ ROUGE-2: {results['rouge2']:.4f}")
        print(f"✓ ROUGE-L: {results['rougeL']:.4f}")
        
        return results
    
    def qualitative_evaluation(self, num_samples: int = 5) -> List[Dict]:
        """Generate qualitative examples"""
        print("\n🔍 Generating qualitative examples...")
        
        examples = []
        samples = self.test_data.select(range(min(num_samples, len(self.test_data))))
        
        for i, example in enumerate(samples, 1):
            print(f"\n{'='*60}")
            print(f"Example {i}/{num_samples}")
            print(f"{'='*60}")
            
            prompt = f"### Instrução:\n{example['instruction']}\n\n### Resposta:\n"
            
            print(f"\n❓ Input:\n{example['instruction'][:200]}...")
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = generated_text.split("### Resposta:")[-1].strip()
            
            print(f"\n✅ Generated:\n{generated_text[:300]}...")
            print(f"\n📌 Reference:\n{example['output'][:300]}...")
            
            examples.append({
                'instruction': example['instruction'],
                'generated': generated_text,
                'reference': example['output']
            })
        
        return examples
    
    def evaluate_all(self, save_results: bool = True) -> Dict:
        """Run complete evaluation"""
        print("\n" + "="*60)
        print("🚀 STARTING COMPREHENSIVE EVALUATION")
        print("="*60)
        
        results = {
            'model_path': str(self.model_path),
            'test_samples': len(self.test_data),
            'metrics': {}
        }
        
        # Perplexity
        results['metrics']['perplexity'] = self.calculate_perplexity(num_samples=100)
        
        # BLEU/ROUGE
        bleu_rouge = self.calculate_bleu_rouge(num_samples=100)
        results['metrics'].update(bleu_rouge)
        
        # Qualitative
        results['qualitative_examples'] = self.qualitative_evaluation(num_samples=5)
        
        # Summary
        print("\n" + "="*60)
        print("📊 EVALUATION SUMMARY")
        print("="*60)
        print(f"Perplexity: {results['metrics']['perplexity']:.2f}")
        print(f"BLEU: {results['metrics']['bleu']:.4f}")
        print(f"ROUGE-L: {results['metrics']['rougeL']:.4f}")
        
        # Save results
        if save_results:
            output_path = self.model_path / "evaluation_results.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Results saved to: {output_path}")
        
        return results

def main():
    """Run evaluation"""
    evaluator = ModelEvaluator(
        model_path="./models/finetuned",
        test_data_path="./data/processed/test.jsonl"
    )
    
    results = evaluator.evaluate_all(save_results=True)

if __name__ == "__main__":
    main()
