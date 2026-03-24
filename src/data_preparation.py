"""
Data Preparation Pipeline for Medical Assistant
Handles PubMedQA and MedQuAD datasets with anonymization
"""

import os
import json
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import re
import hashlib
from datetime import datetime

class MedicalDataPreparator:
    """Prepares medical datasets for fine-tuning"""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        print(f" Data directory: {self.data_dir}")
        
    def download_pubmedqa(self) -> Dict:
        """Download PubMedQA dataset"""
        print("\n" + "="*60)
        print("DOWNLOADING PUBMEDQA DATASET")
        print("="*60)
        
        urls = {
            'train_labeled': 'https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json',
            'dev_unlabeled': 'https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqau.json',
        }
        
        data = {}
        for split, url in urls.items():
            try:
                filepath = self.raw_dir / f"pubmedqa_{split}.json"
                
                if filepath.exists():
                    print(f"✓ {split} already exists, loading from disk...")
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data[split] = json.load(f)
                else:
                    print(f"⬇ Downloading {split}...")
                    response = requests.get(url, timeout=60)
                    response.raise_for_status()
                    data[split] = response.json()
                    
                    # Save to disk
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(data[split], f, indent=2)
                    
                print(f"✓ {split}: {len(data[split])} samples")
                
            except Exception as e:
                print(f"❌ Error downloading {split}: {e}")
                data[split] = {}
                
        return data
    
    def anonymize_text(self, text: str) -> str:
        """Anonymize sensitive information in medical text"""
        # Replace potential patient names (capitalized words that could be names)
        # This is a simple approach - in production, use NER models
        text = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[PATIENT]', text)
        
        # Replace dates
        text = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '[DATE]', text)
        text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '[DATE]', text)
        
        # Replace phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        
        # Replace medical record numbers
        text = re.sub(r'\bMRN:?\s*\d+\b', '[MRN]', text, flags=re.IGNORECASE)
        
        return text
    
    def format_for_instruction_tuning(self, data: Dict, source: str = "pubmedqa") -> List[Dict]:
        """Format data into instruction-following format for fine-tuning"""
        print(f"\n Formatting {source.upper()} for instruction tuning...")
        
        formatted_data = []
        
        if source == "pubmedqa":
            for split_name, split_data in data.items():
                for qid, item in tqdm(split_data.items(), desc=f"Processing {split_name}"):
                    # Extract components
                    question = item.get('QUESTION', '')
                    contexts = item.get('CONTEXTS', [])
                    long_answer = item.get('LONG_ANSWER', '')
                    final_decision = item.get('final_decision', '')
                    
                    if not question or not contexts:
                        continue
                    
                    # Anonymize
                    question = self.anonymize_text(question)
                    contexts = [self.anonymize_text(ctx) for ctx in contexts]
                    long_answer = self.anonymize_text(long_answer) if long_answer else ''
                    
                    # Create context
                    context = ' '.join(contexts[:3])  # Use first 3 contexts to avoid token limit
                    
                    # Format as instruction
                    instruction = (
                        f"Você é um assistente médico especializado. "
                        f"Com base no contexto médico fornecido, responda à seguinte pergunta clínica.\n\n"
                        f"Contexto: {context}\n\n"
                        f"Pergunta: {question}"
                    )
                    
                    # Create response
                    if long_answer and final_decision:
                        response = f"Resposta: {final_decision}\n\nExplicação: {long_answer}"
                    elif long_answer:
                        response = long_answer
                    elif final_decision:
                        response = final_decision
                    else:
                        continue
                    
                    formatted_data.append({
                        'id': f"{source}_{qid}",
                        'instruction': instruction,
                        'input': '',  # Keep empty for instruction-only format
                        'output': response,
                        'source': source,
                        'split': split_name,
                        'created_at': datetime.now().isoformat()
                    })
        
        print(f"✓ Formatted {len(formatted_data)} examples")
        return formatted_data
    
    def create_splits(self, data: List[Dict], 
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split data into train/val/test sets"""
        print("\n Creating train/val/test splits...")
        
        import random
        random.seed(42)
        random.shuffle(data)
        
        n_total = len(data)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train = data[:n_train]
        val = data[n_train:n_train + n_val]
        test = data[n_train + n_val:]
        
        print(f"✓ Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
        return train, val, test
    
    def save_datasets(self, train: List[Dict], val: List[Dict], test: List[Dict]):
        """Save datasets in JSON and JSONL formats for different tools"""
        print("\n Saving datasets...")
        
        # Save as JSON (for inspection)
        splits = {'train': train, 'val': val, 'test': test}
        
        for split_name, split_data in splits.items():
            # JSON format
            json_path = self.processed_dir / f"{split_name}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)
            
            # JSONL format (for HuggingFace)
            jsonl_path = self.processed_dir / f"{split_name}.jsonl"
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for item in split_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            print(f"✓ Saved {split_name}: {len(split_data)} examples")
            print(f"  - {json_path}")
            print(f"  - {jsonl_path}")
        
        # Save dataset statistics
        self.save_statistics(train, val, test)
    
    def save_statistics(self, train: List[Dict], val: List[Dict], test: List[Dict]):
        """Generate and save dataset statistics"""
        print("\n Generating statistics...")
        
        stats = {
            'total_samples': len(train) + len(val) + len(test),
            'train_samples': len(train),
            'val_samples': len(val),
            'test_samples': len(test),
            'sources': {},
            'avg_instruction_length': 0,
            'avg_output_length': 0,
            'created_at': datetime.now().isoformat()
        }
        
        all_data = train + val + test
        
        # Count sources
        for item in all_data:
            source = item.get('source', 'unknown')
            stats['sources'][source] = stats['sources'].get(source, 0) + 1
        
        # Calculate average lengths
        inst_lengths = [len(item['instruction'].split()) for item in all_data]
        out_lengths = [len(item['output'].split()) for item in all_data]
        
        stats['avg_instruction_length'] = sum(inst_lengths) / len(inst_lengths)
        stats['avg_output_length'] = sum(out_lengths) / len(out_lengths)
        stats['max_instruction_length'] = max(inst_lengths)
        stats['max_output_length'] = max(out_lengths)
        
        # Save statistics
        stats_path = self.processed_dir / 'dataset_statistics.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✓ Statistics saved to {stats_path}")
        print(f"\n Dataset Summary:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Avg instruction length: {stats['avg_instruction_length']:.1f} words")
        print(f"  Avg output length: {stats['avg_output_length']:.1f} words")
    
    def prepare_all(self):
        """Main pipeline - download, process, and save all datasets"""
        print("\n" + "="*60)
        print(" STARTING MEDICAL DATA PREPARATION PIPELINE")
        print("="*60)
        
        # Download PubMedQA
        pubmedqa_data = self.download_pubmedqa()
        
        # Format for training
        formatted_data = self.format_for_instruction_tuning(pubmedqa_data, source="pubmedqa")
        
        if not formatted_data:
            print(" No data formatted. Exiting.")
            return
        
        # Create splits
        train, val, test = self.create_splits(formatted_data)
        
        # Save datasets
        self.save_datasets(train, val, test)
        
        print("\n" + "="*60)
        print(" DATA PREPARATION COMPLETE!")
        print("="*60)
        print(f" Processed data saved in: {self.processed_dir}")
        print("\nNext steps:")
        print("1. Review the data in data/processed/")
        print("2. Run the fine-tuning script: python src/finetune_model.py")

def main():
    preparator = MedicalDataPreparator(data_dir="./data")
    preparator.prepare_all()

if __name__ == "__main__":
    main()
