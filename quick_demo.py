"""
Quick Demo Script - Medical Assistant
Test the assistant with sample queries
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from medical_assistant import MedicalAssistant, AssistantConfig

def print_separator():
    print("\n" + "="*70 + "\n")

def demo_basic_query():
    """Demo 1: Basic medical query"""
    print(" DEMO 1: Pergunta Médica Básica")
    print_separator()
    
    # Note: This demo assumes model is already trained
    # If not trained yet, it will show an error
    try:
        config = AssistantConfig(
            model_path="./models/finetuned",
            require_human_validation=True
        )
        
        assistant = MedicalAssistant(config)
        
        query = "Quais são os sintomas comuns de hipertensão arterial?"
        
        print(f" Pergunta: {query}")
        print()
        
        result = assistant.ask(query, use_rag=False)
        
        print(f" Resposta:\n{result['response']}")
        print()
        print(f" Status de Segurança: {'✅ Aprovado' if result['safety_check']['is_safe'] else '⚠️ Requer Validação'}")
        
        if result['safety_check']['warnings']:
            print(f" Avisos: {result['safety_check']['warnings']}")
        
        print_separator()
        
    except FileNotFoundError:
        print(" Erro: Modelo não encontrado!")
        print("   Execute primeiro: python src/finetune_model.py")
        print_separator()
    except Exception as e:
        print(f" Erro: {e}")
        print_separator()

def demo_with_rag():
    """Demo 2: Query with RAG"""
    print("🔹 DEMO 2: Consulta com RAG (Retrieval-Augmented Generation)")
    print_separator()
    
    try:
        config = AssistantConfig(
            model_path="./models/finetuned",
            require_human_validation=True
        )
        
        assistant = MedicalAssistant(config)
        
        query = "Qual o protocolo para tratamento de diabetes tipo 2?"
        
        print(f" Pergunta: {query}")
        print()
        
        result = assistant.ask(query, use_rag=True)
        
        print(f" Resposta:\n{result['response']}")
        print()
        
        if result['sources']:
            print(f"{len(result['sources'])} fonte(s) consultada(s)")
        else:
            print(" Vector store não configurado, usando apenas LLM")
        
        print_separator()
        
    except Exception as e:
        print(f"Erro: {e}")
        print_separator()

def demo_safety_validation():
    """Demo 3: Safety validation"""
    print("🔹 DEMO 3: Validação de Segurança")
    print_separator()
    
    print("Este demo testa como o sistema lida com perguntas que requerem")
    print("prescrição ou diagnóstico definitivo...")
    print()
    
    try:
        config = AssistantConfig(
            model_path="./models/finetuned",
            require_human_validation=True
        )
        
        assistant = MedicalAssistant(config)
        
        # Query that might trigger safety warnings
        query = "Me prescreva um medicamento para dor de cabeça"
        
        print(f"Pergunta: {query}")
        print()
        
        result = assistant.ask(query, use_rag=False)
        
        print(f"Resposta:\n{result['response']}")
        print()
        print(f"Status de Segurança:")
        print(f"   - Seguro: {'Sim' if result['safety_check']['is_safe'] else 'Não'}")
        print(f"   - Violações: {result['safety_check']['violations'] or 'Nenhuma'}")
        print(f"   - Avisos: {result['safety_check']['warnings'] or 'Nenhum'}")
        print(f"   - Requer Validação: {'Sim' if result['safety_check']['requires_validation'] else 'Não'}")
        
        print_separator()
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        print_separator()

def demo_logging():
    """Demo 4: Audit logging"""
    print("🔹 DEMO 4: Sistema de Logging e Auditoria")
    print_separator()
    
    print("Após executar os demos acima, verifique os logs em:")
    print("./logs/audit_YYYYMMDD.log")
    print("./logs/interactions_YYYYMMDD.json")
    print()
    print("Cada interação é registrada com:")
    print("   • Timestamp")
    print("   • Query completa")
    print("   • Resposta gerada")
    print("   • Resultado da validação de segurança")
    print("   • Fontes consultadas (RAG)")
    print("   • Metadata adicional")
    
    print_separator()

def print_menu():
    """Print demo menu"""
    print("\n" + "="*70)
    print(" MEDICAL ASSISTANT - DEMO RÁPIDO")
    print("="*70)
    print()
    print("Escolha um demo:")
    print()
    print("  1. Pergunta médica básica")
    print("  2. Consulta com RAG (Retrieval-Augmented Generation)")
    print("  3. Validação de segurança")
    print("  4. Sistema de logging")
    print("  5. Executar todos os demos")
    print("  0. Sair")
    print()

def main():
    """Main demo loop"""
    
    print("\nBem-vindo ao Demo do Assistente Médico Virtual!")
    print()
    print("IMPORTANTE:")
    print("    Este demo requer que o modelo já tenha sido treinado.")
    print("    Se ainda não treinou, execute primeiro:")
    print("    $ python src/data_preparation.py")
    print("    $ python src/finetune_model.py")
    print()
    input("Pressione ENTER para continuar...")
    
    while True:
        print_menu()
        
        try:
            choice = input("Digite sua escolha (0-5): ").strip()
            
            if choice == "0":
                print("\n👋 Até logo!")
                break
            elif choice == "1":
                demo_basic_query()
            elif choice == "2":
                demo_with_rag()
            elif choice == "3":
                demo_safety_validation()
            elif choice == "4":
                demo_logging()
            elif choice == "5":
                print("\nExecutando todos os demos...")
                demo_basic_query()
                demo_with_rag()
                demo_safety_validation()
                demo_logging()
                print("\nTodos os demos concluídos!")
            else:
                print("\nOpção inválida. Tente novamente.")
            
            input("\nPressione ENTER para continuar...")
            
        except KeyboardInterrupt:
            print("\n\nInterrompido pelo usuário. Até logo!")
            break
        except Exception as e:
            print(f"\nErro inesperado: {e}")
            input("\nPressione ENTER para continuar...")

if __name__ == "__main__":
    main()
