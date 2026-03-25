#!/usr/bin/env python3
"""
Comprehensive example of using Hugging Face models in the EmoDebt system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'emomas'))

from emomas.llm.llm_wrapper import LLMWrapper, QwenLLMWrapper
from langchain_core.messages import HumanMessage

def demo_huggingface_usage():
    """Demonstrate different ways to use Hugging Face models"""
    
    print("🤗 HUGGING FACE INTEGRATION DEMO")
    print("=" * 60)
    
    # Method 1: Using QwenLLMWrapper (Recommended)
    print("\n📋 Method 1: Using QwenLLMWrapper")
    print("-" * 40)
    
    try:
        # This automatically maps to the correct HF model
        creditor_qwen = QwenLLMWrapper("3b", "creditor")  # Uses Qwen/Qwen2.5-3B-Instruct
        debtor_qwen = QwenLLMWrapper("7b", "debtor")      # Uses Qwen/Qwen2.5-7B-Instruct
        
        print("✅ QwenLLMWrapper models initialized successfully")
        
        # Test creditor
        creditor_prompt = "You are a creditor negotiating a $5000 debt. The debtor offers $2000. Respond professionally."
        creditor_response = creditor_qwen.invoke([HumanMessage(content=creditor_prompt)])\n        
        print(f"💼 Creditor (Qwen 3B): {creditor_response.content[:150]}...")
        
    except Exception as e:
        print(f"❌ QwenLLMWrapper error: {e}")
    
    # Method 2: Using LLMWrapper with model names
    print(f"\n📋 Method 2: Using LLMWrapper with model names")
    print("-" * 40)
    
    model_configs = [
        ("qwen-4b", "Creditor with Qwen 4B (mapped to 3B)"),
        ("offline:Qwen/Qwen2.5-7B-Instruct", "Direct HuggingFace model path"),
        ("gpt-4o-mini", "Online OpenAI model (for comparison)")
    ]
    
    for model_name, description in model_configs:
        try:
            print(f"\n🔧 Testing: {description}")
            model = LLMWrapper(model_name, "negotiator")
            
            test_prompt = "What's your opening offer for a $10,000 debt settlement?"
            response = model.invoke([HumanMessage(content=test_prompt)])
            
            print(f"✅ Response: {response.content[:100]}...")
            
        except Exception as e:
            print(f"❌ Error with {model_name}: {str(e)[:80]}...")
    
    # Method 3: Running your actual command with HF models
    print(f"\n📋 Method 3: Your Command with HuggingFace Models")
    print("-" * 40)
    
    print("🚀 To run your experiments with Hugging Face models, use:")
    print("\n# Example command with Qwen models:")
    print("python experiments/run_all_datasets.py \\")
    print("    --model_type vanilla \\")
    print("    --dataset_type debt \\") 
    print("    --model_creditor \"qwen-7b\" \\")
    print("    --model_debtor \"qwen-3b\" \\")
    print("    --scenarios 3 \\")
    print("    --iterations 2")
    
    print("\n# Alternative with direct HuggingFace paths:")
    print("python experiments/run_all_datasets.py \\")
    print("    --model_type vanilla \\")
    print("    --dataset_type debt \\")
    print("    --model_creditor \"offline:Qwen/Qwen2.5-7B-Instruct\" \\")  
    print("    --model_debtor \"offline:Qwen/Qwen2.5-3B-Instruct\" \\")
    print("    --scenarios 3 \\")
    print("    --iterations 2")
    
    # Method 4: Performance and cost considerations
    print(f"\n📋 Method 4: Performance & Cost Benefits")
    print("-" * 40)
    
    print("💰 Benefits of using Hugging Face models:")
    print("   • No API costs (free after download)")
    print("   • Complete privacy (runs locally)")
    print("   • No rate limits")
    print("   • Faster response times (no network calls)")
    print("   • Works offline")
    
    print("\n🔧 Requirements:")
    print("   • GPU recommended for larger models (7B+)")
    print("   • 8GB+ RAM for 3B models")
    print("   • 16GB+ RAM for 7B models") 
    print("   • Initial download time for model weights")
    
    print("\n📊 Model size recommendations:")
    print("   • Qwen 0.5B-1.5B: Very fast, basic performance")
    print("   • Qwen 3B: Good balance of speed/quality")
    print("   • Qwen 7B: Better quality, slower")
    print("   • Qwen 14B+: Best quality, requires more resources")

if __name__ == "__main__":
    demo_huggingface_usage()