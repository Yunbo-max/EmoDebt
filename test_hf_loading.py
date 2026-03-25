#!/usr/bin/env python3
"""
Test script for Hugging Face model loading
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'emomas'))

from emomas.llm.llm_wrapper import LLMWrapper, QwenLLMWrapper
from langchain_core.messages import HumanMessage

def test_huggingface_models():
    """Test different Hugging Face model configurations"""
    
    print("🧪 Testing Hugging Face Model Loading")
    print("=" * 50)
    
    # Test models to try (in order of preference)
    test_models = [
        ("qwen-4b", "Qwen 4B model"),
        ("qwen-3b", "Qwen 3B model"), 
        ("qwen-7b", "Qwen 7B model"),
        ("gpt-4o-mini", "OpenAI GPT-4o-mini (fallback)")
    ]
    
    successful_models = []
    failed_models = []
    
    for model_name, description in test_models:
        print(f"\n🔍 Testing {description} ({model_name})...")
        print("-" * 30)
        
        try:
            # Initialize model
            if model_name.startswith("qwen"):
                # Extract size
                size = model_name.split("-")[1] if "-" in model_name else "7b"
                model = QwenLLMWrapper(size, "test_negotiator")
            else:
                model = LLMWrapper(model_name, "test_negotiator")
            
            print(f"✅ Model initialized successfully")
            
            # Test with a simple prompt
            test_prompt = "Hello! You are a debt negotiator. Please introduce yourself briefly."
            response = model.invoke([HumanMessage(content=test_prompt)])
            
            print(f"✅ Model response received")
            print(f"📝 Response preview: {response.content[:100]}...")
            
            successful_models.append((model_name, description))
            
            # If this is an offline model, we've confirmed HF loading works
            if "qwen" in model_name.lower():
                print(f"🎉 Hugging Face model loading successful!")
                break
                
        except Exception as e:
            print(f"❌ Error with {model_name}: {str(e)[:100]}...")
            failed_models.append((model_name, description, str(e)))
            continue
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 TESTING SUMMARY")
    print(f"{'='*50}")
    
    if successful_models:
        print("✅ Successfully loaded models:")
        for model_name, desc in successful_models:
            print(f"   • {desc} ({model_name})")
    
    if failed_models:
        print("\n❌ Failed to load models:")
        for model_name, desc, error in failed_models:
            print(f"   • {desc} ({model_name})")
            print(f"     Error: {error[:80]}...")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if any("qwen" in model[0] for model in successful_models):
        print("   🎉 Hugging Face integration is working correctly!")
        print("   🚀 You can now use offline models for your negotiations")
        print("   💰 This will save on API costs and provide more control")
    else:
        print("   ⚠️  Offline models not working, check:")
        print("   📦 pip install transformers torch accelerate")
        print("   🔧 GPU/CUDA setup if you want GPU acceleration")
        print("   🌐 Internet connection for model downloads")

if __name__ == "__main__":
    test_huggingface_models()