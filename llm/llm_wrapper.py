"""
LLM wrapper for online/offline models
"""

import os
from typing import Optional, Union, List
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

# Try to import offline models
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available. Install with: pip install transformers torch")

class LLMWrapper:
    """Wrapper for LLM models (online and offline)"""
    
    def __init__(self, model_name: str, role: str = "generic"):
        self.model_name = model_name
        self.role = role
        self.model = self._initialize_model(model_name)
    
    def _initialize_model(self, model_name: str):
        """Initialize LLM model"""
        model_lower = model_name.lower()
        
        # Check for offline models
        if "deepseek" in model_lower or "llama" in model_lower or "offline:" in model_lower:
            if not TRANSFORMERS_AVAILABLE:
                print(f"⚠️ Transformers not available for {self.role}. Falling back to GPT-4o-mini")
                return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
            
            # Remove "offline:" prefix
            actual_model = model_name.replace("offline:", "")
            
            # Handle common model mappings
            if actual_model.lower() == "deepseek-7b" or actual_model.lower() == "deepseek":
                actual_model = "deepseek-ai/DeepSeek-LLM-7B-Chat"
            elif actual_model.lower() == "llama-7b" or actual_model.lower() == "llama":
                actual_model = "meta-llama/Llama-2-7b-chat-hf"
            
            try:
                print(f"🔄 Initializing offline model for {self.role}: {actual_model}")
                return self._initialize_offline_model(actual_model)
            except Exception as e:
                print(f"❌ Failed to load offline model for {self.role}: {e}")
                return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        
        # Online models
        elif "gpt" in model_lower:
            temp = 1.0 if "gpt-5" in model_lower else 0.7
            return ChatOpenAI(model=model_name, temperature=temp)
        elif "claude" in model_lower:
            return ChatAnthropic(model=model_name, temperature=0.7)
        else:
            # Default fallback
            return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    def _initialize_offline_model(self, model_name: str):
        """Initialize offline model using transformers"""
        class OfflineLLM:
            def __init__(self, model_name):
                self.model_name = model_name
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                
                # Determine device
                if torch.cuda.is_available():
                    device_map = "cuda"
                    torch_dtype = torch.float16
                else:
                    device_map = "cpu"
                    torch_dtype = torch.float32
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    trust_remote_code=True
                )
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            
            def invoke(self, messages, temperature=0.7, max_new_tokens=512):
                # Extract prompt
                if isinstance(messages, list) and len(messages) > 0:
                    if hasattr(messages[0], 'content'):
                        prompt = messages[0].content
                    else:
                        prompt = str(messages[0])
                else:
                    prompt = str(messages)
                
                # Format for chat
                formatted_prompt = f"User: {prompt}\n\nAssistant:"
                
                # Tokenize
                inputs = self.tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                )
                
                # Move to device
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=temperature > 0,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode
                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                # Return compatible format
                class MockAIMessage:
                    def __init__(self, content):
                        self.content = content
                
                return MockAIMessage(response)
        
        return OfflineLLM(model_name)
    
    def invoke(self, messages, temperature: float = 0.7, **kwargs):
        """Invoke the LLM model"""
        return self.model.invoke(messages, temperature=temperature, **kwargs)