"""
Multi-Model LLM Interface for Alita Framework
Supports GPT-4o, Claude, Gemini, Mistral, Llama and other models
"""

import os
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import requests
import json
import logging

logger = logging.getLogger(__name__)

# Try to import commercial API libraries, but make them optional
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.info("OpenAI library not available. GPT models will not be available.")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.info("Anthropic library not available. Claude models will not be available.")

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    logger.info("Google AI library not available. Gemini models will not be available.")

try:
    from mistralai.client import MistralClient
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    logger.info("Mistral library not available. Mistral models will not be available.")

@dataclass
class ModelConfig:
    """Configuration for different LLM models"""
    model_name: str
    api_key: str = ""  # Made optional for open source models
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 60
    device: str = "auto"  # For local models: "cpu", "cuda", "auto"
    load_in_8bit: bool = False  # For memory optimization

class BaseLLM(ABC):
    """Base class for all LLM implementations"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_name = config.model_name
        
    @abstractmethod
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response from the model"""
        pass
    
    @abstractmethod
    def generate_sync(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Synchronous version of generate"""
        pass

class GPTClient(BaseLLM):
    """OpenAI GPT-4o client"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        self.client = openai.OpenAI(api_key=config.api_key)
        
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                timeout=self.config.timeout
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"GPT generation error: {e}")
            raise
    
    def generate_sync(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                timeout=self.config.timeout
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"GPT sync generation error: {e}")
            raise

class ClaudeClient(BaseLLM):
    """Anthropic Claude client"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic library not installed. Run: pip install anthropic")
        self.client = anthropic.Anthropic(api_key=config.api_key)
        
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude generation error: {e}")
            raise
    
    def generate_sync(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            response = self.client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude sync generation error: {e}")
            raise

class GeminiClient(BaseLLM):
    """Google Gemini client"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if not GOOGLE_AVAILABLE:
            raise ImportError("Google AI library not installed. Run: pip install google-generativeai")
        genai.configure(api_key=config.api_key)
        self.model = genai.GenerativeModel(config.model_name)
        
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
                
            response = await asyncio.to_thread(
                self.model.generate_content,
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
            )
            
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise
    
    def generate_sync(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
                
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
            )
            
            return response.text
        except Exception as e:
            logger.error(f"Gemini sync generation error: {e}")
            raise

class MistralClient(BaseLLM):
    """Mistral AI client"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if not MISTRAL_AVAILABLE:
            raise ImportError("Mistral library not installed. Run: pip install mistralai")
        self.client = MistralClient(api_key=config.api_key)
        
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = await asyncio.to_thread(
                self.client.chat,
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Mistral generation error: {e}")
            raise
    
    def generate_sync(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Mistral sync generation error: {e}")
            raise

class LlamaClient(BaseLLM):
    """Llama (via Ollama or API) client"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"
        
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
                
            payload = {
                "model": self.config.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens
                }
            }
            
            response = await asyncio.to_thread(
                requests.post,
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.config.timeout
            )
            
            return response.json()["response"]
        except Exception as e:
            logger.error(f"Llama generation error: {e}")
            raise
    
    def generate_sync(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
                
            payload = {
                "model": self.config.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.config.timeout
            )
            
            return response.json()["response"]
        except Exception as e:
            logger.error(f"Llama sync generation error: {e}")
            raise

class HuggingFaceClient(BaseLLM):
    """Hugging Face Transformers client for open source models"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
            self.tokenizer = None
            self.model = None
            self.pipeline = None
            
            # Set device
            if config.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = config.device
                
            logger.info(f"Using device: {self.device}")
            
        except ImportError:
            logger.error("Transformers library not installed. Run: pip install transformers torch")
            raise
    
    def _load_model(self):
        """Lazy load model to save memory"""
        if self.pipeline is None:
            try:
                from transformers import pipeline
                import torch
                
                # Load model with memory optimization
                model_kwargs = {
                    "device_map": "auto" if self.device == "cuda" else None,
                    "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                }
                
                if self.config.load_in_8bit and self.device == "cuda":
                    model_kwargs["load_in_8bit"] = True
                
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.config.model_name,
                    tokenizer=self.config.model_name,
                    **model_kwargs
                )
                logger.info(f"Model {self.config.model_name} loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load model {self.config.model_name}: {e}")
                raise
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            return await asyncio.to_thread(self.generate_sync, prompt, system_prompt)
        except Exception as e:
            logger.error(f"HuggingFace async generation error: {e}")
            raise
    
    def generate_sync(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            self._load_model()
            
            # Format prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
            
            # Generate response
            outputs = self.pipeline(
                full_prompt,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self.pipeline.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            return outputs[0]["generated_text"].strip()
            
        except Exception as e:
            logger.error(f"HuggingFace sync generation error: {e}")
            raise

class QwenClient(BaseLLM):
    """Qwen model client (can work with Ollama or direct API)"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"
        
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            # Format for Qwen models
            if system_prompt:
                formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                
            payload = {
                "model": self.config.model_name,
                "prompt": formatted_prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                    "stop": ["<|im_end|>"]
                }
            }
            
            response = await asyncio.to_thread(
                requests.post,
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.config.timeout
            )
            
            return response.json()["response"].strip()
        except Exception as e:
            logger.error(f"Qwen generation error: {e}")
            raise
    
    def generate_sync(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            # Format for Qwen models
            if system_prompt:
                formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                
            payload = {
                "model": self.config.model_name,
                "prompt": formatted_prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                    "stop": ["<|im_end|>"]
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.config.timeout
            )
            
            return response.json()["response"].strip()
        except Exception as e:
            logger.error(f"Qwen sync generation error: {e}")
            raise

class OpenSourceAPIClient(BaseLLM):
    """Generic client for OpenAI-compatible APIs (like vLLM, Text Generation WebUI, etc.)"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:8000"
        
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.config.model_name,
                "messages": messages,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "stream": False
            }
            
            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            response = await asyncio.to_thread(
                requests.post,
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.config.timeout
            )
            
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"OpenSource API generation error: {e}")
            raise
    
    def generate_sync(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.config.model_name,
                "messages": messages,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "stream": False
            }
            
            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.config.timeout
            )
            
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"OpenSource API sync generation error: {e}")
            raise

class ModelFactory:
    """Factory for creating different LLM clients"""
    
    _models = {
        "gpt-4o": GPTClient,
        "gpt-4": GPTClient,
        "gpt-3.5-turbo": GPTClient,
        "claude-3-opus-20240229": ClaudeClient,
        "claude-3-sonnet-20240229": ClaudeClient,
        "claude-3-haiku-20240307": ClaudeClient,
        "claude-3-5-sonnet-20241022": ClaudeClient,
        "gemini-pro": GeminiClient,
        "gemini-1.5-pro": GeminiClient,
        "mistral-large-latest": MistralClient,
        "mistral-medium-latest": MistralClient,
        "llama3.1": LlamaClient,
        "llama3.2": LlamaClient,
        "codellama": LlamaClient,
        # Open Source Models
        "qwen2.5": QwenClient,
        "qwen2.5-coder": QwenClient,
        "qwen2": QwenClient,
        "qwen-coder": QwenClient,
        "codeqwen": QwenClient,
        # Hugging Face models
        "microsoft/DialoGPT-medium": HuggingFaceClient,
        "microsoft/DialoGPT-large": HuggingFaceClient,
        "gpt2": HuggingFaceClient,
        "gpt2-medium": HuggingFaceClient,
        "gpt2-large": HuggingFaceClient,
        "gpt2-xl": HuggingFaceClient,
        "distilgpt2": HuggingFaceClient,
        "Qwen/Qwen2.5-7B-Instruct": HuggingFaceClient,
        "Qwen/Qwen2.5-14B-Instruct": HuggingFaceClient,
        "Qwen/Qwen2.5-Coder-7B-Instruct": HuggingFaceClient,
        "deepseek-ai/deepseek-coder-6.7b-instruct": HuggingFaceClient,
        "WizardLM/WizardCoder-Python-7B-V1.0": HuggingFaceClient,
        "codellama/CodeLlama-7b-Instruct-hf": HuggingFaceClient,
        "codellama/CodeLlama-13b-Instruct-hf": HuggingFaceClient,
        # OpenAI-compatible API servers
        "vllm": OpenSourceAPIClient,
        "text-generation-webui": OpenSourceAPIClient,
        "localai": OpenSourceAPIClient,
        # Generic mappings
        "huggingface": HuggingFaceClient,
        "qwen": QwenClient,
        "opensourceapi": OpenSourceAPIClient,
    }
    
    @classmethod
    def create_model(cls, config: ModelConfig) -> BaseLLM:
        """Create LLM client based on model name"""
        model_class = None
        
        # Check exact match first
        if config.model_name in cls._models:
            model_class = cls._models[config.model_name]
        else:
            # Check partial matches
            for model_key, model_cls in cls._models.items():
                if model_key in config.model_name.lower():
                    model_class = model_cls
                    break
        
        if not model_class:
            raise ValueError(f"Unsupported model: {config.model_name}")
        
        return model_class(config)
    
    @classmethod
    def get_supported_models(cls) -> List[str]:
        """Get list of supported models"""
        return list(cls._models.keys())

class MultiModelManager:
    """Manager for multiple LLM models"""
    
    def __init__(self):
        self.models: Dict[str, BaseLLM] = {}
        self.default_model: Optional[str] = None
    
    def add_model(self, name: str, config: ModelConfig):
        """Add a new model to the manager"""
        self.models[name] = ModelFactory.create_model(config)
        if self.default_model is None:
            self.default_model = name
    
    def set_default_model(self, name: str):
        """Set the default model"""
        if name not in self.models:
            raise ValueError(f"Model {name} not found")
        self.default_model = name
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                      model_name: Optional[str] = None) -> str:
        """Generate using specified model or default"""
        model_name = model_name or self.default_model
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        return await self.models[model_name].generate(prompt, system_prompt)
    
    def generate_sync(self, prompt: str, system_prompt: Optional[str] = None,
                     model_name: Optional[str] = None) -> str:
        """Synchronous generate using specified model or default"""
        model_name = model_name or self.default_model
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        return self.models[model_name].generate_sync(prompt, system_prompt)
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return list(self.models.keys())

# Convenience function to create model manager with common configurations
def create_model_manager_from_env() -> MultiModelManager:
    """Create model manager using environment variables"""
    manager = MultiModelManager()
    
    # GPT models
    if os.getenv("OPENAI_API_KEY"):
        manager.add_model("gpt-4o", ModelConfig(
            model_name="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY")
        ))
    
    # Claude models
    if os.getenv("ANTHROPIC_API_KEY"):
        manager.add_model("claude-3.5-sonnet", ModelConfig(
            model_name="claude-3-5-sonnet-20241022",
            api_key=os.getenv("ANTHROPIC_API_KEY")
        ))
    
    # Gemini models
    if os.getenv("GOOGLE_API_KEY"):
        manager.add_model("gemini-pro", ModelConfig(
            model_name="gemini-1.5-pro",
            api_key=os.getenv("GOOGLE_API_KEY")
        ))
    
    # Mistral models
    if os.getenv("MISTRAL_API_KEY"):
        manager.add_model("mistral-large", ModelConfig(
            model_name="mistral-large-latest",
            api_key=os.getenv("MISTRAL_API_KEY")
        ))
    
    # Llama models (via Ollama)
    if os.getenv("OLLAMA_BASE_URL") or True:  # Default to localhost
        manager.add_model("llama3.1", ModelConfig(
            model_name="llama3.1",
            api_key="",  # Not needed for Ollama
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ))
    
    # Qwen models (via Ollama)
    manager.add_model("qwen2.5", ModelConfig(
        model_name="qwen2.5",
        api_key="",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ))
    
    manager.add_model("qwen2.5-coder", ModelConfig(
        model_name="qwen2.5-coder",
        api_key="",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ))
    
    # Free Hugging Face models (CPU/GPU)
    device = os.getenv("HF_DEVICE", "auto")  # auto, cpu, cuda
    load_in_8bit = os.getenv("HF_LOAD_IN_8BIT", "false").lower() == "true"
    
    # GPT-2 (completely free)
    manager.add_model("gpt2", ModelConfig(
        model_name="gpt2",
        api_key="",
        device=device,
        load_in_8bit=load_in_8bit,
        max_tokens=512  # GPT-2 has smaller context
    ))
    
    manager.add_model("gpt2-large", ModelConfig(
        model_name="gpt2-large",
        api_key="",
        device=device,
        load_in_8bit=load_in_8bit,
        max_tokens=1024
    ))
    
    # Qwen Instruct models (free via HuggingFace)
    manager.add_model("qwen-7b", ModelConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        api_key="",
        device=device,
        load_in_8bit=load_in_8bit
    ))
    
    manager.add_model("qwen-coder-7b", ModelConfig(
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        api_key="",
        device=device,
        load_in_8bit=load_in_8bit
    ))
    
    # Code-specific models
    manager.add_model("codellama-7b", ModelConfig(
        model_name="codellama/CodeLlama-7b-Instruct-hf",
        api_key="",
        device=device,
        load_in_8bit=load_in_8bit
    ))
    
    # OpenAI-compatible local servers
    if os.getenv("VLLM_BASE_URL"):
        manager.add_model("vllm-local", ModelConfig(
            model_name=os.getenv("VLLM_MODEL_NAME", "local-model"),
            api_key="",
            base_url=os.getenv("VLLM_BASE_URL")
        ))
    
    if os.getenv("TEXT_GENERATION_WEBUI_URL"):
        manager.add_model("webui-local", ModelConfig(
            model_name=os.getenv("WEBUI_MODEL_NAME", "local-model"),
            api_key="",
            base_url=os.getenv("TEXT_GENERATION_WEBUI_URL")
        ))
    
    return manager 