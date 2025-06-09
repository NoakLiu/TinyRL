"""
Multi-Model LLM Interface for Alita Framework
Supports GPT-4o, Claude, Gemini, Mistral, Llama and other models
"""

import os
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import openai
import anthropic
import google.generativeai as genai
from mistralai.client import MistralClient
import requests
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for different LLM models"""
    model_name: str
    api_key: str
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 60

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
    
    return manager 