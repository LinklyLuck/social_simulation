import os
import json
import hashlib
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import random

@dataclass
class LLMResponse:
    """Standardized LLM response"""
    content: str
    model: str
    tokens_used: int = 0
    cached: bool = False

class LLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "", **kwargs) -> LLMResponse:
        pass
    
    @abstractmethod
    def batch_generate(self, prompts: List[str], system_prompt: str = "", **kwargs) -> List[LLMResponse]:
        pass

class MockLLMClient(LLMClient):
    """
    Mock LLM client for testing without API calls.
    Generates realistic-looking responses based on templates.
    """
    
    def __init__(self, model_name: str = "mock-llm"):
        self.model_name = model_name
        self.response_cache: Dict[str, str] = {}
        
        # Response templates based on context
        self.templates = {
            "support": [
                "I completely agree with this! {topic} is definitely something we should support.",
                "This makes total sense. I'm sharing this with my network.",
                "Great point! More people need to see this information about {topic}.",
                "I've been saying this for a while. Glad to see others agree.",
            ],
            "oppose": [
                "I'm skeptical about this claim regarding {topic}. Where's the evidence?",
                "This doesn't seem right. I'd want to see more sources before believing it.",
                "I disagree. The evidence doesn't support this view on {topic}.",
                "Let's not spread unverified information. This needs fact-checking.",
            ],
            "neutral": [
                "Interesting perspective on {topic}. I'd like to hear other viewpoints.",
                "This is worth discussing further. What do others think?",
                "I can see both sides of this argument about {topic}.",
                "Let me look into this more before forming an opinion.",
            ],
            "fact_check": [
                "According to reliable sources, the claim about {topic} is {verdict}.",
                "I've verified this information: {verdict}. Here's why...",
                "Fact check: The original claim appears to be {verdict}.",
            ],
            "share": [
                "Sharing this important update about {topic}.",
                "Everyone should know about this: {topic}",
                "RT: {content}",
                "This is worth spreading: {topic}",
            ],
            "question": [
                "Can someone explain more about {topic}?",
                "Has anyone verified this claim about {topic}?",
                "What's the source for this information?",
                "Is there evidence to support {topic}?",
            ],
            "collaboration": [
                "Based on what I know: {info}. Does anyone have additional details?",
                "Here's my piece of the puzzle: {info}",
                "Combining this with what {other_agent} said, I think {conclusion}.",
                "Let me add to the discussion: {info}",
            ],
        }
    
    def _get_cache_key(self, prompt: str, system_prompt: str) -> str:
        """Generate cache key for response caching"""
        combined = f"{system_prompt}|{prompt}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _extract_context(self, prompt: str) -> Dict[str, str]:
        """Extract context from prompt for template filling"""
        context = {
            "topic": "this topic",
            "content": "the shared content",
            "verdict": random.choice(["accurate", "misleading", "unverified"]),
            "info": "some relevant information",
            "conclusion": "a preliminary conclusion",
            "other_agent": "another user",
        }
        
        # Try to extract actual topic from prompt
        if "about" in prompt.lower():
            parts = prompt.lower().split("about")
            if len(parts) > 1:
                context["topic"] = parts[1][:50].strip()
        
        return context
    
    def _determine_response_type(self, prompt: str, system_prompt: str) -> str:
        """Determine appropriate response type based on context"""
        prompt_lower = prompt.lower()
        system_lower = system_prompt.lower()
        
        if "fact" in system_lower or "verify" in prompt_lower:
            return "fact_check"
        elif "support" in system_lower or "agree" in system_lower:
            return "support"
        elif "oppose" in system_lower or "skeptic" in system_lower:
            return "oppose"
        elif "share" in prompt_lower or "spread" in prompt_lower:
            return "share"
        elif "?" in prompt or "what" in prompt_lower:
            return "question"
        elif "collaborat" in system_lower or "together" in prompt_lower:
            return "collaboration"
        else:
            return "neutral"
    
    def generate(self, prompt: str, system_prompt: str = "", **kwargs) -> LLMResponse:
        """Generate a mock response"""
        cache_key = self._get_cache_key(prompt, system_prompt)
        
        if cache_key in self.response_cache:
            return LLMResponse(
                content=self.response_cache[cache_key],
                model=self.model_name,
                cached=True
            )
        
        response_type = self._determine_response_type(prompt, system_prompt)
        context = self._extract_context(prompt)
        
        templates = self.templates.get(response_type, self.templates["neutral"])
        template = random.choice(templates)
        
        try:
            response = template.format(**context)
        except KeyError:
            response = template
        
        self.response_cache[cache_key] = response
        
        return LLMResponse(
            content=response,
            model=self.model_name,
            tokens_used=len(response.split()) * 2
        )
    
    def batch_generate(self, prompts: List[str], system_prompt: str = "", **kwargs) -> List[LLMResponse]:
        """Generate multiple responses"""
        return [self.generate(p, system_prompt, **kwargs) for p in prompts]


class OpenAIClient(LLMClient):
    """
    OpenAI API client for GPT models.
    Also supports OpenAI-compatible APIs (e.g., DeepSeek, local models).
    """
    
    def __init__(
        self, 
        model: str = "gpt-4", 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.client = None
        
        if self.api_key:
            try:
                from openai import OpenAI
                client_kwargs = {"api_key": self.api_key}
                if self.base_url:
                    client_kwargs["base_url"] = self.base_url
                self.client = OpenAI(**client_kwargs)
            except ImportError:
                print("OpenAI package not installed. Using mock client.")
    
    def generate(self, prompt: str, system_prompt: str = "", **kwargs) -> LLMResponse:
        if not self.client:
            # Fallback to mock
            mock = MockLLMClient(self.model)
            return mock.generate(prompt, system_prompt, **kwargs)
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 150)
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=self.model,
                tokens_used=response.usage.total_tokens if response.usage else 0
            )
        except Exception as e:
            print(f"OpenAI API error: {e}")
            mock = MockLLMClient(self.model)
            return mock.generate(prompt, system_prompt, **kwargs)
    
    def batch_generate(self, prompts: List[str], system_prompt: str = "", **kwargs) -> List[LLMResponse]:
        return [self.generate(p, system_prompt, **kwargs) for p in prompts]


class DeepSeekClient(LLMClient):
    """
    DeepSeek API client - uses OpenAI-compatible interface.
    """
    
    def __init__(
        self, 
        model: str = "deepseek-chat", 
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com/v1"
    ):
        self.model = model
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url
        self.client = None
        
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
            except ImportError:
                print("OpenAI package not installed. Using mock client.")
    
    def generate(self, prompt: str, system_prompt: str = "", **kwargs) -> LLMResponse:
        if not self.client:
            mock = MockLLMClient(self.model)
            return mock.generate(prompt, system_prompt, **kwargs)
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 150)
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=self.model,
                tokens_used=response.usage.total_tokens if response.usage else 0
            )
        except Exception as e:
            print(f"DeepSeek API error: {e}")
            mock = MockLLMClient(self.model)
            return mock.generate(prompt, system_prompt, **kwargs)
    
    def batch_generate(self, prompts: List[str], system_prompt: str = "", **kwargs) -> List[LLMResponse]:
        return [self.generate(p, system_prompt, **kwargs) for p in prompts]


class AnthropicClient(LLMClient):
    """
    Anthropic API client for Claude models.
    Requires ANTHROPIC_API_KEY environment variable.
    """
    
    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = None
        
        if self.api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                print("Anthropic package not installed. Using mock client.")
    
    def generate(self, prompt: str, system_prompt: str = "", **kwargs) -> LLMResponse:
        if not self.client:
            mock = MockLLMClient(self.model)
            return mock.generate(prompt, system_prompt, **kwargs)
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", 150),
                system=system_prompt if system_prompt else None,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return LLMResponse(
                content=response.content[0].text,
                model=self.model,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens
            )
        except Exception as e:
            print(f"Anthropic API error: {e}")
            mock = MockLLMClient(self.model)
            return mock.generate(prompt, system_prompt, **kwargs)
    
    def batch_generate(self, prompts: List[str], system_prompt: str = "", **kwargs) -> List[LLMResponse]:
        return [self.generate(p, system_prompt, **kwargs) for p in prompts]


class LLMService:
    """
    Unified LLM Service that manages different LLM clients.
    Provides caching, rate limiting, and fallback mechanisms.
    """
    
    def __init__(
        self, 
        preferred_client: str = "mock",
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.clients: Dict[str, LLMClient] = {}
        self.preferred_client = preferred_client
        self._preferred_client = preferred_client  # For checking in agent
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.model = model
        self._init_clients()
    
    def _init_clients(self):
        """Initialize available LLM clients"""
        # Always have mock available
        self.clients["mock"] = MockLLMClient()
        
        # Initialize based on preferred client with provided credentials
        if self.preferred_client == "openai" and self.api_key:
            self.clients["openai"] = OpenAIClient(
                model=self.model or "gpt-4o-mini",
                api_key=self.api_key,
                base_url=self.api_base_url
            )
        elif self.preferred_client == "deepseek" and self.api_key:
            self.clients["deepseek"] = DeepSeekClient(
                model=self.model or "deepseek-chat",
                api_key=self.api_key,
                base_url=self.api_base_url or "https://api.deepseek.com/v1"
            )
        elif self.preferred_client == "anthropic" and self.api_key:
            self.clients["anthropic"] = AnthropicClient(
                model=self.model or "claude-3-5-sonnet-20241022",
                api_key=self.api_key
            )
        
        # Also check environment variables for fallback
        if "openai" not in self.clients and os.getenv("OPENAI_API_KEY"):
            self.clients["openai"] = OpenAIClient()
        
        if "anthropic" not in self.clients and os.getenv("ANTHROPIC_API_KEY"):
            self.clients["anthropic"] = AnthropicClient()
        
        if "deepseek" not in self.clients and os.getenv("DEEPSEEK_API_KEY"):
            self.clients["deepseek"] = DeepSeekClient()
    
    def get_client(self, client_name: Optional[str] = None) -> LLMClient:
        """Get specified client or preferred client"""
        name = client_name or self.preferred_client
        if name in self.clients:
            return self.clients[name]
        return self.clients["mock"]
    
    def generate_response(
        self,
        prompt: str,
        system_prompt: str = "",
        client_name: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using specified or preferred client"""
        client = self.get_client(client_name)
        return client.generate(prompt, system_prompt, **kwargs)
    
    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: str = "",
        client_name: Optional[str] = None,
        **kwargs
    ) -> List[LLMResponse]:
        """Generate multiple responses"""
        client = self.get_client(client_name)
        return client.batch_generate(prompts, system_prompt, **kwargs)
    
    def available_clients(self) -> List[str]:
        """Return list of available client names"""
        return list(self.clients.keys())
