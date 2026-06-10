"""LLM client for different providers."""

from abc import ABC, abstractmethod
import httpx


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Generated text response
        """
        pass


class OllamaLLMClient(LLMClient):
    """LLM client for Ollama."""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=httpx.Timeout(timeout=120.0, connect=10.0, pool=10.0))

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]


class OpenAICompatibleLLMClient(LLMClient):
    """LLM client for OpenAI-compatible APIs."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout=120.0, connect=10.0, pool=10.0),
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


class AnthropicLLMClient(LLMClient):
    """LLM client for Anthropic API."""

    def __init__(self, api_key: str, model: str):
        self.model = model
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout=120.0, connect=10.0, pool=10.0),
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt

        response = self._client.post(
            "https://api.anthropic.com/v1/messages",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data["content"][0]["text"]


class GoogleLLMClient(LLMClient):
    """LLM client for Google Gemini API."""

    def __init__(self, api_key: str, model: str):
        self.model = model
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout=120.0, connect=10.0, pool=10.0),
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
        )

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        payload: dict = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
            },
        }

        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}]
            }

        response = self._client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


def create_llm_client(config, model_override: str | None = None) -> LLMClient:
    """Create an LLM client from configuration.

    Args:
        config: Config object with LLM provider settings
        model_override: Optional model name to use instead of the provider default

    Returns:
        LLMClient instance
    """
    provider = config.agent.llm_provider
    provider_config = config.get_llm_config()
    model = model_override or provider_config.model

    if provider == "ollama":
        return OllamaLLMClient(
            base_url=provider_config.base_url,
            model=model,
        )
    elif provider == "openai-compatible":
        return OpenAICompatibleLLMClient(
            base_url=provider_config.base_url,
            api_key=provider_config.api_key,
            model=model,
        )
    elif provider == "anthropic":
        return AnthropicLLMClient(
            api_key=provider_config.api_key,
            model=model,
        )
    elif provider == "google":
        return GoogleLLMClient(
            api_key=provider_config.api_key,
            model=model,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
