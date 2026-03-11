"""
llm_client.py — Thin wrapper around the OpenAI Chat Completions API.

Swap `model` in config.py to use GPT-4o, GPT-4-turbo, etc.
"""

from openai import OpenAI
from config import settings


class LLMClient:
    """
    Handles all interactions with the LLM (OpenAI by default).
    Easily swap for Anthropic, Cohere, or a local Ollama endpoint
    by changing the base_url and api_key in config.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
    ):
        self.model = model or settings.LLM_MODEL
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL or None,
        )
        print(f"🤖 LLMClient ready — model: '{self.model}', temperature: {self.temperature}")

    def complete(self, prompt: str, system: str = "You are a helpful assistant.") -> str:
        """
        Send a single-turn completion request and return the response text.

        Args:
            prompt: The user message / full RAG prompt.
            system: Optional system message to set the assistant's behaviour.

        Returns:
            The model's response as a plain string.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    def chat(self, messages: list[dict]) -> str:
        """
        Multi-turn chat with a full message history.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": str}

        Returns:
            The model's reply as a plain string.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
        )
        return response.choices[0].message.content.strip()
