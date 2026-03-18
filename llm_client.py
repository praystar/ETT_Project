"""
llm_client.py — Thin wrapper around the Google Gemini Chat API.

Swap `model` in config.py to use gemini-2.5-pro, gemini-1.5-flash, etc.
"""

import google.generativeai as genai
from config import settings


class LLMClient:
    """
    Handles all interactions with the LLM (Google Gemini by default).
    Easily swap for other Gemini models by changing LLM_MODEL in config.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
    ):
        self.model = model or settings.LLM_MODEL
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.client = genai.GenerativeModel(
            model_name=self.model,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
            ),
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
        # Gemini uses system instruction separately from message history
        response = self.client.generate_content(
            contents=[
                genai.types.Content(parts=[genai.types.Part(
                    text=f"{system}\n\n{prompt}"
                )]),
            ],
            stream=False,
        )
        return response.text.strip()

    def chat(self, messages: list[dict]) -> str:
        """
        Multi-turn chat with a full message history.

        Args:
            messages: List of {"role": "user"|"assistant"|"model", "content": str}

        Returns:
            The model's reply as a plain string.
        """
        # Convert OpenAI-style messages to Gemini format
        gemini_messages = []
        for msg in messages:
            role = msg["role"]
            # Gemini uses "model" instead of "assistant"
            if role == "assistant":
                role = "model"
            elif role == "system":
                # System messages should be combined with user messages in Gemini
                continue
            gemini_messages.append(
                genai.types.Content(
                    parts=[genai.types.Part(text=msg["content"])],
                    role=role,
                )
            )
        
        response = self.client.generate_content(
            contents=gemini_messages,
            stream=False,
        )
        return response.text.strip()
