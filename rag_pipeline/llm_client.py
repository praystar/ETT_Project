from groq import Groq
import google.genai as genai
from .config import settings


class LLMClient:
    def __init__(self, provider=None, model=None, temperature=None):
        self.provider = provider or settings.LLM_PROVIDER.lower()
        self.model = model or settings.LLM_MODEL
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE

        if self.provider == "groq":
            if not settings.GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY is not set in .env file")
            self.client = Groq(api_key=settings.GROQ_API_KEY)
            print(f"🤖 LLMClient ready — provider: GROQ, model: '{self.model}', temperature: {self.temperature}")

        elif self.provider == "gemini":
            if not settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is not set in .env file")
            self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
            print(f"🤖 LLMClient ready — provider: Gemini, model: '{self.model}', temperature: {self.temperature}")

        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}. Use 'groq' or 'gemini'")

    def complete(self, prompt: str, system: str = "You are a helpful assistant.") -> str:
        try:
            if self.provider == "groq":
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    model=self.model,
                    temperature=self.temperature,
                )
                return chat_completion.choices[0].message.content.strip()

            elif self.provider == "gemini":
                full_prompt = f"{system}\n\n{prompt}"
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=full_prompt,
                )
                return response.text.strip()

        except Exception as e:
            return f"LLM Error: {str(e)}"