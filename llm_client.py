from groq import Groq
from config import settings


class LLMClient:
    def __init__(self, model=None, temperature=None):
        self.model = model or settings.LLM_MODEL
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE

        self.client = Groq(api_key=settings.GROQ_API_KEY)

        print(f"🤖 LLMClient ready — model: '{self.model}', temperature: {self.temperature}")

    def complete(self, prompt: str, system: str = "You are a helpful assistant.") -> str:
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=self.temperature,
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            return f"LLM Error: {str(e)}"