"""LLM client for interacting with OpenRouter API."""

import asyncio
from typing import Any

from openai import AsyncOpenAI

from config import Config, load_config


class LLMClientError(Exception):
    """Raised when LLM API calls fail."""
    pass


class LLMClient:
    """Asynchronous client for LLM interactions via OpenRouter."""
    
    BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(self, config: Config) -> None:
        """
        Initialize the LLM client.
        
        Args:
            config: Application configuration containing API key and model settings.
        """
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=self.BASE_URL,
        )
    
    async def generate_response(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            model: Model identifier. Defaults to config.model_id if not provided.
            temperature: Sampling temperature (0.0 to 1.0). Defaults to 0.7.
        
        Returns:
            The string content of the LLM response.
        
        Raises:
            LLMClientError: If the API call fails.
        """
        model = model or self.config.model_id
        
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
            )
            
            content = response.choices[0].message.content
            if content is None:
                return ""
            return content
            
        except Exception as e:
            raise LLMClientError(f"LLM API call failed: {e}") from e


async def main() -> None:
    """Test the LLM client connection."""
    try:
        config = load_config()
        client = LLMClient(config)
        
        messages = [{"role": "user", "content": "Say hello"}]
        
        print(f"Testing connection with model: {config.model_id}")
        print("Sending message: 'Say hello'")
        print("-" * 40)
        
        response = await client.generate_response(messages)
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
