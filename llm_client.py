from abc import ABC, abstractmethod
from typing import Dict, Any
import openai


class LLMClient(ABC):
    """
    Abstract base class for Language Model clients.
    """

    @abstractmethod
    def generate_text(self, prompt: str, model: str, temperature: float | None, max_output_tokens: int | None, **kwargs) -> str:
        """
        Generates text based on the given prompt.

        Args:
            prompt: The prompt to use for text generation.
            model: The name of the OpenAI model to use.
            temperature: The temperature to use for text generation.
            max_output_tokens: The maximum number of output tokens.
            **kwargs: Additional keyword arguments for the API.

        Returns:
            The generated text.
        """
        pass

    @abstractmethod
    def get_usage_info(self) -> Dict:
        """
        Returns information about the API usage.

        Returns:
            A dictionary containing usage information.
        """
        pass


class OpenAIClient(LLMClient):
    """
    Client for interacting with the OpenAI API.
    """

    def __init__(self, api_key: str):
        """
        Initializes the OpenAI client.

        Args:
            api_key: The OpenAI API key.
        """
        self.api_key = api_key
        openai.api_key = self.api_key
        self.client = openai.OpenAI()
        print("OpenAI client initialized")

    def generate_text(self, prompt: str, model: str, temperature: float | None, max_output_tokens: int | None, **kwargs) -> str:
        """
        Generates text using the OpenAI API.

        Args:
            prompt: The prompt to use for text generation.
            model: The name of the OpenAI model to use.
            temperature: The temperature to use for text generation.
            max_output_tokens: The maximum number of output tokens.
            **kwargs: Additional keyword arguments for the API.

        Returns:
            The generated text.
        """
        try:
            # Common parameters
            params: Dict[str, Any] = {
                "model": model,
                "prompt": prompt,
                **kwargs
            }

            # Add optional parameters if provided
            if temperature is not None:
                params["temperature"] = temperature
            if max_output_tokens is not None:
                params["max_tokens"] = max_output_tokens

            response = self.client.completions.create(**params)
            return response.choices[0].text.strip()
        except Exception as e:
            print(f"Error generating text from OpenAI: {e}")
            return ""

    def get_usage_info(self) -> Dict:
        """
        Returns usage information from the OpenAI API.

        Returns:
            A dictionary containing usage information.
        """
        # This functionality might not be directly available in the 'openai' library.
        # You might need to track token usage separately or use a different API endpoint
        return {"model": self.model_name, "usage": "Not available"}
