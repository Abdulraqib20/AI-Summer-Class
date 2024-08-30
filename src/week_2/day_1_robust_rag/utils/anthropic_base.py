"""
Customise the Anthropic class here using the anthropic base and utils scripts from llama-index.
This script should address the following:
    - Enforce AnthropicVertex client within the Anthropic class, to ensure you can access the models via GCP
    - Update model codes to resolve model code conflicts between Anthropic and AnthropicVertex
"""

"""
Customise the Anthropic class here using the anthropic base and utils scripts from llama-index.
This script should address the following:
    - Enforce AnthropicVertex client within the Anthropic class, to ensure you can access the models via GCP
    - Update model codes to resolve model code conflicts between Anthropic and AnthropicVertex
"""

from typing import Any, Dict, Optional, Sequence

from llama_index.llms.anthropic import Anthropic as BaseAnthropic
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.anthropic.utils import (
    anthropic_modelname_to_contextsize,
    messages_to_anthropic_messages
)

class Anthropic(BaseAnthropic):
    def __init__(
        self,
        model: str,
        vertex_client: Any,  # AnthropicVertex client
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the Anthropic LLM."""
        self.vertex_client = vertex_client
        self.model = self._resolve_model_name(model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.additional_kwargs = additional_kwargs or {}

    def _resolve_model_name(self, model: str) -> str:
        """Resolve model name conflicts between Anthropic and AnthropicVertex."""
        model_mapping = {
            "claude-3-5-sonnet@20240620": "claude-3-sonnet-20240229",
            "claude-3-haiku@20240307": "claude-3-haiku-20240307",
            "claude-3-3-opus@20240229": "claude-3-opus-20240229",
        }
        return model_mapping.get(model, model)

    def complete(self, prompt: str, **kwargs: Any) -> Any:
        """Completion endpoint for Anthropic."""
        messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
        anthropic_messages = messages_to_anthropic_messages(messages)
        
        response = self.vertex_client.completions.create(
            model=self.model,
            messages=anthropic_messages,
            temperature=self.temperature,
            max_tokens_to_sample=self.max_tokens,
            **self.additional_kwargs,
            **kwargs,
        )
        return response.completion

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> Any:
        """Chat endpoint for Anthropic."""
        anthropic_messages = messages_to_anthropic_messages(messages)
        
        response = self.vertex_client.messages.create(
            model=self.model,
            messages=anthropic_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **self.additional_kwargs,
            **kwargs,
        )
        return response.content[0].text

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    @property
    def context_window(self) -> int:
        """Get the context window for the model."""
        return anthropic_modelname_to_contextsize(self.model)