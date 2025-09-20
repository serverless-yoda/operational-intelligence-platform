# application/services/customer_amanagement_service.py

from typing import Any, Dict, List, Optional
from domain.contracts.i_foundry_client import IFoundryClient


class CustomerEngagementService:
    def __init__(self, foundry_client: IFoundryClient, default_model: Optional[str] = None):
        """
        :param foundry_client: Concrete implementation of IFoundryClient (single-endpoint Foundry client).
        :param default_model:  Optional deployment name to use if the caller doesn't pass a model.
        """
        self.foundry_client = foundry_client
        self.default_model = default_model

    async def chat_with_customer(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        """
        Sends a single-turn chat prompt to Azure AI Foundry (Chat Completions) and returns assistant text.

        Args:
            prompt:        End-user message.
            model:         Foundry deployment name (overrides the service default if provided).
            system_prompt: Optional system instruction for behavior control.
            temperature:   Sampling temperature for creativity vs. determinism.
            max_tokens:    Upper bound on completion tokens.

        Returns:
            Assistant's response text.

        Raises:
            RuntimeError: If the response is missing the expected fields.
        """
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Call the generic Foundry Models route for chat completions.
        response = await self.foundry_client.invoke(
            route="chat/completions",
            body={
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            model=model or self.default_model,  # route by deployment name if available
            # extra_params_mode="reject"  # set to "pass-through" to forward provider-specific params
        )

        # Extract the assistant text safely.
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Unexpected chat response format: {response}") from e
