import os
import json
from typing import Any, AsyncIterator, Dict, Optional, Union, List

import aiohttp  # (not used below; safe to remove if unused elsewhere)
import httpx

from domain.contracts.i_ai_service import IOpenAIService
from common.config import settings

from azure.identity.aio import DefaultAzureCredential
from azure.keyvault.secrets.aio import SecretClient


class AzureOpenAIClient(IOpenAIService):
    """
    Azure AI Foundry client using the single Models inference endpoint.
    Supports generic invocation plus helpers for chat, embeddings, and image embeddings.

    References:
      - Endpoints & routing: https://learn.microsoft.com/azure/ai-foundry/foundry-models/concepts/endpoints
      - Chat Completions REST: https://github.com/MicrosoftDocs/azure-ai-docs/blob/main/articles/ai-foundry/foundry-models/includes/use-chat-completions/rest.md
      - Text Embeddings REST: https://learn.microsoft.com/rest/api/aifoundry/model-inference/get-embeddings/get-embeddings
      - Image Embeddings REST: https://learn.microsoft.com/rest/api/aifoundry/model-inference/get-image-embeddings
      - Extra parametercs pass-through: https://learn.microsoft.com/rest/api/aifoundry/modelinference/
    """

    def __init__(self):
        # Key Vault bootstrap (you already had these)
        self.key_vault_url = f"https://{settings.azure_key_vault_name}.vault.azure.net/"
        self.azure_foundry_endpoint = settings.azure_foundry_endpoint  # secret name
        self.azure_foundry_key = settings.azure_foundry_key            # secret name

        # Optional (if you store these in app settings, not KV)
        self.default_model_deployment = getattr(settings, "azure_foundry_deployment", None)
        self.api_version = getattr(settings, "azure_foundry_api_version", "2024-05-01-preview")

        # Resolved values from Key Vault
        self.azure_foundry_endpoint_value: Optional[str] = None  # e.g., https://<resource>.services.ai.azure.com
        self.azure_foundry_key_value: Optional[str] = None

        # Cached default headers (without pass-through)
        self.headers: Optional[Dict[str, str]] = None

    # -------------------- Bootstrap --------------------

    async def initialize(self):
        """
        Resolve endpoint and key from Azure Key Vault using DefaultAzureCredential.
        """
        try:
            async with DefaultAzureCredential() as credential:
                async with SecretClient(vault_url=self.key_vault_url, credential=credential) as client:
                    _endpoint = await client.get_secret(self.azure_foundry_endpoint)
                    _key = await client.get_secret(self.azure_foundry_key)
                    self.azure_foundry_endpoint_value = _endpoint.value.rstrip("/")
                    self.azure_foundry_key_value = _key.value

            # Build default headers after secrets resolve
            self.headers = self._build_headers(extra_params_mode="reject")

        except Exception as e:
            # Re-raise with context so callers can decide how to handle
            raise RuntimeError(f"Failed to initialize AzureOpenAIClient: {e}") from e

    async def _ensure_ready(self):
        """
        Ensure secrets and headers are ready; lazily initialize if not done yet.
        """
        if not self.azure_foundry_endpoint_value or not self.azure_foundry_key_value:
            await self.initialize()
        if self.headers is None:
            self.headers = self._build_headers(extra_params_mode="reject")

    # -------------------- Internals --------------------

    def _build_headers(
        self,
        *,
        extra_params_mode: str = "reject",  # "reject" | "pass-through"
        extra: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        Construct headers. Use extra-parameters: pass-through to forward unknown JSON fields.
        """
        h = {
            "api-key": self.azure_foundry_key_value or "",
            "Content-Type": "application/json",
        }
        if extra_params_mode == "pass-through":
            h["extra-parameters"] = "pass-through"
        if extra:
            h.update(extra)
        return h

    def _url(self, route: str, api_version: Optional[str] = None) -> str:
        route = route.strip().lstrip("/")
        base = self.azure_foundry_endpoint_value or ""
        ver = api_version or self.api_version
        return f"{base}/models/{route}?api-version={ver}"

    # -------------------- Generic invoke --------------------

    async def invoke(
        self,
        route: str,
        body: Dict[str, Any],
        *,
        model: Optional[str] = None,
        stream: bool = False,
        extra_params_mode: str = "reject",
        headers: Optional[Dict[str, str]] = None,
        api_version: Optional[str] = None,
        timeout: Optional[float] = 60.0,
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """
        Generic invoker for any Foundry Models route.

        Args:
            route: e.g., "chat/completions", "embeddings", "images/embeddings"
            body: JSON payload for that route
            model: deployment name to route to (overrides self.default_model_deployment)
            stream: for chat streaming via SSE
            extra_params_mode: "reject" (default) or "pass-through" to forward unknown params
            headers: extra headers to add
            api_version: override API version per-call
            timeout: request timeout (no effect on streaming)

        Returns:
            If stream=False -> dict
            If stream=True  -> async iterator yielding dict chunks from SSE
        """
        await self._ensure_ready()

        mdl = model or self.default_model_deployment
        if mdl and "model" not in body:
            body = {**body, "model": mdl}

        url = self._url(route, api_version=api_version)
        hdrs = self._build_headers(extra_params_mode=extra_params_mode, extra=headers)

        if stream:
            # SSE streaming (primarily for chat)
            body = {**body, "stream": True}
            client = httpx.AsyncClient(timeout=None)
            resp_cm = client.stream("POST", url, headers=hdrs, json=body)

            async def _aiter() -> AsyncIterator[Dict[str, Any]]:
                async with client:
                    async with resp_cm as r:
                        r.raise_for_status()
                        async for line in r.aiter_lines():
                            if not line or not line.startswith("data:"):
                                continue
                            if line.strip() == "data: [DONE]":
                                break
                            yield json.loads(line.removeprefix("data: ").strip())

            return _aiter()

        # Non-streaming requests
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, headers=hdrs, json=body)
            resp.raise_for_status()
            return resp.json()

    # -------------------- Convenience wrappers --------------------

    async def chat(self, prompt: str) -> str:
        """
        Keeps your original interface: single-string prompt -> assistant text.
        Uses chat/completions under the hood.
        """
        messages = [{"role": "user", "content": prompt}]
        data = await self.invoke("chat/completions", body={"messages": messages})
        return data["choices"][0]["message"]["content"]

    async def chat_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_params_mode: str = "reject",
    ) -> str:
        """
        Multi-turn chat helper returning assistant text.
        """
        body: Dict[str, Any] = {"messages": messages}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        data = await self.invoke(
            "chat/completions",
            body=body,
            model=model,
            extra_params_mode=extra_params_mode,
        )
        return data["choices"][0]["message"]["content"]

    async def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        extra_params_mode: str = "reject",
    ) -> AsyncIterator[str]:
        """
        Streaming chat: yields assistant text deltas as they arrive.
        """
        stream_iter = await self.invoke(
            "chat/completions",
            body={"messages": messages},
            model=model,
            stream=True,
            extra_params_mode=extra_params_mode,
        )

        async for chunk in stream_iter:  # type: ignore
            delta = chunk["choices"][0].get("delta", {}).get("content")
            if delta:
                yield delta

    async def embeddings(
        self,
        inputs: Union[str, List[str]],
        *,
        model: Optional[str] = None,
        extra_params_mode: str = "reject",
    ) -> Dict[str, Any]:
        """
        Text embeddings via /models/embeddings.
        """
        return await self.invoke(
            "embeddings",
            body={"input": inputs},
            model=model,
            extra_params_mode=extra_params_mode,
        )

    async def image_embeddings(
        self,
        image_base64_png: str,
        *,
        model: Optional[str] = None,
        extra_params_mode: str = "reject",
    ) -> Dict[str, Any]:
        """
        Image embeddings via /models/images/embeddings.
        Pass raw base64 PNG data (no 'data:image/png;base64,' prefix).
        """
        body = {
            "input": [
                {
                    "type": "image",
                    "image_format": "png",
                    "data": image_base64_png,
                }
            ]
        }
        return await self.invoke(
            "images/embeddings",
            body=body,
            model=model,
            extra_params_mode=extra_params_mode,
        )
