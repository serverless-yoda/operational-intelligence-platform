# azure_openai_client.py
# -----------------------------------------------------------------------------
# Azure AI Foundry client that standardizes on the single Models inference
# endpoint. Applications keep one base URL and one key and route requests by
# supplying a deployment name in the "model" field.
#
# Highlights
# ----------
# - Uses a single async `invoke` method (from IFoundryClient) to call any
#   `/models/<route>` endpoint (chat/completions, embeddings, images/embeddings).
# - Provides convenience helpers for common tasks (chat, streaming chat,
#   text embeddings, image embeddings) without hiding the generic route.
# - Reads endpoint and key from Azure Key Vault via DefaultAzureCredential.
# - Supports provider-specific parameters via `extra-parameters: pass-through`.
#
# Switching auth
# --------------
# - This implementation uses key-based auth (`api-key` header). If the project
#   adopts Microsoft Entra ID, replace the header with a Bearer token and remove
#   `api-key` construction in `_build_headers`.
# -----------------------------------------------------------------------------

import os
import json
from typing import Any, AsyncIterator, Dict, Optional, Union, List

# Retained because other parts of the repository may rely on this import.
# This client uses httpx for HTTP calls; drop aiohttp if not required elsewhere.
import aiohttp  # intentionally retained
import httpx

from domain.contracts.i_foundry_client import IFoundryClient
from common.config import settings

from azure.identity.aio import DefaultAzureCredential
from azure.keyvault.secrets.aio import SecretClient


class AzureFoundryClient(IFoundryClient):
    """
    Azure AI Foundry client using the Models inference endpoint.

    Operational model
    -----------------
    - Base URL:  https://<resource>.services.ai.azure.com/models
    - Routing:   The request body includes `model: "<deployment-name>"`,
                 which directs Foundry to the correct deployment.
    - API shape: Consistent with the Foundry Model Inference API, enabling
                 chat completions, text embeddings, image embeddings, and
                 future routes without SDK churn.

    Public surface
    --------------
    - `invoke()`               : generic dispatcher for any /models route
    - `chat()`                 : backward-compatible single-prompt helper
    - `chat_messages()`       : multi-turn chat (non-streaming)
    - `chat_stream()`         : streaming chat via SSE
    - `embeddings()`          : text embeddings
    - `image_embeddings()`    : image embeddings (base64 PNG)
    """

    def __init__(self):
        # Key Vault secret names (configured in application settings).
        self.key_vault_url = f"https://{settings.azure_key_vault_name}.vault.azure.net/"
        self.azure_foundry_endpoint = settings.azure_foundry_endpoint  # KV secret name for endpoint
        self.azure_foundry_key = settings.azure_foundry_key            # KV secret name for API key

        # Optional defaults for convenience. Calls can override these per request.
        self.default_model_deployment = getattr(settings, "azure_foundry_deployment", None)
        self.api_version = getattr(settings, "azure_foundry_api_version", "2024-05-01-preview")

        # Resolved values after Key Vault lookup.
        self.azure_foundry_endpoint_value: Optional[str] = None
        self.azure_foundry_key_value: Optional[str] = None

        # Cached default headers (without provider pass-through).
        self.headers: Optional[Dict[str, str]] = None

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    async def initialize(self):
        """
        Resolve the Foundry Models endpoint and key from Azure Key Vault.

        Uses DefaultAzureCredential, which supports managed identity, CLI login,
        environment credentials, and other standard Azure auth flows.
        """
        try:
            async with DefaultAzureCredential() as credential:
                async with SecretClient(vault_url=self.key_vault_url, credential=credential) as client:
                    endpoint_secret = await client.get_secret(self.azure_foundry_endpoint)
                    key_secret = await client.get_secret(self.azure_foundry_key)

                    self.azure_foundry_endpoint_value = (endpoint_secret.value or "").rstrip("/")
                    self.azure_foundry_key_value = key_secret.value or ""

            # Build default headers once secrets are available.
            self.headers = self._build_headers(extra_params_mode="reject")

        except Exception as e:
            # Surface a clear startup failure to callers.
            raise RuntimeError(f"Failed to initialize AzureOpenAIClient: {e}") from e

    async def _ensure_ready(self):
        """
        Ensure the client has resolved secrets and prepared headers.

        This lazy guard allows either:
        - explicit initialization at app start, or
        - on-demand initialization at first call.
        """
        if not self.azure_foundry_endpoint_value or not self.azure_foundry_key_value:
            await self.initialize()
        if self.headers is None:
            self.headers = self._build_headers(extra_params_mode="reject")

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    # --- headers: allow content_type & accept to be customized or omitted ---

    # --- headers: allow content_type & accept to be customized or omitted ---

    def _build_headers(
        self,
        *,
        content_type: Optional[str] = "application/json",  # None => omit header
        accept: Optional[str] = None,                      # e.g., "text/event-stream" for SSE
        extra_params_mode: str = "reject",                 # "reject" | "pass-through"
        extra: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        Build HTTP headers for Foundry Models.

        - Key-based authentication uses `api-key`.
        - `content_type=None` omits Content-Type so multipart boundaries can be set by httpx.
        - `accept` allows explicit Accept negotiation (e.g., SSE).
        - `extra-parameters: pass-through` forwards unknown JSON fields to the provider.
        """
        h = {
            "api-key": self.azure_foundry_key_value or "",
        }
        if content_type:
            h["Content-Type"] = content_type
        if accept:
            h["Accept"] = accept
        if extra_params_mode == "pass-through":
            h["extra-parameters"] = "pass-through"
        if extra:
            h.update(extra)
        return h


    # --- invoke: support json, multipart/form-data, and raw binary ---

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

        # New flexibility knobs:
        content_type: Optional[str] = "application/json",        # None => omit, good for multipart
        accept: Optional[str] = None,                            # e.g., "text/event-stream" for SSE
        files: Optional[Dict[str, Any]] = None,                  # for multipart/form-data
        data: Optional[Dict[str, Any]] = None,                   # for multipart or form fields
        raw: Optional[Union[bytes, bytearray, memoryview]] = None,  # for application/octet-stream
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """
        Execute a generic call to any Foundry Models route, supporting:
        - JSON (`json=...`),
        - multipart/form-data (`files=...`, `data=...`),
        - raw binary (`content=...` with `content_type="application/octet-stream"`).

        Notes
        -----
        - Vision (chat) & image embeddings typically remain JSON with base64/URLs.
        - For streaming, Accept may be set to "text/event-stream".
        """
        await self._ensure_ready()

        # Inject deployment name if not present and relevant
        deployment = model or self.default_model_deployment
        if deployment and "model" not in body and not files and raw is None:
            # In multipart/raw scenarios, the model routing may be carried in form fields or URL;
            # leave it to the caller to include the right field in those cases.
            body = {**body, "model": deployment}

        # Auto Accept for streaming if caller didn't set one
        effective_accept = accept or ("text/event-stream" if stream else None)

        # Omit Content-Type when sending multipart (`files` or `data`) so httpx sets boundaries.
        effective_content_type = None if (files or data) else content_type

        url = self._url(route, api_version=api_version)
        hdrs = self._build_headers(
            content_type=effective_content_type,
            accept=effective_accept,
            extra_params_mode=extra_params_mode,
            extra=headers,
        )

        # Build request kwargs based on the chosen body shape
        request_kwargs: Dict[str, Any] = {"headers": hdrs}
        if files or data:
            # multipart/form-data or form-encoded; do not pass json=
            if data:
                request_kwargs["data"] = data
            if files:
                request_kwargs["files"] = files
        elif raw is not None:
            # raw binary payload
            request_kwargs["content"] = raw
        else:
            # default JSON
            request_kwargs["json"] = body

        if stream:
            # Stream SSE; httpx will set the appropriate response handling.
            client = httpx.AsyncClient(timeout=None)
            resp_cm = client.stream("POST", url, **request_kwargs)

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

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, **request_kwargs)
            resp.raise_for_status()
            return resp.json()

    # -------------------------------------------------------------------------
    # Convenience wrappers
    # -------------------------------------------------------------------------

    async def chat(self, prompt: str) -> str:
        """
        Backward-compatible single-turn chat helper.

        Underlying call:
          - POST /models/chat/completions with `messages=[{"role":"user","content":...}]`
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
        Multi-turn chat helper (non-streaming) returning the assistant's text.

        Notes
        -----
        - Multimodal turns are supported by supplying `content` as a list of parts
          (e.g., text and image_url entries), provided the deployment supports vision.
        - Provider-specific parameters (e.g., `logprobs`, `top_k`) can be forwarded
          by enabling `extra_params_mode="pass-through"`.
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
        Streaming chat helper yielding assistant text deltas.

        Implementation details
        ----------------------
        - Sets `stream=true` on the request.
        - Parses SSE "data:" lines and yields the `choices[0].delta.content` text parts.
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
        Text embeddings helper.

        Contract
        --------
        - POST /models/embeddings with `input` as a string or list of strings.
        - The response carries embeddings under `data[i].embedding` and usage info.
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
        Image embeddings helper for base64-encoded PNG data.

        Request shape
        -------------
        {
          "input": [
            { "type": "image", "image_format": "png", "data": "<base64-bytes>" }
          ],
          "model": "<deployment-name>"
        }
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
