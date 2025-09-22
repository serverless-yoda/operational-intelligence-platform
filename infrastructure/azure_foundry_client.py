# infrastructure/azure_foundry_client.py
import json
from typing import Any, AsyncIterator, Dict, Optional, Union, List

import httpx
from azure.identity.aio import DefaultAzureCredential
from azure.keyvault.secrets.aio import SecretClient

from domain.contracts.i_foundry_client import IFoundryClient
from common.config import settings


class AzureFoundryClient(IFoundryClient):
    """
    Azure AI Foundry client using the Models inference endpoint.
    Assumes the Key Vault secret for inference URL already includes `/models`.
    """

    def __init__(self):
        self._kv_url = f"https://{settings.azure_key_vault_name}.vault.azure.net/"
        self._endpoint_secret_name = settings.azure_foundry_endpoint
        self._inference_secret_name = settings.azure_foundry_inference_url
        self._key_secret_name = settings.azure_foundry_key

        self._default_model = getattr(settings, "azure_foundry_deployment", None)
        self._embed_model= getattr(settings, "azure_foundry_deployment_embed", None)
        self._api_version = getattr(settings, "azure_foundry_api_version", "2024-05-01-preview")

        self._inference_base: Optional[str] = None
        self._api_key: Optional[str] = None
        self._client: Optional[httpx.AsyncClient] = None

    async def initialize(self):
        """Fetch secrets from Key Vault and prepare HTTP client."""
        try:
            async with DefaultAzureCredential() as cred:
                async with SecretClient(vault_url=self._kv_url, credential=cred) as kv:
                    inference_url = await kv.get_secret(self._inference_secret_name)
                    api_key = await kv.get_secret(self._key_secret_name)

            self._inference_base = (inference_url.value or "").rstrip("/")
            self._api_key = api_key.value or ""

            if self._client is None:
                self._client = httpx.AsyncClient(timeout=60.0)

        except Exception as e:
            raise RuntimeError(f"Failed to initialize AzureFoundryClient: {e}") from e

    async def _ensure_ready(self):
        if not (self._inference_base and self._api_key):
            await self.initialize()
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)

    def _url(self, route: str, api_version: Optional[str] = None) -> str:
        """Compose full URL: <inference-base>/<route>?api-version=<ver>."""
        ver = api_version or self._api_version
        return f"{self._inference_base}/{route.lstrip('/')}?api-version={ver}"

    def _headers(
        self,
        *,
        content_type: Optional[str] = "application/json",
        accept: Optional[str] = None,
        extra_params_mode: str = "reject",
        extra: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        headers = {"api-key": self._api_key}
        if content_type:
            headers["Content-Type"] = content_type
        if accept:
            headers["Accept"] = accept
        if extra_params_mode == "pass-through":
            headers["extra-parameters"] = "pass-through"
        if extra:
            headers.update(extra)
        return headers

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
        content_type: Optional[str] = "application/json",
        accept: Optional[str] = None,
        files: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        raw: Optional[Union[bytes, bytearray, memoryview]] = None,
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """
        Generic dispatcher for Foundry Models. Supports JSON, multipart/form-data,
        and raw binary; SSE streaming for chat-completions routes.

        Notes
        -----
        - For chat streaming, the API requires BOTH:
            1) Accept: text/event-stream
            2) { "stream": true } in the JSON body
        - Routes that require a deployment name: chat/completions, embeddings, images/embeddings.
        """
        await self._ensure_ready()

        # Routes that require a model (deployment name)
        requires_model = route.lstrip("/") in {
            "chat/completions",
            "embeddings",
            "images/embeddings",
        }

        # Inject deployment if not provided (for JSON path only)
        if "model" not in body and not files and raw is None:
            body = {**body, "model": model or self._default_model}

        # Fail fast if still missing for routes that need a model
        if requires_model and "model" not in body:
            raise ValueError(
                "Missing 'model' (deployment name). "
                "Pass model=... or set settings.azure_foundry_deployment."
            )

        # Streaming: set Accept header and 'stream': true in JSON payload
        effective_accept = accept or ("text/event-stream" if stream else None)
        effective_ct = None if (files or data) else content_type

        url = self._url(route, api_version)
        hdrs = self._headers(
            content_type=effective_ct,
            accept=effective_accept,
            extra_params_mode=extra_params_mode,
            extra=headers,
        )

        if stream and not files and raw is None:
            body = {**body, "stream": True}

        # Build the request kwargs based on body shape
        req_kwargs: Dict[str, Any] = {"headers": hdrs}
        if files or data:
            if data:
                req_kwargs["data"] = data
            if files:
                req_kwargs["files"] = files
        elif raw is not None:
            req_kwargs["content"] = raw
        else:
            req_kwargs["json"] = body

        # ---- Streaming path (SSE) ----
        if stream:
            assert self._client is not None
            # For streams, avoid a hard timeout unless you want to enforce one
            resp_cm = self._client.stream("POST", url, timeout=None, **req_kwargs)

            async def _aiter() -> AsyncIterator[Dict[str, Any]]:
                async with resp_cm as r:
                    r.raise_for_status()
                    ctype = (r.headers.get("Content-Type") or "").lower()
                    # Fallback: if not SSE, try to parse full JSON once and yield
                    if "text/event-stream" not in ctype:
                        full = await r.aread()
                        try:
                            yield json.loads(full)
                        except Exception:
                            # If the server returned non-JSON with 200 OK, nothing to stream
                            return
                        return

                    async for line in r.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        if line.strip() == "data: [DONE]":
                            break
                        yield json.loads(line.removeprefix("data: ").strip())

            return _aiter()

        # ---- Non-streaming path ----
        assert self._client is not None
        resp = await self._client.post(url, timeout=timeout, **req_kwargs)
        resp.raise_for_status()
        return resp.json()

    # ---------------- Convenience wrappers ----------------


    async def chat(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_params_mode: str = "reject",
    ) -> str:
        body: Dict[str, Any] = {"messages": [{"role": "user", "content": prompt}]}
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


    async def chat_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_params_mode: str = "reject",
    ) -> str:
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
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_params_mode: str = "reject",
    ) -> AsyncIterator[str]:
        # Build body including tunables
        body: Dict[str, Any] = {"messages": messages}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        # Call with streaming enabled (invoke adds "stream": true + SSE Accept)
        stream_iter = await self.invoke(
            "chat/completions",
            body=body,
            model=model,
            stream=True,
            extra_params_mode=extra_params_mode,
        )

        async for chunk in stream_iter:  # type: ignore
            # Basic shape guard
            if not isinstance(chunk, dict):
                continue

            # Some error payloads come back in-stream
            if "error" in chunk:
                # surface the provider message, if present
                err = chunk.get("error")
                raise RuntimeError(f"Streaming error: {err}")

            choices = chunk.get("choices") or []
            if not choices:
                # Could be a usage/final or heartbeat-like event; skip quietly
                continue

            choice0 = choices[0] or {}

            # 1) Streaming deltas (most common)
            delta_obj = choice0.get("delta") or {}
            if isinstance(delta_obj, dict):
                # a) Simple string content
                content = delta_obj.get("content")
                if isinstance(content, str) and content:
                    yield content
                    continue

                # b) Multimodal content (list of parts). Yield only text parts if present.
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            t = part.get("text")
                            if t:
                                yield t
                    continue

                # c) Role-only deltas, tool_calls, etc. -> nothing to emit; continue

            # 2) Some providers send a final non-delta message
            message_obj = choice0.get("message") or {}
            final_content = message_obj.get("content")
            if isinstance(final_content, str) and final_content:
                yield final_content
                continue


    async def embeddings(self, inputs: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        return await self.invoke("embeddings", {"input": inputs}, **kwargs)

    async def image_embeddings(self, image_base64_png: str, **kwargs) -> Dict[str, Any]:
        return await self.invoke("images/embeddings", {
            "input": [{"type": "image", "image_format": "png", "data": image_base64_png}]
        }, **kwargs)
