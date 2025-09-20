# domain/contracts/i_foundry_client.py
# -----------------------------------------------------------------------------
# This interface defines a minimal, generic contract for calling Azure AI
# Foundry Models via a single async method, `invoke`. Keeping the surface area
# small allows multiple implementations (httpx, aiohttp, test doubles) to be
# swapped in without touching call sites.
# -----------------------------------------------------------------------------

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Optional, Union


class IFoundryClient(ABC):
    """
    Generic contract for invoking Azure AI Foundry Models routes.

    Design notes
    ------------
    - The goal is to keep the API compact and future-proof. One method covers
      chat completions, text embeddings, image embeddings, and new routes as
      they appear under `/models/`.
    - The return type is flexible: non-streaming returns a JSON dict; streaming
      returns an async iterator of JSON-decoded SSE chunks.
    """

    @abstractmethod
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
        Perform a generic call to any route under `/models/`.

        Expected usage patterns:
          - route="chat/completions" with {"messages": [...]}
          - route="embeddings" with {"input": ...}
          - route="images/embeddings" with {"input": [...] image parts}

        Returns
        -------
        - Non-streaming: a dict with the parsed JSON response.
        - Streaming: an async iterator yielding dict chunks (each an SSE "data:" payload).
        """
        #raise NotImplementedError
        pass
