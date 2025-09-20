# api/v1/routes.py
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from application.services.customer_management_service import CustomerEngagementService
from infrastructure.azure_foundry_client import AzureFoundryClient

router = APIRouter()


# -------------------------------
# Dependencies
# -------------------------------
async def get_foundry_client() -> AzureFoundryClient:
    """
    Creates the AzureFoundryClient and ensures initialization
    (Key Vault secrets, headers, etc.). If your client supports
    lazy init, this still ensures a fast first call.
    """
    client = AzureFoundryClient()
    # If your client exposes initialize(), await it here.
    # If you rely on lazy init, you can comment this out.
    if hasattr(client, "initialize"):
        await client.initialize()
    return client


def get_customer_engagement_service(
    foundry_client: AzureFoundryClient = Depends(get_foundry_client),
) -> CustomerEngagementService:
    """
    Provides the service layer with the Foundry client dependency injected.
    """
    return CustomerEngagementService(foundry_client=foundry_client)


# -------------------------------
# Health
# -------------------------------
@router.get("/health")
async def health_check():
    return {"status": "healthy"}


# -------------------------------
# Chat: GET (query) - existing
# -------------------------------
@router.get("/chat")
async def chat_endpoint(
    prompt: str,
    model: Optional[str] = None,
    service: CustomerEngagementService = Depends(get_customer_engagement_service),
):
    """
    Simple chat endpoint using a query parameter prompt.
    Good for quick testing (curl, browser).
    """
    try:
        text = await service.chat_with_customer(
            prompt=prompt,
            model=model,            # optional: route to a specific Foundry deployment
            # system_prompt="You are a helpful assistant.",  # uncomment to steer behavior
            temperature=0.2,
            max_tokens=512,
        )
        return {"message": text}
    except Exception as e:
        # Normalize upstream errors into an HTTP status for the API consumer
        raise HTTPException(status_code=502, detail=f"Chat failed: {e}") from e


# -------------------------------
# Chat: POST (JSON) - recommended
# -------------------------------
class ChatRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 512


class ChatResponse(BaseModel):
    message: str


@router.post("/chat", response_model=ChatResponse)
async def chat_post_endpoint(
    payload: ChatRequest,
    service: CustomerEngagementService = Depends(get_customer_engagement_service),
):
    """
    Preferred chat endpoint for production:
    - Accepts JSON payloads (better for long prompts & parameters).
    - Returns a structured response model.
    """
    try:
        text = await service.chat_with_customer(
            prompt=payload.prompt,
            model=payload.model,
            system_prompt=payload.system_prompt,
            temperature=payload.temperature or 0.2,
            max_tokens=payload.max_tokens or 512,
        )
        return ChatResponse(message=text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Chat failed: {e}") from e
