# api/v1/routes.py
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException


from application.services.customer_management_service import CustomerEngagementService
from application.services.document_processing_service import DocumentProcessingService
from application.services.personalisation_service import PersonalizationService
from application.services.operational_intelligence_service import OperationalIntelligenceService

from application.dto.document_extract_request_dto import DocumentExtractRequestDTO
from application.dto.document_extract_response_dto import DocumentExtractResponseDTO
from application.dto.chat_request_dto import ChatRequestDTO
from application.dto.chat_response_dto import ChatResponseDTO
from application.dto.recommendation_request_dto import RecommendationRequestDTO
from application.dto.recommendation_response_dto import RecommendationResponseDTO
from application.dto.anomaly_request_dto import AnomalyRequestDTO


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

def get_document_processing_service(
    foundry_client: AzureFoundryClient = Depends(get_foundry_client),
) -> DocumentProcessingService:
    """
    Provides the service layer with the Foundry client dependency injected.
    """
    return DocumentProcessingService(foundry_client=foundry_client)

def get_personalisation_service(
    foundry_client: AzureFoundryClient = Depends(get_foundry_client)
) -> PersonalizationService:
    return PersonalizationService(foundry_client=foundry_client)

def get_operational_intelligence_service(
    foundry_client: AzureFoundryClient = Depends(get_foundry_client)
)-> OperationalIntelligenceService
    return OperationalIntelligenceService(foundry_client=foundry_client)
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


@router.post("/chat", response_model=ChatResponseDTO)
async def chat_post_endpoint(
    payload: ChatRequestDTO,
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
        return ChatResponseDTO(message=text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Chat failed: {e}") from e


@router.post("/document/extract", response_model=DocumentExtractResponseDTO)
async def extract_document_post_endpoint(payload: DocumentExtractRequestDTO,
    service: DocumentProcessingService = Depends(get_document_processing_service)):
    results = await service.extract_strucutured_data(
        document_base64=payload.document_base64,
        model=payload.model,
        document_type=payload.document_type
    )

    return DocumentExtractResponseDTO(
        extracted_data=results
    )

    @route.post("/document/classify", response_model=DocumentExtractResponseDTO)
    async def classify_document_post_endpoint(payload: DocumentExtractRequestDTO,
        service: DocumentProcessingService = Depends(get_document_processing_service)):
        results = await service.extract_strucutured_data(
            document_base64=payload.document_base64,
            model=payload.model,
            document_type=payload.document_type
        )

        return DocumentExtractResponseDTO(
            extracted_data=results
        )


@router.post("/personalize/recommend", response_model=RecommendationResponseDTO)
async def recommend_endpoint(
    payload: RecommendationRequestDTO,
    service: PersonalizationService = Depends(get_personalisation_service)
):
    result = await service.get_recommendations(
        user_id=payload.user_id,
        context=payload.context,
        model=payload.model,
        num_results=payload.num_results
    )
    return RecommendationResponseDTO(recommendations=result)


@router.post("/ops/anomaly")
async def anomaly_endpoint(
    payload: AnomalyRequestDTO,
    service: OperationalIntelligenceService = Depends(get_operational_intelligence_service)
):
    result = await service.detect_anomalies(
        metrics=payload.metrics,
        sensitivity=payload.sensitivity,
        model=payload.model,
    )
    return result