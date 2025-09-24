# application/service/personalisation_service.py

from domain.contracts.i_foundry_client import IFoundryClient
from typing import Any, Dict, Optional, List



class PersonalizationService:
    """
    Service for generating personalized content or recommendations using Azure AI Foundry.
    """

    def __init__(self, foundry_client: IFoundryClient, default_model: Optional[str] = None):
        self.foundry_client = foundry_client
        self.default_model = default_model

    async def get_recommendations(
        self,
        user_id: str,
        context: Dict[str, Any],
        *,
        model: Optional[str] = None,
        num_results: int = 3,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Gets recommended items for a user given their context and preferences.
        Args:
            user_id: The unique identifier for the end-user.
            context: A dictionary of user context (history, session, preferences, etc.).
            model: Optional model/deployment name for Foundry endpoint.
            num_results: How many recommendations to return.
            additional_params: Any extra fields or model options.
        Returns:
            List of recommended items (can be product IDs, content, actions).
        """
        query = {
            "user_id": user_id,
            "context": context,
            "num_results": num_results,
        }
        if additional_params:
            query.update(additional_params)

        response = await self.foundry_client.invoke(
            route="personalize/recommendations",  # Route can be adjusted as per actual Foundry setup
            body=query,
            model=model,
        )
        # Defensive parse: assume Foundry returns list under response["results"]
        return response.get("results", [])

    async def personalize_message(
        self,
        user_id: str,
        base_message: str,
        context: Dict[str, Any],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = 0.25,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Creates or modifies a message/content for a user, personalized with their context."""
        payload = {
            "user_id": user_id,
            "base_message": base_message,
            "context": context,
            "temperature": temperature
        }
        if additional_params:
            payload.update(additional_params)

        response = await self.foundry_client.invoke(
            route="personalize/message",  # Route can be changed to fit your API spec/LLM
            body=payload,
            model=model
        )
        return response.get("personalized_message") or response.get("result") or ""

