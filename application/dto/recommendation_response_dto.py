# application/dto/recommendation_response_dto.py

from typing import  Dict, List, Any
from pydantic import BaseModel

class RecommendationResponseDTO(BaseModel):
     recommendations: List[Dict[str, Any]]
