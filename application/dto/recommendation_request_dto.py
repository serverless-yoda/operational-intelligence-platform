# application/dto/recommendation_request_dto.py

from typing import Optional, Dict
from pydantic import BaseModel

class RecommendationRequestDTO(BaseModel):
    user_id: str
    context: Dict[str, any]
    model: Optional[str] = None
    num_results: Optional[int] = 3
