# application/dto/anomaly_request_dto.py

from pydantic import BaseModel
from typing import List, Optional,Any, Dict

class AnomalyRequestDTO(BaseModel):
    metrics: List[Dict[str, Any]]
    sensitivity: Optional[float] = 0.5
    model: Optional[str] = None