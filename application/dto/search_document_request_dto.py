# application/dto/search_document_request_dto.py

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel


class SearchDocumentsRequestDTO(BaseModel):
    query: str
    index_name: Optional[str] = None
    model: Optional[str] = None
    top_k: Optional[int] = 10
    filters: Optional[Dict[str, Any]] = None