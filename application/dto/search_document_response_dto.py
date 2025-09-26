# applucation/dto/search_document_response_dto.py

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel

class SearchDocumentsResponseDTO(BaseModel):
    results: List[Dict[str, Any]]