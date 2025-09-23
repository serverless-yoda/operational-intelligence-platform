# application/dto/document_extract_request_dto.py

from typing import Optional
from pydantic import BaseModel

class DocumentExtractRequestDTO(BaseModel):
    document_base64: str
    document_type: Optional[str] = None
    model: Optional[str] = None