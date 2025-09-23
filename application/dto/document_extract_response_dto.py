# application/dto/document_extract_response_dto.py

from typing import Optional
from pydantic import BaseModel

class DocumentExtractResponseDTO(BaseModel):
    extracted_data: str