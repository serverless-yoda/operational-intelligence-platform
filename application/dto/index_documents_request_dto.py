# applucation/dto/index_documents_request_dto.py

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel
from domain.entities.document_model import DocumentModel

class IndexDocumentsRequestDTO(BaseModel):
    documents: List[DocumentModel]
    index_name: Optional[str] = None
    model: Optional[str] = None