# applucation/dto/index_documents_request_dto.py

from code import interact
from pydantic import BaseModel

class IndexDocumentResponseDTO(BaseModel):
    indexed: int
    failed: interact