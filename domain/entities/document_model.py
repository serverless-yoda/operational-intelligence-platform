# domain/entities/document_model.py

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel

class DocumentModel(BaseModel):
    doc_id: Optional[Union[str, int]]
    text: str
    metadata: Optional[Dict[str, Any]] = None