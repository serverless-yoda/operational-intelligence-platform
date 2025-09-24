# application/services/document_processing_service.py

from lzma import MODE_NORMAL
from domain.contracts.i_foundry_client import IFoundryClient
from typing import Any, Dict, Union, Optional

class DocumentProcessingService:
    def __init__(self, foundry_client: IFoundryClient, default_model: Optional[str] = None):
        self.foundry_client = foundry_client
        self.default_model = default_model

    
    async def extract_strucutured_data(self, 
            document_base64: str, 
            *, 
            model: Optional[str] = None, 
            document_type: Optional[str]=None, 
            additional_params: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:

        payload={
            "input": [
                {
                    "type": "image",
                    "image_format": "png",
                    "data": document_base64
                }
            ]
        }

        if document_type:
            payload["document_type"] = document_type
        if additional_params:
            payload.update(additional_params)

        response = await self.foundry_client.invoke(route="images/embeddings",body=payload,model=model)

        return response

    
    async def classify_document(self,
            document_base64: str,
            *,
            model: Optional[str]=None,
            additional_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        
        payload = {
            "input": [
                {
                    "type": "image",
                    "image_format": "png",
                    "data": document_base64
                }
            ]
        }

        if additional_params:
            payload.update(additional_params)
        response = await self.foundry_client.invoke(
            route="chat/completions",
            body=payload,
            model=model
        )

        return response