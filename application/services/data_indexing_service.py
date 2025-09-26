from domain.contracts.i_foundry_client import IFoundryClient
from typing import Dict, Any, List, Optional, Union

class DataIndexingService:
    """
    Service for indexing documents and performing search queries via Azure AI Foundry.
    """

    def __init__(self, foundry_client: IFoundryClient, default_model:Optional[str] = None):
        self.foundry_client = foundry_client
        self.default_model = default_model

    async def index_documents(
        self,
        documents: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        index_name: Optional[str] = None,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Indexes a batch of documents (text and/or metadata).
        Args:
            documents: List of dicts representing each document and its fields.
            model: Optional model/deployment name for Foundry or Search endpoint.
            index_name: The name of the search index/dataset (if needed).
            additional_params: Any extra fields for special index options.
        Returns:
            Dict summarizing indexing results (success, failures).
        """
        payload = {
            "documents": documents,
        }
        if index_name:
            payload["index_name"] = index_name
        if additional_params:
            payload.update(additional_params)

        response = await self.foundry_client.invoke(
            route="indexer/index-documents",  # Match to your Foundry/CogSearch endpoint route
            body=payload,
            model=model,
        )
        return response

    async def search_documents(
        self,
        query: str,
        *,
        model: Optional[str] = None,
        index_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Performs a semantic or keyword search against the indexed data.
        Args:
            query: The natural language or keyword search string.
            model: Optional retrieval or rerank model name.
            index_name: Search index to query (if applicable).
            filters: Optional structured search filters.
            top_k: Max results to return.
            additional_params: Any extra options (e.g., NSFW filters, rerank settings).
        Returns:
            List of top-matching document dicts.
        """
        payload = {
            "query": query,
            "top_k": top_k,
        }
        if index_name:
            payload["index_name"] = index_name
        if filters:
            payload["filters"] = filters
        if additional_params:
            payload.update(additional_params)

        response = await self.foundry_client.invoke(
            route="indexer/search-documents",  # Adjust for your actual endpoint
            body=payload,
            model=model,
        )
        return response.get("results", [])

    async def delete_documents(
        self,
        doc_ids: List[Union[str, int]],
        *,
        model: Optional[str] = None,
        index_name: Optional[str] = None,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Deletes documents by ID from the index.
        Args:
            doc_ids: List of document IDs to remove.
            model: (Optional) Backend model name.
            index_name: (Optional) Dataset or index to delete from.
            additional_params: Additional flags or options.
        Returns:
            Dict summarizing deletions.
        """
        payload = {
            "doc_ids": doc_ids,
        }
        if index_name:
            payload["index_name"] = index_name
        if additional_params:
            payload.update(additional_params)

        response = await self.foundry_client.invoke(
            route="indexer/delete-documents",
            body=payload,
            model=model,
        )
        return response
