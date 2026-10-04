from typing import Any, Dict, List

import httpx


class JSCReranker:
    """JSC Blablador reranker (Cohere-style ``/rerank``) over a shared async client."""

    def __init__(
        self,
        api_token: str,
        base_url: str,
        model_name: str = "alias-qwen3-4b-reranking",
    ):
        self.api_token = api_token
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    async def rerank(
        self, client: httpx.AsyncClient, queries: List[str], documents: List[str]
    ) -> List[Dict[str, Any]]:
        response = await client.post(
            f"{self.base_url}/rerank",
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "query": queries[0],
                "documents": documents,
            },
        )
        response.raise_for_status()
        # A non-JSON body (Blablador answers throttling with 200 text/plain)
        # raises ValueError here, which the caller treats as a failed provider.
        data = response.json()
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            raise ValueError("JSC reranker response did not include results")

        scores_with_indices = [
            {
                "index": item["index"],
                "reranking_score": item["relevance_score"],
            }
            for item in results
        ]
        scores_with_indices.sort(key=lambda x: x["reranking_score"], reverse=True)
        return scores_with_indices
