from typing import Any, Dict, List

import httpx


class DeepInfraReranker:
    """DeepInfra ``Qwen/Qwen3-Reranker-4B`` inference endpoint over a shared async client."""

    def __init__(self, api_token: str):
        self.api_token = api_token
        self.base_url = "https://api.deepinfra.com/v1/inference"
        self.model_name = "Qwen/Qwen3-Reranker-4B"

    async def rerank(
        self, client: httpx.AsyncClient, queries: List[str], documents: List[str]
    ) -> List[Dict[str, Any]]:
        response = await client.post(
            f"{self.base_url}/{self.model_name}",
            headers={
                "Authorization": f"bearer {self.api_token}",
                "Content-Type": "application/json",
            },
            json={"queries": queries, "documents": documents},
        )
        response.raise_for_status()
        data = response.json()
        scores_with_indices = [
            {"index": idx, "reranking_score": score}
            for idx, score in enumerate(data["scores"])
        ]
        scores_with_indices.sort(key=lambda x: x["reranking_score"], reverse=True)
        return scores_with_indices
