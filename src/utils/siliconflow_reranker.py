import requests
import json
from typing import List


class SiliconFlowReranker:
    def __init__(self, api_token: str):
        """
        Initialize the SiliconFlow Reranker client.

        Args:
            api_token (str): Your SiliconFlow API token
        """
        self.api_token = api_token
        self.base_url = "https://api.siliconflow.com/v1/rerank"
        self.model_name = "Qwen/Qwen3-Reranker-4B"

    def rerank(self, queries: List[str], documents: List[str]) -> List[float]:
        """
        Rerank documents based on a single query using the Qwen3-Reranker-4B model.

        Args:
            queries (List[str]): List of query strings (only one query supported)
            documents (List[str]): List of document strings to rerank

        Returns:
            List[float]: List of relevance scores corresponding to each document

        Raises:
            requests.RequestException: If the API request fails
            ValueError: If the response is invalid
        """
        if len(queries) != 1:
            raise ValueError(
                "SiliconFlowReranker currently supports only one query per request."
            )

        query = queries[0]

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "return_documents": True,
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            # Extract relevance scores
            if "results" not in data:
                raise ValueError(f"Unexpected response format: {data}")
            reranked_results = data["results"]
            scores_with_indices = [
                {"index": item["index"], "reranking_score": item["relevance_score"]}
                for item in reranked_results
            ]
            scores_with_indices.sort(key=lambda x: x["reranking_score"], reverse=True)
            return scores_with_indices

        except requests.RequestException as e:
            raise requests.RequestException(f"API request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {str(e)}")
