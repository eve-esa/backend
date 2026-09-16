import json
from typing import Any, Dict, List

import requests


class JSCReranker:
    def __init__(
        self,
        api_token: str,
        base_url: str,
        model_name: str = "alias-qwen3-4b-reranking",
    ):
        self.api_token = api_token
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    def rerank(self, queries: List[str], documents: List[str]) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/rerank"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "query": queries[0],
            "documents": documents,
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            results = data.get("results")
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

        except requests.RequestException as e:
            raise requests.RequestException(f"API request failed: {str(e)}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {str(e)}") from e
