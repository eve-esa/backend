import requests
import json
from typing import List, Dict, Any, Optional


class DeepInfraReranker:
    def __init__(self, api_token: str):
        """
        Initialize the DeepInfra Reranker client.

        Args:
            api_token (str): Your DeepInfra API token
        """
        self.api_token = api_token
        self.base_url = "https://api.deepinfra.com/v1/inference"
        self.model_name = "Qwen/Qwen3-Reranker-4B"

    def rerank(self, queries: List[str], documents: List[str]) -> Dict[str, Any]:
        """
        Rerank documents based on queries using the Qwen3-Reranker-4B model.

        Args:
            queries (List[str]): List of query strings
            documents (List[str]): List of document strings to rerank

        Returns:
            Dict[str, Any]: Response from the API containing reranking results

        Raises:
            requests.RequestException: If the API request fails
            ValueError: If the response is invalid
        """
        url = f"{self.base_url}/{self.model_name}"

        headers = {
            "Authorization": f"bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        payload = {"queries": queries, "documents": documents}

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            data = response.json()
            scores_with_indices = [
                {"index": idx, "reranking_score": score}
                for idx, score in enumerate(data["scores"])
            ]
            scores_with_indices.sort(key=lambda x: x["reranking_score"], reverse=True)
            return scores_with_indices

        except requests.RequestException as e:
            raise requests.RequestException(f"API request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {str(e)}")
