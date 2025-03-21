import requests
from typing import List


class LMStudioEmbeddings:
    def __init__(self, api_url: str, model: str, dimensions: int):
        self.api_url = api_url
        self.model = model
        self.dimensions = dimensions

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using the LMStudio API."""
        results = []
        for text in texts:
            embeddings = self._get_embedding(text)
            results.append(embeddings)
        return results

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the LMStudio API."""
        return self._get_embedding(text)

    def _get_embedding(self, text: str) -> List[float]:
        """Call the LMStudio API to get embeddings."""
        headers = {"Content-Type": "application/json"}
        payload = {
            "input": text,
            "model": self.model
        }

        try:
            response = requests.post(
                f"{self.api_url}/embeddings",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            embedding_data = response.json()
            return embedding_data["data"][0]["embedding"]
        except requests.exceptions.RequestException as e:
            print(f"Error getting embedding: {e}")
            # Return a zero vector as fallback
            return [0.0] * self.dimensions