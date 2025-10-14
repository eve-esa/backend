import requests
from typing import List, Tuple
from urllib.parse import urlparse
import trafilatura
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()


class ScrapingDogCrawler:
    def __init__(self, all_urls: List[str], api_key: str) -> None:
        self.api_key = api_key
        self.domains = list({urlparse(u).netloc for u in all_urls})
        self.domain_query = " OR ".join(f"site:{d}" for d in self.domains)
        self.base_url = "https://api.scrapingdog.com/google"

    def _search_scrapingdog(self, query: str, top_k: int) -> List[Tuple[str, str]]:
        q = f"({self.domain_query}) {query}"
        params = {
            "api_key": self.api_key,
            "query": q,
            "country": "us",
            "advance_search": "true",
            "domain": "google.com",
        }
        response = requests.get(self.base_url, params=params)
        if response.status_code != 200:
            print(f"Request failed with {response.status_code}")
            return []
        data = response.json()
        organic_results = data.get("organic_results", [])
        urls = [
            (r.get("title", ""), r.get("link", ""))
            for r in organic_results
            if r.get("title") and r.get("link")
        ]
        print(f"ScrapingDog returned {len(urls)} URLs")
        return urls[
            :top_k
        ]  # because in the api params, it returns the whole page and not top k results

    def _extract_single(self, title_url):
        """Helper for parallel extraction"""
        title, url = title_url
        try:
            r = httpx.get(url, verify=False, timeout=20)
            text = trafilatura.extract(r.text)
            if text:
                print(f"Extracted: {url}")
                return {"title": title, "url": url, "text": text}
        except Exception as e:
            print(f"Failed {url}: {e}")
        return None

    def _extract_text(
        self, urls: List[Tuple[str, str]], max_workers: int = 8
    ) -> List[dict]:  # change the max worker maybe to os.cpu_count() // 2
        """Extract text content using Trafilatura in parallel."""
        extracted_texts = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._extract_single, t) for t in urls]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    extracted_texts.append(result)
        return extracted_texts

    def run(self, query: str, top_k: int = 5) -> List[dict]:
        urls = self._search_scrapingdog(query, top_k)
        return self._extract_text(urls)
