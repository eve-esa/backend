import asyncio
import requests
from typing import List, Tuple
from urllib.parse import urlparse
import trafilatura
import httpx
from dotenv import load_dotenv

load_dotenv()


class ScrapingDogCrawler:
    def __init__(self, all_urls: List[str], api_key: str) -> None:
        self.api_key = api_key
        self.domains = list({urlparse(u).netloc for u in all_urls})
        self.domain_query = " OR ".join(f"site:{d}" for d in self.domains)
        self.base_url = "https://api.scrapingdog.com/google"

    def _search_scrapingdog(self, query: str, top_k: int) -> List[Tuple[str, str]]:
        try:
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
                raise Exception(f"scrapingdog returned error: {response.json()}")

            data = response.json()
            organic_results = data.get("organic_results", [])
            urls = []
            for r in organic_results:
                title = r.get("title", "")
                link = r.get("link", "")
                if link.lower().endswith(".pdf") or ".pdf?" in link.lower():  # skip pdfs
                    print(f"Skipping PDF: {link}")
                    continue
                urls.append((title, link))
            print(f"ScrapingDog returned {len(urls)} URLs")
            return urls[:top_k]
        except Exception:
            raise

    async def _extract_single(
        self, client: httpx.AsyncClient, title_url: Tuple[str, str]
    ):
        title, url = title_url
        try:
            resp = await client.get(url, timeout=20, follow_redirects=True)
            text = trafilatura.extract(resp.text)
            if text:
                print(f"Extracted: {url}")
                return {
                    "collection_name": "Online search",
                    "payload": {
                        "title": title,
                        "url": url,
                    },
                    "text": text,
                }
        except Exception as e:
            raise Exception(f"Failed to extract single URL: {url}: {e}") from e
        return None

    async def _extract_text(
        self, urls: List[Tuple[str, str]], max_concurrent: int = 10
    ):
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []

        async with httpx.AsyncClient(
            verify=False
        ) as client:  # verify false to extract webpages with bad ssl certs

            async def extract(title_url):
                async with semaphore:
                    return await self._extract_single(client, title_url)

            tasks = [extract(t) for t in urls]
            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    if result:
                        results.append(result)
                except Exception:
                    raise
        return results

    async def run(self, query: str, top_k: int = 5):
        try:
            urls = self._search_scrapingdog(query, top_k)
        except Exception as e:
            raise
        if not urls:
            return []
        try:
            return await self._extract_text(urls)
        except Exception as e:
            raise
