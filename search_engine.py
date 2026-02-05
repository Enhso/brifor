"""Search engine module for web research using DuckDuckGo."""

import asyncio
from typing import TypedDict

from asyncddgs import aDDGS

from scraper import scrape_url, ScrapeResult


class SearchResult(TypedDict):
    """A single search result."""
    title: str
    url: str
    snippet: str


async def perform_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """
    Perform an async search using DuckDuckGo.
    
    Args:
        query: The search query string.
        max_results: Maximum number of results to return. Defaults to 5.
    
    Returns:
        List of search results with title, url, and snippet.
    """
    try:
        async with aDDGS() as ddgs:
            results = await ddgs.text(query, max_results=max_results)
            
            return [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", ""),
                )
                for r in results
            ]
    except Exception:
        return []


async def gather_data(
    queries: list[str],
    max_results_per_query: int = 5,
    max_urls: int = 10,
    scrape_concurrency: int = 3,
    scrape_timeout: int = 15,
) -> list[ScrapeResult]:
    """
    Search multiple queries and scrape the resulting unique URLs.
    
    Args:
        queries: List of search queries to execute.
        max_results_per_query: Max results per search query. Defaults to 5.
        max_urls: Maximum unique URLs to scrape. Defaults to 10.
        scrape_concurrency: Max concurrent browser instances. Defaults to 3.
        scrape_timeout: Timeout for each scrape in seconds. Defaults to 15.
    
    Returns:
        List of scraped data dictionaries with url, text, and status.
    """
    # Run all searches concurrently
    search_tasks = [
        perform_search(query, max_results=max_results_per_query)
        for query in queries
    ]
    all_results = await asyncio.gather(*search_tasks)
    
    # Flatten and deduplicate URLs (preserve order)
    seen_urls: set[str] = set()
    unique_urls: list[str] = []
    
    for results in all_results:
        for result in results:
            url = result["url"]
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_urls.append(url)
    
    # Limit to max_urls
    urls_to_scrape = unique_urls[:max_urls]
    
    # Scrape with limited concurrency
    semaphore = asyncio.Semaphore(scrape_concurrency)
    
    async def scrape_with_limit(url: str) -> ScrapeResult:
        async with semaphore:
            return await scrape_url(url, timeout=scrape_timeout)
    
    scrape_tasks = [scrape_with_limit(url) for url in urls_to_scrape]
    scraped_data = await asyncio.gather(*scrape_tasks)
    
    return list(scraped_data)


async def main() -> None:
    """Test the search and scrape pipeline."""
    queries = ["Python async programming", "asyncio best practices"]
    
    print(f"Searching for: {queries}")
    print("-" * 40)
    
    # Test perform_search
    results = await perform_search(queries[0], max_results=3)
    print(f"Found {len(results)} results for '{queries[0]}':")
    for r in results:
        print(f"  - {r['title'][:50]}... ({r['url'][:40]}...)")
    
    print("-" * 40)
    print("Running full gather_data pipeline (this may take a moment)...")
    
    # Test gather_data with limited scope
    scraped = await gather_data(
        queries,
        max_results_per_query=2,
        max_urls=2,
        scrape_concurrency=2,
        scrape_timeout=15,
    )
    
    print(f"Scraped {len(scraped)} pages:")
    for s in scraped:
        status = "✓" if s["status"] == "success" else "✗"
        text_preview = s["text"][:80] + "..." if s["text"] else "(no content)"
        print(f"  {status} {s['url'][:50]}...")
        print(f"    {text_preview}")


if __name__ == "__main__":
    asyncio.run(main())
