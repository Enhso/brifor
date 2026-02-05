"""Heavyweight web scraper using Playwright for JavaScript-rendered pages."""

from typing import TypedDict

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout


class ScrapeResult(TypedDict):
    """Result of a scrape operation."""
    url: str
    text: str
    status: str


async def scrape_url(url: str, timeout: int = 15) -> ScrapeResult:
    """
    Scrape a URL using a headless browser.
    
    Uses Playwright to render JavaScript-heavy pages and extracts
    clean text content from the body.
    
    Args:
        url: The URL to scrape.
        timeout: Timeout in seconds for page load. Defaults to 15.
    
    Returns:
        A dictionary with 'url', 'text', and 'status' keys.
        Status is 'success' on successful scrape, 'error' otherwise.
    """
    async with async_playwright() as p:
        browser = None
        try:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            )
            page = await context.new_page()
            
            # Navigate and wait for network to settle
            await page.goto(url, timeout=timeout * 1000)
            await page.wait_for_load_state("networkidle", timeout=timeout * 1000)
            
            # Extract body content
            body_html = await page.content()
            
            # Clean the HTML
            cleaned_text = _extract_clean_text(body_html)
            
            await context.close()
            
            return ScrapeResult(
                url=url,
                text=cleaned_text,
                status="success",
            )
            
        except PlaywrightTimeout:
            return ScrapeResult(url=url, text="", status="error")
        except Exception:
            return ScrapeResult(url=url, text="", status="error")
        finally:
            if browser:
                await browser.close()


def _extract_clean_text(html: str) -> str:
    """
    Extract clean text from HTML, removing scripts and styles.
    
    Args:
        html: Raw HTML content.
    
    Returns:
        Cleaned text content with normalized whitespace.
    """
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove script and style elements
    for element in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        element.decompose()
    
    # Extract text
    text = soup.get_text(separator=" ", strip=True)
    
    # Normalize whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = " ".join(chunk for chunk in chunks if chunk)
    
    return text


async def main() -> None:
    """Test the scraper."""
    test_url = "https://example.com/"
    print(f"Scraping: {test_url}")
    
    result = await scrape_url(test_url, timeout=15)
    
    print(f"Status: {result['status']}")
    print(f"Text preview: {result['text'][:200]}...")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
