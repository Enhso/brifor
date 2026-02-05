"""Brief writer module for generating the final forecast report."""

from dataclasses import dataclass
from datetime import datetime
from typing import TypedDict
from urllib.parse import urlparse

from config import Config, load_config
from llm_client import LLMClient, LLMClientError
from scraper import ScrapeResult


class SourceInfo(TypedDict):
    """Source information with tier classification."""
    url: str
    text: str
    tier: int
    status: str


@dataclass
class PlanInfo:
    """Information from the search plan."""
    time_window: str
    reference_class: str | None
    search_queries: list[str]
    historical_queries: list[str]


def _extract_domain(url: str) -> str:
    """Extract domain from URL for display."""
    parsed = urlparse(url)
    domain = parsed.netloc or parsed.path
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _format_sources_for_prompt(sources: list[SourceInfo], max_chars: int = 2000) -> str:
    """
    Format scraped sources for inclusion in the LLM prompt.
    
    Args:
        sources: List of source info dictionaries with tier data.
        max_chars: Maximum characters to include per source. Defaults to 2000.
    
    Returns:
        Formatted string of all sources.
    """
    if not sources:
        return "No sources were successfully scraped."
    
    formatted_parts: list[str] = []
    
    for i, source in enumerate(sources, 1):
        domain = _extract_domain(source["url"])
        tier = source["tier"]
        text = source["text"]
        
        # Truncate text if needed
        if len(text) > max_chars:
            text = text[:max_chars] + "... [truncated]"
        
        formatted_parts.append(
            f"[{i}] Source (Tier {tier}): {source['url']}\n"
            f"Domain: {domain}\n"
            f"Content:\n{text}\n"
        )
    
    return "\n---\n".join(formatted_parts)


def _build_system_prompt(
    question: str,
    plan_info: PlanInfo,
    base_rate_summary: str,
    formatted_sources: str,
    source_count: int,
) -> str:
    """Build the comprehensive system prompt for brief generation."""
    
    today = datetime.now().strftime("%Y-%m-%d")
    
    ref_class_text = plan_info.reference_class or "Inferred from question context"
    
    return f"""You are an expert forecasting analyst. Generate a rigorous, structured research brief for the following question.

**FORECASTING QUESTION:** {question}

**TODAY'S DATE:** {today}

**RESEARCH PARAMETERS:**
- Time Window: {plan_info.time_window}
- Reference Class: {ref_class_text}
- Search Queries Used: {', '.join(plan_info.search_queries)}

**BASE RATE ANALYSIS:**
{base_rate_summary}

**SCRAPED SOURCES ({source_count} total):**
{formatted_sources}

---

**OUTPUT REQUIREMENTS:**

Generate a Markdown document following this EXACT structure:

# Forecast Brief: {question}
**Date:** {today}
**Reference Class Used:** {ref_class_text}

## 1. Bottom Line Up Front (BLUF)
*   [Concise synthesis of the most likely outcome based on evidence]
*   [Major signal vs. noise distinction]

## 2. Timeline of Key Events
*   **[YYYY-MM-DD]:** Event description. [1]
*   **[YYYY-MM-DD]:** Event description (note contradictions if any). [2]
(List key events chronologically with footnote citations)

## 3. Base Rates & Reference Classes
*   **Class:** [Description of reference class]
*   **Data:** Found [N] similar events. [X] resulted in Outcome A, [Y] in Outcome B.
*   **Crude Rate:** X/N ([%]).
*   *Note:* [Caveats about sample size or applicability]

## 4. Key Probability Drivers
*   [Factor 1]: Increases probability because... [cite]
*   [Factor 2]: Decreases probability because... [cite]
(List factors that push probability up or down)

## 5. Multiple Scenarios & Contrarian View
*   **Main Scenario:** [Most likely path forward]
*   **Contrarian/tail-risk:** [Alternative scenario that could upend the forecast]

## 6. Critical Uncertainties
*   [Unknown variable 1 that would flip the forecast]
*   [Unknown variable 2]

## References
(List all sources with numbered footnotes matching citations above)
1. [Page Title](URL) - *Tier [N]*
2. [Page Title](URL) - *Tier [N]*

---

**CRITICAL INSTRUCTIONS:**

1. **CITATIONS:** Use numbered footnotes [1], [2], etc. that EXACTLY match the source numbers provided above. Never invent sources.

2. **RECENCY WEIGHTING:** Prioritize newer information. If sources conflict, favor the most recent unless there's a reason not to.

3. **TIER WEIGHTING:** Give more weight to Tier 1-2 sources (academic, major news) than Tier 4-5 (blogs, social media). Note tier in References.

4. **CONTRADICTIONS:** If sources contradict, explicitly call this out in the Timeline or BLUF.

5. **NO HALLUCINATION:** If information is insufficient, say so. Do not invent facts, dates, or statistics.

6. **BULLET-POINT STYLE:** Be concise, rigorous, no fluff. Every bullet should add value.

7. **BASE RATES:** Incorporate the base rate analysis into Section 3. If no historical data was found, state this clearly.

Generate the complete Markdown brief now:"""


async def write_brief(
    client: LLMClient,
    question: str,
    scraped_data: list[SourceInfo],
    base_rate_summary: str,
    plan_info: PlanInfo,
    temperature: float = 0.4,
) -> str:
    """
    Generate the final forecast brief using the LLM.
    
    Args:
        client: Initialized LLMClient instance.
        question: The original forecasting question.
        scraped_data: List of scraped sources with tier information.
        base_rate_summary: Summary text from base rate analysis.
        plan_info: Plan information (time window, reference class, queries).
        temperature: LLM temperature for generation. Defaults to 0.4.
    
    Returns:
        The generated Markdown brief as a string.
    
    Raises:
        LLMClientError: If the LLM call fails.
    """
    # Filter to only successful scrapes
    valid_sources = [s for s in scraped_data if s["status"] == "success" and s["text"]]
    
    # Format sources for the prompt
    formatted_sources = _format_sources_for_prompt(valid_sources)
    
    # Build the comprehensive prompt
    system_prompt = _build_system_prompt(
        question=question,
        plan_info=plan_info,
        base_rate_summary=base_rate_summary,
        formatted_sources=formatted_sources,
        source_count=len(valid_sources),
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Generate the forecast brief now."},
    ]
    
    try:
        response = await client.generate_response(messages, temperature=temperature)
        return response.strip()
    except LLMClientError as e:
        # Return an error brief if generation fails
        return f"""# Forecast Brief: {question}
**Date:** {datetime.now().strftime("%Y-%m-%d")}

## Error
Brief generation failed: {e}

## Raw Data Available
- Sources scraped: {len(valid_sources)}
- Base rate summary: {base_rate_summary[:500] if base_rate_summary else 'None'}
"""


def save_brief(brief: str, question: str, output_dir: str = "./reports") -> str:
    """
    Save the brief to a Markdown file.
    
    Args:
        brief: The generated Markdown content.
        question: The original question (used for filename).
        output_dir: Directory to save the file. Defaults to current directory.
    
    Returns:
        The path to the saved file.
    """
    from pathlib import Path
    import re
    
    # Create filename from date and question snippet
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    # Clean question for filename (first 50 chars, alphanumeric only)
    clean_question = re.sub(r"[^a-zA-Z0-9\s]", "", question)
    clean_question = "_".join(clean_question.split()[:6])
    
    filename = f"{date_str}_{clean_question}.md"
    
    output_path = Path(output_dir) / filename
    output_path.write_text(brief, encoding="utf-8")
    
    return str(output_path)


async def main() -> None:
    """Test the brief writer."""
    from forecasting_logic import SearchPlan
    
    config = load_config()
    client = LLMClient(config)
    
    # Mock data for testing
    question = "Will the Federal Reserve cut interest rates in Q1 2026?"
    
    mock_sources: list[SourceInfo] = [
        {
            "url": "https://reuters.com/markets/fed-policy-outlook",
            "text": "The Federal Reserve signaled a cautious approach to rate cuts in 2026, "
                    "with Chair Powell noting inflation remains above target. Markets expect "
                    "two rate cuts by mid-2026, with the first potentially in March.",
            "tier": 2,
            "status": "success",
        },
        {
            "url": "https://bloomberg.com/fed-watch",
            "text": "Fed officials remain divided on the pace of rate cuts. The December "
                    "minutes showed some members favoring a pause while others pushed for "
                    "gradual easing. Economic data will be key for Q1 decisions.",
            "tier": 2,
            "status": "success",
        },
    ]
    
    plan_info = PlanInfo(
        time_window="6m",
        reference_class="Federal Reserve rate decisions during disinflation periods",
        search_queries=["Fed rate cut 2026", "Federal Reserve policy 2026"],
        historical_queries=["Fed rate cuts history", "rate cut timing patterns"],
    )
    
    base_rate_summary = """Historical analysis found 8 similar disinflation periods since 1990.
In 6 of 8 cases (75%), the Fed began cutting rates within 6 months of inflation dropping below 3%.
Average delay from peak rates to first cut: 4.2 months.
Caveat: Current labor market conditions differ from historical norms."""
    
    print("Generating test brief...")
    print("-" * 50)
    
    brief = await write_brief(
        client=client,
        question=question,
        scraped_data=mock_sources,
        base_rate_summary=base_rate_summary,
        plan_info=plan_info,
    )
    
    print(brief)
    print("-" * 50)
    
    # Optionally save
    filepath = save_brief(brief, question)
    print(f"Saved to: {filepath}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
