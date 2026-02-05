#!/usr/bin/env python3
"""
Forecasting Research Brief Generator CLI.

Usage:
    python main.py "Will X happen by Y date?" [--ref-class "..."] [--lookback "6m"]
"""

import argparse
import asyncio
import re
from datetime import datetime

from config import load_config, Config, ConfigError
from llm_client import LLMClient
from forecasting_logic import (
    generate_search_plan,
    classify_source,
    analyze_base_rates,
    SearchPlan,
)
from search_engine import gather_data
from writer import write_brief, save_brief, PlanInfo, SourceInfo


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a forecasting research brief for a given question.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py "Will the Fed cut rates in Q1 2026?"
    python main.py "Will OpenAI release GPT-5 by 2026?" --ref-class "AI model releases"
    python main.py "Will inflation exceed 3%?" --lookback "1y"
        """,
    )
    
    parser.add_argument(
        "question",
        type=str,
        help="The forecasting question to research.",
    )
    
    parser.add_argument(
        "--ref-class",
        type=str,
        default=None,
        dest="ref_class",
        help="Manually specify the reference class for base rates.",
    )
    
    parser.add_argument(
        "--lookback",
        type=str,
        default=None,
        help="Manually specify the search time window (e.g., '6m', '1y', '2y').",
    )
    
    return parser.parse_args()


async def main() -> None:
    """Main application entry point."""
    args = parse_args()
    
    print("=" * 60)
    print("FORECASTING RESEARCH BRIEF GENERATOR")
    print("=" * 60)
    print(f"\nQuestion: {args.question}")
    if args.ref_class:
        print(f"Reference Class Override: {args.ref_class}")
    if args.lookback:
        print(f"Lookback Override: {args.lookback}")
    print()
    
    # Step A: Load configuration
    print("[Step 1/7] Loading configuration...")
    try:
        config = load_config()
        client = LLMClient(config)
        print(f"  ✓ Using model: {config.model_id}")
    except ConfigError as e:
        print(f"  ✗ Configuration error: {e}")
        return
    
    # Step B: Generate search plan
    print("[Step 2/7] Generating search plan...")
    plan = await generate_search_plan(
        client=client,
        question=args.question,
        user_ref_class=args.ref_class,
    )
    
    # Apply CLI overrides
    if args.lookback:
        plan["time_window"] = args.lookback
    
    print(f"  ✓ Time window: {plan['time_window']}")
    print(f"  ✓ Search queries: {len(plan['search_queries'])}")
    print(f"  ✓ Historical queries: {len(plan['historical_queries'])}")
    
    # Step C: Gather data (search + scrape)
    print("[Step 3/7] Searching and scraping sources...")
    scraped_data = await gather_data(
        queries=plan["search_queries"],
        max_results_per_query=5,
        max_urls=config.search.max_queries,
        scrape_concurrency=3,
        scrape_timeout=config.search.scrape_timeout,
    )
    
    successful_scrapes = [s for s in scraped_data if s["status"] == "success"]
    print(f"  ✓ Scraped {len(successful_scrapes)}/{len(scraped_data)} sources successfully")
    
    # Step D: Analyze base rates (runs concurrently conceptually, but after scraping)
    print("[Step 4/7] Analyzing base rates from historical data...")
    base_rate_summary = await analyze_base_rates(
        client=client,
        question=args.question,
        historical_queries=plan["historical_queries"],
        max_results_per_query=5,
    )
    print(f"  ✓ Base rate analysis complete")
    
    # Step E: Classify source tiers
    print("[Step 5/7] Classifying source credibility tiers...")
    sources_with_tiers: list[SourceInfo] = []
    
    for scrape in scraped_data:
        tier = await classify_source(
            domain_or_url=scrape["url"],
            client=client,
            config=config,
        )
        sources_with_tiers.append(
            SourceInfo(
                url=scrape["url"],
                text=scrape["text"],
                tier=tier,
                status=scrape["status"],
            )
        )
    
    tier_counts = {}
    for s in sources_with_tiers:
        tier_counts[s["tier"]] = tier_counts.get(s["tier"], 0) + 1
    print(f"  ✓ Tier distribution: {dict(sorted(tier_counts.items()))}")
    
    # Step F: Generate the brief
    print("[Step 6/7] Writing forecast brief...")
    plan_info = PlanInfo(
        time_window=plan["time_window"],
        reference_class=args.ref_class,
        search_queries=plan["search_queries"],
        historical_queries=plan["historical_queries"],
    )
    
    brief = await write_brief(
        client=client,
        question=args.question,
        scraped_data=sources_with_tiers,
        base_rate_summary=base_rate_summary,
        plan_info=plan_info,
    )
    print(f"  ✓ Brief generated ({len(brief)} characters)")
    
    # Step G: Save the output
    print("[Step 7/7] Saving output...")
    filepath = save_brief(brief, args.question)
    print(f"  ✓ Saved to: {filepath}")
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    
    # Print brief preview
    print("\n--- BRIEF PREVIEW (first 1000 chars) ---\n")
    print(brief[:1000])
    if len(brief) > 1000:
        print("\n... [truncated, see full output in file]")


if __name__ == "__main__":
    asyncio.run(main())