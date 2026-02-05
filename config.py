"""Configuration management for the forecasting research CLI tool."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
import os


@dataclass
class SearchConfig:
    """Search-related configuration."""
    max_queries: int = 10
    scrape_timeout: int = 15


@dataclass
class DomainTiers:
    """Domain tier configuration for source prioritization."""
    tier_1: list[str] = field(default_factory=list)
    tier_2: list[str] = field(default_factory=list)


@dataclass
class Config:
    """Main application configuration."""
    model_id: str
    api_key: str
    search: SearchConfig
    domain_tiers: DomainTiers


class ConfigError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


def load_config(config_path: Path | str | None = None) -> Config:
    """
    Load and validate application configuration.
    
    Args:
        config_path: Path to the YAML config file. Defaults to config.yaml
                     in the same directory as this module.
    
    Returns:
        A validated Config object.
    
    Raises:
        ConfigError: If the API key is missing or config file is invalid.
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Validate API key exists
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ConfigError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Please add it to your .env file."
        )
    
    # Determine config file path
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)
    
    # Load YAML configuration
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, "r") as f:
            raw_config: dict[str, Any] = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in config file: {e}") from e
    
    # Parse configuration sections
    search_data = raw_config.get("search", {})
    search_config = SearchConfig(
        max_queries=search_data.get("max_queries", 10),
        scrape_timeout=search_data.get("scrape_timeout", 15),
    )
    
    tiers_data = raw_config.get("domain_tiers", {})
    domain_tiers = DomainTiers(
        tier_1=tiers_data.get("tier_1", []),
        tier_2=tiers_data.get("tier_2", []),
    )
    
    return Config(
        model_id=raw_config.get("model_id", ""),
        api_key=api_key,
        search=search_config,
        domain_tiers=domain_tiers,
    )


# Useless function?
def get_config() -> Config:
    """
    Convenience function to load the default configuration.
    
    Returns:
        A validated Config object using default paths.
    """
    return load_config()


if __name__ == "__main__":
    # Quick test when running directly
    try:
        config = load_config()
        print(f"Model ID: {config.model_id}")
        print(f"API Key: {config.api_key[:8]}...")
        print(f"Max queries: {config.search.max_queries}")
        print(f"Tier 1 domains: {config.domain_tiers.tier_1}")
    except ConfigError as e:
        print(f"Configuration error: {e}")
