#!/usr/bin/env python3
"""
Manual test script for the Patronus pipeline.

This script runs the full pipeline (sources → agent → outputs) with visible output,
including tool calls and the final digest structure. Useful for:
- Debugging agent behavior
- Understanding how tools are used
- Verifying digest quality manually
- Development and iteration
- Testing with real Notion context

Unlike integration tests, this script:
- Uses the full DigestPipeline (same as production)
- Shows all LLM interactions via logging
- Prints formatted digest output
- Uses cached Notion context by default (fast)
- Can force fresh context or use test data
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from datetime import datetime, timezone

import numpy as np

from patronus.config import load_config
from patronus.db import Database
from patronus.pipeline import DigestPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def _unit_vec(*values: float) -> np.ndarray:
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def create_test_database(db_path: str) -> Database:
    """Create a test database with realistic content."""
    db = Database(db_path=db_path)
    
    now = datetime.now(timezone.utc)
    recent_ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Create embeddings (these would normally come from real embedding API)
    emb_ml = _unit_vec(1.0, 0.1, 0.0)
    emb_tech = _unit_vec(0.1, 1.0, 0.0)
    emb_phil = _unit_vec(0.0, 0.1, 1.0)
    
    items = [
        {
            "url": "https://arxiv.org/abs/2024.12345",
            "title": "Attention Mechanisms in Transformer Models",
            "author": "Jane Smith et al.",
            "source": "Arxiv RSS",
            "text": "This paper explores novel attention mechanisms that improve transformer efficiency by 40%. We demonstrate that sparse attention patterns can be learned dynamically during training.",
            "embedding": emb_ml,
            "item_type": "paper",
        },
        {
            "url": "https://arxiv.org/abs/2024.54321",
            "title": "Mechanistic Interpretability of Language Models",
            "author": "Alice Johnson",
            "source": "Arxiv RSS",
            "text": "We present a comprehensive framework for understanding the internal representations of large language models. Our analysis reveals distinct computational circuits.",
            "embedding": emb_ml,
            "item_type": "paper",
        },
        {
            "url": "https://arxiv.org/abs/2024.99999",
            "title": "Scaling Laws for Neural Language Models",
            "author": "OpenAI Research",
            "source": "Arxiv RSS",
            "text": "We investigate the relationship between model size, dataset size, and performance. Our findings suggest predictable scaling behavior across multiple orders of magnitude.",
            "embedding": emb_ml,
            "item_type": "paper",
        },
        {
            "url": "https://techcrunch.com/ai-startup-funding",
            "title": "AI Startup Secures $500M in Series C Funding",
            "author": "TechCrunch Staff",
            "source": "TechCrunch",
            "text": "Leading AI infrastructure company raises massive round to compete with OpenAI and Anthropic. The funding will be used to scale compute and hire research talent.",
            "embedding": emb_tech,
            "item_type": "article",
        },
        {
            "url": "https://stratechery.com/product-strategy",
            "title": "The Evolution of Product Strategy in AI-Native Companies",
            "author": "Ben Thompson",
            "source": "Stratechery",
            "text": "AI-native companies are fundamentally different from traditional software businesses. This analysis explores how product development must adapt.",
            "embedding": emb_tech,
            "item_type": "article",
        },
        {
            "url": "https://blog.example.com/philosophy",
            "title": "Consciousness and Computation: A Philosophical Exploration",
            "author": "David Williams",
            "source": "Philosophy Blog",
            "text": "This essay examines the relationship between consciousness and computational systems, drawing on analytic philosophy and cognitive science.",
            "embedding": emb_phil,
            "item_type": "article",
        },
        {
            "url": "https://twitter.com/researcher/status/123",
            "title": "Tweet: Breakthrough in RL from DeepMind",
            "author": "@researcher",
            "source": "Twitter",
            "text": "Just read the new DeepMind paper on hierarchical RL. The results are stunning - they've achieved human-level performance with 10x less compute.",
            "embedding": emb_ml,
            "item_type": "tweet",
        },
    ]
    
    for item in items:
        db.add_item(
            url=item["url"],
            source_type="rss",
            title=item["title"],
            author=item.get("author", ""),
            source=item.get("source", ""),
            text=item["text"],
            embedding=item["embedding"],
            timestamp=recent_ts,
            item_type=item.get("item_type", "article"),
        )
    
    logger.info(f"Created test database with {len(items)} items")
    return db


def print_digest(digest):
    """Pretty-print the digest structure."""
    print("\n" + "="*80)
    print(f"DIGEST GENERATED: {digest.generated_at}")
    print(f"Mode: {digest.mode}")
    print(f"Total items: {digest.item_count}")
    print("="*80 + "\n")
    
    for i, section in enumerate(digest.sections, 1):
        print(f"{i}. {section.title.upper()} ({section.type.value})")
        print("-" * 80)
        
        for j, item in enumerate(section.items, 1):
            print(f"\n   {i}.{j} {item.title}")
            print(f"       URL: {item.url}")
            if item.author:
                print(f"       Author: {item.author}")
            if item.source:
                print(f"       Source: {item.source}")
            print(f"       Summary: {item.summary}")
            print()
        
        print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run full pipeline manually with visible output"
    )
    parser.add_argument(
        "--db-path",
        default="/tmp/patronus_manual_test.db",
        help="Path to database (will be created with test data if doesn't exist)"
    )
    parser.add_argument(
        "--use-prod-db",
        action="store_true",
        help="Use production database instead of test database"
    )
    parser.add_argument(
        "--recreate-db",
        action="store_true",
        help="Recreate the test database (default: reuse if exists)"
    )
    parser.add_argument(
        "--force-notion-refresh",
        action="store_true",
        help="Force fresh Notion context (bypass 24h cache)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    
    # Ensure we're in agent mode
    if config.digest.mode != "agent":
        logger.warning(f"Config has mode={config.digest.mode}, switching to agent")
        config.digest.mode = "agent"
    
    # Create or load database
    if args.use_prod_db:
        logger.info("Using production database from config")
        db = Database()
    else:
        if args.recreate_db:
            import os
            if os.path.exists(args.db_path):
                os.remove(args.db_path)
                logger.info(f"Removed existing database: {args.db_path}")
        
        db = create_test_database(args.db_path)
        logger.info(f"Using test database: {args.db_path}")
    
    # Custom output that just captures the digest
    class CaptureOutput:
        def __init__(self):
            self.digest = None
        
        def send(self, digest, config):
            self.digest = digest
    
    output = CaptureOutput()
    
    # Build pipeline with default sources (interests + notion if configured)
    # This mirrors exactly what send_digest.py does
    logger.info("Building pipeline with default sources (interests + Notion if configured)")
    
    pipeline = DigestPipeline(
        config,
        db,
        sources=None,  # Use default sources
        outputs=[output],
    )
    
    # Run pipeline
    print("\n" + "="*80)
    print("STARTING PIPELINE (full production flow)")
    print("="*80 + "\n")
    
    digest = pipeline.run(notion_force_refresh=args.force_notion_refresh)
    
    # Print results
    print_digest(digest)
    
    # Show what sources were used
    print("\n" + "="*80)
    print("PIPELINE DETAILS")
    print("="*80)
    print(f"Database: {args.db_path if not args.use_prod_db else 'production'}")
    print(f"Notion caching: {'BYPASSED (fresh)' if args.force_notion_refresh else 'ENABLED (24h TTL)'}")
    print(f"Sources: interests.yaml" + (" + Notion" if config.notion and config.notion_token else ""))
    
    # Cleanup
    db.close()
    
    print("\n✓ Manual test complete")
    if not args.use_prod_db:
        print(f"Test database: {args.db_path}")
        print("Rerun with same --db-path to reuse the data (faster)")
    print()


if __name__ == "__main__":
    main()
