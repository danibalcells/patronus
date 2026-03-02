import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tempfile
from datetime import datetime, timedelta, timezone

from patronus.db import Database
from patronus.tools.local import SearchBySource, SearchRecent

REPEATED_PAPER = {
    "url": "https://arxiv.org/abs/2401.00001",
    "title": "★ REPEAT PAPER: Attention Is All You Need",
    "source": "arxiv",
    "source_type": "arxiv_search",
    "item_type": "paper",
    "author": "Vaswani et al.",
}

FRESH_PAPERS_BY_RUN: list[list[dict]] = [
    [
        {"url": "https://arxiv.org/abs/2402.00001", "title": "Run 1 — Paper A (15 days ago)", "source": "arxiv", "source_type": "arxiv_search", "item_type": "paper"},
        {"url": "https://arxiv.org/abs/2402.00002", "title": "Run 1 — Paper B (15 days ago)", "source": "arxiv", "source_type": "arxiv_search", "item_type": "paper"},
    ],
    [
        {"url": "https://arxiv.org/abs/2403.00001", "title": "Run 2 — Paper C (45 days ago, outside window)", "source": "arxiv", "source_type": "arxiv_search", "item_type": "paper"},
        {"url": "https://arxiv.org/abs/2403.00002", "title": "Run 2 — Paper D (45 days ago, outside window)", "source": "arxiv", "source_type": "arxiv_search", "item_type": "paper"},
    ],
    [
        {"url": "https://arxiv.org/abs/2405.00001", "title": "Run 3 — Paper E (fresh)", "source": "arxiv", "source_type": "arxiv_search", "item_type": "paper"},
        {"url": "https://arxiv.org/abs/2405.00002", "title": "Run 3 — Paper F (fresh)", "source": "arxiv", "source_type": "arxiv_search", "item_type": "paper"},
    ],
]

# days_ago for each simulated digest
DIGEST_AGES = [15, 45, 0]


def _add_paper(db: Database, paper: dict, ts: str) -> str:
    return db.add_item(
        url=paper["url"],
        title=paper.get("title"),
        source=paper.get("source"),
        source_type=paper["source_type"],
        item_type=paper.get("item_type", "paper"),
        author=paper.get("author"),
        timestamp=ts,
    )


def _hr(char: str = "─", width: int = 72) -> None:
    print(char * width)


def _section(title: str) -> None:
    print()
    _hr("═")
    print(f"  {title}")
    _hr("═")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.sqlite3")

        _section("Setup")
        print()
        with Database(db_path) as db:
            now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            repeated_id = _add_paper(db, REPEATED_PAPER, now_ts)
            all_fresh_ids: list[list[str]] = []
            for run_papers in FRESH_PAPERS_BY_RUN:
                run_ids = [_add_paper(db, p, now_ts) for p in run_papers]
                all_fresh_ids.append(run_ids)

            print(f"  Rule: items digested within the last 30 days are excluded from searches.")
            print(f"  Repeated paper id : {repeated_id}")
            print(f"  Repeated paper url: {REPEATED_PAPER['url']}")

            for run_num in range(1, 4):
                days_ago = DIGEST_AGES[run_num - 1]
                generated_at = (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")
                fresh_ids = all_fresh_ids[run_num - 1]
                fresh_titles = [FRESH_PAPERS_BY_RUN[run_num - 1][i]["title"] for i in range(len(fresh_ids))]

                _section(f"Run {run_num}  (digest was {days_ago} days ago — {'within' if days_ago < 30 else 'outside'} 30-day window)")

                digest_ids = [repeated_id] + fresh_ids
                items = [{"item_id": iid, "summary": "", "score": 1.0, "matched_topic": "ml"} for iid in digest_ids]
                db.save_digest(generated_at=generated_at, item_count=len(items), formatted_text=None, items=items)

                recently_digested = db.get_recently_digested_item_ids()
                repeat_blocked = repeated_id in recently_digested

                print(f"\n  Digest saved (generated_at={generated_at}). Included items:")
                print(f"    • {REPEATED_PAPER['title']}  ← repeat paper")
                for t in fresh_titles:
                    print(f"    • {t}")
                print()
                if repeat_blocked:
                    print(f"  ⚑  Repeat paper is within the 30-day window → BLOCKED from current searches")
                else:
                    print(f"  ✓  Repeat paper's digest is older than 30 days → eligible again")

            _section("Run 4 — search results (repeat paper was last digested 15 days ago → should be excluded)")

            recently_digested = db.get_recently_digested_item_ids()
            print(f"\n  Items blocked (digested in last 30 days): {len(recently_digested)}")
            print(f"  Repeat paper blocked? {'YES' if repeated_id in recently_digested else 'NO'}")

            print()
            _hr()
            print("  SearchRecent results (days=365, n=50):")
            _hr()
            result = SearchRecent(db).execute(days=365, n=50)
            found_repeat = False
            for item in result.items:
                marker = "  ← REPEAT PAPER (should be absent!)" if item["id"] == repeated_id else ""
                if item["id"] == repeated_id:
                    found_repeat = True
                print(f"    • {item['title']}{marker}")

            print()
            _hr()
            print("  SearchBySource(source_name='arxiv') results (n=50):")
            _hr()
            result2 = SearchBySource(db).execute(source_name="arxiv", n=50)
            found_repeat2 = False
            for item in result2.items:
                marker = "  ← REPEAT PAPER (should be absent!)" if item["id"] == repeated_id else ""
                if item["id"] == repeated_id:
                    found_repeat2 = True
                print(f"    • {item['title']}{marker}")

            print()
            _hr("═")
            if not found_repeat and not found_repeat2:
                print("  PASS  Repeat paper (digested 15 days ago) correctly excluded from Run 4 searches.")
                print("  NOTE  Papers from Run 2 (45 days ago) ARE included — outside the 30-day window.")
            else:
                failures = [name for name, found in [("SearchRecent", found_repeat), ("SearchBySource", found_repeat2)] if found]
                print(f"  FAIL  Repeat paper still appeared in: {', '.join(failures)}")
            _hr("═")
            print()


if __name__ == "__main__":
    main()
