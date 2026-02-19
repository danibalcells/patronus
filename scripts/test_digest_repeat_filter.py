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
        {"url": "https://arxiv.org/abs/2402.00001", "title": "Run 1 — Paper A", "source": "arxiv", "source_type": "arxiv_search", "item_type": "paper"},
        {"url": "https://arxiv.org/abs/2402.00002", "title": "Run 1 — Paper B", "source": "arxiv", "source_type": "arxiv_search", "item_type": "paper"},
    ],
    [
        {"url": "https://arxiv.org/abs/2403.00001", "title": "Run 2 — Paper C", "source": "arxiv", "source_type": "arxiv_search", "item_type": "paper"},
        {"url": "https://arxiv.org/abs/2403.00002", "title": "Run 2 — Paper D", "source": "arxiv", "source_type": "arxiv_search", "item_type": "paper"},
    ],
    [
        {"url": "https://arxiv.org/abs/2404.00001", "title": "Run 3 — Paper E", "source": "arxiv", "source_type": "arxiv_search", "item_type": "paper"},
        {"url": "https://arxiv.org/abs/2404.00002", "title": "Run 3 — Paper F", "source": "arxiv", "source_type": "arxiv_search", "item_type": "paper"},
    ],
    [
        {"url": "https://arxiv.org/abs/2405.00001", "title": "Run 4 — Paper G (fresh)", "source": "arxiv", "source_type": "arxiv_search", "item_type": "paper"},
        {"url": "https://arxiv.org/abs/2405.00002", "title": "Run 4 — Paper H (fresh)", "source": "arxiv", "source_type": "arxiv_search", "item_type": "paper"},
    ],
]


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

            print(f"  Repeated paper id : {repeated_id}")
            print(f"  Repeated paper url: {REPEATED_PAPER['url']}")

            for run_num in range(1, 5):
                _section(f"Run {run_num}")
                fresh_ids = all_fresh_ids[run_num - 1]
                fresh_titles = [FRESH_PAPERS_BY_RUN[run_num - 1][i]["title"] for i in range(len(fresh_ids))]

                if run_num < 4:
                    digest_ids = [repeated_id] + fresh_ids
                    generated_at = (datetime.now(timezone.utc) - timedelta(days=4 - run_num)).strftime("%Y-%m-%dT%H:%M:%SZ")
                    items = [{"item_id": iid, "summary": "", "score": 1.0, "matched_topic": "ml"} for iid in digest_ids]
                    db.save_digest(generated_at=generated_at, item_count=len(items), formatted_text=None, items=items)

                    over = db.get_over_digested_item_ids()
                    repeat_count = run_num
                    blocked = repeated_id in over

                    print(f"\n  Digest saved. Included items:")
                    print(f"    • {REPEATED_PAPER['title']}  ← repeat paper (appearance #{repeat_count})")
                    for t in fresh_titles:
                        print(f"    • {t}")
                    print()
                    if blocked:
                        print(f"  ⚑  Repeat paper has now appeared in {repeat_count} digests → BLOCKED from future runs")
                    else:
                        print(f"  ✓  Repeat paper has appeared in {repeat_count}/3 digests → still eligible")

                else:
                    print(f"\n  Simulating agent search calls...")
                    print(f"\n  Fresh papers available this run:")
                    for t in fresh_titles:
                        print(f"    • {t}")

                    over = db.get_over_digested_item_ids()
                    print(f"\n  Items currently blocked (3+ appearances): {len(over)}")
                    print(f"  Repeat paper blocked? {'YES' if repeated_id in over else 'NO'}")

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
                        print("  PASS  Repeat paper correctly excluded from all Run 4 search results.")
                    else:
                        failures = [name for name, found in [("SearchRecent", found_repeat), ("SearchBySource", found_repeat2)] if found]
                        print(f"  FAIL  Repeat paper still appeared in: {', '.join(failures)}")
                    _hr("═")
                    print()


if __name__ == "__main__":
    main()
