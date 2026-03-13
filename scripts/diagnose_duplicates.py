import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _get_api_client() -> Any:
    from langfuse import Langfuse
    pk = os.getenv("LANGFUSE_PUBLIC_KEY")
    sk = os.getenv("LANGFUSE_SECRET_KEY")
    if not pk or not sk:
        raise RuntimeError("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set in .env")
    lf = Langfuse(public_key=pk, secret_key=sk)
    return lf.api


def _normalize_url(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    for prefix in ["https://", "http://"]:
        if url.startswith(prefix):
            url = url[len(prefix):]
            break
    return url.lower()


def _extract_items_from_compose(compose_output: dict) -> list[dict]:
    items = []
    for section in compose_output.get("sections", []):
        section_type = section.get("type", "unknown")
        for item in section.get("items", []):
            items.append({
                "section": section_type,
                "item_id": item.get("item_id", ""),
                "title": (item.get("title") or "").strip(),
                "url": item.get("url", ""),
            })
    return items


def _find_compose_output(trace: Any) -> dict | None:
    """Extract the final digest structure from any observation in the trace."""
    observations = getattr(trace, "observations", []) or []

    # Priority 1: compose-attempt generation span — look at submit_digest tool call
    for obs in observations:
        name = getattr(obs, "name", "") or ""
        if name.startswith("compose-attempt"):
            out = getattr(obs, "output", None) or {}
            for tc in out.get("tool_calls", []):
                if tc.get("name") == "submit_digest":
                    inp = tc.get("input", {})
                    if isinstance(inp, dict) and "sections" in inp:
                        return inp

    # Priority 2: step4-compose span output (summary only, has section_types)
    for obs in observations:
        name = getattr(obs, "name", "") or ""
        if name == "step4-compose":
            out = getattr(obs, "output", None)
            if isinstance(out, dict) and "sections" in out:
                return out

    # Priority 3: agent-digest-run span output
    for obs in observations:
        name = getattr(obs, "name", "") or ""
        if name == "agent-digest-run":
            out = getattr(obs, "output", None)
            if isinstance(out, dict) and "sections" in out:
                return out

    # Priority 4: top-level trace output
    top_output = getattr(trace, "output", None)
    if isinstance(top_output, dict) and "sections" in top_output:
        return top_output

    return None


def _extract_step_text(observations: list[Any], name_prefix: str) -> str:
    for obs in observations:
        name = getattr(obs, "name", "") or ""
        if name.startswith(name_prefix):
            out = getattr(obs, "output", None)
            if out is None:
                continue
            if isinstance(out, dict):
                return json.dumps(out)
            return str(out)
    return ""


def analyze_trace(trace: Any) -> dict[str, Any]:
    trace_id = getattr(trace, "id", "?")
    created_at = getattr(trace, "timestamp", None)
    observations = getattr(trace, "observations", []) or []

    compose_output = _find_compose_output(trace)
    if compose_output is None:
        return {"trace_id": trace_id, "created_at": str(created_at), "error": "no_compose_output"}

    all_items = _extract_items_from_compose(compose_output)

    # --- Intra-digest duplicate detection ---
    url_to_sections: dict[str, list[str]] = defaultdict(list)
    id_to_sections: dict[str, list[str]] = defaultdict(list)
    title_to_sections: dict[str, list[str]] = defaultdict(list)

    for item in all_items:
        url = _normalize_url(item["url"])
        if url:
            url_to_sections[url].append(item["section"])
        item_id = item["item_id"]
        if item_id and item_id not in ("", "0"):
            id_to_sections[item_id].append(item["section"])
        title = item["title"].lower()
        if title:
            title_to_sections[title].append(item["section"])

    url_dupes = {url: secs for url, secs in url_to_sections.items() if len(secs) > 1}
    id_dupes = {id_: secs for id_, secs in id_to_sections.items() if len(secs) > 1}
    title_dupes = {t: secs for t, secs in title_to_sections.items() if len(secs) > 1}

    # --- Cross-section URL overlap (news vs threads/long_form) ---
    news_sections = {"whats_new", "headlines"}
    thread_sections = {"threads", "long_form_pick", "serendipity"}
    research_sections = {"research_roundup"}

    urls_by_section_group: dict[str, set[str]] = {"news": set(), "threads": set(), "research": set()}
    for item in all_items:
        url = _normalize_url(item["url"])
        sec = item["section"]
        if sec in news_sections:
            urls_by_section_group["news"].add(url)
        elif sec in thread_sections:
            urls_by_section_group["threads"].add(url)
        elif sec in research_sections:
            urls_by_section_group["research"].add(url)

    news_threads_overlap = list(urls_by_section_group["news"] & urls_by_section_group["threads"])
    news_research_overlap = list(urls_by_section_group["news"] & urls_by_section_group["research"])
    threads_research_overlap = list(urls_by_section_group["threads"] & urls_by_section_group["research"])

    # --- Tool call analysis: what did thread-puller retrieve? ---
    tool_calls_made: list[dict] = []
    news_text = _extract_step_text(observations, "step3a-news")
    threads_text = _extract_step_text(observations, "step3c-threads")
    research_text = _extract_step_text(observations, "step3b-research")

    for obs in observations:
        obs_type = getattr(obs, "type", None) or getattr(obs, "observation_type", None)
        name = getattr(obs, "name", "") or ""
        if str(obs_type) == "tool" or name in ("search_similar", "search_recent", "search_by_topic", "search_by_source", "search_arxiv", "search_openalex", "search_notion", "get_citing_papers", "get_referenced_papers"):
            inp = getattr(obs, "input", {}) or {}
            tool_calls_made.append({
                "tool": name,
                "input": inp,
            })

    section_counts = Counter(item["section"] for item in all_items)

    return {
        "trace_id": trace_id,
        "created_at": str(created_at),
        "total_items": len(all_items),
        "section_counts": dict(section_counts),
        "intra_digest_url_dupes": dict(list(url_dupes.items())[:20]),
        "intra_digest_id_dupes": dict(list(id_dupes.items())[:20]),
        "intra_digest_title_dupes": dict(list(title_dupes.items())[:20]),
        "cross_section": {
            "news_threads": news_threads_overlap[:10],
            "news_research": news_research_overlap[:10],
            "threads_research": threads_research_overlap[:10],
        },
        "dupe_count_by_url": len(url_dupes),
        "dupe_count_by_id": len(id_dupes),
        "dupe_count_by_title": len(title_dupes),
        "tool_calls": tool_calls_made[:30],
        "all_items": all_items,
    }


def fetch_and_analyze(api: Any, days: int = 10) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    print(f"Fetching digest-pipeline traces since {cutoff.strftime('%Y-%m-%d')}...")
    page = 1
    trace_ids = []
    while True:
        result = api.trace.list(
            name="digest-pipeline",
            from_timestamp=cutoff,
            page=page,
            limit=20,
        )
        batch = result.data if hasattr(result, "data") else []
        if not batch:
            break
        for t in batch:
            trace_ids.append(getattr(t, "id"))
        if len(batch) < 20:
            break
        page += 1

    print(f"Found {len(trace_ids)} traces. Fetching full observations...")

    analyses = []
    for i, tid in enumerate(trace_ids):
        print(f"  [{i+1}/{len(trace_ids)}] {tid[:12]}...")
        try:
            full_trace = api.trace.get(tid)
            result = analyze_trace(full_trace)
        except Exception as e:
            print(f"    Error: {e}")
            result = {"trace_id": tid, "error": str(e), "created_at": "?"}
        analyses.append(result)

    return analyses


def print_report(analyses: list[dict]) -> None:
    valid = [a for a in analyses if "error" not in a]
    errored = [a for a in analyses if "error" in a]

    print()
    print("=" * 72)
    print(f"  DIGEST DUPLICATE DIAGNOSTICS")
    print(f"  {len(analyses)} traces | {len(valid)} with data | {len(errored)} missing output")
    print("=" * 72)

    if not valid:
        print("\nNo valid analyses to report on.")
        return

    # --- Aggregate stats ---
    total_items_sum = sum(a["total_items"] for a in valid)
    avg_items = total_items_sum / len(valid)
    traces_with_url_dupes = sum(1 for a in valid if a["dupe_count_by_url"] > 0)
    traces_with_title_dupes = sum(1 for a in valid if a["dupe_count_by_title"] > 0)
    traces_with_id_dupes = sum(1 for a in valid if a["dupe_count_by_id"] > 0)
    total_url_dupes = sum(a["dupe_count_by_url"] for a in valid)
    total_title_dupes = sum(a["dupe_count_by_title"] for a in valid)

    def pct(n: int, d: int) -> str:
        return f"{n}/{d} ({100*n//d}%)" if d else "0/0"

    print(f"\n  Average digest size: {avg_items:.1f} items")
    print(f"\n{'─'*72}")
    print("  INTRA-DIGEST DUPLICATES  (same item in multiple sections, same digest)")
    print(f"{'─'*72}")
    print(f"  By URL   : {total_url_dupes} duplicate pairs across {pct(traces_with_url_dupes, len(valid))} digests")
    print(f"  By title : {total_title_dupes} duplicate pairs across {pct(traces_with_title_dupes, len(valid))} digests")
    print(f"  By ID    : {sum(a['dupe_count_by_id'] for a in valid)} pairs across {pct(traces_with_id_dupes, len(valid))} digests")

    # Section pair breakdown
    section_pair_counts: Counter = Counter()
    for a in valid:
        for secs in a.get("intra_digest_url_dupes", {}).values():
            if len(secs) >= 2:
                for i, s1 in enumerate(sorted(set(secs))):
                    for s2 in sorted(set(secs))[i+1:]:
                        section_pair_counts[(s1, s2)] += 1
        for secs in a.get("intra_digest_title_dupes", {}).values():
            if len(secs) >= 2:
                for i, s1 in enumerate(sorted(set(secs))):
                    for s2 in sorted(set(secs))[i+1:]:
                        section_pair_counts[(s1, s2)] += 1

    if section_pair_counts:
        print("\n  Most common section pairs sharing duplicate items:")
        for (s1, s2), cnt in section_pair_counts.most_common(8):
            print(f"    {s1:<22} ↔ {s2:<22}  {cnt}×")

    # Cross-section stats
    nn = sum(len(a["cross_section"].get("news_threads", [])) for a in valid)
    nr = sum(len(a["cross_section"].get("news_research", [])) for a in valid)
    tr = sum(len(a["cross_section"].get("threads_research", [])) for a in valid)
    nn_traces = sum(1 for a in valid if a["cross_section"].get("news_threads"))
    nr_traces = sum(1 for a in valid if a["cross_section"].get("news_research"))
    tr_traces = sum(1 for a in valid if a["cross_section"].get("threads_research"))

    print(f"\n{'─'*72}")
    print("  CROSS-SECTION OVERLAPS  (same URL appearing in two different sections)")
    print(f"{'─'*72}")
    print(f"  news ↔ threads   : {nn} overlapping items across {pct(nn_traces, len(valid))} digests")
    print(f"  news ↔ research  : {nr} overlapping items across {pct(nr_traces, len(valid))} digests")
    print(f"  threads ↔ research: {tr} overlapping items across {pct(tr_traces, len(valid))} digests")

    # --- Cross-digest repeats (same URL appearing in multiple digests) ---
    url_digest_map: dict[str, list[str]] = defaultdict(list)
    title_digest_map: dict[str, list[str]] = defaultdict(list)
    for a in valid:
        date = (a.get("created_at") or "?")[:10]
        for item in a.get("all_items", []):
            url = _normalize_url(item["url"])
            if url:
                url_digest_map[url].append(date)
            title = item["title"].lower().strip()
            if title:
                title_digest_map[title].append(date)

    cross_digest_url = {url: dates for url, dates in url_digest_map.items() if len(set(dates)) > 1}
    cross_digest_title = {t: dates for t, dates in title_digest_map.items() if len(set(dates)) > 1}

    print(f"\n{'─'*72}")
    print("  CROSS-DIGEST REPEATS  (same item appearing in multiple different digests)")
    print(f"{'─'*72}")
    print(f"  By URL   : {len(cross_digest_url)} unique items repeated across multiple digests")
    print(f"  By title : {len(cross_digest_title)} unique items repeated across multiple digests")

    if cross_digest_url:
        print("\n  Top repeated URLs (by number of digest appearances):")
        sorted_repeats = sorted(cross_digest_url.items(), key=lambda x: len(x[1]), reverse=True)
        for url, dates in sorted_repeats[:10]:
            print(f"    [{', '.join(sorted(set(dates)))}]  {url[:65]}")

    if cross_digest_title:
        print("\n  Top repeated titles:")
        sorted_title_repeats = sorted(cross_digest_title.items(), key=lambda x: len(x[1]), reverse=True)
        for title, dates in sorted_title_repeats[:10]:
            print(f"    [{', '.join(sorted(set(dates)))}]  '{title[:60]}'")

    # --- Per-trace breakdown ---
    print(f"\n{'─'*72}")
    print("  PER-DIGEST BREAKDOWN")
    print(f"{'─'*72}")

    valid_sorted = sorted(valid, key=lambda a: a.get("created_at", ""), reverse=True)
    for a in valid_sorted:
        date_str = (a.get("created_at") or "?")[:10]
        n_items = a["total_items"]
        n_url_d = a["dupe_count_by_url"]
        n_title_d = a["dupe_count_by_title"]
        n_id_d = a["dupe_count_by_id"]
        cs = a.get("cross_section", {})
        cs_nn = len(cs.get("news_threads", []))
        cs_nr = len(cs.get("news_research", []))
        cs_tr = len(cs.get("threads_research", []))
        sections = a.get("section_counts", {})
        print(f"\n  [{date_str}]  {a['trace_id'][:10]}  |  {n_items} items  |  url_dupes={n_url_d}  title_dupes={n_title_d}  id_dupes={n_id_d}")
        print(f"    Sections: {dict(sorted(sections.items()))}")
        cross_parts = []
        if cs_nn: cross_parts.append(f"news↔threads={cs_nn}")
        if cs_nr: cross_parts.append(f"news↔research={cs_nr}")
        if cs_tr: cross_parts.append(f"threads↔research={cs_tr}")
        if cross_parts:
            print(f"    Cross-section: {', '.join(cross_parts)}")

        if a.get("intra_digest_url_dupes"):
            print("    URL duplicates (url → sections they appear in):")
            for url, secs in list(a["intra_digest_url_dupes"].items())[:5]:
                print(f"      {url[:65]}")
                print(f"        → {secs}")

        if a.get("intra_digest_title_dupes"):
            print("    Title duplicates (title → sections):")
            for title, secs in list(a["intra_digest_title_dupes"].items())[:5]:
                print(f"      '{title[:60]}'  → {secs}")

        if cs.get("news_threads"):
            print("    URLs in BOTH news and threads sections:")
            for url in cs["news_threads"][:3]:
                print(f"      {url[:70]}")

    if errored:
        print(f"\n{'─'*72}")
        print("  ERRORED TRACES")
        print(f"{'─'*72}")
        for a in errored:
            print(f"  {a['trace_id'][:14]}  {a.get('error', '?')}")


def main() -> None:
    api = _get_api_client()
    analyses = fetch_and_analyze(api, days=10)

    print_report(analyses)

    out_path = Path(__file__).parent.parent / "duplicate_analysis.json"
    # Don't serialize all_items for brevity in dupe report
    export = [{k: v for k, v in a.items() if k != "all_items"} for a in analyses]
    with open(out_path, "w") as f:
        json.dump(export, f, indent=2, default=str)
    print(f"\n  Full JSON saved to: {out_path}")


if __name__ == "__main__":
    main()
