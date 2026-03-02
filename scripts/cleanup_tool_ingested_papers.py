import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite3

_TOOL_SOURCE_TYPES = {
    "arxiv_search",
    "openalex_search",
    "openalex_citing",
    "openalex_references",
}

DB_PATH = "db.sqlite3"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

placeholders = ",".join("?" for _ in _TOOL_SOURCE_TYPES)
cur.execute(
    f"SELECT source_type, COUNT(*) FROM items WHERE source_type IN ({placeholders}) GROUP BY source_type",
    list(_TOOL_SOURCE_TYPES),
)
counts = cur.fetchall()
total = sum(c for _, c in counts)

print("Items to delete:")
for source_type, count in sorted(counts):
    print(f"  {source_type}: {count}")
print(f"  Total: {total}")

if total == 0:
    print("Nothing to do.")
    conn.close()
    sys.exit(0)

confirm = input(f"\nDelete {total} items? [y/N] ").strip().lower()
if confirm != "y":
    print("Aborted.")
    conn.close()
    sys.exit(0)

cur.execute(
    f"DELETE FROM items WHERE source_type IN ({placeholders})",
    list(_TOOL_SOURCE_TYPES),
)
conn.commit()
print(f"Deleted {cur.rowcount} items.")
conn.close()
