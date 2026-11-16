"""Retrieval-only evaluation for the code index.

Agent-level success rate confounds three things: retrieval quality, model
behaviour, and luck in tool sequencing. This script isolates the first one so a
retrieval change can be judged without paying for 10 full agent runs.

Gold is a set of `path:line` locations, not a keyword. Several chunks can
legitimately answer one question (the argparse block and the README section both
list the approval modes), so a hit anywhere in the accepted set counts. A hit
means a returned chunk's line range covers an accepted line, with a small window
because a chunk boundary can fall right at the answer.

Modes mirror what the agent could use:
    substring  upstream behaviour: `pattern.lower() in line.lower()`
    bm25       identifier + CJK-bigram keyword ranking
    vector     embedding cosine only
    hybrid     weighted RRF (the shipped setting)

Usage:
    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --no-context   # ablate contextual headers
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mini_coding_agent as mca  # noqa: E402
from code_index import CodeIndex, tokenize, _normalize, RRF_K, W_VECTOR, W_KEYWORD  # noqa: E402

TOP_K = 5
# A boundary cut can leave the answer line one line outside a chunk; +-3 is fair.
WINDOW = 3

# Cases live in `eval/retrieval_cases.json`, not inline here.
# Learned the hard way twice: an inlined question makes this script the best
# match for its own query, so the agent reads the harness instead of the source.
# Anything that describes the corpus has to live outside the corpus.
CASE_FILE = ROOT / "eval" / "retrieval_cases.json"


def load_cases():
    """Read cases and resolve every anchor to a line number in the current tree.

    Anchors exist because the first version stored raw line numbers. The source
    then grew by ~300 lines and all 11 cases kept scoring, silently, against
    whatever happened to sit at the old offsets: hybrid R@3 read 18% instead of
    55% and the number looked like a retrieval regression rather than a broken
    ruler. An unresolved anchor is a hard error, not a silent miss.
    """
    cases = json.loads(CASE_FILE.read_text(encoding="utf-8"))["cases"]
    resolved = []
    for case in cases:
        gold = []
        for target in case["gold"]:
            path = ROOT / target["path"]
            anchor = target.get("anchor")
            if anchor is None:
                gold.append(target)
                continue
            line = None
            for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if anchor in text:
                    line = number
                    break
            if line is None:
                raise SystemExit(f"锚点失效：{target['path']} 中找不到 {anchor!r}（用例 {case['query']}）")
            gold.append({"path": target["path"], "line": line})
        resolved.append({"query": case["query"], "gold": gold})
    return resolved


def rank_of(keys, gold):
    for position, key in enumerate(keys, start=1):
        path, _, span = key.partition(":")
        start, _, end = span.partition("-")
        for target in gold:
            if path != target["path"]:
                continue
            line = target["line"]
            if int(start) <= line + WINDOW and line - WINDOW <= int(end):
                return position
    return None


def orders(index, query):
    """Rankings from each strategy, for the ablation table.

    `hybrid` calls the shipped `search()` rather than re-deriving the fusion
    here. The first version rebuilt the RRF sum inline and silently omitted the
    rare-term bonus that `search()` applies, so the number in the table was not
    the number the agent would experience. A benchmark that reimplements the
    thing it measures measures the reimplementation.
    """
    terms = tokenize(query)
    bm25 = index._bm25_scores(terms)
    bm25_order = sorted(range(len(index.chunks)), key=lambda i: -bm25[i])
    query_vector = _normalize(index.client.embed([query])[0])
    cosine = [sum(a * b for a, b in zip(query_vector, v)) for v in index.vectors]
    vector_order = sorted(range(len(index.chunks)), key=lambda i: -cosine[i])

    position = {chunk.key: i for i, chunk in enumerate(index.chunks)}
    hybrid_order = [
        position[chunk.key]
        for chunk, _ in index.search(query, top_k=len(index.chunks))
    ]

    # Upstream search: literal substring, one pass, in file order.
    needle = query.lower()
    substring_order = [
        i for i, chunk in enumerate(index.chunks) if needle in chunk.text.lower()
    ]
    return {
        "substring": substring_order,
        "bm25": bm25_order,
        "vector": vector_order,
        "hybrid": hybrid_order,
    }


def evaluate(index, label):
    rows = []
    for case in load_cases():
        started = time.time()
        table = orders(index, case["query"])
        elapsed = (time.time() - started) * 1000
        row = {"query": case["query"], "latency_ms": elapsed}
        for mode, order in table.items():
            keys = [index.chunks[i].key for i in order[:TOP_K]]
            row[mode] = rank_of(keys, case["gold"])
        rows.append(row)

    total = len(rows)
    print(f"\n{label}  共 {total} 条 / top_k={TOP_K}")
    header = f"{'检索方式':<12}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'MRR':>8}{'耗时(ms)':>10}"
    print(header)
    print("-" * 54)
    summary = {}
    for mode in ("substring", "bm25", "vector", "hybrid"):
        hit1 = [r for r in rows if r[mode] == 1]
        hit3 = [r for r in rows if r[mode] and r[mode] <= 3]
        hit5 = [r for r in rows if r[mode] and r[mode] <= 5]
        mrr = sum(1 / r[mode] for r in rows if r[mode]) / total
        print(
            f"{mode:<12}{len(hit1) / total:>8.1%}{len(hit3) / total:>8.1%}"
            f"{len(hit5) / total:>8.1%}{mrr:>8.3f}"
            f"{sum(r['latency_ms'] for r in rows) / total:>10.0f}"
        )
        summary[mode] = {
            "recall@1": round(len(hit1) / total, 3),
            "recall@3": round(len(hit3) / total, 3),
            "recall@5": round(len(hit5) / total, 3),
            "mrr": round(mrr, 3),
        }
    misses = [r for r in rows if not r["hybrid"]]
    if misses:
        print("\n混合检索未命中：")
        for r in misses:
            print(f"  {r['query']}  (vector={r['vector']}, bm25={r['bm25']})")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate code retrieval quality.")
    parser.add_argument("--no-context", action="store_true", help="Ablate contextual headers.")
    parser.add_argument("--json", default=None, help="Write summary JSON to this path.")
    opts = parser.parse_args(argv)

    mca.load_env_file(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        print("缺少 OPENAI_API_KEY，无法向量化")
        return 1

    index = CodeIndex(ROOT, contextual=not opts.no_context).build()
    label = "有上下文头" if index.contextual else "无上下文头（消融）"
    print(f"索引 {index.build_stats['chunks']} 块 | 新向量 {index.build_stats['embedded']} | {label}")
    summary = evaluate(index, label)

    if opts.json:
        import json
        path = ROOT / opts.json
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
